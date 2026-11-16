#!/usr/bin/env python3
"""Summarize public timing/shape results only, without dumping key payloads."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])

def distribution(values):
    return {'samples': len(values), 'median_s': statistics.median(values),
            'min_s': min(values), 'max_s': max(values)}

def traces(path, prefix):
    if not path.exists():
        return []
    return [json.loads(line[len(prefix):]) for line in path.read_text().splitlines()
            if line.startswith(prefix)]

out = {'scope': 'Fresh-process measurements. Validation excluded from named upload stages; no physical cold-disk assertion.'}
base = []
for rep in (1, 2, 3):
    path = root / f'poseidon-q32-omp1-r{rep}/probe.json'
    if path.exists():
        j = json.loads(path.read_text())
        assert j['status'] == 'ok', path
        base.append(j)
if base:
    stages = ('cpu_context', 'ordinary_keygen', 'relin_generation', 'galois_generation', 'relin_upload', 'galois_upload')
    out['poseidon_q32_omp1'] = {s: distribution([j['stages'][s]['wall_ms']/1000 for j in base]) for s in stages}
    for label, names in [('evalkey_generation', ('relin_generation', 'galois_generation')),
                         ('evalkey_upload', ('relin_upload', 'galois_upload'))]:
        out['poseidon_q32_omp1'][label] = distribution([
            sum(j['stages'][s]['wall_ms'] for s in names)/1000 for j in base])
    out['poseidon_q32_omp1']['gpu_key_payload_bytes'] = sum(
        base[0]['gpu_keys'][s]['field_payload_bytes'] for s in ('relin', 'galois'))
    out['poseidon_q32_omp1']['checks'] = [j['checks'] for j in base]
out['thread_diagnostics'] = {}
for name in ('poseidon-q32-omp7-passive', 'poseidon-q32-omp7-active',
             'poseidon-q32-default64', 'poseidon-q32-omp64-active'):
    path = root/name/'probe.json'
    if path.exists():
        j = json.loads(path.read_text())
        out['thread_diagnostics'][name] = {'galois_generation_s': j['stages']['galois_generation']['wall_ms']/1000,
                'omp_max_threads': j['environment']['omp_max_threads'],
                'affinity_cpu_count': j['environment']['affinity_cpu_count']}
out['parameter_diagnostics'] = {}
for name in ('poseidon-q50-full', 'poseidon-q50-sparse'):
    path = root/name/'probe.json'
    if path.exists():
        j = json.loads(path.read_text())
        out['parameter_diagnostics'][name] = {'build_and_upload_s': j['stages']['gpu_parameters_build_and_upload']['wall_ms']/1000,
                **{k: j['gpu_parameters'][k] for k in ('mode', 'level_count', 'allocated_bytes', 'seven_root_families_total_bytes')}}
out['native_profiles'] = {}
for name in ('native-off-omp1', 'native-qp-trace-omp1'):
    path = root/name/'probe.json'
    if path.exists():
        j = json.loads(path.read_text())
        out['native_profiles'][name] = {**j,
            'phases': traces(path.parent/'stderr.log', 'BOOT_SETUP_PHASE '),
            'key_upload_profiles': traces(path.parent/'stderr.log', 'KEYUPLOAD_PROFILE '),
            'qp_upload_profiles': traces(path.parent/'stderr.log', 'QP_UPLOAD_PROFILE ')}
cerium = []
for rep in (1, 2, 3):
    path = root/f'cerium-q32-r{rep}/stdout.log'
    if path.exists():
        j = json.loads(path.read_text())
        cerium.append(j)
if cerium:
    for j in cerium:
        assert j['ok'] and j['verified_host_exact'] and j['verified_gpu_exact']
        assert j['actual_payload_bytes'] == 452984832 and j['actual_keypair_count'] == 18
        assert j['temporary_files_removed'] and not j['secret_key_serialized']
    out['cerium_q32'] = {s: distribution([j['timing_ms'][s]/1000 for j in cerium])
        for s in cerium[0]['timing_ms'] if cerium[0]['timing_ms'][s] is not None}
    for label, names in [('reload_to_device', ('deserialize_evalkeys_including_host_alloc', 'device_pool_alloc', 'h2d')),
                         ('generate_serialize_reload_to_device', ('generate_evalkeys', 'serialize_evalkeys_no_fsync',
                            'deserialize_evalkeys_including_host_alloc', 'device_pool_alloc', 'h2d'))]:
        out['cerium_q32'][label] = distribution([sum(j['timing_ms'][s] for s in names)/1000 for j in cerium])
    out['cerium_q32_raw'] = cerium
print(json.dumps(out, indent=2))
