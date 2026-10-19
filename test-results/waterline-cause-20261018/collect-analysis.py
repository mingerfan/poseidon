#!/usr/bin/env python3
"""Collect existing GPU evidence and the CPU scheduler ablation; no new GPU runs."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

out = Path(__file__).resolve().parent
repo = out.parents[1]
prior = out.parent / 'waterline-legacy-20261018'
cpu = json.loads((out / 'local-cpu/summary.json').read_text())
raw_cpu = [json.loads(p.read_text()) for p in (out / 'local-cpu').glob('n*-r*-*.json')]
assert len(raw_cpu) == cpu['verified_samples'] == 63
assert all(x['passed'] and x['compute_calls'] == 2*x['weights']-1
           and x['execution_tasks'] == 5*x['weights']-2 for x in raw_cpu)

def reports(folder, pattern):
    samples = []
    for p in sorted((prior / folder).glob(pattern + '/summary.json')):
        x = json.loads(p.read_text())['samples']
        assert len(x) == 1 and x[0]['passed']
        samples.append(dict(source=str(p.relative_to(repo)), **x[0]))
    assert len(samples) == 3, (pattern, len(samples))
    return samples

def distribution(values):
    return dict(median=statistics.median(values), samples=values)

eager = {}
for variant in ('legacy', 'current'):
    samples = reports('server-runs', f'server-pressure-r*-{variant}-eager')
    eager[variant] = dict(sources=[s['source'] for s in samples])
    for key, fn in {
        'online_ms': lambda s: s['online_ms'],
        'forward_ms': lambda s: s['forward_ms'],
        'initialization_ms': lambda s: s['init_ms'],
        'compute_api_ms': lambda s: s['report']['runtime_timing']['compute_submission_including_boot_ms'],
        'outside_compute_api_ms': lambda s: s['online_ms'] - s['report']['runtime_timing']['compute_submission_including_boot_ms'],
    }.items():
        eager[variant][key] = distribution([fn(s) for s in samples])

waterline_samples = reports('server-runs', 'server-pressure-r*-current-waterline')
waterline = dict(sources=[s['source'] for s in waterline_samples])
for key, fn in {
    'online_ms': lambda s: s['online_ms'],
    'forward_ms': lambda s: s['forward_ms'],
    'bundle_read_ms': lambda s: s['report']['runtime_timing']['bundle_read_ms'],
    'prepare_api_ms': lambda s: s['report']['runtime_timing']['encode_submission_ms'],
    'compute_api_ms': lambda s: s['report']['runtime_timing']['compute_submission_including_boot_ms'],
    **{key: lambda s, key=key: sum(d[key] for d in s['report']['device_measurements']) for key in (
        'gpu_encode_calls', 'gpu_encode_ms', 'gpu_raw_h2d_bytes', 'gpu_raw_h2d_ms', 'consumer_wait_ms')},
}.items():
    waterline[key] = distribution([fn(s) for s in waterline_samples])
assert waterline['gpu_encode_calls']['samples'] == [4096]*3
assert waterline['gpu_raw_h2d_bytes']['samples'] == [268435456]*3
assert all(s['report']['preparation_measurements']['cpu_prepare_calls'] == 0 for s in waterline_samples)

auto_samples = reports('local-runs', 'local-qproj-r*-current-auto-w8')
auto = dict(sources=[s['source'] for s in auto_samples], online_ms=distribution([s['online_ms'] for s in auto_samples]))
for key in ('cpu_prepare_calls', 'cpu_encode_ms', 'cpu_plain_h2d_bytes'):
    auto[key] = distribution([s['report']['preparation_measurements'][key] for s in auto_samples])

profile = json.loads((prior / 'nsys-mlp.json').read_text())
for value in profile['profiles'].values():
    del value['kernel_shapes']

files = [repo/'third_party/ckks-runtime/runtime/runtime.hpp',
         repo/'src/poseidon/runtime_api/poseidon_gpu_api.cpp',
         out/'cpu-probe.cpp', out/'build-probes.py', out/'run-probes.py']
sha = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert (repo/'third_party/ckks-runtime/runtime/runtime.hpp').read_bytes() == (prior/'headers/current.hpp').read_bytes()
# Production hashes are recorded, not edited. Historical hashes live in the prior provenance file.
build = repo / 'build-waterline-cause-20261018-v3'
shutil.copyfile(build/'commands.json', out/'local-build-commands.json')
shutil.copyfile(build/'prepared_direct/include/runtime/runtime.hpp', out/'prepared-direct.hpp')
analysis = dict(
    scope='Historical GPU reports plus new local CPU-only Runtime controls. No production changes. No new V100 profile: SSH returned SOCKSv5 Host unreachable.',
    timing_caveats=[
        'Historical comparison swaps exact Runtime headers only, keeping the current GPU backend; it is not a historical whole-repository rebuild.',
        'Eager online excludes prior encoding/upload; Waterline online includes runtime preparation. Both include final output synchronization and drain.',
        'Compute API is host wall time, not GPU kernel time. Its complement includes dispatch, Release, metadata and final GPU drain, not pure CPU or pure idle time.',
        'CPU controls use scalar tokens and instant communication, with no CUDA or weight I/O. Their durations cannot be subtracted from V100 timings.',
        'Context switch counts cover the full Runtime call (including initialization/setup), whereas online_ms covers online execution.',
        'Prepared-direct preserves new metadata source construction and bypasses central per-task coordination. Its residual difference is not a clean metadata-only cost measurement.',
        'GPU encoding/upload/consumer CUDA-event intervals include stream scheduling waits. They overlap other timings and cannot be summed into an exclusive wall-time breakdown.',
        'Bundle reading includes validation, copying and conversion; OS file cache was not flushed. It is not pure physical disk time.',
        'Nsight kernel span covers first-to-last kernel in the process, not Runtime online. Inter-kernel gaps can include copies; profiles are excluded from latency medians.',
        'q_proj is synthetic proxy work, not a fresh full-Qwen measurement.',
    ],
    server_pressure_execution=dict(weights=4096, arithmetic_tasks=8191, release_tasks=12287, total_tasks=20478),
    server_pressure_eager=eager,
    server_pressure_waterline=waterline,
    local_cpu_control=cpu,
    local_qproj_auto8=auto,
    local_mlp_nsys=profile,
    source_sha256=sha,
)
(out/'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2)+'\n')
print('Verified 63 CPU samples, 6 V100 eager reports, 3 V100 Waterline reports and 3 q_proj Auto/8 reports.')
print('V100 online delta_ms:', eager['current']['online_ms']['median']-eager['legacy']['online_ms']['median'])
print('V100 compute API delta_ms:', eager['current']['compute_api_ms']['median']-eager['legacy']['compute_api_ms']['median'])
