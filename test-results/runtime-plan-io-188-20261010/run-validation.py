"""Validate the completed remote export; do not regenerate or reschedule IR."""
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

root = Path('/home/xuming/poseidon-runtime-io-20261010')
original = Path('/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1')
old = original / 'artifacts/qwen24-unit-plaintext/gpu-plans/gpu4'
new = root / 'artifacts/gpu4'
stem = 'qwen24._hecate_qwen25_24layer'
tool = root / 'build-o2/runtime_plan_io_benchmark'
results = root / 'results'


def limit():
    resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
    os.nice(10)


def benchmark(name, *arguments):
    command = [str(tool), *map(str, arguments)]
    start = time.monotonic()
    with (results / (name + '.json')).open('w') as output, \
            (results / (name + '.log')).open('w') as errors:
        process = subprocess.run(command, stdout=output, stderr=errors,
                                 preexec_fn=limit)
    (results / (name + '-process.json')).write_text(json.dumps({
        'command': command, 'returncode': process.returncode,
        'elapsed_seconds': time.monotonic() - start,
        'address_space_limit_bytes': 32 * 1024**3, 'nice': 10,
    }, indent=2) + '\n')
    process.check_returncode()
    return json.loads((results / (name + '.json')).read_text())


export_process = json.loads((results / 'stage-c-gpu4-export-process.json').read_text())
if export_process['returncode'] != 0:
    raise RuntimeError('export did not finish successfully')
export_lines = (results / 'stage-c-gpu4-export.log').read_text().splitlines()
reports = [json.loads(line.removeprefix('runtime-plan-io '))
           for line in export_lines if line.startswith('runtime-plan-io ')]
if len(reports) != 1:
    raise RuntimeError('expected one complete export report')
export = reports[0]
(results / 'stage-c-gpu4-export.json').write_text(json.dumps(export, indent=2) + '\n')

benchmark('stage-c-manifest-equivalence', '--compare-manifests',
          old / (stem + '.bundle/manifest.json'), new / (stem + '.bundle/manifest.json'))
benchmark('stage-c-plan-equivalence', '--compare-plans',
          old / (stem + '.runtime-plan.json'), new / (stem + '.runtime-plan.json'))
original_loaded = benchmark('stage-c-original-load-o2', '--plan',
                            old / (stem + '.runtime-plan.json'))
if original_loaded['source_sha256'] != 'sha256:e7b0f54fb91ca0a606b021ba9dd7b7ac8f47e345aa8fedfdd440933575e15547':
    raise RuntimeError('original plan source digest changed')
loaded = benchmark('stage-c-compact-load-verify-o2', '--plan',
                   new / (stem + '.runtime-plan.json'),
                   original / 'profiles/operator-spec.json', new / (stem + '.bundle'))
if loaded['source_sha256'] != export['plan_sha256']:
    raise RuntimeError('independently loaded plan digest differs from writer')
if loaded['bytes'] != export['plan_bytes']:
    raise RuntimeError('independently loaded plan byte count differs from writer')
if loaded['values'] != export['values'] or \
        sum(loaded[phase] for phase in ('initialization', 'execution', 'finalization')) != export['instructions']:
    raise RuntimeError('independently loaded plan record counts differ from writer')

# This independent integrity check is outside loader performance measurements.
# Python keeps the small manifest DOM, never the plan or all blob payloads.
start = time.monotonic()
manifest_bytes = (new / (stem + '.bundle/manifest.json')).read_bytes()
manifest_sha256 = 'sha256:' + hashlib.sha256(manifest_bytes).hexdigest()
manifest = json.loads(manifest_bytes)
del manifest_bytes
blob_bytes = 0
for entry in manifest['blobs']:
    path = new / (stem + '.bundle/data') / (entry['content'][7:] + '.bin')
    digest = hashlib.sha256()
    length = 0
    with path.open('rb') as data:
        while block := data.read(64 * 1024):
            digest.update(block)
            length += len(block)
    if length != entry['byte_length'] or 'sha256:' + digest.hexdigest() != entry['content']:
        raise RuntimeError('new bundle blob length or digest mismatch: ' + str(path))
    blob_bytes += length
if len(manifest['blobs']) != export['unique_blobs'] or blob_bytes != export['bundle_bytes']:
    raise RuntimeError('new bundle totals differ from writer')
(results / 'stage-c-bundle-integrity.json').write_text(json.dumps({
    'verified_blobs': len(manifest['blobs']), 'blob_bytes': blob_bytes,
    'manifest_sha256': manifest_sha256, 'seconds': time.monotonic() - start,
    'input_buffer_bytes': 64 * 1024,
    'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
}, indent=2) + '\n')
