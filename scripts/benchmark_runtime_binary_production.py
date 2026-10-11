#!/usr/bin/env python3
"""Benchmark the production reader using the already checked full Qwen binary."""
import argparse
import json
import resource
import statistics
import subprocess
from pathlib import Path


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (32 << 30, 32 << 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / 'production-binary-20261011'
    work.mkdir()
    original = root / 'full-qwen-binary-corrected-20261011/qwen24.v3.plan.bin'
    packed = root / 'artifacts/gpu4/qwen24._hecate_qwen25_24layer.packed.bundle'
    output = work / 'qwen24.runtime-plan.bin'
    bundle = work / 'qwen24.bundle'
    bundle.mkdir()
    (bundle / 'data.bin').symlink_to(packed / 'data.bin')
    converter = root / 'build-o2/runtime_binary_io_experiment'
    benchmark = root / 'build-o2/runtime_plan_io_benchmark'
    spec = args.spec.resolve()
    inputs = {'source_binary': str(original), 'source_bytes': original.stat().st_size,
              'packed_bundle': str(packed), 'operator_spec': str(spec),
              'source': 'existing checked binary, no full JSON conversion or Qwen recompilation',
              'threads': [1, 4, 8], 'runs_4_and_8': 3,
              'cache_condition': 'shared server, no cache eviction',
              'whole_file_hashing': False, 'blob_payload_hashing': False,
              'address_space_limit_bytes': 32 << 30}
    (work / 'inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')

    def invoke(label, tool, *arguments):
        print(f'starting {label}', flush=True)
        with (work / f'{label}.json').open('w') as stdout, (work / f'{label}.stderr').open('w') as stderr:
            subprocess.run(['nice', '-n', '10', str(tool), *map(str, arguments)],
                           check=True, stdout=stdout, stderr=stderr, timeout=180, preexec_fn=limits)
        result = json.loads((work / f'{label}.json').read_text())
        print(json.dumps({'label': label, **result}), flush=True)
        return result

    invoke('promote-plan', converter, 'promote-plan', original, output)
    invoke('promote-manifest', converter, 'promote-manifest', packed / 'manifest.json', bundle / 'manifest.bin')
    comparison = invoke('field-comparison', converter, 'compare-production-plan', original, output)
    assert comparison['all_fields_equal'] and comparison['hash_seconds'] == 0
    rows = []
    for run, order in enumerate(([1, 4, 8], [8, 4], [4, 8]), 1):
        for threads in order:
            result = invoke(f'load-{threads}-run-{run}', benchmark, '--plan', output, spec, bundle, threads)
            assert result['binary'] and result['hash_seconds'] == 0 and result['source_sha256'] == ''
            assert result['values'] == 11064667 and result['execution'] == 22149239
            assert result['capabilities'] == 4 and result['keys'] == 42566
            rows.append({'threads': threads, 'run': run, **result})
    resident = invoke('resident-4', benchmark, '--plan', output, spec, bundle, 4, 16 << 30)
    assert resident['bundle_resident_bytes'] == 14701969408
    summary = {}
    for threads in (1, 4, 8):
        selected = [r for r in rows if r['threads'] == threads]
        summary[str(threads)] = {'runs': len(selected)}
        for field in ('load_seconds', 'verify_seconds', 'bundle_index_seconds', 'total_load_verify_index_seconds', 'peak_rss_bytes'):
            values = [r[field] for r in selected]
            summary[str(threads)][field] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
    summary['resident_4'] = resident
    (work / 'loads.json').write_text(json.dumps(rows, indent=2) + '\n')
    (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
