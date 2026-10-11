#!/usr/bin/env python3
"""Cross-check compiler JSON/binary exports using the production runtime reader."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--runtime-tool', type=Path, required=True)
    parser.add_argument('--ir', type=Path, required=True)
    parser.add_argument('--reference-plan', type=Path, required=True)
    parser.add_argument('--compiler-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compiler, tool, ir = [p.resolve() for p in (args.compiler, args.runtime_tool, args.ir)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.reference_plan.read_text())
    target, spec = plan['target'], plan['target']['operator_spec']
    options = (f'plan-id={plan["plan_id"]} target-id={target["target_id"]} '
               f'capability-version={target["capability_version"]} operator-spec-id={spec["id"]} '
               f'operator-spec-version={spec["version"]} operator-spec-sha256={spec["source_sha256"]} '
               f'context-id={plan["values"][0]["context"]} device-count={target["device_counts"][0]} '
               'boot-profile=poseidon-gpu-host-boot-v1 boot-implementation=decrypt_reencrypt '
               'inline-payload-max-bytes=4096 bundle-format=pack report-io=true')
    reports = {}
    paths = {}
    for fmt in ('json', 'binary'):
        prefix = output / fmt
        command = [str(compiler), str(ir),
                   f'-p=builtin.module(func.func(emit-runtime-plan{{prefix={prefix} plan-format={fmt} {options}}}))',
                   '--ckks-config=' + str(args.compiler_profile.resolve()), '--mlir-print-op-on-diagnostic=false', '-o', '/dev/null']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        (output / f'{fmt}.log').write_text(result.stderr)
        reports[fmt] = json.loads(next(line[len('runtime-plan-io '):] for line in result.stderr.splitlines()
                                      if line.startswith('runtime-plan-io ')))
        paths[fmt] = next(output.glob(f'{fmt}.*.runtime-plan.{"json" if fmt == "json" else "bin"}'))
    comparison = subprocess.run([str(tool), 'compare-production-json', str(paths['json']), str(paths['binary'])],
                                check=True, capture_output=True, text=True)
    reports['comparison'] = json.loads(comparison.stdout)
    assert reports['comparison']['all_fields_equal']
    assert reports['binary']['plan_hash_seconds'] == 0
    manifest_json = json.loads(next(output.glob('json.*.bundle/manifest.json')).read_text())
    manifest_bin = next(output.glob('binary.*.bundle/manifest.bin')).read_bytes()
    import struct
    assert manifest_bin[:8] == b'CKKSMF01'
    version, size = struct.unpack_from('<II', manifest_bin, 8)
    assert version == 1
    metadata = json.loads(manifest_bin[16:16 + size])
    count, = struct.unpack_from('<Q', manifest_bin, 16 + size)
    assert count == len(manifest_json['blobs'])
    assert len(manifest_bin) == 24 + size + 48 * count
    entries = []
    for offset in range(24 + size, len(manifest_bin), 48):
        digest, position, length = struct.unpack_from('<32sQQ', manifest_bin, offset)
        entries.append({'content': 'sha256:' + digest.hex(), 'offset': position, 'byte_length': length})
    metadata['blobs'] = entries
    assert metadata == manifest_json
    reports['all_manifest_fields_equal'] = True
    assert not list(output.glob('*.tmp'))
    # Repeated output is deterministic, and a blocked temporary record path
    # must report failure without damaging the already published entry point.
    published = paths['binary'].read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert paths['binary'].read_bytes() == published
    blocked = Path(str(output / 'binary') + '.' + paths['binary'].name.split('.', 1)[1].removesuffix('.runtime-plan.bin') + '.records.tmp')
    blocked.mkdir()
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and 'cannot open binary record file' in failed.stderr
    assert paths['binary'].read_bytes() == published and blocked.is_dir()
    blocked.rmdir()
    reports['publication_failure_preserved_output'] = True
    (output / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
