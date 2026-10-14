#!/usr/bin/env python3
"""One preparation/conversion, then three alternating metadata load pairs."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / 'binary-io-experiment'
    data = args.data_dir.resolve() if args.data_dir else work / 'data'
    binary = root / 'build-o2/runtime_binary_io_experiment'
    source = work / '1gpu-stream._hecate_MLP.runtime-plan.json'
    spec = work / 'operator-spec.json'
    manifest = root / 'artifacts/gpu4/qwen24._hecate_qwen25_24layer.packed.bundle/manifest.json'

    def invoke(*arguments):
        output = subprocess.check_output(['nice', '-n', '10', str(binary), *map(str, arguments)], text=True)
        return json.loads(output)

    inputs = work / 'inputs.jsonl'
    with inputs.open('w') as output:
        subprocess.run(['python3', str(work / 'prepare_binary_io_experiment.py'), str(source), str(data),
                        '--copies', '81', '324'], stdout=output, check=True)
    conversions = []
    for copies in [81, 324]:
        path = data / f'plan-c{copies}.json'
        conversions.append({'input': str(path), **invoke('convert-plan', path, path.with_suffix('.bin'))})
        print(json.dumps(conversions[-1]), flush=True)
    manifest_binary = data / 'manifest.bin'
    conversions.append({'input': str(manifest), **invoke('convert-manifest', manifest, manifest_binary)})
    print(json.dumps(conversions[-1]), flush=True)
    (work / 'conversions.json').write_text(json.dumps(conversions, indent=2) + '\n')

    with (work / 'loads.jsonl').open('w') as output:
        for copies in [81, 324]:
            path = data / f'plan-c{copies}.json'
            for run in range(1, 4):
                modes = ['plan-json', 'plan-binary'] if run % 2 else ['plan-binary', 'plan-json']
                for mode in modes:
                    report = invoke(mode, path if mode == 'plan-json' else path.with_suffix('.bin'), spec)
                    report.update({'copies': copies, 'run': run})
                    output.write(json.dumps(report) + '\n'); output.flush()
                    print(json.dumps(report), flush=True)
        for run in range(1, 4):
            modes = ['manifest-json', 'manifest-binary'] if run % 2 else ['manifest-binary', 'manifest-json']
            for mode in modes:
                report = invoke(mode, manifest if mode == 'manifest-json' else manifest_binary)
                report['run'] = run
                output.write(json.dumps(report) + '\n'); output.flush()
                print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
