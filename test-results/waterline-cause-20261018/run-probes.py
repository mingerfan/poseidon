#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--suite', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=7)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
samples = []
variants = ['legacy', 'current', 'prepared_direct']
for weights in (1024, 4096, 16384):
    for repeat in range(args.repeats):
        shift = repeat % len(variants)
        for variant in variants[shift:] + variants[:shift]:
            command = [str((args.build / variant / 'probe').resolve()),
                str((args.suite / 'eager.json').resolve()),
                str((args.suite / 'operator-spec.json').resolve()), str(weights)]
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            result = json.loads(run.stdout)
            assert result['passed'] and result['compute_calls'] == 2 * weights - 1
            assert result['execution_tasks'] == 5 * weights - 2
            samples.append(dict(variant=variant, repeat=repeat, command=command, **result))
            (args.output / f'n{weights}-r{repeat}-{variant}.json').write_text(json.dumps(samples[-1], indent=2)+'\n')
rows = []
for weights in (1024, 4096, 16384):
    for variant in variants:
        selected = [s for s in samples if s['weights'] == weights and s['variant'] == variant]
        metrics = ['online_ms', 'compute_api_ms', 'main_collections', 'worker_collections',
                   'voluntary_context_switches', 'involuntary_context_switches', 'process_cpu_seconds']
        row = dict(weights=weights, variant=variant, repeats=len(selected),
            **{key: dict(median=statistics.median(s[key] for s in selected),
                         min=min(s[key] for s in selected), max=max(s[key] for s in selected)) for key in metrics})
        rows.append(row)
        print(weights, variant, row['online_ms'], 'switches', row['voluntary_context_switches'], flush=True)
(args.output / 'summary.json').write_text(json.dumps(dict(
    scope='CPU scalar-token control using the real Runtime. No CUDA, CKKS encoding, weight I/O or GPU event queries. Prepared-direct retains new metadata construction but bypasses per-task dispatcher handoffs. These are not FHE/GPU latency measurements.',
    verified_samples=len(samples), comparisons=rows), indent=2)+'\n')
