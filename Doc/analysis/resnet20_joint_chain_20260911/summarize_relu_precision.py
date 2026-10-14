"""Summarize completed diagnostic logs without hiding failed stage guards."""
import argparse
import json
from pathlib import Path


def fields(tokens):
    result = {}
    for token in tokens:
        name, value = token.split('=', 1)
        try:
            result[name] = float(value)
        except ValueError:
            result[name] = value
    return result


def parse(path):
    run = dict(log=str(path), checks={})
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == 'CHECK':
            run['checks'][words[1]] = fields(words[2:])
        elif words[0] in ('SOURCE_REFERENCE', 'RESULT', 'FIXTURE', 'APPROXIMATION'):
            run[words[0].lower()] = fields(words[1:])
    if 'result' not in run:
        raise ValueError(f'incomplete run: {path}')
    return run


def summarize(paths):
    runs = [parse(p) for p in paths]
    if any(r['fixture'] != runs[0]['fixture'] for r in runs):
        raise ValueError('cannot merge different fixtures')
    selected = ['encrypted_input', 'P15_stage1', 'P15_stage2', 'P27_stage3',
                'ReLU_final_vs_original_polynomial', 'ReLU_final_vs_ideal_ReLU',
                'stage1_vs_polynomial_of_decrypted_input',
                'stage2_vs_polynomial_of_decrypted_input',
                'stage3_vs_polynomial_of_decrypted_input',
                'ReLU_final_vs_polynomial_of_decrypted_input']
    summary = {}
    for name in selected:
        rows = [r['checks'][name] for r in runs if name in r['checks']]
        if rows:
            summary[name] = dict(runs=len(rows), q=rows[0]['q'], log_scale=rows[0]['log_scale'],
                max_abs=minmax(rows, 'max_abs'), rms=minmax(rows, 'rms'))
    final = 'ReLU_final_vs_original_polynomial'
    return dict(scope='ISOLATED_RELU_ACCURACY_ONLY', security_approved=False,
                bootstrap_tested=False, full_network_tested=False,
                global_parameters=dict(N=65536, Q=50, P=25, dnum=2, secret_hamming_weight=192),
                input_description='32768 real slots: [-1,1] grid and corners/near-zero powers of two',
                intermediate_tolerance_is_conservative_diagnostic_not_network_acceptance=True,
                final_output_tolerance=1e-5,
                all_final_outputs_pass=all(r['checks'][final]['max_abs']<=1e-5 for r in runs),
                all_stage_guards_pass=all(r['result']['precision']=='PASS' for r in runs),
                summary=summary, runs=runs)


def minmax(rows, name):
    return dict(min=min(r[name] for r in rows), max=max(r[name] for r in rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('logs', nargs='+', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    report = summarize(args.logs)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'}, indent=2))
