"""Re-audit the four frozen construction cohorts; never calls a paid provider."""
import argparse
import json
from pathlib import Path
import shlex
import tempfile

from audit_agent_lineage import RESULTS, require


def audit():
    from audit_construction_batch import audit as construction
    from audit_object_arithmetic_batch import audit as arithmetic
    from audit_scalar_conversion_batch import audit as scalar
    from audit_object_unary_batch import audit as unary

    supplements = [RESULTS / ('agent-deepseek-' + name) / 'report.json' for name in
                   ('iuzitx7d', 'hs3u8qvi', '9tcer_u5', '7s3yo4bv', '4bsrkh3q')]
    cohorts = {
        'v19_public_construction': construction(RESULTS / 'agent-batch-64es49_l', supplements),
        'v20_object_arithmetic': arithmetic(RESULTS / 'agent-batch-6u_j0fug'),
        'v21_scalar_conversion': scalar(RESULTS / 'agent-batch-1codrfmv'),
        'v22_object_unary': unary(RESULTS / 'agent-batch-l_nd65sn'),
    }
    rows = []
    for name, evidence in cohorts.items():
        require(evidence['status'] == 'covered', name + ': incomplete coverage')
        matrix = evidence['feature_matrix']
        require(all(item.get('cases', item.get('passing_cases')) for item in matrix),
                name + ': feature without a passing real Agent candidate')
        rows.append(dict(cohort=name,
            passed=evidence.get('passed', evidence.get('passing_cases')),
            features=len(matrix), compared_values=evidence['compared_values'],
            max_absolute_error=evidence['max_absolute_error']))
    require([r['passed'] for r in rows] == [30, 10, 15, 8], 'Frozen case counts changed')
    require([r['features'] for r in rows] == [117, 10, 19, 12], 'Frozen feature counts changed')
    count = sum(r['compared_values'] for r in rows)
    return dict(schema=1, status='covered', cohorts=cohorts, summary=rows,
        selected_cases=sum(r['passed'] for r in rows), compared_values=count,
        max_absolute_error=max(r['max_absolute_error'] for r in rows),
        weighted_mae=sum(c['weighted_mae'] * c['compared_values'] for c in cohorts.values()) / count,
        new_api_calls=0, new_fhe_executions=0, poseidon_gpu_validated=False,
        all_upstream_semantics_proven=False,
        limitations=[
            'Feature counts are version-specific observations, not distinct FHE operators.',
            'Do not add feature counts and claim coverage of the full upstream DSL.',
            'Execution structure evidence is weaker than output-sensitive evidence.',
            'Frozen targeted implementation forms are not unseen-model generalization.',
        ])


def main():
    from hecate_python_env import ROOT, VENV, enter_nix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Wrong workspace')
    if not args.inside:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' + shlex.join(
            [str(VENV / 'bin/python'), str(Path(__file__).resolve()), '--inside'])
        return enter_nix(command, seconds=600)
    report = audit()
    output = Path(tempfile.mkdtemp(prefix='recent-construction-audit-', dir=RESULTS))
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'cohorts'}, indent=2))
    print('Audit evidence:', output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
