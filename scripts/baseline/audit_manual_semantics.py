"""Read-only numeric/trace/artifact audit of the saved manual CPU cohorts.

Writes a new audit under results; does not generate candidates, call a provider,
execute FHE, install dependencies, change historical results or read private keys.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile

from audit_agent_lineage import RESULTS, metadata, read, require
from manual_semantic_evidence import audit_manual_case

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
COHORTS = (
    ('rotation', 'rotation-batch-i9qaiev4', 7),
    ('multi_input', 'schema3-golden-batch-ht5no6ir', 5),
    ('conv_pool', 'spatial-golden-batch-z8dy657u', 11),
    ('wide_linear', 'wide-linear-goldens-02h0bwbh', 2),
    ('grouped_dilated_conv', 'grouped-spatial-goldens-ffbvf58_', 8),
    ('broadcast', 'broadcast-golden-batch-cq9e4lbd', 10),
    ('encrypted_zero', 'zero-golden-batch-w9vuz53i', 9),
    ('explicit_power', 'explicit-power-goldens-a3ovzyp7', 4),
    ('activation_approximation', 'approximation-golden-v08aa_ux', 2),
)
ARITHMETIC = (
    ('candidate-replay-gq2y3dwl', True, 'subneg_linear/native_golden.py'),
    ('candidate-replay-yovkx6a5', True, 'cipher_subtract/golden.py'),
    ('candidate-replay-8vm9gphy', False, 'cipher_subtract/wrong_order.py'),
)


def collect(root=RESULTS):
    rows, sources = [], []
    for name, passed, golden in ARITHMETIC:
        rows.append(dict(cohort='arithmetic', run=str(root/name), expected_pass=passed,
                         golden=str(BASE/'golden_cases'/golden)))
    for label, directory, count in COHORTS:
        path = root/directory/'report.json'
        batch, digest = metadata(path, root)
        require(batch['status'] == 'passed' and batch.get('agent_calls', 0) == 0 and
                batch['poseidon_gpu_validated'] is False and len(batch['cases']) == count,
                'Cohort count/status/backend changed')
        sources.append(dict(report=str(path), sha256=digest, cohort=label, cases=count))
        for item in batch['cases']:
            require(type(item['counterexample']) is bool and item['matched_expected'] is True,
                    'Missing explicit expected manual outcome')
            command = item['command']
            require(command.count('--golden-file') == 1, 'Missing unique manual-golden source')
            golden = command[command.index('--golden-file')+1]
            row = dict(cohort=label, run=item['run'], expected_pass=not item['counterexample'], golden=golden)
            if label == 'activation_approximation':
                row['approximation_row'] = item
            rows.append(row)
    require(len({r['run'] for r in rows}) == len(rows), 'Duplicate execution in manual denominator')
    return rows, sources


def approximation_check(row, root):
    import numpy as np
    from approximation_errors import relu_and_quadratic, decompose
    run = Path(row['run'])
    with np.load(io.BytesIO(read(run/'arrays.npz', root)[0]), allow_pickle=False) as arrays:
        inputs = arrays['inputs'].copy()
    actual_raw, digest = read(run/'attempt-00/output/decrypted.npy', root)
    actual = np.load(io.BytesIO(actual_raw), allow_pickle=False)
    saved = row['approximation_row']
    require(saved['inputs'] == inputs.tolist() and saved['decrypted_sha256'] == digest,
            'Approximation inputs/decrypted evidence changed')
    f, p = relu_and_quadratic(inputs.reshape(-1).tolist())
    errors = decompose(f, p, actual.reshape(-1).tolist())
    require(errors == saved['errors'] and errors['ckks_passed'] is row['expected_pass'] and
            errors['original_semantic_equivalence_verified'] is False,
            'Approximation decomposition must use actual saved ciphertext outputs')
    return dict(approximation_max_error=errors['approximation_error']['max_absolute_error'],
                execution_max_error=errors['ckks_execution_error']['max_absolute_error'],
                original_semantic_equivalence_verified=False)


def operator_matrix(cases):
    from model_graph import OPS
    from model_semantic_coverage import OP_SEMANTICS
    from dsl_semantic_inventory import SEMANTICS
    rows = []
    for op in sorted(OPS):
        positive = [c for c in cases if c['status'] == 'verified' and c['expected_numerical_pass'] and
                    op in c['model_semantics']['operators']]
        negative = [c for c in cases if c['status'] == 'verified' and not c['expected_numerical_pass'] and
                    op in c['model_semantics']['operators']]
        semantic = OP_SEMANTICS[op]
        rows.append(dict(operator=op, semantic=semantic, support_status=SEMANTICS[semantic]['status'],
            positive_evidence=[c['evidence'] for c in positive],
            counterexample_programs_involving_operator=[c['evidence'] for c in negative],
            tests=SEMANTICS[semantic]['tests'], tests_executed_by_this_audit=False,
            observation='finite_manual_graph_cases' if positive else 'not_observed_here',
            fault_localization_proved_by_operator_presence=False))
    return rows


def audit_all(root=RESULTS):
    rows, sources = collect(root)
    cases = []
    for row in rows:
        try:
            source, source_hash = read(Path(row['golden']), ROOT, limit=65536)
            result = audit_manual_case(Path(row['run']), row['expected_pass'], root=root,
                                       golden_source=source.decode('utf-8'))
            result.update(status='verified', cohort=row['cohort'], golden=row['golden'], golden_sha256=source_hash)
            if 'approximation_row' in row:
                result['approximation'] = approximation_check(row, root)
        except (ValueError, OSError, KeyError, TypeError) as error:
            result = dict(status='failed', cohort=row['cohort'], evidence=row['run'],
                          expected_numerical_pass=row['expected_pass'], diagnostic=str(error))
        cases.append(result)
    require(all(metadata(Path(s['report']), root)[1] == s['sha256'] for s in sources),
            'Cohort source changed during audit')
    good = [c for c in cases if c['status'] == 'verified']
    positives = [c for c in good if c['expected_numerical_pass']]
    return dict(status='passed' if len(good) == len(cases) else 'failed',
        scope='saved_manual_cpu_fixture_evidence_not_live_Agent_or_new_execution',
        planned=len(rows), verified=len(good), failures=len(rows)-len(good),
        correct_goldens=len(positives), detected_counterexamples=len(good)-len(positives),
        input_batches=sum(c['input_batches'] for c in good),
        compared_values=sum(c['compared_values'] for c in good),
        max_positive_absolute_error=max((c['max_absolute_error'] for c in positives), default=None),
        legacy_key_probe_not_recorded=sum(c['rotation_key_evidence']=='not_recorded_in_legacy_v1' for c in good),
        sources=sources, cases=cases, operator_matrix=operator_matrix(cases),
        analysis_source_hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in
            ('manual_semantic_evidence.py','audit_manual_semantics.py','model_graph.py','spatial_ops.py',
             'candidate_contract.py','seal_cpu_golden.py','seal_artifact_gate.py')},
        agent_calls=0, new_fhe_executions=0, original_reports_changed=False,
        all_semantics_verified=False, all_goal_requirements_complete=False,
        limitations=['Counts are executions, not unique model families or an Agent success rate.',
                     'A counterexample containing an operator does not localize its bug to that operator.',
                     'The wide manual golden covers width8; widths5..8 rule evidence is separate.',
                     'Approximate ReLU is not declared equivalent to original ReLU.',
                     'Historical v1 runs lack later key-file probe fields; this is reported, not fabricated.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.plan:
        rows, sources = collect()
        print(json.dumps(dict(mode='plan_only', cases=len(rows), sources=sources, agent_calls=0), indent=2))
        return 0
    from hecate_python_env import VENV, WORK, enter_nix
    require(Path.cwd().resolve() == ROOT, 'Run from source root')
    if not args.inside:
        command = 'LD_LIBRARY_PATH='+chr(36)+'HECATE_PYTHON_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'), str(Path(__file__).resolve()), '--inside'])
        return enter_nix(command, seconds=150)
    require(Path(sys.prefix) == VENV, 'Requires pinned numerical environment')
    folder = Path(tempfile.mkdtemp(prefix='manual-semantics-audit-', dir=WORK/'results'))
    report = audit_all()
    (folder/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Manual semantic audit: '+str(folder/'report.json'), flush=True)
    print(json.dumps({k:report[k] for k in ('status','planned','verified','failures','correct_goldens',
        'detected_counterexamples','input_batches','compared_values','max_positive_absolute_error')}, indent=2))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
