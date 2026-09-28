"""Read-only r6 replay audit. Run with locked Python and PYTHONPATH=scripts/baseline.

Reports main rule-generated tasks and supplemental manual programs separately.
No generation, compilation, credential access or changes to existing evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from benchmark_graph import digest, require, samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from benchmark_runner import ROOT, DEFAULT, load
from semantic_benchmark_execution import runtime_sources
from audit_semantic_benchmark import audit as audit_main
from unified_graph_contract import validate_request, validate_candidate
from unified_public_coverage import verify_trace_coverage
from candidate_contract import request_input_names, request_rotations
from cipher_abi import artifact_options
from compiler_configuration import verify_artifact_configuration
from seal_artifact_gate import inspect_artifacts
from seal_cpu_golden import compare, PROFILE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_files(folder, hashes):
    for name, expected in hashes.items():
        path = folder / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe evidence path')
        require(path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink(),
                'Indirect evidence path')
        require(sha(path) == expected, 'Changed evidence: ' + name)


def verify_candidate(folder):
    report = read(folder / 'report.json')
    require(report['status'] == 'passed' and report['agent_calls'] == 0,
            'Expected manual passed candidate')
    require(not report['llm_generation_validated'], 'Not an Agent result')
    verify_files(ROOT / 'scripts/baseline', report['source_hashes'])
    verify_files(folder, report['frozen_hashes'])
    require(read(folder / 'key-cleanup-outcome.json')['complete'], 'Key cleanup incomplete')
    request = read(folder / 'request.json'); validate_request(request)
    model = read(folder / 'model.json')
    require(model == request['model'], 'Model/request mismatch')
    passed = [a for a in report['attempts'] if a.get('status') == 'passed']
    require(len(passed) == 1, 'Expected one successful manual attempt')
    attempt = passed[0]; attempt_dir = folder / ('attempt-%02d' % attempt['index'])
    out = attempt_dir / 'output'
    verify_files(out, attempt['artifact_hashes'])
    require(all(attempt[k] for k in ('compiled','executed','numerically_correct')),
            'Missing real execution')
    require(attempt['trace']['frontend'] == 'real_Hecate', 'Not real tracing')
    require(read(out / 'execution.json') == attempt['execution'] and
            attempt['execution']['encrypted_execution'], 'Execution binding')
    checked = validate_candidate(dict(schema=1, request_id=request['request_id'],
                                      hecate_source=(attempt_dir / 'candidate.py').read_text()), request)
    coverage = verify_trace_coverage(checked['construction_exercise'],
                                     read(out / 'public-construction-events.json'))
    require(coverage == attempt['public_expression_coverage'], 'Directed trace mismatch')
    expected_inputs = []; references = []
    period = request['layout']['input_slot_period']
    for inputs in samples(model, 4):
        first = mathematical(model, inputs); second = torch_reference(model, inputs)
        for name in first:
            np.testing.assert_allclose(first[name], second[name], atol=1e-12, rtol=1e-12)
        references.append(np.concatenate([first[o['name']].reshape(-1) for o in model['outputs']]))
        expected_inputs.append(np.stack([np.pad(inputs[i['name']].reshape(-1),
                              (0, period-math.prod(i['shape']))) for i in model['inputs']]))
    packed = np.stack(expected_inputs)
    if len(model['inputs']) == 1:
        packed = packed[:,0,:]
    with np.load(folder / 'arrays.npz', allow_pickle=False) as data:
        require(np.array_equal(data['inputs'], packed) and
                np.array_equal(data['reference'], np.stack(references)), 'Reference/input mismatch')
        comparison = compare(np.load(out / 'decrypted.npy', allow_pickle=False),
                             data['reference'], 1e-5, 1e-4)
    require(comparison == attempt['comparison'] and comparison['passed'], 'Numerical mismatch')
    gate = inspect_artifacts((out / 'lowered._hecate_golden.hevm').read_bytes(),
                             (out / '_hecate_golden.cst').read_bytes(),
                             rotation_steps=request_rotations(request),
                             expected_inputs=len(request_input_names(request)),
                             **artifact_options(request['layout']))
    verify_artifact_configuration(request, gate, sha(PROFILE))
    require(gate == attempt['artifact_gate'], 'Artifact gate mismatch')
    # Include post-execution arrays/cleanup as well as the originally frozen artifacts.
    files = {str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    return dict(evidence=str(folder), files=files, request_id=request['request_id'],
                model_sha256=digest(model), exercise=request['construction_exercise']['id'],
                comparison=comparison, coverage=coverage, period=period,
                inputs=len(model['inputs']), outputs=len(model['outputs']))


def run(root):
    require(runtime_sources() == read(root/'coordinator-plan.json')['sources'], 'Runtime changed')
    batches = [root / n for n in ('pilot48','contexts-1','contexts-2','old-regression24')]
    original = audit_main(DEFAULT, root.parent/'public-r6-plaintext', batches)
    require(original == read(root/'audit.json'), 'Original main audit changed')
    main = []
    for batch in original['batches']:
        main.extend(verify_candidate(Path(r['evidence'])) for r in batch['summary']['reports'])
    supplemental = root/'supplemental-packed'
    verify_files(ROOT, read(supplemental/'fixture-hashes.json'))
    fixture_dir = Path(__file__).resolve().parent
    specs = read(fixture_dir/'manifest.json')
    records = read(supplemental/'run-summary.json')
    require(len(records) == len(specs) == 3, 'Supplemental denominator changed')
    tasks = {t['id']:t for t in read(DEFAULT/'coverage.json')['directed_tasks']}
    extras = []
    for spec, record in zip(specs, records):
        require(spec['task_id'] == record['task_id'], 'Task order changed')
        task = tasks[spec['task_id']]
        require(task['task_sha256'] == spec['task_sha256'], 'Frozen task changed')
        folder = Path(record['evidence'])
        require(sha(folder/'report.json') == record['report_sha256'], 'Supplemental report changed')
        require(sha(supplemental/(spec['task_id']+'.log')) == record['log_sha256'], 'Log changed')
        verified = verify_candidate(folder)
        require(verified['model_sha256'] == spec['model_sha256'] and
                verified['exercise'] == spec['exercise'], 'Wrong supplemental model/exercise')
        require((folder/'attempt-00/candidate.py').read_text() ==
                (fixture_dir/spec['source_file']).read_text(), 'Manual source changed')
        verified['task_id'] = spec['task_id']; extras.append(verified)
    def totals(items):
        values = sum(x['comparison']['compared_values'] for x in items)
        return dict(passed=len(items), failed=0, skipped=0, compared_values=values,
                    max_absolute_error=max(x['comparison']['max_absolute_error'] for x in items),
                    weighted_mae=sum(x['comparison']['mae']*x['comparison']['compared_values'] for x in items)/values,
                    periods=dict(Counter(x['period'] for x in items)),
                    inputs=dict(Counter(x['inputs'] for x in items)),
                    outputs=dict(Counter(x['outputs'] for x in items)),
                    numeric=sum(not x['coverage']['structural_only'] for x in items),
                    structural=sum(x['coverage']['structural_only'] for x in items))
    return dict(schema=1, audit_source_sha256=sha(Path(__file__)),
                source_snapshot_sha256=sha(root/'frozen-source.tar.gz'),
                original_main_audit_sha256=sha(root/'audit.json'),
                main=totals(main), supplemental_manual=totals(extras),
                main_records=main, supplemental_records=extras,
                default_rule_blocks_retained=3, agent_calls=0,
                all_dsl_semantics_covered=False, formal_equivalence_proven=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing audit')
    result = run(args.root)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('main','supplemental_manual','default_rule_blocks_retained','agent_calls')},indent=2))
