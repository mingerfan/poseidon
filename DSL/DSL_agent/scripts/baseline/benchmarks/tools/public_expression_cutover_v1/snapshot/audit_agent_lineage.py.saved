"""Audit cumulative saved Agent evidence, never generate or execute a model.

Different providers/prompts/retries are a cumulative coverage ledger, not one
homogeneous experiment. Recheck immutable inputs/artifacts and compare saved
decrypted arrays against saved independent references at the frozen tolerance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

from run_agent_batch import batch_plan, case_metrics, load_failed_source, row_family
from interrupted_batch import load_remaining_source

from workspace_paths import RESULTS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, root, limit=8*1024**2):
    path, root = Path(path), Path(root).resolve()
    require(not path.is_symlink() and path.resolve().is_relative_to(root),
            'Evidence path outside root or symbolic link')
    require(path.is_file() and path.stat().st_size <= limit, 'Evidence size/type')
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def metadata(path, root):
    raw, digest = read(path, root)
    return json.loads(raw), digest


def catalog_for_report(path, report, root=RESULTS):
    custom = report.get('custom_manifest')
    if custom is None:
        return [r['descriptor'] for r in batch_plan(True)[0]]
    require(type(custom) is dict and set(custom) == {'file', 'sha256', 'case_count'} and
            custom['file'] == 'custom-manifest.json', 'Invalid custom cohort metadata')
    snapshot = Path(path).parent/'custom-manifest.json'
    read(snapshot, root, limit=1024**2)
    from custom_batch_manifest import load_manifest
    rows, _, _ = load_manifest(snapshot, custom['sha256'])
    require(type(custom['case_count']) is int and custom['case_count'] == len(rows),
            'Custom cohort count mismatch')
    return [row['descriptor'] for row in rows]


def lineage(latest, root=RESULTS):
    reports, seen = [], set()
    path = Path(latest).resolve()
    report, _ = metadata(path, root)
    catalog = catalog_for_report(path, report, root)
    while True:
        require(path not in seen and len(seen) < 16, 'Cyclic or excessive lineage')
        seen.add(path)
        report, digest = metadata(path, root)
        require(catalog_for_report(path, report, root) == catalog, 'Lineage cohort changed')
        require(report['status'] in ('passed', 'completed_with_failures', 'interrupted'),
                'Only terminal or sealed interrupted reports may be audited')
        selection = report.get('selection')
        expected = catalog
        if selection:
            parent = Path(selection['source_report']).resolve()
            _, parent_hash = metadata(parent, root)
            require(parent_hash == selection['source_sha256'], 'Parent hash mismatch')
            kind = selection['kind']
            require(kind in ('previous_failures_only', 'previous_failures_and_unfinished'),
                    'Unknown lineage selection')
            loader = load_remaining_source if kind == 'previous_failures_and_unfinished' else load_failed_source
            _, expected = loader(catalog, parent, root)
        rows = report['cases']
        require([r['descriptor'] for r in rows] == expected, 'Case selection/order mismatch')
        if report['status'] == 'interrupted':
            # Validates original checkpoint hash, case identity and worker-stop record.
            load_remaining_source(catalog, path, root)
        for row in rows:
            require(row['status'] in ('passed', 'failed', 'pending'), 'Invalid row status')
            if row['status'] == 'pending':
                require(report['status'] == 'interrupted' and 'metrics' not in row,
                        'Unsealed pending row')
            else:
                require(type(row['metrics'].get('passed')) is bool and
                        row['metrics']['passed'] == (row['status'] == 'passed'),
                        'Row outcome mismatch')
        if report['status'] == 'passed':
            require(all(r['status'] == 'passed' for r in rows), 'False batch success')
        reports.append((path, report, digest))
        if not selection:
            return list(reversed(reports))
        path = parent


def audit_case(row, root=RESULTS):
    import numpy as np
    from seal_cpu_golden import compare
    run = Path(row['evidence']).resolve()
    require(run.parent == Path(root).resolve(), 'Case outside results root')
    report, report_hash = metadata(run/'report.json', root)
    descriptor, _ = metadata(run/'model.json', run)
    require(descriptor == row['descriptor'], 'Model descriptor mismatch')
    require(case_metrics(report) == row['metrics'], 'Saved case metrics mismatch')
    require(report.get('llm_generation_validated') is True and report.get('agent_calls', 0) > 0,
            'Not a real Agent-generated pass')
    require(report.get('backend') == 'upstream_SEAL_HEVM_CPU' and
            report.get('poseidon_gpu_validated') is False, 'Unexpected backend claim')
    require(report.get('tolerance') == dict(atol=1e-5, rtol=1e-4), 'Tolerance changed')
    params = report['parameters']
    require(params['seal_version'] == '4.0.0' and params['polynomial_degree'] == 32768 and
            params['security_check'] == 'tc128' and params['parameters_set'] is True and
            params['modulus_bits'] == [60]*14, 'Security profile changed')
    frozen = report['frozen_hashes']
    require({'arrays.npz', 'weights.npz', 'request.json', 'model.json'} <= set(frozen),
            'Missing immutable inputs')
    for name, expected in frozen.items():
        require(read(run/name, run)[1] == expected, 'Frozen input hash mismatch')
    attempt = next(a for a in reversed(report['attempts']) if a.get('numerically_correct'))
    output = run/f"attempt-{attempt['index']:02d}"/'output'
    hashes = attempt['artifact_hashes']
    require({'lowered._hecate_golden.hevm', '_hecate_golden.cst', 'lowered.ckks.mlir'} <= set(hashes),
            'Missing compiler artifact')
    for name, expected in hashes.items():
        require(read(output/name, output)[1] == expected, 'Compiler artifact hash mismatch')
    execution = attempt['execution']
    require(execution['encrypted_execution'] is True and execution['bootstrap_executed'] is False and
            execution['input_batches'] == 4, 'Missing four real encrypted input executions')
    require(execution['rotation_key_check']['actual_key_file_verified'] is True, 'Keys not verified')
    decrypted = output/'decrypted.npy'
    _, decrypted_hash = read(decrypted, output)
    with np.load(run/'arrays.npz', allow_pickle=False) as data:
        reference = data['reference']
        request, _ = metadata(run/'request.json', run)
        from zero_evidence import verify_zero_execution
        from cipher_abi import has_zero_argument, execution_options
        verify_zero_execution(request['layout'], execution)
        if has_zero_argument(request['layout']):
            from candidate_contract import request_input_names
            # request_input_names validates the versioned zero ABI, including
            # newer construction contracts; do not hard-code only request v5.
            require(len(request_input_names(request)) == execution['encrypted_input_count'], 'Zero ABI arity mismatch')
            options = execution_options(request['layout'])
            logical = options['logical_inputs'];period=options.get('input_period',4)
            require(data['inputs'].shape == ((4,period) if logical == 1 else (4,logical,period)),
                    'Auxiliary zero must not replace or enlarge the original user input arrays')
            require(all(type(report.get(k)) is str and len(report[k]) == 64 for k in
                    ('metadata_observer_sha256', 'metadata_observer_source_sha256')), 'Missing native observer provenance')
        if descriptor.get('schema')==5:
            from packed_input_abi import ABI,validate_request,pack_inputs,gate_options
            from model_graph import evaluate_reference
            from seal_artifact_gate import inspect_artifacts
            from candidate_contract import request_rotations,request_input_names
            plan=validate_request(request)
            require(execution.get('execution_abi')==ABI and
                    execution.get('input_slot_period')==plan['slot_period'],'Packed runtime period mismatch')
            require(np.array_equal(data['inputs'],pack_inputs(data['logical_inputs'],descriptor['input_shape'])),
                    'Packed input bytes do not match original logical input')
            independent=np.asarray([evaluate_reference(descriptor,x.tolist()) for x in data['logical_inputs']])
            require(np.array_equal(reference,independent),'Packed reference does not match original model')
            if len(request['layout']['output_shape'])>1:
                from packed_input_abi import decode_output
                logical=np.load(output/'decrypted-logical.npy',allow_pickle=False)
                require(read(output/'decrypted-logical.npy',output)[1]==attempt['logical_output_sha256'],
                        'Logical output hash mismatch')
                require(np.array_equal(data['reference_logical'],independent.reshape(4,*request['layout']['output_shape'])) and
                        np.array_equal(logical,decode_output(np.load(decrypted,allow_pickle=False),request['layout'])),
                        'Logical output shape/order mismatch')
            gate=inspect_artifacts((output/'lowered._hecate_golden.hevm').read_bytes(),
                (output/'_hecate_golden.cst').read_bytes(),rotation_steps=request_rotations(request),
                expected_inputs=len(request_input_names(request)),**gate_options(request['layout']))
            require(execution['rotation_key_check']['required_steps']==gate['rotation_steps'],
                    'Packed executed rotations differ from artifact')
            require(all(type(report.get(k)) is str and len(report[k])==64 for k in
                ('packed_observer_sha256','packed_observer_source_sha256','packed_key_helper_sha256',
                 'packed_key_source_sha256')),'Missing packed observer/key provenance')
    actual = np.load(decrypted, allow_pickle=False)
    comparison = compare(actual, reference, 1e-5, 1e-4)
    require(comparison['passed'], 'Recomputed numerical comparison failed')
    recorded = attempt['comparison']
    for name in ('actual', 'reference', 'compared_values', 'elementwise_pass', 'max_absolute_error'):
        require(comparison[name] == recorded[name], 'Saved numerical evidence mismatch')
    return dict(case=descriptor['id'], family=row_family(row), evidence=str(run),
                report_sha256=report_hash, decrypted_sha256=decrypted_hash,
                max_absolute_error=comparison['max_absolute_error'],
                compared_values=comparison['compared_values'],
                input_executions=execution['input_batches'],
                private_keys_retained=(run/'private-keys').exists(),
                provider=report['provider_metrics']['service_provider'],
                model=report['provider_metrics']['model'])


def audit(latest, root=RESULTS):
    reports = lineage(latest, root)
    successes, batch_rows = {}, []
    for path, report, digest in reports:
        batch_rows.append(dict(report=str(path), sha256=digest, status=report['status'],
                               service_provider=report.get('service_provider', 'deepseek'),
                               model=report['model'], loopback_proxy_port=report.get('loopback_proxy_port', 0),
                               passed=sum(r['status'] == 'passed' for r in report['cases'])))
        for row in report['cases']:
            if row['status'] == 'passed':
                case = row['descriptor']['id']
                require(case not in successes, 'Already successful case repeated in failure-only lineage')
                successes[case] = audit_case(row, root)
    catalog = [dict(descriptor=d) for d in catalog_for_report(reports[0][0], reports[0][1], root)]
    missing = [r['descriptor']['id'] for r in catalog if r['descriptor']['id'] not in successes]
    rows = list(successes.values())
    return dict(status='coverage_complete' if not missing else 'coverage_incomplete',
                audit_only=True, new_api_calls=0, new_fhe_executions=0,
                interpretation='Cumulative heterogeneous coverage, NOT a single-configuration success rate.',
                poseidon_gpu_validated=False, all_goal_requirements_complete=False,
                planned=len(catalog), passed=len(rows), missing=missing, batches=batch_rows,
                family_passes=dict(Counter(r['family'] for r in rows)), cases=rows,
                compared_values=sum(r['compared_values'] for r in rows),
                input_executions=sum(r['input_executions'] for r in rows),
                max_absolute_error=max((r['max_absolute_error'] for r in rows), default=None),
                private_key_directories_retained=sum(r['private_keys_retained'] for r in rows))


def main():
    from hecate_python_env import ROOT, VENV, enter_nix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from the configured source root')
    require(args.report.resolve().is_relative_to(RESULTS), 'Report outside results root')
    if not args.inside:
        command = ('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' +
                   shlex.join([str(VENV/'bin/python'), str(Path(__file__).resolve()),
                               str(args.report.resolve()), '--inside']))
        # No keys or provider environment crosses this offline audit boundary.
        return enter_nix(command, seconds=180)
    require(Path(sys.prefix) == VENV, 'Requires the pinned NumPy environment')
    report = audit(args.report)
    out = Path(tempfile.mkdtemp(prefix='agent-lineage-audit-', dir=RESULTS))
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases', 'batches')}, indent=2))
    print('Audit evidence:', out)
    return 0 if report['status'] == 'coverage_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
