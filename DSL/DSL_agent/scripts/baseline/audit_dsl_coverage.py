"""Re-audit saved CPU outputs and bind grammar coverage to traced payloads.

No API calls, no execution of candidate Python, no new FHE executions. An absent
feature is a coverage gap, not a failed model and not an unsupported operator.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys
import tempfile

from audit_agent_lineage import RESULTS, audit, metadata, read, require
from candidate_contract import TASK_RULES, request_input_names, validate_candidate
from dsl_grammar_coverage import FEATURES, analyze_source
from dsl_semantic_inventory import SEMANTICS


def source_evidence(run, report):
    """Use the frozen payload actually submitted to tracing, not a loose .py file."""
    run = Path(run)
    attempt = next(a for a in reversed(report['attempts']) if a.get('numerically_correct'))
    prefix = f"attempt-{attempt['index']:02d}"
    name = prefix + '/trace-payload.json'
    require(name in report['frozen_hashes'], 'Missing frozen trace payload')
    payload, digest = metadata(run/name, run)
    require(digest == report['frozen_hashes'][name], 'Trace payload hash mismatch')
    request, request_hash = metadata(run/'request.json', run)
    require(request_hash == report['frozen_hashes']['request.json'], 'Request hash mismatch')
    require(payload['request'] == request, 'Traced request differs from frozen request')
    candidate = payload['candidate']
    check = validate_candidate(candidate, request)
    require(check == attempt['static_check'], 'Recomputed static check differs from recorded check')
    source = candidate['hecate_source']
    sidecar = run/prefix/'candidate.py'
    if sidecar.exists():
        require(read(sidecar, run)[0].decode('utf-8') == source,
                'Candidate sidecar differs from actual traced payload')
    features = analyze_source(source, request['public_constants'],
                              request['layout']['output_ciphertexts'],
                              contract=TASK_RULES[request['task']][0],
                              input_names=request_input_names(request))
    trace = attempt['trace']
    require(trace['frontend'] == 'real_Hecate' and trace['candidate_python_executed'] is False and
            trace['request_id'] == request['request_id'], 'Missing safe real Hecate tracing evidence')
    return dict(trace_payload_sha256=digest, request_sha256=request_hash, grammar=features,
                hevm_opcode_counts=attempt['artifact_gate']['opcode_counts'])


def summarize_cases(cases):
    rows = []
    for feature, semantic in FEATURES.items():
        present = [r['case'] for r in cases if feature in r['grammar']['present_counts']]
        live = [r['case'] for r in cases if feature in r['grammar']['output_dependency_counts']]
        rows.append(dict(feature=feature, semantic=semantic,
                         support_status=SEMANTICS[semantic]['status'],
                         tests=SEMANTICS[semantic]['tests'],
                         coverage_test='test_dsl_grammar_coverage.GrammarCoverageTests',
                         present_cases=present, output_dependency_cases=live,
                         evidence_status='observed_in_cpu_passing_agent_programs' if live else
                                         'not_observed_in_audited_agent_outputs'))
    missing = [r['feature'] for r in rows if not r['output_dependency_cases']]
    return dict(features=rows, feature_partition_count=len(rows),
                observed_feature_partitions=len(rows)-len(missing), missing_feature_partitions=missing,
                all_feature_partitions_observed=not missing, all_semantics_verified=False,
                semantics={name: dict(row) for name, row in SEMANTICS.items()},
                scope='Local restricted Hecate v0-v4 grammar, not the entire upstream DSL',
                auxiliary_encrypted_zero_cases=sum(c['grammar'].get('auxiliary_encrypted_zero', False) for c in cases),
                limitations=[
                    'Finite feature partitions do not enumerate all grammar combinations or inputs.',
                    'SSA reachability does not establish numerical influence or survival through optimization.',
                    'Compiler opcodes are observed separately; AST and HEVM counts need not match.',
                    'Referenced test names are traceability links, not tests executed by this audit.',
                    'Manual golden evidence is separate; absence here does not erase it.',
                    'Model approximation error is not CKKS execution error; no implicit activation rewrite.',
                ])


def audit_coverage(latest, root=RESULTS):
    numerical = audit(latest, root)
    cases = []
    for case in numerical['cases']:
        run = Path(case['evidence'])
        report, digest = metadata(run/'report.json', root)
        require(digest == case['report_sha256'], 'Case changed during audit')
        cases.append(dict(case, **source_evidence(run, report)))
    result = summarize_cases(cases)
    return dict(result, schema=1, status='audited', audit_only=True,
                backend='upstream_SEAL_HEVM_CPU', new_api_calls=0, new_fhe_executions=0,
                poseidon_gpu_required=False, all_goal_requirements_complete=False,
                numerical_coverage_status=numerical['status'], source_batches=numerical['batches'],
                model_cases=len(cases), family_passes=numerical['family_passes'],
                input_executions=numerical['input_executions'], compared_values=numerical['compared_values'],
                max_absolute_error=numerical['max_absolute_error'], cases=cases)


def audit_cohorts(reports, root=RESULTS):
    """Union evidence, not success rates; shared ancestor runs count only once."""
    require(type(reports) in (list, tuple) and 1 <= len(reports) <= 8, 'One to eight cohort reports required')
    paths = [Path(path).resolve() for path in reports]
    require(len(set(paths)) == len(paths), 'Duplicate cohort report')
    combined, cohorts, sources = {}, [], {}
    for path in paths:
        checked = audit_coverage(path, root)
        cohorts.append(dict(report=str(path), numerical_coverage_status=checked['numerical_coverage_status'],
                            model_cases=checked['model_cases'], family_passes=checked['family_passes']))
        for source in checked['source_batches']:
            previous = sources.setdefault(source['report'], source)
            require(previous == source, 'Shared source batch changed during audit')
        for case in checked['cases']:
            identity = str(Path(case['evidence']).resolve())
            if identity in combined:
                prior = {k: v for k, v in combined[identity].items() if k != 'cohort_reports'}
                require(prior == case, 'Shared candidate evidence changed during audit')
                combined[identity]['cohort_reports'].append(str(path))
            else:
                combined[identity] = dict(case, cohort_reports=[str(path)])
    cases = list(combined.values())
    # Distinguish equal user IDs in unrelated cohorts. Never use an ID as run identity.
    labeled = [dict(case, case=Path(case['evidence']).name + ':' + case['case']) for case in cases]
    coverage = summarize_cases(labeled)
    return dict(coverage, schema=1, status='audited', audit_only=True,
                backend='upstream_SEAL_HEVM_CPU', new_api_calls=0, new_fhe_executions=0,
                poseidon_gpu_required=False, all_goal_requirements_complete=False,
                interpretation='Heterogeneous evidence union; not a single-configuration success rate.',
                cohorts=cohorts, source_batches=list(sources.values()), model_cases=len(cases),
                input_executions=sum(case['input_executions'] for case in cases),
                compared_values=sum(case['compared_values'] for case in cases),
                max_absolute_error=max((case['max_absolute_error'] for case in cases), default=None),
                numerical_coverage_status='all_cohorts_complete' if all(
                    c['numerical_coverage_status'] == 'coverage_complete' for c in cohorts) else 'incomplete_cohorts',
                cases=cases)


def main():
    from hecate_python_env import ROOT, VENV, enter_nix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--additional-report', type=Path, action='append', default=[],
                        help='Re-audit and union another cohort; never pool success rates')
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from configured source root')
    reports = [args.report, *args.additional_report]
    require(len(reports) <= 8 and all(path.resolve().is_relative_to(RESULTS) for path in reports),
            'Too many reports or report outside results root')
    if not args.inside:
        command = ('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' +
                   shlex.join([str(VENV/'bin/python'), str(Path(__file__).resolve()),
                               str(args.report.resolve()), '--inside', *[
                                   flag for path in args.additional_report for flag in ('--additional-report', str(path.resolve()))]]))
        return enter_nix(command, seconds=240 * len(reports))
    require(Path(sys.prefix) == VENV, 'Requires pinned NumPy environment')
    report = audit_cohorts(reports) if args.additional_report else audit_coverage(args.report)
    out = Path(tempfile.mkdtemp(prefix='dsl-grammar-audit-', dir=RESULTS))
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in
                     ('features', 'semantics', 'cases', 'source_batches')}, indent=2))
    print('DSL grammar audit:', out)
    return 0  # Coverage gaps are data, not an infrastructure failure.


if __name__ == '__main__':
    raise SystemExit(main())
