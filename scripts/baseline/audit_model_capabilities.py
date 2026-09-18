"""Join real Agent/FHE evidence to user-model capabilities, with no paid calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

from audit_agent_lineage import RESULTS, metadata, require
from audit_dsl_coverage import audit_cohorts
from model_semantic_coverage import analyze_graph, summarize_models


def audit_models(paths, root=RESULTS):
    checked = audit_cohorts(paths, root)
    graph_rows, legacy = [], []
    for case in checked['cases']:
        run = Path(case['evidence'])
        model, digest = metadata(run/'model.json', run)
        if model.get('schema') == 1:
            legacy.append(dict(case=case['case'], evidence=str(run), reason='catalog_descriptor_not_free_graph'))
            continue
        request, _ = metadata(run/'request.json', run)
        require(request['model'] == model, 'Model differs from the model sent to Agent')
        graph_rows.append(dict(case=case['case'], evidence=str(run), model_sha256=digest,
                              model_semantics=analyze_graph(model)))
    summary = summarize_models(graph_rows)
    return dict(summary, schema=1, audit_only=True, new_api_calls=0, new_fhe_executions=0,
                analysis_source_hashes={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ('audit_model_capabilities.py','model_semantic_coverage.py','model_graph.py')},
                cohorts=checked['cohorts'], total_agent_cases=checked['model_cases'],
                graph_cases=len(graph_rows), legacy_catalog_cases=len(legacy),
                graph_case_details=graph_rows, legacy_cases=legacy,
                missing_graph_operators=[r['operator'] for r in summary['operator_matrix'] if not r['observed_cases']],
                numerical_coverage_status=checked['numerical_coverage_status'],
                poseidon_gpu_required=False, all_goal_requirements_complete=False,
                limitations=['Legacy catalog evidence is retained but not reclassified as user-defined graphs.',
                    'Observed parameters do not exhaust the allowed shape/weight/depth combinations.',
                    'Dead model nodes do not contribute to output-reachable operator coverage.',
                    'Linked test names are not proof that those tests were executed in this audit.'])


def main():
    from hecate_python_env import ROOT, VENV, enter_nix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='+')
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT and 1 <= len(args.reports) <= 8, 'Requires source root and one to eight reports')
    require(all(p.resolve().is_relative_to(RESULTS) for p in args.reports), 'Report outside results root')
    if not args.inside:
        command = ('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' +
            shlex.join([str(VENV/'bin/python'), str(Path(__file__).resolve()), '--inside', *map(str,args.reports)]))
        return enter_nix(command, seconds=240*len(args.reports))
    require(Path(sys.prefix) == VENV, 'Requires pinned numerical dependencies')
    report = audit_models(args.reports)
    out = Path(tempfile.mkdtemp(prefix='model-capability-audit-', dir=RESULTS))
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('graph_case_details','legacy_cases','operator_matrix','feature_matrix')}, indent=2))
    print('Model capability audit:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
