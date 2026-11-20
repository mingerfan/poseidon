"""Publish closed S2 cases and actual progress; never dispatch experiments."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from .. import paths
from .r10_paper_metrics import write_json
from .s2_heir_serial import same_birth

MODELS = ('resnet20', 'vgg16', 'alexnet', 'bert_tiny')


def read(path):
    return json.loads(Path(path).read_text())


def locked_rows(path):
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        return [json.loads(x) for x in stream if x.strip()]


def actual_processes():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors='replace') for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
            if not argv or any('python3 - <<' in x for x in argv):
                continue
            if ('pt_agent.experiments.s2_heir_compile' in argv or 'pt_agent.experiments.s2_heir_serial' in argv or
                    Path(argv[0]).name == 'heir-opt' and any('s2-heir-' in x for x in argv)):
                raw = (p / 'stat').read_text()
                fields = raw[raw.rfind(')') + 2:].split()
                found.append(dict(pid=int(p.name), birth=fields[19], state=fields[0], executable=Path(argv[0]).name))
        except (OSError, ValueError):
            pass
    return found


def refresh(args, helper, current_view):
    utc = datetime.now(timezone.utc).isoformat()
    queue_file = args.queue / ('result.json' if (args.queue / 'result.json').is_file() else 'progress.json')
    queue = read(queue_file)
    alive = same_birth(args.controller_pid, args.controller_birth)
    all_rows = locked_rows(args.paper / 'runs.jsonl')
    rows = {x['model']: x for x in all_rows if x.get('task') == 'S2-HEIR' and x.get('system') == 'heir'}
    science = actual_processes()
    observer_state = 'closed' if not alive and (args.queue / 'completion.json').is_file() else 'active'
    progress = dict(schema='s2-heir-actual-progress-v1', observed_utc=utc,
                    controller=dict(pid=args.controller_pid, birth=args.controller_birth, alive=alive),
                    queue_source=str(queue_file), queue_status=queue['status'], observer_source_state=observer_state,
                    current_scientific_processes=science,
                    task_source=str(args.task), current_required_extension='S2-HEIR',
                    models=[dict(model=m, status=rows[m]['status'], run_id=rows[m]['run_id'],
                                 compile_s=rows[m]['compile_s'], boots_compiled=rows[m]['boots_compiled'],
                                 source=rows[m]['source']) if m in rows else
                            dict(model=m, status=next((j['status'] for j in queue.get('jobs', []) if j['model'] == m), 'pending'),
                                 compile_s=None, boots_compiled=None) for m in MODELS],
                    new_scientific_actions=0, new_API=0, new_CKKS=0)
    helper.refresh(SimpleNamespace(runs=args.paper / 'runs.jsonl', supersessions=args.paper / 'supersessions.jsonl',
                                   current=args.paper / 'current_runs.json'), current_view)
    write_json(args.mirror / 'current_runs.json', read(args.paper / 'current_runs.json'))
    selector = read(args.paper / 'paper_cell_selector.json')
    for cell in selector['cells']:
        if cell['system'] != 'heir' or cell['model'] not in MODELS:
            continue
        if 'pre_S2_cell' not in cell:
            cell['pre_S2_cell'] = dict(cell)
        model = cell['model']
        if model not in rows:
            cell.update(cell_state='pending', compilation_status=None, reason='S2 restored HEIR by the current task; actual progress is in S2_HEIR_PROGRESS.json')
            continue
        row = rows[model]
        case = Path(row['source']).parent
        result, completion = read(case / 'result.json'), read(case / 'completion.json')
        if not completion.get('registry_persisted') or not completion.get('required_artifacts_persisted'):
            raise ValueError('S2 source not durably closed')
        if row['status'] != result['status']:
            raise ValueError('S2 paper and scientific source outcomes differ')
        for key in ('compile_s', 'boots_compiled', 'boot_sites_static', 'exec_ops', 'pred_latency_s', 'peak_rss_gib', 'budget_s'):
            cell[key] = row.get(key)
        cell.update(cell_state='terminal', compilation_status=row['status'], selected_run_id=row['run_id'],
                    scientific_run_id=row['run_id'], expected_scientific_run_id=row['run_id'],
                    source_result=row['source'], source_completion=str(case / 'completion.json'),
                    clock_kind='actual_compile_only' if row['compile_s'] is not None else 'frontend_admission_only',
                    generation_e2e_s=None, actual_compile_only_case_s=row.get('complete_compile_case_s'),
                    output_kind='native Lattigo MLIR' if row['status'] == 'ok' else None,
                    intended_output_kind='native Lattigo MLIR', compiler_attempts=row.get('compiler_attempts', 0),
                    boot_kind=row.get('boot_kind'), parameter_contract=row.get('parameter_contract'),
                    prediction_scope=row.get('prediction_unavailable_reason'),
                    source_binding=dict(result_status=result['status'], completion_status=completion['status'],
                                        registry_persisted=True, required_artifacts_persisted=True),
                    reason=row.get('status_detail') or 'One native HEIR compile completed; serialized straight-line IR bootstrap inventory is separate from CKKS execution')
    selector['generated_utc'] = utc
    selector['current_user_policy'].update(HEIR_new_work='S2 restored; one native attempt per CNN and existing BERT entry check',
                                           current_scientific_priority='S2-HEIR' if alive else 'S2-HEIR terminal sources published')
    selector['S2_HEIR_current_progress'] = str(args.paper / 'S2_HEIR_PROGRESS.json')
    index = read(args.paper / 'current_experiment_evidence.json')
    for row in index['rows']:
        if row['system'] == 'heir' and row['model'] in MODELS:
            selected = next(x for x in selector['cells'] if x['system'] == row['system'] and x['model'] == row['model'])
            row.update({k: v for k, v in selected.items() if k in row or k in ('selected_run_id', 'scientific_run_id', 'source_completion', 'source_binding', 'boot_kind', 'parameter_contract')})
    index.update(last_metadata_revision_utc=utc, status='additional_S2_HEIR_work_active' if alive else 'S2_HEIR_terminal_sources_published',
                 cell_states=dict(Counter(x['cell_state'] for x in selector['cells'])),
                 compilation_outcomes=dict(Counter(x['compilation_status'] for x in selector['cells'] if x['cell_state'] == 'terminal')),
                 unique_terminal_sources_verified=len([x for x in selector['cells'] if x['cell_state'] == 'terminal']))
    index['linked_evidence']['S2_HEIR_current_progress'] = str(args.paper / 'S2_HEIR_PROGRESS.json')
    index['cancelled_work'] = [x for x in index.get('cancelled_work', []) if x != 'HEIR new work']
    delivery = read(args.paper / 'EXPERIMENT_DELIVERY.json')
    delivery['previous_completed_scope_status'] = delivery.get('previous_completed_scope_status', delivery['status'])
    delivery.update(status=index['status'], last_metadata_revision_utc=utc,
                    current_scientific_background_processes=len(science),
                    S2_HEIR_extension=progress,
                    prior_audit_scope='282-check audit covers the pre-S2 completed snapshot; S2 durable source checks are separate',
                    matrix_states=index['cell_states'], measured_outcomes=index['compilation_outcomes'])
    delivery['cancelled_work'] = [x for x in delivery['cancelled_work'] if x != 'HEIR new work']
    delivery['links']['S2_HEIR_current_progress'] = str(args.paper / 'S2_HEIR_PROGRESS.json')
    delivery['prior_complete_delivery_snapshot'] = str(args.out / 'before-C-EXPERIMENT_DELIVERY.json')
    for name, value in [('S2_HEIR_PROGRESS.json', progress), ('paper_cell_selector.json', selector),
                        ('current_experiment_evidence.json', index), ('EXPERIMENT_DELIVERY.json', delivery)]:
        write_json(args.paper / name, value)
        write_json(args.mirror / name, value)
    policy = read(args.paper / 'paper_comparison_policy.json')
    policy.update(updated_utc=utc, HEIR_new_work='S2 restored by the current task; one native attempt per CNN',
                  current_required_experiment_scope_complete=observer_state == 'closed' and len(rows) == 4,
                  current_S2_HEIR_progress=str(args.paper / 'S2_HEIR_PROGRESS.json'))
    for root in (args.paper, args.mirror):
        write_json(root / 'paper_comparison_policy.json', policy)
        a = read(root / 'A_GROUP_PROGRESS.json')
        a.update(updated_utc=utc, S2_HEIR=progress,
                 current_required_experiment_scope_complete=observer_state == 'closed' and len(rows) == 4,
                 HEIR_new_work=policy['HEIR_new_work'], background=science,
                 background_observed_utc=utc)
        write_json(root / 'A_GROUP_PROGRESS.json', a)
    write_json(args.mirror / 'LATEST_EXECUTION_STATUS.json',
               dict(schema='s2-heir-current-execution-status-v1', observed_utc=utc,
                    source_progress=str(args.paper / 'S2_HEIR_PROGRESS.json'),
                    current_scientific_processes=science, controller_alive=alive,
                    actual_heavy_activity=science, new_scientific_actions=0))
    return progress, observer_state == 'closed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('paper', 'mirror', 'queue', 'task', 'out', 'view-helper', 'view-function'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--controller-pid', type=int, required=True)
    p.add_argument('--controller-birth', required=True)
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('s2_existing_current_view_helper', args.view_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    current_view = helper.load_view_function(args.view_function)
    for root, tag in ((args.paper, 'C'), (args.mirror, 'D')):
        for name in ('paper_cell_selector.json', 'current_experiment_evidence.json', 'EXPERIMENT_DELIVERY.json',
                     'current_runs.json', 'paper_comparison_policy.json', 'A_GROUP_PROGRESS.json'):
            (args.out / ('before-' + tag + '-' + name)).write_text((root / name).read_text())
    started = time.monotonic()
    while True:
        progress, closed = refresh(args, helper, current_view)
        write_json(args.out / 'progress.json', progress)
        if closed or time.monotonic() - started > 9 * 3600:
            write_json(args.out / 'result.json', dict(status='ok' if closed else 'observation_window_elapsed',
                                                    last_observation=progress, new_scientific_actions=0))
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
