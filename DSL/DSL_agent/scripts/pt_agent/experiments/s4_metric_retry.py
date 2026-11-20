"""One fresh S4 retry after S5, under the updated metric-fix authorization."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace

from .. import paths
from . import s4_qwen_block as s4
from . import s4_qwen_cold as cold
from . import s4_qwen_publication as publication
from . import s4_qwen_post as post
from .a5_a6_rule63_search import guarded_process
from .r10_paper_metrics import write_json
from .s2_heir_serial import record_registry


def publish(args, state):
    state.update(updated_utc=datetime.now(timezone.utc).isoformat(),
        task_source=str(args.task), task_source_mtime_ns=args.task.stat().st_mtime_ns,
        prior_S5=str(args.prior_case), cold_case=str(args.cold_case),
        cutoff_utc='2026-11-18T12:00:00+00:00', additional_cold_attempts_authorized=1)
    write_json(args.out / 'progress.json', state)
    for root in (args.paper, args.mirror):
        write_json(root / 'S4_QWEN_METRIC_RETRY_PROGRESS.json', state)
        delivery = s4.read(root / 'EXPERIMENT_DELIVERY.json')
        delivery.update(current_required_experiment_scope_complete=False,
            S4_metric_retry=state)
        delivery['links']['S4_Qwen_metric_retry_progress'] = str(args.paper / 'S4_QWEN_METRIC_RETRY_PROGRESS.json')
        write_json(root / 'EXPERIMENT_DELIVERY.json', delivery)


def command(args):
    return [sys.executable, '-B', '-m', cold.MODULE,
        '--model', str(args.model), '--inputs', str(args.inputs),
        '--original-info', str(args.original_info), '--out', str(args.cold_case),
        '--checkpoint-out', str(args.checkpoint_out), '--paper-runs', str(args.paper / 'runs.jsonl'),
        '--metadata-template', str(args.metadata_template), '--heldout', '256', '--probe', str(args.probe)]


def main(args):
    gate = s4.read(args.tests)
    if gate.get('status') != 'ok' or gate.get('quality_tests_passed') != 8:
        raise ValueError('The completed generic classification/regression regressions are required')
    if args.cold_case.exists() or args.out.exists():
        raise FileExistsError('Preserve every previous controller or scientific attempt')
    args.out.mkdir()
    source = args.out / 'source'
    source.mkdir()
    base = paths.REPO / 'DSL/DSL_agent/scripts/pt_agent'
    for name in ('quality.py', 'planners/agent.py', 'planners/tuner.py', 'planners/transformer_tuner.py',
                 'experiments/s4_qwen_cold.py', 'experiments/s4_metric_retry.py', 'tests/test_s4_quality_metrics.py'):
        shutil.copyfile(base / name, source / name.replace('/', '_'))
    state = dict(status='waiting_for_all_S5_cases_and_reports', cold_attempts=0,
        calibration_or_candidate_reused=False, tests=str(args.tests), quality_policy='regression_relative<=0.01',
        original_budgets=dict(total_s=10800, search_ceiling_s=9000, tail_reserve_s=1800, tuner_s=2400))
    publish(args, state)
    while not (args.prior_case / 'result.json').is_file():
        if time.time() >= s4.DEADLINE:
            raise TimeoutError('Publication cutoff before all S5 cases closed')
        time.sleep(2)
    prior = s4.read(args.prior_case / 'result.json')
    if len(prior.get('rows', [])) != 3:
        raise ValueError('The three actual CNN cold chains and their reports must close first')
    if s4.remaining_before_deadline(10800) < 10800:
        state.update(status='timeout', cold_attempts=0,
            reason='The submission cutoff leaves less than the required full three-hour cold allocation',
            complete_three_hour_availability_test=False)
        write_json(args.out / 'result.json', state)
        publish(args, state)
        return 1
    pub_args = SimpleNamespace(out=args.out / 'publisher', paper=args.paper, mirror=args.mirror,
                              task=args.task, q1_case=args.cold_case, finalize=False)
    stop = threading.Event()
    failures = []
    def follow():
        try:
            while not (args.cold_case / 'result.json').is_file():
                if stop.wait(.2):
                    return
            publication.main(pub_args)
        except Exception as exc:
            failures.append(type(exc).__name__ + ': ' + str(exc))
            write_json(args.out / 'publication-error.json', dict(error=failures[-1]))
    watcher = threading.Thread(target=follow, daemon=True)
    argv = command(args)
    write_json(args.out / 'cold-command.json', argv)
    state.update(status='running', cold_attempts=1)
    publish(args, state)
    watcher.start()
    try:
        observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
            log=args.out / 'cold.log', resources=args.out / 'resources.jsonl',
            timeout=s4.remaining_before_deadline(10815), mem_gib=24, fsize_gib=16)
        write_json(args.out / 'process.json', observed)
    finally:
        stop.set()
        watcher.join(timeout=15)
    if watcher.is_alive() or failures or not (args.cold_case / 'completion.json').is_file():
        raise RuntimeError('Retain the source and close its publication before post-freeze work')
    post_args = SimpleNamespace(worker=None, resume_wait=False, prior_case=args.prior_case,
        source_case=args.cold_case, publisher=pub_args.out, out=args.out / 'isolated-post',
        metadata_template=args.metadata_template, paper=args.paper, mirror=args.mirror)
    state.update(status='isolated_post', source_status=s4.read(args.cold_case / 'result.json')['status'])
    publish(args, state)
    post.main(post_args)
    state.update(status='complete', cold_result=str(args.cold_case / 'result.json'),
        post_result=str(post_args.out / 'result.json'), publication_source=str(pub_args.out / 'result.json'))
    write_json(args.out / 'result.json', state)
    publish(args, state)
    record_registry(args.out, 'S4-METRIC-RETRY-CONTROLLER', 'Qwen2-0.5B-layer0',
        'one-fresh-cold-after-S5-and-isolated-post', state['status'], dict(cold_attempts=1))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('tests', 'out', 'prior-case', 'paper', 'mirror', 'task', 'cold-case', 'checkpoint-out',
                'model', 'inputs', 'original-info', 'metadata-template', 'probe'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = main(args)
    except Exception as exc:
        if args.out.is_dir():
            value = dict(status='error', error=type(exc).__name__ + ': ' + str(exc))
            write_json(args.out / 'result.json', value)
            publish(args, value)
        raise
    raise SystemExit(result)
