"""Finish the untouched ResNet report after the S5 cold matrix has closed."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import fcntl

from .. import paths
from . import s4_qwen_block as s4
from . import s5_tuner_off as off
from .r10_paper_metrics import write_json
from .a5_a6_rule63_search import guarded_process


def main(args):
    args.out.mkdir(exist_ok=False)
    old = s4.read(args.failed_report / 'result.json')
    if old.get('report_samples_completed') != 0 or old.get('error') != 'SubprocessError: Exception occurred in preexec_fn.':
        raise ValueError('Only an untouched preexec failure may continue report access')
    state = dict(status='waiting_for_CNN_cold_matrix', source=str(args.source_case),
        preserved_report_entry_failure=str(args.failed_report), original_report_samples_read=0)
    write_json(args.out / 'progress.json', state)
    while not (args.prior_case / 'result.json').is_file():
        if time.time() >= s4.DEADLINE:
            raise TimeoutError('Publication deadline before the independent report continuation')
        time.sleep(2)
    prior = s4.read(args.prior_case / 'result.json')
    if len(prior.get('rows', [])) != 3:
        raise ValueError('Close all three actual CNN cold chains before report continuation')
    report_case = args.out / 'resnet20-independent-report'
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        argv = [sys.executable, '-B', '-m', off.MODULE, '--cnn-report', '--model', 'resnet20_silu',
            '--source-case', str(args.source_case), '--out', str(report_case),
            '--seconds', str(s4.remaining_before_deadline(1800)), '--memory-gib', '6']
        state.update(status='independent_report')
        write_json(args.out / 'progress.json', state)
        observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=args.out / 'report.log',
            resources=args.out / 'resources.jsonl', timeout=s4.remaining_before_deadline(1900), mem_gib=24, fsize_gib=8)
        write_json(args.out / 'process.json', observed)
        report = s4.read(report_case / 'result.json')
        if observed['remaining_pids']:
            raise RuntimeError('The independent report must close every owned child')
    state.update(status='complete', rows=prior['rows'], reports=prior['reports'],
        report_result=str(report_case / 'result.json'), report_status=report['status'])
    state['reports']['resnet20_silu'] = dict(status=report['status'], source=state['report_result'],
        report_samples=report.get('report_samples_completed'), task_policy_passed=report.get('task_policy_passed'))
    write_json(args.out / 'result.json', state)
    for root in (args.paper, args.mirror):
        target = root / 'S5_TUNER_OFF_RESULTS.json'
        current = s4.read(target)
        current['reports']['resnet20_silu'] = state['reports']['resnet20_silu']
        current.update(status='complete', report_entry_repair=str(args.out / 'result.json'))
        write_json(target, current)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('out', 'prior-case', 'source-case', 'failed-report', 'paper', 'mirror'):
        parser.add_argument('--'+key, type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
