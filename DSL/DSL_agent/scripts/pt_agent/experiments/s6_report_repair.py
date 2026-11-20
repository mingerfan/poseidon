"""Finish the first untouched S6 report after its missing worker wrapper."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

from .. import paths
from . import s4_qwen_block as s4
from . import s6_no_llm as driver
from .r10_paper_metrics import write_json
from .s2_heir_serial import record_registry


def main(args):
    from .a5_a6_rule63_search import guarded_process
    args.out.mkdir(exist_ok=False)
    state = dict(status='waiting_for_S6_batch', original_case=str(args.source_case),
        original_report_samples_read=0, feedback_to_Agent=False, new_API_calls=0)
    write_json(args.out / 'progress.json', state)
    while not (args.prior_case / 'result.json').is_file():
        if time.time() >= s4.DEADLINE:
            state.update(status='not_run', reason='Publication cutoff before the original S6 batch closed')
            write_json(args.out / 'result.json', state)
            return 1
        time.sleep(2)
    report_case = args.out / 'resnet20-independent-report'
    with (paths.RESULTS_DIR / 'autonomous-heavy-experiments.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if s4.remaining_before_deadline(1800) < 30:
            raise TimeoutError('Insufficient time before the publication cutoff')
        argv = [sys.executable, '-B', '-m', driver.MODULE, '--cnn-report', '--model', 'resnet20_silu',
            '--source-case', str(args.source_case), '--out', str(report_case), '--seconds',
            str(s4.remaining_before_deadline(1800)), '--memory-gib', '6']
        state.update(status='independent_report')
        write_json(args.out / 'progress.json', state)
        observed = guarded_process(argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
            log=args.out / 'report.log', resources=args.out / 'resources.jsonl',
            timeout=s4.remaining_before_deadline(1900), mem_gib=24, fsize_gib=8)
        write_json(args.out / 'process.json', observed)
        if observed.get('remaining_pids'):
            raise RuntimeError('Close every report worker before returning')
        report = s4.read(report_case / 'result.json')
    state.update(status='complete', report_status=report['status'], report_source=str(report_case / 'result.json'))
    write_json(args.out / 'result.json', state)
    for root in (args.paper, args.mirror):
        value = s4.read(root / 's6_no_llm.json')
        summary = dict(status=report['status'], source=state['report_source'],
            report_samples=report.get('report_samples_completed'),
            task_policy_passed=report.get('report_policy_passed'))
        value['reports']['resnet20_silu'] = summary
        for row in value['rows']:
            if row['model'] == 'resnet20':
                row['report'] = summary
        value.update(report_entry_repair=str(args.out / 'result.json'))
        write_json(root / 's6_no_llm.json', value)
    record_registry(args.out, 'S6-REPORT-ENTRY-REPAIR', 'resnet20', 'once-after-zero-read-adapter-failure',
                    state['status'], dict(API_calls=0, report_status=report['status']))
    return int(report['status'] != 'ok')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'prior-case', 'source-case', 'paper', 'mirror'):
        p.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
