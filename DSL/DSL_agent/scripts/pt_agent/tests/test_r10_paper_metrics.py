"""R10 attribution and failure supervision, with no model or native compiler."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments.r10_paper_metrics import (
    StageRecorder, read_events, run_process, summarize_events, _process)
from pt_agent.experiments.r10_paper_preflight import finalize_result


def interval(key, parent, start, end):
    begin = dict(type='begin', event_id=key, parent_id=parent, stage_id=key, start_ns=start)
    return [begin, dict(begin, type='end', end_ns=end, status='ok')]


class AttributionTest(unittest.TestCase):
    def test_nested_exclusive_intervals_sum_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = iter([0, 1, 3, 10])
            recorder = StageRecorder(Path(tmp) / 'events', clock=lambda: next(ticks))
            with recorder.span('root'):
                with recorder.span('lookup', cache_hit=True, source_seconds=613.801):
                    pass
            summary = summarize_events(read_events(recorder.path))
        self.assertTrue(summary['attribution_ready'])
        self.assertEqual([row['exclusive_ns'] for row in summary['spans']], [8, 2])
        self.assertEqual(sum(row['exclusive_ns'] for row in summary['spans']), 10)
        self.assertTrue(summary['spans'][1]['cache_hit'])

    def test_overlapping_siblings_are_rejected_for_plotting(self):
        events = interval('root', None, 0, 10) + interval('a', 'root', 0, 8) + interval('b', 'root', 2, 10)
        summary = summarize_events(events)
        self.assertEqual(summary['status'], 'timing_invalid')
        self.assertFalse(summary['attribution_ready'])
        self.assertTrue(all(row['exclusive_ns'] is None for row in summary['spans']))

    def test_interrupted_stage_has_no_fabricated_end(self):
        result = summarize_events([dict(type='begin', event_id='x', parent_id=None,
                                       stage_id='import', start_ns=2)], termination_observed_ns=99)
        row = result['spans'][0]
        self.assertEqual(row['status'], 'interrupted')
        self.assertIsNone(row['end_ns'])
        self.assertIsNone(row['inclusive_ns'])
        self.assertEqual(row['termination_observed_ns'], 99)
        self.assertFalse(result['attribution_ready'])

    def test_exception_closes_the_stage_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = StageRecorder(Path(tmp) / 'events')
            with self.assertRaisesRegex(ValueError, 'fixture'):
                with recorder.span('probe'):
                    raise ValueError('fixture')
            rows = read_events(recorder.path)
        self.assertEqual([row['type'] for row in rows], ['begin', 'end'])
        self.assertEqual(rows[-1]['status'], 'error')

    def test_parent_closed_before_child_is_invalid_even_when_intervals_fit(self):
        root = [dict(row, pid=101) for row in interval('root', None, 0, 10)]
        child = [dict(row, pid=101) for row in interval('child', 'root', 2, 8)]
        summary = summarize_events([root[0], child[0], root[1], child[1]])
        self.assertEqual(summary['status'], 'timing_invalid')
        self.assertFalse(summary['attribution_ready'])
        self.assertTrue(any('non-LIFO' in reason for reason in summary['diagnostics']))
        self.assertTrue(all(row['exclusive_ns'] is None for row in summary['spans']))

    def test_appended_worker_stream_uses_its_own_pid_stack(self):
        root = [dict(row, pid=101) for row in interval('root', None, 0, 10)]
        launched = [dict(row, pid=101) for row in interval('launch', 'root', 1, 9)]
        worker = [dict(row, pid=202) for row in interval('worker', 'launch', 2, 9)]
        probe = [dict(row, pid=202) for row in interval('probe', 'worker', 3, 8)]
        outer_events = [root[0], launched[0], launched[1], root[1]]
        worker_events = [worker[0], probe[0], probe[1], worker[1]]
        summary = summarize_events(outer_events + worker_events)
        self.assertEqual(summary['status'], 'ok')
        self.assertTrue(summary['attribution_ready'])
        self.assertEqual(sum(row['exclusive_ns'] for row in summary['spans']), 10)

    def test_capabilities_preserves_raw_events_separately_from_spans(self):
        from pt_agent.experiments import r10_paper_preflight as preflight

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(phase='capabilities', out=root / 'run',
                                   model=None, inputs=None, export_record=None)
            with patch.object(preflight.paths, 'RESULTS_DIR', root), patch.object(
                    preflight, '_git_state', return_value={'branch': 'fixture', 'working_tree_changed': False}):
                result = preflight.execute(args)
            raw = read_events(args.out / 'events.jsonl')
            spans = read_events(args.out / 'stages.jsonl')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([row['type'] for row in raw], ['begin', 'begin', 'end', 'end'])
        self.assertEqual(len(spans), 2)
        self.assertTrue(all('inclusive_ns' in row and 'exclusive_ns' in row for row in spans))
        self.assertTrue(all(row['end_ns'] is not None for row in spans))

    def test_child_beyond_parent_is_invalid(self):
        summary = summarize_events(interval('root', None, 1, 9) + interval('a', 'root', 2, 12))
        self.assertEqual(summary['status'], 'timing_invalid')

    def test_killed_or_failed_worker_keeps_measurements_null(self):
        row = finalize_result(phase='qwen-import', worker=None, processes=[dict(exit_code=-9)],
                              timing={}, status='failed', error='killed', git_state={}, out=Path('/r/run'))
        self.assertIsNone(row['metrics']['compiled_bootstraps'])
        self.assertIsNone(row['metrics']['predicted_latency'])
        self.assertIsNone(row['metrics']['import_completed'])
        self.assertEqual(row['metrics']['real_ckks_runs'], 0)


@unittest.skipUnless(sys.platform == 'linux' and Path('/proc').is_dir(), 'Linux /proc supervision')
class ProcessTreeTest(unittest.TestCase):
    def test_separate_session_descendant_is_terminated_and_reaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / 'child-pid'
            program = ('import os,time\n'
                       'child=os.fork()\n'
                       'if child==0:\n'
                       ' os.setsid()\n'
                       f' open({str(marker)!r},"w").write(str(os.getpid()))\n'
                       ' time.sleep(60)\n'
                       'else:\n'
                       ' time.sleep(60)\n')
            result = run_process([sys.executable, '-c', program], cwd=root,
                                 log=root/'stdout', resources=root/'resources',
                                 timeout=1, mem_gib=4, term_grace=0.2)
            child = int(marker.read_text())
            self.assertTrue(result['timed_out'])
            self.assertEqual(result['remaining_pids'], [])
            self.assertIsNone(_process(child))
            self.assertTrue((root/'resources').is_file())

    def test_nonzero_child_retains_exit_and_resource_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_process([sys.executable, '-c', 'raise SystemExit(7)'], cwd=root,
                                 log=root/'stdout', resources=root/'resources',
                                 timeout=10, mem_gib=4)
            samples = [json.loads(line) for line in (root/'resources').read_text().splitlines()]
        self.assertEqual(result['exit_code'], 7)
        self.assertFalse(result['timed_out'])
        self.assertEqual(result['remaining_pids'], [])
        self.assertTrue(samples)
        self.assertGreater(result['outer_peak_rss_kib'], 0)
        self.assertEqual(samples[0]['outer_process']['pid'], os.getpid())


if __name__ == '__main__':
    unittest.main()
