"""Small independent BERT bookkeeping checks; no model or API execution."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_sst2 as outer


class ApiAccountingChecks(unittest.TestCase):
    def test_explicit_paths_ignore_missing_or_legacy_pt_results_and_deduplicate(self):
        for legacy in (None, '/tmp/unrelated-legacy-sdk-results'):
            with self.subTest(legacy=legacy), TemporaryDirectory() as temporary:
                root = Path(temporary)
                ledger, registry = root / 'explicit-ledger.jsonl', root / 'explicit-registry.jsonl'
                work = root / 'run/phases/003-step'
                work.mkdir(parents=True)
                calls = [dict(ts='synthetic-' + str(i), status=200, purpose='agent-iter-' + str(i),
                              model='fixture', seconds=0.1) for i in (1, 2)]
                ledger.write_text(''.join(json.dumps(call) + '\n' for call in calls))
                env = dict(os.environ)
                env.pop('PT_RESULTS', None)
                if legacy is not None:
                    env['PT_RESULTS'] = legacy
                with patch.dict(os.environ, env, clear=True):
                    first = outer.account_api_calls(ledger, registry, work,
                        before=1, after=2, git_rev='synthetic')
                    second = outer.account_api_calls(ledger, registry, work,
                        before=1, after=2, git_rev='synthetic')
                self.assertEqual(first['registered_calls'], [2])
                self.assertEqual(second['registered_calls'], [])
                self.assertEqual(second['already_registered_calls'], [2])
                rows = [json.loads(line) for line in registry.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(set(rows[0]), {'run_id', 'round', 'model', 'mode', 'git_rev',
                                               'metrics', 'artifacts_dir', 'ts'})
                self.assertEqual(rows[0]['metrics']['global_call'], 2)
                self.assertEqual(rows[0]['metrics']['source_ledger'], str(ledger))
                self.assertEqual(rows[0]['metrics']['purpose'], 'agent-iter-2')
                self.assertEqual(rows[0]['artifacts_dir'], str(work))

    def test_completed_phase_registers_new_call_without_legacy_sdk_configuration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(out=root / 'fresh-run', ledger=root / 'ledger.jsonl', mem_gib=4)
            worker = dict(status='ok', phase='step', model=outer.MODEL, next_step='evaluate',
                native_test_reads=0, compiler_runs=0, keygen_runs=0, real_ckks_attempts=0,
                real_ckks_completed=0, report_samples=0, frozen=False,
                **dict.fromkeys(outer.FORMAL))
            def fake_run(command, **kwargs):
                work = Path(kwargs['log']).parent
                (work / 'worker_result.json').write_text(json.dumps(worker))
                args.ledger.write_text(json.dumps(dict(ts='synthetic-call', status=200,
                    purpose='agent-iter-1', model='fixture', seconds=0.1)) + '\n')
                return dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[])
            with patch.object(outer.paths, 'RESULTS_DIR', root), \
                    patch.dict(os.environ, {'PT_RESULTS': '/tmp/legacy-unrelated'}), \
                    patch.object(outer, 'source_preflight', return_value={'synthetic': True}), \
                    patch.object(outer, 'command', return_value=['synthetic-worker']), \
                    patch.object(outer, 'budget_guard', return_value=nullcontext({})), \
                    patch.object(outer, 'check_budget', return_value=0), \
                    patch.object(outer, 'run_process', side_effect=fake_run), \
                    patch('pt_agent.registry.git_rev', return_value='synthetic'):
                result = outer.execute_phase(args, 'step')
            self.assertEqual(result['status'], 'ok')
            self.assertNotIn('accounting_error', result)
            self.assertEqual(result['api_registration']['registered_calls'], [1])
            rows = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
            self.assertEqual([row['mode'] for row in rows], ['llm-call', 'sst2-step'])
            self.assertEqual(rows[0]['metrics']['global_call'], 1)
            self.assertEqual(result['metrics']['llm_calls'], 1)


if __name__ == '__main__':
    unittest.main()
