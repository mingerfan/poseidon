"""E8b rerun: driver-passed ledger cap, session outcome and call counting (API 0)."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pt_agent.experiments import a1_transformer_agent as a1
from pt_agent.experiments import e8_align_fallback as e8


class LedgerCapTest(unittest.TestCase):
    def test_default_keeps_the_earlier_caps(self):
        self.assertEqual(a1.ledger_cap(SimpleNamespace(), 403), 403)
        self.assertEqual(a1.ledger_cap(SimpleNamespace(ledger_cap=None), 450), 450)

    def test_driver_cap_replaces_the_default(self):
        self.assertEqual(a1.ledger_cap(SimpleNamespace(ledger_cap=427), 403), 427)

    def test_round_override(self):
        self.assertEqual(a1.round_of(SimpleNamespace(), 'E6r'), 'E6r')
        self.assertEqual(a1.round_of(SimpleNamespace(experiment_round='E8'), 'E6r'), 'E8')

    def test_call_rows_use_the_round_override(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'out'; out.mkdir()
            (out/'api-accounting.json').write_text(json.dumps(dict(api_before=403)))
            ledger = root/'calls.jsonl'
            values = [{}]*403+[dict(ts='now', purpose='agent-iter-1'), dict(ts='now', purpose='tune-select-iter2-round1')]
            ledger.write_text(''.join(json.dumps(item)+'\n' for item in values))
            rounds = {}
            for label, override in (('r2b', 'E8'), ('r2b', None), ('r1', None)):
                appended = []
                with patch('pt_agent.registry.load', return_value=[]), \
                     patch('pt_agent.registry.append', side_effect=appended.append):
                    a1.account_calls(ledger, out, model='qwen', prompt_version='r8-v11', label=label,
                                     split_version='split-v2', experiment_round=override)
                rounds[(label, override)] = [row['round'] for row in appended]
                if override:
                    self.assertTrue(all(row['run_id'].startswith('E8-qwen-api-') for row in appended))
                    self.assertEqual([row['metrics']['budget_namespace'] for row in appended],
                                     [['qwen', 'r8-v11', 'r2b'], ['qwen', 't1', 'r2b']])
            self.assertEqual(rounds, {('r2b', 'E8'): ['E8', 'E8'], ('r2b', None): ['E6', 'E6'], ('r1', None): ['E6r', 'E6r']})

    def test_session_command_passes_the_approved_ceiling(self):
        command = e8.session_command(Path('/r'), Path('/m'), 'r2b')
        self.assertEqual(command[command.index('--ledger-cap')+1], str(e8.API_CEILING))
        self.assertEqual(command[command.index('--experiment-round')+1], 'E8')
        self.assertEqual(command[command.index('--session-label')+1], 'r2b')


class OutcomeTest(unittest.TestCase):
    def test_outcomes(self):
        self.assertEqual(e8.session_outcome(0, dict(complete=True, frozen=True)), 'frozen')
        self.assertEqual(e8.session_outcome(0, dict(complete=True, frozen=False)), 'not_frozen')
        self.assertEqual(e8.session_outcome(1, dict(complete=False, error='budget_exhausted', failure_class='provider')),
                         'budget_exhausted')
        self.assertEqual(e8.session_outcome(1, dict(complete=False, error='http_error', failure_class='provider')),
                         'provider_failed')
        self.assertIsNone(e8.session_outcome(1, dict(complete=False, error='KeyError')))


class CountingTest(unittest.TestCase):
    def test_refused_attempts_are_counted_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'attempts'
            folder.mkdir()
            for name in ('call-001-001', 'call-002-001', 'call-003-001'):
                (folder/(name+'.record.json')).write_text('{}')
            (folder/'call-002-001.error.json').write_text(json.dumps(dict(error='http_error')))
            (folder/'call-003-001.error.json').write_text(json.dumps(dict(error='budget_exhausted')))
            self.assertEqual(e8.namespace_calls(Path(tmp)), (2, 1))
            self.assertEqual(e8.namespace_calls(Path(tmp)/'missing'), (0, 0))

    def test_session_labels_include_first_attempts(self):
        b = dict(r1=dict(status='frozen'), r2=dict(status='frozen', label='r2b', first_attempt=dict(label='r2')))
        self.assertEqual(e8.session_labels(b), ['r1', 'r2', 'r2b'])
        self.assertEqual(e8.session_labels(dict(r1=dict(status='frozen'), r2=dict(status='provider_failed'))), ['r1', 'r2'])


if __name__ == '__main__':
    unittest.main()
