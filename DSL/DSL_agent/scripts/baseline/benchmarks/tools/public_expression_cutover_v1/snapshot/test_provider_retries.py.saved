"""Offline retry fixtures: no API key or paid requests."""
import copy
import json
import unittest
from unittest.mock import patch

from deepseek_provider import Config, DeepSeekProvider, ProviderError, generation_deadline
from test_deepseek_provider import FixtureTransport, completion
from test_candidate_pipeline import request_fixture
from run_agent_batch import case_metrics


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.request = request_fixture()
        self.transport = None

    def provider(self, replies, diagnostics=None, **settings):
        self.transport = FixtureTransport(replies)
        self.transport.is_live = True  # Synthetic accounting only; post is a local fixture.
        self.transport.last_diagnostics = diagnostics or {}
        provider = DeepSeekProvider(Config(provider_retries=3, **settings), transport=self.transport)
        self.snapshots = []
        provider.on_attempt = lambda: self.snapshots.append(provider.metrics())
        return provider

    @patch('deepseek_provider.time.sleep')
    def test_missing_done_then_success_preserves_payload(self, sleep):
        provider = self.provider([ProviderError('transport_invalid_stream'), completion('{}')],
                                 {'stream_error': 'missing_done'})
        self.assertEqual(provider.generate(self.request, []), '{}')
        self.assertEqual(self.transport.sent[0], self.transport.sent[1])
        self.assertEqual(sleep.call_args.args, (5,))
        metrics = provider.metrics()
        self.assertEqual((metrics['agent_calls'], metrics['generation_attempts'], metrics['transport_retries']), (2, 1, 1))
        self.assertEqual(metrics['calls'][0]['retry_index'], 0)
        self.assertTrue(metrics['calls'][0]['retry_scheduled'])
        self.assertEqual(self.snapshots[0]['calls'][0]['status'], 'in_flight')
        report = dict(status='provider_failed', agent_calls=2, provider_metrics=metrics)
        self.assertEqual(case_metrics(report)['repairs_attempted'], 0)
        self.assertEqual(case_metrics(report)['calls_without_usage'], 1)

    @patch('deepseek_provider.time.sleep')
    def test_four_attempts_exhaust_and_no_fifth(self, sleep):
        provider = self.provider([TimeoutError('PRIVATE')] * 4)
        with self.assertRaisesRegex(ProviderError, 'transport_timeout'):
            provider.generate(self.request, [])
        with self.assertRaisesRegex(ProviderError, 'no_retry'):
            provider.generate(self.request, [])
        self.assertEqual(len(self.transport.sent), 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5, 15, 30])
        self.assertNotIn('PRIVATE', json.dumps(provider.metrics()))
        self.assertFalse(provider.metrics()['calls'][-1]['retry_scheduled'])

    @patch('deepseek_provider.time.sleep')
    def test_nontransient_failures_never_retry(self, sleep):
        failures = [(status, b'PRIVATE') for status in (301, 400, 401, 402, 403, 404, 429)]
        failures += [ProviderError(code) for code in ('transport_certificate_failed', 'transport_tls_failed',
                      'transport_invalid_stream', 'response_model_mismatch', 'transport_worker_failed')]
        failures += [completion('{}', model='wrong-model')]
        for failure in failures:
            provider = self.provider([failure], {'stream_error': 'invalid_json'})
            with self.assertRaises(ProviderError):
                provider.generate(self.request, [])
            self.assertEqual(len(self.transport.sent), 1)
        sleep.assert_not_called()

    @patch('deepseek_provider.time.sleep')
    def test_server_failures_retry(self, sleep):
        for status in (408, 500, 502, 503, 504):
            provider = self.provider([(status, b'PRIVATE'), completion('{}')])
            self.assertEqual(provider.generate(self.request, []), '{}')
            self.assertEqual(len(self.transport.sent), 2)

    @patch('deepseek_provider.time.sleep')
    def test_repair_history_and_separate_budgets(self, sleep):
        provider = self.provider([TimeoutError(), completion('first'),
                                  TimeoutError(), completion('second')], max_calls=2)
        provider.generate(self.request, [])
        feedback = dict(status='failed', layer='response_parse', category='candidate')
        provider.generate(self.request, [feedback])
        self.assertEqual(self.transport.sent[2], self.transport.sent[3])
        self.assertEqual(self.transport.sent[2][0]['messages'][2]['content'], 'first')
        with self.assertRaisesRegex(ProviderError, 'call_budget_exhausted'):
            provider.generate(self.request, [feedback, feedback])
        self.assertEqual(provider.metrics()['generation_attempts'], 2)
        self.assertEqual(provider.metrics()['transport_retries'], 2)

    def test_limits_forwarding_and_deadlines(self):
        for retries in (-1, 4, True, 1.5):
            with self.assertRaisesRegex(ProviderError, 'retry_limit'):
                Config(provider_retries=retries)
        self.assertEqual(generation_deadline(1200, 4, 3), 19400)
        from run_candidate import parse_args, forward_options
        args = parse_args(['--case', 'fixture.json', '--live', '--provider-retries', '3'])
        self.assertIn('--provider-retries 3', forward_options(args))
        # Legacy metrics must retain their shape for sealed report validation.
        self.assertNotIn('transport_retries', case_metrics(dict(agent_calls=2)))
        self.assertEqual(case_metrics(dict(agent_calls=2))['repairs_attempted'], 1)


if __name__ == '__main__':
    unittest.main()
