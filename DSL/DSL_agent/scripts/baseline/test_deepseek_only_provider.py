"""Offline provider removal regression; fake keys and mocked HTTP only."""
import contextlib
import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_credentials import PROVIDERS
from deepseek_provider import Config, DeepSeekProvider, HTTPSTransport, ProviderError
import deepseek_http_worker as worker
from test_candidate_pipeline import request_fixture
from test_deepseek_provider import FixtureTransport, completion


class DeepSeekOnlyTests(unittest.TestCase):
    def test_default_candidate_and_batch_plan_are_deepseek(self):
        from run_candidate import parse_args
        from run_agent_batch import main
        args = parse_args(['--case', 'fixture.json', '--prepare'])
        self.assertEqual((args.provider, args.model, args.reasoning_effort),
                         ('deepseek', 'deepseek-flash', 'high'))
        output = io.StringIO()
        with patch('agent_credentials.load_api_key', side_effect=AssertionError('credential read')):
            with patch.object(sys, 'argv', ['run_agent_batch.py', '--plan']):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report['service_provider'], 'deepseek')
        self.assertEqual(report['api_concurrency'], 10)
        self.assertEqual(report['agent_calls'], 0)

    def test_historical_report_provider_identity_is_not_rewritten(self):
        from run_agent_batch import generation_settings
        for name in ('opencode-go', 'commandcode-goat'):
            report = dict(service_provider=name, model='deepseek-v4-flash',
                          max_tokens=384000, max_repairs=3, api_timeout=900)
            self.assertEqual(generation_settings(report)['service_provider'], name)

    def test_only_deepseek_is_registered(self):
        self.assertEqual(PROVIDERS, {'deepseek': 'DEEPSEEK_API_KEY'})
        self.assertEqual(worker.ROUTES, {'deepseek': ('api.deepseek.com', '/chat/completions')})
        for name in ('opencode-go', 'commandcode-goat', 'https://evil.invalid'):
            with self.subTest(provider=name), self.assertRaises(ProviderError):
                Config(service_provider=name)
        with self.assertRaisesRegex(ProviderError, 'unsupported_model'):
            Config(model='glm-5.3-flash')

    def test_removed_routes_fail_before_network(self):
        with patch.object(worker.http.client, 'HTTPSConnection', side_effect=AssertionError('network')):
            for name in ('opencode-go', 'commandcode-goat', 'evil'):
                with self.subTest(provider=name), self.assertRaises(KeyError):
                    worker.exchange(json.dumps(dict(api_key='FAKE', body={}, provider=name)).encode())

    def test_candidate_and_batch_cli_reject_removed_options(self):
        from run_candidate import parse_args
        from run_agent_batch import main
        rejected = [
            ['--provider', 'opencode-go'], ['--provider', 'commandcode-goat'],
            ['--model', 'glm-5.3-flash'],
            ['--connection-report', 'obsolete.json'], ['--accept-connection-risk'],
        ]
        with patch('agent_credentials.load_api_key', side_effect=AssertionError('credential read')):
            for flags in rejected:
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_args(['--case', 'fixture.json', '--live', *flags])
                    self.assertEqual(error.exception.code, 2)
                    with patch.object(sys, 'argv', ['run_agent_batch.py', '--plan', *flags]):
                        with self.assertRaises(SystemExit) as error:
                            main()
                    self.assertEqual(error.exception.code, 2)

    def test_deepseek_wire_settings_and_transport_remain_intact(self):
        config = Config(model='deepseek-flash', reasoning_effort='high',
                        max_tokens=384000, timeout_seconds=1200, stream=True)
        fixture = FixtureTransport([completion('{}', model=config.model)])
        provider = DeepSeekProvider(config, transport=fixture)
        provider.generate(request_fixture(), [])
        body, timeout = fixture.sent[0]
        self.assertEqual(body['model'], 'deepseek-flash')
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertEqual(body['reasoning_effort'], 'high')
        self.assertEqual(body['max_tokens'], 384000)
        self.assertEqual(timeout, 1200)
        transport = HTTPSTransport(api_key='FAKE', approved_request_id=request_fixture()['request_id'],
                                  enabled=True, approved_config=config)
        with patch('deepseek_provider.subprocess.run',
                   return_value=SimpleNamespace(returncode=0, stdout=b'401\n', stderr=b'')) as run:
            transport.post(json.dumps(body).encode(), timeout)
        frame = json.loads(run.call_args.kwargs['input'])
        self.assertEqual(frame['provider'], 'deepseek')
        self.assertNotIn('session_id', frame)

    def test_deepseek_headers_and_no_redirect_fallback(self):
        calls = []

        class Connection:
            def __init__(self, host, **kwargs):
                calls.append((host, kwargs))

            def connect(self):
                pass

            def request(self, method, path, **kwargs):
                calls.append((method, path, kwargs))

            def getresponse(self):
                return SimpleNamespace(status=302)

            def close(self):
                pass

        wire = json.dumps(dict(api_key='FAKE', provider='deepseek', timeout=1200, body={})).encode()
        with patch.object(worker.http.client, 'HTTPSConnection', Connection):
            self.assertEqual(worker.exchange(wire), b'302\n')
        self.assertEqual(calls[0][0], 'api.deepseek.com')
        self.assertEqual(calls[1][1], '/chat/completions')
        self.assertEqual(set(calls[1][2]['headers']),
                         {'Authorization', 'Content-Type', 'Accept', 'User-Agent'})

    def test_response_and_candidate_limits_unchanged(self):
        from deepseek_provider import MAX_HTTP_BYTES, MAX_OUTPUT_TOKENS
        from candidate_contract import MAX_BYTES
        self.assertEqual(MAX_HTTP_BYTES, worker.MAX_RESPONSE_BYTES)
        self.assertEqual(MAX_OUTPUT_TOKENS, 384000)
        self.assertEqual(MAX_BYTES, 131072)


if __name__ == '__main__':
    unittest.main()
