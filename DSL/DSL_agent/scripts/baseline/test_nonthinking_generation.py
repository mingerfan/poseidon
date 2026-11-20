"""Offline boundary tests; no credentials, sockets, compiler or ciphertexts."""
import ast
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmark_graph import canonical, digest
from deepseek_provider import Config, DeepSeekProvider, HTTPSTransport, ProviderError
from generation_receipt import verify, safe_metrics
from test_candidate_pipeline import request_fixture
from test_deepseek_provider import FixtureTransport, completion


class NonthinkingGeneration(unittest.TestCase):
    def fixture(self, config):
        request = request_fixture()
        fixture = FixtureTransport([completion('{}')])
        provider = DeepSeekProvider(config, transport=fixture)
        provider.generate(request, [])
        return request, fixture.sent[0], provider

    def transport(self, request, config):
        return HTTPSTransport(api_key='FAKE-KEY', approved_request_id=request['request_id'],
                              enabled=True, approved_config=config)

    def test_default_and_all_enabled_settings_preserve_original_bytes(self):
        self.assertEqual((Config().reasoning_effort, Config().max_tokens,
                          Config().timeout_seconds), ('high', 8192, 120))
        for model in ('deepseek-flash', 'deepseek-v4-pro', 'deepseek-v4-flash'):
            for effort in ('low', 'high', 'max'):
                for stream in (False, True):
                    with self.subTest(model=model, effort=effort, stream=stream):
                        config = Config(model=model, reasoning_effort=effort, stream=stream)
                        expected = dict(model=model, reasoning_effort=effort, max_tokens=8192,
                                        response_format={'type': 'json_object'}, stream=stream)
                        if stream:
                            expected['stream_options'] = {'include_usage': True}
                        expected['thinking'] = {'type': 'enabled'}
                        self.assertEqual(canonical(config.settings()), canonical(expected))

    def test_none_has_exact_disabled_settings_and_finite_limits(self):
        for stream in (False, True):
            config = Config(reasoning_effort='none', max_tokens=65536, max_calls=2,
                            timeout_seconds=1200, stream=stream)
            expected = dict(model='deepseek-flash', reasoning_effort='none',
                            max_tokens=65536, response_format={'type': 'json_object'}, stream=stream)
            if stream:
                expected['stream_options'] = {'include_usage': True}
            expected['thinking'] = {'type': 'disabled'}
            self.assertEqual(canonical(config.settings()), canonical(expected))
            self.assertEqual(Config(**asdict(config)), config)
        for update in (dict(max_tokens=384001), dict(max_calls=5),
                       dict(timeout_seconds=1201), dict(provider_retries=4)):
            with self.assertRaises(ProviderError):
                Config(reasoning_effort='none', **update)

    def test_invalid_efforts_and_types_are_rejected(self):
        for effort in ('None', 'disabled', 'medium', '', None, True, 0, [], {}):
            with self.subTest(effort=effort), self.assertRaises(ProviderError):
                Config(reasoning_effort=effort)

    def test_none_provider_body_config_roundtrip_and_safe_metrics(self):
        config = Config(reasoning_effort='none')
        request, (body, timeout), provider = self.fixture(config)
        self.assertEqual(self.transport(request, config).validate(canonical(body), timeout), body)
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        metrics = safe_metrics(provider.metrics())
        self.assertEqual(metrics['reasoning_effort'], 'none')
        self.assertEqual((metrics['request_attempts'], metrics['generation_attempts']), (1, 1))
        self.assertNotIn('DO NOT STORE OR FORWARD', json.dumps(metrics))

    def test_none_transport_rejects_config_and_toggle_substitution(self):
        config = Config(reasoning_effort='none')
        request, (body, timeout), _ = self.fixture(config)
        transport = self.transport(request, config)
        for update in (dict(thinking={'type': 'enabled'}), dict(reasoning_effort='low'),
                       dict(reasoning_effort='high'), dict(max_tokens=65536), dict(stream=True)):
            with self.subTest(update=update), self.assertRaises(ProviderError):
                transport.validate(canonical(dict(body, **update)), timeout)
        for effort in ('low', 'high', 'max'):
            with self.assertRaises(ProviderError):
                self.transport(request, Config(reasoning_effort=effort)).validate(canonical(body), timeout)

    def test_none_transport_preserves_bounded_public_repair_validation(self):
        config = Config(reasoning_effort='none', max_calls=2)
        request, (body, timeout), _ = self.fixture(config)
        transport = self.transport(request, config)
        pair = [dict(role='assistant', content='{bad json'),
                dict(role='user', content=json.dumps(dict(feedback=dict(status='failed', layer='response_parse'))))]
        body['messages'].extend(pair)
        self.assertEqual(transport.validate(canonical(body), timeout), body)
        body['messages'].extend(pair)
        with self.assertRaises(ProviderError):
            transport.validate(canonical(body), timeout)
        body['messages'] = body['messages'][:2] + [dict(role='assistant', content='{}'),
            dict(role='user', content=json.dumps(dict(feedback=dict(status='failed', reference=[123]))))]
        with self.assertRaises(ProviderError):
            transport.validate(canonical(body), timeout)

    def application_fixture(self):
        from test_application_generation import GenerationTests
        fixture = GenerationTests('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_normal_Application_none_receipt_roundtrip_and_tamper_rejection(self):
        fixture = self.application_fixture()
        provider = fixture.provider([fixture.source], reasoning_effort='none', max_calls=1)
        task = fixture.app.submit(fixture.model, limits=dict(generations=1, http_attempts=1,
                wall_seconds=120, api_timeout_seconds=120, max_tokens=8192))
        result = fixture.app.run(task, provider)
        self.assertEqual(result['status'], 'succeeded')
        value = fixture.backend.receipt
        self.assertEqual(value['configuration']['reasoning_effort'], 'none')
        self.assertEqual(value['metrics']['reasoning_effort'], 'none')
        self.assertEqual(verify(value, fixture.request, fixture.source)['generations'], 1)
        changed = copy.deepcopy(value)
        changed['configuration']['reasoning_effort'] = 'low'
        changed['binding'] = digest({k: v for k, v in changed.items() if k != 'binding'})
        with self.assertRaises(ValueError):
            verify(changed, fixture.request, fixture.source)

    def test_Application_none_does_not_bypass_HTTP_budget(self):
        fixture = self.application_fixture()
        provider = fixture.provider([fixture.source], reasoning_effort='none', max_calls=1,
                                    max_tokens=65536)
        task = fixture.app.submit(fixture.model, limits=dict(generations=1, http_attempts=1,
                wall_seconds=120, api_timeout_seconds=120, max_tokens=8192))
        result = fixture.app.run(task, provider)
        self.assertNotEqual(result['status'], 'succeeded')
        self.assertEqual(provider.agent_calls, 0)

    def test_single_case_CLI_forwards_none(self):
        from run_candidate import parse_args, forward_options
        args = parse_args(['--case', 'model.json', '--deepseek', '--reasoning-effort', 'none'])
        self.assertIn('--reasoning-effort none', forward_options(args))

    def test_all_public_CLI_choices_admit_none_and_preserve_default(self):
        # Inspect actual parser calls without invoking a CLI's execution workflow.
        for name in ('run_candidate.py', 'run_agent_batch.py', 'semantic_benchmark_execution.py'):
            module = __import__(name[:-3])
            tree = ast.parse(Path(module.__file__).read_text())
            matches = [n for n in ast.walk(tree) if type(n) is ast.Call and
                type(n.func) is ast.Attribute and n.func.attr == 'add_argument' and
                n.args and type(n.args[0]) is ast.Constant and n.args[0].value == '--reasoning-effort']
            self.assertEqual(len(matches), 1, name)
            keywords = {k.arg: ast.literal_eval(k.value) for k in matches[0].keywords}
            self.assertEqual(keywords['choices'], ('none', 'low', 'high', 'max'))
            self.assertEqual(keywords['default'], 'high')


if __name__ == '__main__':
    unittest.main()
