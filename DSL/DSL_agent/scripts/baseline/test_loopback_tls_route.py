"""Offline tests only; no API credentials or external networking."""
import json
import shlex
import ssl
import unittest
from unittest.mock import patch

import deepseek_http_worker as worker
from deepseek_provider import Config, ProviderError, retryable_failure
from run_agent_batch import generation_settings, configuration_delta
from run_candidate import parse_args, forward_options


class LoopbackRouteTests(unittest.TestCase):
    def test_strict_destination_tls_and_credential_free_connect(self):
        events = []
        class Connection:
            def __init__(self, host, **kw):
                events.append(('init', host, kw))
            def set_tunnel(self, host, port, **kw):
                events.append(('tunnel', host, port, kw))
            def connect(self):
                events.append(('connect',))
            def request(self, method, path, **kw):
                events.append(('request', method, path, kw))
            def getresponse(self):
                class Response:
                    status = 200
                    def read(self, size):
                        return b'{}'
                return Response()
            def close(self):
                events.append(('close',))
        wire = json.dumps(dict(api_key='FAKE-KEY', provider='deepseek', timeout=12,
                               loopback_proxy_port=7897, body={})).encode()
        with patch.object(worker.http.client, 'HTTPSConnection', Connection):
            self.assertEqual(worker.exchange(wire), b'200\n{}')
        self.assertEqual(events[0][1], '127.0.0.1')
        self.assertEqual(events[0][2]['port'], 7897)
        ctx = events[0][2]['context']
        self.assertTrue(ctx.check_hostname)
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(events[1], ('tunnel', 'api.deepseek.com', 443, {}))
        self.assertEqual(events[3][1:3], ('POST', '/chat/completions'))
        self.assertEqual(events[3][3]['headers']['Authorization'], 'Bearer FAKE-KEY')
        self.assertEqual(events[-1], ('close',))

    def test_invalid_ports_and_other_providers_rejected_before_network(self):
        for port in (True, -1, 65536, '7897', 'http://remote.invalid'):
            with self.subTest(port=port), self.assertRaises(ProviderError):
                Config(loopback_proxy_port=port)
            wire = json.dumps(dict(api_key='FAKE', body={}, timeout=1,
                                   loopback_proxy_port=port)).encode()
            with patch.object(worker.http.client, 'HTTPSConnection') as connection:
                with self.assertRaises(ValueError):
                    worker.exchange(wire)
                connection.assert_not_called()
        with self.assertRaises(ProviderError):
            Config(service_provider='opencode-go', loopback_proxy_port=7897)

    def test_cli_forwards_explicit_route(self):
        args = parse_args(['--case', 'scripts/baseline/cases/linear-example.json',
                           '--deepseek', '--provider', 'deepseek',
                           '--loopback-proxy-port', '7897'])
        options = shlex.split(forward_options(args))
        self.assertEqual(options[options.index('--loopback-proxy-port')+1], '7897')

    def test_legacy_default_and_route_delta_are_audited(self):
        before = generation_settings(dict(model='deepseek-v4-flash', max_tokens=384000,
                                          max_repairs=3, api_timeout=900))
        self.assertEqual(before['loopback_proxy_port'], 0)
        after = dict(before, loopback_proxy_port=7897)
        self.assertEqual(configuration_delta(before, after),
                         {'loopback_proxy_port': {'before': 0, 'after': 7897}})
        self.assertNotIn('loopback_proxy_port', Config(loopback_proxy_port=7897).settings())

    def test_integrity_and_certificate_errors_still_fail_closed(self):
        for code in ('transport_tls_failed', 'transport_certificate_failed'):
            self.assertFalse(retryable_failure(code, dict(tls_reason='DECRYPTION_FAILED_OR_BAD_RECORD_MAC')))


if __name__ == '__main__':
    unittest.main()
