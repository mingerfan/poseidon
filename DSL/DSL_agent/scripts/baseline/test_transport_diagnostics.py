import io
import json
import http.client
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import time

import deepseek_http_worker as worker
from deepseek_provider import Config, HTTPSTransport, DeepSeekProvider, SYSTEM, canonical, ProviderError
from test_candidate_pipeline import request_fixture


class TransportDiagnosticsTests(unittest.TestCase):
    def test_distinct_failures_do_not_echo_exception_text(self):
        for error, code in ((http.client.RemoteDisconnected('SECRET'), 'transport_remote_disconnected'),
                            (http.client.IncompleteRead(b'SECRET'), 'transport_incomplete_read'),
                            (http.client.BadStatusLine('SECRET'), 'transport_bad_status_line')):
            self.assertEqual(worker.failure_code(error), code)

    def test_error_body_and_diagnostics_never_preserve_arbitrary_text(self):
        for raw in (b'{"error":{"message":"Bearer SECRET max_tokens exceeded", "code":"SECRET", "param":"SECRET"}}',
                    b'<html>SECRET</html>', b'{"error":"SECRET"}', b'[]'):
            result = worker.safe_error(raw)
            self.assertNotIn('SECRET', json.dumps(result))
        result = worker.safe_diagnostics(dict(stage='SECRET', error_code='SECRET', elapsed_seconds=float('nan'),
            received_bytes=True, unexpected='SECRET', status=400, error_hint='max_tokens'))
        self.assertEqual(result, dict(status=400, error_hint='max_tokens'))

    def stream(self, events, done=True):
        data = b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)
        return io.BytesIO(data + (b'data: [DONE]\n\n' if done else b''))

    def test_stream_assembles_content_and_usage(self):
        events = [dict(model='deepseek-v4-flash', choices=[dict(index=0, delta=dict(content='a', reasoning_content='r'))]),
                  dict(choices=[dict(index=0, delta=dict(content='b'), finish_reason='stop')]),
                  dict(choices=[], usage=dict(prompt_tokens=1, completion_tokens=2, total_tokens=3))]
        diagnostics = {}
        result = json.loads(worker.collect_sse(self.stream(events), diagnostics, time.monotonic()))
        self.assertEqual(result['choices'][0]['message']['content'], 'ab')
        self.assertEqual(result['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(result['usage']['total_tokens'], 3)
        self.assertEqual(diagnostics['sse_events'], 3)
        self.assertIn('first_body_byte_seconds', diagnostics)

    def test_truncated_and_tool_streams_rejected(self):
        for events, done in (([], False), ([dict(choices=[dict(delta=dict(tool_calls=[{}]))])], True)):
            with self.assertRaises(worker.StreamError):
                worker.collect_sse(self.stream(events, done), {}, time.monotonic())

    def test_error_response_classification_has_stages_not_body(self):
        class Connection:
            sock = None
            def __init__(self, *args, **kwargs): pass
            def connect(self): pass
            def request(self, *args, **kwargs): pass
            def getresponse(self):
                return SimpleNamespace(status=400, read=lambda n: b'{"error":{"message":"SECRET max_tokens", "param":"max_tokens"}}')
            def close(self): pass
        diagnostics = {}
        wire = json.dumps(dict(api_key='SECRET', provider='deepseek',timeout=900,body={})).encode()
        with patch.object(worker.http.client, 'HTTPSConnection', Connection):
            self.assertEqual(worker.exchange(wire, diagnostics), b'400\n')
        self.assertEqual(diagnostics['error_param'], 'max_tokens')
        self.assertEqual(diagnostics['stage'], 'complete')
        self.assertNotIn('SECRET', json.dumps(diagnostics))

    def test_parent_collects_diagnostics_on_failure(self):
        request = request_fixture()
        config = Config(model='deepseek-v4-flash', service_provider='deepseek', stream=True)
        transport = HTTPSTransport(api_key='SECRET', approved_request_id=request['request_id'], enabled=True, approved_config=config)
        provider = DeepSeekProvider(config, transport=transport)
        process = SimpleNamespace(returncode=1, stdout=b'', stderr=b'transport_remote_disconnected\nDIAG {"stage":"headers","secret":"SECRET","headers_seconds":1}\n')
        with patch('deepseek_provider.subprocess.run', return_value=process):
            with self.assertRaisesRegex(ProviderError, 'remote_disconnected'):
                provider.generate(request, [])
        metrics = provider.metrics()
        self.assertEqual(metrics['calls'][0]['transport_diagnostics']['stage'], 'headers')
        self.assertNotIn('SECRET', json.dumps(metrics))


if __name__ == '__main__':
    unittest.main()
