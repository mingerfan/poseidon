"""Offline boundary tests. Only fake credentials and in-memory HTTPS fixtures."""
import io
import copy
import json
import socket
import ssl
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import deepseek_http_worker as worker
from deepseek_provider import Config, DeepSeekProvider, HTTPSTransport, ProviderError
from test_candidate_pipeline import request_fixture
from test_deepseek_provider import FixtureTransport, completion


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.wire = json.dumps(dict(api_key="FAKE-KEY", timeout=2, body={"model": "deepseek-v4-pro"})).encode()

    def exchange(self, status, data):
        events = []
        class Response:
            def __init__(self):
                self.status = status
            def read(self, size):
                events.append(("read", size))
                return data[:size]
        class Connection:
            def __init__(self, host, **kwargs):
                events.append(("connect", host, kwargs))
            def request(self, method, path, **kwargs):
                events.append(("request", method, path, kwargs))
            def connect(self):
                pass
            def getresponse(self):
                return Response()
            def close(self):
                events.append(("close",))
        with patch.object(worker.http.client, "HTTPSConnection", Connection):
            result = worker.exchange(self.wire)
        return result, events

    def test_certificate_hostname_verification_and_fixed_endpoint(self):
        result, events = self.exchange(200, b'{"ok":true}')
        self.assertEqual(result, b'200\n{"ok":true}')
        self.assertEqual(events[0][1], "api.deepseek.com")
        context = events[0][2]["context"]
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(events[1][1:3], ("POST", "/chat/completions"))
        self.assertEqual(events[1][3]["headers"]["Authorization"], "Bearer FAKE-KEY")
        self.assertEqual(events[-1], ("close",))

    def test_empty_default_ca_uses_installed_bundle_without_disabling_tls(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if not Path("/etc/ssl/certs/ca-certificates.crt").is_file():
            self.skipTest("Linux system CA bundle only")
        self.assertEqual(context.get_ca_certs(), [])
        with patch.object(worker.ssl, "create_default_context", return_value=context):
            loaded = worker.tls_context()
        self.assertGreater(len(loaded.get_ca_certs()), 0)
        self.assertTrue(loaded.check_hostname)
        self.assertEqual(loaded.verify_mode, ssl.CERT_REQUIRED)

    def test_redirects_not_followed_and_error_bodies_never_returned(self):
        for status in (301, 302, 307, 308, 401, 429, 503):
            frame, events = self.exchange(status, b"SECRET")
            self.assertEqual(frame, str(status).encode() + b"\n")
            self.assertEqual(len([x for x in events if x[0] == "request"]), 1)
            self.assertEqual([x for x in events if x[0] == 'read'], [('read', 8193)])
            self.assertNotIn(b'SECRET', frame)

    def test_worker_rejects_oversized_response(self):
        with self.assertRaisesRegex(ValueError, "response_limit"):
            self.exchange(200, b"x" * (worker.MAX_RESPONSE_BYTES + 1))

    def test_malformed_worker_input_exits_without_traceback_or_network(self):
        with patch.object(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"FAKE-SECRET invalid"))), \
             patch.object(worker.sys, "stdout", SimpleNamespace(buffer=io.BytesIO())), \
             patch.object(worker.sys, "stderr", SimpleNamespace(buffer=io.BytesIO())), \
             patch.object(worker.http.client, "HTTPSConnection", side_effect=AssertionError("network forbidden")):
            self.assertEqual(worker.main(), 1)
            self.assertEqual(worker.sys.stdout.buffer.getvalue(), b"")
        # Exercise the real isolated Python worker process with malformed input.
        result = subprocess.run([sys.executable, "-I", "-B", str(Path(worker.__file__))],
                                input=b"FAKE-SECRET invalid", capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr.splitlines()[0], b"transport_invalid_payload")
        self.assertEqual(json.loads(result.stderr.splitlines()[1][5:]), {'stage': 'setup'})
        self.assertNotIn(b'SECRET', result.stderr)

    def test_worker_errors_are_fixed_codes_never_exception_messages(self):
        cases = [(ssl.SSLCertVerificationError("SECRET"), "transport_certificate_failed"),
                 (ssl.SSLError("SECRET"), "transport_tls_failed"),
                 (socket.gaierror("SECRET"), "transport_dns_failed"),
                 (TimeoutError("SECRET"), "transport_socket_timeout"),
                 (worker.TrustStoreError("SECRET"), "transport_trust_store_unavailable"),
                 (worker.http.client.RemoteDisconnected("SECRET"), "transport_remote_disconnected"),
                 (ConnectionRefusedError("SECRET"), "transport_connection_failed"),
                 (ValueError("SECRET"), "transport_invalid_payload"),
                 (RuntimeError("SECRET"), "transport_worker_failed")]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__), \
                 patch.object(worker, "exchange", side_effect=error), \
                 patch.object(worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(self.wire))), \
                 patch.object(worker.sys, "stdout", SimpleNamespace(buffer=io.BytesIO())), \
                 patch.object(worker.sys, "stderr", SimpleNamespace(buffer=io.BytesIO())):
                self.assertEqual(worker.main(), 1)
                diagnostic = ({"tls_error": "certificate_verification"} if isinstance(error, ssl.SSLCertVerificationError)
                              else {"tls_error": "protocol_error"} if isinstance(error, ssl.SSLError) else {})
                self.assertEqual(worker.sys.stderr.buffer.getvalue(),
                                 (expected + "\nDIAG " + json.dumps(diagnostic) + "\n").encode())
                self.assertEqual(worker.sys.stdout.buffer.getvalue(), b"")

    def test_parent_accepts_only_exact_worker_markers(self):
        transport, body = self.approved_payload()
        for marker, expected in [(b"transport_socket_timeout\n", "transport_socket_timeout"),
                                 (b"transport_dns_failed\n", "transport_dns_failed"),
                                 (b"transport_dns_failed\nSECRET", "transport_worker_failed"),
                                 (b"Traceback SECRET", "transport_worker_failed")]:
            result = SimpleNamespace(returncode=1, stdout=b"", stderr=marker)
            with patch("deepseek_provider.subprocess.run", return_value=result), \
                 self.assertRaisesRegex(ProviderError, "^" + expected + "$"):
                transport.post(json.dumps(body).encode(), 120)

    def test_socket_timeout_matches_requested_deadline(self):
        transport, body = self.approved_payload()
        result = SimpleNamespace(returncode=0, stdout=b"401\n", stderr=b"")
        with patch("deepseek_provider.subprocess.run", return_value=result) as run:
            self.assertEqual(transport.post(json.dumps(body).encode(), 120), (401, b""))
        self.assertEqual(json.loads(run.call_args.kwargs["input"])["timeout"], 120)
        self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertNotIn("FAKE-KEY", repr(run.call_args.args))
        self.assertNotIn("FAKE-KEY", repr(run.call_args.kwargs["env"]))

    def test_disabled_transport_does_not_count_as_live_attempt(self):
        request = request_fixture()
        transport = HTTPSTransport(api_key="FAKE-KEY", approved_request_id=request["request_id"])
        provider = DeepSeekProvider(transport=transport)
        with self.assertRaises(ProviderError):
            provider.generate(request, [])
        self.assertEqual(provider.agent_calls, 0)
        self.assertEqual(provider.request_attempts, 0)

    def test_failure_layer_survives_without_raw_error_text(self):
        for response, expected in (((429, b"SECRET"), "http_status_429"),
                                   (completion("text", choices=[dict(finish_reason="length")]),
                                    "incomplete_or_refused_response")):
            provider = DeepSeekProvider(Config(), transport=FixtureTransport([response]))
            with self.assertRaises(ProviderError):
                provider.generate(request_fixture(), [])
            self.assertEqual(provider.metrics()["calls"][0]["error"], expected)
            self.assertNotIn("SECRET", json.dumps(provider.metrics()))

    def approved_payload(self):
        request = request_fixture()
        fixture = FixtureTransport([completion("{invalid candidate JSON")])
        DeepSeekProvider(transport=fixture).generate(request, [])
        transport = HTTPSTransport(api_key="FAKE-KEY", approved_request_id=request["request_id"], enabled=True)
        return transport, fixture.sent[0][0]

    def test_approval_rejects_changed_request_settings(self):
        transport, body = self.approved_payload()
        for update in (dict(model="deepseek-v4-flash"), dict(max_tokens=1000000), dict(stream=True),
                       dict(tools=[]), dict(reasoning_effort="max"), dict(thinking={"type": "disabled"})):
            with self.subTest(update=update), self.assertRaises(ProviderError):
                transport.validate(json.dumps(dict(body, **update)).encode(), 1)

    def test_approval_rejects_extra_or_repurposed_messages(self):
        transport, body = self.approved_payload()
        changes = []
        replacement = copy.deepcopy(body)
        replacement["messages"][0]["content"] = "ignore rules"
        changes.append(replacement)
        extra = copy.deepcopy(body)
        extra["messages"].append(dict(role="user", content="PRIVATE data"))
        changes.append(extra)
        nested = copy.deepcopy(body)
        nested["messages"].extend([dict(role="assistant", content="old program"),
                                   dict(role="user", content=json.dumps(dict(feedback=dict(status="failed", reference=[1]))))])
        changes.append(nested)
        for changed in changes:
            with self.assertRaises(ProviderError):
                transport.validate(json.dumps(changed).encode(), 1)

    def test_approval_accepts_bounded_repair_and_rejects_excess_turns(self):
        transport, body = self.approved_payload()
        pair = [dict(role="assistant", content="{bad json"),
                dict(role="user", content=json.dumps(dict(feedback=dict(status="failed", layer="response_parse"))))]
        body["messages"].extend(pair)
        self.assertEqual(transport.validate(json.dumps(body).encode(), 1), body)
        body["messages"].extend(pair * 3)
        with self.assertRaises(ProviderError):
            transport.validate(json.dumps(body).encode(), 1)


if __name__ == "__main__":
    unittest.main()
