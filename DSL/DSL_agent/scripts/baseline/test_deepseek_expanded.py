"""Offline tests for the explicitly authorized expanded generation experiment."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepseek_provider import Config, DeepSeekProvider, HTTPSTransport, ProviderError
from run_agent_batch import load_failed_source, configuration_delta, generation_settings
from test_candidate_pipeline import request_fixture
from test_deepseek_provider import FixtureTransport, completion


class ExpandedGenerationTests(unittest.TestCase):
    def test_defaults_and_finite_expanded_caps(self):
        self.assertEqual((Config().reasoning_effort, Config().max_tokens, Config().timeout_seconds),
                         ("high", 8192, 120))
        Config(reasoning_effort="low", max_tokens=65536, timeout_seconds=900)
        Config(timeout_seconds=1200)
        for change in (dict(max_tokens=384001), dict(timeout_seconds=1201),
                       dict(timeout_seconds=float("inf")), dict(max_tokens=True)):
            with self.assertRaises(ProviderError):
                Config(**change)

    def test_expanded_payload_and_transport_deadline_are_bound_to_approval(self):
        config = Config(reasoning_effort="low", max_tokens=65536, timeout_seconds=900)
        request = request_fixture()
        fixture = FixtureTransport([completion("{}")] )
        provider = DeepSeekProvider(config, transport=fixture)
        provider.generate(request, [])
        body, timeout = fixture.sent[0]
        body = json.dumps(body).encode()
        transport = HTTPSTransport(api_key="FAKE-KEY", approved_request_id=request["request_id"],
                                  enabled=True, approved_config=config)
        result = SimpleNamespace(returncode=0, stdout=b"401\n", stderr=b"")
        with patch("deepseek_provider.subprocess.run", return_value=result) as run:
            transport.post(body, timeout)
        self.assertEqual(timeout, 900)
        self.assertEqual(run.call_args.kwargs["timeout"], 900)
        payload = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(payload["body"]["max_tokens"], 65536)
        self.assertEqual(payload["body"]["reasoning_effort"], "low")
        self.assertNotIn("FAKE-KEY", repr(run.call_args.args))
        with self.assertRaises(ProviderError):
            HTTPSTransport(api_key="FAKE-KEY", approved_request_id=request["request_id"],
                          enabled=True).validate(body, 900)

    @unittest.skipUnless(sys.platform == "linux", "Pinned Linux runner")
    def test_single_case_cli_forwards_new_configuration(self):
        from run_candidate import parse_args, forward_options
        args = parse_args(["--case", "model.json", "--deepseek", "--reasoning-effort", "low",
                           "--max-tokens", "65536", "--api-timeout", "900"])
        options = forward_options(args)
        for text in ("--reasoning-effort low", "--max-tokens 65536", "--api-timeout 900"):
            self.assertIn(text, options)

    def test_recursive_selection_and_hash_guard(self):
        catalog = [dict(id=str(i)) for i in range(3)]
        def report(items, passed):
            return dict(status="completed_with_failures", cases=[dict(descriptor=d,
                status="passed" if d["id"] in passed else "failed",
                metrics=dict(passed=d["id"] in passed)) for d in items])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = root / "first.json", root / "second.json"
            first.write_text(json.dumps(report(catalog, {"0"})))
            subset = report(catalog[1:], {"1"})
            subset["selection"] = dict(source_report=str(first),
                source_sha256=hashlib.sha256(first.read_bytes()).hexdigest())
            second.write_text(json.dumps(subset))
            _, selected = load_failed_source(catalog, second, root)
            self.assertEqual(selected, [catalog[2]])
            first.write_text(first.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_failed_source(catalog, second, root)

    def test_configuration_delta_records_only_intended_fields(self):
        old = generation_settings(dict(model="deepseek-v4-pro", max_repairs=3,
                                       api_timeout=120, max_tokens=8192))
        new = dict(old, reasoning_effort="low", max_tokens=65536, timeout_seconds=900)
        self.assertEqual(set(configuration_delta(old, new)),
                         {"reasoning_effort", "max_tokens", "timeout_seconds"})


if __name__ == "__main__":
    unittest.main()
