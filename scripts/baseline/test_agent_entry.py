"""Agent CLI wiring tests; no model service or real credentials."""
import json
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

if sys.platform == "linux":
    import run_candidate as runner
    import hecate_python_env as environment
from test_candidate_pipeline import request_fixture


@unittest.skipUnless(sys.platform == "linux", "runner requires WSL/Linux")
class AgentEntryTests(unittest.TestCase):
    def test_cli_loads_local_file_only_for_live_mode_and_cleans_up(self):
        args = runner.parse_args(["--case", "model.json", "--deepseek", "--provider", "deepseek"])
        def enter(command, **kwargs):
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "FAKE-FILE-KEY")
            self.assertNotIn("FAKE-FILE-KEY", command)
            self.assertEqual(kwargs["keep_env"], ("DEEPSEEK_API_KEY",))
            return 0
        with patch.object(runner, "parse_args", return_value=args), \
             patch.object(runner.Path, "cwd", return_value=runner.ROOT), \
             patch.dict(os.environ, {}, clear=True), \
             patch("agent_credentials.load_api_key", return_value="FAKE-FILE-KEY") as loader, \
             patch.object(runner, "enter_nix", side_effect=enter):
            self.assertEqual(runner.main(), 0)
            loader.assert_called_once_with(runner.ROOT, provider="deepseek")
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
            args.deepseek = False
            args.prepare = True
            with patch.object(runner, "enter_nix", return_value=0):
                self.assertEqual(runner.main(), 0)
            loader.assert_called_once()

    def test_credential_environment_restored_on_nix_failure(self):
        args = runner.parse_args(["--case", "model.json", "--deepseek"])
        with patch.object(runner, "parse_args", return_value=args), \
             patch.object(runner.Path, "cwd", return_value=runner.ROOT), \
             patch.dict(os.environ, {"DEEPSEEK_API_KEY": "FAKE-OVERRIDE"}, clear=True), \
             patch("agent_credentials.load_api_key", return_value="FAKE-OVERRIDE"), \
             patch.object(runner, "enter_nix", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                runner.main()
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "FAKE-OVERRIDE")

    def test_terminal_outcome_does_not_echo_unknown_provider_errors(self):
        for error, expected in [("transport_socket_timeout", "transport_socket_timeout"),
                                ("http_status_401", "http_status_401"),
                                ("SECRET raw error", "transport_worker_failed")]:
            report = dict(status="provider_failed", provider_metrics=dict(calls=[dict(error=error)]))
            with patch.object(sys, "stdout", io.StringIO()) as output:
                runner.print_outcome(report)
                text = output.getvalue()
            self.assertIn("Candidate status: provider_failed", text)
            self.assertIn("Provider failure: " + expected, text)
            self.assertNotIn("SECRET", text)

    def test_live_cli_and_repair_limit(self):
        args = runner.parse_args(["--case", "model.json", "--deepseek", "--max-repairs", "2"])
        self.assertTrue(args.deepseek)
        self.assertEqual(args.max_repairs, 2)
        self.assertIn("--deepseek", runner.forward_options(args))
        self.assertNotIn("--replay", runner.forward_options(args))

    def test_excess_repairs_rejected(self):
        with self.assertRaises((ValueError, SystemExit)):
            runner.parse_args(["--case", "model.json", "--deepseek", "--max-repairs", "4"])

    def test_provider_selection_uses_deepseek_and_never_rule_answer(self):
        args = runner.parse_args(["--case", "model.json", "--deepseek", "--max-repairs", "1"])
        with patch("deepseek_provider.HTTPSTransport.post", side_effect=AssertionError("no network")):
            provider = runner.select_provider(args, request_fixture(), api_key="FAKE-KEY")
        self.assertEqual(provider.kind, "deepseek_api")
        self.assertEqual(provider.config.max_calls, 2)
        self.assertEqual(provider.agent_calls, 0)
        self.assertNotIn("FAKE-KEY", json.dumps(provider.metrics()))

    def test_missing_key_stops_without_fallback(self):
        args = runner.parse_args(["--case", "model.json", "--deepseek"])
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            runner.select_provider(args, request_fixture(), api_key="")

    def test_replay_never_uses_api_key(self):
        args = runner.parse_args(["--case", "model.json", "--replay", "answers.json"])
        provider = runner.select_provider(args, request_fixture(), responses=["saved answer"])
        self.assertEqual(provider.kind, "scripted_replay")
        self.assertEqual(provider.agent_calls, 0)

    def test_manual_golden_never_discovers_credentials(self):
        args = runner.parse_args(["--case", "custom.json", "--golden-file", "manual.py"])
        self.assertFalse(args.deepseek)
        self.assertIn("--golden-file", runner.forward_options(args))
        with patch.object(runner, "parse_args", return_value=args), \
             patch.object(runner.Path, "cwd", return_value=runner.ROOT), \
             patch("agent_credentials.load_api_key", side_effect=AssertionError("offline key read")), \
             patch.object(runner, "enter_nix", return_value=0):
            self.assertEqual(runner.main(), 0)

    def test_pure_environment_keeps_only_explicit_agent_key(self):
        calls = []
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            if "eval" in argv:
                return SimpleNamespace(stdout=json.dumps(dict(shell_bash="/nix/store/fake/bin/bash")))
            return SimpleNamespace(returncode=0)
        with patch.object(environment.subprocess, "run", side_effect=run):
            environment.enter_nix("trusted-command", seconds=60, keep_env=("DEEPSEEK_API_KEY",))
        command = calls[-1][0]
        self.assertIn("--pure", command)
        self.assertEqual(command[command.index("--keep")+1], "DEEPSEEK_API_KEY")
        with self.assertRaises(ValueError):
            environment.enter_nix("trusted-command", keep_env=("UNRELATED_SECRET",))

    def test_live_report_needs_encrypted_numeric_success(self):
        provider = SimpleNamespace(kind="deepseek_api", agent_calls=1,
                                   metrics=lambda: dict(agent_calls=1, calls=[]))
        report = dict(status="passed", attempts=[dict(checked=True, executed=False)])
        runner.update_provider_report(report, provider)
        self.assertEqual(report["agent_calls"], 1)
        self.assertFalse(report["llm_generation_validated"])
        report["attempts"] = [dict(numerically_correct=True, executed=True,
                                   execution=dict(encrypted_execution=True))]
        runner.update_provider_report(report, provider)
        self.assertTrue(report["llm_generation_validated"])
        offline = SimpleNamespace(kind="deepseek_offline", agent_calls=0, metrics=lambda: {})
        runner.update_provider_report(report, offline)
        self.assertFalse(report["llm_generation_validated"])


if __name__ == "__main__":
    unittest.main()
