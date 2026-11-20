"""Real shared leases with offline Provider fixtures; no API or FHE evidence."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import candidate_shared as entry
import run_candidate as runner
from candidate_contract import ReplayProvider, run_feedback_loop
from component_control import QualificationCancelled
from deepseek_provider import Config, DeepSeekProvider
from shared_execution import Coordinator, KINDS, LIMITS
from test_candidate_pipeline import request_fixture
from test_deepseek_provider import FixtureTransport, completion


class ControlledTransport(FixtureTransport):
    is_live = True  # Live control branch only; inherited post has no socket path.


def hold_api(storage, results, limits, output, ready, release):
    c = Coordinator(storage, [results], limits=limits)
    plan = dict(branch="lhy-resnet20", source_binding="b"*64, reserve_bytes=1024,
                wall_seconds=30, stages={k: int(k == "api") for k in KINDS})
    batch, _ = c.register("b-fixture-api", plan, output)
    with batch.stage("api"):
        ready.send(True)
        if not release.poll(8):
            raise RuntimeError("Bounded cross-branch lock fixture")
        release.recv()
    batch.finish("succeeded")


class CandidateSharedTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="shared-candidate-unit-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.results = self.root / "results"
        self.results.mkdir(mode=0o700)
        self.case = self.root / "model.json"
        self.case.write_text('{"unit_fixture":true}')
        self.c = Coordinator(self.root / "control", [self.results],
            limits=dict(LIMITS, evidence_bytes=8*1024**2, batch_bytes=1024**2))
        self.host = dict(backend_identity={"fixture": True})
        for name, options in [
                ("shared_worker.environment", dict(return_value=(self.c, self.results, "lhy-generalization"))),
                ("shared_worker.snapshot", dict(side_effect=lambda: copy.deepcopy(self.host))),
                ("application_identity.source_closure", dict(return_value={"unit_sources": True}))]:
            p = patch(name, **options)
            p.start()
            self.addCleanup(p.stop)
        self.request = request_fixture()
        self.args = runner.parse_args(["--case", str(self.case), "--deepseek", "--max-repairs", "1",
                                      "--api-timeout", "60", "--max-tokens", "8192"])
        self.outputs = []

    def rows(self):
        return self.c.snapshot()[0]["batches"]

    def own(self):
        return next(b for name, b in self.rows().items() if name.startswith("shared-component-"))

    def worker(self, provider, evaluator):
        def worker(args, *, report_sink, shared, **options):
            output = Path(tempfile.mkdtemp(prefix="candidate-fixture-", dir=self.results))
            shared.observe(output)
            self.assertIn(str(output), [x["path"] for x in self.own()["artifacts"]])
            self.outputs.append(output)
            with shared.provider(provider):
                loop = run_feedback_loop(self.request, provider, shared.evaluator(evaluator),
                                         lambda *_: None, max_repairs=args.max_repairs)
            report = dict(status=loop["status"], loop=loop, attempts=[], llm_generation_validated=False)
            report_sink(report, output)
            return 0 if loop["status"] == "passed" else 1
        return worker

    def passed(self, raw, index):
        self.assertEqual(self.own()["active"]["kind"], "qualify")
        return dict(status="passed", layer="complete")

    def test_public_inside_uses_shared_entry_and_removes_credential_first(self):
        def run(args, worker, **options):
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)
            self.assertEqual(options["api_key"], "FAKE-ONLY")
            self.assertIs(worker, runner._inside)
            return 7
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "FAKE-ONLY"}), \
                patch.object(entry, "run", side_effect=run):
            self.assertEqual(runner.inside(self.args), 7)

    def test_replay_registers_before_writes_and_never_takes_api_stage(self):
        self.args.deepseek = False
        provider = ReplayProvider(["offline source fixture"])
        original = provider.generate
        self.assertEqual(entry.run(self.args, self.worker(provider, self.passed)), 0)
        self.assertEqual(provider.generate, original)
        row = self.own()
        self.assertEqual(row["status"], "succeeded")
        self.assertEqual((row["used"]["api"], row["used"]["qualify"]), (0, 1))
        report = json.loads((self.outputs[0] / "report.json").read_text())
        self.assertTrue(report["shared_control"]["globally_verified"])

    def test_original_provider_http_retry_and_repair_counters_stay_separate(self):
        self.args.provider_retries = 1
        transport = ControlledTransport([TimeoutError(), completion("first"), completion("second")])
        provider = DeepSeekProvider(Config(max_calls=2, provider_retries=1), transport=transport)
        observed = []
        provider.on_attempt = lambda: observed.append(self.own()["active"]["kind"])
        original = provider.generate
        def evaluate(raw, index):
            self.assertEqual(self.own()["active"]["kind"], "qualify")
            return dict(status="failed" if index == 0 else "passed", layer="static_check", category="candidate")
        with patch("deepseek_provider.time.sleep"):
            self.assertEqual(entry.run(self.args, self.worker(provider, evaluate)), 0)
        row = self.own()
        self.assertEqual((row["used"]["api"], row["used"]["qualify"]), (2, 2))
        self.assertEqual(provider.generate, original)
        self.assertNotIn("generate", vars(provider))
        self.assertEqual(provider.agent_calls, 3)
        self.assertEqual(len(transport.sent), 3)
        self.assertTrue(observed and all(x == "api" for x in observed))

    def test_provider_failure_keeps_used_api_and_does_not_evaluate(self):
        provider = DeepSeekProvider(Config(max_calls=2), transport=ControlledTransport([TimeoutError()]))
        self.assertEqual(entry.run(self.args, self.worker(provider, self.passed)), 1)
        row = self.own()
        self.assertEqual((row["used"]["api"], row["used"]["qualify"]), (1, 0))
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["last_stage"]["execution_uncertain"])

    def test_mutated_model_is_rejected_before_original_provider(self):
        provider = DeepSeekProvider(transport=ControlledTransport([]))
        def worker(args, *, shared, **options):
            self.case.write_text('{"unit_fixture":false}')
            with shared.provider(provider):
                provider.generate(self.request, [])
            self.fail("Mutated model was accepted")
        self.assertEqual(entry.run(self.args, worker), 1)
        self.assertEqual(provider.agent_calls, 0)
        self.assertEqual(self.own()["used"]["api"], 0)

    def test_cancelled_evaluation_preserves_uncertain_qualification(self):
        self.args.deepseek = False
        def cancel(*_):
            raise QualificationCancelled("unit fixture")
        self.assertEqual(entry.run(self.args, self.worker(ReplayProvider(["fixture"]), cancel)), 130)
        row = self.own()
        self.assertEqual(row["status"], "cancelled")
        self.assertEqual(row["used"]["qualify"], 1)
        self.assertTrue(row["last_stage"]["execution_uncertain"])

    def test_late_shared_failure_cannot_leave_success_report_or_sink(self):
        received = []
        self.args.deepseek = False
        original = self.worker(ReplayProvider(["fixture"]), self.passed)
        def worker(*args, **options):
            value = original(*args, **options)
            self.host["backend_identity"]["changed"] = True
            return value
        self.assertEqual(entry.run(self.args, worker, report_sink=lambda r, p: received.append(r)), 1)
        self.assertEqual(received[0]["status"], "infrastructure_failed")
        self.assertFalse(received[0]["llm_generation_validated"])
        self.assertFalse(received[0]["shared_control"]["globally_verified"])
        self.assertEqual(self.own()["status"], "failed")
        self.assertTrue((self.outputs[0] / "report.json").is_file())

    def test_cli_qualification_waits_for_other_branch_api(self):
        ctx = mp.get_context("fork")
        ready_a, ready_b = ctx.Pipe()
        release_a, release_b = ctx.Pipe()
        output = self.results / "b-output"
        output.mkdir(mode=0o700)
        child = ctx.Process(target=hold_api, args=(self.c.storage, self.results, self.c.limits,
                                                  output, ready_b, release_b))
        child.start()
        try:
            self.assertTrue(ready_a.poll(8) and ready_a.recv())
            entered = threading.Event()
            def release():
                time.sleep(.2)
                self.assertFalse(entered.is_set())
                release_a.send(True)
            thread = threading.Thread(target=release)
            thread.start()
            def evaluate(*args):
                entered.set()
                return self.passed(*args)
            self.args.deepseek = False
            self.assertEqual(entry.run(self.args, self.worker(ReplayProvider(["fixture"]), evaluate)), 0)
            thread.join(8)
            self.assertFalse(thread.is_alive())
            self.assertTrue(entered.is_set())
            child.join(8)
            self.assertEqual(child.exitcode, 0)
        finally:
            if child.is_alive():
                child.terminate()
                child.join(5)
            for pipe in (ready_a, ready_b, release_a, release_b):
                pipe.close()


if __name__ == "__main__":
    unittest.main()
