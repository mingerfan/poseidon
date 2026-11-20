"""Shared Application lifecycle with synthetic backend/transport, never FHE evidence."""
import copy
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from application_shared import SharedApplication, FIXED_LIMITS
from benchmark_suite import Builder
from shared_execution import Coordinator, LIMITS, KINDS, digest, process
from test_application_component import Backend, Provider
from test_application_generation import SyntheticTransport
from deepseek_provider import Config, DeepSeekProvider
import test_application_identity as identity_fixture


def hold_task(path, ready, release):
    import fcntl
    with (Path(path)/"lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        ready.send(True)
        if not release.poll(8):
            raise RuntimeError("Bounded task-lock fixture wait")
        release.recv()


def interrupt_api(storage, results, limits, task, output, ready, release):
    c=Coordinator(storage,[results],limits=limits)
    path=Path(task)
    inputs=dict(task_store=str(path.parent),task_id=path.name)
    plan=dict(branch="lhy-generalization",source_binding=digest(inputs),reserve_bytes=1024**2,
        wall_seconds=30,stages={kind:int(kind=="api")for kind in KINDS})
    b,_=c.register("shared-component-crash",plan,output)
    b.watch_growth(path)
    (Path(output)/"plan.json").write_text(json.dumps(dict(authority=dict(
        action="application-run",input_sha256=digest(inputs)))))
    record=dict(format="poseidon-application-shared-operation-v1",action="application-run",
        task_store=str(path.parent),task_id=path.name,operation_input=inputs,
        batch_name=b.name,plan_binding=b.binding,scope=str(output),owner=process())
    record["binding"]=digest(record)
    (path/(b.name+".json")).write_text(json.dumps(record))
    with b.stage("api"):
        ready.send(True)
        if not release.poll(8):raise RuntimeError("Bounded interrupted-stage fixture wait")
        release.recv()
        os._exit(23)


class SharedApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shared-app-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir(mode=0o700)
        self.c = Coordinator(self.root / "coordination", [self.results],
            limits=dict(LIMITS, evidence_bytes=32*1024**2, batch_bytes=8*1024**2))
        fixture = identity_fixture.IdentityTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        identities = fixture.manifest()
        test = self
        class ObservedBackend(Backend):
            artifact_monitor_version = "poseidon-artifact-monitor-v1"
            reject_request = False
            receipt = None
            def operational_artifact_roots(self):
                return (test.results,)
            def identity_manifest(self):
                return copy.deepcopy(identities)
            def check_request(self, request, level):
                if self.reject_request:
                    raise ValueError("Request/SDK configuration binding mismatch")
            def qualify(self, request, candidate, level, *, artifact_observer=None):
                out = Path(tempfile.mkdtemp(prefix="candidate-replay-", dir=test.results))
                if artifact_observer is not None:
                    artifact_observer(out)
                return super().qualify(request, candidate, level)
            def export(self, result, destination, level, *, generation_provenance=None, model_source=None, model_provenance=None):
                self.receipt = generation_provenance
                super().export(result, destination, level)
            def execute(self, folder, samples, *, artifact_observer=None, sample_format="named"):
                out = Path(tempfile.mkdtemp(prefix="poseidon-program-execution-", dir=test.results))
                if artifact_observer is not None:
                    artifact_observer(out)
                (out / "fixture").write_text("mock execution")
                return dict(status="executed", encrypted_execution=False, fixture=True, agent_calls=0, compiler_calls=0)
        self.backend = ObservedBackend()
        env = (self.c, self.results, "lhy-generalization")
        for name in ["application_shared.environment", "shared_worker.environment"]:
            p = patch(name, return_value=env)
            p.start()
            self.addCleanup(p.stop)
        p = patch("shared_worker.snapshot", return_value={"backend_identity":{"unit_fixture":True}})
        p.start();self.addCleanup(p.stop)
        p = patch("application_shared.authority", return_value={"sources":{"fixture.py":"a"*64}})
        p.start();self.addCleanup(p.stop)
        p = patch("poseidon_runtime_paths.selected_sdk", return_value=None)
        p.start();self.addCleanup(p.stop)
        self.app = SharedApplication(self.root / "tasks", self.backend)
        builder = Builder([(2,)])
        self.model = builder.finish(builder.node("square", ["input0"]))

    def rows(self):
        return sorted(self.c.snapshot()[0]["batches"].values(), key=lambda row: row["started_monotonic"])

    def test_external_store_submission_is_queued_and_globally_counted(self):
        task = self.app.submit(self.model)
        self.assertEqual(self.app.status(task)["status"], "queued")
        self.assertEqual(self.app._checked_task_metadata(task)["operational_limits"], FIXED_LIMITS)
        row = self.rows()[0]
        self.assertEqual(row["status"], "succeeded")
        self.assertFalse(any(row["used"].values()))
        self.assertEqual(len(row["growth_scopes"]), 1)
        self.assertEqual(row["growth_scopes"][0]["path"], str(self.app._dir(task)))
        self.assertGreater(row["observed_bytes"], 0)

    def test_replay_qualifies_once_and_cache_reuse_does_not_generate(self):
        first = self.app.submit(self.model)
        provider = Provider()
        self.assertEqual(self.app.run(first, provider)["status"], "succeeded")
        second = self.app.submit(self.model)
        unused = Provider()
        result = self.app.run(second, unused)
        self.assertTrue(result["package"]["reused"])
        self.assertEqual(unused.calls, 0)
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(sum(r["used"]["qualify"] for r in self.rows()), 1)
        self.assertEqual(sum(r["used"]["api"] for r in self.rows()), 0)
        self.assertIs(self.app.backend, self.backend)

    def test_sdk_preflight_failure_precedes_provider_and_backend(self):
        task = self.app.submit(self.model)
        self.backend.reject_request = True
        provider = Provider()
        result = self.app.run(task, provider)
        self.assertEqual(result["status"], "service_error")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(self.backend.calls, 0)
        self.assertFalse(any(self.rows()[-1]["used"].values()))

    def test_original_http_hooks_type_check_and_generation_receipt_survive(self):
        task = self.app.submit(self.model)
        request = self.app._checked_task_metadata(task)["request"]
        source = '@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x*x]\n'
        raw = json.dumps(dict(schema=1, request_id=request["request_id"], hecate_source=source))
        provider = DeepSeekProvider(Config(max_calls=1), transport=SyntheticTransport([raw]))
        result = self.app.run(task, provider)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["http_attempts"], 1)
        self.assertIsNone(provider.before_attempt)
        self.assertIsNone(provider.on_attempt)
        self.assertNotIn("generate", vars(provider))
        self.assertEqual(sum(r["used"]["api"] for r in self.rows()), 1)
        self.assertEqual(sum(r["used"]["qualify"] for r in self.rows()), 1)
        self.assertIsNotNone(self.backend.receipt)

    def test_execute_uses_encrypted_lease_without_generation_or_compilation(self):
        task = self.app.submit(self.model)
        self.assertEqual(self.app.run(task, Provider())["status"], "succeeded")
        result = self.app.execute(task, [{"input0":[.1,.2]}])
        self.assertEqual(result["status"], "executed")
        self.assertTrue(result["fixture"])
        row = self.rows()[-1]
        self.assertEqual(row["used"]["encrypted"], 1)
        self.assertEqual(row["used"]["api"], 0)
        self.assertEqual(row["used"]["qualify"], 0)

    def test_cancelled_queued_task_never_generates(self):
        task = self.app.submit(self.model)
        self.app.cancel(task)
        provider = Provider()
        self.assertEqual(self.app.run(task, provider)["status"], "cancelled")
        self.assertEqual(provider.calls, 0)
        self.assertEqual(self.backend.calls, 0)

    def test_shared_final_check_cannot_leave_a_successful_cache_entry(self):
        task = self.app.submit(self.model)
        with patch("shared_worker.Session.verify", side_effect=[None,ValueError("fixture source drift")]):
            result = self.app.run(task, Provider())
        self.assertEqual(result["status"], "service_error")
        self.assertIsNone(self.app.result(task)["package"])
        self.assertTrue((self.app._dir(task)/"program").is_dir())

    def test_second_worker_does_not_overwrite_an_active_task(self):
        task=self.app.submit(self.model)
        path=self.app._dir(task)
        before=(path/"state.json").read_bytes()
        ctx=mp.get_context("spawn")
        ready,child_ready=ctx.Pipe(duplex=False)
        child_release,release=ctx.Pipe(duplex=False)
        child=ctx.Process(target=hold_task,args=(str(path),child_ready,child_release))
        child.start();child_ready.close();child_release.close()
        try:
            self.assertTrue(ready.poll(10));ready.recv()
            provider=Provider()
            with self.assertRaises(BlockingIOError):self.app.run(task,provider)
            self.assertEqual(provider.calls,0)
            self.assertEqual((path/"state.json").read_bytes(),before)
        finally:
            release.send(True);child.join(3)
            if child.is_alive():child.kill();child.join(3)
            ready.close();release.close()
        self.assertEqual(child.exitcode,0)

    def test_explicit_recovery_preserves_unknown_api_attempt_and_does_not_replay(self):
        task=self.app.submit(self.model)
        path=self.app._dir(task)
        state=self.app._load(path/"state.json")
        state.update(status="running",phase="generating")
        self.app._write(path/"state.json",state)
        output=self.results/"interrupted-worker"
        output.mkdir(mode=0o700)
        ctx=mp.get_context("spawn")
        ready,child_ready=ctx.Pipe(duplex=False)
        child_release,release=ctx.Pipe(duplex=False)
        child=ctx.Process(target=interrupt_api,args=(str(self.c.storage),str(self.results),self.c.limits,
            str(path),str(output),child_ready,child_release))
        child.start();child_ready.close();child_release.close()
        try:
            self.assertTrue(ready.poll(10));ready.recv()
            release.send(True);child.join(3)
            self.assertEqual(child.exitcode,23)
            provider=Provider()
            with self.assertRaisesRegex(ValueError,"Recover interrupted"):
                self.app.run(task,provider)
            self.assertEqual(provider.calls,0)
            result=self.app.recover(task)
            self.assertEqual(result["failure"]["code"],"interrupted_outcome_unknown")
            row=self.c.snapshot()[0]["batches"]["shared-component-crash"]
            self.assertEqual(row["status"],"interrupted")
            self.assertEqual(row["used"]["api"],1)
            self.assertEqual(row["uncertain_stage"]["kind"],"api")
            self.assertEqual(self.app.run(task,provider)["status"],"service_error")
            self.assertEqual(provider.calls,0)
        finally:
            if child.is_alive():child.kill();child.join(3)
            ready.close();release.close()


if __name__ == "__main__":
    unittest.main()
