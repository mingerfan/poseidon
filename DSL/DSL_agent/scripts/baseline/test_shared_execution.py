"""Real cross-process lease/accounting tests; no API, compiler or FHE calls."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from shared_execution import (Batch, Coordinator, CoordinatedBackend, coordinated_provider,
                              GIB, KINDS, LIMITS, digest)


def plan(branch="lhy-generalization", reserve=1024, **counts):
    return dict(branch=branch, source_binding="a"*64, reserve_bytes=reserve,
                wall_seconds=60, stages={kind: counts.get(kind, 0) for kind in KINDS})


def hold(storage, root, limits, output, name, kind, ready, release, crash=False):
    c=Coordinator(storage,[root],limits=limits)
    b,_=c.register(name,plan(api=1,encrypted=1,compile=1,link=1,build=1,qualify=1),output)
    with b.stage(kind):
        ready.send(name)
        if not release.poll(8):
            raise RuntimeError("Lease test bounded wait expired")
        release.recv()
        if crash:
            os._exit(23)
    b.finish("succeeded")


class SharedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="poseidon-shared-control-")
        self.base=Path(self.temp.name)
        self.root=self.base/"results"
        self.root.mkdir(mode=0o700)
        self.storage=self.base/"coordination"
        self.limits=dict(LIMITS,evidence_bytes=8192,batch_bytes=2048)
        self.c=Coordinator(self.storage,[self.root],limits=self.limits)
        self.children=[]
        self.ctx=mp.get_context("spawn")

    def tearDown(self):
        for child,release in self.children:
            try:
                release.send(True)
            except (BrokenPipeError, OSError):
                pass
            child.join(3)
            if child.is_alive():
                child.kill()
                child.join(3)
        self.temp.cleanup()

    def output(self,name):
        p=self.root/name
        p.mkdir(mode=0o700)
        return p

    def register(self,name="batch-a",value=None):
        return self.c.register(name,plan() if value is None else value,self.output(name))[0]

    def child(self,name,kind,crash=False):
        ready,ready_child=self.ctx.Pipe(duplex=False)
        release_child,release=self.ctx.Pipe(duplex=False)
        child=self.ctx.Process(target=hold,args=(str(self.storage),str(self.root),self.limits,
            str(self.output(name)),name,kind,ready_child,release_child,crash))
        child.start()
        self.children.append((child,release))
        ready_child.close()
        release_child.close()
        self.assertTrue(ready.poll(10),"Lease test child did not start")
        self.assertEqual(ready.recv(),name)
        ready.close()
        return child,release

    def test_shared_reservations_are_not_added_per_branch(self):
        a=self.register(value=plan(reserve=2048))
        other=Coordinator(self.storage,[self.root],limits=self.limits)
        b,_=other.register("batch-b",plan("lhy-resnet20",reserve=2048),self.output("batch-b"))
        self.assertEqual(other.snapshot()[1]["remaining_reserved_bytes"],4096)
        a.finish("cancelled")
        self.assertEqual(self.c.snapshot()[1]["remaining_reserved_bytes"],2048)
        b.finish("succeeded")

    def test_global_cap_rejects_without_creating_a_ledger_entry(self):
        c=Coordinator(self.base/"small-coordination",[self.root],limits=dict(self.limits,evidence_bytes=1536))
        c.register("one",plan(),self.output("one"))
        with self.assertRaisesRegex(ValueError,"Global evidence"):
            c.register("two",plan("lhy-resnet20"),self.output("two"))
        self.assertEqual(set(c.snapshot()[0]["batches"]),{"one"})

    def test_observed_bytes_replace_the_reservation_without_double_counting(self):
        b=self.register()
        (self.root/"batch-a"/"retained").write_bytes(b"x"*200)
        usage=b.checkpoint()
        self.assertEqual(usage["observed_bytes"],200)
        self.assertEqual(usage["remaining_reserved_bytes"],824)
        b.finish("succeeded")
        self.assertEqual(self.c.snapshot()[1]["remaining_reserved_bytes"],0)
        self.assertEqual(self.c.snapshot()[1]["observed_bytes"],200)

    def test_per_batch_overrun_is_not_success(self):
        b=self.register(value=plan(reserve=8))
        (self.root/"batch-a"/"retained").write_bytes(b"x"*9)
        with self.assertRaisesRegex(ValueError,"Batch artifact"):
            b.checkpoint()
        with self.assertRaisesRegex(ValueError,"Finished batch"):
            b.finish("succeeded")
        self.assertEqual(b.finish("failed")["observed_bytes"],9)

    def test_disk_reserve_admits_no_new_work(self):
        with patch("shared_execution.shutil.disk_usage") as usage:
            usage.return_value.free=4*GIB
            with self.assertRaisesRegex(ValueError,"disk reserve"):
                self.register()
        self.assertFalse(self.c.snapshot()[0]["batches"])

    def test_policy_cannot_expand_limits_or_reinterpret_existing_ledger(self):
        with self.assertRaisesRegex(ValueError,"expansion"):
            Coordinator(self.base/"expanded",[self.root],limits=dict(LIMITS,evidence_bytes=LIMITS["evidence_bytes"]+GIB))
        with self.assertRaisesRegex(ValueError,"binding/policy"):
            Coordinator(self.storage,[self.root],limits=dict(self.limits,batch_bytes=1024))

    def test_stage_attempts_are_nonrefundable_and_zero_budget_never_enters(self):
        b=self.register(value=plan(api=1))
        with self.assertRaisesRegex(RuntimeError,"transport unknown"):
            with b.stage("api"):
                raise RuntimeError("transport unknown")
        row=self.c.snapshot()[0]["batches"]["batch-a"]
        self.assertEqual(row["used"]["api"],1)
        self.assertTrue(row["last_stage"]["execution_uncertain"])
        with self.assertRaisesRegex(ValueError,"budget exhausted"):
            with b.stage("api"):
                self.fail("A spent generation may not be repeated")
        with self.assertRaisesRegex(ValueError,"budget exhausted"):
            with b.stage("encrypted"):
                self.fail("Unplanned native execution")

    def test_api_and_encrypted_leases_are_serial_across_processes(self):
        child,release=self.child("api-owner","api")
        b=self.register(value=plan(encrypted=1))
        with self.assertRaises(TimeoutError):
            with b.stage("encrypted",timeout=0.08):
                self.fail("Global API/FHE lease overlap")
        self.assertEqual(self.c.snapshot()[0]["batches"]["batch-a"]["used"]["encrypted"],0)
        release.send(True)
        child.join(5)
        self.assertEqual(child.exitcode,0)
        with b.stage("encrypted",timeout=1):
            pass

    def test_compile_two_link_one_and_build_leases_share_the_same_slots(self):
        first,release_first=self.child("compile-one","compile")
        second,release_second=self.child("compile-two","compile")
        b=self.register(value=plan(compile=1,link=1,build=1))
        for kind in ("compile","link","build"):
            with self.assertRaises(TimeoutError):
                with b.stage(kind,timeout=0.06):
                    self.fail("Global compiler capacity exceeded")
        release_first.send(True)
        first.join(5)
        self.assertEqual(first.exitcode,0)
        with b.stage("link",timeout=1):
            other=self.register("another-link",plan(link=1))
            with self.assertRaises(TimeoutError):
                with other.stage("link",timeout=0.06):
                    self.fail("Two global links overlapped")
        release_second.send(True)
        second.join(5)
        self.assertEqual(second.exitcode,0)
        with b.stage("build",timeout=1):
            pass

    def test_killed_api_owner_keeps_uncertainty_until_explicit_recovery(self):
        child,release=self.child("interrupted-api","api",crash=True)
        with self.assertRaisesRegex(ValueError,"still alive"):
            self.c.recover_interrupted("interrupted-api")
        release.send(True)
        child.join(5)
        self.assertEqual(child.exitcode,23)
        before=self.c.snapshot()[0]["batches"]["interrupted-api"]
        self.assertEqual(before["status"],"running")
        self.assertEqual(before["active"]["kind"],"api")
        after=self.c.recover_interrupted("interrupted-api")
        self.assertEqual(after["status"],"interrupted")
        self.assertEqual(after["used"]["api"],1)
        self.assertEqual(after["uncertain_stage"]["kind"],"api")
        b=self.register(value=plan(api=1))
        with b.stage("api",timeout=1):
            pass
        with self.assertRaisesRegex(ValueError,"already recorded"):
            self.c.register("interrupted-api",plan(),self.output("not-replayed"))

    def test_terminal_batch_cannot_resume(self):
        b=self.register(value=plan(api=1))
        b.finish("cancelled")
        with self.assertRaisesRegex(ValueError,"ownership/plan"):
            with b.stage("api"):
                self.fail("Cancelled batch resumed")

    def test_artifact_registration_preserves_ownership_and_root_identity(self):
        b=self.register()
        extra=self.output("external-native-output")
        b.register_artifact(extra)
        (extra/"trace").write_bytes(b"x"*100)
        self.assertEqual(b.checkpoint()["observed_bytes"],100)
        c=self.register("batch-b",plan("lhy-resnet20"))
        with self.assertRaisesRegex(ValueError,"overlap"):
            c.register_artifact(self.output("external-native-output/child"))
        moved=self.root/"moved"
        extra.rename(moved)
        extra.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError,"identity changed"):
            b.checkpoint()

    def test_symlink_outputs_and_lock_files_are_rejected(self):
        link=self.root/"link"
        link.symlink_to(self.base,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"Indirect"):
            self.c.register("link",plan(),link)
        link.unlink()
        b=self.register(value=plan(api=1))
        lock=self.storage/"api-encrypted.lock"
        lock.symlink_to(self.base/"outside")
        with self.assertRaises(OSError):
            with b.stage("api"):
                self.fail("Symlink lock opened")

    def test_budget_clock_and_ledger_tampering_fail_closed(self):
        b=self.register()
        row=self.c.snapshot()[0]["batches"]["batch-a"]
        with patch("shared_execution.time.monotonic",return_value=row["started_monotonic"]+61):
            with self.assertRaisesRegex(ValueError,"wall budget"):
                b.checkpoint()
        data=json.loads(self.c.journal.read_bytes())
        data["batches"]["batch-a"]["plan"]["reserve_bytes"]=2048
        self.c.journal.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"binding/policy"):
            b.checkpoint()

    def test_wrappers_preserve_existing_provider_hooks_and_artifact_observers(self):
        b=self.register(value=plan(api=1,qualify=1,encrypted=1))
        class Provider:
            kind="test_fixture"
            before_attempt=None
            def generate(self,request,history):
                self.before_attempt()
                return "retained response"
        called=[]
        raw=Provider()
        with coordinated_provider(b,raw) as provider:
            self.assertIs(provider,raw)
            provider.before_attempt=lambda:called.append("existing HTTP reservation")
            self.assertEqual(provider.generate({},[]),"retained response")
        self.assertEqual(called,["existing HTTP reservation"])
        test=self
        class Backend:
            def qualify(self,request,candidate,level,*,artifact_observer):
                p=test.output("qualified-output")
                artifact_observer(p)
                return {"fixture":True}
            def execute(self,folder,samples,*,artifact_observer,sample_format="named"):
                return sample_format
        backend=CoordinatedBackend(b,Backend())
        observed=[]
        self.assertEqual(backend.qualify({},"source","numerical",artifact_observer=observed.append),{"fixture":True})
        self.assertEqual(len(observed),1)
        self.assertEqual(backend.execute("package",[],sample_format="forward_call"),"forward_call")

    def test_existing_application_keeps_paid_provider_type_check_and_http_accounting(self):
        from application_component import Application
        from deepseek_provider import DeepSeekProvider
        from test_application_generation import GenerationTests
        fixture=GenerationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        # The production adapter accepts an artifact observer. Preserve that
        # signature on the existing fixture while keeping its real lifecycle.
        class Backend(type(fixture.backend)):
            def qualify(self,request,candidate,level,*,artifact_observer=None):
                return super().qualify(request,candidate,level)
        backend=Backend()
        backend.failures=["static_check"]
        results=self.base/"application-results"
        results.mkdir(mode=0o700)
        output=results/"batch"
        output.mkdir(mode=0o700)
        c=Coordinator(self.base/"application-coordination",[results],
            limits=dict(LIMITS,evidence_bytes=64*1024**2,batch_bytes=32*1024**2))
        b,_=c.register("application",plan(reserve=32*1024**2,api=2,qualify=2),output)
        app=Application(output/"tasks",CoordinatedBackend(b,backend))
        changed=fixture.source.replace("x*x","x*x+zero_ct")
        provider=fixture.provider([TimeoutError(),fixture.source,changed],provider_retries=1)
        original=provider.generate
        with patch("deepseek_provider.time.sleep"),coordinated_provider(b,provider) as same:
            self.assertIs(same,provider)
            self.assertIsInstance(same,DeepSeekProvider)
            result=app.run(app.submit(fixture.model),same)
        self.assertEqual(result["status"],"succeeded")
        self.assertEqual(result["generation"]["generations"],2)
        self.assertEqual(result["generation"]["http_attempts"],3)
        self.assertEqual(provider.generate,original)
        self.assertIsNone(provider.before_attempt)
        self.assertIsNone(provider.on_attempt)
        row=b.finish("succeeded")
        self.assertEqual(row["used"]["api"],2)
        self.assertEqual(row["used"]["qualify"],2)


if __name__ == "__main__":
    unittest.main()
