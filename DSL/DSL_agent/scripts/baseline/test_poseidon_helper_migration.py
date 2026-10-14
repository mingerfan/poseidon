"""Poseidon helper source binding and explicit compiler configuration boundaries."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmark_suite import Builder
from benchmark_graph import digest
from compiler_configuration import configuration,validate_configuration
from component_contract import prepare_task,reconstruct_request,capabilities
from unified_graph_contract import validate_request
from upstream_candidate_helpers import PROFILE,manifest,verify_sources,POSEIDON_LOCK
from poseidon_runtime_compile import command
class HelperMigrationTests(unittest.TestCase):
    def graph(self):
        b=Builder([(2,)])
        return b.finish(b.node("square",["input0"]))
    def test_source_locks_and_requests_are_distinct(self):
        old=prepare_task(self.graph(),dict(compiler_configuration="seal-cpu-eva-w40-v1",helper_profile=PROFILE))
        new=prepare_task(self.graph(),dict(compiler_configuration="poseidon-cpu-eva-w40-v1",helper_profile=PROFILE))
        self.assertNotIn("source_backend",old["upstream_helpers"])
        self.assertEqual(new["upstream_helpers"]["source_backend"],"poseidon_cpu")
        self.assertNotEqual(old["upstream_helpers"]["source_lock_sha256"],new["upstream_helpers"]["source_lock_sha256"])
        self.assertNotEqual(old["request_id"],new["request_id"])
        for r in (old,new):self.assertEqual(reconstruct_request(r),r)
    def test_cross_backend_manifest_rejected_even_when_rehashed(self):
        r=prepare_task(self.graph(),dict(helper_profile=PROFILE))
        r["upstream_helpers"]=manifest()
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(r)
    def test_new_lock_checks_every_file(self):
        from workspace_paths import ROOT
        lock=json.loads(POSEIDON_LOCK.read_text())
        self.assertEqual(verify_sources(backend="poseidon_cpu")["gitlink"],lock["gitlink"])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in [*lock["sources"],"src/poseidon/tools/dacapo/poly-python-wheels.lock.json"]:
                p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
            verify_sources(root,backend="poseidon_cpu")
            p=root/next(iter(lock["sources"]));p.write_bytes(p.read_bytes()+b"\n")
            with self.assertRaises(ValueError):verify_sources(root,backend="poseidon_cpu")
    def test_explicit_w40_changes_binding_and_command(self):
        a=configuration("poseidon-cpu-eva-w40-v1");b=configuration("poseidon-cpu-eva-w45-v1")
        self.assertEqual(a["parameter_identity"],b["parameter_identity"])
        self.assertNotEqual(a["identity_sha256"],b["identity_sha256"])
        self.assertIn("--waterline=40",command(40));self.assertIn("--waterline=45",command())
        with self.assertRaises(ValueError):command(44)
        with self.assertRaises(ValueError):validate_configuration(dict(a,waterline=45),a["ckks_config_sha256"])
    def test_application_accepts_helpers_with_same_contract(self):
        from application_policy import prepare,BACKEND
        r,_=prepare(self.graph(),BACKEND,"numerical",dict(helper_profile=PROFILE))
        self.assertEqual(r["upstream_helpers"]["source_backend"],"poseidon_cpu")
        self.assertIn(PROFILE,capabilities()["helper_profiles"])
        with self.assertRaises(ValueError):prepare(self.graph(),BACKEND,"numerical",dict(helper_profile="unknown"))
    def test_adapter_drift_changes_application_identity(self):
        from application_backend import PoseidonBackend
        import poseidon_backend_identity
        with tempfile.TemporaryDirectory() as d,patch("hecate_python_env.ROOT",Path(d)),patch("poseidon_backend_identity.identity",return_value={"test_only":True}):
            folder=Path(d)/"scripts/baseline/upstream_adapters";folder.mkdir(parents=True)
            p=folder/"adapter.py";p.write_text("before")
            before=PoseidonBackend().identity()
            p.write_text("after")
            self.assertNotEqual(before,PoseidonBackend().identity())
if __name__=="__main__":unittest.main()
