"""Proposal guards run without model generation, compilation, or FHE."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parents[2]))
import offline_source_r1405 as runner
from benchmark_graph import digest
class ScopeGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=runner.read(runner.PROPOSAL)
    def bound(self,mutator):
        plan=copy.deepcopy(self.plan);mutator(plan)
        plan["binding"]=digest({k:v for k,v in plan.items()if k!="binding"})
        return plan
    def actual(self):
        return dict(manifest_sha256=self.plan["sdk_identity"],backend=self.plan["backend_identity"])
    def verify(self,plan):
        with patch("application_identity.current_identity_manifest",return_value=self.actual()):
            return runner.verify(plan)
    def rejected(self,mutator,code):
        with self.assertRaisesRegex(ValueError,code):self.verify(self.bound(mutator))
    def test_current_plan_static_guards(self):
        self.assertEqual(self.verify(self.plan),self.actual())
    def test_original_digest_mismatch(self):
        p=copy.deepcopy(self.plan);p["binding"]="0"*64
        with self.assertRaisesRegex(ValueError,"control_proposal_binding"):self.verify(p)
    def test_duplicate_case_rejected(self):
        self.rejected(lambda p:p["cases"].append(copy.deepcopy(p["cases"][0])),"control_case_inventory")
    def test_call_inflation_rejected(self):
        self.rejected(lambda p:p["scope"].update(maximum_encrypted_calls=1000),"control_declared_calls")
    def test_native_limits_rejected(self):
        self.rejected(lambda p:p["scope"].update(native_concurrency=2),"control_native_limits")
    def test_wall_extension_rejected(self):
        self.rejected(lambda p:p["scope"].update(wall_seconds=2401),"control_bounded_scope")
    def test_space_extension_rejected(self):
        self.rejected(lambda p:p["scope"].update(new_bytes=1024**3),"control_bounded_scope")
    def test_provider_rejected(self):
        self.rejected(lambda p:p.update(provider_calls=1),"control_scope")
    def test_sdk_drift_rejected(self):
        self.rejected(lambda p:p.update(sdk_identity="0"*64),"control_current_sdk")
    def test_backend_drift_rejected(self):
        self.rejected(lambda p:p.update(backend_identity={}),"control_backend_identity")
    def test_rule_blocker_retained_not_executed(self):
        self.rejected(lambda p:p["cases"][0].update(preflight_status="blocked"),"control_preflight_inventory")
    def test_source_tamper_rejected(self):
        def change(p):
            files=p["cases"][0]["files"];files[next(iter(files))]="0"*64
        self.rejected(change,"control_fixture_changed")
    def test_control_tool_tamper_rejected(self):
        def change(p):
            tools=p["tools"];tools[next(iter(tools))]="0"*64
        self.rejected(change,"control_tool_changed")
    def test_missing_authorization_does_not_enter_nix(self):
        with tempfile.TemporaryDirectory()as t,patch.object(runner,"AUTH",Path(t)/"absent.json"),\
             patch.object(sys,"argv",["offline_source_r1405.py","--run-approved"]),patch.object(runner,"enter_nix")as nix:
            with self.assertRaisesRegex(ValueError,"control_explicit_authorization_required"):runner.main()
            nix.assert_not_called()
if __name__=="__main__":unittest.main(verbosity=2)
