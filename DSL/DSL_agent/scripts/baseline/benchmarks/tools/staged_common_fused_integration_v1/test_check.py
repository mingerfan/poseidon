"""Control-layer regressions; synthetic fixtures never count as FHE evidence."""
import copy, json, os, shutil, tempfile, unittest
from pathlib import Path
import check as guard

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="poseidon-source-integration-test-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.here=self.root/"integration"
        self.here.mkdir()
        self.plan=guard.read(guard.HERE/"plan.json")
        # All files are public source and developer control records. Credentials,
        # raw provider responses and runtime outputs are not in this fixture.
        self.copy_paths=set(self.plan["original_production_hashes"])
        for entry in self.plan["targets"].values():
            self.copy_paths.add(entry["source"])
            self.copy_paths.update(layer["source"] for layer in entry["overlay_history"])
        for entries in self.plan["dependencies"].values():self.copy_paths.update(entries)
        for item in self.plan["source_acceptance"].values():self.copy_paths.add(item["path"])
        for rel in self.copy_paths:
            src=guard.ROOT/rel;dst=self.root/rel
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        shutil.copytree(guard.HERE/"targets",self.here/"targets")
        shutil.copytree(guard.HERE/"recovery-originals",self.here/"recovery-originals")
    def check(self):
        return guard.validate(self.plan,self.root,self.here)
    def bind(self):
        self.plan["binding"]=guard.digest({k:v for k,v in self.plan.items() if k!="binding"})

class IntegrityTests(Fixture):
    def test_exact_combined_source_is_prepared_without_activation(self):
        state=self.check()
        self.assertEqual(state["targets"],10)
        self.assertEqual(state["combined_sources"],706)
        self.assertEqual(state["originals_retained"],8)
        self.assertFalse(state["production_applied"])
        self.assertEqual(state["backend_acceptance"],"not_run")

    def test_plan_changed_without_new_binding_rejected(self):
        self.plan["branch"]="other"
        with self.assertRaisesRegex(ValueError,"identity"):self.check()

    def test_modified_user_source_rejected(self):
        target=next(name for name,item in self.plan["targets"].items() if not item["new_file"])
        (self.root/"scripts/baseline"/target).write_text("# user edit\n")
        with self.assertRaisesRegex(ValueError,"Production freeze"):self.check()

    def test_unrelated_production_edit_also_rejected(self):
        p=next(name for name in self.plan["original_production_hashes"]
               if Path(name).name not in self.plan["targets"])
        (self.root/p).write_text("# unrelated edit\n")
        with self.assertRaisesRegex(ValueError,"Production freeze"):self.check()

    def test_staged_final_source_change_rejected(self):
        (self.here/"targets/pytorch_model_intake.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError,"source changed"):self.check()

    def test_preceding_overlay_change_rejected(self):
        item=next(i for i in self.plan["targets"].values() if len(i["overlay_history"])>1)
        (self.root/item["overlay_history"][0]["source"]).write_text("# changed\n")
        with self.assertRaisesRegex(ValueError,"Prerequisite layer"):self.check()

    def test_overlay_priority_cannot_be_rebound(self):
        item=next(i for i in self.plan["targets"].values() if len(i["overlay_history"])>1)
        item["overlay_history"].reverse();self.bind()
        with self.assertRaisesRegex(ValueError,"precedence"):self.check()

    def test_recovery_original_change_rejected(self):
        name=next(n for n,i in self.plan["targets"].items() if not i["new_file"])
        (self.here/"recovery-originals"/name).write_text("# broken recovery\n")
        with self.assertRaisesRegex(ValueError,"Recovery"):self.check()

    def test_new_module_collision_rejected(self):
        name=next(n for n,i in self.plan["targets"].items() if i["new_file"])
        (self.root/"scripts/baseline"/name).write_text("# user module\n")
        with self.assertRaisesRegex(ValueError,"collision"):self.check()

    def test_dangling_new_module_symlink_rejected(self):
        name=next(n for n,i in self.plan["targets"].items() if i["new_file"])
        (self.root/"scripts/baseline"/name).symlink_to(self.root/"missing")
        with self.assertRaisesRegex(ValueError,"collision"):self.check()

    def test_existing_source_symlink_rejected(self):
        name=next(n for n,i in self.plan["targets"].items() if not i["new_file"])
        p=self.root/"scripts/baseline"/name
        raw=p.read_bytes();p.unlink()
        copied=self.root/"user_source.py";copied.write_bytes(raw)
        p.symlink_to(copied)
        with self.assertRaisesRegex(ValueError,"Regular"):self.check()

    def test_outside_checkout_source_rejected(self):
        self.plan["targets"]["pytorch_model_intake.py"]["source"]="../external.py"
        self.bind()
        with self.assertRaisesRegex(ValueError,"Checkout-relative"):self.check()

    def test_prerequisite_plan_change_rejected(self):
        p=next(iter(self.plan["dependencies"]["plans"]))
        (self.root/p).write_text("{}")
        with self.assertRaisesRegex(ValueError,"Prerequisite plan"):self.check()

    def test_existing_process_helper_change_rejected(self):
        p=next(iter(self.plan["dependencies"]["control_helpers"]))
        (self.root/p).write_text("# changed guard\n")
        with self.assertRaisesRegex(ValueError,"control helper"):self.check()

    def test_source_acceptance_change_rejected(self):
        p=self.plan["source_acceptance"]["fused_matrix_source_draft_v1"]["path"]
        (self.root/p).write_text("{}")
        with self.assertRaisesRegex(ValueError,"Accepted evidence"):self.check()

    def test_unverified_backend_cannot_be_marked_passed(self):
        self.plan["backend_qualification"]["status"]="passed"
        self.bind()
        with self.assertRaisesRegex(ValueError,"remains unverified"):self.check()

    def test_combined_identity_cannot_drop_module(self):
        self.plan["inside_source_hashes"].pop("scripts/baseline/pytorch_fused_matrix_normalization.py")
        self.bind()
        with self.assertRaisesRegex(ValueError,"Combined source"):self.check()

    def test_original_permission_change_rejected(self):
        name=next(n for n,i in self.plan["targets"].items() if not i["new_file"])
        p=self.root/"scripts/baseline"/name
        p.chmod(self.plan["targets"][name]["original_mode"] ^ 0o020)
        with self.assertRaisesRegex(ValueError,"Original permission"):self.check()

    def test_recovery_permission_change_rejected(self):
        name=next(n for n,i in self.plan["targets"].items() if not i["new_file"])
        p=self.here/"recovery-originals"/name
        p.chmod(self.plan["targets"][name]["original_mode"] ^ 0o020)
        with self.assertRaisesRegex(ValueError,"Recovery permission"):self.check()

    def test_target_permission_change_rejected(self):
        p=self.here/"targets/pytorch_model_intake.py"
        p.chmod(self.plan["targets"][p.name]["target_mode"] ^ 0o020)
        with self.assertRaisesRegex(ValueError,"Target permission"):self.check()

    def test_new_module_permission_cannot_be_rebound(self):
        name=next(n for n,i in self.plan["targets"].items() if i["new_file"])
        self.plan["targets"][name]["target_mode"]=0o777
        (self.here/"targets"/name).chmod(0o777)
        self.bind()
        with self.assertRaisesRegex(ValueError,"New module permission"):self.check()

class BoundaryTests(unittest.TestCase):
    def test_real_guest_is_readonly_and_waits_for_backend(self):
        result=guard.check()
        self.assertFalse(result["production_applied"])
        self.assertFalse(result["activation_supported"])
        self.assertIn(result["status"],("awaiting_backend_qualification","waiting_for_process_exit"))
        self.assertEqual(result["compiler_calls"],0)
        self.assertEqual(result["encrypted_executions"],0)
        self.assertEqual(result["provider_calls"],0)
    def test_process_exit_required(self):
        result=guard.observe_boundary([1234,2345,1234])
        self.assertEqual(result["status"],"waiting_for_process_exit")
        self.assertEqual(result["active_processes"],[1234,2345])
        self.assertFalse(result["activation_supported"])
    def test_no_live_process_does_not_prove_backend(self):
        result=guard.observe_boundary([])
        self.assertEqual(result["status"],"awaiting_backend_qualification")
        self.assertFalse(result["activation_supported"])
    def test_invalid_process_observation_rejected(self):
        for value in (None,["1234"],[True],[0],[-1]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                guard.observe_boundary(value)
    def test_duplicate_control_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"record.json";p.write_text('{"binding":"a","binding":"b"}')
            with self.assertRaisesRegex(ValueError,"Duplicate"):guard.read(p)

if __name__=="__main__":unittest.main(verbosity=2)
