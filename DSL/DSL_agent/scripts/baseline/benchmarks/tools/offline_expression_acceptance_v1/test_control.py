"""Control tests: synthetic approvals never authorize a real batch."""
import ast,copy,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import control
import run
P=control.read(control.PROPOSAL)
HASHES=control.local_driver_hashes()
def receipt():
    return dict(format="poseidon-manual-expression-offline-approval-v1",approved=True,
      proposal_binding=P["binding"],proposal_sha256=control.sha(control.PROPOSAL),
      scope=dict(control.SCOPE),driver_hashes=dict(HASHES),human_message="Synthetic unit-test fixture")
def check(value):
    return control.check_approval(P,value,control.sha(control.PROPOSAL),HASHES)
class ControlTests(unittest.TestCase):
    def test_approval_is_missing_and_no_authoritative_receipt_is_created(self):
        self.assertFalse(control.APPROVAL.exists())
        with self.assertRaisesRegex(ValueError,"Pending r1286"):control.approved_plan()
        self.assertFalse(control.APPROVAL.exists());self.assertFalse(control.RUN.exists())
        self.assertNotIn("hecate_python_env",sys.modules)
        self.assertNotIn("component_backend",sys.modules)

    def test_synthetic_original_scope_validates_as_data_only(self):
        self.assertEqual(check(receipt())["scope"],control.SCOPE)
        self.assertEqual(check(receipt())["paid_calls"],0)
        self.assertFalse(control.APPROVAL.exists())

    def test_false_integer_and_missing_approval_are_rejected(self):
        for flag in (False,1,0,None,"true"):
            with self.subTest(flag=flag):
                value=receipt();value["approved"]=flag
                with self.assertRaises(ValueError):check(value)
        value=receipt();value.pop("approved")
        with self.assertRaises(ValueError):check(value)

    def test_other_proposal_and_changed_proposal_bytes_are_rejected(self):
        for field in ("proposal_binding","proposal_sha256"):
            value=receipt();value[field]="f"*64
            with self.subTest(field=field):
                with self.assertRaises(ValueError):check(value)

    def test_changed_proposal_content_cannot_borrow_original_binding(self):
        value=copy.deepcopy(P);value["model_sha256"]="f"*64
        with self.assertRaises(ValueError):
            control.check_approval(value,receipt(),control.sha(control.PROPOSAL),HASHES)

    def test_every_scope_field_is_exact_and_typed(self):
        for name,old in control.SCOPE.items():
            with self.subTest(field=name):
                value=receipt()
                value["scope"][name]=not old if type(old) is bool else old+1
                with self.assertRaises(ValueError):check(value)
        for name in ("compiler_attempts_max","encrypted_phases_max","provider_calls"):
            value=receipt();value["scope"][name]=True if control.SCOPE[name]==1 else False
            with self.subTest(field=name):
                with self.assertRaises(ValueError):check(value)

    def test_missing_extra_or_borrowed_scope_fields_are_rejected(self):
        for replacement in ({},dict(control.SCOPE,live=True)):
            value=receipt();value["scope"]=replacement
            with self.assertRaises(ValueError):check(value)

    def test_control_source_change_and_empty_inventory_are_rejected(self):
        value=receipt();value["driver_hashes"]["run.py"]="f"*64
        with self.assertRaises(ValueError):check(value)
        value=receipt();value["driver_hashes"]={}
        with self.assertRaises(ValueError):
            control.check_approval(P,value,control.sha(control.PROPOSAL),{})

    def test_other_requests_remain_explicitly_excluded(self):
        value=copy.deepcopy(P);value["exclusions"]=[]
        with self.assertRaises(ValueError):
            control.check_approval(value,receipt(),control.sha(control.PROPOSAL),HASHES)
        self.assertIn("paid_r1232",P["exclusions"])
        self.assertIn("fused_matrix_V14_backend_qualification",P["exclusions"])

    def test_original_live_sources_candidate_request_and_math_bindings_match(self):
        request=control.check_sources(P)
        self.assertEqual(request["request_id"],P["request_id"])
        self.assertEqual(request["compiler_configuration"]["backend"],"poseidon_cpu")
        self.assertEqual(len(P["source_hashes"]),704)
        self.assertEqual(len(P["protected_sources"]),654)
        self.assertFalse(control.APPROVAL.exists())

    def test_fresh_result_rejects_foreign_paths(self):
        for path in (Path("/tmp"),control.RUN.with_name("other"),Path("relative")):
            with self.subTest(path=str(path)):
                with self.assertRaises(ValueError):control.fresh_result(path)
        control.fresh_result()
        self.assertFalse(control.RUN.exists())

    def test_exclusive_json_creation_preserves_a_prior_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"attempt.json"
            control.write(path,{"original":True});saved=path.read_bytes()
            with self.assertRaises(FileExistsError):control.write(path,{"original":False})
            self.assertEqual(path.read_bytes(),saved)

    def test_duplicate_and_nonfinite_control_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"control.json"
            for raw in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
                path.write_text(raw)
                with self.subTest(raw=raw):
                    with self.assertRaises(ValueError):control.read(path)

    def test_cli_run_denies_before_nix_backend_lock_or_output(self):
        result=subprocess.run([sys.executable,"-B",str(control.HERE/"run.py"),"--run"],
            stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,2)
        value=json.loads(result.stderr)
        self.assertEqual(value["status"],"control_rejected")
        self.assertIn("Pending r1286",value["diagnostic"])
        self.assertFalse(control.RUN.exists());self.assertFalse(control.APPROVAL.exists())
        self.assertEqual(value["provider_calls"],0)

    def test_cli_plan_is_readonly_and_does_not_promote_acceptance(self):
        result=subprocess.run([sys.executable,"-B",str(control.HERE/"run.py"),"--plan"],
            stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0)
        value=json.loads(result.stdout)
        self.assertEqual(value["status"],"approval_required")
        for name in ("compiler_calls","encrypted_executions","provider_calls"):self.assertEqual(value[name],0)
        self.assertFalse(value["production_applied"]);self.assertFalse(control.RUN.exists())

    def test_each_internal_cli_stage_also_requires_human_approval(self):
        for action in ("--nix-worker","--namespace","--inside","--qualify","--mount-layer"):
            args=[sys.executable,"-B",str(control.HERE/"run.py"),action,str(control.RUN)]
            if action=="--mount-layer":args.append("0")
            result=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=10)
            with self.subTest(action=action):
                self.assertEqual(result.returncode,2)
                self.assertIn("Pending r1286",result.stderr)
        self.assertFalse(control.RUN.exists())

    def test_dispatch_has_no_approval_writer_or_budget_extension(self):
        source=(control.HERE/"run.py").read_text()
        tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        qualify=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="qualify"]
        self.assertEqual(len(qualify),1)
        self.assertEqual([kw.value.value for kw in qualify[0].keywords if kw.arg=="validation_level"],["numerical"])
        self.assertNotIn("DEEPSEEK_API_KEY",run.env())
        self.assertEqual(control.SCOPE["provider_calls"],0)
        self.assertFalse(control.SCOPE["new_input_execution"]);self.assertFalse(control.SCOPE["cache_execution"])

    def test_backend_identity_corruption_is_rejected(self):
        original=control.expected_backend(P)
        control.check_backend(original,original)
        for key in ("compiler_sha256","frontend_sha256","executor_sha256","poseidon_library_sha256",
                    "operator_spec_sha256","compiler_profile_sha256","runtime_dacapo_tree_sha256","api_tree_sha256"):
            bad=copy.deepcopy(original);bad[key]="f"*64
            with self.subTest(field=key):
                with self.assertRaises(ValueError):control.check_backend(bad,original)
        bad=copy.deepcopy(original);bad["build_configuration"]["cmake_cache_sha256"]["compiler"]="f"*64
        with self.assertRaises(ValueError):control.check_backend(bad,original)
        bad=copy.deepcopy(original);bad["platform"]["configured_platform"]="x86_64-linux"
        with self.assertRaises(ValueError):control.check_backend(bad,original)

    def test_actual_current_sdk_bytes_match_original_native_identity(self):
        # Data-only identity reader in the system interpreter; no SDK or Nix invocation.
        code="import sys,json;sys.path[:0]=["+repr(str(control.HERE))+","+repr(str(control.BASE))+"];"+\
             "import control;from poseidon_backend_identity import identity;"+\
             "actual=identity();control.check_backend(actual,control.expected_backend(control.read(control.PROPOSAL)));"+\
             "print(json.dumps(dict(status='matched',identity_sha256=control.digest(actual))))"
        variables=dict(PATH="/usr/bin:/bin",POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work")
        result=subprocess.run([sys.executable,"-B","-c",code],stdin=subprocess.DEVNULL,
                              capture_output=True,text=True,timeout=10,env=variables)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"],"matched")
        self.assertFalse(control.RUN.exists());self.assertFalse(control.APPROVAL.exists())

if __name__=="__main__":unittest.main(verbosity=2)
