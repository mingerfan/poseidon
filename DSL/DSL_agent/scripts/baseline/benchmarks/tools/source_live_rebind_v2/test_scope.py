"""Pure stdlib rejection controls; no source capture, backend or Provider."""
import copy, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import scope

class Controls(unittest.TestCase):
    def sample(self):
        old=dict(branch="lhy-agent-dsl",head="h",backend_identity=dict(format="poseidon-backend-identity-v2"),
            cases=[dict(id=n,source_format="poseidon-trusted-pytorch-v12",source_manifest=n+"/manifest.json",
                source_files={n+"/model.py":"a"*64},request_id="b"*64,request_file=n+"/old.json",
                request_file_sha256="c"*64,rule_qualification_evidence="/tmp/"+n,
                rule_qualification_evidence_sha256="d"*64) for n in scope.CASES],
            disclosure={"fees_unknown":True},original_terminal_scope={"planned":76,"passed":56,"failed":20})
        cut=dict(binding="cut",production_sources={"current.py":"e"*64},protected_sources={"backend.cpp":"f"*64})
        cases=[]
        for c in old["cases"]:
            cases.append(dict(id=c["id"],source_format=c["source_format"],source_manifest=c["source_manifest"],
                source_files=c["source_files"],free_generation=True,directed_construction=False,
                historical=dict(request_id=c["request_id"],request_file=str(scope.ROOT/c["request_file"]),
                    request_file_sha256=c["request_file_sha256"],qualification=c["rule_qualification_evidence"],
                    qualification_sha256=c["rule_qualification_evidence_sha256"],current_identity=False),
                request_file=str(scope.CASE_ROOTS[c["id"]]/"request.json"),
                prepared={"task_path":str(scope.CASE_ROOTS[c["id"]]/"tasks"/("a"*32))}))
        proposal=dict(format=scope.FORMAT,approved=False,approval_reason="pending",source_root=str(scope.ROOT),
            branch=old["branch"],head=old["head"],historical=dict(path=str(scope.OLD),sha256=scope.OLD_SHA,
                binding=scope.OLD_BINDING,validation_identity_current=False),
            cutover=dict(path=str(scope.CUTOVER),sha256=scope.CUTOVER_SHA,binding="cut"),
            source_hashes=cut["production_sources"],protected_sources=cut["protected_sources"],
            provider=scope.PROVIDER,budget=scope.BUDGET,options=scope.OPTIONS,resource_profile=scope.PROFILE,
            cases=cases,preparation=dict(output=str(scope.PREPARATION),scope=scope.PREP_SCOPE,current_identity_not_qualification=True),
            tool_hashes=scope.tools(),application_backend_identity=dict(backend="poseidon_cpu",
                tools=old["backend_identity"],implementation_digest="g"*64),backend_identity=old["backend_identity"],
            application_stage_identity={"format":"poseidon-application-identities-v1"},
            disclosure=old["disclosure"],original_terminal_scope=old["original_terminal_scope"],
            validation=dict(backend="poseidon_cpu",level="numerical",new_agent_generation=True,
                no_compiler_or_runtime_changes=True,required=["source_capture","numerical_qualification",
                "new_input_execution","package_cache_reuse"],tolerance=dict(atol=1e-5,rtol=1e-4)))
        return scope.rebound(copy.deepcopy(proposal)),copy.deepcopy(old),copy.deepcopy(cut)
    def mutate(self,change,code):
        p,old,cut=self.sample();change(p)
        with self.assertRaisesRegex(ValueError,code):scope.validate(scope.rebound(p),old,cut)
    def test_exact_fixed_plan(self):
        p,old,cut=self.sample();self.assertIs(scope.validate(p,old,cut),p)
    def test_case_addition_rejected(self):self.mutate(lambda p:p["cases"].append(p["cases"][0]),"reviewed_cases")
    def test_case_reorder_rejected(self):self.mutate(lambda p:p["cases"].reverse(),"reviewed_cases")
    def test_case_replacement_rejected(self):self.mutate(lambda p:p["cases"][0].update(id="common_alpha"),"reviewed_cases")
    def test_provider_change_rejected(self):self.mutate(lambda p:p["provider"].update(model="other"),"reviewed_provider_or_budget")
    def test_token_increase_rejected(self):self.mutate(lambda p:p["provider"].update(max_tokens=384001),"reviewed_provider_or_budget")
    def test_time_extension_rejected(self):self.mutate(lambda p:p["budget"].update(batch_wall_seconds=43201),"reviewed_provider_or_budget")
    def test_bool_budget_rejected(self):self.mutate(lambda p:p["budget"].update(api_concurrency=True),"reviewed_provider_or_budget")
    def test_options_change_rejected(self):self.mutate(lambda p:p["options"].update(chunk_period=8),"reviewed_preparation_options")
    def test_fixture_replacement_rejected(self):self.mutate(lambda p:p["cases"][0]["source_files"].update(x="0"*64),"reviewed_case_source")
    def test_source_omission_rejected(self):self.mutate(lambda p:p["source_hashes"].clear(),"current_source_namespace")
    def test_protected_change_rejected(self):self.mutate(lambda p:p["protected_sources"].clear(),"current_source_namespace")
    def test_historical_identity_cannot_be_current(self):self.mutate(lambda p:p["historical"].update(validation_identity_current=True),"historical_separate")
    def test_low_level_identity_cannot_replace_application(self):self.mutate(lambda p:p.update(application_backend_identity=p["backend_identity"]),"backend_identity_types")
    def test_sdk_change_rejected(self):self.mutate(lambda p:p["backend_identity"].update(format="other"),"backend_identity_types")
    def test_preparation_is_not_qualification(self):self.mutate(lambda p:p["preparation"].update(current_identity_not_qualification=False),"preparation_scope")
    def test_tool_replacement_rejected(self):self.mutate(lambda p:p["tool_hashes"].clear(),"frozen_tools")
    def test_approval_cannot_be_embedded(self):self.mutate(lambda p:p.update(approved=True),"unapproved_proposal")
    def receipt(self,p):
        return dict(format="poseidon-source-live-rebound-approval-v2",approved=True,proposal_binding=p["binding"],
            proposal_sha256="a"*64,tool_hashes=p["tool_hashes"],authorization="human_explicit_case_budget_approval")
    def test_proposal_cannot_authorize(self):
        p,_,_=self.sample()
        with self.assertRaisesRegex(ValueError,"approval_record_fields"):scope.authority(p,p,"a"*64)
    def test_old_receipt_format_rejected(self):
        p,_,_=self.sample();r=self.receipt(p);r["format"]="poseidon-user-campaign-approval-v1"
        with self.assertRaisesRegex(ValueError,"approval_required"):scope.authority(r,p,"a"*64)
    def test_receipt_tools_changed_rejected(self):
        p,_,_=self.sample();r=self.receipt(p);r["tool_hashes"]={}
        with self.assertRaisesRegex(ValueError,"approval_required"):scope.authority(r,p,"a"*64)
    def test_receipt_hash_changed_rejected(self):
        p,_,_=self.sample();r=self.receipt(p);r["proposal_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"approval_required"):scope.authority(r,p,"a"*64)
    def test_no_receipt_rejected_before_live_import(self):
        import run
        with mock.patch.object(run,"verify",side_effect=AssertionError("must not prepare")):
            with self.assertRaisesRegex(ValueError,"approval_record_required"):run.execute(scope.RUN,None)
        self.assertNotIn("deepseek_provider",sys.modules);self.assertNotIn("agent_credentials",sys.modules)
    def test_hidden_case_reserves_once(self):
        import run
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            folder=Path(d);(folder/"case-boundary.json").write_bytes(scope.canonical({"binding":"c"}));run.reserve_case(folder,"linear",{"binding":"b"})
            with self.assertRaises(FileExistsError):run.reserve_case(folder,"linear",{"binding":"b"})
    def test_ledger_unknown_outcome_stops(self):
        s=dict(generations=1,http_attempts=1,ledger=[dict(event="http_reserved",index=0)])
        v=scope.terminal([dict(case="linear",status="passed")],[s])
        self.assertEqual(v["status"],"stopped");self.assertFalse(v["calls_resolved"])
    def test_duplicate_result_case_rejected(self):
        with self.assertRaisesRegex(ValueError,"result_case_scope"):
            scope.terminal([dict(case="linear",status="passed")]*3,[])
    def test_duplicate_json_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate_json_field"):scope.decode('{"x":1,"x":2}')
    def test_nonfinite_json_rejected(self):
        with self.assertRaisesRegex(ValueError,"nonfinite_json"):scope.decode('{"x":NaN}')
    def test_import_guard_blocks_paid_and_backend(self):
        import prepare
        guard=prepare.OfflineImports()
        for module in ("agent_credentials","deepseek_provider","component_backend","hecate"):
            with self.assertRaisesRegex(RuntimeError,"offline_preparation_import"):guard.find_spec(module)
    def test_owned_lock_denies_other_process_and_releases_after_exception(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            path=Path(d)/"owned.lock"
            code="import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            with self.assertRaisesRegex(RuntimeError,"owned body"):
                with scope.owned_lock(path):
                    child=subprocess.run([sys.executable,"-B","-c",code,str(path)],capture_output=True,timeout=5)
                    self.assertNotEqual(child.returncode,0)
                    self.assertIn(b"BlockingIOError",child.stderr)
                    raise RuntimeError("owned body")
            child=subprocess.run([sys.executable,"-B","-c",code,str(path)],capture_output=True,timeout=5)
            self.assertEqual(child.returncode,0,child.stderr)
    def test_owned_lock_remains_through_child_lifetime(self):
        import subprocess
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            path=Path(d)/"owned.lock"
            code="import fcntl,sys;f=open(sys.argv[1],'a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            with scope.owned_lock(path):
                for _ in range(2):
                    child=subprocess.run([sys.executable,"-B","-c",code,str(path)],capture_output=True,timeout=5)
                    self.assertNotEqual(child.returncode,0)
            self.assertEqual(subprocess.run([sys.executable,"-B","-c",code,str(path)],capture_output=True,timeout=5).returncode,0)
    def test_complete_actual_prepared_requests_and_reference_closure(self):
        initial=scope.read(scope.PREPARATION/"proposal-initial.json")
        for c in initial["cases"]:
            q=scope.validate_case(c,scope.Reader(),initial["application_stage_identity"])
            self.assertEqual(q["request_id"],c["historical"]["request_id"])
            self.assertEqual(c["request_differences"],[])
    def test_same_request_id_does_not_allow_changed_layout(self):
        initial=scope.read(scope.PREPARATION/"proposal-initial.json");c=initial["cases"][0]
        class ChangedReader(scope.Reader):
            def json(self,path):
                value=super().json(path)
                if Path(path)==Path(c["request_file"]):
                    value=copy.deepcopy(value);value["layout"]["output_ciphertexts"]+=1
                return value
        with self.assertRaisesRegex(ValueError,"complete_request_binding"):
            scope.validate_case(c,ChangedReader(),initial["application_stage_identity"])
    def test_metadata_only_reference_pass_is_rejected(self):
        initial=scope.read(scope.PREPARATION/"proposal-initial.json");c=initial["cases"][0]
        report=Path(c["prepared"]["task_path"])/"model-import/reference-report.json"
        class ChangedReader(scope.Reader):
            def json(self,path):
                value=super().json(path)
                if Path(path)==report:
                    value=copy.deepcopy(value);value["finite_reference"]["status"]="failed"
                return value
        with self.assertRaisesRegex(ValueError,"reference_capture_cross_binding"):
            scope.validate_case(c,ChangedReader(),initial["application_stage_identity"])
    def test_final_reader_rejects_early_file_changed_during_later_read(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            first=Path(d)/"first";second=Path(d)/"second"
            first.write_bytes(b"first");second.write_bytes(b"second")
            with mock.patch.object(scope,"regular",side_effect=lambda p:Path(p)):
                reader=scope.Reader();reader.hash(first);reader.hash(second)
                first.write_bytes(b"outside edit")
                with self.assertRaisesRegex(ValueError,"file_changed_during_read"):reader.finish()
    def test_self_rebound_initial_cannot_replace_original_preparation(self):
        import finalize
        original=finalize.sha
        with mock.patch.object(finalize,"sha",side_effect=lambda p:"0"*64 if Path(p)==finalize.INITIAL else original(p)):
            with self.assertRaisesRegex(ValueError,"fixed_initial_preparation"):finalize.candidate()
    def test_origin_freeze_sha_is_fixed_not_read_from_proposal(self):
        value=scope.read(scope.PREPARATION/"proposal-initial.json")
        # Build the authentic origin entries without invoking SDK or final tools.
        origins={}
        for name in scope.CASES:
            attempt=scope.CASE_ROOTS[name].parent
            frozen=scope.read(attempt/"freeze.json")
            snapshots={}
            for rel,want in frozen["tool_hashes"].items():
                original=scope.ROOT/rel
                path=(attempt/"tool-snapshot"/original.name) if original.parent==scope.HERE else original
                snapshots[str(path)]=want
            origins[name]=dict(type="reuse_completed_capture" if name=="linear" else "new_capture",
                case_root=str(scope.CASE_ROOTS[name]),attempt_root=str(attempt),freeze_sha256=scope.sha(attempt/"freeze.json"),
                producer_tool_hashes=frozen["tool_hashes"],tool_snapshot_hashes=snapshots)
        value["preparation"]["producer_tool_hashes"]=origins["norm"]["producer_tool_hashes"]
        value["preparation"]["capture_origins"]=origins
        scope.verify_origins(value,scope.Reader())
        value["preparation"]["capture_origins"]["linear"]["freeze_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"capture_origin_freeze"):scope.verify_origins(value,scope.Reader())
    def test_missing_current_plan_fails_closed(self):
        with mock.patch.object(scope,"PROPOSAL",scope.HERE/"does-not-exist.json"):
            with self.assertRaises((ValueError,FileNotFoundError)):scope.verify()
if __name__=="__main__":unittest.main()
