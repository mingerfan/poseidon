"""Joint receipt rejection tests; no source mutation."""
import copy,unittest
from guarded_apply_joint import validate_receipt,validate_preflight
class JointReceiptTests(unittest.TestCase):
    def setUp(self):
        self.expected={"tools/targets/source.py":"a"*64}
        common=dict(task_status="queued",generations=0,http_attempts=0,encrypted_execution=False,public_context_verified=True)
        rows=[dict(common,case=name,source_format="poseidon-trusted-pytorch-v12",generation_guidance="explicit-v11")
              for name in ("linear","norm","split_weight")]
        rows.extend(dict(common,case=name,source_api="snapshot_then_admit_checked_capture",capture_supervisor_executed=False)
                    for name in ("linear","norm","split_weight"))
        self.report=dict(status="passed",tests=251,failed=0,errors=0,skipped=0,provider_calls=0,
            compilations=0,encrypted_executions=0,production_applied=False,application_submissions=rows,
            runtime=dict(python=[3,10,14],torch="2.0.1",numpy="1.25.2"))
        self.sandbox=dict(exit_code=0,freeze_before={"source_hashes":695},freeze_after={"source_hashes":695},
            clearenv=True,unshare_all=True,production_applied=False,source_hashes=self.expected.copy())
    def test_complete_staged_receipt(self):validate_receipt(self.report,self.sandbox,self.expected)
    def test_untested_target_rejected(self):
        with self.assertRaisesRegex(ValueError,"Exact tested"):validate_receipt(self.report,self.sandbox,{"tools/targets/new.py":"b"*64})
    def test_changed_target_rejected(self):
        with self.assertRaisesRegex(ValueError,"Exact tested"):validate_receipt(self.report,self.sandbox,{"tools/targets/source.py":"b"*64})
    def test_failed_error_and_skip_rejected(self):
        for field in ("failed","errors","skipped"):
            report=copy.deepcopy(self.report);report[field]=1
            with self.assertRaises(ValueError):validate_receipt(report,self.sandbox,self.expected)
    def test_extra_api_compiler_or_ciphertext_claim_rejected(self):
        for field in ("provider_calls","compilations","encrypted_executions"):
            report=copy.deepcopy(self.report);report[field]=1
            with self.assertRaises(ValueError):validate_receipt(report,self.sandbox,self.expected)
    def test_unpinned_runtime_rejected(self):
        report=copy.deepcopy(self.report);report["runtime"]["torch"]="2.0.1+cpu"
        with self.assertRaises(ValueError):validate_receipt(report,self.sandbox,self.expected)
    def test_changed_production_freeze_rejected(self):
        self.sandbox["freeze_after"]["source_hashes"]=694
        with self.assertRaises(ValueError):validate_receipt(self.report,self.sandbox,self.expected)
    def test_missing_isolation_rejected(self):
        self.sandbox["clearenv"]=False
        with self.assertRaises(ValueError):validate_receipt(self.report,self.sandbox,self.expected)
    def test_synthetic_submit_cannot_claim_numeric_acceptance(self):
        self.report["application_submissions"][0]["encrypted_execution"]=True
        with self.assertRaises(ValueError):validate_receipt(self.report,self.sandbox,self.expected)
    def test_source_subset_cannot_replace_import_subset(self):
        self.report["application_submissions"]=self.report["application_submissions"][3:]*2
        with self.assertRaisesRegex(ValueError,"Independent import"):validate_receipt(self.report,self.sandbox,self.expected)
    def test_direct_admission_cannot_claim_full_capture_supervisor(self):
        self.report["application_submissions"][3]["capture_supervisor_executed"]=True
        with self.assertRaisesRegex(ValueError,"Source supervisor"):validate_receipt(self.report,self.sandbox,self.expected)
    def test_repeated_context_does_not_count_as_three(self):
        self.report["application_submissions"][1]["case"]="linear"
        with self.assertRaisesRegex(ValueError,"Independent import"):validate_receipt(self.report,self.sandbox,self.expected)
class PreflightReceiptTests(unittest.TestCase):
    def setUp(self):
        self.expected={"targets/source.py":"a"*64}
        self.rows=[dict(case=name,static_status="static_accepted_not_executed",compiled=False,encrypted_execution=False,
            generations=0,http_attempts=0,public_context_checked=True,source_format="poseidon-trusted-pytorch-v12",
            resource_profile="general-tensors-v1",generation_guidance="explicit-v11",public_weight_policy="descriptors-v1",
            request_id="a"*64,rule_source_sha256="b"*64) for name in ("linear","norm","split_weight")]
        self.report=dict(format="poseidon-joint-post-cutover-request-preflight-v1",status="passed",models=3,rows=self.rows,
            provider_calls=0,compilations=0,encrypted_executions=0,runtime=dict(python=[3,10,14],torch="2.0.1",numpy="1.25.2"))
        self.sandbox=dict(exit_code=0,test_scope="request_preflight",freeze_before={"sources":695},freeze_after={"sources":695},
            clearenv=True,unshare_all=True,source_hashes=self.expected)
    def test_all_three_static_program_bindings(self):validate_preflight(self.report,self.sandbox,self.expected)
    def test_static_result_cannot_claim_numerical_execution(self):
        for field in ("compiled","encrypted_execution"):
            report=copy.deepcopy(self.report);report["rows"][0][field]=True
            with self.assertRaisesRegex(ValueError,"No numerical"):validate_preflight(report,self.sandbox,self.expected)
    def test_invalid_profile_or_guidance_is_rejected(self):
        for key,value in (("resource_profile","general-v1"),("generation_guidance","explicit-v10")):
            report=copy.deepcopy(self.report);report["rows"][0][key]=value
            with self.assertRaisesRegex(ValueError,"Exact post-cutover"):validate_preflight(report,self.sandbox,self.expected)
    def test_missing_request_or_source_binding_is_rejected(self):
        for key in ("request_id","rule_source_sha256"):
            report=copy.deepcopy(self.report);del report["rows"][0][key]
            with self.assertRaisesRegex(ValueError,"Exact preflight program"):validate_preflight(report,self.sandbox,self.expected)
    def test_duplicate_context_is_not_diversity(self):
        report=copy.deepcopy(self.report);report["rows"][1]["case"]="linear"
        with self.assertRaisesRegex(ValueError,"Independent preflight"):validate_preflight(report,self.sandbox,self.expected)
    def test_other_worker_scope_is_not_preflight(self):
        self.sandbox["test_scope"]="full_component"
        with self.assertRaisesRegex(ValueError,"Preflight isolation"):validate_preflight(self.report,self.sandbox,self.expected)
if __name__=="__main__":unittest.main(verbosity=2)
