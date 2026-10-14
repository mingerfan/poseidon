"""Control tests for evidence integrity and false qualification; no backend."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
import audit

class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="audit-control-", dir=Path(__file__).parent)
        self.root = Path(self.tmp.name).resolve()
        self.reader = audit.Reader((self.root,))
    def tearDown(self):
        self.tmp.cleanup()
    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value if isinstance(value, bytes) else audit.canonical(value))
        return p
    def comparison(self):
        return dict(passed=True,atol=1e-5,rtol=1e-4,
                    actual=[[0.0,2.00002],[-2.0,0.25]],
                    reference=[[0.0,2.0],[-2.0,0.25]],compared_values=4)
    def history(self):
        folder = self.root / "case"
        model = {"format":"fixture-model-v1","inputs":[{"name":"x","shape":[2]}]}
        request = {"request_id":"request","model":model}
        source = "def forward(x):\n    return x\n"
        payload = {"candidate":{"schema":1,"request_id":"request","hecate_source":source},"request":request}
        files = {"model.json":model,"request.json":request,
                 "attempt-00/trace-payload.json":payload,"rule-answer.py":source.encode(),
                 "attempt-00/candidate.py":source.encode(),
                 "attempt-00/response.txt":{"schema":1,"request_id":"request","hecate_source":source},
                 "attempt-00/output/bound/plan.json":{"format":"fixture-runtime-plan"}}
        hashes={}
        for name, data in files.items():
            p=self.write("case/"+name,data)
            hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        raw = dict(backend="poseidon_cpu",validation_level="numerical",status="passed",
                   llm_generation_validated=False,request_id="request",
                   frozen_hashes={k:hashes[k]for k in
                                  ["model.json","request.json","attempt-00/trace-payload.json","rule-answer.py"]},
                   source_hashes={"legacy.py":"legacy_hash"},backend_identity={"version":"legacy"},
                   compiler_configuration={"identity_sha256":"legacy_compiler"},
                   platform_identity={"machine":"aarch64"},
                   attempts=[dict(compiled=True,executed=True,numerically_correct=True,
                                  trace={"frontend":"real_Hecate"},
                                  execution={"encrypted_execution":True,"backend":"poseidon_cpu",
                                             "executor_secret_key":False},
                                  artifact_hashes={"bound/plan.json":hashes["attempt-00/output/bound/plan.json"]},
                                  comparison=self.comparison())])
        report=self.write("case/report.json",raw)
        row=dict(evidence=str(folder),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                 agent_generated=False,model_sha256=audit.digest(model),id="case",partition="fixture",
                 operator="add",input_shapes=[[2]],attributes=[],binding_origin="historical")
        return folder,raw,row
    def resave(self,raw,row):
        p=self.write("case/report.json",raw)
        row["report_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()

    def test_valid_history_keeps_legacy_identity_and_rechecks_each_value(self):
        folder,raw,row=self.history()
        result=audit.historical_case(self.reader,row,{"sdk_identity":"current"})
        self.assertEqual(result["historical_comparison"]["compared_values"],4)
        self.assertTrue(result["historical_comparison"]["elementwise_rechecked"])
        self.assertEqual(result["current_sdk_qualification"],"not_run")
        self.assertIsNone(result["recorded_stage_identities"])
        self.assertIsNone(result["application_task_id"])
        self.assertEqual(result["historical_artifact_hashes_verified"],1)
    def test_report_hash_drift_rejected(self):
        folder,raw,row=self.history()
        self.write("case/report.json",{**raw,"status":"failed"})
        with self.assertRaisesRegex(ValueError,"evidence_content_drift"):
            audit.historical_case(self.reader,row,{})
    def test_provider_origin_cannot_be_invented(self):
        folder,raw,row=self.history()
        raw["llm_generation_validated"]=True;self.resave(raw,row)
        with self.assertRaisesRegex(ValueError,"historical_backend_or_origin"):
            audit.historical_case(self.reader,row,{})
    def test_seal_backend_not_qualified_as_poseidon(self):
        folder,raw,row=self.history()
        raw["backend"]="seal";self.resave(raw,row)
        with self.assertRaisesRegex(ValueError,"historical_backend_or_origin"):
            audit.historical_case(self.reader,row,{})
    def test_mock_trace_not_qualified(self):
        folder,raw,row=self.history()
        raw["attempts"][0]["trace"]["frontend"]="mock";self.resave(raw,row)
        with self.assertRaisesRegex(ValueError,"historical_actual_stages"):
            audit.historical_case(self.reader,row,{})
    def test_artifact_tamper_rejected(self):
        folder,raw,row=self.history()
        self.write("case/attempt-00/output/bound/plan.json",{"changed":True})
        with self.assertRaisesRegex(ValueError,"evidence_content_drift"):
            audit.historical_case(self.reader,row,{})
    def test_candidate_tamper_rejected_without_changing_historical_response(self):
        folder,raw,row=self.history()
        self.write("case/attempt-00/candidate.py",b"def forward(x): return -x")
        with self.assertRaisesRegex(ValueError,"historical_candidate_provenance"):
            audit.historical_case(self.reader,row,{})
    def test_comparison_rejects_claimed_pass_above_tolerance(self):
        c=self.comparison();c["actual"][0][0]=0.1
        with self.assertRaisesRegex(ValueError,"historical_elementwise_comparison"):
            audit.comparison_audit(c)
    def test_comparison_rejects_relaxed_threshold(self):
        c=self.comparison();c["atol"]=1e-3
        with self.assertRaisesRegex(ValueError,"historical_numerical_tolerance"):
            audit.comparison_audit(c)
    def test_comparison_rejects_wrong_shape_same_flat_count(self):
        c=self.comparison();c["actual"]=[[0.0,2.00002,-2.0,0.25]]
        with self.assertRaisesRegex(ValueError,"historical_comparison_status_or_shape"):
            audit.comparison_audit(c)
    def test_comparison_rejects_boolean_numeric_data(self):
        c=self.comparison();c["actual"][0][0]=False
        with self.assertRaisesRegex(ValueError,"comparison_finite_value"):
            audit.comparison_audit(c)
    def test_comparison_rejects_nonfinite_data(self):
        c=self.comparison();c["actual"][0][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"comparison_finite_value"):
            audit.comparison_audit(c)
    def test_json_duplicate_field_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate_json_field"):
            audit.strict('{"a":1,"a":2}')
    def test_json_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError,"nonfinite_json"):
            audit.strict('{"a":NaN}')
    def test_secret_path_rejected(self):
        p=self.write(".env",b"not-a-credential")
        with self.assertRaisesRegex(ValueError,"credential_or_git_path"):
            self.reader.read(p)
    def test_symbolic_link_rejected(self):
        p=self.write("real.json",{})
        link=self.root/"link.json";link.symlink_to(p)
        with self.assertRaisesRegex(ValueError,"evidence_regular_file"):
            self.reader.read(link)
    def test_outside_root_rejected(self):
        with self.assertRaisesRegex(ValueError,"evidence_allowed_root"):
            self.reader.read(Path("/etc/hosts"))
    def test_current_status_cannot_be_promoted_by_resigning_index(self):
        value=dict(format=audit.FORMAT,files={},historical_cpu={"rows":[{"current_sdk_qualification":"passed"}]})
        value["binding"]=audit.digest(value)
        with self.assertRaisesRegex(ValueError,"historical_current_identity_not_promoted"):
            audit.verify_index(value,self.reader)
    def test_index_file_drift_rejected(self):
        p=self.write("evidence.json",{"old":True})
        value=dict(format=audit.FORMAT,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()},
                   historical_cpu={"rows":[]})
        value["binding"]=audit.digest(value)
        self.write("evidence.json",{"new":True})
        with self.assertRaisesRegex(ValueError,"evidence_content_drift"):
            audit.verify_index(value,self.reader)
    def test_static_registry_only_reads_names(self):
        raw=b'REGISTRY = MappingProxyType({"dot": Operator(frozenset(), "ref", "_rule")})'
        self.assertEqual(audit.registered_operators(raw),["dot"])

    def test_static_polynomial_registry_expansion(self):
        raw=b'REGISTRY = MappingProxyType({"dot": Op(), **{name: Op() for name in ("relu","log")}})'
        self.assertEqual(audit.registered_operators(raw),["dot","log","relu"])
    def test_dynamic_registry_iterable_not_executed(self):
        raw=b'REGISTRY = MappingProxyType({**{name: Op() for name in load_remote()}})'
        with self.assertRaises(ValueError):
            audit.registered_operators(raw)

    def current_index(self):
        value=dict(format=audit.FORMAT,files={},historical_cpu={"rows":[]},
                   current_snapshot={"not_a_new_qualification":True},
                   operators=[{"current":dict(registered=True,reference="not_reexecuted",
                                             compiled="not_run",encrypted="not_run",agent="not_run")}])
        value["binding"]=audit.digest(value)
        return value
    def test_operator_status_cannot_be_promoted_by_resigning(self):
        value=self.current_index();value["operators"][0]["current"]["encrypted"]="passed"
        value["binding"]=audit.digest({k:v for k,v in value.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"operator_current_identity_not_promoted"):
            audit.verify_index(value,self.reader)
    def test_snapshot_cannot_be_relabeled_as_new_qualification(self):
        value=self.current_index();value["current_snapshot"]["not_a_new_qualification"]=False
        value["binding"]=audit.digest({k:v for k,v in value.items()if k!="binding"})
        with self.assertRaisesRegex(ValueError,"source_snapshot_not_new_qualification"):
            audit.verify_index(value,self.reader)
    def test_test_result_bound_to_tested_auditor(self):
        test=self.write("test_audit.py",b"tests");tool=self.write("audit.py",b"old-auditor")
        log=self.write("controls-final-v2.log",b"Ran 26 tests OK")
        self.write("controls.json",dict(format="poseidon-qualification-index-controls-v1",
                 status="passed",tests=26,failed=0,errors=0,skipped=0,
                 test_source_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
                 audit_source_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
                 log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        self.assertEqual(audit.verify_controls(self.reader,self.root)["tests"],26)
        self.write("audit.py",b"new-auditor")
        with self.assertRaises(ValueError):
            audit.verify_controls(audit.Reader((self.root,)),self.root)
    def test_bound_request_model_matches_model_file(self):
        folder,raw,row=self.history()
        request=json.loads((folder/"request.json").read_text());request["model"]={"different":True}
        payload=json.loads((folder/"attempt-00/trace-payload.json").read_text());payload["request"]=request
        for name,data in [("request.json",request),("attempt-00/trace-payload.json",payload)]:
            path=self.write("case/"+name,data)
            raw["frozen_hashes"][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        self.resave(raw,row)
        with self.assertRaisesRegex(ValueError,"historical_model_request_binding"):
            audit.historical_case(self.reader,row,{})

if __name__=="__main__":
    unittest.main(verbosity=2)
