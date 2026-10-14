"""Finite MHA input-domain audit; no DSL generation, compilation or FHE."""
import copy,hashlib,json,sys,time,unittest
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import canonical,digest,require
from execution_input_domain import ApproximationDomainRejected,check_approximation_domains,declared_profiles,DOMAIN_FAILURE
from operator_graph import evaluate_reference
HERE=Path(__file__).resolve().parent
CAPTURE=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-layout-component-backend-v1/capture")
PRIOR=CAPTURE.parent.parent/"agent-mha-layout-component-backend-v2/report.json"
PROPOSAL=ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1317.json"
CASES=("self_post_linear_residual__declared_shallow_v1","linear_into_separate_cross_none__declared_shallow_v1","boolean_padding_residual__declared_shallow_v1")
MODELS={}; REFERENCES={}; OBSERVATIONS={}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    proposal=read(PROPOSAL);maps=[proposal["source_stage_identity"][s]["sources"] for s in ("generation","compilation","execution","validation")]
    inventory={}
    for m in maps:
        for name,expected in m.items():
            require(name not in inventory or inventory[name]==expected,"Stage source disagreement")
            inventory[name]=expected
            require(sha(BASE/name)==expected,"Frozen source changed")
    return inventory
def load():
    prior=read(PRIOR)
    rows={r["id"]:r for r in prior["cases"]}
    for case in CASES:
        d=CAPTURE/case;m=read(d/"original.json");r=read(d/"capture-reference.json")
        require(digest(m)==r["original_sha256"],"Captured graph mismatch")
        require(sha(d/"capture-reference.json")==rows[case]["capture_reference_sha256"],"Historical reference changed")
        require(r["reference_status"]=="passed" and r["approximation_accuracy_certified"] is False,"Reference scope")
        MODELS[case]=m;REFERENCES[case]=r
def zero(m):return {i["name"]:np.zeros(i["shape"],dtype=np.float64) for i in m["inputs"]}
def prefix(m,inputs,count):
    p=dict(m,nodes=m["nodes"][:count])
    p["outputs"]=[dict(name="probe",value=p["nodes"][-1]["outputs"][0])]
    return evaluate_reference(p,inputs,ideal=False)
def violation(m):
    last=max(i for i,n in enumerate(m["nodes"]) if declared_profiles(n))
    for exponent in range(1,13):
        sample={}
        for spec in m["inputs"]:
            shape=spec["shape"]
            values=np.linspace(.5,1.5,num=int(np.prod(shape))).reshape(shape)
            sample[spec["name"]]=np.asarray(values*(2.**exponent),dtype=np.float64)
        try:check_approximation_domains(m,[sample])
        except ApproximationDomainRejected as error:return sample,error,last
    raise AssertionError("No finite discriminating violation found")
class DomainAudit(unittest.TestCase):
    def test_three_actual_graphs_have_both_profiles(self):
        for case,m in MODELS.items():
            with self.subTest(case=case):
                prof=[(n["op"],declared_profiles(n)) for n in m["nodes"] if declared_profiles(n)]
                self.assertEqual([op for op,_ in prof],["exp","reciprocal"])
                self.assertTrue(all(set(p)=={"approximation"} for _,p in prof))
    def test_zero_and_signed_inputs_are_checked_without_certification(self):
        for case,m in MODELS.items():
            with self.subTest(case=case):
                z=zero(m)
                signed={k:np.linspace(-.05,.05,num=v.size).reshape(v.shape).astype(np.float64) for k,v in z.items()}
                result=check_approximation_domains(m,[z,signed])
                self.assertEqual(result["status"],"passed")
                self.assertEqual(result["profile_count"],2)
                self.assertEqual(result["samples"],2)
                self.assertEqual(result["source_sha256"],digest(m))
                self.assertNotIn("numerically_validated",result)
                self.assertNotIn("application_accuracy_certified",result)
                OBSERVATIONS.setdefault(case,{})["in_domain_samples_checked"]=2
    def test_three_actual_graphs_reject_and_locate_first_violation(self):
        for case,m in MODELS.items():
            with self.subTest(case=case):
                sample,error,last=violation(m);first=None
                for i in range(last+1):
                    try:prefix(m,sample,i+1)
                    except ValueError as exc:
                        if str(exc)!=DOMAIN_FAILURE:raise
                        first=i;break
                self.assertIsNotNone(first)
                d=error.public_failure
                self.assertEqual(d["location"]["index"],first)
                self.assertEqual(d["location"]["id"],m["nodes"][first]["id"])
                self.assertEqual(d["location"]["operator"],"exp")
                self.assertEqual(d["context"]["declared_domains"],declared_profiles(m["nodes"][first]))
                OBSERVATIONS.setdefault(case,{})["first_violation"]=d["location"]
    def test_later_sample_rejection_is_exact_and_redacted(self):
        for case,m in MODELS.items():
            with self.subTest(case=case):
                sample,_,_=violation(m)
                with self.assertRaises(ApproximationDomainRejected) as caught:
                    check_approximation_domains(m,[zero(m),sample])
                d=caught.exception.public_failure
                self.assertEqual(d["context"]["sample_index"],1)
                self.assertEqual(set(d),{"format","code","layer","location","context","retry_policy"})
                self.assertEqual(set(d["context"]),{"sample_index","declared_domains"})
                self.assertEqual(set(d["location"]),{"kind","index","id","operator"})
                for key in ("actual","expected","reference","values","inputs","coefficients","path"):
                    self.assertNotIn('"'+key+'":',json.dumps(d))
                snapshot=copy.deepcopy(d)
                d["context"]["declared_domains"]["approximation"][0]=-999.
                self.assertEqual(caught.exception.public_failure,snapshot)
    def test_sixteen_sample_boundary_and_snapshot_preservation(self):
        for case,m in MODELS.items():
            with self.subTest(case=case):
                before=canonical(m);sample=zero(m);arrays={k:v.copy() for k,v in sample.items()}
                self.assertEqual(check_approximation_domains(m,[sample]*16)["samples"],16)
                for k,v in arrays.items():np.testing.assert_array_equal(v,sample[k])
                self.assertEqual(canonical(m),before)
                with self.assertRaises(ValueError):check_approximation_domains(m,[sample]*17)
    def test_value_input_alone_does_not_change_attention_domains(self):
        for case in CASES[1:]:
            m=MODELS[case];sample=zero(m);sample["v"].fill(2.**20)
            self.assertEqual(check_approximation_domains(m,[sample])["status"],"passed")
            OBSERVATIONS.setdefault(case,{})["value_only_domain_check"]="passed_not_output_certification"
    def test_public_padding_mask_removes_masked_key_from_domain_check(self):
        case=CASES[2];m=MODELS[case];r=REFERENCES[case]
        entry=r["capture_metadata"]["mha_source_capture"]["entries"][0]
        mask=entry["source_request"]["options"]["key_padding_mask"]
        self.assertEqual(mask["dtype"],"bool")
        masked=np.asarray(mask["values"],dtype=bool).all(axis=0)
        self.assertTrue(masked.any());self.assertTrue((~masked).any())
        sample=zero(m);sample["q"].fill(1.)
        sample["k"][:,masked,:]=2.**20
        self.assertEqual(check_approximation_domains(m,[sample])["status"],"passed")
        sample["k"][:,~masked,:]=2.**20
        with self.assertRaises(ApproximationDomainRejected):
            check_approximation_domains(m,[sample])
        OBSERVATIONS.setdefault(case,{})["masked_vs_unmasked_key_discrimination"]="passed"
    def test_package_uses_exact_bound_original_graph(self):
        from program_execution import package_execution_metadata
        folder=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-live-delivery-recovery-v2/tasks/5384076bf9c04dc097254d1b1db30886/program")
        manifest=read(folder/"manifest.json");model=read(folder/"model.json")
        metadata=package_execution_metadata(folder,manifest,model)
        self.assertEqual(canonical(metadata["original_model"]),canonical(MODELS[CASES[2]]))
        self.assertEqual(metadata["output_structure_policy"],"poseidon-mha-ordered-source-v1")
        OBSERVATIONS["package_original_binding_verified"]=manifest["binding"]
def main():
    started=time.monotonic();before=sources();load()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DomainAudit))
    after=sources()
    require(after==before,"Frozen source changed during audit")
    report=dict(format="poseidon-mha-input-domain-audit-v1",status="passed" if result.wasSuccessful() else "failed",
      tests=dict(run=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)),
      scope="Three retained mathematical capture graphs; pre-encryption approximation-domain control only",
      cases=list(CASES),observations=OBSERVATIONS,source_files=before,
      capture_files={str((CAPTURE/c/n)):sha(CAPTURE/c/n) for c in CASES for n in ("original.json","capture-reference.json")},
      pending_proposal_sha256=sha(PROPOSAL),tool_sha256=sha(Path(__file__)),
      runtime=dict(python=sys.version.split()[0],numpy=np.__version__),elapsed_seconds=time.monotonic()-started,
      agent_generation=False,api_calls=0,compiler_calls=0,encrypted_execution=False,
      numerically_validated=False,original_application_accuracy_certified=False)
    report["binding"]=digest(report)
    with (HERE/"report.json").open("x") as f:json.dump(report,f,sort_keys=True,separators=(",",":"),allow_nan=False)
    print(json.dumps({k:report[k] for k in ("status","tests","elapsed_seconds","api_calls","compiler_calls","encrypted_execution","binding")}))
    return 0 if result.wasSuccessful() else 1
if __name__=="__main__":raise SystemExit(main())
