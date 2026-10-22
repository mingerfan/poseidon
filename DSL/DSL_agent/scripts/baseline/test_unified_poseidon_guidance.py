"""Poseidon context selection, immutable historical hashes and privacy."""
import copy,json,unittest
from benchmark_suite import Builder
from benchmark_graph import digest
from compiler_configuration import configuration,PROFILE_SHA256
from unified_graph_contract import prepare,validate_request,static_repair_hint
from unified_poseidon_guidance import specification,repair_hint
from component_contract import reconstruct_request

class PoseidonGuidanceTests(unittest.TestCase):
    def graph(self,size=4,kind="mean"):
        b=Builder([(size,)])
        if kind=="mean":
            s=b.node("mean",["input0"],axes=[0],keepdims=True)
            v=b.node("multiply",["input0",s])
            y=b.node("multiply",[v,b.const([.5]*size)])
        elif kind=="polynomial":
            y=b.node("polynomial",["input0",b.const([.1,.2,.3])],basis="power")
        else:y=b.node("square",["input0"])
        return b.finish(y)
    def request(self,version="explicit-v10",model=None,public=False):
        c=configuration("poseidon-cpu-eva-w45-v1")
        return prepare(model or self.graph(),c["ckks_config_sha256"],c,generation_guidance=version,
                       construction_profile="hecate-unified-public-v1" if public else None)
    def test_v9_unchanged_and_v10_is_new_exact_binding(self):
        old=self.request("explicit-v9");new=self.request()
        self.assertNotEqual(old["request_id"],new["request_id"])
        for r in (old,new):
            self.assertEqual(reconstruct_request(r),r);validate_request(r)
        self.assertIn("actual SEAL",old["generation_guidance"]["value_and_scale_discipline"]["scale_semantics"])
        self.assertNotIn("SEAL",json.dumps(new["generation_guidance"]))
    def test_relevant_features_and_no_model_name_dispatch(self):
        r=self.request();context=specification(r)
        self.assertEqual(context["selection"],["packed_reduction","public_factor_placement","public_tensor_coordinates","scalar_broadcast"])
        renamed=copy.deepcopy(r);renamed["model"]["id"]="previously_unseen_model"
        self.assertEqual(specification(renamed),context)
        only=self.request(model=self.graph(kind="polynomial"))
        self.assertEqual(specification(only)["selection"],["polynomial_grouping"])
    def test_legacy_backend_cannot_select_poseidon_rules(self):
        with self.assertRaisesRegex(ValueError,"Poseidon CPU"):
            prepare(self.graph(),PROFILE_SHA256,generation_guidance="explicit-v10")
    def test_tampered_rehashed_guidance_is_rejected(self):
        r=self.request();r["generation_guidance"]["poseidon_expression_rules"]["rules"]["packed_reduction"]="Skip tails"
        r["request_id"]=digest({k:v for k,v in r.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Changed unified generation guidance"):validate_request(r)
    def test_public_request_excludes_answers_and_private_context(self):
        from deepseek_provider import public_request
        from unified_graph_lowering import lower
        r=self.request();payload=public_request(r);text=json.dumps(payload)
        self.assertNotIn(lower(r),text)
        for key in ("reference_outputs","test_inputs","hecate_source","PRIVATE"):
            self.assertNotIn(key,payload)
            self.assertNotIn(key,payload["generation_guidance"])
        self.assertIn("hecate_source",payload["response_schema"]["required"])
    def test_application_selects_relevant_new_requests_and_honors_explicit_version(self):
        from application_policy import prepare as app_prepare,BACKEND
        r,c=app_prepare(self.graph(),BACKEND,"numerical",{})
        self.assertEqual(c["guidance"],"explicit-v10")
        self.assertEqual(r["generation_guidance"]["version"],"explicit-v10")
        _,c=app_prepare(self.graph(kind="square"),BACKEND,"numerical",{})
        self.assertIsNone(c["guidance"])
        r,c=app_prepare(self.graph(),BACKEND,"numerical",{"generation_guidance":"explicit-v9"})
        self.assertEqual(c["guidance"],"explicit-v9")
        self.assertEqual(r["generation_guidance"]["version"],"explicit-v9")
    def test_repair_messages_are_bounded_fixed_and_reach_static_route(self):
        from unified_graph_lowering import lower
        r=self.request();source=lower(r)
        text=repair_hint(r,source,"/private SECRET",stage="compiler")
        self.assertIn("shallower",text);self.assertIn("slot zero",text)
        self.assertNotIn("SECRET",text);self.assertNotIn("/private",text)
        self.assertLessEqual(len(text),5000)
        self.assertEqual(repair_hint(r,"bad syntax!",stage="compiler"),"")
        self.assertIn("16 positional",static_repair_hint(source,r,"Native function parameter limit"))
    def test_cli_sandbox_and_public_profile_keep_old_limits(self):
        from run_candidate import parse_args
        from sandbox_policy import MODULES
        self.assertEqual(parse_args(["--prepare","--case","x","--unified-guidance","explicit-v10"]).unified_guidance,"explicit-v10")
        self.assertIn("unified_poseidon_guidance.py",MODULES)
        r=self.request(public=True)
        self.assertIn("undecorated",r["generation_guidance"]["expression_construction_rules"]["native_helpers"])
        self.assertEqual(r["rules"],self.request("explicit-v9",public=True)["rules"])

    def test_validation_adapter_routes_compiler_failure_to_poseidon_guidance(self):
        import tempfile
        from pathlib import Path
        from unified_graph_lowering import lower
        from validation_adapter import ValidationContext,CandidateValidationAdapter
        r=self.request();source=lower(r)
        with tempfile.TemporaryDirectory() as folder:
            report=dict(attempts=[])
            def native(payload,output,command,log):
                if command[0]=="/hecate-opt":
                    log.write_text("synthetic compiler admission failure")
                    return 1
                (output/"trace-evidence.json").write_text(json.dumps(dict(frontend="mock_for_unit_test")))
                log.write_text("mock tracing")
                return 0
            ctx=ValidationContext(Path(folder),r,report,{},dict(atol=1e-5,rtol=1e-4),45,
                                  Path(folder)/"keys",True,native,lambda:None)
            raw=json.dumps(dict(schema=1,request_id=r["request_id"],hecate_source=source))
            feedback=CandidateValidationAdapter(ctx).evaluate(raw,0)
            self.assertEqual(feedback["layer"],"compiler")
            self.assertIn("shallower",feedback["diagnostic"])
            self.assertIn("slot zero",feedback["diagnostic"])
            self.assertFalse(report["attempts"][0]["executed"])
            self.assertNotIn("SEAL",feedback["diagnostic"])

if __name__=="__main__":unittest.main()
