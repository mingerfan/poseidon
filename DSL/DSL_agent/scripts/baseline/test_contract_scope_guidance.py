"""Effective native/public guidance, gate feedback and immutable legacy requests."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from benchmark_suite import Builder
from component_contract import prepare_task,request_options,reconstruct_request
from unified_graph_contract import validate_request,validate_candidate,static_repair_hint,explicit_generation_guidance
from unified_contract_scope_guidance import VERSION,base_version,ROUTE_MEAN
from candidate_check_diagnostic import CandidateCheckError
from application_diagnostics import attach

class ContractScopeTests(unittest.TestCase):
    def model(self):
        b=Builder([(4,)])
        model=b.finish(b.node("square",["input0"]))
        model.update(format="poseidon-model-graph-v2",resource_profile="medium-v1")
        return model
    def request(self,public=False,alias=False):
        opts=dict(generation_guidance=VERSION)
        if public:opts["construction_profile"]="hecate-unified-public-v1"
        if alias:opts["output_binding_policy"]="exact-values-v1"
        model=self.model()
        if alias:model["outputs"].append(dict(name="copy",value=model["outputs"][0]["value"]))
        return prepare_task(model,opts)
    def source(self,r,body):
        return r["generation_guidance"]["entry_header"]+"\n"+"".join("    "+line+"\n"for line in body.splitlines())
    def candidate(self,r,body):
        return dict(schema=1,request_id=r["request_id"],hecate_source=self.source(r,body))
    def x(self,r):return r["layout"]["inputs"][0]["dsl_name"]
    def test_regular_and_exact_output_requests_reconstruct(self):
        for alias in(False,True):
            r=self.request(alias=alias)
            self.assertEqual(base_version(r),"explicit-v15"if alias else"explicit-v14")
            self.assertEqual(reconstruct_request(r),r)
            self.assertEqual(validate_request(r),r["layout"]["input_slot_period"])
    def test_explicit_guidance_only_changes_request_identity_and_context(self):
        old=prepare_task(self.model(),dict(generation_guidance="explicit-v14"))
        new=self.request()
        self.assertNotEqual(old["request_id"],new["request_id"])
        for key in old:
            if key not in("generation_guidance","request_id"):self.assertEqual(new[key],old[key],key)
    def test_native_guidance_matches_actual_control_gate(self):
        r=self.request();x=self.x(r);body=f"a = {x}\nif 1 < 2:\n    a = {x} * 2\nreturn [a]"
        c=self.candidate(r,body)
        with self.assertRaises(CandidateCheckError):validate_candidate(c,r)
        hint=static_repair_hint(c["hecate_source"],r,"Unsupported native loop syntax")
        self.assertIn("does not admit if",hint)
        self.assertIn("static",hint)
    def test_public_profile_keeps_public_if(self):
        r=self.request(public=True);x=self.x(r)
        c=self.candidate(r,f"a = {x}\nif 1 < 2:\n    a = {x} * {x}\nreturn [a]")
        validate_candidate(c,r)
        self.assertNotIn("does not admit if",json.dumps(r["generation_guidance"]))
    def test_object_array_is_still_admitted_and_full_is_rejected(self):
        r=self.request();x=self.x(r)
        validate_candidate(self.candidate(r,f"a = np.array([{x}],dtype=object)\nreturn [a.item()*a.item()]"),r)
        with self.assertRaises(ValueError):
            validate_candidate(self.candidate(r,f"a = np.full((1,),{x},dtype=object)\nreturn [{x}*{x}]"),r)
    def test_plain_formal_is_not_a_rotation_index(self):
        r=self.request();x=self.x(r)
        src='@hc.func("c,p")\ndef rot(v,k):\n    return v.rotate(k)\n'+self.source(r,f"return [rot({x},1)]")
        with self.assertRaises(ValueError):
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=src),r)
        self.assertIn("not a public Python number",r["generation_guidance"]["helper_rule"])
    def test_application_static_feedback_uses_scoped_contract(self):
        r=self.request();x=self.x(r);c=self.candidate(r,f"a = {x}\nif 1 < 2:\n    a = {x} * 2\nreturn [a]")
        try:validate_candidate(c,r)
        except CandidateCheckError as e:
            from application_diagnostics import make
            d=make(e.reason,location=e.location,context=e.context)
        else:self.fail("Native if passed")
        result=attach(dict(status="failed",structured_diagnostic=d),candidate=c,request=r)
        text=result["public_feedback"]["diagnostic"]
        self.assertIn("does not admit if",text)
        self.assertNotIn("Ordering and equality require",text)
    def test_public_dtype_feedback_preserves_public_admission(self):
        r=self.request(public=True)
        from unified_contract_scope_guidance import type_repair_hint
        text=type_repair_hint(r,"dtype failure")
        self.assertIn("numeric",text)
        self.assertNotIn("not constructors",text)
    def test_numeric_feedback_explains_already_normalized_mean_without_answers(self):
        import json
        root=Path(__file__).resolve().parent/"benchmarks/tools/capacity_shallow_live_v1/prepared/suffix_three/request.json"
        old=json.loads(root.read_text())
        opts=request_options(old)
        opts.update(generation_guidance=VERSION,capacity_packing_version="capacity-public-tensors-v10")
        r=prepare_task(old["model"],opts)
        x=self.x(r);c=self.candidate(r,f"return [{x}]")
        result=attach(dict(status="failed",failure=dict(layer="numerical_comparison",diagnostic="SECRET /private/ref")),
                      candidate=c,request=r)
        text=result["public_feedback"]["diagnostic"]
        # aligned suffix has its own normalized gather instruction; use a generic route graph separately.
        self.assertNotIn("SECRET",text);self.assertNotIn("/private",text)
        self.assertNotIn("reference=[",text)
    def test_full_mean_route_metadata_is_selected_by_role(self):
        root=Path(__file__).resolve().parent/"benchmarks/tools/capacity_shallow_live_v1/prepared/rank_axis_8/request.json"
        old=json.loads(root.read_text());opts=request_options(old)
        opts.update(generation_guidance=VERSION,capacity_packing_version="capacity-public-tensors-v10")
        r=prepare_task(old["model"],opts)
        self.assertEqual(r["generation_guidance"]["bound_route_semantics"]["normalization"],ROUTE_MEAN)
        x=self.x(r);c=self.candidate(r,f"return [{x},{x}*2]")
        result=attach(dict(status="failed",failure=dict(layer="numerical_comparison",diagnostic="SECRET /private/ref")),
                      candidate=c,request=r)
        self.assertIn("without applying that reciprocal again",result["public_feedback"]["diagnostic"])
    def test_tampered_guidance_or_profile_does_not_validate(self):
        r=self.request()
        r["generation_guidance"]["helper_rule"]="p may control arbitrary Python"
        with self.assertRaises(ValueError):validate_request(r)
        r=self.request();r["construction_profile"]="invented"
        with self.assertRaises(ValueError):validate_request(r)
    def test_validation_adapter_stops_before_native_and_returns_same_scope(self):
        from validation_adapter import ValidationContext,CandidateValidationAdapter
        r=self.request();x=self.x(r);c=self.candidate(r,f"a = {x}\nif 1 < 2:\n    a = {x} * 2\nreturn [a]")
        with tempfile.TemporaryDirectory()as folder:
            def native(*a,**k):self.fail("Static refusal entered native execution")
            ctx=ValidationContext(Path(folder),r,dict(attempts=[]),{},dict(atol=1e-5,rtol=1e-4),45,
                                  Path(folder)/"keys",True,native,lambda:None)
            feedback=CandidateValidationAdapter(ctx).evaluate(json.dumps(c),0)
            self.assertEqual(feedback["layer"],"static_check")
            self.assertIn("does not admit if",feedback["diagnostic"])
            self.assertFalse(ctx.report["attempts"][0]["compiled"])
            self.assertFalse(ctx.report["attempts"][0]["executed"])
if __name__=="__main__":unittest.main(verbosity=2)
