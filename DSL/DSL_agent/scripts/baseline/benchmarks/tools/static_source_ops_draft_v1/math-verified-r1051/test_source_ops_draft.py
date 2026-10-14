import copy,unittest
from benchmark_graph import digest
from source_ops_draft import (public_division,functional_max_pool,verify_plan,
                              SourcePlanError,DIVISION_CALLS,VERSION)
APPROX=dict(method="polynomial",basis="power",domain=[-2.,2.],coefficients=[0.,0.,1.])
class SourceDraftTests(unittest.TestCase):
    def rejected(self,code,fn,*args,**kwargs):
        with self.assertRaises(SourcePlanError) as cm:fn(*args,**kwargs)
        self.assertEqual(cm.exception.code,code)
        payload=cm.exception.feedback()
        self.assertEqual(payload["retry_policy"],"never_automatic")
        self.assertNotIn("traceback",payload)
        return payload
    def test_division_aliases_share_mathematical_graph(self):
        plans=[public_division([2,3],2.,call=c) for c in DIVISION_CALLS]
        self.assertEqual(len({digest(p["mathematical_graph"]) for p in plans}),1)
        self.assertEqual(len({p["source_request_sha256"] for p in plans}),len(DIVISION_CALLS))
    def test_static_broadcast_public_tensor(self):
        p=public_division([2,1,2,3],[[[[2.,-4.,8.]],[[.5,1.25,-2.5]]]])
        self.assertEqual(p["lowered_graph"]["inputs"][0]["shape"],[2,1,2,3])
        self.assertEqual(p["production_intake_integrated"],False)
    def test_rank32_and_legacy_rank_limit(self):
        p=public_division([1]*31+[2],[2.,4.])
        self.assertEqual(len(p["mathematical_graph"]["inputs"][0]["shape"]),32)
        self.rejected("source_shape_rejected",public_division,[1]*5,2.,resource_profile="legacy-v1")
    def test_input_and_output_capacity(self):
        self.rejected("source_shape_rejected",public_division,[4097],2.)
        self.rejected("division_broadcast_rejected",public_division,[4096,1],[[2.,3.]])
    def test_zero_scalar_and_array(self):
        for value in (0.,[[2.,0.]]):self.rejected("division_public_zero",public_division,[2,2],value)
    def test_finite_types_and_public_bounds(self):
        for value in (float("nan"),float("inf"),-float("inf")):
            self.rejected("division_public_nonfinite",public_division,[2],value)
        for value in (True,"not-public",[],[[1.],[2.,3.]]):
            self.rejected("division_public_type_rejected",public_division,[2],value)
        for value in (1e-9,2048.,10**1000):
            self.rejected("division_public_budget",public_division,[2],value)
    def test_broadcast_mismatch(self):
        self.rejected("division_broadcast_rejected",public_division,[2,3],[[2.,3.]])
    def test_rounding_and_out_rejected(self):
        self.rejected("division_rounding_unsupported",public_division,[2],2.,rounding_mode="trunc")
        self.rejected("source_out_mutation_unsupported",public_division,[2],2.,out=object())
    def test_unknown_call_feedback_does_not_echo_source(self):
        marker="private_source_secret_path"
        payload=self.rejected("source_call_unregistered",public_division,[2],2.,call=marker)
        self.assertNotIn(marker,str(payload))
        self.assertEqual(payload["location"],dict(operator="public_division",parameter="call"))
    def test_names_and_source_copy(self):
        denominator=[2.,4.];p=public_division([2],denominator,input_name="public_factor")
        denominator[0]=0.
        self.assertEqual(p["source_request"]["divisor"],[2.,4.])
        self.assertNotIn("public_factor",p["mathematical_graph"]["constants"])
        verify_plan(p,expected_source_request_sha256=p["source_request_sha256"])
    def test_resigned_model_tamper_rejected(self):
        p=public_division([2],2.);expected=p["source_request_sha256"];p=copy.deepcopy(p)
        key=next(iter(p["mathematical_graph"]["constants"]));p["mathematical_graph"]["constants"][key]=9.
        p["binding"]=digest({k:v for k,v in p.items() if k!="binding"})
        self.rejected("source_plan_integrity_rejected",verify_plan,p,expected_source_request_sha256=expected)
    def test_resigned_source_tamper_rejected(self):
        p=public_division([2],2.);expected=p["source_request_sha256"];p=copy.deepcopy(p)
        p["source_request"]["divisor"]=4.;p["source_request_sha256"]=digest(p["source_request"])
        p["binding"]=digest({k:v for k,v in p.items() if k!="binding"})
        self.rejected("source_plan_integrity_rejected",verify_plan,p,expected_source_request_sha256=expected)
    def test_pool_static_defaults_and_padding(self):
        p=functional_max_pool([1,1,5,5],2,padding=1,dilation=[2,1],stride=2,approximation=APPROX)
        self.assertEqual(p["source_request"]["padding"],[1,1])
        self.assertEqual(p["source_request"]["dilation"],[2,1])
        verify_plan(p,expected_source_request_sha256=p["source_request_sha256"])
    def test_pool_rank_and_boolean_parameters(self):
        self.rejected("pool_rank_rejected",functional_max_pool,[1,4],2,rank=1,approximation=APPROX)
        self.rejected("pool_rank_rejected",functional_max_pool,[1,1,4],2,rank=True,approximation=APPROX)
        self.rejected("pool_public_parameters_rejected",functional_max_pool,[1,1,4,4],True,approximation=APPROX)
        self.rejected("pool_public_parameters_rejected",functional_max_pool,[1,1,4,4],[2],approximation=APPROX)
    def test_pool_floor_values_only(self):
        self.rejected("pool_ceil_unsupported",functional_max_pool,[1,1,4,4],2,ceil_mode=True,approximation=APPROX)
        self.rejected("pool_indices_unsupported",functional_max_pool,[1,1,4,4],2,return_indices=True,approximation=APPROX)
    def test_pool_explicit_profile_required(self):
        self.rejected("pool_approximation_rejected",functional_max_pool,[1,1,4,4],2)
        self.rejected("pool_approximation_rejected",functional_max_pool,[1,1,4,4],2,approximation=dict(APPROX,coefficients=[True]))
    def test_pool_empty_window_and_excess_padding(self):
        self.rejected("source_graph_contract_rejected",functional_max_pool,[1,1,1,1],2,padding=1,dilation=2,approximation=APPROX)
        self.rejected("source_graph_contract_rejected",functional_max_pool,[1,1,4,4],2,padding=2,approximation=APPROX)
    def test_no_success_inference_from_math_plan(self):
        p=public_division([2],2.)
        self.assertEqual(p["format"],VERSION)
        self.assertTrue(p["no_dsl_answer"])
        for k in ("physical_layout_status","compiler_status","encrypted_status","agent_status"):
            self.assertEqual(p[k],"unverified")
if __name__=="__main__":unittest.main()
