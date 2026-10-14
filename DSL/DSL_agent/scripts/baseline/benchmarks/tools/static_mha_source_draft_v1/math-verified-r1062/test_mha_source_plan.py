import copy,unittest
import torch
from benchmark_graph import digest
from mha_source_plan import plan,verify,snapshot_module,public_mask,MhaError
from fixtures import CASES,APPROX,capture,module

class MhaSourceTests(unittest.TestCase):
    def reject(self,code,fn,*args,**kwargs):
        with self.assertRaises(MhaError) as cm:fn(*args,**kwargs)
        self.assertEqual(cm.exception.code,code)
        payload=cm.exception.feedback();self.assertEqual(payload["retry_policy"],"never_automatic")
        self.assertNotIn("private_source",str(payload))
    def request(self,index=0):
        return copy.deepcopy(capture(CASES[index],index)[2]["source_request"])
    def test_all_static_contexts_are_mapped(self):
        for index,case in enumerate(CASES):
            with self.subTest(case=case["id"]):
                fixture,graph,value=capture(case,index)
                self.assertFalse(value["production_intake_integrated"])
                self.assertEqual(value["output_schema"]["items"][1]["kind"],"tensor" if case.get("need",True) else "none")
    def test_self_alias_is_one_logical_input(self):
        value=capture(CASES[0])[2]
        self.assertEqual(len(value["mathematical_graph"]["inputs"]),1)
        self.assertEqual(len({x["name"] for x in value["source_request"]["operands"].values()}),1)
    def test_separate_projection_weights(self):
        value=capture(CASES[3],3)[2]
        self.assertFalse(value["source_request"]["module"]["config"]["same_projection"])
        self.assertIsNone(value["source_request"]["module"]["parameters"]["in_proj_weight"])
    def test_weight_output_structure_is_explicit(self):
        a=capture(CASES[1],1)[2];b=capture(CASES[2],2)[2]
        self.assertEqual(len(a["mathematical_graph"]["outputs"]),2)
        self.assertEqual(len(b["mathematical_graph"]["outputs"]),1)
        self.assertEqual(b["output_schema"]["items"][1],{"kind":"none"})
    def test_training_rejected(self):
        m=module(CASES[0],0).train()
        self.reject("mha_module_rejected",snapshot_module,m)
    def test_module_subclass_rejected(self):
        class Custom(torch.nn.MultiheadAttention):pass
        self.reject("mha_module_rejected",snapshot_module,Custom(2,1,dtype=torch.float64).eval())
    def test_float32_state_rejected(self):
        self.reject("mha_state_rejected",snapshot_module,module(CASES[0],0).float())
    def test_projection_shape_rejected(self):
        request=self.request();request["module"]["parameters"]["in_proj_weight"]=[[0.]*2]*5
        self.reject("mha_state_rejected",plan,**request)
    def test_incompatible_heads_rejected(self):
        request=self.request();request["module"]["config"]["num_heads"]=3
        self.reject("mha_state_rejected",plan,**request)
    def test_unknown_parameter_rejected(self):
        request=self.request();request["module"]["parameters"]["private_source"]=[1.]
        self.reject("mha_state_rejected",plan,**request)
    def test_wrong_input_feature_rejected(self):
        request=self.request(2);request["operands"]["query"]["shape"][-1]=3
        self.reject("mha_shape_rejected",plan,**request)
    def test_alias_shape_conflict_rejected(self):
        request=self.request(2);request["operands"]["key"]["name"]=request["operands"]["query"]["name"]
        self.reject("mha_shape_rejected",plan,**request)
    def test_wrong_mask_shape_rejected(self):
        request=self.request(4);request["options"]["attn_mask"]["values"]=[[False,True]]
        self.reject("mha_mask_rejected",plan,**request)
    def test_all_masked_rows_rejected(self):
        request=self.request(4);request["options"]["attn_mask"]["values"]=[[True]*3]*2
        self.reject("mha_all_masked_unsupported",plan,**request)
    def test_causal_hint_requires_actual_causal_mask(self):
        request=self.request(10);request["options"]["attn_mask"]["values"]=[[False]*3]*2
        self.reject("mha_arguments_rejected",plan,**request)
    def test_negative_infinity_has_exact_public_representation(self):
        mask=public_mask(torch.tensor([[0.,float("-inf")]],dtype=torch.float64))
        self.assertEqual(mask,dict(dtype="float64",values=[[0.,0.]],negative_infinity=[[False,True]]))
        for value in (float("nan"),float("inf")):
            self.reject("mha_mask_rejected",public_mask,torch.tensor([value],dtype=torch.float64))
    def test_unknown_approximation_rejected(self):
        request=self.request();request["approximation"]={}
        self.reject("mha_approximation_rejected",plan,**request)
    def test_logical_capacity_rejected(self):
        request=self.request(2)
        for x in request["operands"].values():x["shape"]=[3000,2,2]
        self.reject("mha_resource_rejected",plan,**request)
    def test_resigned_weight_tamper_rejected(self):
        value=capture(CASES[0])[2];expected=value["source_request_sha256"]
        value["source_request"]["module"]["parameters"]["in_proj_weight"][0][0]+=.125
        value["source_request_sha256"]=digest(value["source_request"]);value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
        self.reject("mha_integrity_rejected",verify,value,expected_source_request_sha256=expected)
    def test_resigned_output_tamper_rejected(self):
        value=capture(CASES[0])[2];expected=value["source_request_sha256"]
        value["output_schema"]["items"][1]=dict(kind="none")
        value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
        self.reject("mha_integrity_rejected",verify,value,expected_source_request_sha256=expected)
    def test_metadata_hooks_and_cycle_rejected(self):
        class Unsafe:
            def __deepcopy__(self,memo):raise AssertionError("Unsafe hook invoked")
        for cycle in (False,True):
            request=self.request()
            request["extra"]=request if cycle else Unsafe()
            self.reject("mha_arguments_rejected" if not cycle else "mha_resource_rejected",plan,
                module=request if cycle else dict(request["module"],unsafe=Unsafe()),
                operands=request["operands"],options=request["options"],approximation=APPROX)
    def test_only_mathematical_draft_status_is_returned(self):
        value=capture(CASES[0])[2]
        for key in ("physical_layout_status","compiler_status","encrypted_status","agent_status"):self.assertEqual(value[key],"unverified")
        self.assertFalse(value["original_function_accuracy_certified"])

if __name__=="__main__":unittest.main()
