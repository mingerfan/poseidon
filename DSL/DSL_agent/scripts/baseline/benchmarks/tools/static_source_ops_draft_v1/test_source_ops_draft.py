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

    def test_call_positional_and_keyword_binding(self):
        from source_ops_draft import source_call,verify_call
        x=dict(kind="cipher",name="x",shape=[2,3],dtype="float64");d=dict(kind="public",value=2.)
        a=source_call("torch.div",[x,d],{})
        b=source_call("torch.div",[],dict(input=x,other=d))
        self.assertEqual(a["plan"]["mathematical_graph"],b["plan"]["mathematical_graph"])
        self.assertNotEqual(a["source_call_sha256"],b["source_call_sha256"])
        verify_call(a,expected_source_call_sha256=a["source_call_sha256"])
    def test_call_duplicate_and_unknown_arguments(self):
        from source_ops_draft import source_call
        x=dict(kind="cipher",name="x",shape=[2],dtype="float64");d=dict(kind="public",value=2.)
        self.rejected("source_call_arguments_rejected",source_call,"torch.div",[x,d],dict(input=x))
        self.rejected("source_call_arguments_rejected",source_call,"torch.div",[x,d],dict(unknown_private="secret"))
        self.rejected("source_call_arguments_rejected",source_call,"operator.truediv",[x,d],dict(rounding_mode=None))
    def test_call_no_silent_dtype_conversion(self):
        from source_ops_draft import source_call
        x=dict(kind="cipher",name="x",shape=[2],dtype="float32");d=dict(kind="public",value=2.)
        self.rejected("source_cipher_metadata_rejected",source_call,"torch.div",[x,d],{})
    def test_call_cipher_divisor_not_public_inverse(self):
        from source_ops_draft import source_call
        x=dict(kind="cipher",name="x",shape=[2],dtype="float64")
        self.rejected("division_cipher_denominator_unsupported",source_call,"torch.div",[x,x],{})
    def test_pool_call_standard_signature(self):
        from source_ops_draft import source_call,verify_call
        x=dict(kind="cipher",name="x",shape=[1,1,4,4],dtype="float64")
        a=source_call("F.max_pool2d",[x,2],{},approximation=APPROX)
        b=source_call("torch.nn.functional.max_pool2d",[],dict(input=x,kernel_size=2),approximation=APPROX)
        self.assertEqual(a["plan"]["mathematical_graph"],b["plan"]["mathematical_graph"])
        verify_call(a,expected_source_call_sha256=a["source_call_sha256"])
    def test_call_resigned_tamper_rejected(self):
        from source_ops_draft import source_call,verify_call
        x=dict(kind="cipher",name="x",shape=[2],dtype="float64");d=dict(kind="public",value=2.)
        a=source_call("torch.div",[x,d],{});expected=a["source_call_sha256"]
        a["source_call"]["args"][1]["value"]=4.
        a["source_call_sha256"]=digest(a["source_call"]);a["binding"]=digest({k:v for k,v in a.items() if k!="binding"})
        self.rejected("source_plan_integrity_rejected",verify_call,a,expected_source_call_sha256=expected)


    def test_alias_argument_admission_matches_fixed_torch(self):
        import operator,torch
        from source_ops_draft import source_call
        x=torch.tensor([.25,-.5],dtype=torch.float64)
        node=dict(kind="cipher",name="x",shape=[2],dtype="float64")
        divisor=dict(kind="public",value=2.)
        for alias in DIVISION_CALLS:
            method=alias.startswith("Tensor.")
            native=(operator.truediv if alias=="operator.truediv" else
                    getattr(x if method else torch,alias.split(".")[1]))
            native_args=(2.,) if method else (x,2.)
            tests=[
                ("base",native_args,{},[node,divisor],{}),
                ("rounding_none",native_args,dict(rounding_mode=None),[node,divisor],dict(rounding_mode=None)),
                ("out_none",native_args,dict(out=None),[node,divisor],dict(out=None)),
                ("all_keywords",(),dict(input=x,other=2.),[],dict(input=node,other=divisor)),
                ("other_keyword",() if method else (x,),dict(other=2.),[node],dict(other=divisor)),
                ("third_positional",native_args+(None,),{},[node,divisor,None],{}),
                ("duplicate_input",native_args,dict(input=x),[node,divisor],dict(input=node)),
            ]
            for label,nargs,nkwargs,args,kwargs in tests:
                with self.subTest(alias=alias,variant=label):
                    try:native(*nargs,**nkwargs);native_ok=True
                    except TypeError:native_ok=False
                    try:source_call(alias,args,kwargs);planned_ok=True
                    except SourcePlanError:planned_ok=False
                    self.assertEqual(planned_ok,native_ok)

    def test_pool_argument_admission_matches_fixed_torch(self):
        import torch
        from torch.nn import functional as F
        from source_ops_draft import source_call
        for rank in (1,2):
            input_shape=[1,1]+[4]*rank
            x=torch.arange(4**rank,dtype=torch.float64).reshape(input_shape)/32
            node=dict(kind="cipher",name="x",shape=input_shape,dtype="float64")
            native=getattr(F,"max_pool"+str(rank)+"d")
            for prefix in ("F.","torch.nn.functional."):
                alias=prefix+"max_pool"+str(rank)+"d"
                tests=[
                    ("default",(x,2),{},[node,2],{}),
                    ("all_keywords",(),dict(input=x,kernel_size=2),[],dict(input=node,kernel_size=2)),
                    ("full_positional",(x,2,1,0,1,False,False),{},[node,2,1,0,1,False,False],{}),
                    ("duplicate",(x,2),dict(kernel_size=2),[node,2],dict(kernel_size=2)),
                    ("unknown",(x,2),dict(unknown_private="secret"),[node,2],dict(unknown_private="secret")),
                ]
                for label,nargs,nkwargs,args,kwargs in tests:
                    with self.subTest(alias=alias,variant=label):
                        try:native(*nargs,**nkwargs);native_ok=True
                        except TypeError:native_ok=False
                        try:source_call(alias,args,kwargs,approximation=APPROX);planned_ok=True
                        except SourcePlanError:planned_ok=False
                        self.assertEqual(planned_ok,native_ok)

if __name__=="__main__":unittest.main()
