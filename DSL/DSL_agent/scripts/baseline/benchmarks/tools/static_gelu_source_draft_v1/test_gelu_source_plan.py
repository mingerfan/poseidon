import copy,math,unittest
import torch
from torch.nn import functional as F
from benchmark_graph import digest
from gelu_source_plan import (plan,source_call,verify_plan,verify_call,from_fx_call,GeluSourceError)
from gelu_source_reference import original_scalar,check_domain
A=1/math.sqrt(2*math.pi)
POWER=dict(method="polynomial",basis="power",domain=[-1.,1.],coefficients=[0.,.5,A,0.,-A/6])
CHEB=dict(method="polynomial",basis="chebyshev",domain=[-1.,1.],coefficients=[7*A/16,.5,5*A/12,0.,-A/48])

def operand(shape=(4,),name="x",dtype="float64"):
    return dict(kind="cipher",name=name,shape=list(shape),dtype=dtype)

class Fixture(torch.nn.Module):
    def __init__(self,variant,source):
        super().__init__();self.variant=variant;self.source=source
        self.activation=torch.nn.GELU(approximate=variant)
    def forward(self,x):
        if self.source=="module":return self.activation(x)
        if self.source=="input_keyword":return F.gelu(input=x,approximate=self.variant)
        return F.gelu(x,approximate=self.variant)

def capture(variant,source,shape,approximation):
    fixture=Fixture(variant,source).eval()
    graph=torch.fx.symbolic_trace(fixture)
    nodes=list(graph.graph.nodes);placeholder=next(n for n in nodes if n.op=="placeholder")
    calls=[n for n in nodes if n.op in ("call_function","call_module")]
    if len(calls)!=1:raise ValueError("Known single-operation fixture")
    node=calls[0];module=graph.get_submodule(node.target) if node.op=="call_module" else None
    value=from_fx_call(node,{placeholder:operand(shape)},module=module,approximation=approximation)
    verify_call(value,expected_source_call_sha256=value["source_call_sha256"])
    return fixture,graph,value

class GeluSourceTests(unittest.TestCase):
    def reject(self,code,fn,*args,**kwargs):
        with self.assertRaises(GeluSourceError) as cm:fn(*args,**kwargs)
        self.assertEqual(cm.exception.code,code)
        self.assertEqual(cm.exception.feedback()["retry_policy"],"never_automatic")
        return cm.exception.feedback()

    def test_variants_have_distinct_source_bindings_same_requested_polynomial(self):
        a=plan(operand(),approximate="none",approximation=POWER)
        b=plan(operand(),approximate="tanh",approximation=POWER)
        self.assertNotEqual(a["source_request_sha256"],b["source_request_sha256"])
        self.assertNotEqual(a["source_function"],b["source_function"])
        self.assertEqual(a["approximate_mathematical_graph"],b["approximate_mathematical_graph"])

    def test_original_tanh_is_not_erf(self):
        x=-.75
        exact=original_scalar(x,"none");tanh=original_scalar(x,"tanh")
        self.assertGreater(abs(exact-tanh),1e-5+1e-4*abs(tanh))
        self.assertAlmostEqual(tanh,F.gelu(torch.tensor(x,dtype=torch.float64),approximate="tanh").item(),places=15)

    def test_polynomial_is_explicit(self):
        for value in (None,{},dict(POWER,coefficients=[True]),dict(POWER,basis="other")):
            self.reject("gelu_source_approximation_rejected",plan,operand(),approximate="tanh",approximation=value)

    def test_unknown_variants_not_echoed(self):
        for value in (None,True,2,"private_source_path"):
            payload=self.reject("gelu_source_variant_rejected",plan,operand(),approximate=value,approximation=POWER)
            self.assertNotIn("private_source_path",str(payload))

    def test_cipher_metadata_and_dtype(self):
        self.reject("gelu_source_input_rejected",plan,operand(dtype="float32"),approximate="tanh",approximation=POWER)
        self.reject("gelu_source_input_rejected",plan,dict(operand(),extra=True),approximate="tanh",approximation=POWER)

    def test_rank_and_element_limits(self):
        self.reject("gelu_source_resource_rejected",plan,operand([1]*5),approximate="tanh",approximation=POWER,resource_profile="legacy-v1")
        p=plan(operand([1]*31+[2]),approximate="tanh",approximation=POWER)
        self.assertEqual(len(p["lowered_graph"]["inputs"][0]["shape"]),32)
        self.reject("gelu_source_resource_rejected",plan,operand([4097]),approximate="tanh",approximation=POWER)

    def test_name_collision_is_avoided(self):
        p=plan(operand(name="public_coefficients"),approximate="tanh",approximation=POWER)
        self.assertNotIn("public_coefficients",p["lowered_graph"]["constants"])

    def test_profile_and_source_are_copied(self):
        x=operand();profile=copy.deepcopy(POWER);p=plan(x,approximate="tanh",approximation=profile)
        x["shape"][0]=2;profile["coefficients"][0]=9.
        self.assertEqual(p["source_request"]["operand"]["shape"],[4])
        self.assertEqual(p["source_request"]["approximation"]["coefficients"][0],0.)

    def test_resigned_variant_tamper_rejected(self):
        p=plan(operand(),approximate="tanh",approximation=POWER);expected=p["source_request_sha256"]
        p["source_request"]["approximate"]="none";p["source_request_sha256"]=digest(p["source_request"])
        p["binding"]=digest({k:v for k,v in p.items() if k!="binding"})
        self.reject("gelu_source_integrity_rejected",verify_plan,p,expected_source_request_sha256=expected)

    def test_resigned_polynomial_tamper_rejected(self):
        p=plan(operand(),approximate="tanh",approximation=POWER);expected=p["source_request_sha256"]
        key=next(iter(p["lowered_graph"]["constants"]));p["lowered_graph"]["constants"][key][0]=9.
        p["binding"]=digest({k:v for k,v in p.items() if k!="binding"})
        self.reject("gelu_source_integrity_rejected",verify_plan,p,expected_source_request_sha256=expected)

    def test_function_argument_admission_matches_fixed_torch(self):
        x=torch.tensor([-.75,.75],dtype=torch.float64);meta=operand([2])
        cases=[("default",(x,),{},[meta],{}),
               ("tanh",(x,),dict(approximate="tanh"),[meta],dict(approximate="tanh")),
               ("none",(x,),dict(approximate="none"),[meta],dict(approximate="none")),
               ("tanh_positional",(x,"tanh"),{},[meta,"tanh"],{}),
               ("input_keyword",(),dict(input=x,approximate="tanh"),[],dict(input=meta,approximate="tanh")),
               ("duplicate",(x,),dict(input=x),[meta],dict(input=meta)),
               ("invalid",(x,),dict(approximate="unknown"),[meta],dict(approximate="unknown")),
               ("null",(x,),dict(approximate=None),[meta],dict(approximate=None)),
               ("out_none",(x,),dict(out=None),[meta],dict(out=None)),
               ("unknown",(x,),dict(private_source="secret"),[meta],dict(private_source="secret"))]
        for alias in ("F.gelu","torch.nn.functional.gelu"):
            for label,nargs,nkwargs,args,kwargs in cases:
                with self.subTest(alias=alias,variant=label):
                    try:F.gelu(*nargs,**nkwargs);native=True
                    except (TypeError,RuntimeError):native=False
                    try:source_call(alias,args,kwargs,approximation=POWER);planned=True
                    except GeluSourceError:planned=False
                    self.assertEqual(native,planned)

    def test_out_mutation_is_rejected(self):
        self.reject("gelu_source_out_rejected",source_call,"F.gelu",[operand()],dict(out=operand()),approximation=POWER)

    def test_actual_module_and_function_fx(self):
        for variant in ("none","tanh"):
            for source in ("module","functional","input_keyword"):
                with self.subTest(variant=variant,source=source):
                    fixture,graph,value=capture(variant,source,[2,3],POWER)
                    self.assertEqual(value["plan"]["source_function"]["variant"],variant)

    def test_module_variant_cannot_be_forward_keyword(self):
        self.reject("gelu_source_arguments_rejected",source_call,"nn.GELU",[operand()],dict(approximate="tanh"),module_variant="tanh",approximation=POWER)

    def test_module_subclass_rejected(self):
        class Custom(torch.nn.GELU):
            def forward(self,x):return x
        g=torch.fx.Graph();x=g.placeholder("x");node=g.call_module("activation",(x,))
        self.reject("gelu_source_call_rejected",from_fx_call,node,{x:operand()},module=Custom(),approximation=POWER)

    def test_function_name_spoof_rejected(self):
        def private_source(x):return x
        private_source.__name__="gelu";private_source.__module__="torch.nn.functional"
        g=torch.fx.Graph();x=g.placeholder("x");node=g.call_function(private_source,(x,))
        self.reject("gelu_source_call_rejected",from_fx_call,node,{x:operand()},approximation=POWER)

    def test_metadata_hooks_and_cycle_rejected(self):
        class Unsafe:
            def __deepcopy__(self,memo):raise AssertionError("Unsafe hook called")
        for x in (dict(operand(),dtype=Unsafe()),dict(operand())):
            if type(x["dtype"]) is str:x["cycle"]=x
            self.reject("gelu_source_arguments_rejected",plan,x,approximate="tanh",approximation=POWER)

    def test_domain_rejection_does_not_silently_clamp(self):
        self.reject("gelu_source_domain_rejected",check_domain,[1.001],POWER)

    def test_call_resigned_source_tamper_rejected(self):
        value=source_call("F.gelu",[operand()],dict(approximate="tanh"),approximation=POWER)
        expected=value["source_call_sha256"];value["source_call"]["kwargs"]["approximate"]="none"
        value["source_call_sha256"]=digest(value["source_call"]);value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
        self.reject("gelu_source_integrity_rejected",verify_call,value,expected_source_call_sha256=expected)

    def test_only_draft_math_status_is_returned(self):
        p=plan(operand(),approximate="tanh",approximation=POWER)
        self.assertFalse(p["production_intake_integrated"]);self.assertFalse(p["original_function_accuracy_certified"])
        for field in ("physical_layout_status","compiler_status","encrypted_status","agent_status"):
            self.assertEqual(p[field],"unverified")

if __name__=="__main__":unittest.main()
