"""Pinned-Torch fused matrix semantics and independent graph references."""
import copy,math,operator,unittest
import numpy as np,torch
from benchmark_graph import digest
from pytorch_static_trace import trace,ALIAS_POLICY,COMMON_POLICY
from pytorch_fused_matrix_normalization import POLICY,arguments,plan,normalize,check_record
from pytorch_ordered_capture import prepare_module,COMMON_FORMAT
from pytorch_ordered_reference import validate
from pytorch_model_capture import ADAPTIVE_FUNCTION_POLICY
PROFILE="general-tensors-v1"
EMPTY=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[])
REFERENCES=[]

def shapes(op):
    return ([2],[2,3],[3,2]) if op=="addmm" else ([2],[2,2,3],[2,3,2])
def signature(op):
    return [dict(name=n,shape=s) for n,s in zip(("bias","left","right"),shapes(op))]
class Fusion(torch.nn.Module):
    def __init__(self,op,style="function",context="cipher",alpha=1.25,beta=-.5,composed=False):
        super().__init__();self.op,self.style,self.context=op,style,context
        self.alpha,self.beta,self.composed=alpha,beta,composed
        dims=shapes(op)
        index={"public_left":1,"public_right":2,"public_bias":0}.get(context)
        if index is not None:
            count=math.prod(dims[index])
            value=(torch.arange(count,dtype=torch.float64).reshape(dims[index])-count/2)/7
            self.register_buffer("weight",value)
    def forward(self,bias,left,right):
        previous=left.clone()
        if self.context=="effect":left.transpose(-1,-2).mul_(.75)
        b=self.weight if self.context=="public_bias" else bias
        l=self.weight if self.context=="public_left" else left
        r=self.weight if self.context=="public_right" else right
        if self.style=="method":value=getattr(b,self.op)(l,r,alpha=self.alpha,beta=self.beta)
        elif self.style=="keywords":
            if self.op=="addmm":value=torch.addmm(input=b,mat1=l,mat2=r,alpha=self.alpha,beta=self.beta)
            elif self.op=="baddbmm":value=torch.baddbmm(input=b,batch1=l,batch2=r,alpha=self.alpha,beta=self.beta)
            else:value=torch.addbmm(input=b,batch1=l,batch2=r,alpha=self.alpha,beta=self.beta)
        else:value=getattr(torch,self.op)(b,l,r,alpha=self.alpha,beta=self.beta)
        if self.composed:
            value=value.reshape(value.size(0),-1).square().sum(dim=-1,keepdim=True)
        left.view(-1).add_(.125)
        return dict(result=value,current=left,previous=previous)
def capture(model):
    return prepare_module(model.double().eval(),signature(model.op),["result","current","previous"],
        approximations=EMPTY,resource_profile=PROFILE,matrix_policy="static-tensor-matmul-v1",
        functional_policy=ADAPTIVE_FUNCTION_POLICY,approximation_policy="node-module-default-v1",
        common_calls=True,fused_matrix_policy=POLICY)
def exercise(case,model):
    value=capture(model)
    high,low,binding,_=value
    alias=binding["pytorch_capture"]["ordered_alias_capture"]
    adapter=alias["capture_adapter"];record=adapter["fused_matrix_normalization"]
    assert adapter["format"]=="poseidon-trusted-ordered-fused-matrix-capture-draft-v1"
    assert record["entries"][0]["operation"]==model.op
    ref=validate(value,signature(model.op),["result","current","previous"],resource_profile=PROFILE,count=16)
    REFERENCES.append(dict(case=case,operation=model.op,style=model.style,context=model.context,
        high_sha256=digest(high),lowered_sha256=digest(low),capture_binding_sha256=binding["binding_sha256"],
        normalization_binding=record["binding_sha256"],reference=ref))
    return value
class ReferenceTests(unittest.TestCase):
    def test_addmm_five_contexts_both_call_styles(self):self.group("addmm")
    def test_baddbmm_five_contexts_both_call_styles(self):self.group("baddbmm")
    def test_addbmm_five_contexts_both_call_styles(self):self.group("addbmm")
    def group(self,op):
        for style in ("function","method"):
            for context in ("cipher","public_left","public_right","public_bias","effect"):
                with self.subTest(style=style,context=context):
                    exercise(op+"_"+style+"_"+context,Fusion(op,style,context,composed=context=="effect"))
    def test_keyword_forms_three_contexts(self):
        for op in ("addmm","baddbmm","addbmm"):
            with self.subTest(op=op):exercise(op+"_keywords",Fusion(op,"keywords","cipher",composed=True))
    def test_zero_beta_three_contexts(self):
        for op in ("addmm","baddbmm","addbmm"):
            with self.subTest(op=op):exercise(op+"_beta_zero",Fusion(op,"method","effect",beta=0))
    def test_zero_alpha_three_contexts(self):
        for op in ("addmm","baddbmm","addbmm"):
            with self.subTest(op=op):exercise(op+"_alpha_zero",Fusion(op,alpha=0))
    def test_weight_content_changes_binding_and_math(self):
        a=Fusion("addmm",context="public_right");b=copy.deepcopy(a);b.weight.add_(.375)
        va,vb=capture(a),capture(b)
        self.assertNotEqual(va[2]["binding_sha256"],vb[2]["binding_sha256"])
        x=[torch.ones(s,dtype=torch.float64) for s in shapes("addmm")]
        self.assertFalse(torch.equal(a(*[v.clone() for v in x])["result"],b(*[v.clone() for v in x])["result"]))
        exercise("addmm_weight_changed",b)
    def test_reference_inputs_state_and_source_preserved(self):
        model=Fusion("baddbmm",context="public_right");before=model.weight.clone()
        value=exercise("baddbmm_state_preserved",model)
        torch.testing.assert_close(before,model.weight,rtol=0,atol=0)
        self.assertIsNot(value[3]["original_module"],value[3]["normalized_module"])
class NormalizationTests(unittest.TestCase):
    def raw(self,model):
        return trace(model.double().eval(),signature(model.op),resource_profile=PROFILE,
                     policy=ALIAS_POLICY,fused_matrix_policy=POLICY)
    def test_beta_zero_ignores_nan_bias_pinned_torch(self):
        for op in ("addmm","baddbmm","addbmm"):
            with self.subTest(op=op):
                model=Fusion(op,beta=0);raw=self.raw(model);normalized,record=normalize(raw,signature(op),resource_profile=PROFILE)
                check_record(record,raw,normalized)
                args=[torch.full(s,float("nan") if i==0 else .25,dtype=torch.float64) for i,s in enumerate(shapes(op))]
                a=model(*[x.clone() for x in args]);b=normalized(*[x.clone() for x in args])
                self.assertTrue(torch.isfinite(a["result"]).all())
                torch.testing.assert_close(a["result"],b["result"],rtol=1e-14,atol=1e-14)
                self.assertEqual(record["entries"][0]["plan"]["bias_policy"],"ignored")
    def test_matrix_multiply_order_matches_independent_loops(self):
        for op in ("addmm","baddbmm","addbmm"):
            with self.subTest(op=op):
                model=Fusion(op);raw=self.raw(model);normalized,_=normalize(raw,signature(op),resource_profile=PROFILE)
                args=[np.arange(math.prod(s),dtype=np.float64).reshape(s)/11-.375 for s in shapes(op)]
                bias,left,right=args
                if left.ndim==2:
                    product=np.array([[sum(left[i,k]*right[k,j] for k in range(left.shape[1]))
                        for j in range(right.shape[1])] for i in range(left.shape[0])])
                else:
                    product=np.array([[[sum(left[b,i,k]*right[b,k,j] for k in range(left.shape[2]))
                        for j in range(right.shape[2])] for i in range(left.shape[1])] for b in range(left.shape[0])])
                    if op=="addbmm":product=np.array([[sum(product[b,i,j] for b in range(product.shape[0]))
                        for j in range(product.shape[2])] for i in range(product.shape[1])])
                expected=model.alpha*product+model.beta*bias
                actual=normalized(*[torch.from_numpy(v.copy()) for v in args])["result"].detach().numpy()
                np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-14)
    def test_prior_effect_before_ignored_bias_still_executed(self):
        class Witness(torch.nn.Module):
            op="addmm"
            def forward(self,bias,left,right):
                alias=left.view(-1);left.add_(.5)
                result=torch.addmm(bias,left,right,beta=0)
                left.mul_(.75)
                return result,alias
        model=Witness();raw=self.raw(model);normalized,record=normalize(raw,signature("addmm"),resource_profile=PROFILE)
        args=[torch.full(s,.25,dtype=torch.float64) for s in shapes("addmm")]
        a=model(*[x.clone() for x in args]);b=normalized(*[x.clone() for x in args])
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-14,atol=1e-14)
        self.assertEqual(record["dead_code_policy"],"retain-source-effects-and-order")
    def test_record_tamper_with_recomputed_digest_rejected(self):
        model=Fusion("addmm");raw=self.raw(model);normalized,record=normalize(raw,signature("addmm"),resource_profile=PROFILE)
        for field in ("alpha","beta","result_shape","source_shapes"):
            bad=copy.deepcopy(record)
            bad["entries"][0]["plan"][field]=0 if field in ("alpha","beta") else []
            bad["binding_sha256"]=digest({k:v for k,v in bad.items() if k!="binding_sha256"})
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"provenance"):check_record(bad,raw,normalized)
    def test_public_state_change_detected(self):
        model=Fusion("addmm",context="public_right");raw=self.raw(model);normalized,record=normalize(raw,signature("addmm"),resource_profile=PROFILE)
        normalized.weight.add_(.125)
        with self.assertRaisesRegex(ValueError,"provenance"):check_record(record,raw,normalized)
    def test_wrong_normalized_operator_detected(self):
        model=Fusion("addmm");raw=self.raw(model);normalized,record=normalize(raw,signature("addmm"),resource_profile=PROFILE)
        for n in normalized.graph.nodes:
            if n.op=="call_function" and n.target is torch.mm:n.target=torch.add;break
        normalized.recompile()
        with self.assertRaisesRegex(ValueError,"provenance"):check_record(record,raw,normalized)
    def test_input_signature_binding_tamper_rejected(self):
        model=Fusion("addmm");raw=self.raw(model)
        sig=signature("addmm");sig[0]["shape"]=[1]
        with self.assertRaisesRegex(ValueError,"trace policy"):normalize(raw,sig,resource_profile=PROFILE)
    def test_trace_output_metadata_tamper_rejected(self):
        model=Fusion("addmm");raw=self.raw(model)
        for n in raw.graph.nodes:
            if n.op=="call_function" and n.target is torch.addmm:
                n.meta["poseidon_static_tensor"]["shape"]=[1];break
        with self.assertRaisesRegex(ValueError,"result metadata"):
            normalize(raw,signature("addmm"),resource_profile=PROFILE)
class RejectionTests(unittest.TestCase):
    def test_invalid_scalar_types_and_budgets(self):
        for name in ("alpha","beta"):
            for bad in (True,float("nan"),float("inf"),1024.01,torch.tensor(1.),"1"):
                with self.subTest(name=name,bad=str(bad)),self.assertRaisesRegex(ValueError,"scalar"):
                    arguments((1,2,3),{name:bad},"addmm")
    def test_argument_ambiguity_out_and_unknown_fields(self):
        for args,kwargs,method in (((1,2,3),{"input":4},False),((1,2),{},False),
                 ((1,2,3,4),{},False),((1,2,3),{"out":4},False),
                 ((1,2,3),{"out":None},True),((1,2,3),{"rounding_mode":"floor"},False)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):arguments(args,kwargs,"addmm",method=method)
    def test_matrix_rank_batch_contraction_and_bias_mismatches(self):
        for op,ss in (("addmm",[[2],[2,2,3],[2,3,2]]),("baddbmm",[[2],[1,2,3],[2,3,2]]),
                      ("addmm",[[2],[2,4],[3,2]]),("addmm",[[3],[2,3],[3,2]]),
                      ("addbmm",[[2,2,2],[2,2,3],[2,3,2]])):
            with self.subTest(op=op,shapes=ss),self.assertRaises(ValueError):plan(op,ss,resource_profile=PROFILE)
    def test_expansion_shape_budget(self):
        with self.assertRaisesRegex(ValueError,"budget"):
            plan("baddbmm",[[1],[2,300,1],[2,1,300]],resource_profile=PROFILE)
    def test_beta_zero_does_not_disable_shape_checks(self):
        with self.assertRaisesRegex(ValueError,"bias broadcast"):
            plan("addmm",[[3],[2,3],[3,2]],beta=0,resource_profile=PROFILE)
        with self.assertRaises(RuntimeError):torch.addmm(torch.zeros(3),torch.zeros(2,3),torch.zeros(3,2),beta=0)
    def test_unregistered_policy_and_wrong_trace_policy(self):
        model=Fusion("addmm")
        for policy,fused in ((ALIAS_POLICY,"anything"),(COMMON_POLICY,POLICY)):
            with self.subTest(policy=policy),self.assertRaisesRegex(ValueError,"metadata policy"):
                trace(model,signature("addmm"),resource_profile=PROFILE,policy=policy,fused_matrix_policy=fused)
        with self.assertRaisesRegex(ValueError,"capture policy"):
            prepare_module(model,signature("addmm"),["result","current","previous"],
                resource_profile=PROFILE,fused_matrix_policy=POLICY)
    def test_pure_public_matrix_product_explicitly_rejected(self):
        class PublicProduct(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("left",torch.ones(2,3,dtype=torch.float64))
                self.register_buffer("right",torch.ones(3,2,dtype=torch.float64))
            def forward(self,x):return torch.addmm(x,self.left,self.right)
        sig=[dict(name="x",shape=[2])]
        raw=trace(PublicProduct(),sig,resource_profile=PROFILE,policy=ALIAS_POLICY,fused_matrix_policy=POLICY)
        with self.assertRaisesRegex(ValueError,"encrypted operand"):normalize(raw,sig,resource_profile=PROFILE)
    def test_default_intake_does_not_open_new_calls(self):
        model=Fusion("addmm")
        with self.assertRaises(ValueError):
            prepare_module(model,signature("addmm"),["result","current","previous"],
                approximations=EMPTY,resource_profile=PROFILE,matrix_policy="static-tensor-matmul-v1",common_calls=True)
class CompatibilityTests(unittest.TestCase):
    def test_common_default_keeps_original_format(self):
        class Retained(torch.nn.Module):
            def forward(self,x):
                old=x.clone();value=x.add(x,alpha=.25);x.add_(.125);return value,x,old
        sig=[dict(name="x",shape=[2,3])];names=["result","current","previous"]
        value=prepare_module(Retained().double(),sig,names,approximations=EMPTY,resource_profile=PROFILE,
              matrix_policy="static-tensor-matmul-v1",common_calls=True)
        adapter=value[2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]
        self.assertEqual(adapter["format"],COMMON_FORMAT);self.assertNotIn("fused_matrix_normalization",adapter)
        validate(value,sig,names,resource_profile=PROFILE,count=16)
    def test_metadata_default_has_no_fused_policy(self):
        class Retained(torch.nn.Module):
            def forward(self,x):return x.square()
        raw=trace(Retained(),[dict(name="x",shape=[2,3])],resource_profile=PROFILE,policy=ALIAS_POLICY)
        self.assertNotIn("fused_matrix_policy",raw.meta)
