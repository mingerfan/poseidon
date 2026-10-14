"""Registered common PyTorch calls, finite references and retained source versions."""
import copy, math, operator, unittest
import torch
from torch.nn import functional as F
from benchmark_graph import canonical, digest
from pytorch_static_trace import trace, COMMON_POLICY as SHAPE_POLICY, OPERATOR_POLICY, INDEX_POLICY, policy_record
from pytorch_model_capture import capture, ADAPTIVE_FUNCTION_POLICY
from pytorch_arithmetic_normalization import (normalize_common, normalize, rename_record,
    bind_record, bind_common_record, check_common_record, sdpa_shape)
import pytorch_model_intake as intake

PROFILE="general-tensors-v1"
APPROX=dict(exp=dict(method="taylor",domain=[-8.,8.],degree=2),
            reciprocal=dict(method="goldschmidt",domain=[0.5,100.],iterations=1))

def graph(target, *, names=("x","y"), args=None, kwargs=None, root=None, method=False):
    g=torch.fx.Graph();ports=[g.placeholder(n) for n in names]
    operands=tuple(ports) if args is None else args(g,ports)
    node=g.call_method(target,operands,kwargs or {}) if method else g.call_function(target,operands,kwargs or {})
    g.output(node)
    return torch.fx.GraphModule(root or torch.nn.Module(),g)

def captured(module, signatures):
    raw=trace(module.eval(),signatures,resource_profile=PROFILE,policy=SHAPE_POLICY)
    common,change=normalize_common(raw,signatures,resource_profile=PROFILE)
    normalized,arithmetic=normalize(common,resource_profile=PROFILE,preserve_unused_state=True)
    final=trace(normalized,signatures,resource_profile=PROFILE,policy=INDEX_POLICY)
    arithmetic=rename_record(arithmetic,normalized,final)
    high,_,metadata=capture(final,signatures,["result"],approximations=dict(format="poseidon-pytorch-approximations-v1",defaults=APPROX,modules=[],nodes=[]),
        approximation_policy="node-module-default-v1",
        resource_profile=PROFILE,dtype_policy="strict-float64-v1",preserve_output_structure=True,
        record_state_sources=True,capture_policy=INDEX_POLICY,matrix_policy="static-tensor-matmul-v1",
        functional_policy=ADAPTIVE_FUNCTION_POLICY)
    arithmetic=bind_record(arithmetic,high,captured_fx=final)
    change=bind_common_record(change,high,arithmetic,metadata["pytorch_capture"]["state_constant_sources"])
    return high,change,arithmetic

class Projected(torch.nn.Module):
    def __init__(self,mask_kind="none",style=0):
        super().__init__();self.mask_kind=mask_kind;self.style=style
        self.q=torch.nn.Linear(4,4);self.k=torch.nn.Linear(4,4);self.v=torch.nn.Linear(4,4)
        if mask_kind=="boolean":self.register_buffer("mask",torch.tensor([[True,False],[True,True]]))
        elif mask_kind=="additive":self.register_buffer("mask",torch.tensor([[0.,-0.75],[0.125,0.]],dtype=torch.float64))
        self.double().eval()
        with torch.no_grad():
            for i,p in enumerate(self.parameters()):
                p.copy_(torch.arange(p.numel(),dtype=torch.float64).reshape(p.shape)*.007-.03+i*.002)
    def forward(self,x):
        q=self.q(x).reshape(1,2,2,2).transpose(1,2)
        k=F.linear(x,self.k.weight,self.k.bias).reshape(1,2,2,2).transpose(1,2)
        v=self.v(x).reshape(1,2,2,2).transpose(1,2)
        if self.style==1:q=F.dropout(q,p=.2,training=False)
        if self.style==2:v=F.dropout2d(v,p=.3,training=False)
        mask=self.mask if self.mask_kind in ("boolean","additive") else None
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,
                                         is_causal=self.mask_kind=="causal")
        return y.reshape(y.shape[0],-1)

class CommonNormalizationTests(unittest.TestCase):
    def test_alpha_functions_methods_broadcast_and_static_values(self):
        signatures=[dict(name="x",shape=[2,3]),dict(name="y",shape=[3])]
        x=torch.arange(6,dtype=torch.float64).reshape(2,3)/3
        y=torch.tensor([.25,-2.,3.],dtype=torch.float64)
        for fn,method in ((torch.add,False),(torch.sub,False),(torch.subtract,False),("add",True),("sub",True)):
            for alpha in (-2.,0.,2.,.125):
                with self.subTest(fn=fn,alpha=alpha):
                    model=graph(fn,method=method,kwargs=dict(alpha=alpha));before=str(model.graph)
                    result,record=normalize_common(model,signatures,resource_profile=PROFILE)
                    torch.testing.assert_close(result(x,y),model(x,y),atol=1e-12,rtol=1e-12)
                    self.assertEqual(str(model.graph),before)
                    self.assertEqual(record["entries"][0]["attrs"]["alpha"],alpha)
    def test_public_scalar_alpha_is_folded_without_public_arithmetic(self):
        model=graph(torch.add,names=("x",),args=lambda g,p:(p[0],.25),kwargs=dict(alpha=-2))
        result,record=normalize_common(model,[dict(name="x",shape=[3])],resource_profile=PROFILE)
        result,_=normalize(result,resource_profile=PROFILE)
        x=torch.arange(3,dtype=torch.float64)
        torch.testing.assert_close(result(x),model(x),atol=0,rtol=0)
        self.assertEqual(record["generated_constants"],[])
    def test_public_tensor_alpha_is_bound_and_source_preserved(self):
        root=torch.nn.Module();root.register_buffer("bias",torch.tensor([.2,-.4,.6],dtype=torch.float64))
        model=graph(torch.sub,names=("x",),root=root,args=lambda g,p:(p[0],g.get_attr("bias")),kwargs=dict(alpha=2))
        signature=[dict(name="x",shape=[3])]
        result,record=normalize_common(model,signature,resource_profile=PROFILE)
        torch.testing.assert_close(result(torch.ones(3,dtype=torch.float64)),model(torch.ones(3,dtype=torch.float64)),atol=0,rtol=0)
        self.assertEqual(record["generated_constants"][0]["kind"],"alpha_constant")
        torch.testing.assert_close(model.bias,root.bias,atol=0,rtol=0)
    def test_invalid_alpha_and_out_rejected(self):
        signature=[dict(name=n,shape=[2]) for n in ("x","y")]
        for alpha in (True,float("nan"),float("inf")):
            with self.subTest(alpha=alpha),self.assertRaisesRegex(ValueError,"Finite public scalar"):
                normalize_common(graph(torch.add,kwargs=dict(alpha=alpha)),signature,resource_profile=PROFILE)
        dynamic=graph(torch.add,names=("x","y","a"),args=lambda g,p:(p[0],p[1]),kwargs={})
        node=next(n for n in dynamic.graph.nodes if n.op=="call_function")
        node.kwargs={"alpha":next(n for n in dynamic.graph.nodes if n.name=="a")}
        with self.assertRaisesRegex(ValueError,"Finite public scalar"):
            normalize_common(dynamic,signature+[dict(name="a",shape=[2])],resource_profile=PROFILE)
        model=graph(torch.add,args=lambda g,p:(p[0],p[1]))
        node=next(n for n in model.graph.nodes if n.op=="call_function");node.kwargs={"out":next(n for n in model.graph.nodes if n.name=="x")}
        with self.assertRaisesRegex(ValueError,"out mutation"):
            normalize_common(model,signature,resource_profile=PROFILE)
    def test_inplace_arithmetic_retains_existing_refusal(self):
        model=graph("add_",method=True)
        with self.assertRaisesRegex(ValueError,"In-place arithmetic"):
            common,_=normalize_common(model,[dict(name=n,shape=[2]) for n in ("x","y")],resource_profile=PROFILE)
            normalize(common,resource_profile=PROFILE,preserve_unused_state=True)
    def test_eval_dropout_variants(self):
        variants=((F.dropout,[2,4]),(F.dropout1d,[1,2,4]),(F.dropout2d,[1,2,2,2]),
                  (F.dropout3d,[1,2,1,2,2]),(F.alpha_dropout,[2,4]),(F.feature_alpha_dropout,[1,2,2,2]))
        for fn,dims in variants:
            with self.subTest(fn=fn):
                model=graph(fn,names=("x",),kwargs=dict(training=False,p=.25))
                out,record=normalize_common(model,[dict(name="x",shape=dims)],resource_profile=PROFILE)
                x=torch.arange(math.prod(dims),dtype=torch.float64).reshape(dims)
                torch.testing.assert_close(out(x),model(x),atol=0,rtol=0)
                self.assertEqual(record["entries"][0]["rule"],"inference_dropout")
    def test_dropout_probability_training_inplace_and_rank_rejected(self):
        for kw in (dict(training=True),dict(training=False,inplace=True),dict(training=False,p=-.1),dict(training=False,p=True)):
            with self.subTest(kwargs=kw),self.assertRaises(ValueError):
                normalize_common(graph(F.dropout,names=("x",),kwargs=kw),[dict(name="x",shape=[2])],resource_profile=PROFILE)
        with self.assertRaisesRegex(ValueError,"Spatial Dropout input rank"):
            normalize_common(graph(F.dropout1d,names=("x",),kwargs=dict(training=False)),
                             [dict(name="x",shape=[1,2,2,2])],resource_profile=PROFILE)
    def test_sdpa_three_direct_shapes_and_determinism(self):
        for dims in (([2,3],[4,3],[4,2]),([1,2,2,3],[1,2,4,3],[1,2,4,2]),
                     ([2,1,2,3],[1,2,4,3],[2,2,4,2])):
            with self.subTest(dims=dims):
                model=graph(F.scaled_dot_product_attention,names=("q","k","v"))
                signatures=[dict(name=n,shape=d) for n,d in zip(("q","k","v"),dims)]
                first,r1=normalize_common(model,signatures,resource_profile=PROFILE)
                second,r2=normalize_common(model,signatures,resource_profile=PROFILE)
                gen=torch.Generator().manual_seed(183);args=[torch.randn(d,generator=gen,dtype=torch.float64)*.25 for d in dims]
                torch.testing.assert_close(first(*args),model(*args),atol=1e-12,rtol=1e-12)
                self.assertEqual(canonical(r1),canonical(r2))
    def test_projected_sdpa_each_mask_three_contexts(self):
        signatures=[dict(name="x",shape=[1,2,4])]
        for mask in ("none","boolean","additive","causal"):
            for style in range(3):
                with self.subTest(mask=mask,style=style):
                    model=Projected(mask,style);source=trace(model,signatures,resource_profile=PROFILE,policy=SHAPE_POLICY)
                    transformed,record=normalize_common(source,signatures,resource_profile=PROFILE)
                    for value in (torch.zeros(1,2,4,dtype=torch.float64),
                                  torch.arange(8,dtype=torch.float64).reshape(1,2,4)/8-.5):
                        torch.testing.assert_close(transformed(value),model(value),atol=1e-12,rtol=1e-12)
                    entry=next(e for e in record["entries"] if e["rule"]=="sdpa")
                    self.assertEqual(entry["attrs"]["mask_kind"],mask)
    def test_wrong_dimensions_and_capacity_rejected(self):
        for dims in (([2,3],[4,2],[4,2]),([2,3],[4,3],[3,2]),([512,2],[512,2],[512,2])):
            with self.subTest(dims=dims),self.assertRaises(ValueError):
                sdpa_shape(*dims,resource_profile=PROFILE)
    def test_sdpa_dropout_dynamic_causal_and_mask_combo_rejected(self):
        signatures=[dict(name=n,shape=[2,2]) for n in ("q","k","v")]
        for kw in (dict(dropout_p=.2),dict(is_causal=1),dict(is_causal=True,attn_mask=True)):
            with self.subTest(kwargs=kw),self.assertRaises(ValueError):
                normalize_common(graph(F.scaled_dot_product_attention,names=("q","k","v"),kwargs=kw),signatures,resource_profile=PROFILE)
    def test_encrypted_mask_rejected(self):
        g=torch.fx.Graph();ports=[g.placeholder(n) for n in ("q","k","v","mask")]
        n=g.call_function(F.scaled_dot_product_attention,tuple(ports),{});g.output(n)
        with self.assertRaisesRegex(ValueError,"public state"):
            normalize_common(torch.fx.GraphModule(torch.nn.Module(),g),[dict(name=n,shape=[2,2]) for n in ("q","k","v","mask")],resource_profile=PROFILE)
    def test_mask_nonfinite_fully_masked_dtype_and_broadcast_rejected(self):
        masks=(torch.tensor([[0.,float("-inf")],[0.,0.]],dtype=torch.float64),
               torch.tensor([[False,False],[True,False]]),torch.ones(2,2,dtype=torch.int64),
               torch.ones(3,2,dtype=torch.float64))
        for mask in masks:
            with self.subTest(mask=mask),self.assertRaises((ValueError,RuntimeError)):
                root=torch.nn.Module();root.register_buffer("mask",mask)
                m=graph(F.scaled_dot_product_attention,names=("q","k","v"),root=root,
                        args=lambda g,p:tuple(p)+(g.get_attr("mask"),))
                normalize_common(m,[dict(name=n,shape=[2,2]) for n in ("q","k","v")],resource_profile=PROFILE)
    def test_intermediate_shape_requires_registered_metadata(self):
        model=graph(F.scaled_dot_product_attention,names=("q","k","v"),args=lambda g,p:(g.call_function(torch.flip,(p[0],(0,))),p[1],p[2]))
        with self.assertRaisesRegex(ValueError,"Registered static tensor metadata"):
            normalize_common(model,[dict(name=n,shape=[2,2]) for n in ("q","k","v")],resource_profile=PROFILE)
    def test_dead_generated_constants_are_not_reported_as_executed(self):
        root=torch.nn.Module();root.register_buffer("bias",torch.tensor([.2,.3],dtype=torch.float64))
        g=torch.fx.Graph();x=g.placeholder("x");bias=g.get_attr("bias")
        g.call_function(torch.add,(x,bias),{"alpha":2});g.output(x)
        model=torch.fx.GraphModule(root,g)
        normalized,record=normalize_common(model,[dict(name="x",shape=[2])],resource_profile=PROFILE)
        self.assertEqual(record["generated_constants"],[])
        self.assertEqual(record["entries"][0]["normalized_nodes"],[])
        normalize(normalized,resource_profile=PROFILE,preserve_unused_state=True)
    def test_common_clone_retains_unused_registered_public_state(self):
        model=Projected("boolean")
        signatures=[dict(name="x",shape=[1,2,4])]
        raw=trace(model,signatures,resource_profile=PROFILE,policy=SHAPE_POLICY)
        common,_=normalize_common(raw,signatures,resource_profile=PROFILE)
        normalized,_=normalize(common,resource_profile=PROFILE,preserve_unused_state=True)
        self.assertEqual(set(common.state_dict()),set(normalized.state_dict()))
        torch.testing.assert_close(normalized.mask,model.mask,atol=0,rtol=0)
    def test_unknown_call_not_rewritten_or_used_for_shape(self):
        model=graph(torch.flip,names=("x",),args=lambda g,p:(p[0],(0,)))
        result,record=normalize_common(model,[dict(name="x",shape=[2])],resource_profile=PROFILE)
        self.assertEqual(str(result.graph),str(model.graph));self.assertEqual(record["entries"],[])
        class Unknown(torch.nn.Module):
            def forward(self,x):
                y=torch.flip(x,[0]);return y.reshape(y.shape[0],-1)
        with self.assertRaisesRegex(ValueError,"Static shape unavailable"):
            trace(Unknown(),[dict(name="x",shape=[2,2])],resource_profile=PROFILE,policy=SHAPE_POLICY)
    def test_unmatched_approximation_not_silently_accepted(self):
        model=Projected("boolean")
        raw=trace(model,[dict(name="x",shape=[1,2,4])],resource_profile=PROFILE,policy=SHAPE_POLICY)
        common,_=normalize_common(raw,[dict(name="x",shape=[1,2,4])],resource_profile=PROFILE)
        norm,_=normalize(common,resource_profile=PROFILE,preserve_unused_state=True)
        final=trace(norm,[dict(name="x",shape=[1,2,4])],resource_profile=PROFILE,policy=INDEX_POLICY)
        with self.assertRaises(ValueError):
            capture(final,[dict(name="x",shape=[1,2,4])],["result"],
                approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
                approximation_policy="node-module-default-v1",resource_profile=PROFILE,dtype_policy="strict-float64-v1",
                preserve_output_structure=True,capture_policy=INDEX_POLICY,matrix_policy="static-tensor-matmul-v1")

class CommonBindingTests(unittest.TestCase):
    def setUp(self):
        self.high,self.record,self.arithmetic=captured(Projected("boolean"),[dict(name="x",shape=[1,2,4])])
    def resign(self,record):
        record.pop("binding_sha256",None);record["binding_sha256"]=digest(record);return record
    def test_bound_mask_matches_actual_graph_constant(self):
        check_common_record(self.record,self.high,self.arithmetic)
        item=self.record["generated_constants"][0]
        self.assertEqual(item["kind"],"mask")
        self.assertIn(item["captured_constant"],self.high["constants"])
    def test_constant_content_tamper_rejected_after_report_resign(self):
        changed=copy.deepcopy(self.high);name=self.record["generated_constants"][0]["captured_constant"]
        changed["constants"][name][0][0]=.25
        record=copy.deepcopy(self.record);record["captured_graph_sha256"]=digest(changed)
        with self.assertRaisesRegex(ValueError,"generated mask binding"):
            check_common_record(self.resign(record),changed,self.arithmetic)
    def test_wrong_generated_constant_binding_rejected(self):
        record=copy.deepcopy(self.record);record["generated_constants"][0]["captured_constant"]="missing"
        with self.assertRaises(ValueError):check_common_record(self.resign(record),self.high,self.arithmetic)
    def test_name_map_and_arithmetic_source_binding_checked(self):
        for key,value in (("capture_name_map",{}),("arithmetic_source_fx_sha256","0"*64)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                record=copy.deepcopy(self.record);record[key]=value
                check_common_record(self.resign(record),self.high,self.arithmetic)
    def test_source_text_and_policy_identity_checked(self):
        for key,value in (("source_fx_text","different"),("format","poseidon-fx-common-calls-v0"),("public_state_preserved",False)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                record=copy.deepcopy(self.record);record[key]=value
                check_common_record(self.resign(record),self.high,self.arithmetic)
    def test_sdpa_shape_scale_and_dependency_fields_checked(self):
        for field,value in (("scale",1.),("score_shape",[2,3]),("mask_kind","private")):
            with self.subTest(field=field),self.assertRaises(ValueError):
                record=copy.deepcopy(self.record)
                next(e for e in record["entries"] if e["rule"]=="sdpa")["attrs"][field]=value
                check_common_record(self.resign(record),self.high,self.arithmetic)
        record=copy.deepcopy(self.record);record["entries"][0]["normalized_nodes"]=["invented"]
        with self.assertRaises(ValueError):check_common_record(self.resign(record),self.high,self.arithmetic)

class RetainedVersionTests(unittest.TestCase):
    def test_new_policy_is_explicit_and_old_policy_records_unchanged(self):
        signature=[dict(name="x",shape=[1,2,4])]
        for policy in (INDEX_POLICY,OPERATOR_POLICY):
            self.assertNotIn("common_call_rules",policy_record(signature,PROFILE,policy=policy))
        new=policy_record(signature,PROFILE,policy=SHAPE_POLICY)
        self.assertEqual(new["common_call_rules"],"poseidon-fx-common-calls-v1")
        self.assertEqual(intake.source_shape_policy(dict(format=intake.COMMON_FORMAT)),SHAPE_POLICY)
        self.assertEqual(intake.source_shape_policy(dict(format=intake.OPERATOR_SHAPE_FORMAT)),OPERATOR_POLICY)
    def test_old_policy_cannot_use_fused_sdpa_intermediate_shape(self):
        with self.assertRaisesRegex(ValueError,"Static shape unavailable"):
            trace(Projected(),[dict(name="x",shape=[1,2,4])],resource_profile=PROFILE,policy=OPERATOR_POLICY)
    def test_old_arithmetic_alpha_rule_still_rejects(self):
        with self.assertRaisesRegex(ValueError,"alpha=1"):
            normalize(graph(torch.add,kwargs=dict(alpha=2)),resource_profile=PROFILE)
    def test_capabilities_do_not_claim_backend_or_agent_acceptance(self):
        from application_capabilities import input_sources
        entry=input_sources()["trusted_pytorch"]
        self.assertIn(intake.COMMON_FORMAT,entry["formats"])
        self.assertEqual(entry["source_application_entry"],"submit_source")
        for layer in ("compiler_status","encrypted_status","agent_status"):
            self.assertEqual(entry["common_calls"][layer],"unverified")
if __name__=="__main__":unittest.main()
