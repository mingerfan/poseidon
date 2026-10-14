import copy,hashlib,json,importlib.util,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from benchmark_graph import digest
from pytorch_static_trace import trace,ALIAS_POLICY,INDEX_POLICY
from pytorch_arithmetic_normalization import normalize
from pytorch_model_capture import capture
from pytorch_alias_composition import AliasError,verify
from operator_graph import evaluate_reference
from benchmark_math import evaluate as base_reference

PROFILE="general-tensors-v1"
RATIO=dict(method="newton",domain=[.1,2.],iterations=1)
POLY=dict(method="polynomial",basis="power",domain=[-2.,2.],coefficients=[.125,.5,.125])
MEASUREMENTS=[]
def freeze_state(module):
    return {k:v.detach().clone() for k,v in module.state_dict().items()}
def equal_state(module,before):
    return before.keys()==module.state_dict().keys() and all(torch.equal(v,module.state_dict()[k]) for k,v in before.items())
def flatten(value):
    if isinstance(value,torch.Tensor):return [value.detach().cpu().numpy().copy()]
    if isinstance(value,dict):return [v for item in value.values() for v in flatten(item)]
    if isinstance(value,(tuple,list)):return [v for item in value for v in flatten(item)]
    raise ValueError("Unsupported test output")
def probes(specs):
    for kind in range(16):
        result={}
        for j,s in enumerate(specs):
            size=int(np.prod(s["shape"]))
            if kind==0:a=np.zeros(size)
            elif kind==1:a=np.resize([-.2,.1,-.1,.2],size)
            elif kind==2:a=np.linspace(-.24,.24,size)
            else:a=np.random.default_rng(601000+kind*100+j).uniform(-.24,.24,size)
            result[s["name"]]=a.reshape(s["shape"])
        yield result
def staged(module,specs,names,approx=None,profile=PROFILE):
    from pytorch_ordered_capture import capture_module
    result=capture_module(module,specs,names,approximations=approx or {},resource_profile=profile)
    return result,result[2]["pytorch_capture"]["ordered_alias_capture"]["capture_adapter"]["arithmetic"]
def qualify(test,module,specs,names,approx=None,tag="case",expected=None):
    before=freeze_state(module)
    (high,lowered,binding),record=staged(module,specs,names,approx)
    test.assertTrue(equal_state(module,before))
    alias=binding["pytorch_capture"]["ordered_alias_capture"]
    verify(alias["normalization"],expected_source_request_sha256=alias["normalization"]["source_request_sha256"])
    count=0;max_ideal=0.;max_lower=0.;approx_error=0.
    for inputs in probes(specs):
        # Separate fresh input storage for each original or normalized reference.
        want=flatten(module(*(torch.from_numpy(inputs[s["name"]].copy()) for s in specs)))
        ideal=evaluate_reference(high,{k:v.copy() for k,v in inputs.items()},ideal=True)
        reference=evaluate_reference(high,{k:v.copy() for k,v in inputs.items()})
        actual=base_reference(lowered,{k:v.copy() for k,v in inputs.items()},resource_profile=PROFILE)
        test.assertEqual(list(ideal),names)
        for name,v in zip(names,want):
            w=v.reshape(1) if v.ndim==0 else v
            np.testing.assert_allclose(ideal[name],w,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(actual[name],reference[name],atol=1e-12,rtol=1e-12)
            max_ideal=max(max_ideal,float(np.max(np.abs(ideal[name]-w))))
            max_lower=max(max_lower,float(np.max(np.abs(actual[name]-reference[name]))))
            approx_error=max(approx_error,float(np.max(np.abs(reference[name]-w))))
            count+=w.size
        test.assertTrue(equal_state(module,before))
    if expected:expected(alias,high,lowered,binding)
    MEASUREMENTS.append(dict(context=tag,groups=16,values=count,original_torch_vs_independent_ideal=max_ideal,
        independent_approximate_vs_decomposition=max_lower,approximation_error=approx_error,
        source_graph_sha256=alias["source_graph_sha256"],graph_sha256=digest(high),
        mutation_events=sum(e["rule"]=="mutation_ssa" for e in alias["normalization"]["events"])))
    return high,lowered,binding
class Composed(torch.nn.Module):
    def __init__(self,group,variant):
        super().__init__();self.group=group;self.variant=variant
        if group=="linear":self.op=torch.nn.Linear(3,2)
        elif group=="conv1d":self.op=torch.nn.Conv1d(1,1,2)
        elif group=="conv2d":self.op=torch.nn.Conv2d(1,1,2)
        elif group=="layer_norm":self.op=torch.nn.LayerNorm(2,eps=.1)
        elif group=="group_norm":self.op=torch.nn.GroupNorm(2,2,eps=.1)
        elif group=="batch_norm":self.op=torch.nn.BatchNorm1d(2,eps=.1)
        generator=torch.Generator().manual_seed(8833)
        with torch.no_grad():
            for p in self.parameters():p.copy_(torch.rand(p.shape,generator=generator)*.25-.125)
        self.double().eval()
    def forward(self,x,b=None):
        if self.group=="mean":y=x+x.mean(-1,keepdim=True)
        elif self.group=="matmul":y=x@b
        elif self.group=="join":y=torch.cat((x,x.square()),dim=0)
        else:y=self.op(x)
        previous=y.square()
        if self.variant==0:result=y.add_(.125)
        elif self.variant==1:result=y[...,0:1].mul_(.25)
        else:
            result=y.clone()
            result.sub_(.125)
        return {"current":y,"previous":previous,"result":result}
class SingleComposed(Composed):
    def forward(self,x):return super().forward(x,None)
class CaptureTests(unittest.TestCase):
    def test_registered_groups_three_alias_contexts(self):
        shapes={"linear":[2,3],"conv1d":[1,1,4],"conv2d":[1,1,3,3],
            "layer_norm":[2,2],"group_norm":[1,2,2],"batch_norm":[1,2,3],
            "mean":[2,3],"matmul":[2,2],"join":[2,2]}
        for group,shape in shapes.items():
            for variant in range(3):
                with self.subTest(group=group,variant=variant):
                    specs=[dict(name="x",shape=shape)]
                    if group=="matmul":specs.append(dict(name="b",shape=[2,2]))
                    approximations={"rsqrt":RATIO} if group in ("layer_norm","group_norm") else {}
                    qualify(self,(Composed if group=="matmul" else SingleComposed)(group,variant),specs,["current","previous","result"],
                        approximations,tag=group+":"+str(variant))
    def test_split_returns_and_mutates_shared_view(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                left,right=x.split([2,2],dim=1)
                previous=left.square()
                right.add_(.25)
                return (x,[left,right],{"prior":previous})
        qualify(self,Model(),[dict(name="x",shape=[2,4])],["whole","left","right","prior"],tag="split")
    def test_transposed_clone_and_contiguous_write(self):
        for kind in ("view","reshape","clone","contiguous"):
            class Model(torch.nn.Module):
                def forward(self,x):
                    y=x.transpose(0,1)
                    if kind=="view":z=y.view(3,2)
                    elif kind=="reshape":z=y.reshape(6)
                    elif kind=="clone":z=y.clone()
                    else:z=y.contiguous()
                    previous=x.square()
                    z.add_(.125)
                    return x,z,previous
            qualify(self,Model(),[dict(name="x",shape=[2,3])],["x","z","previous"],tag="transpose_"+kind)
    def test_input_side_effect_is_preserved(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                before=x.square();x.mul_(.5);return {"x":x,"before":before}
        qualify(self,Model(),[dict(name="x",shape=[2,3])],["x","before"],tag="input_write")
    def test_add_alpha_and_self_alias(self):
        class Model(torch.nn.Module):
            def forward(self,x,b):
                y=x.clone();y.add_(b,alpha=.25);y.sub_(y,alpha=.125);return y
        qualify(self,Model(),[dict(name="x",shape=[2,2]),dict(name="b",shape=[1,2])],["y"],tag="alpha_self")
    def test_public_left_arithmetic_keeps_effect_order(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                y=.25-x;before=y.square();y.add_(.125);return before,y
        qualify(self,Model(),[dict(name="x",shape=[2,2])],["before","y"],tag="public_left")
    def test_cipher_copy_broadcast(self):
        class Model(torch.nn.Module):
            def forward(self,x,b):
                before=x.square();x.copy_(b);return x,before
        qualify(self,Model(),[dict(name="x",shape=[2,3]),dict(name="b",shape=[1,3])],["x","before"],tag="copy")
    def test_advanced_index_is_an_independent_result(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                y=x[:,[0,2]];y.add_(.25);return x,y
        qualify(self,Model(),[dict(name="x",shape=[2,3])],["x","y"],tag="advanced_index")
    def test_explicit_inplace_relu(self):
        for source in ("module","functional","method","function"):
            class Model(torch.nn.Module):
                def __init__(self):super().__init__();self.relu=torch.nn.ReLU(inplace=True)
                def forward(self,x):
                    y=x+.125;before=y.square()
                    if source=="module":z=self.relu(y)
                    elif source=="functional":z=F.relu(y,inplace=True)
                    elif source=="method":z=y.relu_()
                    else:z=torch.relu_(y)
                    return y,before,z
            qualify(self,Model(),[dict(name="x",shape=[2,2])],["y","before","z"],
                {"relu":POLY},tag="relu_"+source)
    def test_default_capture_stays_identical(self):
        spec=importlib.util.spec_from_file_location("original_capture","/originals/pytorch_model_capture.py")
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        class Model(torch.nn.Module):
            def forward(self,x):return (x+.25).square()
        inputs=[dict(name="x",shape=[2,2])];kwargs=dict(resource_profile=PROFILE,
            dtype_policy="strict-float64-v1",preserve_output_structure=True,
            record_state_sources=True,capture_policy=INDEX_POLICY)
        a=original.capture(Model(),inputs,["y"],**kwargs);b=capture(Model(),inputs,["y"],**kwargs)
        self.assertEqual(a,b)
        self.assertNotIn("ordered_alias_capture",b[2]["pytorch_capture"])
    def test_unmarked_graph_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"source policy binding"):
            capture(torch.fx.symbolic_trace(torch.nn.Identity()),[dict(name="x",shape=[2,2])],["y"],
                resource_profile=PROFILE,capture_policy=INDEX_POLICY,preserve_output_structure=True,capture_aliases=True)
    def test_inplace_requires_explicit_approximation(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.relu_()
        with self.assertRaisesRegex(ValueError,"Explicit approximation"):
            staged(Model(),[dict(name="x",shape=[2,2])],["y"])
    def test_public_weight_mutation_rejected_without_mutating_caller(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("w",torch.ones(2,2,dtype=torch.float64))
            def forward(self,x):self.w.add_(.25);return x+self.w
        module=Model();before=freeze_state(module)
        with self.assertRaisesRegex(ValueError,"State-mutating"):
            staged(module,[dict(name="x",shape=[2,2])],["y"])
        self.assertTrue(equal_state(module,before))
    def test_overlapping_partial_views_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):x[:,1:3].add_(x[:,0:2]);return x
        with self.assertRaises((ValueError,RuntimeError,AliasError)):
            staged(Model(),[dict(name="x",shape=[2,3])],["y"])
    def test_repeated_expand_write_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):x.expand(2,3).add_(.25);return x
        with self.assertRaises((ValueError,RuntimeError,AliasError)):
            staged(Model(),[dict(name="x",shape=[1,3])],["y"])
    def test_incompatible_view_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.transpose(0,1).view(6)
        with self.assertRaises((ValueError,RuntimeError,AliasError)):
            staged(Model(),[dict(name="x",shape=[2,3])],["y"])
    def test_dynamic_branch_is_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                if x.sum()>0:return x.square()
                return x
        with self.assertRaises(Exception):
            staged(Model(),[dict(name="x",shape=[2,2])],["y"])
    def test_nondefault_clone_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.clone(memory_format=torch.contiguous_format)
        with self.assertRaisesRegex(ValueError,"clone/detach"):
            staged(Model(),[dict(name="x",shape=[2,2])],["y"])
    def test_original_graph_not_affected_by_shapeprop(self):
        class Model(torch.nn.Module):
            def forward(self,x):x.add_(.25);return x
        graph=trace(Model(),[dict(name="x",shape=[2,2])],resource_profile=PROFILE,policy=ALIAS_POLICY)
        text=str(graph.graph);state=freeze_state(graph)
        capture(graph,[dict(name="x",shape=[2,2])],["y"],resource_profile=PROFILE,
            capture_policy=INDEX_POLICY,preserve_output_structure=True,capture_aliases=True)
        self.assertEqual(str(graph.graph),text);self.assertTrue(equal_state(graph,state))

    def test_rank_32_alias_carrier(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                before=x.square();x.add_(.125);return x,before
        qualify(self,Model(),[dict(name="x",shape=[1]*31+[2])],["x","before"],tag="rank32")
    def test_public_state_sources_are_bound(self):
        module=SingleComposed("linear",0)
        inputs=[dict(name="x",shape=[2,3])]
        (high,low,binding),_=staged(module,inputs,["current","previous","result"])
        record=binding["pytorch_capture"]["ordered_alias_capture"]
        names=binding["pytorch_capture"]["state_constant_sources"]
        self.assertEqual(set(names.values()),{"op.weight","op.bias"})
        for name,state in names.items():
            np.testing.assert_array_equal(np.asarray(high["constants"][name]),module.state_dict()[state].numpy())
        self.assertEqual(record["source_capture_policy"]["version"],ALIAS_POLICY)
        changed=copy.deepcopy(module)
        with torch.no_grad():changed.op.weight[0,0]+=.03125
        (new,_,new_binding),_=staged(changed,inputs,["current","previous","result"])
        self.assertNotEqual(digest(high),digest(new));self.assertNotEqual(binding["binding_sha256"],new_binding["binding_sha256"])
    def test_nonrepeating_large_public_weight_alias_map(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.op=torch.nn.Linear(160,64,bias=False)
                with torch.no_grad():
                    self.op.weight.copy_(torch.arange(10240).reshape(64,160)/65536-.0625)
                self.double()
            def forward(self,x):
                y=self.op(x);before=y.square();y.add_(.125);return y,before
        # Admission and alias storage maps, not a physical resource or FHE claim.
        model=Model()
        (graph,_,binding),_=staged(model,[dict(name="x",shape=[1,160])],["y","before"])
        self.assertEqual(sum(np.asarray(v).size for v in graph["constants"].values()),10241)
        self.assertIn("op.weight",binding["pytorch_capture"]["state_constant_sources"].values())
        values={"x":np.linspace(-.2,.2,160).reshape(1,160)}
        got=evaluate_reference(graph,values);want=flatten(model(torch.from_numpy(values["x"].copy())))
        for name,w in zip(["y","before"],want):np.testing.assert_allclose(got[name],w,atol=1e-12,rtol=1e-12)
    def test_mutation_source_metadata_is_immutable(self):
        class Model(torch.nn.Module):
            def forward(self,x):x.add_(.25);return x.square()
        (graph,low,binding),_=staged(Model(),[dict(name="x",shape=[2,2])],["y"])
        alias=binding["pytorch_capture"]["ordered_alias_capture"]
        record=alias["normalization"]
        self.assertNotIn("pytorch_capture",record["decomposition"])
        encoded=json.dumps(binding,allow_nan=False)
        self.assertEqual(binding["binding_sha256"],digest({k:v for k,v in binding.items() if k!="binding_sha256"}))
        record["events"][0]["rule"]="tampered"
        with self.assertRaises(AliasError):verify(record,expected_source_request_sha256=record["source_request_sha256"])
    def test_alias_path_cannot_bypass_original_fx_budget(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                for _ in range(65):x=x.square()
                return x
        with self.assertRaisesRegex(ValueError,"Static FX expansion budget"):
            staged(Model(),[dict(name="x",shape=[1])],["y"],profile="legacy-v1")
    def test_capture_rejects_tampered_source_policy(self):
        class Model(torch.nn.Module):
            def forward(self,x):x.add_(.125);return x
        specs=[dict(name="x",shape=[2,2])]
        fx=trace(Model(),specs,resource_profile=PROFILE,policy=ALIAS_POLICY)
        fx.meta["ordered_alias_capture_policy"]["input_signature_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"source policy binding"):
            capture(fx,specs,["y"],resource_profile=PROFILE,capture_policy=INDEX_POLICY,
                preserve_output_structure=True,capture_aliases=True)
    def test_unsupported_copy_signature_rejected(self):
        class Model(torch.nn.Module):
            def forward(self,x,b):x.copy_(b,non_blocking=True);return x
        with self.assertRaisesRegex(ValueError,"CPU blocking"):
            staged(Model(),[dict(name="x",shape=[2,2]),dict(name="b",shape=[2,2])],["y"])

    def test_strict_dtype_checked_before_any_conversion(self):
        from pytorch_ordered_capture import capture_module
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Flatten())
        before=freeze_state(model)
        with self.assertRaisesRegex(ValueError,"Strict float64"):
            capture_module(model,[dict(name="x",shape=[1,2])],["y"])
        self.assertTrue(equal_state(model,before))
    def test_promotion_preserves_original_dtype_identity(self):
        from pytorch_ordered_capture import capture_module
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Flatten())
        before=freeze_state(model)
        high,low,binding=capture_module(model,[dict(name="x",shape=[1,2])],["y"],
                                       dtype_policy="convert-float64-v1")
        info=binding["pytorch_capture"]
        self.assertEqual(set(info["source_state_dtypes"].values()),{"float32"})
        self.assertFalse(info["source_precision_execution_compared"])
        self.assertTrue(equal_state(model,before))
    def test_nonfinite_public_weights_rejected(self):
        from pytorch_ordered_capture import capture_module
        module=torch.nn.Sequential(torch.nn.Linear(2,2)).double()
        with torch.no_grad():module[0].weight[0,0]=float("nan")
        with self.assertRaises((ValueError,AliasError)):
            capture_module(module,[dict(name="x",shape=[1,2])],["y"])
