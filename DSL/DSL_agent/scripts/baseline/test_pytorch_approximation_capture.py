"""Trusted v5 mappings checked against independent original/decomposed math."""
import copy,importlib.util,json,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from benchmark_graph import samples,digest
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as tensor_reference
from operator_reference import evaluate as high
from pytorch_model_capture import capture
from pytorch_source_contract import output_tree
from pytorch_approximation import check_record,FORMAT

FIXTURES=Path(__file__).parent/"cases/trusted-pytorch-approximation-v5"
spec=importlib.util.spec_from_file_location("approximation_fixture",FIXTURES/"models.py")
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
P=dict(method="polynomial",basis="power",domain=[-2.,2.],coefficients=[0.,.5,.125])

def config(defaults=None,modules=None,nodes=None):
    return dict(format=FORMAT,defaults=defaults or {},modules=modules or [],nodes=nodes or [])

class ApproximationCaptureTests(unittest.TestCase):
    def capture(self,module,shape,configuration,outputs=("out",)):
        return capture(module.double().eval(),[dict(name="x",shape=shape)],list(outputs),
            approximations=configuration,resource_profile="general-tensors-v1",
            dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True,
            capture_policy="signature-static-v1",approximation_policy="node-module-default-v1")
    def compare(self,module,shape,configuration,outputs=("out",)):
        original,lowered,binding=self.capture(module,shape,configuration,outputs)
        self.assertEqual((original,lowered,binding),self.capture(module,shape,configuration,outputs))
        check_record(binding["pytorch_capture"]["approximation_selection"],original,configuration)
        for values in samples(lowered,16):
            with torch.no_grad():result=module(torch.from_numpy(values["x"]))
            leaves,tree=output_tree(result,lambda v:isinstance(v,torch.Tensor),list(outputs),
                                    scalar_tensor=lambda v:v.ndim==0)
            self.assertEqual(tree,binding["pytorch_capture"]["output_structure"])
            ideal=high(original,values,ideal=True)
            expected=high(original,values)
            for name,value in zip(outputs,leaves):
                np.testing.assert_allclose(value.detach().numpy().reshape(ideal[name].shape),
                                           ideal[name],atol=1e-12,rtol=1e-12)
                for evaluator in (mathematical,tensor_reference):
                    actual=evaluator(lowered,values,resource_profile="general-tensors-v1")
                    np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
        return original,lowered,binding
    def case(self,name):
        item=json.loads((FIXTURES/(name+".json")).read_text())
        return self.compare(fixture.make(name),item["inputs"][0]["shape"],item["approximations"],item["outputs"])
    def test_module_specific_profiles(self):self.case("modules")
    def test_shared_module_calls_node_override_and_default(self):
        _,_,binding=self.case("nodes")
        rows=binding["pytorch_capture"]["approximation_selection"]["selections"]
        self.assertEqual([r["source"]["kind"] for r in rows],["modules","nodes","defaults"])
    def test_nested_function_owner_scope_and_public_shape_loop(self):
        _,_,binding=self.case("nested")
        self.assertEqual([r["module_paths"] for r in binding["pytorch_capture"]["approximation_selection"]["selections"]],
                         [["left"],["right"]])
    def test_different_functional_activations_and_named_outputs(self):self.case("functions")
    def test_norm_module_rsqrt_components(self):self.case("norms")
    def test_functional_node_selector_third_context(self):
        item=json.loads((FIXTURES/"functions_nodes.json").read_text())
        _,_,binding=self.compare(fixture.make("functions"),[2],item["approximations"],item["outputs"])
        gelu=[r for r in binding["pytorch_capture"]["approximation_selection"]["selections"] if r["function"]=="gelu"]
        self.assertEqual(len(gelu),1)
        self.assertEqual(gelu[0]["source"],dict(kind="nodes",selector=0))

    def test_composite_softmax_components(self):
        _,_,binding=self.case("compound")
        self.assertEqual([r["function"] for r in binding["pytorch_capture"]["approximation_selection"]["selections"]],
                         ["exp","reciprocal","exp","reciprocal"])
    def test_new_function_spellings_three_shapes_each(self):
        for name,fn,role in [
            ("relu",torch.relu,"relu"),("relu_F",F.relu,"relu"),("leaky",lambda x:F.leaky_relu(x,.2),"relu"),
            ("gelu",F.gelu,"gelu"),("silu",F.silu,"silu"),
            ("sigmoid",torch.sigmoid,"sigmoid"),("tanh",torch.tanh,"tanh"),("exp",torch.exp,"exp"),
            ("reciprocal",torch.reciprocal,"reciprocal"),("rsqrt",torch.rsqrt,"rsqrt"),("sqrt",torch.sqrt,"rsqrt"),
            ("method_relu",lambda x:x.relu(),"relu"),("method_sigmoid",lambda x:x.sigmoid(),"sigmoid"),
            ("method_tanh",lambda x:x.tanh(),"tanh"),("method_exp",lambda x:x.exp(),"exp"),
            ("method_reciprocal",lambda x:x.reciprocal(),"reciprocal"),
            ("method_rsqrt",lambda x:x.rsqrt(),"rsqrt"),("method_sqrt",lambda x:x.sqrt(),"rsqrt")]:
            value=(dict(method="taylor",domain=[-2.,2.],degree=2) if role=="exp" else
                   dict(method="goldschmidt",domain=[.5,2.],iterations=1) if role=="reciprocal" else
                   dict(method="newton",domain=[.5,2.],iterations=1) if role=="rsqrt" else P)
            class Model(torch.nn.Module):
                def forward(self,x):
                    result=fn(x*.125+.75)
                    return result.reshape(result.shape)
            for shape in ([2],[1,2],[1,1,2]):
                with self.subTest(name=name,shape=shape):self.compare(Model(),shape,config({role:value}))
    def test_prelu_function_three_shapes(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("alpha",torch.tensor([.2],dtype=torch.float64))
            def forward(self,x):return F.prelu(x,self.alpha)
        for shape in ([2],[1,2],[1,1,2]):
            with self.subTest(shape=shape):self.compare(Model(),shape,config(dict(relu=P)))
    def test_softmax_and_logsoftmax_three_spellings_each(self):
        for log in (False,True):
            for spelling in ("torch","functional","method"):
                class Model(torch.nn.Module):
                    def forward(self,x):
                        y=x*.125
                        if spelling=="torch":return torch.log_softmax(y,-1) if log else torch.softmax(y,-1)
                        if spelling=="functional":return F.log_softmax(y,dim=-1) if log else F.softmax(y,dim=-1)
                        return y.log_softmax(-1) if log else y.softmax(-1)
                second="log" if log else "reciprocal"
                profile=dict(P,domain=[.5,8.]) if log else dict(method="goldschmidt",domain=[.5,8.],iterations=1)
                value=config(dict(exp=dict(method="taylor",domain=[-2.,2.],degree=2),**{second:profile}))
                with self.subTest(log=log,spelling=spelling):self.compare(Model(),[2],value)
    def test_scope_conflict_unused_and_wrong_component_rejected(self):
        item=json.loads((FIXTURES/"nodes.json").read_text())
        for edit in (lambda v:v["nodes"].append(dict(name="missing",function="relu",profile=P)),
                     lambda v:v["modules"].append(dict(path="missing",function="relu",profile=P)),
                     lambda v:v["defaults"].update(gelu=P)):
            bad=copy.deepcopy(item["approximations"]);edit(bad)
            with self.assertRaisesRegex(ValueError,"Unmatched"):self.capture(fixture.make("nodes"),[2],bad)
        with self.assertRaisesRegex(ValueError,"Explicit approximation"):
            self.capture(fixture.make("norms"),[1,2],config(dict(relu=P)))
    def test_config_change_changes_graph_and_request(self):
        from component_contract import prepare_task
        item=json.loads((FIXTURES/"nodes.json").read_text())
        changed=copy.deepcopy(item["approximations"])
        changed["nodes"][0]["profile"]["coefficients"][1]=.3125
        a,low_a,_=self.capture(fixture.make("nodes"),[2],item["approximations"])
        b,low_b,_=self.capture(fixture.make("nodes"),[2],changed)
        self.assertNotEqual(digest(a),digest(b))
        options=dict(generation_guidance="explicit-v10",compiler_configuration="poseidon-cpu-eva-w45-v1",chunk_period=4)
        self.assertNotEqual(prepare_task(low_a,options)["request_id"],prepare_task(low_b,options)["request_id"])
    def test_opt_in_and_old_graph_hash_preserved(self):
        module=torch.nn.Sequential(torch.nn.ReLU())
        args=([dict(name="x",shape=[2])],["out"])
        old=capture(module,*args,approximations=dict(relu=P),resource_profile="general-tensors-v1")
        old_explicit=capture(module,*args,approximations=dict(relu=P),resource_profile="general-tensors-v1",
                             approximation_policy="operator-defaults-v1")
        self.assertEqual(old,old_explicit)
        scoped=self.capture(module,[2],config(dict(relu=P)))
        self.assertEqual(old[:2],scoped[:2])
        self.assertNotIn("approximation_selection",old[2]["pytorch_capture"])
        class Functional(torch.nn.Module):
            def forward(self,x):return F.relu(x)
        with self.assertRaisesRegex(ValueError,"Unsupported PyTorch"):
            capture(Functional(),*args,approximations=dict(relu=P))
    def test_side_effect_or_unconfigured_approximation_rejected(self):
        with self.assertRaises(ValueError):
            self.capture(torch.nn.Sequential(torch.nn.ReLU()),[2],config())
        class Inplace(torch.nn.Module):
            def forward(self,x):return F.relu(x,inplace=True)
        with self.assertRaisesRegex(ValueError,"In-place"):
            self.capture(Inplace(),[2],config(dict(relu=P)))
        class ApproximateGELU(torch.nn.Module):
            def forward(self,x):return F.gelu(x,approximate="tanh")
        with self.assertRaisesRegex(ValueError,"Exact GELU"):
            self.capture(ApproximateGELU(),[2],config(dict(gelu=P)))
    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            capture(torch.nn.ReLU(),[dict(name="x",shape=[2])],["out"],approximation_policy="auto")

if __name__=="__main__":unittest.main()
