"""Shape controls use declared metadata, scalar leaves keep their source ABI."""
import copy, unittest
import numpy as np
import torch
from benchmark_graph import samples, digest
from benchmark_math import evaluate as mathematics
from benchmark_torch import evaluate as independent
from pytorch_model_capture import capture
from pytorch_source_contract import output_tree, restore_outputs, validate_output_tree
from pytorch_static_trace import POLICY, policy_record

class StaticCaptureTests(unittest.TestCase):
    def compare(self, module, shapes, outputs=("out",), *, profile="legacy-v1"):
        signature=[dict(name="input"+str(i),shape=list(s)) for i,s in enumerate(shapes)]
        options=dict(dtype_policy="strict-float64-v1",preserve_output_structure=True,
                     record_state_sources=True,capture_policy=POLICY,resource_profile=profile)
        original, lowered, binding=capture(module,signature,list(outputs),**options)
        self.assertEqual((original,lowered,binding),capture(module,signature,list(outputs),**options))
        metadata=binding["pytorch_capture"]
        self.assertEqual(metadata["static_capture"],policy_record(signature,profile))
        tree=metadata["output_structure"]
        validate_output_tree(tree,list(outputs),allow_scalars=True)
        for case in samples(lowered,16):
            raw=module(*(torch.from_numpy(case[i["name"]]) for i in signature))
            leaves,actual_tree=output_tree(raw,lambda v:isinstance(v,torch.Tensor),list(outputs),
                                          scalar_tensor=lambda v:v.ndim==0)
            self.assertEqual(actual_tree,tree)
            expected={name:(value.detach().numpy().reshape(1) if value.ndim==0
                           else value.detach().numpy()) for name,value in zip(outputs,leaves)}
            for evaluate in (mathematics,independent):
                actual=evaluate(lowered,case,resource_profile=profile)
                for name in outputs:
                    np.testing.assert_allclose(actual[name],expected[name],atol=1e-12,rtol=1e-12)
                restored=restore_outputs({k:v.tolist() for k,v in actual.items()},tree)
                if tree.get("shape")==[]:self.assertIsInstance(restored,float)
        return original,lowered,binding

    def test_signature_loop_shape_three_contexts(self):
        for variant,shape in enumerate(([3],[2,3],[1,2,2])):
            class Model(torch.nn.Module):
                def forward(self,x):
                    n=x.shape[-1] if variant==0 else x.size(-1) if variant==1 else x.numel()
                    result=x
                    for _ in range(n):result=result+x*.125
                    return result
            self.compare(Model(),[shape])

    def test_intermediate_shapes_and_module_result(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(3,2).double()
            def forward(self,x):
                y=self.linear(x)
                for _ in range(y.size(-1)):y=y+x.sum(-1,keepdim=True)*.125
                return y
        self.compare(Model(),[[1,3]])

    def test_root_linear_parameters_do_not_become_control_values(self):
        self.compare(torch.nn.Linear(3,2).double(),[[1,3]])

    def test_public_weight_budget_is_separate_from_intermediate_tensor_budget(self):
        self.compare(torch.nn.Linear(32,32).double(),[[1,32]])

    def test_reshape_shape_arithmetic_and_public_branch(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                y=x.reshape(x.shape[0],-1)
                if y.dim()==2 and len(y)==2 and y.numel()==8:
                    for _ in range(y.shape[1]-1):y=y+x.reshape(2,4)*.25
                return y
        self.compare(Model(),[[2,2,2]])

    def test_static_shape_values_are_not_tensor_output_leaves(self):
        class Model(torch.nn.Module):
            def forward(self,x):return dict(tensor=x+x,shape=x.shape)
        with self.assertRaisesRegex(ValueError,"Output"):self.compare(Model(),[[2]],("a","b"))

    def test_direct_scalars_three_contexts(self):
        for variant in range(3):
            class Model(torch.nn.Module):
                def forward(self,x):
                    if variant==0:return x.sum()
                    if variant==1:return (x*x).mean()
                    return torch.dot(x,x+.25)
            self.compare(Model(),[[3]])

    def test_nested_scalar_vector_and_matrix_output(self):
        class Model(torch.nn.Module):
            def forward(self,x,y):
                return {"score":torch.dot(x,y),"features":(x+y,[x.reshape(1,3)])}
        _,_,binding=self.compare(Model(),[[3],[3]],("score","features","matrix"))
        tree=binding["pytorch_capture"]["output_structure"]
        actual=restore_outputs(dict(score=[.5],features=[1.,2.,3.],matrix=[[1.,2.,3.]]),tree)
        self.assertEqual(actual,dict(score=.5,features=([1.,2.,3.],[[[1.,2.,3.]]])))

    def test_scalar_intermediate_shape_query(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                value=x.sum()
                if value.ndim==0 and value.numel()==1:return value*.5
                return value
        self.compare(Model(),[[3]])

    def test_legacy_scalar_rejection_is_preserved(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x.mean()
        with self.assertRaisesRegex(ValueError,"Scalar outputs"):
            capture(Model(),[dict(name="x",shape=[3])],["out"],
                    dtype_policy="strict-float64-v1",preserve_output_structure=True)

    def test_legacy_public_shape_loop_rejection_is_preserved(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                y=x
                for _ in range(x.size(-1)):y=y+x
                return y
        with self.assertRaises(Exception):
            capture(Model(),[dict(name="x",shape=[2])],["out"])

    def test_non_shape_legacy_graph_and_request_identity_unchanged(self):
        from component_contract import prepare_task
        class Model(torch.nn.Module):
            def forward(self,x):return x*x+.25
        signature=[dict(name="x",shape=[3])]
        options=dict(dtype_policy="strict-float64-v1",preserve_output_structure=True)
        old=capture(Model(),signature,["out"],**options)
        new=capture(Model(),signature,["out"],capture_policy=POLICY,**options)
        self.assertEqual(old[:2],new[:2])
        for guidance in ("explicit-v9","explicit-v10"):
            kwargs=dict(generation_guidance=guidance)
            self.assertEqual(prepare_task(old[1],kwargs)["request_id"],
                             prepare_task(new[1],kwargs)["request_id"])

    def test_data_dependent_controls_three_rejections(self):
        for variant in range(3):
            class Model(torch.nn.Module):
                def forward(self,x):
                    if variant==0:
                        if x.sum()>0:return x+x
                    elif variant==1:
                        for _ in range(int(x[0])):x=x+x
                    else:
                        for value in x:x=x+value
                    return x
            with self.assertRaises(Exception):self.compare(Model(),[[2]])

    def test_unknown_metadata_and_encrypted_axis_rejected(self):
        class Unknown(torch.nn.Module):
            def forward(self,x):
                y=torch.sin(x)
                for _ in range(y.shape[0]):y=y+y
                return y
        with self.assertRaisesRegex(ValueError,"Static shape unavailable"):
            self.compare(Unknown(),[[2]])
        class Axis(torch.nn.Module):
            def forward(self,x):return x+x.size(x[0])
        with self.assertRaisesRegex(ValueError,"Static size axis"):self.compare(Axis(),[[2]])

    def test_expansion_budget_stops_before_real_tensor_execution(self):
        class Model(torch.nn.Module):
            def forward(self,x):
                for _ in range(10000):x=x+x
                return x
        with self.assertRaisesRegex(ValueError,"Static FX expansion budget"):self.compare(Model(),[[2]])

    def test_random_and_state_mutation_still_rejected(self):
        class Random(torch.nn.Module):
            def forward(self,x):return x+torch.rand_like(x)
        with self.assertRaisesRegex(ValueError,"Random forward"):self.compare(Random(),[[2]])
        class State(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("value",torch.zeros(2,dtype=torch.float64))
            def forward(self,x):
                self.value.add_(1)
                return x+self.value
        with self.assertRaisesRegex(ValueError,"State-mutating"):self.compare(State(),[[2]])

    def test_shape_budget_and_extra_input_rejected(self):
        class Huge(torch.nn.Module):
            def forward(self,x):
                y=x.expand(257,2)
                return y+y
        with self.assertRaisesRegex(ValueError,"tensor budget"):self.compare(Huge(),[[1,2]])
        class One(torch.nn.Module):
            def forward(self,x):return x+x
        with self.assertRaisesRegex(ValueError,"Input declaration count"):self.compare(One(),[[2],[2]])

    def test_policy_explicit_and_metadata_required(self):
        class Model(torch.nn.Module):
            def forward(self,x):return x+x
        with self.assertRaisesRegex(ValueError,"Capture shape policy"):
            capture(Model(),[dict(name="x",shape=[2])],["out"],capture_policy="automatic")
        with self.assertRaisesRegex(ValueError,"requires output metadata"):
            capture(Model(),[dict(name="x",shape=[2])],["out"],capture_policy=POLICY)

if __name__=="__main__":unittest.main()
