"""Actual FX normalization, public-state replay and source semantics."""
import copy,math,unittest
import numpy as np
import torch
from torch.fx.passes.shape_prop import ShapeProp
from benchmark_graph import digest
from pytorch_static_trace import ALIAS_POLICY
from gather_fx_normalization import POLICY,normalize,check,public_arrays,arguments
from pytorch_fused_matrix_normalization import inventory

class Gather(torch.nn.Module):
    def __init__(self,index,axis,method=False):
        super().__init__();self.register_buffer("indices",torch.tensor(index,dtype=torch.int64))
        self.axis=axis;self.method=method
    def forward(self,x):
        if self.method:return x.gather(dim=self.axis,index=self.indices)
        return torch.gather(x,self.axis,self.indices)

def traced(module,shape):
    graph=torch.fx.symbolic_trace(module)
    ShapeProp(graph).propagate(torch.zeros(shape,dtype=torch.float64))
    for node in graph.graph.nodes:
        info=node.meta.get("tensor_meta")
        if node.op!="output" and info is not None and hasattr(info,"shape"):
            node.meta["poseidon_static_tensor"]=dict(policy=ALIAS_POLICY,shape=list(info.shape),
                dtype=str(info.dtype).removeprefix("torch."))
    graph.meta["tensor_gather_source_policy"]=POLICY
    return graph

class GatherFXTests(unittest.TestCase):
    def context(self,shape,index,axis,method,profile):
        original=traced(Gather(index,axis,method),shape)
        before=inventory(original);state=public_arrays(original)
        result,record=normalize(original,[dict(name="x",shape=shape)],resource_profile=profile)
        self.assertEqual(inventory(original),before)
        self.assertEqual(check(record,[dict(name="x",shape=shape)],state,resource_profile=profile),inventory(result))
        for seed in range(3):
            x=torch.from_numpy(np.random.default_rng(seed).uniform(-.25,.25,shape).astype(np.float64))
            torch.testing.assert_close(result(x),original(x),rtol=1e-12,atol=1e-12)
        self.assertFalse(any(n.target in(torch.gather,"gather")for n in result.graph.nodes))
        for name,value in state.items():np.testing.assert_array_equal(public_arrays(result)[name],value)
    def test_vector_and_repetition(self):self.context([5],[4,0,4,2],0,False,"legacy-v1")
    def test_different_row_indices_and_method_kwargs(self):self.context([2,4],[[3,0,3],[1,2,0]],1,True,"legacy-v1")
    def test_negative_middle_axis_and_partial_dimension(self):self.context([2,3,2],[[[2],[0]],[[1],[2]]],-2,False,"legacy-v1")
    def test_rank32(self):
        index=[[2,0],[1,2]]
        for _ in range(30):index=[index]
        self.context([1]*30+[2,3],index,-1,True,"general-tensors-v1")
    def test_tampered_route_rejected_even_after_hash_rebinding(self):
        g=traced(Gather([[3,0],[1,2]],1),[2,4]);_,record=normalize(g,[dict(name="x",shape=[2,4])],resource_profile="legacy-v1")
        bad=copy.deepcopy(record);bad["entries"][0]["plan"]["steps"][1]["attrs"]["indices"][0]=1
        bad.pop("binding");bad["binding"]=digest(bad)
        with self.assertRaises(ValueError):check(bad,[dict(name="x",shape=[2,4])],public_arrays(g),resource_profile="legacy-v1")
    def test_changed_public_state_rejected(self):
        g=traced(Gather([0,1],0),[3]);_,record=normalize(g,[dict(name="x",shape=[3])],resource_profile="legacy-v1")
        arrays=public_arrays(g);arrays["indices"][0]=2
        with self.assertRaises(ValueError):check(record,[dict(name="x",shape=[3])],arrays,resource_profile="legacy-v1")
    def test_missing_explicit_policy_rejected(self):
        g=traced(Gather([0,1],0),[3]);g.meta={}
        with self.assertRaises(ValueError):normalize(g,[dict(name="x",shape=[3])],resource_profile="legacy-v1")
    def test_signature_restrictions(self):
        for args,kwargs in (((0,1,2,False),{}),((0,1,2),dict(dim=1)),((0,1,2),dict(sparse_grad=True)),
                            ((0,1,2),dict(out=0)),((0,True,2),{})):
            with self.assertRaises(ValueError):arguments(args,kwargs)
if __name__=="__main__":unittest.main(verbosity=2)
