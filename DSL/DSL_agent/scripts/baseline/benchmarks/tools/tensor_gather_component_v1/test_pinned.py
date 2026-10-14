"""Pinned Torch gather and independent references of the existing core decomposition."""
import copy,itertools,math,unittest
import numpy as np
import torch
from tensor_gather_plan import plan
from operator_graph import FORMAT,decompose
from benchmark_math import evaluate as math_reference
from benchmark_torch import evaluate as torch_reference

def contextual_models():
    nested=[[2,0],[1,2]]
    for _ in range(30):nested=[nested]
    return [
        ("vector",[5],[4,0,4,2],0,"legacy-v1"),
        ("per_row",[2,4],[[3,0,3],[1,2,0]],1,"legacy-v1"),
        ("middle",[2,3,2],[[[2],[0]],[[1],[2]]],-2,"legacy-v1"),
        ("partial_rows",[3,4],[[3,0],[2,1]],1,"legacy-v1"),
        ("last_axis",[2,2,3],[[[2,1],[0,2]],[[1,0],[2,2]]],2,"legacy-v1"),
        ("rank32",[1]*30+[2,3],nested,-1,"general-tensors-v1"),
    ]

def high_model(shape, planned):
    nodes=[];previous="x"
    for i,step in enumerate(planned["steps"]):
        name="v"+str(i)
        nodes.append(dict(id="g"+str(i),op=step["op"],inputs=[previous],
                          attrs=copy.deepcopy(step["attrs"]),outputs=[name]))
        previous=name
    return dict(format=FORMAT,id="tensor_gather_normalized",inputs=[dict(name="x",shape=shape)],
                constants={},nodes=nodes,outputs=[dict(name="out",value=previous)])

class PinnedGatherTests(unittest.TestCase):
    def context(self, shape, indices, axis, profile):
        index=np.asarray(indices,dtype=np.int64)
        source=(np.arange(math.prod(shape),dtype=np.float64)*.375-.875).reshape(shape)
        # Independent mathematical expected result reads original coordinates.
        expected=np.empty(index.shape,dtype=np.float64)
        for coordinate in np.ndindex(index.shape):
            origin=list(coordinate);origin[axis%len(shape)]=int(index[coordinate])
            expected[coordinate]=source[tuple(origin)]
        tensor=torch.from_numpy(source.copy());keys=torch.from_numpy(index.copy())
        actual=torch.gather(tensor,axis,keys).numpy()
        method=tensor.gather(axis,keys).numpy()
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(method,expected)
        planned=plan(shape,indices,axis,resource_profile=profile)
        core,binding=decompose(high_model(shape,planned),resource_profile=profile)
        self.assertEqual(binding["original_sha256"],__import__("benchmark_graph").digest(high_model(shape,planned)))
        self.assertTrue(binding["no_dsl_answer"])
        for reference in (math_reference,torch_reference):
            output=reference(core,dict(x=source))["out"]
            np.testing.assert_allclose(output,expected,rtol=1e-12,atol=1e-12)

    def test_pinned_gather_requires_long_indices(self):
        with self.assertRaises(RuntimeError):
            torch.gather(torch.arange(3,dtype=torch.float64),0,torch.tensor([0,1],dtype=torch.int32))

    def test_pinned_gather_rejects_negative_index(self):
        with self.assertRaises(RuntimeError):
            torch.gather(torch.arange(3,dtype=torch.float64),0,torch.tensor([-1],dtype=torch.int64))

for name,shape,indices,axis,profile in contextual_models():
    def test(self,shape=shape,indices=indices,axis=axis,profile=profile):
        self.context(shape,indices,axis,profile)
    setattr(PinnedGatherTests,"test_"+name,test)
if __name__=="__main__":unittest.main(verbosity=2)
