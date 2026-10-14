"""Capacity split: independent references, chunk routing and unchanged limits."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,samples
from benchmark_math import evaluate as math_reference
from benchmark_torch import evaluate as torch_reference
from component_contract import prepare_task,reconstruct_request
from unified_graph_lowering import candidate_source
from unified_capacity_lowering import split_tensor
from test_unified_capacity_linear import interpreted_output


def model(shape,axis,sections):
    outputs=["part_"+str(i) for i in range(len(sections))]
    return dict(format="poseidon-model-graph-v2",id="capacity_split",
        resource_profile="general-tensors-v1",
        inputs=[dict(name="input",shape=shape)],constants={},
        nodes=[dict(id="partition",op="split",inputs=["input"],outputs=outputs,
                    attrs=dict(axis=axis,sections=sections))],
        outputs=[dict(name=n,value=n) for n in outputs])


class CapacitySplitTests(unittest.TestCase):
    def test_three_contexts_independent_references_and_checked_source(self):
        for shape,axis,sections in [([9],0,[2,3,4]),([2,3],1,[1,2]),
                                   ([1,2,3,2],-2,[1,2])]:
            with self.subTest(shape=shape,axis=axis):
                graph=model(shape,axis,sections)
                request=prepare_task(graph,dict(chunk_period=4))
                source,_=candidate_source(request)
                for probe in samples(graph,16):
                    first,second=math_reference(graph,probe),torch_reference(graph,probe)
                    for name in first:
                        np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
                    expected=np.concatenate([first[o["name"]].reshape(-1) for o in graph["outputs"]])
                    np.testing.assert_allclose(interpreted_output(source,request,probe),expected,
                                               atol=1e-12,rtol=1e-12)

    def test_request_and_public_bindings_are_unchanged(self):
        graph=model([2,3],-1,[1,2])
        request=prepare_task(graph,dict(chunk_period=4))
        original=canonical(request)
        candidate_source(request)
        self.assertEqual(canonical(request),original)
        self.assertEqual(canonical(reconstruct_request(request)),original)

    def test_input_and_every_split_result_can_be_reused(self):
        graph=model([9],0,[3,3,3])
        graph["nodes"].append(dict(id="mix",op="subtract",inputs=["part_0","part_2"],
                                  outputs=["delta"],attrs={}))
        graph["outputs"]=[dict(name="original",value="input"),
                          dict(name="middle",value="part_1"),dict(name="delta",value="delta")]
        request=prepare_task(graph,dict(chunk_period=4));source,_=candidate_source(request)
        for probe in samples(graph,16):
            original=copy.deepcopy(probe)
            values=math_reference(graph,probe)
            expected=np.concatenate([values[o["name"]].reshape(-1) for o in graph["outputs"]])
            np.testing.assert_allclose(interpreted_output(source,request,probe),expected,
                                       atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(probe["input"],original["input"])


    def test_partial_view_tail_is_excluded_from_reductions(self):
        for shape,axis,sections in [([9],0,[2,3,4]),([3,2],0,[1,2]),([1,3,2],1,[1,2])]:
            for operation in ("sum","mean"):
                with self.subTest(shape=shape,operation=operation):
                    graph=model(shape,axis,sections)
                    rank=len(shape)
                    graph["nodes"].append(dict(id="reduce",op=operation,inputs=["part_0"],
                        outputs=["result"],attrs=dict(axes=list(range(rank)),keepdims=True)))
                    graph["outputs"]=[dict(name="reduced",value="result"),
                                      dict(name="remaining",value="part_1")]
                    request=prepare_task(graph,dict(chunk_period=4));source,_=candidate_source(request)
                    count=int(np.prod(shape))
                    inputs={"input":np.asarray([2.,-1.]+[1000.+i*17 for i in range(count-2)]).reshape(shape)}
                    first,second=math_reference(graph,inputs),torch_reference(graph,inputs)
                    for name in first:
                        np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
                    expected=np.concatenate([first[o["name"]].reshape(-1) for o in graph["outputs"]])
                    np.testing.assert_allclose(interpreted_output(source,request,inputs),expected,
                                               atol=1e-12,rtol=1e-12)

    def test_public_data_is_not_mistaken_for_ciphertext(self):
        graph=model([4],0,[2,2])
        with self.assertRaisesRegex(ValueError,"ciphertext input"):
            split_tensor(None,np.arange(4),graph["nodes"][0],{"part_0":[2],"part_1":[2]})


    def test_contiguous_full_chunks_retain_exact_outputs(self):
        graph=model([1024],0,[512,512])
        request=prepare_task(graph,dict(chunk_period=256));source,_=candidate_source(request)
        inputs={"input":np.arange(1024,dtype=np.float64)-513.}
        first,second=math_reference(graph,inputs),torch_reference(graph,inputs)
        for name in first:
            np.testing.assert_array_equal(first[name],second[name])
        expected=np.concatenate([first[o["name"]].reshape(-1) for o in graph["outputs"]])
        np.testing.assert_array_equal(interpreted_output(source,request,inputs),expected)

    def test_large_fallback_retains_existing_source_and_work_limits(self):
        graph=model([32,32],1,[16,16])
        request=prepare_task(graph,dict(chunk_period=256))
        with self.assertRaisesRegex(ValueError,"budget|limit"):
            candidate_source(request)


if __name__=="__main__":unittest.main()
