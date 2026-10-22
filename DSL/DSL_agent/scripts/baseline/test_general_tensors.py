"""Rank expansion, immutable physical limits and old request compatibility."""
import copy
import unittest
import numpy as np
from benchmark_graph import validate, infer, shape, samples, canonical
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as independent
from model_resources import identity, resolve
from general_tensor_cases import PROFILE, Builder, tensor_shape, models
from component_contract import prepare_task, reconstruct_request
from unified_capacity_layout import ABI, EXTENDED_ABI, WIDE_ABI, physical_limits, layout
from unified_graph_contract import validate_request, validate_candidate
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output

class GeneralTensorTests(unittest.TestCase):
    def test_old_profile_identities_unchanged(self):
        expected={"legacy-v1":"e2de24eaf8d93604d8e532b3afcdd005c99cec8b4760189947561b16ada49f58",
                  "medium-v1":"358c2914e21971dbc1bfe9a8f5e134b53c01f978794e6dfd3bf81f2f89581084",
                  "resnet20-v1":"be93e276b6b97edba7c1d6fdd1e48741fd9977aa6d88645b0b4792e8671f7934"}
        for name,sha in expected.items():self.assertEqual(identity(name)["sha256"],sha)

    def test_rank_expansion_keeps_all_other_logical_limits(self):
        old=identity("resnet20-v1");new=identity(PROFILE)
        self.assertEqual(new["rank"],32)
        for name in old:
            if name not in ("name","rank","sha256"):self.assertEqual(new[name],old[name])

    def test_rank_boundaries_and_legacy_rejection(self):
        for rank in (1,4,5,8,16,32):
            b=Builder("rank_boundary",[tensor_shape(rank)])
            graph=b.finish(b.node("negate",["input0"]))
            validate(graph)
            if rank>4:
                for profile in ("legacy-v1","medium-v1","resnet20-v1"):
                    old=copy.deepcopy(graph);old["resource_profile"]=profile
                    with self.assertRaises(ValueError):validate(old)
        for rank in (0,33):
            b=Builder("rank_overflow",[[1]*rank])
            with self.assertRaises(ValueError):b.finish("input0")

    def test_public_constant_and_reshape_rank_boundaries(self):
        value=0.125
        for _ in range(32):value=[value]
        self.assertEqual(shape(value,resource_profile=PROFILE),(1,)*32)
        with self.assertRaises(ValueError):shape([value],resource_profile=PROFILE)
        self.assertEqual(infer("reshape",[(4,)],dict(shape=[1]*31+[4]),resource_profile=PROFILE),[(1,)*31+(4,)])
        with self.assertRaises(ValueError):infer("reshape",[(4,)],dict(shape=[1]*32+[4]),resource_profile=PROFILE)
        with self.assertRaises(ValueError):infer("reshape",[(4,)],dict(shape=[1]*4+[4]))

    def test_stack_cannot_exceed_rank(self):
        b=Builder("rank_stack",[tensor_shape(32)]*2)
        with self.assertRaises(ValueError):b.finish(b.node("stack",["input0","input1"],axis=0))

    def test_element_axis_and_node_budgets_still_apply(self):
        b=Builder("rank_elements",[[4097]])
        with self.assertRaises(ValueError):b.finish("input0")
        b=Builder("rank_axis",[[65537]])
        with self.assertRaises(ValueError):b.finish("input0")

    def test_operator_specific_rank_still_applies(self):
        for op in ("conv1d","conv2d","avg_pool1d","avg_pool2d"):
            b=Builder("rank_conv",[tensor_shape(5)])
            rank=1 if op.endswith("1d") else 2
            if op.startswith("conv"):
                refs=["input0",b.constant([[[0.125]]] if rank==1 else [[[[0.125]]]])]
                attrs=dict(stride=[1]*rank,padding=[0]*rank,dilation=[1]*rank,groups=1)
            else:
                refs=["input0"];attrs=dict(kernel=[1]*rank,stride=[1]*rank,padding=[0]*rank,count_include_pad=True)
            with self.assertRaises(ValueError):b.node(op,refs,**attrs)

    def test_no_unregistered_extended_physical_authority(self):
        for profile in ("legacy-v1","medium-v1"):
            for abi in (EXTENDED_ABI,WIDE_ABI):
                with self.assertRaises(ValueError):physical_limits(profile,abi)
        self.assertEqual(physical_limits(PROFILE),(EXTENDED_ABI,16,16))
        self.assertEqual(physical_limits(PROFILE,WIDE_ABI),(WIDE_ABI,16,16))
        self.assertEqual(physical_limits(PROFILE,ABI),(ABI,4,4))
        graph=models()[0];graph["inputs"][0]["shape"]=[65]
        with self.assertRaises(ValueError):layout(graph,4)

    def test_same_canonical_layout_and_request_reconstruction(self):
        for graph in models():
            r=prepare_task(graph,dict(chunk_period=4,generation_guidance="explicit-v10"))
            self.assertEqual(validate_request(r),4)
            self.assertEqual(r["layout"]["execution_abi"],EXTENDED_ABI)
            self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
            self.assertNotIn("Only resnet20-v1 uses this ABI.",r["rules"])
            self.assertIn("general-tensors-v1",r["rules"])

    def test_independent_references_and_rule_candidates(self):
        for graph in models():
            r=prepare_task(graph,dict(chunk_period=4,generation_guidance="explicit-v10"))
            source,_=candidate_source(r)
            validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
            for probe in samples(graph,4):
                first=mathematical(graph,probe);second=independent(graph,probe)
                for name in first:np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
                expected=np.concatenate([first[o["name"]].reshape(-1) for o in graph["outputs"]])
                np.testing.assert_allclose(interpreted_output(source,r,probe),expected,atol=1e-12,rtol=1e-12)

    def test_static_permutations_cross_chunks_and_tail(self):
        b=Builder("rank_permutation_tail",[[1,1,1,2,3]])
        value=b.node("permute",["input0"],dims=[0,1,2,4,3])
        graph=b.finish(value)
        r=prepare_task(graph,dict(chunk_period=4,generation_guidance="explicit-v10"))
        source,_=candidate_source(r)
        self.assertEqual(len(r["layout"]["inputs"]),2)
        for probe in samples(graph,4):
            expected=mathematical(graph,probe)["output0"].reshape(-1)
            np.testing.assert_allclose(interpreted_output(source,r,probe),expected,atol=1e-12,rtol=1e-12)

    def test_capability_matrix_uses_profile_physical_authority(self):
        from application_capabilities import describe, assess_model
        matrix=describe()
        self.assertEqual(matrix["resource_profiles"][PROFILE]["rank"],32)
        for item in matrix["layouts"]:
            if item["abi"] in (EXTENDED_ABI,WIDE_ABI):self.assertIn(PROFILE,item["resource_profiles"])
        for graph in models():
            report=assess_model(graph,options=dict(chunk_period=4,generation_guidance="explicit-v10"))
            self.assertEqual(report["admission"],"accepted_pending_validation")
            self.assertEqual(report["execution_readiness"],"unknown")
            self.assertFalse(report["encrypted_execution"])

if __name__=="__main__":unittest.main()
