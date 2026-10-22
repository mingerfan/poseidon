"""Independent references, source arithmetic, index coverage and old request binding."""
import copy,json,math,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import samples,canonical,digest,validate
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from component_contract import prepare_task,reconstruct_request
from unified_graph_contract import validate_request,validate_candidate
from unified_capacity_constants import packing,VERSION,TILE_VERSION
from unified_capacity_routes import contributions,vectors,runs
from unified_capacity_tensor_cases import models
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output

class CapacityTensorTests(unittest.TestCase):
    def test_four_partitions_and_inner_axis_boundary(self):
        from collections import Counter
        cases=models()
        self.assertEqual(len(cases),13)
        self.assertEqual(Counter(g["nodes"][-1]["op"] for g in cases[:-1]),dict(concat=3,stack=3,sum=3,mean=3))
        self.assertEqual({len(g["inputs"][0]["shape"]) for g in cases},{1,2,3,4})
        self.assertTrue(any(g["nodes"][-1]["attrs"].get("axis")==-1 for g in cases))

    def test_independent_references_and_emitted_arithmetic(self):
        for g in models():
            probes=samples(g,16)
            for inputs in probes:
                a,b=math_ref(g,inputs),torch_ref(g,inputs)
                for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
            r=prepare_task(g,{"chunk_period":256});source,_=candidate_source(r)
            for inputs in probes:
                expected=math_ref(g,inputs)["result"].reshape(-1)
                np.testing.assert_allclose(interpreted_output(source,r,inputs),expected,atol=1e-12,rtol=1e-12)

    def test_grouped_routes_exactly_match_static_index_contributions(self):
        for g in models():
            node=g["nodes"][-1];shapes=validate(g)["shapes"];routes=vectors(node,shapes,256)
            expected={(operand,src,dst):factor for operand,src,dst,factor in contributions(node,shapes)}
            actual={}
            for (operand,source,target,step),vector in routes.items():
                for slot,factor in enumerate(vector):
                    if not factor:continue
                    key=(operand,source*256+(slot+step)%256,target*256+slot)
                    self.assertNotIn(key,actual);actual[key]=factor
            self.assertEqual(actual,expected)
            if node["op"] in ("concat","stack"):
                self.assertEqual(len({dst for _,_,dst in actual}),len(actual))

    def test_mapping_and_normalization_are_bound(self):
        r=prepare_task(models()[0],{"chunk_period":256});d=r["constant_origins"]["capacity_tensor_packing"]
        self.assertEqual(d["version"],VERSION);self.assertEqual(canonical(r),canonical(reconstruct_request(r)))
        for field,val in (("step",1),("input_chunk",3),("output_chunk",3),("operand",3)):
            bad=copy.deepcopy(r);bad["constant_origins"]["capacity_tensor_packing"]["bindings"][0][field]=val
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaises(ValueError):validate_request(bad)
        r=prepare_task(next(g for g in models() if g["id"]=="capacity_mean_axes_0"),{"chunk_period":256})
        b=copy.deepcopy(r);k=b["constant_origins"]["capacity_tensor_packing"]["bindings"][0]["key"]
        b["public_constants"][k]=[1. if v else 0. for v in b["public_constants"][k]]
        b["request_id"]=digest({k:v for k,v in b.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(b)

    def test_repeated_mask_rotation_sums_are_bounded(self):
        g=next(g for g in models() if g["id"]=="capacity_sum_axes_1")
        r=prepare_task(g,{"chunk_period":256});source,_=candidate_source(r)
        result=validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
        self.assertLess(result["functions"]["golden"]["expanded_cost"],1024)
        self.assertLess(len(source.splitlines()),300)
        self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["tensor_strategies"],{"n1":"direct"})
        self.assertEqual(runs([0,1,2,7,8]),[(0,3),(7,2)])

    def test_suffix_reduction_uses_public_row_mapping(self):
        for op in ("sum","mean"):
            g=next(g for g in models() if g["id"]=="capacity_"+op+"_axes_2")
            r=prepare_task(g,{"chunk_period":256});d=r["constant_origins"]["capacity_tensor_packing"]
            self.assertEqual(d["tensor_strategies"],{"n1":"tiled_suffix"})
            self.assertEqual(d["linear_strategies"],{"n1":"tiled_diagonal"})

    def test_inner_axis_stack_has_bound_operand_tiles(self):
        g=models()[-1];r=prepare_task(g,{"chunk_period":256})
        d=r["constant_origins"]["capacity_tensor_packing"]
        self.assertEqual(d["tensor_strategies"],{"n2":"tiled_stack"})
        self.assertEqual([c["operand"] for c in d["stack_operands"]["n2"]],[0,1])
        bad=copy.deepcopy(r);bad["constant_origins"]["capacity_tensor_packing"]["stack_operands"]["n2"][1]["operand"]=0
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(bad)
        from unified_graph_contract import prepare
        from compiler_configuration import configuration
        c=configuration("poseidon-cpu-eva-w45-v1")
        old=prepare(g,c["ckks_config_sha256"],c,chunk_period=256,capacity_packing_version=TILE_VERSION)
        with self.assertRaisesRegex(ValueError,"inner-axis join"):candidate_source(old)

    def test_layout_nodes_cannot_alias_user_nodes(self):
        g=copy.deepcopy(models()[-1])
        g["id"]="capacity_tensor_name_collision"
        for spec in g["inputs"]:spec["shape"]=[17]
        g["constants"]["weight"]=[[0.125]*17]
        g["nodes"].insert(2,dict(id="n2_operand0",op="linear",inputs=["shifted0","weight"],attrs={},outputs=["linear_result"]))
        g["outputs"].append(dict(name="linear_result",value="linear_result"))
        r=prepare_task(g,{"chunk_period":256})
        children=r["constant_origins"]["capacity_tensor_packing"]["stack_operands"]["n2"]
        original={n["id"] for n in g["nodes"]}
        self.assertFalse(original & {c["node_id"] for c in children})
        self.assertEqual(children[0]["node_id"],"n2_operand0_")
        source,_=candidate_source(r)
        for probe in samples(g,16):
            expected=math_ref(g,probe)
            for name,v in torch_ref(g,probe).items():np.testing.assert_allclose(expected[name],v,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(interpreted_output(source,r,probe),
                np.concatenate([expected[o["name"]].reshape(-1) for o in g["outputs"]]),atol=1e-12,rtol=1e-12)
        g["nodes"][2]["id"]="n2_operand0_"
        r=prepare_task(g,{"chunk_period":256})
        self.assertFalse({n["id"] for n in g["nodes"]} & {
             c["node_id"] for c in r["constant_origins"]["capacity_tensor_packing"]["stack_operands"]["n2"]})

    def test_prior_v3_and_absent_descriptor_requests_reconstruct(self):
        from hecate_python_env import WORK
        rows=list((WORK/"results/poseidon-capacity-tiles-numerical-r422").glob("*/request.json"))
        self.assertEqual(len(rows),8)
        for f in rows:
            r=json.loads(f.read_text());self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
            d=r["constant_origins"].get("capacity_tensor_packing")
            if d:self.assertEqual(d["version"],TILE_VERSION)
        total=absent=0
        for batch in ("poseidon-capacity-minimal-r382","poseidon-capacity-remainder-r383"):
            folder=WORK/"results"/batch;self.assertTrue(folder.is_dir())
            for f in folder.glob("*/request.json"):
                r=json.loads(f.read_text());self.assertEqual(canonical(reconstruct_request(r)),canonical(r));total+=1
                absent+=int(r["constant_origins"].get("capacity_tensor_packing") is None)
        self.assertEqual(total,21);self.assertEqual(absent,21)

if __name__=="__main__":unittest.main()
