"""Exact pool factoring and physical Linear choice with old request compatibility."""
import ast,copy,json,math,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import samples,validate,canonical,digest
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from component_contract import prepare_task,reconstruct_request
from unified_graph_contract import validate_request,prepare
from unified_capacity_constants import VERSION,WINDOW_VERSION,SPATIAL_VERSION,packing,row_sum_cost
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output

def pool_models():
    result=[]
    for i,(shape,kernel,stride) in enumerate((
        ([1,2,17,17],[2,2],[2,2]),([2,19,11],[3,2],[2,2]),
        ([1,1,13,23],[2,3],[2,3]))):
        result.append(dict(format="poseidon-model-graph-v2",resource_profile="medium-v1",id="window_"+str(i),
            inputs=[dict(name="x",shape=shape)],constants={},
            nodes=[dict(id="pool",op="avg_pool2d",inputs=["x"],outputs=["p"],
                attrs=dict(kernel=kernel,stride=stride,padding=[0,0],count_include_pad=False))],
            outputs=[dict(name="pooled",value="p")]))
    return result

class SpatialWindowTests(unittest.TestCase):
    def compare(self,g,version=None):
        if version is None:request=prepare_task(g,dict(chunk_period=256,generation_guidance="explicit-v10"))
        else:
            from compiler_configuration import configuration
            config=configuration("poseidon-cpu-eva-w45-v1")
            request=prepare(g,config["ckks_config_sha256"],config,chunk_period=256,
                            generation_guidance="explicit-v10",capacity_packing_version=version)
        source,_=candidate_source(request)
        for inputs in samples(g,16):
            a=math_ref(g,inputs,resource_profile="medium-v1")
            b=torch_ref(g,inputs,resource_profile="medium-v1")
            expected=np.concatenate([a[o["name"]].reshape(-1) for o in g["outputs"]])
            np.testing.assert_allclose(expected,np.concatenate([b[o["name"]].reshape(-1) for o in g["outputs"]]),atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(interpreted_output(source,request,inputs),expected,atol=1e-12,rtol=1e-12)
        return request,source

    def test_three_odd_spatial_contexts_and_cross_chunk_windows(self):
        for g in pool_models():
            r,_=self.compare(g)
            self.assertIn("Public packing capacity-public-tensors-v7:",r["rules"])
            self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["tensor_strategies"]["pool"],"spatial_stream")
            self.assertLessEqual(len(r["public_constants"]),256)
            self.assertEqual(canonical(r),canonical(reconstruct_request(r)))

    def test_padding_and_divisor_do_not_enter_unpadded_strategy(self):
        g=pool_models()[0];g["inputs"][0]["shape"]=[1,1,17,17];g["nodes"][0]["attrs"]["padding"]=[1,1]
        r=self.compare(g)[0]
        self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["tensor_strategies"]["pool"],"spatial_bsgs")

    def test_shift_anchor_and_window_plan_tampering_rejected(self):
        r=self.compare(pool_models()[0])[0]
        for role,field in (("pool_stream_gather","stream"),("pool_stream_gather","step"),
                           ("pool_stream_gather","input_chunk"),("pool_stream_gather","output_chunk")):
            bad=copy.deepcopy(r);b=next(b for b in bad["constant_origins"]["capacity_tensor_packing"]["bindings"] if b["role"]==role)
            b[field]+=1;bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaises(ValueError):validate_request(bad)
        bad=copy.deepcopy(r)
        bad["constant_origins"]["capacity_tensor_packing"]["spatial_streams"]["pool"][0]["offsets"][0]+=1
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(bad)

    def test_row_sum_chosen_for_three_wide_small_output_linears(self):
        for width,outputs,batches in ((65,4,2),(67,3,2),(71,5,3)):
            weights=[[(j+i*width)%17*.002-.016 for j in range(width)] for i in range(outputs)]
            g=dict(format="poseidon-model-graph-v2",resource_profile="medium-v1",id="wide_"+str(width),
                inputs=[dict(name="x",shape=[batches,width])],constants=dict(w=weights),
                nodes=[dict(id="linear",op="linear",inputs=["x","w"],attrs={},outputs=["p"])],
                outputs=[dict(name="result",value="p")])
            r,_=self.compare(g)
            self.assertIn(r["constant_origins"]["capacity_tensor_packing"]["linear_strategies"]["linear"],("row_sum","tiled_row_sum"))
            self.assertLess(row_sum_cost(weights,batches,256),274 if width==65 else 1024)

    def test_original_medium_models_fit_unchanged_source_and_ast_limits(self):
        from hecate_python_env import WORK
        from candidate_contract import request_file_bytes,trace_payload_json
        for label in ("spatial_residual","linear_fusion","reduction_fusion"):
            g=json.loads((Path(__file__).parent/"cases/trusted-pytorch-medium-v1/captured-graphs"/(label+".json")).read_text())
            r,source=self.compare(g)
            self.assertLessEqual(len(request_file_bytes(r)),131072)
            self.assertLessEqual(len(trace_payload_json(r,dict(schema=1,request_id=r["request_id"],hecate_source=source)).encode()),131072)
            self.assertLessEqual(sum(1 for _ in ast.walk(ast.parse(source))),4096)

    def test_old_v5_request_hashes_remain_exact(self):
        from hecate_python_env import WORK
        for label in ("linear_fusion","reduction_fusion"):
            g=json.loads((Path(__file__).parent/"cases/trusted-pytorch-medium-v1/captured-graphs"/(label+".json")).read_text())
            from compiler_configuration import configuration
            config=configuration("poseidon-cpu-eva-w45-v1")
            r=prepare(g,config["ckks_config_sha256"],config,chunk_period=256,
                      generation_guidance="explicit-v10",capacity_packing_version=SPATIAL_VERSION)
            self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["version"],SPATIAL_VERSION)
            self.assertEqual(canonical(r),canonical(reconstruct_request(r)))
            prior=WORK/"results/poseidon-pytorch-medium-r470"/label/"request.json"
            if prior.exists():
                historical=json.loads(prior.read_text());validate_request(historical)
                self.assertEqual(historical["request_id"],r["request_id"])
                self.assertEqual(canonical(historical),canonical(r))

    def test_v6_window_semantics_and_executed_requests_stay_bound(self):
        from hecate_python_env import WORK
        for g in pool_models():
            r,_=self.compare(g,WINDOW_VERSION)
            self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["tensor_strategies"]["pool"],"spatial_window")
            self.assertEqual(canonical(r),canonical(reconstruct_request(r)))
        for p in (WORK/"results/poseidon-pytorch-medium-r484").glob("*/request.json"):
            r=json.loads(p.read_text());validate_request(r)
            self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["version"],WINDOW_VERSION)
            self.assertEqual(canonical(r),canonical(reconstruct_request(r)))

    def test_stream_fragments_in_three_1d_contexts(self):
        for i,(shape,kernel,stride) in enumerate((([2,259],3,2),([1,3,171],5,3),([3,93],4,1))):
            g=dict(format="poseidon-model-graph-v2",resource_profile="medium-v1",id="stream1d_"+str(i),
                inputs=[dict(name="x",shape=shape)],constants={},
                nodes=[dict(id="pool",op="avg_pool1d",inputs=["x"],outputs=["p"],
                    attrs=dict(kernel=[kernel],stride=[stride],padding=[0],count_include_pad=True))],
                outputs=[dict(name="result",value="p")])
            r,_=self.compare(g)
            self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["tensor_strategies"]["pool"],"spatial_stream")

    def test_tiled_row_masks_and_method_tampering_rejected(self):
        g=json.loads((Path(__file__).parent/"cases/trusted-pytorch-medium-v1/captured-graphs/linear_fusion.json").read_text())
        r=self.compare(g)[0]
        descriptor=r["constant_origins"]["capacity_tensor_packing"]
        self.assertIn("tiled_row_sum",descriptor["linear_strategies"].values())
        for role in ("linear_tile_row_weight","linear_tile_row_mask"):
            bad=copy.deepcopy(r);key=next(b["key"] for b in descriptor["bindings"] if b["role"]==role)
            bad["public_constants"][key][0]+=.01
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaises(ValueError):validate_request(bad)

if __name__=="__main__":unittest.main()
