"""Channel-affine BatchNorm and aligned suffix reductions with frozen public bindings."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import samples,validate,canonical,digest
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_request
from component_contract import reconstruct_request
from unified_capacity_constants import packing,CHANNEL_VERSION,VERSION
from unified_capacity_routes import aligned_suffix_plan
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output

CASES=Path(__file__).parent/"cases/capacity-channel-v1"
def models():
    return [json.loads(p.read_text()) for p in sorted(CASES.glob("*.json"))]
def request(g,version=CHANNEL_VERSION):
    c=configuration("poseidon-cpu-eva-w45-v1")
    return prepare(g,c["ckks_config_sha256"],c,chunk_period=256,
                   capacity_packing_version=version,generation_guidance="explicit-v10")
class ChannelLayoutTests(unittest.TestCase):
    def test_component_and_cli_select_the_same_explicit_packing(self):
        from component_contract import prepare_task,runner_options
        from run_candidate import parse_args
        g=models()[0];r=request(g)
        self.assertEqual(canonical(r),canonical(prepare_task(g,dict(chunk_period=256,
            capacity_packing_version=CHANNEL_VERSION,generation_guidance="explicit-v10"))))
        flags=runner_options(r);i=flags.index("--unified-packing")
        self.assertEqual(flags[i+1],CHANNEL_VERSION)
        args=parse_args(["--case",str(CASES/(g["id"]+".json")),"--prepare",*flags])
        self.assertEqual(args.unified_packing,CHANNEL_VERSION)
    def test_three_contexts_per_partition(self):
        gs=models();self.assertEqual(len(gs),6)
        self.assertEqual(sum(any(n["op"]=="batch_norm" for n in g["nodes"]) for g in gs),3)
        self.assertEqual(sum(g["id"].startswith("capacity_suffix_") for g in gs),3)
    def test_dual_references_and_independent_generated_arithmetic(self):
        for g in models():
            with self.subTest(model=g["id"]):
                r=request(g);source,_=candidate_source(r)
                for probe in samples(g,16):
                    a,b=math_ref(g,probe),torch_ref(g,probe)
                    for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
                    expected=np.concatenate([a[o["name"]].reshape(-1) for o in g["outputs"]])
                    np.testing.assert_allclose(interpreted_output(source,r,probe),expected,atol=1e-12,rtol=1e-12)
    def test_request_reconstruction_and_wire_roundtrip(self):
        from candidate_contract import request_file_bytes
        for g in models():
            r=request(g);self.assertEqual(canonical(r),canonical(reconstruct_request(r)))
            raw=json.loads(request_file_bytes(r));validate_request(raw)
            self.assertEqual(canonical(raw),canonical(r))
    def test_tail_padding_and_rank_channel_mapping(self):
        for g in models()[:3]:
            r=request(g);packed=packing(g,r["layout"],CHANNEL_VERSION);checked=validate(g)
            node=next(n for n in g["nodes"] if n["op"]=="batch_norm")
            self.assertEqual(packed["tensor_strategies"][node["id"]],"channel_affine")
            refs=node["inputs"];self.assertTrue(set(refs[1:])<=packed["excluded_scalars"])
            size=np.prod(checked["shapes"][node["outputs"][0]])
            for b in packed["bindings"]:
                if b["node_id"]==node["id"] and b["role"].startswith("bn_"):
                    n=min(256,size-256*b["chunk"])
                    self.assertTrue(all(v==0. for v in packed["vectors"][b["key"]][n:]))
    def test_raw_stat_reused_elsewhere_retains_scalar_binding(self):
        g=copy.deepcopy(models()[0])
        g["nodes"].append(dict(id="poly",op="polynomial",inputs=["bn","gain"],outputs=["polyout"],attrs={"basis":"power"}))
        g["outputs"]=[dict(name="result",value="polyout")]
        r=request(g);packed=packing(g,r["layout"],CHANNEL_VERSION)
        self.assertNotIn("gain",packed["excluded_scalars"])
        self.assertTrue(set(g["constants"]["gain"])<={v for v in r["public_constants"].values() if type(v) is float})
    def test_changed_gather_binding_and_public_values_rejected(self):
        r=request(models()[-1])
        for field in ("step","baby_step","giant_step","input_chunk","output_chunk"):
            bad=copy.deepcopy(r);row=next(b for b in bad["constant_origins"]["capacity_tensor_packing"]["bindings"] if b["role"]=="suffix_gather")
            row[field]+=1;bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaisesRegex(ValueError,"Changed capacity public tensor"):validate_request(bad)
        g=models()[0];r=request(g);bad=copy.deepcopy(r)
        key=next(b["key"] for b in bad["constant_origins"]["capacity_tensor_packing"]["bindings"] if b["role"]=="bn_gain")
        bad["public_constants"][key][0]+=.125
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Public registry changed"):validate_request(bad)
    def test_row_boundaries_have_discriminating_nonuniform_inputs(self):
        g=models()[-1];r=request(g);src,_=candidate_source(r)
        x=np.arange(np.prod(g["inputs"][0]["shape"]),dtype=np.float64).reshape(g["inputs"][0]["shape"])/512
        probe={"x":x};expected=math_ref(g,probe)["result"].reshape(-1)
        np.testing.assert_allclose(interpreted_output(src,r,probe),expected,atol=1e-12,rtol=1e-12)
        self.assertGreater(len(set(expected.tolist())),1)
    def test_nonpower_and_non_suffix_are_not_silently_selected(self):
        g=models()[-1];node=g["nodes"][-1];shapes=validate(g)["shapes"]
        self.assertIsNotNone(aligned_suffix_plan(node,shapes,256))
        bad=copy.deepcopy(node);bad["attrs"]["axes"]=[0]
        self.assertIsNone(aligned_suffix_plan(bad,shapes,256))
        changed=dict(shapes);changed[node["inputs"][0]]=[4,4,31]
        self.assertIsNone(aligned_suffix_plan(node,changed,256))
    def test_previous_v7_requests_and_sources_stay_exact(self):
        from hecate_python_env import WORK
        count=0
        for f in (WORK/"results/poseidon-pytorch-extended-r532").glob("*/request.json"):
            raw=json.loads(f.read_text());validate_request(raw)
            self.assertEqual(canonical(raw),canonical(reconstruct_request(raw)))
            source,_=candidate_source(raw)
            self.assertEqual(source,(f.parent/"candidate.py").read_text());count+=1
        self.assertEqual(count,5)

    def test_tiled_wire_is_lossless_and_preserves_numeric_types(self):
        from unified_request_storage import compact_request,restore_public_constants,TILES_VERSION
        pattern=[-0.0,0.0,0]+[.125]*28+[1]
        original=dict(layout=dict(input_slot_period=256),
            constant_origins=dict(capacity_tensor_packing=dict(version=CHANNEL_VERSION)),
            public_constants=dict(v=pattern*8))
        wire=compact_request(original)
        self.assertEqual(wire["public_constants"]["v"]["format"],TILES_VERSION)
        self.assertLess(len(canonical(wire)),len(canonical(original)))
        decoded=json.loads(canonical(wire));restore_public_constants(decoded)
        self.assertEqual(canonical(decoded),canonical(original))
    def test_tile_metadata_expansion_and_indices_rejected(self):
        from unified_request_storage import compact_request,restore_public_constants
        original=dict(layout=dict(input_slot_period=256),
            constant_origins=dict(capacity_tensor_packing=dict(version=CHANNEL_VERSION)),
            public_constants=dict(v=([0.]+[.125]*30+[0.])*8))
        wire=compact_request(original);record=wire["public_constants"]["v"]
        self.assertIsInstance(record,dict)
        changes=[("length",257),("block_size",3),("indices",[[257,0]]),
                 ("indices",[[8,99]]),("blocks",[[[32,float("inf")]]]),
                 ("blocks",record["blocks"]+record["blocks"])]
        for key,value in changes:
            bad=copy.deepcopy(wire);bad["public_constants"]["v"][key]=value
            with self.assertRaises(ValueError):restore_public_constants(bad)
    def test_tile_encoding_is_not_accepted_under_historical_v7(self):
        from unified_request_storage import compact_request,restore_public_constants
        original=dict(layout=dict(input_slot_period=256),
            constant_origins=dict(capacity_tensor_packing=dict(version=CHANNEL_VERSION)),
            public_constants=dict(v=([0.]+[.125]*30+[0.])*8))
        wire=compact_request(original);wire["constant_origins"]["capacity_tensor_packing"]["version"]=VERSION
        with self.assertRaisesRegex(ValueError,"require packing v8"):restore_public_constants(wire)
if __name__=="__main__":unittest.main()
