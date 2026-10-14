"""Double-reference and independent source checks for public spatial routes."""
import copy,json,math,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import validate,samples,canonical,digest
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from component_contract import prepare_task,reconstruct_request
from unified_graph_contract import validate_request,validate_candidate,prepare
from unified_capacity_constants import VERSION,TENSOR_VERSION
from unified_capacity_spatial import contributions,vectors
from unified_capacity_spatial_cases import models
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output

class CapacitySpatialTests(unittest.TestCase):
    def compare(self,g,count=16):
        r=prepare_task(g,{"chunk_period":256});source,_=candidate_source(r)
        for inputs in samples(g,count):
            a,b=math_ref(g,inputs),torch_ref(g,inputs)
            for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12,rtol=1e-12)
            expected=np.concatenate([a[o["name"]].reshape(-1) for o in g["outputs"]])
            np.testing.assert_allclose(interpreted_output(source,r,inputs),expected,atol=1e-12,rtol=1e-12)
        return r,source

    def test_four_operator_partitions_and_three_compositions(self):
        from collections import Counter
        cases=models();self.assertEqual(len(cases),15)
        self.assertEqual(Counter(g["nodes"][1]["op"] for g in cases[:12]),
                         dict(conv1d=3,conv2d=3,avg_pool1d=3,avg_pool2d=3))
        self.assertTrue(all(math.prod(g["inputs"][0]["shape"])>256 for g in cases))
        self.assertEqual({len(g["inputs"][0]["shape"]) for g in cases},{2,3,4})
        self.assertEqual(len(cases[-2]["outputs"]),2)

    def test_two_independent_references_and_generated_arithmetic(self):
        for g in models():self.compare(g)

    def test_grouped_vectors_preserve_each_public_coefficient(self):
        for g in models()[:12]:
            node=g["nodes"][1];shapes=validate(g)["shapes"]
            expected={(op,s,d):v for op,s,d,v in contributions(node,shapes,g["constants"])}
            actual={}
            for (op,s,t,step),vector in vectors(node,shapes,g["constants"],256).items():
                for slot,v in enumerate(vector):
                    if v:
                        key=(op,s*256+(slot+step)%256,t*256+slot)
                        self.assertNotIn(key,actual);actual[key]=v
            self.assertEqual(actual,expected)

    def test_weight_bias_operand_and_route_tampering_rejected(self):
        r=prepare_task(models()[3],{"chunk_period":256})
        self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["version"],VERSION)
        self.assertEqual(canonical(r),canonical(reconstruct_request(r)))
        bindings=r["constant_origins"]["capacity_tensor_packing"]["bindings"]
        self.assertTrue(any(b["role"]=="spatial_bias" for b in bindings))
        for field in ("step","baby_step","giant_step","input_chunk","output_chunk","operand","source"):
            bad=copy.deepcopy(r);b=next(b for b in bad["constant_origins"]["capacity_tensor_packing"]["bindings"] if b["role"]=="tensor_route")
            b[field]="offset" if field=="source" else b[field]+1
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaises(ValueError):validate_request(bad)
        for role in ("spatial_bias","tensor_route"):
            bad=copy.deepcopy(r);key=next(b["key"] for b in bindings if b["role"]==role)
            bad["public_constants"][key][0]+=0.125
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaises(ValueError):validate_request(bad)

    def test_zero_weights_and_bias_only_retained(self):
        g=copy.deepcopy(models()[3])
        g["constants"]["weight"]=np.zeros_like(g["constants"]["weight"]).tolist()
        self.compare(g)
        g["constants"]["bias"]=[0.,0.]
        r,src=self.compare(g)
        self.assertEqual(r["constant_origins"]["capacity_tensor_packing"]["tensor_strategies"],{"n1":"spatial_bsgs"})
        self.assertTrue(all(b["role"]=="spatial_bias" for b in r["constant_origins"]["capacity_tensor_packing"]["bindings"]))

    def test_pool_edge_divisor_is_mathematical_input(self):
        g=copy.deepcopy(models()[7]);h=copy.deepcopy(g)
        h["nodes"][1]["attrs"]["count_include_pad"]=True
        probe=samples(g,4)[0]
        self.assertFalse(np.array_equal(math_ref(g,probe)["result"],math_ref(h,probe)["result"]))
        first,_=self.compare(g,4);second,_=self.compare(h,4)
        self.assertNotEqual(first["request_id"],second["request_id"])

    def test_mapping_work_budget_and_legacy_spatial_rejection(self):
        g=copy.deepcopy(models()[0]);g["inputs"][0]["shape"]=[1,1024]
        g["constants"]["weight"]=[[[.125]*257]];g["constants"]["bias"]=[.125]
        g["nodes"][1]["attrs"]["padding"]=[128]
        validate(g)
        with self.assertRaisesRegex(ValueError,"spatial map work budget"):prepare_task(g,{"chunk_period":256})
        from compiler_configuration import configuration
        c=configuration("poseidon-cpu-eva-w45-v1")
        old=prepare(models()[0],c["ckks_config_sha256"],c,chunk_period=256,capacity_packing_version=TENSOR_VERSION)
        with self.assertRaisesRegex(ValueError,"No bounded capacity spatial binding"):candidate_source(old)

    def test_request_and_trace_wire_roundtrip_and_size(self):
        from candidate_contract import request_file_bytes,trace_payload_json
        for g in models():
            r=prepare_task(g,{"chunk_period":256});src,_=candidate_source(r)
            data=request_file_bytes(r);self.assertLessEqual(len(data),131072)
            raw=json.loads(data);validate_request(raw)
            self.assertEqual(canonical(raw),canonical(r))
            payload=json.loads(trace_payload_json(r,dict(schema=1,request_id=r["request_id"],hecate_source=src)))
            validate_request(payload["request"])
            self.assertEqual(canonical(payload["request"]),canonical(r))
            validate_candidate(payload["candidate"],payload["request"])

    def test_storage_preserves_signed_zero_and_numeric_types(self):
        from unified_request_storage import compact_request,restore_public_constants
        original=dict(constant_origins=dict(capacity_tensor_packing=dict(version=VERSION)),
                      layout=dict(input_slot_period=256),
                      public_constants=dict(vector=[-0.0,0.0,0,1.,1]+[0.]*251))
        decoded=json.loads(canonical(compact_request(original)))
        restore_public_constants(decoded)
        self.assertEqual(canonical(decoded),canonical(original))

    def test_storage_expansion_and_malformed_runs_rejected(self):
        from candidate_contract import request_file_bytes
        r=prepare_task(models()[1],{"chunk_period":256})
        raw=json.loads(request_file_bytes(r))
        key=next(k for k,v in raw["public_constants"].items() if type(v) is dict)
        for value in (dict(format="capacity-public-vector-rle-v1",length=257,runs=[[257,0.]]),
                      dict(format="capacity-public-vector-rle-v1",length=256,runs=[[0,0.]]),
                      dict(format="capacity-public-vector-rle-v1",length=256,runs=[[257,0.]]),
                      dict(format="capacity-public-vector-rle-v1",length=256,runs=[[256,float("inf")]])):
            bad=copy.deepcopy(raw);bad["public_constants"][key]=value
            with self.assertRaises(ValueError):validate_request(bad)

    def test_previous_v4_requests_reconstruct_without_relabeling(self):
        from hecate_python_env import WORK
        count=0
        for name in ("poseidon-capacity-tensor-numerical-r435","poseidon-capacity-tensor-composition-r437","poseidon-capacity-tensor-collision-r439"):
            for f in (WORK/"results"/name).glob("*/request.json"):
                r=json.loads(f.read_text())
                self.assertEqual(canonical(r),canonical(reconstruct_request(r)))
                d=r["constant_origins"].get("capacity_tensor_packing")
                if d:self.assertEqual(d["version"],TENSOR_VERSION)
                from candidate_contract import request_file_bytes
                self.assertEqual(request_file_bytes(r),canonical(r))
                count+=1
        self.assertEqual(count,24)

if __name__=="__main__":unittest.main()
