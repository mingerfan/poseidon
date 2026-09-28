
"""Actual public helper evaluation is separate from encrypted acceptance."""
import copy,unittest
import numpy as np
from benchmark_graph import samples,digest
from benchmark_math import evaluate
from benchmark_torch import evaluate as reference
from unified_graph_contract import layout,prepare,validate_candidate,validate_request
from compiler_configuration import PROFILE_SHA256,configuration
from upstream_adapters import fused_conv_bn,spatial_mapped
from upstream_fused_cases import cases,corpus
from test_upstream_spatial import Vector
from upstream_candidate_helpers import manifest,FUSED_PROFILE,MAPPED_PROFILE

class FusedVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
    def test_actual_fused_functions_original_and_asymmetric_models(self):
        for row in cases():
            g=row["model"];p=layout(g)["input_slot_period"];cap=manifest(g,FUSED_PROFILE)
            selected={cap["helpers"][name]["binding"]["node_id"]:cap["helpers"][name]["binding"]
                      for name in row["required_helpers"]}
            required_values={b["input_value"] for b in selected.values()}|{o["value"] for o in g["outputs"]}
            with self.subTest(case=row["name"]):
                for inputs in samples(g,16):
                    expected=evaluate(g,inputs);ref=reference(g,inputs)
                    env={k:Vector(np.tile(np.pad(v.reshape(-1),(0,p-v.size)),16384//p)) for k,v in inputs.items()}
                    for node in g["nodes"]:
                        if node["id"] in selected:
                            b=selected[node["id"]];x=env[b["input_value"]]
                            adapter=fused_conv_bn if b.get("adapter")==fused_conv_bn.ADAPTER else spatial_mapped
                            value,record=adapter.apply(b,x,self.helpers,self.mpcb);adapter.verify_record(b,record)
                            if adapter is fused_conv_bn:
                                np.testing.assert_allclose(value.data[:p],adapter.probe(b,tuple(x.data[:p])),atol=1e-12,rtol=1e-12)
                        elif node["op"]=="negate":value=Vector(-env[node["inputs"][0]].data)
                        elif node["op"]=="square":value=Vector(env[node["inputs"][0]].data**2)
                        else:
                            self.assertFalse(set(node["outputs"])&required_values);continue
                        env[node["outputs"][0]]=value
                    for out in g["outputs"]:
                        values=env[out["value"]].data;n=expected[out["name"]].size
                        np.testing.assert_allclose(values[:n],expected[out["name"]].reshape(-1),atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(expected[out["name"]],ref[out["name"]],atol=1e-12,rtol=1e-12)
                        np.testing.assert_array_equal(values,np.tile(values[:p],16384//p))
    def test_record_parameter_and_family_tampering_rejected(self):
        row=cases()[0];g=row["model"];p=layout(g)["input_slot_period"]
        b=manifest(g,FUSED_PROFILE)["helpers"]["HE_ConvBN0"]["binding"]
        _,record=fused_conv_bn.apply(b,Vector(np.tile(np.arange(p)/32,16384//p)),self.helpers,self.mpcb)
        for key in ("bias","mean","helper","calls","work"):
            bad=copy.deepcopy(record)
            if key=="bias":bad["inner_calls"][0]["convolution_bias"][0]+=.1
            if key=="mean":bad["inner_calls"][0]["normalization_parameters"]["running_mean"][0]+=.1
            if key=="helper":bad["inner_calls"][0]["helper"]="HE_Conv"
            if key=="calls":bad["inner_calls"]=[]
            if key=="work":bad["operation_count"]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):fused_conv_bn.verify_record(b,bad)

class FusedContractTests(unittest.TestCase):
    def request(self,row):
        return prepare(row["model"],PROFILE_SHA256,configuration(row["configuration"]),
                       helper_profile=FUSED_PROFILE,helper_exercise=row["required_helpers"])
    def test_all_programs_and_finite_contribution(self):
        for row in cases():
            with self.subTest(case=row["name"]):
                request=self.request(row)
                validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=row["source"]),request)
    def test_equivalent_parameter_binding_is_not_candidate_editable(self):
        request=self.request(cases()[0])
        for field in ("original","derived","input","family","work"):
            bad=copy.deepcopy(request);b=bad["upstream_helpers"]["helpers"]["HE_ConvBN0"]["binding"]
            if field=="original":b["original"]["bias"][0]+=.1
            if field=="derived":b["calls"][0]["inner"]["parameters"]["batch_norm"]["running_mean"][0]+=.1
            if field=="input":b["input_value"]=b["output_value"]
            if field=="family":b["helper"]="HE_DwConv"
            if field=="work":b["work"]=1
            bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.subTest(field=field),self.assertRaises(ValueError):validate_request(bad)
    def test_old_profile_and_default_do_not_enable_fused_helpers(self):
        g=corpus()[0][0]
        self.assertFalse(any(k.startswith(("HE_ConvBN","HE_DwConv")) for k in manifest(g,MAPPED_PROFILE)["helpers"]))
        self.assertNotIn("upstream_helpers",prepare(g,PROFILE_SHA256))
        with self.assertRaises(ValueError):prepare(g,PROFILE_SHA256,helper_profile=FUSED_PROFILE,chunk_period=4)
    def test_cancellation_is_not_helper_contribution(self):
        request=self.request(cases()[0])
        source='@hc.func("c,c")'+chr(10)+'def golden(x,zero_ct):'+chr(10)+'    h = HE_ConvBN0(x)'+chr(10)+'    return h-h+x'+chr(10)
        with self.assertRaisesRegex(ValueError,"contribution"):
            validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
    def test_compact_trace_payload_preserves_content_and_original_size_gate(self):
        import json
        from candidate_contract import trace_payload_json,strict_json,MAX_BYTES
        row=next(r for r in cases() if r["name"]=="fused_asymmetric_5")
        request=self.request(row)
        candidate=dict(schema=1,request_id=request["request_id"],hecate_source=row["source"])
        payload=dict(request=request,candidate=candidate)
        encoded=trace_payload_json(request,candidate)
        self.assertEqual(strict_json(encoded),payload)
        self.assertLess(len(encoded.encode()),MAX_BYTES)
        self.assertGreater(len(json.dumps(payload,indent=2).encode()),MAX_BYTES)
        with self.assertRaisesRegex(ValueError,"size limit"):
            trace_payload_json(request,dict(hecate_source="x"*MAX_BYTES))
        with self.assertRaises(ValueError):trace_payload_json(dict(value=float("nan")),candidate)

    def test_depthwise_requires_exact_channel_groups(self):
        g=corpus()[4][0];bn=next(n for n in g["nodes"] if n["op"]=="batch_norm")
        with self.assertRaisesRegex(ValueError,"filter per input"):fused_conv_bn.bind_node(g,bn["id"],layout(g)["input_slot_period"],"HE_DwConv")
