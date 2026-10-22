"""Versioned static-index routes and preservation of historical packing."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,digest,samples
from benchmark_math import evaluate as mathematical
from component_contract import prepare_task,reconstruct_request,runner_options
from unified_capacity_constants import CHANNEL_VERSION,INDEX_VERSION,packing
from unified_graph_contract import validate_request
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output
from test_pytorch_index_capture import fixture,captured,CASES
from hecate_python_env import WORK

class StaticIndexPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records={}
        for path in sorted(CASES.glob("*.json")):
            spec=json.loads(path.read_text())
            _,model,_=captured(fixture().make(path.stem),spec["inputs"][0]["shape"],spec["outputs"])
            options=dict(chunk_period=4,generation_guidance="explicit-v10",capacity_packing_version=INDEX_VERSION)
            request=prepare_task(model,options)
            cls.records[path.stem]=(model,request)
    def test_fifteen_routes_and_generated_arithmetic_match_independent_reference(self):
        self.assertEqual(len(self.records),15)
        for name,(model,request) in self.records.items():
            with self.subTest(case=name):
                source,info=candidate_source(request)
                self.assertEqual(info["strategy"],"capacity-chunks-v10")
                for probe in samples(model,4):
                    expected=mathematical(model,probe)
                    actual=interpreted_output(source,request,probe)
                    np.testing.assert_allclose(actual,np.concatenate([expected[o["name"]].reshape(-1)
                        for o in model["outputs"]]),atol=1e-12,rtol=1e-12)
    def test_request_reconstruction_codec_and_cli_select_v9(self):
        from candidate_contract import request_file_bytes
        from run_candidate import parse_args
        for name,(_,request) in self.records.items():
            self.assertEqual(canonical(reconstruct_request(request)),canonical(request))
            wire=json.loads(request_file_bytes(request));validate_request(wire)
            self.assertEqual(canonical(wire),canonical(request))
            options=runner_options(request)
            self.assertEqual(options[options.index("--unified-packing")+1],INDEX_VERSION)
            args=parse_args(["--case",str(CASES/(name+".json")),"--prepare",*options])
            self.assertEqual(args.unified_packing,INDEX_VERSION)
    def test_masks_and_chunk_route_tampering_rejected_after_envelope_rehash(self):
        _,request=self.records["axes_0"]
        for field in ("step","baby_step","giant_step","input_chunk","output_chunk"):
            bad=copy.deepcopy(request)
            row=next(b for b in bad["constant_origins"]["capacity_tensor_packing"]["bindings"] if b["role"]=="tensor_route")
            row[field]+=1;bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
            with self.assertRaisesRegex(ValueError,"Changed capacity public tensor"):validate_request(bad)
        bad=copy.deepcopy(request)
        key=next(b["key"] for b in bad["constant_origins"]["capacity_tensor_packing"]["bindings"] if b["role"]=="tensor_route")
        bad["public_constants"][key][0]+=.25
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Public registry changed"):validate_request(bad)
    def test_old_default_v8_keeps_small_slice_closed(self):
        model,_=self.records["axes_0"]
        request=prepare_task(model,dict(chunk_period=4,generation_guidance="explicit-v10"))
        with self.assertRaisesRegex(ValueError,"No bounded capacity static route"):candidate_source(request)
    def test_six_existing_v8_requests_and_rule_sources_are_unchanged(self):
        batch=WORK/"results/agent-normalization-lowering-backend-r743"
        rows=json.loads((batch/"report.json").read_text())["rows"]
        self.assertEqual(len(rows),6)
        for row in rows:
            package=Path(row["package"]);request=json.loads((package/"request.json").read_text())
            validate_request(request);self.assertEqual(request["request_id"],row["request_id"])
            self.assertEqual(canonical(reconstruct_request(request)),canonical(request))
            source,_=candidate_source(request)
            self.assertEqual(source,(package/"candidate.py").read_text())
    def test_wide_v9_is_explicit_and_preserves_original_v8_default(self):
        from test_unified_capacity_wide import cases
        for model,period in cases():
            original=prepare_task(model,dict(chunk_period=period))
            request=prepare_task(model,dict(chunk_period=period,capacity_packing_version=INDEX_VERSION))
            self.assertEqual(original["constant_origins"]["capacity_tensor_packing"]["version"],CHANNEL_VERSION)
            self.assertEqual(request["constant_origins"]["capacity_tensor_packing"]["version"],INDEX_VERSION)
            self.assertNotEqual(request["request_id"],original["request_id"])
            self.assertEqual(canonical(reconstruct_request(request)),canonical(request))
    def test_new_version_is_bound_even_without_tensor_routes(self):
        model=copy.deepcopy(self.records["slicing_2"][0])
        model["nodes"]=[dict(id="identity",op="reshape",inputs=["x"],outputs=["identity_out"],attrs={"shape":[4]})]
        model["outputs"]=[dict(name="out",value="identity_out")]
        request=prepare_task(model,dict(chunk_period=4,capacity_packing_version=INDEX_VERSION))
        self.assertEqual(request["constant_origins"]["capacity_tensor_packing"]["version"],INDEX_VERSION)
        self.assertEqual(canonical(reconstruct_request(request)),canonical(request))
    def test_v9_inherits_lossless_tiled_public_storage(self):
        from unified_request_storage import compact_request,restore_public_constants,TILES_VERSION
        original=dict(layout=dict(input_slot_period=256),
            constant_origins=dict(capacity_tensor_packing=dict(version=INDEX_VERSION)),
            public_constants=dict(v=([0.]+[.125]*30+[0.])*8))
        wire=compact_request(original)
        self.assertEqual(wire["public_constants"]["v"]["format"],TILES_VERSION)
        restored=json.loads(canonical(wire));restore_public_constants(restored)
        self.assertEqual(canonical(restored),canonical(original))

if __name__=="__main__":unittest.main()
