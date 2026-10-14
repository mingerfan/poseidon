"""Suffix stage contracts and independent coordinates; no API or FHE."""
import copy,json,math,unittest
from pathlib import Path
import numpy as np
from unified_suffix_stage_guidance import VERSION,suffix_context,WINDOW,NORMALIZATION
from component_contract import prepare_task,request_options,reconstruct_request
from unified_graph_contract import validate_request
from application_diagnostics import attach
B=Path(__file__).resolve().parent
def old_request(case="rsqrt_mean"):
    plan=json.loads((B/"benchmarks/tools/source_preparation_live_v1/live-plan.json").read_text())
    row=next(c for c in plan["cases"]if c["id"]==case)
    return json.loads(Path(row["request_file"]).read_text())
def request(model=None,period=None,version=VERSION):
    old=old_request();opts=request_options(old);opts["generation_guidance"]=version
    if period is not None:opts["chunk_period"]=period
    return prepare_task(old["model"]if model is None else model,opts)
def graph(shape,operator):
    return dict(format="poseidon-model-graph-v2",id="suffix_context",resource_profile="general-tensors-v1",
      inputs=[dict(name="x",shape=list(shape))],constants={},
      nodes=[dict(id="reduce",op=operator,inputs=["x"],attrs=dict(axes=[-1],keepdims=True),outputs=["reduced"])],
      outputs=[dict(name="out",value="reduced")])
class SuffixStageTests(unittest.TestCase):
    def test_original_failure_has_explicit_complete_stage_and_normalization(self):
        r=request();c=r["generation_guidance"]["bound_suffix_reduction_stages"]
        row=c["examples"][0]
        self.assertEqual((row["row_width"],row["rows"],row["mask_normalization"]),(2,2,.5))
        self.assertEqual(row["gather_source"],"complete_window_sum")
        self.assertEqual(row["gather_output"],"compact_rows")
        self.assertTrue(row["divisor_already_in_masks"])
        self.assertNotIn("hecate_source",json.dumps(c));self.assertNotIn("golden(",json.dumps(c))
    def test_three_contexts_match_independent_rows_and_reject_missing_window(self):
        contexts=[((2,2),"mean",4),((2,3),"sum",8),((3,4),"mean",4)]
        for shape,operator,period in contexts:
            with self.subTest(shape=shape,operator=operator):
                r=request(graph(shape,operator),period)
                ctx=suffix_context(r);self.assertIsNotNone(ctx)
                spec=ctx["examples"][0];width=shape[-1];rows=math.prod(shape)//width
                self.assertEqual((spec["row_width"],spec["rows"]),(width,rows))
                values=np.arange(math.prod(shape),dtype=np.float64)*.7-.3
                chunks=[np.pad(values[i:i+period],(0,max(0,period-len(values[i:i+period]))))
                        for i in range(0,len(values),period)]
                out=np.zeros(math.ceil(rows/period)*period)
                wrong=np.zeros_like(out)
                bindings=r["constant_origins"]["capacity_tensor_packing"]["bindings"]
                for v in bindings:
                    if v["role"]!="suffix_gather":continue
                    source=chunks[v["input_chunk"]]
                    window=np.array([sum(source[(s+i)%period]for i in range(width))for s in range(period)])
                    mask=r["public_constants"][v["key"]];baby=v["baby_step"];giant=v["giant_step"]
                    for d in range(period):
                        dst=v["output_chunk"]*period+d
                        out[dst]+=mask[(d+giant)%period]*window[(d+baby+giant)%period]
                        wrong[dst]+=mask[(d+giant)%period]*source[(d+baby+giant)%period]
                expected=np.array([sum(float(values[row*width+i])for i in range(width))/
                      (width if operator=="mean"else 1)for row in range(rows)])
                np.testing.assert_allclose(out[:rows],expected,atol=1e-12,rtol=1e-12)
                self.assertFalse(np.allclose(wrong[:rows],expected,atol=1e-12,rtol=1e-12))
                if operator=="mean":
                    self.assertFalse(np.allclose(out[:rows]/width,expected))
    def test_rank_three_keeps_nonreduced_coordinates(self):
        r=request(graph((2,2,2),"mean"),8);c=suffix_context(r)["examples"][0]
        self.assertEqual(c["rows"],4);self.assertEqual(c["reduced_axes"],[2])
        self.assertEqual(c["output_shape"],[2,2,1])
    def test_wrong_public_geometry_and_operation_rejected(self):
        r=request()
        cases=[]
        a=copy.deepcopy(r);next(iter(a["constant_origins"]["capacity_tensor_packing"]["suffix_windows"].values()))["rows"]=3;cases.append(a)
        a=copy.deepcopy(r);key=next(iter(a["constant_origins"]["capacity_tensor_packing"]["suffix_windows"]))
        next(n for n in a["model"]["nodes"]if n["id"]==key)["op"]="negate";cases.append(a)
        for value in cases:
            with self.assertRaises(ValueError):suffix_context(value)
    def test_prior_versions_keep_identical_request_bytes_and_reconstruction(self):
        old=old_request();validate_request(old)
        self.assertEqual(reconstruct_request(old),old)
        newer=request();validate_request(newer)
        self.assertEqual(reconstruct_request(newer),newer)
        self.assertNotEqual(old["request_id"],newer["request_id"])
        for key,value in old.items():
            if key not in("request_id","generation_guidance"):
                self.assertEqual(newer[key],value,key)
        previous=request(version="explicit-v19");self.assertNotIn("bound_suffix_reduction_stages",previous["generation_guidance"])
    def test_feedback_uses_only_public_window_rules(self):
        r=request();src=r["generation_guidance"]["entry_header"]+"\n    return x0\n"
        d=attach(dict(status="failed",failure=dict(layer="numerical_comparison",
           diagnostic="PRIVATE_REFERENCE /private/reference.npy")),candidate=dict(schema=1,
           request_id=r["request_id"],hecate_source=src),request=r)
        diagnostic=d["public_feedback"]["diagnostic"]
        self.assertIn("window aggregate",diagnostic);self.assertIn("mask already contains",diagnostic)
        self.assertNotIn("PRIVATE_REFERENCE",diagnostic);self.assertNotIn("/private/reference.npy",diagnostic)
    def test_context_is_not_added_to_models_without_suffix_strategy(self):
        old=old_request("gelu_base");opts=request_options(old);opts["generation_guidance"]=VERSION
        r=prepare_task(old["model"],opts)
        self.assertNotIn("bound_suffix_reduction_stages",r["generation_guidance"])
    def test_forged_annotations_are_rejected(self):
        r=request();r["generation_guidance"]["bound_suffix_reduction_stages"]["examples"][0]["row_width"]=1
        with self.assertRaises(ValueError):validate_request(r)
if __name__=="__main__":unittest.main(verbosity=2)
