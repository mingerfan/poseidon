"""Explicit shallow packing, independent numeric maps and old request identity."""
import copy,json,math,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,digest,samples,validate
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from component_contract import prepare_task,reconstruct_request,runner_options
from unified_capacity_constants import INDEX_VERSION,SHALLOW_VERSION,packing
from unified_capacity_routes import aligned_suffix_plan
from unified_graph_contract import validate_request
from unified_graph_lowering import candidate_source
from test_unified_capacity_linear import interpreted_output

BASE=Path(__file__).resolve().parent
CASES=BASE/"benchmarks/tools/logical_view_backend_v1/prepared"

def original(name):
    return json.loads((CASES/name/"request.json").read_text())

def current(name):
    old=original(name)
    # Keep the exact frozen configuration and optional binding policies.
    from component_contract import request_options
    return old,prepare_task(old["model"],dict(request_options(old),capacity_packing_version=SHALLOW_VERSION))

class ShallowPackingTests(unittest.TestCase):
    def test_original_requests_reconstruct_byte_for_byte(self):
        for name in ("rank_axis_8","rank_linear_16","rank_permute_32"):
            old=original(name)
            self.assertEqual(canonical(old),canonical(reconstruct_request(old)))
            self.assertEqual(old["constant_origins"]["capacity_tensor_packing"]["version"],INDEX_VERSION)

    def test_new_packing_is_explicit_bound_and_inherits_codec_and_cli(self):
        from candidate_contract import request_file_bytes
        from run_candidate import parse_args
        for name in ("rank_axis_8","rank_linear_16","rank_permute_32"):
            old,q=current(name)
            self.assertNotEqual(q["request_id"],old["request_id"])
            self.assertEqual(q["model"],old["model"]);self.assertEqual(q["layout"],old["layout"])
            self.assertEqual(canonical(q),canonical(reconstruct_request(q)))
            wire=json.loads(request_file_bytes(q));validate_request(wire)
            self.assertEqual(canonical(q),canonical(wire))
            args=runner_options(q)
            self.assertIn(SHALLOW_VERSION,args)
            self.assertEqual(parse_args(["--case","unused.json","--prepare",*args]).unified_packing,SHALLOW_VERSION)

    def test_original_failures_use_bounded_shallow_routes_and_independent_reference(self):
        for name in ("rank_axis_8","rank_linear_16"):
            _,q=current(name);src,info=candidate_source(q)
            self.assertEqual(info["strategy"],"capacity-chunks-v11")
            self.assertLessEqual(len(q["public_constants"]),256)
            if name=="rank_linear_16":
                pk=q["constant_origins"]["capacity_tensor_packing"]
                self.assertEqual(set(pk["linear_strategies"].values()),{"diagonal"})
                self.assertIn("aligned_suffix",pk["tensor_strategies"].values())
                self.assertIn("For linear_diagonal",q["rules"])
                self.assertIn("(j+step) mod P",q["rules"])
                self.assertTrue(all(c["compiler_feasibility"]=="unknown" for c in pk["planning_costs"].values()))
            for probe in samples(q["model"],4):
                expected=mathematical(q["model"],probe);other=torch_reference(q["model"],probe)
                for key in expected:np.testing.assert_allclose(expected[key],other[key],atol=1e-12,rtol=1e-12)
                flat=np.concatenate([expected[o["name"]].reshape(-1) for o in q["model"]["outputs"]])
                np.testing.assert_allclose(interpreted_output(src,q,probe),flat,atol=1e-12,rtol=1e-12)

    def test_complete_suffix_including_nonpowers_and_tails(self):
        for period in (4,8,16,32):
            for width in range(1,period+1):
                for rows in (1,2,3,7):
                    shape=[rows,width]
                    node=dict(id="r",op="mean",inputs=["x"],outputs=["y"],attrs=dict(axes=[-1],keepdims=False))
                    shapes={"x":shape,"y":[rows]}
                    legacy=aligned_suffix_plan(node,shapes,period)
                    if width&(width-1):self.assertIsNone(legacy)
                    plan=aligned_suffix_plan(node,shapes,period,allow_complete_rows=True)
                    complete=all((i*width)//period==(i*width+width-1)//period for i in range(rows))
                    self.assertEqual(plan is not None,complete)
                    if plan is None:continue
                    raw=np.arange(rows*width,dtype=float)-2.5
                    chunks=[np.pad(raw[start:start+period],(0,period-len(raw[start:start+period])),constant_values=173.)
                            for start in range(0,len(raw),period)]
                    windows=[sum(np.roll(v,-j) for j in range(width)) for v in chunks]
                    actual=np.zeros(math.ceil(rows/period)*period)
                    for (_,source,target,step),mask in plan["routes"].items():
                        actual[target*period:(target+1)*period]+=np.roll(windows[source],-step)*np.array(mask)
                    np.testing.assert_allclose(actual[:rows],raw.reshape(rows,width).mean(axis=-1),atol=1e-12,rtol=1e-12)

    def test_planning_or_binding_tampering_is_rejected(self):
        _,q=current("rank_linear_16")
        key=next(iter(q["constant_origins"]["capacity_tensor_packing"]["planning_costs"]))
        bad=copy.deepcopy(q);bad["constant_origins"]["capacity_tensor_packing"]["planning_costs"][key]["compiler_feasibility"]="passed"
        bad["request_id"]=digest({k:v for k,v in bad.items() if k!="request_id"})
        with self.assertRaisesRegex(ValueError,"Changed capacity public tensor"):validate_request(bad)

if __name__=="__main__":unittest.main()
