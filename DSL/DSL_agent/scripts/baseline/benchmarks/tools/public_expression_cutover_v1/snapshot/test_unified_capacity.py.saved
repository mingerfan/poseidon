"""Logical/physical capacity boundaries, independent references and legacy identities."""
import copy,json,math,unittest
from pathlib import Path
from benchmark_graph import validate,CAPACITY_FORMAT,FORMAT,samples,canonical
from component_contract import prepare_task,reconstruct_request
from unified_graph_contract import validate_request,validate_candidate
from unified_capacity_layout import layout,validate_layout,ABI
from unified_capacity_cases import models

class CapacityTests(unittest.TestCase):
    def test_version_required(self):
        g=models()[0];g.pop("resource_profile")
        with self.assertRaises(ValueError):validate(g)
    def test_legacy_cannot_claim_profile(self):
        g=models()[0];g["format"]=FORMAT
        with self.assertRaises(ValueError):validate(g)
    def test_explicit_profile_conflict(self):
        with self.assertRaisesRegex(ValueError,"Conflicting"):validate(models()[0],resource_profile="legacy-v1")
    def test_logical_overflow(self):
        g=models()[0];g["inputs"][0]["shape"]=[1025]
        with self.assertRaisesRegex(ValueError,"Total input"):validate(g)
    def test_original_v2_resnet_remains_physically_bounded(self):
        g=models()[0];g["resource_profile"]="resnet20-v1";g["inputs"][0]["shape"]=[4096]
        validate(g)
        with self.assertRaisesRegex(ValueError,"bounded physical layout"):layout(g,execution_abi=ABI)
    def test_too_small_period(self):
        with self.assertRaisesRegex(ValueError,"input ciphertext"):layout(models()[0],4)
    def test_offset_and_selector_are_bound(self):
        for mutate in (lambda x:x["inputs"][1].update(offset=0),
                       lambda x:x["output_selectors"].__setitem__(256,[0,0]),
                       lambda x:x["outputs"][0]["chunks"][1].update(elements=2)):
            v=layout(models()[0],256);mutate(v)
            with self.assertRaisesRegex(ValueError,"Changed capacity physical"):validate_layout(v)
    def test_resource_identity_is_bound(self):
        v=layout(models()[0],256);v["resource_profile"]["input_elements"]=4096
        with self.assertRaisesRegex(ValueError,"Changed capacity resource"):validate_layout(v)
    def test_tail_masks_include_bias_padding(self):
        g=next(g for g in models() if g["id"]=="capacity_mean_tail_1")
        r=prepare_task(g,{"chunk_period":256})
        self.assertEqual(r["public_constants"]["tail1"],[1.]+[0.]*255)
        self.assertEqual(r["layout"]["logical_inputs"][0]["shape"],[3,171])
        self.assertEqual(len(r["layout"]["inputs"]),3)
    def test_source_output_arity_remains_physical(self):
        g=next(g for g in models() if g["id"]=="capacity_reshape_output_2");r=prepare_task(g,{"chunk_period":256})
        wrong='@hc.func("c,c,c,c,c")\ndef golden(x,y,z,t,zero_ct):\n    return [x]\n'
        with self.assertRaises(ValueError):validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=wrong),r)
    def test_capacity_layout_auto_matches_explicit(self):
        for g in models():
            r=prepare_task(g)
            self.assertLessEqual(len(r["layout"]["inputs"]),4)
            self.assertLessEqual(r["layout"]["output_ciphertexts"],4)
            self.assertEqual(validate_request(r),r["layout"]["input_slot_period"])
            self.assertEqual(canonical(reconstruct_request(r)),canonical(r))
    def test_each_partition_has_three_distinct_contexts(self):
        from collections import Counter
        rows=models()
        self.assertEqual(len(rows),18)
        c=Counter(g["id"].rsplit("_",1)[0] for g in rows)
        self.assertEqual(set(c.values()),{3})
        self.assertTrue(all(sum(math.prod(x["shape"]) for x in g["inputs"])>256 for g in rows))
    def test_independent_references(self):
        import numpy as np
        from benchmark_math import evaluate as math_ref
        from benchmark_torch import evaluate as torch_ref
        for g in models():
            for probe in samples(g,16):
                a=math_ref(g,probe);b=torch_ref(g,probe)
                self.assertEqual(set(a),set(b))
                for n in a:np.testing.assert_allclose(a[n],b[n],atol=1e-12,rtol=1e-12)
    def test_previous_poseidon_requests_unchanged(self):
        from hecate_python_env import WORK
        for name in ("poseidon-historical-agent-replay-r371","poseidon-historical-helper-replay-r375"):
            report=json.loads((WORK/"results"/name/"report.json").read_text())
            for row in report["rows"]:
                r=json.loads((Path(row["program"])/"request.json").read_text())
                self.assertEqual(canonical(reconstruct_request(r)),canonical(r))

if __name__=="__main__":unittest.main()
