"""Bounded duplicate-result handling for trusted baselines, no provider rewriting."""
import json,unittest
from pathlib import Path
import numpy as np
from benchmark_graph import samples,canonical
from benchmark_math import evaluate
from component_contract import prepare_task,reconstruct_request
from unified_capacity_constants import INDEX_VERSION
from unified_graph_lowering import candidate_source
from capacity_output_identity import distinct_outputs
from test_unified_capacity_linear import interpreted_output
from hecate_python_env import WORK

class OutputIdentityTests(unittest.TestCase):
    def models(self):
        return [json.loads(p.read_text()) for p in (Path(__file__).parent/"cases/capacity-output-identity-v1").glob("*.json")]
    def test_three_repeated_result_contexts_match_independent_math(self):
        folder=WORK/"results/agent-pytorch-index-backend-r754/imports/axes_2"
        models=self.models()+[json.loads((folder/"model.json").read_text())]
        self.assertEqual(len(models),3)
        for model in models:
            request=prepare_task(model,dict(chunk_period=4,capacity_packing_version=INDEX_VERSION))
            before=canonical(request);source,record=candidate_source(request)
            self.assertEqual(record["output_identity"]["scope"],"rule_baseline_only")
            self.assertEqual(canonical(request),before)
            self.assertEqual(canonical(reconstruct_request(request)),before)
            for probe in samples(model,16):
                expected=evaluate(model,probe)
                np.testing.assert_allclose(interpreted_output(source,request,probe),
                    np.concatenate([expected[o["name"]].reshape(-1) for o in model["outputs"]]),atol=1e-12,rtol=1e-12)
    def test_distinct_source_bytes_unchanged(self):
        source="@hc.func('c,c')\ndef golden(x,zero_ct):\n    return [x,x.square()]\n"
        self.assertEqual(distinct_outputs(source),(source,None))
    def test_named_aliases_and_repeated_arithmetic_are_detected(self):
        source="@hc.func('c,c')\ndef golden(x,zero_ct):\n    a=x*x\n    b=x*x\n    return [a,b,b]\n"
        result,record=distinct_outputs(source)
        self.assertEqual([v["encrypted_zero_additions"] for v in record["bindings"]],[1,2])
        self.assertIn("b + zero_ct + zero_ct",result)
    def test_missing_encrypted_zero_input_rejected(self):
        with self.assertRaisesRegex(ValueError,"Bound encrypted zero"):
            distinct_outputs("@hc.func('c')\ndef golden(x):\n    return [x,x]\n")
    def test_alias_cycle_is_not_executed(self):
        with self.assertRaisesRegex(ValueError,"alias cycle"):
            distinct_outputs("@hc.func('c,c')\ndef golden(x,zero_ct):\n    a=b\n    b=a\n    return [a,b]\n")

if __name__=="__main__":unittest.main()
