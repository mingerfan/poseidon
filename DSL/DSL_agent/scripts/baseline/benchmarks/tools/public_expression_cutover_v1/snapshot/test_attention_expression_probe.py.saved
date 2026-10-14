"""Independent references for the original Attention/RoPE/KV expression probe."""
import copy,json,unittest
from pathlib import Path
from benchmark_graph import digest
from model_decomposition import decompose
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_candidate
from attention_expression_probe import variants,MODEL_SHA256
from rmsnorm_expression_probe import plaintext_check

class AttentionExpressionTests(unittest.TestCase):
    def request(self):
        high=json.loads((Path(__file__).parent/"cases/operator-decomposition-v1/attention-decode.json").read_text())
        model,_=decompose(high)
        self.assertEqual(digest(model),MODEL_SHA256)
        c=configuration("poseidon-cpu-eva-w45-v1")
        return prepare(model,c["ckks_config_sha256"],c)
    def test_original_three_inputs_outputs_and_all_references(self):
        request=self.request();before=copy.deepcopy(request)
        for source in variants(request).values():
            validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
            self.assertEqual(plaintext_check(source,request)["probes"],16)
        self.assertEqual(request,before)
    def test_early_score_keeps_all_outputs_and_declared_approximation(self):
        request=self.request(); source=variants(request)["packed-early-score"]
        self.assertIn("return [prediction, keys, values]",source)
        self.assertIn("residual4 = residual2 * residual2",source)
        self.assertIn("s0_high",source)
        self.assertEqual(plaintext_check(source,request)["probes"],16)
    def test_early_score_rotation_sign_error_is_rejected(self):
        request=self.request(); source=variants(request)["packed-early-score"]
        source=source.replace("cached_q = x *", "cached_q = -x *")
        with self.assertRaises(AssertionError):plaintext_check(source,request)
    def test_mutated_projection_rejected(self):
        request=self.request();request["model"]["constants"]["wq"][0][0]=.4
        with self.assertRaisesRegex(ValueError,"Original Attention"):variants(request)
    def test_missing_kv_return_is_a_reference_failure(self):
        request=self.request()
        source=next(iter(variants(request).values())).replace("return [prediction, keys, values]","return [prediction, values, keys]")
        with self.assertRaises(AssertionError):plaintext_check(source,request)

if __name__=="__main__":unittest.main()
