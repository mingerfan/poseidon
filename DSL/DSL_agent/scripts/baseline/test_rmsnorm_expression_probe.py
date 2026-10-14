"""Equivalent-expression tests for the original three-step RMSNorm fixture."""
import copy
import json
import unittest
from pathlib import Path
from model_decomposition import decompose
from compiler_configuration import configuration
from unified_graph_contract import prepare, validate_candidate
from rmsnorm_expression_probe import variants, plaintext_check, MODEL_SHA256
from benchmark_graph import digest

class RmsnormExpressionTests(unittest.TestCase):
    def request(self):
        high=json.loads((Path(__file__).parent/"cases/operator-decomposition-v1/rmsnorm.json").read_text())
        model,_=decompose(high)
        self.assertEqual(digest(model),MODEL_SHA256)
        c=configuration("poseidon-cpu-eva-w45-v1")
        return prepare(model,c["ckks_config_sha256"],c)
    def test_all_variants_keep_request_and_match_two_references(self):
        r=self.request(); before=copy.deepcopy(r)
        for name,source in variants(r).items():
            with self.subTest(name=name):
                validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
                self.assertEqual(plaintext_check(source,r)["probes"],16)
        self.assertEqual(r,before)
    def test_different_model_not_silently_substituted(self):
        r=self.request();r["model"]["constants"]["gain"][0]=2.
        with self.assertRaisesRegex(ValueError,"original RMSNorm"):variants(r)
    def test_output_mask_change_rejected(self):
        r=self.request();r["public_constants"]["mask1"]=[1.,0.,0.,0.]
        with self.assertRaisesRegex(ValueError,"selectors"):variants(r)
    def test_probe_is_manual_and_keeps_three_iterations(self):
        r=self.request();rows=variants(r)
        self.assertEqual(len(rows),4)
        self.assertIn("h3",rows["packed-original"])
        self.assertIn("q2",rows["packed-public-first"])

if __name__=="__main__":unittest.main()
