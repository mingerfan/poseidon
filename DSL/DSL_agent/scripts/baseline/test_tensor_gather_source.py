"""Explicit v37 policy and old protocol compatibility."""
import json,unittest
from pathlib import Path
import torch
from pytorch_ordered_contract import package_version,reference_version,SOURCE_FORMATS
from pytorch_model_intake import ordered
from provenance_json_transport import source_selected as enabled
from pytorch_public_expression_capture import capture_format,source_policy
from gather_fx_normalization import POLICY,metadata
from component_contract import prepare_task,reconstruct_request,request_options

class GatherSourcePolicyTests(unittest.TestCase):
    def test_new_protocol_is_explicit_and_complete(self):
        spec=dict(format="poseidon-trusted-pytorch-v37")
        self.assertTrue(ordered(spec));self.assertIn(spec["format"],SOURCE_FORMATS)
        self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v37")
        self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v37")
        self.assertEqual(capture_format(spec),"poseidon-public-expression-source-capture-v14")
        self.assertTrue(enabled(spec))
    def test_parent_protocol_stays_parent(self):
        spec=dict(format="poseidon-trusted-pytorch-v36")
        self.assertTrue(ordered(spec));self.assertTrue(enabled(spec))
        self.assertEqual(package_version(spec),"poseidon-trusted-pytorch-provenance-v36")
        self.assertEqual(reference_version(spec),"poseidon-trusted-pytorch-reference-v36")
        self.assertNotEqual(capture_format(spec),"poseidon-public-expression-source-capture-v14")
        self.assertEqual(source_policy(spec),source_policy(dict(format="poseidon-trusted-pytorch-v37")))
    def test_real_metadata_checks_rank_axes_and_dtype(self):
        result=metadata("call_function",torch.gather,
            (torch.empty((2,4),device="meta",dtype=torch.float64),-1,
             torch.empty((2,3),device="meta",dtype=torch.int64)),{},
            resolve_value=lambda value:value,resource_profile="legacy-v1")
        self.assertEqual(tuple(result.shape),(2,3));self.assertEqual(result.dtype,torch.float64)
        with self.assertRaises(ValueError):
            metadata("call_method","gather",(torch.empty((2,4),device="meta",dtype=torch.float64),1,
                torch.empty((2,3),device="meta",dtype=torch.int32)),{},
                resolve_value=lambda value:value,resource_profile="legacy-v1")
    def test_wrong_nonindex_dimension_rejected(self):
        with self.assertRaises(ValueError):
            metadata("call_function",torch.gather,(torch.empty((2,4),device="meta",dtype=torch.float64),1,
                torch.empty((3,1),device="meta",dtype=torch.int64)),{},
                resolve_value=lambda value:value,resource_profile="legacy-v1")
    def test_old_requests_and_hashes_stay_exact(self):
        base=Path(__file__).resolve().parent
        for relative in ("benchmarks/tools/linear_phase_live_v1/prepared/wide_tail_linear/request.json",
                         "benchmarks/tools/provenance_transport_live_v2/prepared/rank_permute_32/request.json"):
            old=json.loads((base/relative).read_bytes())
            self.assertEqual(reconstruct_request(old),old)
            self.assertEqual(prepare_task(old["model"],request_options(old)),old)
    def test_unrelated_call_does_not_gain_metadata_capability(self):
        self.assertIsNone(metadata("call_function",torch.sin,(),{},
            resolve_value=lambda value:value,resource_profile="legacy-v1"))
if __name__=="__main__":unittest.main(verbosity=2)
