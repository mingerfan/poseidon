"""Host capability descriptions match the registered v37 source contract."""
import unittest
from application_capabilities import input_sources
from application_policy import describe
from pytorch_tensor_gather_source import SOURCE_FORMAT,POLICY
class HostGatherCapabilityTests(unittest.TestCase):
    def test_public_contract_reports_gather_without_global_execution_claim(self):
        source=input_sources()["trusted_pytorch"]
        self.assertIn(SOURCE_FORMAT,source["formats"])
        item=source["public_tensor_gather"]
        self.assertEqual(item["version"],SOURCE_FORMAT);self.assertEqual(item["policy"],POLICY)
        self.assertEqual(item["calls"],["torch.gather","Tensor.gather"])
        self.assertEqual(item["indices"],"fixed public int64 tensor")
        self.assertFalse(item["encrypted_indices"]);self.assertFalse(item["negative_indices"])
        self.assertEqual(item["compiler_status"],"unverified")
        self.assertEqual(item["encrypted_status"],"unverified")
        self.assertEqual(item["agent_status"],"unverified")
    def test_inherited_features_name_the_new_explicit_contract(self):
        source=input_sources()["trusted_pytorch"]
        for key in ("provenance_storage","logical_axis_views","static_broadcast","static_tensor_transforms",
                    "public_constructors","public_mask_selection","source_package"):
            self.assertIn(SOURCE_FORMAT,source[key]["versions"],key)
        self.assertIn(SOURCE_FORMAT,source["public_constructors"]["source_policies"])
    def test_application_uses_the_same_capability_description(self):
        result=describe()
        from application_capabilities import describe as direct
        self.assertEqual(result["capability_matrix"],direct())
if __name__=="__main__":unittest.main(verbosity=2)
