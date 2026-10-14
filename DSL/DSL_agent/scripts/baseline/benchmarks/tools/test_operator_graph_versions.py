"""Dev-tool guard tests; no tracing, FHE or API."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from benchmarks.tools import audit_operator_graph_versions as audit

class VersionAuditTests(unittest.TestCase):
    def test_signature_ignores_identifiers_and_weights(self):
        item=audit.fixture("rms_norm",1)
        changed=copy.deepcopy(item["model"])
        changed["id"]="other_model"
        changed["nodes"][0]["id"]="other_node"
        changed["constants"]["gain"]=[.25,-.125,.375,.5]
        self.assertEqual(audit.structural_signature(item["model"]),audit.structural_signature(changed))
        changed["inputs"][0]["shape"]=[1,4]
        self.assertNotEqual(audit.structural_signature(item["model"]),audit.structural_signature(changed))

    def test_all_registered_operators_have_distinct_planned_contexts(self):
        rows,counts=audit.cases()
        self.assertEqual(set(counts),set(audit.REGISTRY))
        self.assertTrue(all(len(v)>=3 for v in counts.values()))
        self.assertTrue(all(row["model"]["format"]==audit.V1 for row in rows))
        self.assertTrue(any(row["origin"]=="existing_published_graph" for row in rows))
        self.assertTrue(any(row["origin"]=="supplemental_reference_context" for row in rows))

    def test_wrong_independent_tensor_result_cannot_pass(self):
        original=audit.tensor_reference
        def corrupted(model,inputs):
            return {key:value+.125 for key,value in original(model,inputs).items()}
        with patch.object(audit,"tensor_reference",side_effect=corrupted):
            with self.assertRaises(AssertionError):
                audit.check(audit.fixture("exp",0),probes=4)

    def test_nonfinite_tensor_reference_cannot_pass(self):
        original=audit.tensor_reference
        def nonfinite(model,inputs):
            return {key:np.full(value.shape,np.nan) for key,value in original(model,inputs).items()}
        with patch.object(audit,"tensor_reference",side_effect=nonfinite):
            with self.assertRaisesRegex(ValueError,"shape/nonfinite"):
                audit.check(audit.fixture("dropout",0),probes=4)

    def test_source_graph_remains_unchanged_under_version_check(self):
        item=audit.fixture("reciprocal",1)
        snapshot=copy.deepcopy(item["model"])
        result=audit.check(item,probes=4)
        self.assertEqual(item["model"],snapshot)
        self.assertEqual(result["status"],"passed")
        self.assertTrue(result["core_graph_bytes_equal"])
        self.assertFalse(result["compiled"])
        self.assertFalse(result["encrypted_execution"])
        self.assertFalse(result["agent_generated"])

    def test_changed_reference_runtime_or_proposal_changes_binding(self):
        rows=[audit.fixture("exp",0)]
        source={"code.py":"a"*64};runtime=dict(python=[3,10,14],numpy="1.25.2",torch="2.0.1",
                                              files={"native.so":"b"*64})
        proposal=dict(binding="c"*64,matched=True)
        first=audit.audit_binding(source,"d"*64,{},rows,runtime,proposal)
        changed=copy.deepcopy(runtime);changed["files"]["native.so"]="e"*64
        self.assertNotEqual(first,audit.audit_binding(source,"d"*64,{},rows,changed,proposal))
        other=copy.deepcopy(proposal);other["binding"]="f"*64
        self.assertNotEqual(first,audit.audit_binding(source,"d"*64,{},rows,runtime,other))

    def test_registry_fixture_typo_fails_without_fallback(self):
        with self.assertRaisesRegex(ValueError,"No supplemental fixture"):
            audit.fixture("unknown_operator",0)

if __name__=="__main__":
    unittest.main()
