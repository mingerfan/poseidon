"""Metadata adoption must not rewrite task definitions or grant execution credit."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import enrich_semantic_ledger as subject
class LedgerAdoptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite=subject.BASE/"benchmarks/semantic-v1-chunk-helpers-r29"
        cls.rows,cls.index,cls.enriched,cls.components=subject.build(cls.suite)
        cls.old=subject.strict_file(cls.suite/"coverage.json",4*1024**2)
    def test_complete_partition_metadata_preserves_denominators(self):
        e=self.enriched
        self.assertEqual(len(e["requirements"]),401)
        self.assertEqual(len(self.rows),1200)
        self.assertEqual(len(self.components["math_contexts"]["supplemental_models"]),92)
        for r in e["requirements"]:
            self.assertTrue(r["source_basis"] and r["negative_examples"] and r["applicable_contracts"])
            if r["layer"]!="rejection" and not r["blocker"]:
                self.assertGreaterEqual(len({p["topology"] for p in r["positive_examples"]}),3)
    def test_changed_original_task_rejected(self):
        e=copy.deepcopy(self.enriched);e["directed_tasks"][0]["instruction"]="different contract"
        with self.assertRaisesRegex(ValueError,"Original tasks"):subject.validate_enrichment(e,self.old)
    def test_removed_requirement_rejected(self):
        e=copy.deepcopy(self.enriched);e["requirements"].pop()
        with self.assertRaisesRegex(ValueError,"denominator"):subject.validate_enrichment(e,self.old)
    def test_false_execution_credit_rejected(self):
        e=copy.deepcopy(self.enriched);e["evidence_counts"]["agent_cases"]=1
        with self.assertRaisesRegex(ValueError,"states changed"):subject.validate_enrichment(e,self.old)
    def test_helper_negatives_cover_every_binding(self):
        negatives=self.components["helper_negatives"]["fixtures"]
        expected={(t["id"],name) for t in self.old["helper_directed_tasks"] for name in t["required_helpers"]}
        self.assertEqual({(t["positive_task_id"],t["omitted_helper"]) for t in negatives},expected)
        self.assertEqual(len(negatives),76)
    def test_bootstrap_stays_blocked_and_api_inventory_is_not_execution(self):
        blocked=[r for r in self.enriched["requirements"] if r["blocker"]=="bootstrap_backend"]
        self.assertEqual(len(blocked),3)
        self.assertTrue(all(r["applicable_contracts"]==["blocked_real_bootstrap"] for r in blocked))
        self.assertTrue(all(r["execution_evidence"]=="not_individually_audited"
                            for r in self.enriched["upstream_api"]))
if __name__=="__main__":unittest.main()
