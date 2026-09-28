"""Verify classification completeness and important non-equivalence boundaries."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("classify_api",HERE/"classify_upstream_semantic_api.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger=json.loads((module.BASE/"benchmarks/semantic-v1-chunk-helpers-r29/coverage.json").read_text())
        cls.data=module.build(cls.ledger)
        cls.records={(Path(r["source"]).name,r["symbol"]):r for r in cls.data["records"]}

    def test_exhaustive_exact_inventory_not_execution(self):
        self.assertEqual(len(self.records),107)
        self.assertEqual(set(self.records),{(Path(r["source"]).name,r["symbol"]) for r in self.ledger["upstream_api"]})
        self.assertFalse(self.data["changes_candidate_permissions"])
        self.assertTrue(all(r["execution_evidence"]=="not_individually_audited" for r in self.records.values()))

    def test_new_symbol_fails_closed(self):
        changed=copy.deepcopy(self.ledger)
        changed["upstream_api"].append(dict(changed["upstream_api"][0],symbol="UnreviewedNewHelper"))
        with self.assertRaisesRegex(ValueError,"API drift"):module.build(changed)

    def test_missing_symbol_fails_closed(self):
        changed=copy.deepcopy(self.ledger);changed["upstream_api"].pop()
        with self.assertRaisesRegex(ValueError,"API drift"):module.build(changed)

    def test_source_and_location_changes_rejected(self):
        changed=copy.deepcopy(self.ledger)
        source=changed["upstream_api"][0]["source"]
        changed["upstream_source_hashes"][source]="0"*64
        with self.assertRaisesRegex(ValueError,"Pinned source drift"):module.build(changed)
        changed=copy.deepcopy(self.ledger);changed["upstream_api"][0]["line"]+=1
        with self.assertRaisesRegex(ValueError,"location drift"):module.build(changed)

    def test_real_bootstrap_remains_blocked(self):
        for symbol in ("HE_ReLU","HE_Max","HE_MaxPad"):
            self.assertEqual(self.records[("Func.py",symbol)]["backend_blocker"],"real_bootstrap")
        for symbol in ("maxx","maxx.sign"):
            self.assertEqual(self.records[("Poly.py",symbol)]["backend_blocker"],"real_bootstrap")
        # Shape propagation for Max does not itself execute a ciphertext Max.
        r=self.records[("MPCB.py","CascadeMax")]
        self.assertEqual(r["role"],"public_packing_shape_propagation")
        self.assertIsNone(r["backend_blocker"])

    def test_stub_is_not_a_functional_helper(self):
        r=self.records[("MPCB.py","shapeClosure.AvgMidSelecting")]
        node=module.nodes(module.ROOT/r["source"])[r["symbol"]]
        self.assertTrue(all(isinstance(n,ast.Pass) for n in node.body))
        self.assertEqual(r["backend_blocker"],"upstream_stub")

    def test_defined_inplace_is_not_installed_operator(self):
        r=self.records[("expr.py","hecateMetaBinary.__new__.binaryFactory.binaryInplaceMethod")]
        tree=module.nodes(module.ROOT/r["source"])["hecateMetaBinary.__new__.binaryFactory"]
        installations=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="setattr"]
        registered={n.args[-1].id for n in installations if isinstance(n.args[-1],ast.Name)}
        self.assertNotIn("binaryInplaceMethod",registered)
        self.assertIn("binaryMethod",registered)
        self.assertEqual(r["role"],"defined_but_unregistered_inplace")

    def test_plaintext_math_and_framework_io_not_exposed(self):
        for name in ("ReLU","rms","nprelu"):
            r=self.records[("Poly.py",name)]
            self.assertEqual(r["role"],"plaintext_diagnostic_utility")
            self.assertEqual(r["candidate_surface"],"no_direct_candidate_access")
        for name in ("save","removeCtxt","getProperFrame"):
            self.assertEqual(self.records[("expr.py",name)]["candidate_surface"],"no_direct_candidate_access")

    def test_deterministic_hash(self):
        self.assertEqual(module.build(self.ledger),self.data)
        body={k:v for k,v in self.data.items() if k!="classification_sha256"}
        self.assertEqual(module.digest(body),self.data["classification_sha256"])

if __name__=="__main__":unittest.main()
