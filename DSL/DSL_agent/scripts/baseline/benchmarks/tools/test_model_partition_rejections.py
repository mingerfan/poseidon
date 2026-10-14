"""Gate-level negative examples; never score these as FHE or positive models."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("model_rejects",HERE/"build_model_partition_rejections.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,_,cls.ledger=module.read_suite(module.BASE/"benchmarks/semantic-v1-chunk-helpers-r29")
        cls.tasks=module.build(cls.rows,cls.ledger)
        cls.by={t["requirement"]:t for t in cls.tasks}
        cls.models={r["model"]["id"]:r["model"] for r in cls.rows}

    def test_all_partitions_and_positive_controls(self):
        expected={r["id"] for r in self.ledger["requirements"] if r["layer"]=="model"}
        self.assertEqual(set(self.by),expected)
        for t in self.tasks:
            module.validate(self.models[t["positive_model_id"]])
            self.assertEqual(module.digest(self.models[t["positive_model_id"]]),t["positive_model_sha256"])

    def test_each_counterexample_is_rejected_for_recorded_reason(self):
        for t in self.tasks:
            with self.subTest(feature=t["requirement"]):
                try:module.validate(t["negative_model"])
                except ValueError as e:self.assertEqual(str(e),t["observed_rejection"])
                else:self.fail("Negative fixture passed")
                self.assertEqual(module.digest(t["negative_model"]),t["negative_model_sha256"])

    def test_operand_order_rejects_at_encryption_gate(self):
        for f,t in self.by.items():
            if f.startswith("overload."):
                self.assertEqual(t["observed_rejection"],"First operand must be encrypted")
        self.assertEqual(self.by["op.linear"]["observed_rejection"],"Weights/stats must be public")

    def test_representative_specific_attribute_gates(self):
        examples={"op.reshape":"Reshape dimensions","op.rotate":"Rotation step",
                  "op.split":"Split sections","op.sum":"Duplicate reduction axes",
                  "op.power":"Power exponent","op.conv1d":"Conv groups"}
        for f,reason in examples.items():self.assertEqual(self.by[f]["observed_rejection"],reason)

    def test_unrelated_partition_is_not_silently_substituted(self):
        model=self.rows[0]["model"]
        with self.assertRaisesRegex(ValueError,"Positive control lacks feature"):
            module.counterexample(model,"op.unreviewed")

    def test_original_models_remain_bound(self):
        for r in self.rows:self.assertEqual(module.digest(r["model"]),r["model_sha256"])
        self.assertTrue(all(not t["counts_as_positive_model"] and not t["encrypted_execution"] for t in self.tasks))

    def test_deterministic_and_json_safe(self):
        self.assertEqual(module.build(self.rows,self.ledger),self.tasks)
        self.assertEqual(json.loads(json.dumps(self.tasks,allow_nan=False)),self.tasks)
        for t in self.tasks:
            self.assertEqual(module.digest({k:v for k,v in t.items() if k!="task_sha256"}),t["task_sha256"])

if __name__=="__main__":unittest.main()
