"""Pure offline invariants for the proposed mathematical-context bundle."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("model_context_builder", HERE/"build_model_semantic_contexts.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
SUITE = builder.BASE/"benchmarks/semantic-v1-chunk-helpers-r29"


class ModelContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.index, cls.ledger = builder.read_suite(SUITE)
        cls.binding = dict(test_only=True)
        cls.bundle = builder.build(cls.rows, cls.ledger, cls.binding)

    def test_every_partition_has_three_distinct_contexts(self):
        grouped = {}
        for t in self.bundle["tasks"]:
            grouped.setdefault(t["requirement"], []).append(t)
        expected = {r["id"] for r in self.ledger["requirements"] if r["layer"] == "model"}
        self.assertEqual(set(grouped), expected)
        for name, tasks in grouped.items():
            with self.subTest(requirement=name):
                self.assertEqual(len(tasks), 3)
                self.assertEqual(len({t["topology"] for t in tasks}), 3)

    def test_actual_graphs_bind_each_target(self):
        models = {r["model"]["id"]:r for r in self.rows+self.bundle["supplemental_models"]}
        for task in self.bundle["tasks"]:
            with self.subTest(task=task["id"]):
                row = models[task["model_id"]]
                builder.validate(row["model"])
                self.assertIn(task["requirement"], builder.features(row["model"]))
                self.assertEqual(builder.digest(row["model"]), task["model_sha256"])
                self.assertEqual(builder.signature(row["model"], True), task["topology"])
                body = {k:v for k,v in task.items() if k != "task_sha256"}
                self.assertEqual(builder.digest(body), task["task_sha256"])

    def test_no_rename_or_weight_only_diversity(self):
        all_rows = self.rows+self.bundle["supplemental_models"]
        self.assertEqual(len({r["signature"] for r in all_rows}), len(all_rows))
        for r in self.bundle["supplemental_models"]:
            renamed = copy.deepcopy(r["model"])
            renamed["id"] = "other_name"
            self.assertEqual(builder.signature(renamed), r["signature"])

    def test_group_and_parent_split_isolation(self):
        splits = {}
        parents = {r["model"]["id"]:r for r in self.rows}
        for r in self.rows+self.bundle["supplemental_models"]:
            self.assertEqual(splits.setdefault(r["topology"],r["split"]),r["split"])
        for r in self.bundle["supplemental_models"]:
            p = parents[r["provenance"]["parent_id"]]
            self.assertEqual(r["split"],p["split"])
            self.assertEqual(r["provenance"]["parent_sha256"],p["model_sha256"])

    def test_original_models_are_not_mutated_or_recounted(self):
        self.assertEqual(len(self.rows),1200)
        for r in self.rows:
            self.assertEqual(builder.digest(r["model"]),r["model_sha256"])
        self.assertFalse(self.bundle["supplemental_models_counted_in_original"])
        self.assertFalse(self.bundle["selection_uses_agent_or_FHE_success"])
        self.assertFalse(self.bundle["all_executed"])
        self.assertTrue(all(t["state"]=="planned_not_run" for t in self.bundle["tasks"]))

    def test_deterministic_bundle(self):
        self.assertEqual(builder.build(self.rows,self.ledger,self.binding),self.bundle)

    def test_false_feature_mapping_is_rejected(self):
        ledger = copy.deepcopy(self.ledger)
        req = next(r for r in ledger["requirements"] if r["layer"]=="model")
        req["id"] = "op.nonexistent"
        with self.assertRaisesRegex(ValueError,"Incorrect existing feature mapping"):
            builder.build(self.rows,ledger,self.binding)

    def test_missing_fixture_remains_failure(self):
        ledger = copy.deepcopy(self.ledger)
        req = next(r for r in ledger["requirements"] if r["layer"]=="model")
        req["models"] = []
        with self.assertRaisesRegex(ValueError,"Missing positive model"):
            builder.build(self.rows,ledger,self.binding)

    def test_existing_split_leakage_rejected(self):
        rows = copy.deepcopy(self.rows)
        source = rows[0]
        duplicate = copy.deepcopy(source)
        duplicate["split"] = "holdout" if source["split"]!="holdout" else "development"
        with self.assertRaisesRegex(ValueError,"Existing split leakage"):
            builder.build(rows+[duplicate],self.ledger,self.binding)

    def test_recipes_preserve_interface_and_do_not_mutate_parent(self):
        model = next(r["model"] for r in self.rows if len(r["model"]["nodes"])<8)
        before = builder.digest(model)
        old = builder.validate(model)
        for recipe in builder.RECIPES:
            g = builder.augmented(model,recipe,"recipe_test")
            info = builder.validate(g)
            self.assertEqual(g["inputs"],model["inputs"])
            self.assertEqual([o["name"] for o in g["outputs"]],[o["name"] for o in model["outputs"]])
            self.assertEqual([info["shapes"][o["value"]] for o in g["outputs"]],
                             [old["shapes"][o["value"]] for o in model["outputs"]])
            self.assertEqual(builder.digest(model),before)
        with self.assertRaisesRegex(ValueError,"Unknown context recipe"):
            builder.augmented(model,"unrestricted","bad")


if __name__ == "__main__":
    unittest.main()
