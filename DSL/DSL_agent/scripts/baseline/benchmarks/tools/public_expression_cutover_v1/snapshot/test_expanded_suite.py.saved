"""16-family benchmark identity, real model semantics and encrypted evidence."""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from expanded_model_suite import BASE_FAMILIES, NEW_FAMILIES, FAMILIES, entries, manifest, digest
from model_graph import validate_graph


class ExpandedSuiteTests(unittest.TestCase):
    def test_96_cases_16_explicit_families_without_descriptor_labels(self):
        items = entries()
        self.assertEqual(len(items), 96)
        self.assertEqual(len(FAMILIES), 16)
        self.assertEqual(Counter(e["family"] for e in items), {f: 6 for f in FAMILIES})
        self.assertEqual(len({e["descriptor"]["id"] for e in items}), 96)
        for item in items[48:]:
            validate_graph(item["descriptor"])
            self.assertNotIn("family", item["descriptor"])
            self.assertNotIn("configuration", item["descriptor"])
            self.assertEqual(digest(item["descriptor"]), item["descriptor_sha256"])

    def test_manifest_is_reproducible_and_mutation_does_not_change_future_entries(self):
        original = manifest()
        self.assertEqual(original, manifest())
        changed = entries()
        changed[-1]["descriptor"]["constants"].clear()
        self.assertEqual(original, manifest())
        self.assertFalse(original["live_agent_validated"])
        self.assertFalse(original["poseidon_gpu_validated"])

    def test_families_have_structural_or_privacy_distinctions(self):
        items = entries()[48:]
        for family in NEW_FAMILIES:
            group = [i["descriptor"] for i in items if i["family"] == family]
            self.assertEqual(len(group), 6)
            signatures = []
            for desc in group:
                shape = copy.deepcopy(desc)
                shape.pop("id")
                signatures.append(digest(shape))
            self.assertEqual(len(set(signatures)), 6, family)
            if family.startswith("dual_"):
                self.assertTrue(all(len(d["inputs"]) == 2 for d in group))
        self.assertEqual(len(set(BASE_FAMILIES) | set(NEW_FAMILIES)), 16)


try:
    import numpy as np
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires pinned Torch")
class ExpandedTorchTests(unittest.TestCase):
    def test_exact_original_48_preserved_and_all_new_models_trace(self):
        from model_catalog import descriptors, FAMILIES as original, build_model, test_inputs
        from fx_to_hecate import translate
        from model_graph import evaluate_reference
        from multi_input_fixtures import fixture_inputs
        self.assertEqual(BASE_FAMILIES, original)
        self.assertEqual([e["descriptor"] for e in entries()[:48]], descriptors())
        for item in entries()[48:]:
            data = item["descriptor"]
            model, shape = build_model(data)
            translation = translate(model, shape)
            self.assertTrue(translation["static_check"]["syntax_type_layout_checked"])
            if data["schema"] == 2:
                batches = [(x,) for x in test_inputs(shape)]
            else:
                batches = [tuple(np.asarray(v).reshape(s["shape"]) for v, s in zip(batch, data["inputs"]))
                           for batch in fixture_inputs(len(data["inputs"]))]
            for batch in batches:
                supplied = (batch[0].tolist() if data["schema"] == 2 else
                            {s["name"]: v.tolist() for s, v in zip(data["inputs"], batch)})
                expected = evaluate_reference(data, supplied)
                with torch.no_grad():
                    actual = model(*[torch.from_numpy(v.copy()) for v in batch]).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, err_msg=item["family"])


@unittest.skipUnless(os.environ.get("POSEIDON_EXPANDED_RULE_RESULTS"), "Requires actual 96-case encrypted run")
class ExpandedEvidenceTests(unittest.TestCase):
    def test_exact_96_descriptors_and_families_have_real_execution(self):
        root = Path(os.environ["POSEIDON_EXPANDED_RULE_RESULTS"])
        report = json.loads((root / "report.json").read_text())
        items = entries()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_descriptors"], [i["descriptor"] for i in items])
        self.assertEqual(len(report["cases"]), 96)
        self.assertEqual(report["agent_calls"], 0)
        self.assertFalse(report["poseidon_gpu_validated"])
        self.assertEqual(Counter(i["benchmark_family"] for i in report["cases"]), {f: 6 for f in FAMILIES})
        self.assertEqual(report["parameters"]["security_check"], "tc128")
        self.assertEqual(report["parameters"]["modulus_bits"], [60]*14)
        for row, expected in zip(report["cases"], items):
            self.assertEqual(row["benchmark_family"], expected["family"])
            self.assertEqual(row["status"], "passed")
            self.assertTrue(row["execution"]["encrypted_execution"])
            self.assertFalse(row["execution"]["bootstrap_executed"])
            self.assertEqual(row["execution"]["input_batches"], 4)
            self.assertTrue(row["comparison"]["passed"])
            self.assertEqual(row["comparison"]["atol"], 1e-5)
            self.assertEqual(row["comparison"]["rtol"], 1e-4)
            for name, hashed in row["frozen_hashes"].items():
                self.assertEqual(hashlib.sha256((root / row["folder"] / name).read_bytes()).hexdigest(), hashed)


if __name__ == "__main__":
    unittest.main()
