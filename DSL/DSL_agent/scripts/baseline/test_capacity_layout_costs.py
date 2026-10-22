import copy
import math
import unittest

from capacity_layout_costs import (
    candidate_costs, choose_candidate, template_costs, tiled_costs,
)
from unified_capacity_tiling import template, row_stages, template_cost


class CapacityLayoutCostsTest(unittest.TestCase):
    def test_diagonal_depth_beats_cheaper_row_sum(self):
        direct = candidate_costs("direct_diagonal", 1, 900, 100)
        rows = candidate_costs("row_sum", 2, 40, 3)
        self.assertEqual(choose_candidate([rows, direct])["name"], "direct_diagonal")

    def test_equal_depth_uses_work_and_stable_name(self):
        a = candidate_costs("a", 1, 20, 3)
        b = candidate_costs("b", 1, 19, 3)
        self.assertEqual(choose_candidate([a, b])["name"], "b")
        b["estimated_work"] = 20
        self.assertEqual(choose_candidate([b, a])["name"], "a")

    def test_selection_is_inclusive_and_has_constant_budget(self):
        candidate = candidate_costs("boundary", 1, 1023, 256)
        self.assertEqual(choose_candidate([candidate])["name"], "boundary")
        self.assertIsNone(choose_candidate([candidate], max_work=1022))
        self.assertIsNone(choose_candidate([candidate], max_constants=255))
        for field, value in (("estimated_work", 1024), ("constant_count", 257)):
            other = dict(candidate, **{field: value})
            self.assertIsNone(choose_candidate([other]))

    def test_unknown_metrics_are_not_selectable(self):
        for field in ("plaintext_depth_upper_bound", "estimated_work", "constant_count"):
            values = dict(name="unknown", plaintext_depth_upper_bound=1,
                          estimated_work=2, constant_count=1)
            values[field] = None
            self.assertIsNone(choose_candidate([candidate_costs(**values)]))

    def test_cost_does_not_certify_backend_or_change_input(self):
        candidate = candidate_costs("direct_diagonal", 1, 3, 1)
        selected = choose_candidate([candidate])
        self.assertEqual(selected["compiler_feasibility"], "unknown")
        self.assertEqual(selected["execution_feasibility"], "unknown")
        selected["certainty"]["work"] = "changed"
        self.assertEqual(candidate["certainty"]["work"], "estimate")
        corrupt = dict(candidate, compiler_feasibility="passed")
        with self.assertRaises(ValueError):
            choose_candidate([corrupt])

    def test_template_matches_public_row_stages(self):
        weights = [[0.5, -0.25, 0.125], [0.25, 0.5, -0.125]]
        t = template(weights, 4, 16)
        report = template_costs(t, 16)
        expected = len(row_stages(3, 4, 4, 16)) + 1 + len(row_stages(2, 4, 4, 16, compress=True))
        self.assertEqual(report["plaintext_depth_upper_bound"], expected)
        self.assertEqual(report["estimated_work"], template_cost(t, 16))
        self.assertEqual(report["depth_stages"], dict(expand=2, weight=1, row_mask=0, compress=2))

    def test_row_sum_adds_one_serial_mask(self):
        weights = [[0.5, -0.25, 0.125], [0.25, 0.5, -0.125]]
        direct = template_costs(template(weights, 4, 16), 16)
        rows = template_costs(template(weights, 4, 16, row_sum=True), 16)
        self.assertEqual(rows["plaintext_depth_upper_bound"],
                         direct["plaintext_depth_upper_bound"] + 1)

    def test_identity_and_zero_elision_keep_conservative_depth(self):
        for weights in ([[1.]], [[0.]]):
            report = template_costs(template(weights, 1, 4), 4)
            self.assertEqual(report["plaintext_depth_upper_bound"], 1)
            self.assertEqual(report["zero_one_elision"],
                             "not_discounted_in_depth_upper_bound")
        report = template_costs(template([[1.]], 1, 4, row_sum=True), 4)
        self.assertEqual(report["plaintext_depth_upper_bound"], 2)

    def tiled(self, *, cross=False, blocks=1, row_sum=False):
        t = template([[0.5, -0.25, 0.125], [0.25, 0.5, -0.125]], 2, 16,
                     row_sum=row_sum)
        single = [dict(chunk=0, lo=0, hi=6, step=0)]
        multiple = [dict(chunk=0, lo=0, hi=2, step=1),
                    dict(chunk=1, lo=2, hi=6, step=15)]
        spans = multiple if cross else single
        plan = dict(blocks=[dict(batch_offset=i * 2, batches=2,
                                gather=copy.deepcopy(spans),
                                scatter=copy.deepcopy(spans)) for i in range(blocks)])
        return plan, {2: t}

    def test_cross_chunk_masks_add_two_not_number_of_spans(self):
        plain, templates = self.tiled()
        cross, _ = self.tiled(cross=True)
        a = tiled_costs(plain, templates, 16)
        b = tiled_costs(cross, templates, 16)
        self.assertEqual(b["plaintext_depth_upper_bound"], a["plaintext_depth_upper_bound"] + 2)
        self.assertEqual(b["block_depths"][0]["depth_stages"]["gather"], 1)
        self.assertEqual(b["block_depths"][0]["depth_stages"]["scatter"], 1)
        self.assertGreater(b["estimated_work"], a["estimated_work"])

    def test_independent_blocks_take_max_depth_not_sum(self):
        one, templates = self.tiled()
        many, _ = self.tiled(blocks=3)
        a = tiled_costs(one, templates, 16)
        b = tiled_costs(many, templates, 16)
        self.assertEqual(a["plaintext_depth_upper_bound"], b["plaintext_depth_upper_bound"])
        self.assertGreater(b["estimated_work"], a["estimated_work"])

    def test_existing_registry_deduplicates_contents(self):
        plan, templates = self.tiled()
        t = templates[2]
        base = tiled_costs(plan, templates, 16)
        same = tiled_costs(plan, templates, 16, existing_vectors=[t["expand"][0]["fixed"]])
        self.assertEqual(base["constant_count"], same["constant_count"])
        added = tiled_costs(plan, templates, 16, existing_vectors=[[0.123] * 16])
        self.assertEqual(added["constant_count"], base["constant_count"] + 1)
        self.assertEqual(added["estimated_work"], base["estimated_work"] + 2)

    def test_tiled_plan_does_not_mutate_and_does_not_trust_supplied_work(self):
        plan, templates = self.tiled()
        plan["estimated_work"] = -999
        frozen = copy.deepcopy((plan, templates))
        result = tiled_costs(plan, templates, 16)
        self.assertGreaterEqual(result["estimated_work"], 0)
        self.assertEqual((plan, templates), frozen)

    def test_partial_tail_uses_its_own_template(self):
        plan, templates = self.tiled(blocks=2)
        plan["blocks"][-1]["batches"] = 1
        templates[1] = template([[0.5, -0.25, 0.125], [0.25, 0.5, -0.125]], 1, 16)
        report = tiled_costs(plan, templates, 16)
        self.assertEqual(len(report["block_depths"]), 2)
        self.assertGreater(report["block_depths"][0]["plaintext_depth_upper_bound"],
                           report["block_depths"][1]["plaintext_depth_upper_bound"])

    def test_existing_plan_work_matches_for_rectangular_and_tail_blocks(self):
        from unified_capacity_tiling import plan as existing_plan
        for width, outputs, batches, period in ((3, 2, 12, 16), (5, 2, 20, 64),
                                               (2, 3, 16, 32), (7, 3, 31, 64)):
            weights = [[0.125 * (1 + (i + j) % 3) for j in range(width)]
                       for i in range(outputs)]
            for row_sum in (False, True):
                with self.subTest(width=width, outputs=outputs, row_sum=row_sum):
                    selected = existing_plan(weights, batches, period, 1024, row_sum=row_sum)
                    self.assertIsNotNone(selected)
                    record = tiled_costs(*selected, period)
                    self.assertEqual(record["estimated_work"], selected[0]["estimated_work"])
                    self.assertEqual(record["compiler_feasibility"], "unknown")

    def test_malformed_public_plan_is_rejected(self):
        plan, templates = self.tiled()
        for change in ("period", "vector", "span", "unused_template", "nonfinite"):
            p, ts = copy.deepcopy(plan), copy.deepcopy(templates)
            period = 16
            if change == "period": period = 15
            if change == "vector": ts[2]["expand"][0]["fixed"].pop()
            if change == "span": p["blocks"][0]["gather"][0]["hi"] = 17
            if change == "unused_template": ts[1] = template([[1.]], 1, 16)
            if change == "nonfinite": next(iter(ts[2]["weights"].values()))[0] = math.inf
            with self.subTest(change=change), self.assertRaises(ValueError):
                tiled_costs(p, ts, period)

    def test_bad_budgets_duplicate_names_and_boolean_counts_rejected(self):
        candidate = candidate_costs("direct", 1, 2, 1)
        for kwargs in (dict(max_work=-1), dict(max_constants=257), dict(max_work=True)):
            with self.assertRaises(ValueError):
                choose_candidate([candidate], **kwargs)
        with self.assertRaises(ValueError):
            choose_candidate([candidate, copy.deepcopy(candidate)])
        with self.assertRaises(ValueError):
            candidate_costs("bad", True, 1, 1)


if __name__ == "__main__":
    unittest.main()
