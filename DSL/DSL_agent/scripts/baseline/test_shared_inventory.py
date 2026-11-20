"""Retained-store accounting with real files; no Provider or backend calls."""
from pathlib import Path
import tempfile
import unittest

from shared_execution import Coordinator, KINDS, LIMITS, digest
from application_artifact_budget import effective_limits, FORMAT


class StoreAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shared-store-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir(mode=0o700)
        self.store = self.root / "tasks"
        self.store.mkdir(mode=0o700)
        self.limits = dict(LIMITS, evidence_bytes=8192, batch_bytes=2048)
        self.c = Coordinator(self.root / "coordination", [self.results], limits=self.limits)

    def batch(self, name="one", reserve=1024):
        out = self.results / name
        out.mkdir(mode=0o700)
        plan = dict(branch="lhy-generalization", source_binding="a"*64, reserve_bytes=reserve,
                    wall_seconds=60, stages={kind: int(kind == "api") for kind in KINDS})
        return self.c.register(name, plan, out)[0]

    def test_existing_store_is_counted_without_changing_its_bytes(self):
        old = self.store / "retained-response"
        old.write_bytes(b"historical"*31)
        original = old.read_bytes(), old.stat().st_mtime_ns
        self.c.register_inventory(self.store)
        self.assertEqual(self.c.snapshot()[1]["observed_bytes"], len(original[0]))
        self.assertEqual((old.read_bytes(), old.stat().st_mtime_ns), original)

    def test_nested_and_common_result_inventories_do_not_double_count(self):
        child = self.store / "nested"
        child.mkdir(mode=0o700)
        (child / "old").write_bytes(b"x"*123)
        (self.results / "retained").write_bytes(b"y"*77)
        self.c.register_inventory(self.results)
        self.c.register_inventory(self.store)
        self.c.register_inventory(child)
        self.assertEqual(self.c.snapshot()[1]["observed_bytes"], 200)

    def test_overfull_old_store_rejects_admission_without_rewriting_ledger(self):
        (self.store / "large").write_bytes(b"x"*8193)
        before = self.c.journal.read_bytes()
        with self.assertRaisesRegex(ValueError, "Global evidence"):
            self.c.register_inventory(self.store)
        self.assertEqual(self.c.journal.read_bytes(), before)

    def test_growth_uses_existing_baseline_and_reservation_only_once(self):
        (self.store / "retained").write_bytes(b"x"*500)
        self.c.register_inventory(self.store)
        batch = self.batch()
        batch.watch_growth(self.store)
        (self.store / "new").write_bytes(b"y"*200)
        usage = batch.checkpoint()
        self.assertEqual(usage["observed_bytes"], 700)
        self.assertEqual(usage["remaining_reserved_bytes"], 824)
        self.assertEqual(batch.finish("succeeded")["observed_bytes"], 200)
        self.assertEqual(self.c.snapshot()[1]["observed_bytes"], 700)

    def test_growth_overrun_never_enters_an_api_stage(self):
        self.c.register_inventory(self.store)
        batch = self.batch(reserve=100)
        batch.watch_growth(self.store)
        (self.store / "new").write_bytes(b"x"*101)
        with self.assertRaisesRegex(ValueError, "Batch artifact"):
            with batch.stage("api"):
                self.fail("Over-budget work entered")
        self.assertEqual(batch.record()["used"]["api"], 0)
        self.assertEqual(batch.finish("failed")["observed_bytes"], 101)

    def test_active_task_growth_cannot_be_adopted_by_a_second_batch(self):
        self.c.register_inventory(self.store)
        first, second = self.batch(), self.batch("two")
        first.watch_growth(self.store)
        with self.assertRaisesRegex(ValueError, "already active"):
            second.watch_growth(self.store)
        first.finish("succeeded")
        second.watch_growth(self.store)
        second.finish("succeeded")

    def test_fresh_native_child_of_watched_task_is_not_counted_twice(self):
        task = self.results / "task"
        task.mkdir(mode=0o700)
        batch = self.batch()
        batch.watch_growth(task)
        output = task / "output"
        output.mkdir(mode=0o700)
        batch.register_artifact(output)
        (output / "result").write_bytes(b"x"*21)
        self.assertEqual(batch.checkpoint()["observed_bytes"], 21)
        self.assertEqual(batch.finish("succeeded")["observed_bytes"], 21)

    def test_store_identity_replacement_is_rejected(self):
        self.c.register_inventory(self.store)
        self.store.rename(self.root / "preserved")
        self.store.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError, "inventory identity"):
            self.c.snapshot()

    def test_unregistered_growth_and_broad_store_roots_are_rejected(self):
        batch = self.batch()
        with self.assertRaisesRegex(ValueError, "Registered task"):
            batch.watch_growth(self.store)
        with self.assertRaisesRegex(ValueError, "overlap/scope"):
            self.c.register_inventory(self.root)

    def test_old_ledger_can_be_read_without_implicit_migration(self):
        before = self.c.journal.read_bytes()
        other = Coordinator(self.c.storage, [self.results], limits=self.limits)
        self.assertEqual(other.snapshot()[1]["observed_bytes"], 0)
        self.assertEqual(self.c.journal.read_bytes(), before)
        batch = self.batch()
        self.c.register_inventory(self.store)
        batch.watch_growth(self.store)
        data, _ = other.snapshot()
        self.assertIn("store_accounting", data)
        self.assertEqual(data["batches"]["one"]["plan_binding"], batch.binding)

    def test_store_extension_cannot_change_the_fixed_resource_policy(self):
        self.c.register_inventory(self.store)
        batch = self.batch()
        batch.watch_growth(self.store)
        data = self.c._read()
        data["store_accounting"]["format"] = "unreviewed"
        self.c._write(self.c.journal, data)
        with self.assertRaisesRegex(ValueError, "Store accounting"):
            self.c.snapshot()

    def test_deployment_caps_intersect_without_expanding_request_allowance(self):
        fixed = dict(format=FORMAT, max_new_artifact_bytes=2048, min_free_disk_bytes=4096)
        asked = dict(format=FORMAT, max_new_artifact_bytes=1024, min_free_disk_bytes=8192)
        self.assertEqual(effective_limits(asked, fixed), asked)
        asked.update(max_new_artifact_bytes=8192, min_free_disk_bytes=0)
        self.assertEqual(effective_limits(asked, fixed), fixed)
        self.assertIsNone(effective_limits(None))
        self.assertEqual(effective_limits(None, fixed), fixed)


if __name__ == "__main__":
    unittest.main()
