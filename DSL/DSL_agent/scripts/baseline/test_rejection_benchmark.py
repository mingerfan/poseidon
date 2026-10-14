"""Negative evidence must fail if a gate accepts, crashes, or rejects its control."""
import copy,unittest
from unittest.mock import patch
from rejection_benchmark import PARTITIONS,tasks,run_task,fixture,check
from benchmark_graph import digest,signature

class RejectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalog=tasks()
    def test_frozen_ledger_binds_every_partition(self):
        from benchmark_runner import load,DEFAULT,strict_file
        load(DEFAULT);ledger=strict_file(DEFAULT/"coverage.json",4*1024**2)
        self.assertEqual(ledger["rejection_tasks"],self.catalog)
        self.assertEqual(len(self.catalog),48)
        self.assertEqual(len({t["task_sha256"] for t in self.catalog}),48)
        for name in PARTITIONS:
            contexts=[t for t in self.catalog if t["requirement"]=="reject."+name]
            self.assertEqual(len(contexts),3)
            from rejection_benchmark import model
            self.assertEqual(len({signature(model(t["context"]),True) for t in contexts}),3)
    def test_all_static_gates_and_positive_controls(self):
        for task in self.catalog:
            with self.subTest(task=task["id"]):
                r=run_task(task,self.catalog)
                self.assertEqual(r["status"],"passed")
                self.assertTrue(r["positive_control_accepted"])
                self.assertTrue(r["negative_rejected"])
    def test_accepting_negative_is_not_pass(self):
        with patch("rejection_benchmark.check",return_value={}):
            with self.assertRaisesRegex(ValueError,"unexpectedly accepted"):
                run_task(self.catalog[0],self.catalog)
    def test_wrong_failure_layer_is_not_pass(self):
        with patch("rejection_benchmark.check",side_effect=[{},ValueError("unrelated")]):
            with self.assertRaisesRegex(ValueError,"Unexpected rejection reason"):
                run_task(self.catalog[0],self.catalog)
    def test_crash_is_not_expected_rejection(self):
        with patch("rejection_benchmark.check",side_effect=[{},KeyError("broken")]):
            with self.assertRaises(KeyError):run_task(self.catalog[0],self.catalog)
    def test_rejected_positive_control_is_not_pass(self):
        with patch("rejection_benchmark.check",side_effect=ValueError("broken control")):
            with self.assertRaisesRegex(ValueError,"broken control"):
                run_task(self.catalog[0],self.catalog)
    def test_changed_task_even_rehashed_is_rejected(self):
        t=copy.deepcopy(self.catalog[0]);t["expected_reason"]="anything"
        t["task_sha256"]=digest({k:v for k,v in t.items() if k!="task_sha256"})
        with self.assertRaisesRegex(ValueError,"Frozen rejection task"):
            run_task(t,self.catalog)

if __name__=="__main__":unittest.main()
