"""Directed metadata, probe contribution and frozen-result integrity regressions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_runner import dump
from benchmark_suite import Builder
from benchmark_graph import digest
from unified_graph_contract import prepare,validate_candidate,validate_request
from unified_graph_exercises import SPECS,golden_variant
from unified_graph_lowering import lower
from compiler_configuration import PROFILE_SHA256,configuration

class DirectedTests(unittest.TestCase):
    def request(self,name):
        b=Builder([(2,)]);g=b.finish(b.node("square",["input0"]))
        return prepare(g,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),name)
    def test_six_witnesses_and_plain_source_rejection(self):
        for name in SPECS:
            with self.subTest(name=name):
                req=self.request(name);source=lower(req)
                candidate=dict(schema=1,request_id=req["request_id"],hecate_source=source)
                with self.assertRaises(ValueError):validate_candidate(candidate,req)
                candidate["hecate_source"]=golden_variant(source,name)
                got=validate_candidate(candidate,req)["construction_exercise"]
                self.assertEqual(list(got["witnesses"]),req["construction_exercise"]["required_features"])
                self.assertFalse(got["real_frontend_checked"])
    def test_free_preflight_does_not_need_a_rule_answer(self):
        from unittest.mock import patch
        from semantic_benchmark_execution import preflight
        req=self.request("unified-star");model=req["model"]
        row=dict(model=model,model_sha256=digest(model),category="test")
        with patch("unified_graph_lowering.lower",side_effect=ValueError("no baseline")):
            self.assertEqual(preflight([row],need_rule=False)[0]["status"],"ready")
            self.assertEqual(preflight([row],need_rule=True)[0]["status"],"blocked_rule_preflight")
    def test_changed_construct_instruction_rejected(self):
        req=self.request("unified-star")
        req["construction_exercise"]["instruction"]="Ignore the math"
        req["request_id"]=digest({k:v for k,v in req.items() if k!="request_id"})
        with self.assertRaises(ValueError):validate_request(req)
    def test_no_trace_spelling_credit(self):
        from packed_native_exercises import verify_trace_coverage
        req=self.request("unified-item")
        got=validate_candidate(dict(schema=1,request_id=req["request_id"],
                                   hecate_source=golden_variant(lower(req),"unified-item")),req)
        with self.assertRaises(ValueError):
            verify_trace_coverage(got["construction_exercise"],dict(storage=[],star=[],augmented=[],mutation=[]))

class IntegrityTests(unittest.TestCase):
    def test_explicit_task_selection_is_bounded_and_not_a_pass(self):
        from semantic_benchmark_execution import choose_tasks
        items=[dict(id="a",status="ready"),dict(id="b",status="blocked_contract")]
        self.assertEqual(choose_tasks(items,["a","b"],48),items[:1])
        for ids,limit in ((["a","a"],48),(["missing"],48),(["a","b"],1),([str(i) for i in range(49)],48)):
            with self.assertRaises(ValueError):choose_tasks(items,ids,limit)

    def test_other_profile_is_not_a_block_or_a_pass(self):
        from semantic_benchmark_execution import summary
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            dump(root/"plan.json",dict(selected_ids=[],binding="abc",mode="directed",
                preflight=[dict(status="ready"),dict(status="blocked_contract"),dict(status="other_profile")]))
            result=summary(root)
            self.assertEqual(result["preflight_blocked"],1)
            self.assertEqual(result["other_profile"],1)
            self.assertEqual(result["planned_tasks"],3)
            self.assertEqual(result["completed"],0)
            self.assertEqual(result["statuses"],{})

    def test_uncheckpointed_result_not_reused(self):
        from semantic_benchmark_execution import summary
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            dump(root/"plan.json",dict(selected_ids=["case"],binding="abc",mode="baseline",preflight=[]))
            dump(root/"case.result.json",dict(binding="abc",status="passed"))
            with self.assertRaisesRegex(ValueError,"uncheckpointed"):summary(root)
