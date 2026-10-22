import unittest
import contextlib
import io
import json
from unittest.mock import patch
from run_agent_batch import case_metrics, summarize, batch_plan, main, select_failed


def successful():
    return dict(status="passed", provider="deepseek_api", llm_generation_validated=True,
                agent_calls=1, attempts=[dict(parsed=True, source_parsed=True, checked=True,
                compiled=True, executed=True, numerically_correct=True, status="passed",
                execution=dict(encrypted_execution=True, input_batches=4),
                comparison=dict(compared_values=8, max_absolute_error=1e-8))])


class AgentBatchTests(unittest.TestCase):
    def test_public_control_plan_and_resume_identity(self):
        from types import SimpleNamespace
        from run_agent_batch import construction_options,validate_construction_continuation
        args = SimpleNamespace(public_control=True,public_numbers=False)
        self.assertEqual(construction_options(args),['--public-control'])
        validate_construction_continuation({'public_control':True},args)
        with self.assertRaisesRegex(ValueError,'fresh batch'):
            validate_construction_continuation({'public_numbers':True},args)
        output = io.StringIO()
        with patch('sys.argv',['run_agent_batch.py','--plan','--public-control']), \
             patch('agent_credentials.load_api_key',side_effect=AssertionError('credential access')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(),0)
        data = json.loads(output.getvalue())
        self.assertTrue(data['public_control'])
        self.assertFalse(data['public_numbers'])
        self.assertEqual(data['agent_calls'],0)

    def test_public_numeric_plan_and_retry_contract(self):
        from types import SimpleNamespace
        from run_agent_batch import construction_options,validate_construction_continuation
        args=SimpleNamespace(public_numbers=True)
        self.assertEqual(construction_options(args),['--public-numbers'])
        self.assertEqual(construction_options(SimpleNamespace()),[])
        validate_construction_continuation({'public_numbers':True},args)
        with self.assertRaisesRegex(ValueError,'fresh batch'):
            validate_construction_continuation({},args)
        output=io.StringIO()
        with patch('sys.argv',['run_agent_batch.py','--plan','--public-numbers']), \
             patch('agent_credentials.load_api_key',side_effect=AssertionError('credential access')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(),0)
        data=json.loads(output.getvalue())
        self.assertTrue(data['public_numbers'])
        self.assertEqual(data['api_concurrency'],10)
        self.assertEqual(data['agent_calls'],0)

    def test_expanded_plan_keeps_96_cases_and_independent_family_labels(self):
        rows, benchmark = batch_plan(True)
        data = summarize(rows)
        self.assertEqual(data["planned"], 96)
        self.assertEqual(len(data["families"]), 16)
        self.assertEqual(data["api_calls"], 0)
        self.assertTrue(all(f["planned"] == 6 for f in data["families"].values()))
        self.assertTrue(all("family" not in r["descriptor"] for r in rows[48:]))
        legacy, metadata = batch_plan()
        self.assertIsNone(metadata)
        self.assertEqual([r["descriptor"] for r in legacy], [r["descriptor"] for r in rows[:48]])
        self.assertFalse(benchmark["live_agent_validated"])

    def test_expanded_plan_cli_never_loads_credentials_or_enters_execution(self):
        output = io.StringIO()
        with patch("sys.argv", ["run_agent_batch.py", "--plan", "--extended"]), \
             patch("agent_credentials.load_api_key", side_effect=AssertionError("credential access")), \
             patch("run_agent_batch.inside", side_effect=AssertionError("execution")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data["mode"], "plan_only")
        self.assertEqual(data["summary"]["final_success"]["denominator"], 96)

    def test_extended_failure_selection_is_exact_and_custom_rows_summarize(self):
        rows, _ = batch_plan(True)
        for row in rows:
            row.update(status="passed", metrics=case_metrics(successful()))
        rows[-1].update(status="failed", metrics=case_metrics({"status": "provider_failed"}))
        prior = dict(status="completed_with_failures", cases=rows)
        chosen = select_failed([r["descriptor"] for r in rows], prior)
        self.assertEqual(chosen, [rows[-1]["descriptor"]])
        data = summarize(rows)
        self.assertEqual(data["final_success"]["numerator"], 95)
        self.assertEqual(data["families"]["dual_linear"]["final_success"], 5)
        self.assertEqual(summarize([dict(descriptor=chosen[0])])["families"]["custom_graph"]["planned"], 1)
        with self.assertRaisesRegex(ValueError, "exact full catalog"):
            select_failed([r["descriptor"] for r in rows[:48]], prior)

    def test_planned_denominator_keeps_unrun_and_provider_failures(self):
        rows = [dict(descriptor=dict(family="linear"), metrics=case_metrics(successful())),
                dict(descriptor=dict(family="linear")),
                dict(descriptor=dict(family="residual"), metrics=case_metrics(dict(status="provider_failed",
                     agent_calls=1, provider_metrics=dict(calls=[dict(error="transport_timeout")]))))]
        data = summarize(rows)
        self.assertEqual(data["final_success"], dict(numerator=1, denominator=3, rate=1/3))
        self.assertEqual(data["not_completed"], 1)
        self.assertEqual(data["final_success_given_candidate"]["denominator"], 1)
        self.assertEqual(data["provider_errors"], dict(transport_timeout=1))

    def test_repaired_is_not_first_pass(self):
        report = successful()
        report["agent_calls"] = 2
        report["attempts"].insert(0, dict(status="failed", parsed=True, failure_layer="static_check", category="candidate"))
        data = case_metrics(report)
        self.assertTrue(data["passed"])
        self.assertFalse(data["first_passed"])
        self.assertEqual(data["repairs_attempted"], 1)
        self.assertFalse(data["first"]["compiled"])
        self.assertTrue(data["ever"]["compiled"])

    def test_offline_fixture_cannot_count_as_live_success(self):
        report = successful()
        report["provider"] = "deepseek_offline"
        self.assertFalse(case_metrics(report)["passed"])

    def test_incomplete_provider_output_separate_from_network(self):
        data = case_metrics(dict(status="provider_failed", provider_metrics=dict(
            calls=[dict(error="incomplete_or_refused_response")])))
        self.assertEqual(data["failure_category"], "provider_output")


if __name__ == "__main__":
    unittest.main()
