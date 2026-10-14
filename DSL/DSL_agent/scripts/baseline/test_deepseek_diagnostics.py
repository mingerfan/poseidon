import json
import unittest

from deepseek_provider import Config, DeepSeekProvider, ProviderError, response_diagnostics
from test_deepseek_provider import FixtureTransport, completion
from test_candidate_pipeline import request_fixture
from run_agent_batch import select_failed


class ResponseDiagnosticsTests(unittest.TestCase):
    def response(self, finish="length", **extra):
        return completion("unused", choices=[dict(finish_reason=finish,
            message=dict(role="assistant", content="SECRET partial", reasoning_content="SECRET thinking"))], **extra)

    def test_truncation_retains_usage_but_never_text_or_candidate(self):
        provider = DeepSeekProvider(transport=FixtureTransport([self.response()]))
        with self.assertRaisesRegex(ProviderError, "incomplete_or_refused_response"):
            provider.generate(request_fixture(), [])
        entry = provider.metrics()["calls"][0]
        self.assertEqual(entry["finish_reason"], "length")
        self.assertEqual(entry["usage"]["total_tokens"], 130)
        self.assertEqual(entry["usage"]["reasoning_tokens"], 10)
        self.assertNotIn("SECRET", json.dumps(entry))
        self.assertEqual(provider._responses, [])
        with self.assertRaisesRegex(ProviderError, "no_retry"):
            provider.generate(request_fixture(), [])

    def test_unknown_finish_reason_is_not_echoed(self):
        result = response_diagnostics(self.response("SECRET unexpected")[1], Config())
        self.assertEqual(result["finish_reason"], "unknown")
        self.assertNotIn("SECRET", json.dumps(result))

    def test_missing_and_invalid_usage_do_not_hide_finish(self):
        for usage, expected in [(None, "missing"), ({"prompt_tokens": "SECRET"}, "invalid"),
                                (dict(prompt_tokens=True, completion_tokens=1, total_tokens=2), "invalid")]:
            result = response_diagnostics(self.response(usage=usage)[1], Config())
            self.assertEqual(result["finish_reason"], "length")
            self.assertEqual(result["usage_status"], expected)
            self.assertNotIn("usage", result)
            self.assertNotIn("SECRET", json.dumps(result))

    def test_malformed_response_metadata_unavailable(self):
        self.assertEqual(response_diagnostics(b"SECRET", Config()), dict(response_metadata_status="unavailable"))

    def test_maximum_budget_and_reasoning_counts_preserved(self):
        usage = dict(prompt_tokens=100, completion_tokens=8192, total_tokens=8292,
                     completion_tokens_details=dict(reasoning_tokens=8192))
        result = response_diagnostics(self.response(usage=usage)[1], Config())
        self.assertEqual(result["usage"]["reasoning_tokens"], 8192)
        self.assertEqual(result["usage_status"], "valid")


class FailedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = [dict(id="linear-"+str(i)) for i in range(3)]
        self.prior = dict(status="completed_with_failures", cases=[dict(descriptor=d,
            status="failed" if i == 1 else "passed", metrics=dict(passed=i != 1)) for i,d in enumerate(self.catalog)])

    def test_only_previous_failures_selected_and_source_not_changed(self):
        before = json.dumps(self.prior)
        self.assertEqual(select_failed(self.catalog, self.prior), [self.catalog[1]])
        self.assertEqual(json.dumps(self.prior), before)

    def test_incomplete_or_changed_catalog_rejected(self):
        with self.assertRaises(ValueError):
            select_failed(self.catalog, dict(self.prior, status="running"))
        with self.assertRaises(ValueError):
            select_failed(list(reversed(self.catalog)), self.prior)

    def test_inconsistent_outcome_rejected(self):
        self.prior["cases"][1]["status"] = "passed"
        with self.assertRaises(ValueError):
            select_failed(self.catalog, self.prior)


if __name__ == "__main__":
    unittest.main()
