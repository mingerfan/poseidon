"""Provider migration diagnostics never turn mismatched output into acceptance."""
import json
import unittest

from deepseek_provider import Config, ProviderError, response_content, response_diagnostics
from run_agent_batch import should_pause_batch
from test_deepseek_provider import completion


class ModelIdentityDiagnosticsTests(unittest.TestCase):
    def test_retired_alias_mismatch_is_visible_but_still_rejected(self):
        config = Config(model='deepseek-v4-flash')
        _, raw = completion('UNTRUSTED CONTENT', model='deepseek-flash')
        diagnostics = response_diagnostics(raw, config)
        self.assertEqual(diagnostics['reported_model'], 'deepseek-flash')
        self.assertFalse(diagnostics['response_model_matches'])
        self.assertEqual(diagnostics['usage']['total_tokens'], 130)
        self.assertNotIn('UNTRUSTED CONTENT', json.dumps(diagnostics))
        with self.assertRaisesRegex(ProviderError, 'response_model_mismatch'):
            response_content(raw, config)

    def test_unknown_model_strings_are_redacted_not_logged(self):
        for value in ('SECRET TOKEN', ['SECRET'], None):
            _, raw = completion('SECRET', model=value)
            diagnostics = response_diagnostics(raw, Config())
            self.assertNotIn('SECRET', json.dumps(diagnostics))
            self.assertFalse(diagnostics['response_model_matches'])

    def test_invalid_usage_does_not_become_billing_evidence(self):
        _, raw = completion('{}', model='deepseek-flash', usage={'total_tokens': -1})
        diagnostics = response_diagnostics(raw, Config(model='deepseek-v4-flash'))
        self.assertEqual(diagnostics['usage_status'], 'invalid')
        self.assertNotIn('usage', diagnostics)

    def test_model_identity_failure_pauses_all_providers_but_not_candidate_errors(self):
        for provider in ('deepseek',):
            self.assertTrue(should_pause_batch(provider, dict(failure_layer='provider',
                            provider_error='response_model_mismatch')))
            self.assertFalse(should_pause_batch(provider, dict(failure_layer='numerical_comparison')))
        self.assertFalse(should_pause_batch('deepseek', dict(failure_layer='provider', provider_error='transport_timeout')))


if __name__ == '__main__':
    unittest.main()
