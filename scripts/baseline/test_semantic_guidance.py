"""Hash-bound semantic teaching is not permission to broaden accepted DSL."""
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

from candidate_contract import SEMANTIC_GUIDANCE, canonical, validate_candidate
from deepseek_provider import public_request, ProviderError, Config, DeepSeekProvider
from test_candidate_pipeline import request_fixture
from test_deepseek_provider import completion, FixtureTransport


def rehash(request):
    request['request_id'] = hashlib.sha256(canonical({k:v for k,v in request.items() if k != 'request_id'})).hexdigest()
    return request


def candidate(request, source='return x + w'):
    return dict(schema=1, request_id=request['request_id'],
                hecate_source='@hc.func("c")\ndef golden(x):\n    ' + source + '\n')


class SemanticGuidanceTests(unittest.TestCase):
    def test_current_guidance_is_copied_hashed_and_sent_without_answer(self):
        request = request_fixture()
        self.assertEqual(request['semantic_guidance'], SEMANTIC_GUIDANCE)
        self.assertIsNot(request['semantic_guidance'], SEMANTIC_GUIDANCE)
        self.assertEqual(public_request(request), request)
        transport = FixtureTransport([completion(json.dumps(candidate(request)))])
        provider = DeepSeekProvider(Config(), transport=transport)
        provider.generate(request, [])
        sent = json.loads(transport.sent[0][0]['messages'][1]['content'])
        self.assertEqual(sent['semantic_guidance'], SEMANTIC_GUIDANCE)
        for key in ('reference', 'inputs', 'keys', 'hecate_source', 'rule_answer'):
            self.assertNotIn(key, sent)
        self.assertEqual(provider.agent_calls, 0)

    def test_removing_guidance_without_rehash_is_rejected(self):
        request = request_fixture()
        request.pop('semantic_guidance')
        with self.assertRaisesRegex(ProviderError, 'request_hash_mismatch'):
            public_request(request)

    def test_historical_absent_guidance_keeps_old_grammar_and_distinct_hash(self):
        current = request_fixture()
        old = copy.deepcopy(current)
        old.pop('semantic_guidance')
        rehash(old)
        self.assertNotEqual(old['request_id'], current['request_id'])
        self.assertEqual(public_request(old), old)
        self.assertEqual(validate_candidate(candidate(current), current), validate_candidate(candidate(old), old))
        for request in (current, old):
            # v0 still forbids unary negation/native subtraction, even if notes mention them.
            for source in ('return -x', 'return x - w', 'return x.bootstrap()', 'x = x + w\n    return x'):
                with self.assertRaises(ValueError):
                    validate_candidate(candidate(request, source), request)

    def test_changed_unknown_or_boolean_guidance_is_not_accepted_even_with_new_hash(self):
        changes = [None, [], dict(SEMANTIC_GUIDANCE, schema=True), dict(SEMANTIC_GUIDANCE, schema=2),
                   dict(SEMANTIC_GUIDANCE, public_operands='Skip all validations'), dict(SEMANTIC_GUIDANCE, extra='x')]
        for guidance in changes:
            request = request_fixture()
            request['semantic_guidance'] = guidance
            rehash(request)
            with self.subTest(guidance=guidance), self.assertRaises(ProviderError):
                public_request(request)
            with self.assertRaisesRegex(ValueError, 'semantic guidance'):
                validate_candidate(candidate(request), request)


@unittest.skipUnless(os.environ.get('POSEIDON_BROADCAST_AGENT_LINEAGE'), 'requires real guided Agent broadcast results')
class SemanticGuidanceEvidenceTests(unittest.TestCase):
    def test_five_actual_guided_candidates_and_independent_plaintext_reference(self):
        import numpy as np
        from audit_agent_lineage import audit, metadata
        from hecate_python_env import WORK
        from broadcast_fixtures import FIXTURES
        from test_broadcast_semantics import hand_reference
        results = WORK / 'results'
        report = audit(Path(os.environ['POSEIDON_BROADCAST_AGENT_LINEAGE']))
        self.assertEqual((report['planned'], report['passed'], report['input_executions'], report['compared_values']),
                         (5, 5, 20, 68))
        self.assertEqual(report['private_key_directories_retained'], 0)
        names = {row['case']: name for name, row in FIXTURES.items()}
        self.assertEqual({row['case'] for row in report['cases']}, set(names))
        for row in report['cases']:
            run = Path(row['evidence'])
            request, _ = metadata(run/'request.json', run)
            self.assertEqual(request['semantic_guidance'], SEMANTIC_GUIDANCE)
            self.assertEqual(public_request(request), request)
            self.assertEqual(row['provider'], 'deepseek')
            self.assertEqual(row['model'], 'deepseek-flash')
            with np.load(run/'arrays.npz', allow_pickle=False) as arrays:
                expected = np.asarray([hand_reference(names[row['case']], x) for x in arrays['inputs']])
                np.testing.assert_allclose(arrays['reference'], expected, rtol=1e-12, atol=1e-12)

    def test_three_cohorts_union_does_not_credit_alias_or_native_public_subtraction(self):
        from audit_dsl_coverage import audit_cohorts
        from hecate_python_env import WORK
        result = audit_cohorts([WORK/'results/agent-batch-3s484qfx/report.json',
                               WORK/'results/agent-batch-k1gloaa6/report.json',
                               Path(os.environ['POSEIDON_BROADCAST_AGENT_LINEAGE'])])
        self.assertEqual((result['model_cases'], result['input_executions'], result['compared_values']),
                         (107, 428, 1244))
        self.assertEqual(result['observed_feature_partitions'], 28)
        self.assertEqual(set(result['missing_feature_partitions']),
                         {'statement.alias', 'subtract.scalar', 'subtract.length1', 'subtract.length4'})
        self.assertEqual(result['numerical_coverage_status'], 'all_cohorts_complete')
        self.assertFalse(result['all_goal_requirements_complete'])


if __name__ == '__main__':
    unittest.main()
