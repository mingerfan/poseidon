import unittest

from pt_agent.experiments.s4_qwen_post import eligibility


class IsolatedPostGateTests(unittest.TestCase):
    def test_native_failure_preserves_frozen_semantic_count_and_valid_report(self):
        result = eligibility(dict(frozen=True, compile_accepted=False, phases=dict(math=dict(math_passed=True))))
        self.assertTrue(result['count'])
        self.assertTrue(result['report'])
        self.assertFalse(result['compiled'])

    def test_report_requires_strict_math_and_count_requires_freeze(self):
        frozen = eligibility(dict(frozen=True, compile_accepted=False, phases={}))
        self.assertTrue(frozen['count'])
        self.assertFalse(frozen['report'])
        empty = eligibility(dict(frozen=False, compile_accepted=False, phases={}))
        self.assertFalse(empty['count'])
        self.assertFalse(empty['report'])


if __name__ == '__main__':
    unittest.main()
