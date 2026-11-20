import unittest

from pt_agent.experiments.agent_cold_stage_partition import nanoseconds, partition


def row(event, stage, parent, inclusive, exclusive):
    return dict(event_id=event, stage_id=stage, parent_id=parent, status='ok',
        inclusive_ns=str(inclusive), exclusive_ns=str(exclusive))


class StagePartitionTest(unittest.TestCase):
    def fixture(self):
        return [row('a:1', 'fresh_end_to_end', '', 100, 10),
            row('a:2', 'agent', 'a:1', 70, 5),
            row('b:1', 'llm_call', 'a:2', 30, 30),
            row('b:2', 'agent_tune', 'a:2', 35, 35),
            row('a:3', 'compile', 'a:1', 20, 3),
            row('c:1', 'native_compile', 'a:3', 17, 17)]

    def test_exact_partition_with_cross_process_parents_and_persistence(self):
        got = partition(self.fixture(), '0.000000123')
        self.assertEqual(got['attribution_sum_ns'], 123)
        self.assertEqual(got['category_ns']['llm_requests'], 30)
        self.assertEqual(got['category_ns']['candidate_tuning'], 35)
        self.assertEqual(got['category_ns']['agent_context_and_session'], 5)
        self.assertEqual(got['category_ns']['native_compile_stage'], 20)
        self.assertEqual(got['category_ns']['outside_root_span_setup_and_completion'], 23)

    def test_duplicate_or_missing_parent_rejected(self):
        rows = self.fixture()
        with self.assertRaises(ValueError):partition(rows+[rows[1]], '1')
        with self.assertRaises(ValueError):partition(rows[:2]+rows[3:], '1')

    def test_child_double_count_or_overlap_rejected(self):
        rows = self.fixture()
        rows[1]['exclusive_ns'] = '6'
        with self.assertRaises(ValueError):partition(rows, '1')

    def test_wrong_root_failed_or_negative_span_rejected(self):
        for field, value in [('stage_id', 'continuation'), ('status', 'error'), ('exclusive_ns', '-1')]:
            rows = self.fixture();rows[0][field] = value
            with self.assertRaises(ValueError):partition(rows, '1')

    def test_completion_preceding_root_rejected(self):
        with self.assertRaises(ValueError):partition(self.fixture(), '0.000000099')

    def test_decimal_nanoseconds_are_exact(self):
        self.assertEqual(nanoseconds('6880.651302895'), 6880651302895)
        for value in ['NaN', '-1', '0.0000000001']:
            with self.assertRaises(ValueError):nanoseconds(value)


if __name__ == '__main__':unittest.main()
