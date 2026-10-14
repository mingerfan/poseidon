"""Union metrics retain provenance and do not inflate shared candidate evidence."""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from audit_dsl_coverage import audit_cohorts


def case(identity, feature):
    return dict(case='same-user-id', evidence=str(Path(identity).resolve()), report_sha256='a'*64,
                input_executions=4, compared_values=16, max_absolute_error=1e-8,
                grammar=dict(present_counts={feature: 1}, output_dependency_counts={feature: 1}))


def cohort(cases, source='source', status='coverage_complete'):
    return dict(cases=cases, source_batches=[dict(report=source, sha256='b'*64)],
                numerical_coverage_status=status, model_cases=len(cases), family_passes={'custom_graph': len(cases)})


class CoverageCohortTests(unittest.TestCase):
    def test_shared_runs_count_once_but_both_provenances_remain(self):
        shared = case('run1', 'add.length1')
        with patch('audit_dsl_coverage.audit_coverage', side_effect=[cohort([shared]), cohort([shared, case('run2', 'negate.cipher')])]):
            result = audit_cohorts(['report1', 'report2'])
        self.assertEqual(result['model_cases'], 2)
        self.assertEqual(result['input_executions'], 8)
        self.assertEqual(result['compared_values'], 32)
        self.assertEqual(len(result['source_batches']), 1)
        self.assertEqual(len(result['cases'][0]['cohort_reports']), 2)
        self.assertEqual(result['observed_feature_partitions'], 2)
        self.assertNotIn('success_rate', result)
        self.assertFalse(result['all_semantics_verified'])
        labels = [name for feature in result['features'] for name in feature['output_dependency_cases']]
        self.assertEqual(len(set(labels)), 2)

    def test_duplicate_reports_and_bad_count_rejected_before_audit(self):
        for values in ([], ['r']*2, [str(i) for i in range(9)]):
            with patch('audit_dsl_coverage.audit_coverage', side_effect=AssertionError('must not run')):
                with self.assertRaises(ValueError):
                    audit_cohorts(values)

    def test_changed_shared_evidence_or_batch_rejected(self):
        row = case('run1', 'add.length1')
        first = cohort([row])
        for mode in ('run', 'batch'):
            second = copy.deepcopy(first)
            if mode == 'run':
                second['cases'][0]['report_sha256'] = 'c'*64
            else:
                second['source_batches'][0]['sha256'] = 'c'*64
            with patch('audit_dsl_coverage.audit_coverage', side_effect=[first, second]):
                with self.assertRaises(ValueError):
                    audit_cohorts(['r1', 'r2'])

    def test_missing_cohort_cases_are_not_declared_complete(self):
        with patch('audit_dsl_coverage.audit_coverage', return_value=cohort([], status='coverage_incomplete')):
            result = audit_cohorts(['r1'])
        self.assertEqual(result['numerical_coverage_status'], 'incomplete_cohorts')
        self.assertEqual(result['model_cases'], 0)
        self.assertIsNone(result['max_absolute_error'])


@unittest.skipUnless(os.environ.get('POSEIDON_COVERAGE_COHORTS'), 'requires saved Agent cohort reports')
class CoverageCohortEvidenceTests(unittest.TestCase):
    def test_real_old96_new6_cpu_evidence_union(self):
        import json
        paths = json.loads(os.environ['POSEIDON_COVERAGE_COHORTS'])
        result = audit_cohorts(paths)
        self.assertEqual(result['model_cases'], 102)
        self.assertEqual(result['input_executions'], 408)
        self.assertEqual(result['compared_values'], 1176)
        self.assertEqual(result['observed_feature_partitions'], 26)
        self.assertEqual(result['numerical_coverage_status'], 'all_cohorts_complete')
        self.assertEqual(set(result['missing_feature_partitions']), {'statement.alias', 'add.length1',
                         'multiply.length1', 'subtract.scalar', 'subtract.length1', 'subtract.length4'})
        self.assertEqual(result['new_api_calls'], 0)
        self.assertFalse(result['all_goal_requirements_complete'])


if __name__ == '__main__':
    unittest.main()
