"""Offline audits of real rank-preserving rule and Agent executions."""
import json
import os
from pathlib import Path
import unittest

from hecate_python_env import digest
from tensor_model_cases import cases
from test_packed_evidence import check_run


class TensorEvidenceTests(unittest.TestCase):
    def test_rule_baseline_and_wrong_group(self):
        setting=os.environ.get('POSEIDON_TENSOR_GOLDENS')
        if not setting:self.skipTest('Set POSEIDON_TENSOR_GOLDENS to real FHE evidence')
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
        self.assertEqual(report['generator'],'deterministic_tensor_fx')
        self.assertEqual(len(report['cases']),14)
        self.assertEqual([r['id'] for r in report['cases'][:12]],[d['id'] for d in cases()])
        for row in report['cases']:
            with self.subTest(case=row['id'],wrong=row['counterexample']):
                run=Path(row['run']);self.assertEqual(digest(run/'report.json'),row['report_sha256'])
                check_run(self,run,not row['counterexample'])
        last=report['cases'][-1]
        self.assertTrue(last['counterexample']);self.assertEqual(last['failure_layer'],'numerical_comparison')

    def test_real_agent_source_logical_shape_and_reference(self):
        setting=os.environ.get('POSEIDON_TENSOR_PAID_BATCH')
        if not setting:self.skipTest('Set POSEIDON_TENSOR_PAID_BATCH to paid FHE evidence')
        from audit_agent_lineage import audit
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        self.assertEqual([r['descriptor'] for r in report['cases']],cases())
        self.assertEqual(report['status'],'passed')
        evidence=audit(root/'report.json')
        self.assertEqual(evidence['passed'],12);self.assertFalse(evidence['all_goal_requirements_complete'])
        for row in report['cases']:
            child=json.loads((Path(row['evidence'])/'report.json').read_text())
            self.assertTrue(child['llm_generation_validated']);self.assertGreater(child['agent_calls'],0)
            check_run(self,Path(row['evidence']))


if __name__=='__main__':unittest.main()
