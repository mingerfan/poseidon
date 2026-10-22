"""Reaudit real BatchNorm/concat outputs without generating new paid requests."""
import json
import os
from pathlib import Path
import unittest

from hecate_python_env import digest
from packed_composition_cases import cases
from test_packed_evidence import check_run


class PackedCompositionEvidenceTests(unittest.TestCase):
    def test_rule_programs_and_semantic_counterexamples(self):
        setting=os.environ.get('POSEIDON_PACKED_COMPOSITION_GOLDENS')
        if not setting:self.skipTest('Set POSEIDON_PACKED_COMPOSITION_GOLDENS to real evidence')
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        self.assertEqual(digest(root/'report.json'),
                         '3f7e421fcb727076baaf2c2af4f66bd8d886acce575d6bfccf432780e3b4d1af')
        self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
        self.assertEqual(report['generator'],'deterministic_packed_composition_fx')
        self.assertEqual(len(report['cases']),14)
        self.assertEqual([r['id'] for r in report['cases'][:12]],[d['id'] for d in cases()])
        self.assertEqual(sum(r['counterexample'] for r in report['cases']),2)
        for row in report['cases']:
            with self.subTest(case=row['id'],wrong=row['counterexample']):
                run=Path(row['run']);self.assertEqual(digest(run/'report.json'),row['report_sha256'])
                check_run(self,run,not row['counterexample'])
                if row['counterexample']:self.assertEqual(row['failure_layer'],'numerical_comparison')

    def test_paid_responses_and_independent_original_references(self):
        setting=os.environ.get('POSEIDON_PACKED_COMPOSITION_PAID_BATCH')
        if not setting:self.skipTest('Set POSEIDON_PACKED_COMPOSITION_PAID_BATCH to real paid evidence')
        from audit_agent_lineage import audit
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        self.assertEqual(digest(root/'report.json'),
                         '0ab6ba6f2f292f872ef2f3d4bd8574324661d5d96b6f7e6ded9ce4f82521f429')
        self.assertEqual([r['descriptor'] for r in report['cases']],cases())
        self.assertEqual(report['status'],'passed')
        checked=audit(root/'report.json')
        self.assertEqual(checked['passed'],12);self.assertFalse(checked['all_goal_requirements_complete'])
        for row in report['cases']:
            run=Path(row['evidence']);child=json.loads((run/'report.json').read_text())
            self.assertTrue(child['llm_generation_validated']);self.assertGreater(child['agent_calls'],0)
            check_run(self,run)


if __name__=='__main__':unittest.main()
