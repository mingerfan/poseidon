"""Read-only audits of real spatial FHE evidence, including 256-input cases."""
import json
import os
from pathlib import Path
import unittest

from hecate_python_env import digest
from packed_spatial_cases import cases,large_cases,paid_cases
from test_packed_evidence import check_run


class PackedSpatialEvidenceTests(unittest.TestCase):
    def test_two_rule_cohorts_and_wrong_programs(self):
        for variable,expected,wrong_count,sha in [
            ('POSEIDON_PACKED_SPATIAL_GOLDENS',cases(),2,
             '1eaa22ae87e00d406c0cb5e270cecdf289b6ac6c0035e6a9a68c8a56570bf8e7'),
            ('POSEIDON_PACKED_SPATIAL_MAX_GOLDENS',large_cases(),0,
             '84e298173a10c16558867ad18c335f290cab858fed601e4cb28ee2d0da0594c0')]:
            setting=os.environ.get(variable)
            if not setting:self.skipTest('Set '+variable+' to real FHE evidence')
            root=Path(setting);report=json.loads((root/'report.json').read_text())
            self.assertEqual(digest(root/'report.json'),sha)
            self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
            self.assertEqual(report['generator'],'deterministic_packed_spatial_fx')
            self.assertEqual(len(report['cases']),len(expected)+wrong_count)
            self.assertEqual([r['id'] for r in report['cases'][:len(expected)]],[d['id'] for d in expected])
            for row in report['cases']:
                with self.subTest(case=row['id'],wrong=row['counterexample']):
                    run=Path(row['run']);self.assertEqual(digest(run/'report.json'),row['report_sha256'])
                    check_run(self,run,not row['counterexample'])
                    if row['counterexample']:self.assertEqual(row['failure_layer'],'numerical_comparison')

    def test_paid_agent_original_response_and_independent_reference(self):
        setting=os.environ.get('POSEIDON_PACKED_SPATIAL_PAID_BATCH')
        if not setting:self.skipTest('Set POSEIDON_PACKED_SPATIAL_PAID_BATCH to real paid evidence')
        from audit_agent_lineage import audit
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        self.assertEqual(digest(root/'report.json'),
                         'ed8515adc7a7f16be623460a7b304ba8c9dce178ead220881dd59b80ff95aeba')
        self.assertEqual([r['descriptor'] for r in report['cases']],paid_cases())
        self.assertEqual(report['status'],'passed')
        evidence=audit(root/'report.json')
        self.assertEqual(evidence['passed'],16);self.assertFalse(evidence['all_goal_requirements_complete'])
        for row in report['cases']:
            child=json.loads((Path(row['evidence'])/'report.json').read_text())
            self.assertTrue(child['llm_generation_validated']);self.assertGreater(child['agent_calls'],0)
            check_run(self,Path(row['evidence']))


if __name__=='__main__':unittest.main()
