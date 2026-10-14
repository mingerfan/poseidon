"""Frozen original responses, versioned prompts and real FHE layout evidence."""
import json
import os
from pathlib import Path
import unittest

from candidate_contract import PACKED_GUIDANCE,PACKED_GUIDANCE_V2
from hecate_python_env import digest
from layout_guidance_cases import cases
from test_packed_evidence import check_run


def read(path):return json.loads(path.read_text())


class LayoutGuidanceEvidenceTests(unittest.TestCase):
    def test_rule_baseline(self):
        setting=os.environ.get('POSEIDON_LAYOUT_GUIDANCE_GOLDENS')
        if not setting:self.skipTest('Set POSEIDON_LAYOUT_GUIDANCE_GOLDENS to real FHE evidence')
        root=Path(setting);report=read(root/'report.json')
        self.assertEqual(digest(root/'report.json'),
                         '7a941d16497ce4539076adf5eb55685160af6605950a8df8f065f0454029402e')
        self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
        self.assertEqual(report['generator'],'deterministic_layout_guidance_fx')
        self.assertEqual([r['id'] for r in report['cases']],[d['id'] for d in cases()])
        for row in report['cases']:
            with self.subTest(case=row['id']):
                self.assertFalse(row['counterexample']);run=Path(row['run'])
                self.assertEqual(digest(run/'report.json'),row['report_sha256'])
                self.assertEqual(read(run/'request.json')['semantic_guidance'],PACKED_GUIDANCE_V2)
                check_run(self,run)

    def test_paid_original_responses_and_unchanged_historical_failures(self):
        setting=os.environ.get('POSEIDON_LAYOUT_GUIDANCE_PAID_BATCH')
        previous=os.environ.get('POSEIDON_PERMUTATION_PAID_BATCH')
        if not setting or not previous:self.skipTest('Set both current and prior real paid cohorts')
        from audit_agent_lineage import audit
        root=Path(setting);report=read(root/'report.json')
        oldroot=Path(previous);old=read(oldroot/'report.json')
        self.assertEqual(digest(root/'report.json'),
                         '1dc8abc94e6caee2067c06a2f0bd9de6120439b985cf04bbcd99601b0de46547')
        self.assertEqual(digest(oldroot/'report.json'),
                         '5e8a22652c0f69f4e5bd260beab9102d2f2e2bc629b4beb1ff05b404dc6fc3a5')
        self.assertEqual(report['status'],'passed')
        self.assertEqual([r['descriptor'] for r in report['cases']],cases())
        checked=audit(root/'report.json')
        self.assertEqual(checked['passed'],6);self.assertFalse(checked['all_goal_requirements_complete'])
        self.assertEqual((checked['input_executions'],checked['compared_values']),(24,512))
        for row in report['cases']:
            run=Path(row['evidence']);child=read(run/'report.json');request=read(run/'request.json')
            self.assertTrue(child['llm_generation_validated']);self.assertGreater(child['agent_calls'],0)
            self.assertEqual(request['semantic_guidance'],PACKED_GUIDANCE_V2)
            check_run(self,run)
            if row['descriptor']['id'] in {d['id'] for d in cases()[:2]}:
                prior=next(r for r in old['cases'] if r['descriptor']['id']==row['descriptor']['id'])
                priorrun=Path(prior['evidence']);priorrequest=read(priorrun/'request.json')
                self.assertEqual(priorrequest['semantic_guidance'],PACKED_GUIDANCE)
                self.assertNotEqual(priorrequest['request_id'],request['request_id'])
                # Only guidance/request hash changes; model, constants, FX and ABI stay fixed.
                self.assertEqual({k:v for k,v in priorrequest.items() if k not in ('request_id','semantic_guidance')},
                                 {k:v for k,v in request.items() if k not in ('request_id','semantic_guidance')})
                self.assertEqual(read(priorrun/'report.json')['attempts'][0]['failure_layer'],'numerical_comparison')


if __name__=='__main__':unittest.main()
