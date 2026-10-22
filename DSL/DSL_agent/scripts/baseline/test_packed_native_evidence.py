"""Read-only real native + packed evidence; fixtures and Agent results distinct."""
import json
import ast
import os
from pathlib import Path
import unittest
from candidate_contract import PACKED_NATIVE_GUIDANCE
from hecate_python_env import digest
from packed_input_abi import NATIVE_TASK,NATIVE_CONTRACT
from packed_native_cases import cases
from test_packed_evidence import check_run

def read(path):return json.loads(path.read_text())

def check_native(test,run,expected=True):
    report=read(run/'report.json');request=read(run/'request.json')
    test.assertEqual(request['task'],NATIVE_TASK)
    test.assertEqual(request['semantic_guidance'],PACKED_NATIVE_GUIDANCE)
    check_run(test,run,expected)
    attempt=report['attempts'][-1];folder=run/f"attempt-{attempt['index']:02d}"/'output'
    checked=attempt['static_check'];test.assertEqual(checked['contract'],NATIVE_CONTRACT)
    test.assertEqual(read(folder/'native-function-plan.json'),json.loads(json.dumps(checked['native_functions'])))
    trace=read(folder/'trace-evidence.json')
    test.assertEqual(trace['construction'],'validated_native_AST_to_Hecate_functions')
    test.assertFalse(trace['candidate_python_executed'])
    test.assertEqual(checked['native_functions']['packed_binding']['slot_period'],request['layout']['input_slot_period'])
    # Every recorded runtime augmented event must preserve the native alias contract.
    for item in read(folder/'native-augmented-events.json'):
        test.assertTrue(item['fresh_expr'] and item['original_expr_unchanged'])
    for item in read(folder/'native-array-mutation-events.json'):
        test.assertTrue(item['same_array_object'] and item['original_exprs_unchanged'])
    from native_array_alias import trace_projection
    test.assertCountEqual(read(folder/'native-array-mutation-events.json'),
                          trace_projection(checked['native_functions']['array_mutation']['sites']))
    return folder

class PackedNativeEvidenceTests(unittest.TestCase):
    def test_rule_and_wrong_alias_or_order(self):
        setting=os.environ.get('POSEIDON_PACKED_NATIVE_GOLDENS')
        if not setting:self.skipTest('Set POSEIDON_PACKED_NATIVE_GOLDENS to real FHE report')
        root=Path(setting);report=read(root/'report.json')
        self.assertEqual(digest(root/'report.json'),
                         'a7ac662dd85eb2ecb7043d1a5f0cd7b4d63d81023a8260e6e81efa20e69c977d')
        self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
        self.assertEqual(report['generator'],'deterministic_packed_native_fixture')
        self.assertEqual(len(report['cases']),8)
        self.assertEqual([r['id'] for r in report['cases'][:6]],[d['id'] for d in cases()])
        self.assertEqual(sum(r['counterexample'] for r in report['cases']),2)
        for i,row in enumerate(report['cases']):
            with self.subTest(case=row['id'],wrong=row['counterexample']):
                run=Path(row['run']);self.assertEqual(digest(run/'report.json'),row['report_sha256'])
                output=check_native(self,run,not row['counterexample'])
                self.assertTrue(read(output/'native-call-events.json'))
                if i==0:self.assertEqual(len(read(output/'native-array-mutation-events.json')),2)
                if i==1:self.assertEqual(len(read(output/'native-augmented-events.json')),2)
                if row['counterexample']:self.assertEqual(row['failure_layer'],'numerical_comparison')

    def test_paid_original_candidate_and_reference(self):
        setting=os.environ.get('POSEIDON_PACKED_NATIVE_PAID_BATCH')
        if not setting:self.skipTest('Set POSEIDON_PACKED_NATIVE_PAID_BATCH to actual Agent cohort')
        from audit_agent_lineage import audit
        root=Path(setting);report=read(root/'report.json')
        self.assertEqual([r['descriptor'] for r in report['cases']],cases())
        self.assertEqual(digest(root/'report.json'),
                         'f1bcbbf4d29e669292f005a04e3c5f1ce72f2c47a88d31db3d71db58888391b7')
        self.assertEqual(report['status'],'passed');checked=audit(root/'report.json')
        self.assertEqual(checked['passed'],6);self.assertFalse(checked['all_goal_requirements_complete'])
        for row in report['cases']:
            run=Path(row['evidence']);child=read(run/'report.json')
            self.assertTrue(child['llm_generation_validated']);self.assertGreater(child['agent_calls'],0)
            check_native(self,run)

    def test_paid_construct_usage_and_failed_original_interfaces(self):
        setting=os.environ.get('POSEIDON_PACKED_NATIVE_PAID_BATCH')
        if not setting:self.skipTest('Requires actual paid cohort')
        root=Path(setting);report=read(root/'report.json');helper_programs=0;calls=0
        for row in report['cases']:
            run=Path(row['evidence']);child=read(run/'report.json');a=child['attempts'][-1]
            folder=run/f"attempt-{a['index']:02d}";plan=a['static_check']['native_functions']
            reachable=set();todo=['golden']
            while todo:
                name=todo.pop()
                if name in reachable:continue
                reachable.add(name);todo.extend(plan['functions'][name]['calls'])
            helper_programs+=len(reachable)>1
            events=read(folder/'output/native-call-events.json');calls+=len(events)
            nodes=[n for fun in ast.parse((folder/'candidate.py').read_text()).body
                   if fun.name in reachable for n in ast.walk(fun)]
            self.assertFalse(any(isinstance(n,(ast.For,ast.AugAssign,ast.Starred)) for n in nodes))
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                                 and n.func.attr=='array' for n in nodes))
        self.assertEqual((helper_programs,calls),(2,22))
        self.assertEqual(report['summary']['attempt_failure_layers'],{'static_check':4})
        self.assertEqual(report['summary']['first_success']['numerator'],3)
        self.assertEqual(report['summary']['api_calls'],10)
        rows=report['cases']
        affine=Path(rows[0]['evidence'])
        source=read(affine/'attempt-00/response.txt')['hecate_source']
        self.assertEqual([n.arg for n in ast.parse(source).body[0].args.args],['x','c0','c1'])
        zero=Path(rows[3]['evidence'])
        self.assertIn('rotate(v1, 32)',read(zero/'attempt-00/response.txt')['hecate_source'])
        self.assertIn('hc.rotate(v1, 32)',read(zero/'attempt-01/response.txt')['hecate_source'])
        self.assertIn('v1.rotate(32)',read(zero/'attempt-02/response.txt')['hecate_source'])

if __name__=='__main__':unittest.main()
