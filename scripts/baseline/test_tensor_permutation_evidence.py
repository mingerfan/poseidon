"""Read-only real-FHE evidence checks for logical and physical axis ordering."""
import ast
import copy
import json
import os
from pathlib import Path
import unittest

from hecate_python_env import digest
from permutation_model_cases import cases
from test_packed_evidence import check_run


class TensorPermutationEvidenceTests(unittest.TestCase):
    def test_rule_and_wrong_permutations(self):
        setting=os.environ.get('POSEIDON_PERMUTATION_GOLDENS')
        if not setting:self.skipTest('Set POSEIDON_PERMUTATION_GOLDENS to real evidence')
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        self.assertEqual(digest(root/'report.json'),
                         '6eb796008ce3a82f1357a4a74215c42f0d44ad7beaad2416cb5935b02a5a1059')
        self.assertEqual(report['status'],'passed');self.assertEqual(report['agent_calls'],0)
        self.assertEqual(report['generator'],'deterministic_tensor_permutation_fx')
        self.assertEqual(len(report['cases']),14)
        self.assertEqual([r['id'] for r in report['cases'][:12]],[d['id'] for d in cases()])
        self.assertEqual(sum(r['counterexample'] for r in report['cases']),2)
        for row in report['cases']:
            with self.subTest(case=row['id'],wrong=row['counterexample']):
                run=Path(row['run']);self.assertEqual(digest(run/'report.json'),row['report_sha256'])
                check_run(self,run,not row['counterexample'])
                if row['counterexample']:self.assertEqual(row['failure_layer'],'numerical_comparison')

    def test_paid_original_response_and_independent_reference(self):
        setting=os.environ.get('POSEIDON_PERMUTATION_PAID_BATCH')
        if not setting:self.skipTest('Set POSEIDON_PERMUTATION_PAID_BATCH to real paid evidence')
        from audit_agent_lineage import audit
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        self.assertEqual(digest(root/'report.json'),
                         '5e8a22652c0f69f4e5bd260beab9102d2f2e2bc629b4beb1ff05b404dc6fc3a5')
        self.assertEqual([r['descriptor'] for r in report['cases']],cases())
        self.assertEqual(report['status'],'passed')
        checked=audit(root/'report.json')
        self.assertEqual(checked['passed'],12);self.assertFalse(checked['all_goal_requirements_complete'])
        for row in report['cases']:
            run=Path(row['evidence']);child=json.loads((run/'report.json').read_text())
            self.assertTrue(child['llm_generation_validated']);self.assertGreater(child['agent_calls'],0)
            check_run(self,run)

    def test_original_numeric_failures_have_specific_layout_causes(self):
        setting=os.environ.get('POSEIDON_PERMUTATION_PAID_BATCH')
        if not setting:self.skipTest('Set POSEIDON_PERMUTATION_PAID_BATCH to real paid evidence')
        import numpy as np
        from model_graph import evaluate_reference
        from seal_cpu_golden import compare
        root=Path(setting);report=json.loads((root/'report.json').read_text())
        rows={r['descriptor']['id']:r for r in report['cases']}
        for name in ('perm-negative-axes','perm-nhwc-conv'):
            run=Path(rows[name]['evidence']);child=json.loads((run/'report.json').read_text())
            first=child['attempts'][0]
            self.assertEqual(first['failure_layer'],'numerical_comparison')
            self.assertTrue(first['compiled'] and first['executed'])
            self.assertFalse(first['numerically_correct'])
            self.assertEqual(child['status'],'passed')
        # Exact saved mask-then-rotate programs differ by rotation sign, not
        # by different public masks or a changed reference.
        run=Path(rows['perm-negative-axes']['evidence'])
        def masked_shifts(path):
            fn=ast.parse(path.read_text()).body[0];result={};current=None
            for statement in fn.body:
                if isinstance(statement,ast.Assign):
                    target=statement.targets[0].id;value=statement.value
                    if isinstance(value,ast.BinOp):
                        self.assertIsInstance(value.op,ast.Mult)
                        self.assertEqual(value.left.id,'x')
                        current=[value.right.id,0]
                        if target=='out':result[current[0]]=0
                        else:self.assertEqual(target,'t')
                    else:
                        self.assertIsInstance(value,ast.Call);self.assertEqual(target,'t')
                        self.assertEqual(value.func.value.id,'t');self.assertEqual(value.func.attr,'rotate')
                        current[1]+=value.args[0].value
                elif isinstance(statement,ast.AugAssign):
                    self.assertIsInstance(statement.op,ast.Add)
                    self.assertEqual((statement.target.id,statement.value.id),('out','t'))
                    self.assertNotIn(current[0],result);result[current[0]]=current[1]
                else:
                    self.assertIsInstance(statement,ast.Return);self.assertEqual(statement.value.id,'out')
            return result
        before=masked_shifts(run/'attempt-00/candidate.py');after=masked_shifts(run/'attempt-01/candidate.py')
        self.assertEqual(set(before),set(after));self.assertNotEqual(before,after)
        period=json.loads((run/'request.json').read_text())['layout']['input_slot_period']
        for key in before:self.assertEqual((before[key]+after[key])%period,0)
        # First NHWC candidate computes the otherwise identical Conv with a
        # reshape-only view. This is a diagnostic counter-model, NEVER the
        # immutable correctness reference or a reason to accept that candidate.
        run=Path(rows['perm-nhwc-conv']['evidence']);model=json.loads((run/'model.json').read_text())
        wrong=copy.deepcopy(model);first=wrong['nodes'][0]
        wrong['nodes'][0]=dict(id=first['id'],op='reshape',inputs=first['inputs'],shape=[1,2,4,4])
        with np.load(run/'arrays.npz',allow_pickle=False) as data:
            alternative=np.asarray([evaluate_reference(wrong,x.tolist()) for x in data['logical_inputs']])
            original=data['reference'].copy()
        for attempt in ('attempt-00','attempt-01'):
            actual=np.load(run/attempt/'output/decrypted.npy',allow_pickle=False)
            self.assertFalse(compare(actual,original,1e-5,1e-4)['passed'])
            self.assertTrue(compare(actual,alternative,1e-5,1e-4)['passed'])


if __name__=='__main__':unittest.main()
