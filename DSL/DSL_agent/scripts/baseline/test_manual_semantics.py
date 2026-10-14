"""Evidence audit tests: forged metrics/reference must not create a FHE pass."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from manual_semantic_evidence import recompute_arrays


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'requires pinned numerical environment')
class RecomputedArrayTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from model_graph import evaluate_reference
        from seal_cpu_golden import compare
        self.np, self.compare = np, compare
        self.model = dict(schema=2,id='user-affine',input_shape=[4],constants={'w':.5,'b':.125},
            nodes=[dict(id='m',op='multiply',inputs=['x','w']),dict(id='out',op='add',inputs=['m','b'])],output='out')
        self.inputs=np.asarray([[0,0,0,0],[.5,-1,.25,-.75],[.1,-.2,.3,-.4],[-1,1,-1,1]],dtype=np.float64)
        self.reference=np.asarray([evaluate_reference(self.model,x.tolist()) for x in self.inputs])
        self.actual=self.reference+1e-9
        self.recorded=compare(self.actual,self.reference,1e-5,1e-4)

    def audit(self, **updates):
        args=dict(model=self.model,inputs=self.inputs,reference=self.reference,actual=self.actual,
                  recorded=self.recorded,expected_pass=True)
        args.update(updates)
        return recompute_arrays(**args)

    def test_original_input_reference_and_small_execution_error(self):
        self.assertTrue(self.audit()['passed'])
        before=copy.deepcopy(self.model)
        self.audit()
        self.assertEqual(self.model,before)

    def test_reference_replaced_with_decrypted_answer_is_rejected(self):
        fake=self.reference+.5
        with self.assertRaisesRegex(ValueError,'independent'):
            self.audit(reference=fake,actual=fake,recorded=self.compare(fake,fake,1e-5,1e-4))

    def test_changed_array_cannot_keep_old_green_report(self):
        with self.assertRaisesRegex(ValueError,'actual arrays'):
            self.audit(actual=self.actual+.25)

    def test_real_negative_outcome_is_not_counted_as_a_correct_program(self):
        wrong=self.reference+.25
        recorded=self.compare(wrong,self.reference,1e-5,1e-4)
        self.assertFalse(self.audit(actual=wrong,recorded=recorded,expected_pass=False)['passed'])
        with self.assertRaisesRegex(ValueError,'Unexpected numerical outcome'):
            self.audit(actual=wrong,recorded=recorded)
        with self.assertRaisesRegex(ValueError,'explicit'):
            self.audit(expected_pass=1)

    def test_dtype_shape_nonfinite_and_changed_tolerance_fail_closed(self):
        invalid=[dict(inputs=self.inputs.astype('float32')),dict(inputs=self.inputs.reshape(2,8)),
                 dict(reference=self.reference[:,:2]),dict(actual=self.actual*float('nan')),
                 dict(recorded=self.compare(self.actual,self.reference,1.,1.))]
        for update in invalid:
            with self.subTest(update=list(update)),self.assertRaises(ValueError): self.audit(**update)

    def test_multi_input_names_shapes_and_order_are_independent_of_outputs(self):
        from model_graph import evaluate_reference
        model=dict(schema=3,id='user-dual',inputs=[dict(name='left',shape=[2,2]),dict(name='right',shape=[1,4])],
            constants={},nodes=[dict(id='l',op='flatten',inputs=['left']),dict(id='r',op='flatten',inputs=['right']),
                               dict(id='out',op='subtract',inputs=['l','r'])],output='out')
        inputs=self.np.stack([self.inputs,self.inputs[:,::-1]*.25],axis=1)
        ref=self.np.asarray([evaluate_reference(model,{'left':x[0].reshape(2,2).tolist(),
                                    'right':x[1].reshape(1,4).tolist()}) for x in inputs])
        recorded=self.compare(ref,ref,1e-5,1e-4)
        self.assertTrue(self.audit(model=model,inputs=inputs,reference=ref,actual=ref,recorded=recorded)['passed'])
        with self.assertRaisesRegex(ValueError,'independent'):
            self.audit(model=model,inputs=inputs[:,::-1,:],reference=ref,actual=ref,recorded=recorded)

    def test_approximation_summary_uses_actual_arrays_not_only_report_decrypted(self):
        import hashlib
        import io
        from approximation_errors import decompose,relu_and_quadratic
        from audit_manual_semantics import approximation_check
        f,p=relu_and_quadratic(self.inputs.reshape(-1).tolist())
        actual=self.np.asarray(p).reshape(4,4)+1e-9
        inputs_buffer,actual_buffer=io.BytesIO(),io.BytesIO()
        self.np.savez(inputs_buffer,inputs=self.inputs)
        self.np.save(actual_buffer,actual,allow_pickle=False)
        raw=actual_buffer.getvalue()
        row=dict(run='/synthetic/case',expected_pass=True,approximation_row=dict(inputs=self.inputs.tolist(),
            decrypted_sha256=hashlib.sha256(raw).hexdigest(),errors=decompose(f,p,actual.reshape(-1).tolist())))
        def read(path,root):
            value=inputs_buffer.getvalue() if path.name=='arrays.npz' else raw
            return value,hashlib.sha256(value).hexdigest()
        with patch('audit_manual_semantics.read',side_effect=read):
            result=approximation_check(row,Path('/synthetic'))
            self.assertEqual(result['approximation_max_error'],.125)
            self.assertLess(result['execution_max_error'],1e-8)
            self.assertFalse(result['original_semantic_equivalence_verified'])
            row['approximation_row']['errors']['decrypted'][0]+=1.
            with self.assertRaisesRegex(ValueError,'actual saved ciphertext outputs'):
                approximation_check(row,Path('/synthetic'))


class AcceptanceMatrixTests(unittest.TestCase):
    def test_wrong_program_presence_does_not_establish_positive_operator_coverage(self):
        from audit_manual_semantics import operator_matrix
        cases=[dict(status='verified',expected_numerical_pass=True,evidence='positive',model_semantics={'operators':{'add':1}}),
               dict(status='verified',expected_numerical_pass=False,evidence='negative',model_semantics={'operators':{'power':1}}),
               dict(status='failed',expected_numerical_pass=True,evidence='missing')]
        matrix={r['operator']:r for r in operator_matrix(cases)}
        self.assertEqual(matrix['add']['positive_evidence'],['positive'])
        self.assertEqual(matrix['power']['positive_evidence'],[])
        self.assertEqual(matrix['power']['counterexample_programs_involving_operator'],['negative'])
        self.assertFalse(matrix['power']['fault_localization_proved_by_operator_presence'])
        self.assertFalse(matrix['add']['tests_executed_by_this_audit'])

    def test_manifest_counts_and_explicit_negative_flags_cannot_be_silently_changed(self):
        from audit_manual_semantics import collect,COHORTS
        root=Path('/synthetic-results')
        def metadata(path,scope):
            label,_,count=next(c for c in COHORTS if c[1]==path.parent.name)
            return dict(status='passed',agent_calls=0,poseidon_gpu_validated=False,cases=[
                dict(counterexample=False,matched_expected=True,run=str(root/(label+'-'+str(i))),
                     command=['--golden-file','/synthetic.py']) for i in range(count)]),'a'*64
        with patch('audit_manual_semantics.metadata',side_effect=metadata):
            rows,_=collect(root)
            self.assertEqual(len(rows),61)
        def bad(path,scope):
            value,digest=metadata(path,scope)
            value['cases'].pop()
            return value,digest
        with patch('audit_manual_semantics.metadata',side_effect=bad),self.assertRaisesRegex(ValueError,'count'):
            collect(root)


@unittest.skipUnless(os.environ.get('POSEIDON_MANUAL_SEMANTICS_REPORT'),'requires saved real manual audit')
class ManualAuditEvidenceTests(unittest.TestCase):
    def test_new_contract_cannot_lose_required_key_file_evidence(self):
        from audit_agent_lineage import metadata,RESULTS
        from manual_semantic_evidence import audit_manual_case
        summary,_=metadata(Path(os.environ['POSEIDON_MANUAL_SEMANTICS_REPORT']),RESULTS)
        row=next(c for c in summary['cases'] if c['cohort']=='rotation' and c['expected_numerical_pass'])
        def modified(path,root):
            data,digest=metadata(path,root)
            if path.name=='report.json':
                data['attempts'][0]['execution'].pop('rotation_key_check',None)
            elif path.name=='execution.json':
                data.pop('rotation_key_check',None)
            return data,digest
        with patch('manual_semantic_evidence.metadata',side_effect=modified), \
             self.assertRaisesRegex(ValueError,'Missing required rotation-key evidence'):
            audit_manual_case(Path(row['evidence']),True)

    def test_all_recorded_manual_arrays_and_artifacts_recompute(self):
        from audit_agent_lineage import metadata,read,RESULTS
        from audit_manual_semantics import audit_all,BASE
        report,_=metadata(Path(os.environ['POSEIDON_MANUAL_SEMANTICS_REPORT']),RESULTS)
        self.assertEqual((report['status'],report['planned'],report['verified']),('passed',61,61))
        self.assertEqual((report['correct_goldens'],report['detected_counterexamples']),(42,19))
        self.assertEqual((report['agent_calls'],report['new_fhe_executions']),(0,0))
        self.assertEqual(report['legacy_key_probe_not_recorded'],3)
        self.assertFalse(report['all_goal_requirements_complete'])
        for name,digest in report['analysis_source_hashes'].items():
            self.assertEqual(read(BASE/name,BASE)[1],digest)
        actual=audit_all()
        for key in ('cases','sources','operator_matrix','compared_values'):
            self.assertEqual(report[key],actual[key])


if __name__=='__main__':
    unittest.main()
