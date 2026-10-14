"""Finite star-argument influence, real frontend witness protocol, frozen cohort."""
import ast
import copy
import json
import os
from pathlib import Path
import unittest

from candidate_contract import make_request,validate_candidate,canonical
from native_star_exercises import (EXERCISES,TASK,descriptor,check_exercise,verify_trace_coverage,
                                  run_probe,samples,_analyze)
from native_function_core_fixtures import declaration
from test_native_array_core import CONSTANTS,request as base_request
from test_native_starred import source


def request(name):
    old=base_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
    return make_request(payload,descriptor(name),old['compiler_profile_sha256'],
                        native_starred=True,construction_exercise=name)


class NativeStarExerciseTests(unittest.TestCase):
    def test_eight_goldens_have_all_argument_influence_or_explicit_empty_structure(self):
        for name,(fixture,required,_) in EXERCISES.items():
            with self.subTest(name=name):
                req=request(name);self.assertEqual(req['task'],TASK)
                checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source(fixture)),req)
                result=checked['construction_exercise'];self.assertEqual(result['required'],required)
                self.assertFalse(result['encrypted_execution_verified'] or result['real_native_trace_verified'])
                for witnesses in result['influence'].values():
                    self.assertTrue(witnesses)
                    for witness in witnesses:
                        self.assertEqual(witness['status'],'trace_structural_only' if name=='ns-empty' else 'finite_argument_influence')
                        if name!='ns-empty':self.assertEqual(witness['indices'],list(range(len(witness['event']['argument_kinds']))))

    def test_probe_math_matches_independent_affine_without_using_test_arrays(self):
        for fixture,_,_ in EXERCISES.values():
            nodes,plan=_analyze(source(fixture),CONSTANTS,1,('x',),starred_calls=True)
            for inputs in samples(('x',)):
                actual,_=run_probe(nodes,plan,CONSTANTS,inputs,[0])
                for value,x in zip(actual,inputs['x'].data):self.assertAlmostEqual(value,1.5*x+.375,delta=1e-15)

    def test_unused_declarations_and_ignored_argument_rejected(self):
        for name,(fixture,_,_) in EXERCISES.items():
            tree=ast.parse(source(fixture));golden=next(n for n in tree.body if n.name=='golden')
            golden.body=ast.parse('return x*c0+x+c1').body
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'argument influence'):
                check_exercise(ast.unparse(tree),request(name))
        # Numerically correct on the model, but its third positional argument is dead.
        ignored=source('list').replace('a*weight+b+bias','a*weight+a+bias')
        with self.assertRaisesRegex(ValueError,'argument influence'):check_exercise(ignored,request('ns-list'))

    def test_cancelled_call_and_fake_matrix_flatten_rejected(self):
        tree=ast.parse(source('list'));golden=next(n for n in tree.body if n.name=='golden')
        call=golden.body[-1].value
        golden.body[-1:]=[ast.Assign(targets=[ast.Name(id='unused',ctx=ast.Store())],value=call),
                          *ast.parse('return x*c0+x+c1+unused-unused').body]
        text=ast.unparse(ast.fix_missing_locations(tree))
        with self.assertRaisesRegex(ValueError,'argument influence'):check_exercise(text,request('ns-list'))
        text=source('matrix_flatten').replace('[[x,x],[c0,c1]]','[x,c0,x,c1]').replace('items.T.flatten()','items.flatten()')
        with self.assertRaisesRegex(ValueError,'argument influence'):check_exercise(text,request('ns-matrix-flatten'))

    def test_actual_call_shape_order_and_span_required(self):
        checked=check_exercise(source('multiple'),request('ns-multiple'))
        observed=[w['event']['trace'] for ws in checked['influence'].values() for w in ws]
        self.assertTrue(verify_trace_coverage(checked,observed)['verified'])
        for field in ('callee','span','segments'):
            changed=copy.deepcopy(observed)
            for r in changed:r[field]='different' if field=='callee' else [0]*4 if field=='span' else []
            with self.assertRaises(ValueError):verify_trace_coverage(checked,changed)
        changed=copy.deepcopy(observed);changed[0]['segments'][0]['count']=True
        with self.assertRaises(ValueError):verify_trace_coverage(checked,changed)

    def test_wrong_math_is_not_mislabelled_as_semantic_pass(self):
        text=(Path(__file__).parent/'golden_cases/native_starred_exercises/wrong_reverse.py').read_text()
        result=check_exercise(text,request('ns-reverse'))
        self.assertTrue(result['influence']);self.assertFalse(result['encrypted_execution_verified'])

    def test_frozen_prompt_provider_and_plan_never_call_api(self):
        import contextlib,io
        from types import SimpleNamespace
        from unittest.mock import patch
        from deepseek_provider import public_request
        from run_agent_batch import main,validate_construction_continuation
        from custom_batch_manifest import load_manifest
        req=request('ns-list');self.assertEqual(public_request(req),req)
        manifest=Path(__file__).with_name('cases')/'native-star-exercises-8-manifest.json'
        rows,_,_=load_manifest(manifest)
        self.assertEqual([r['descriptor'] for r in rows],[descriptor(n) for n in EXERCISES])
        for name in EXERCISES:
            self.assertEqual(json.loads((manifest.parent/('exercise-'+name+'.json')).read_text()),descriptor(name))
        output=io.StringIO()
        with (patch('sys.argv',['run_agent_batch.py','--plan','--native-star-exercises',
                               '--compiler-configuration','seal-cpu-eva-w45-v1']),
              patch('agent_credentials.load_api_key',side_effect=AssertionError('credential access')),
              contextlib.redirect_stdout(output)):
            self.assertEqual(main(),0)
        plan=json.loads(output.getvalue())
        self.assertEqual((len(plan['cases']),plan['agent_calls']),(8,0))
        self.assertTrue(plan['native_star_exercises'] and plan['native_starred'])
        with self.assertRaisesRegex(ValueError,'requirements cannot be dropped'):
            validate_construction_continuation({'native_star_exercises':{'schema':1}},SimpleNamespace())
        for extra in ('--native-array-exercises','--native-array-arithmetic','--native-public-loops','--extended'):
            with (patch('sys.argv',['run_agent_batch.py','--plan','--native-star-exercises',extra]),
                  contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)):main()
        old=base_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
        for flags in ({},{'native_arrays':True},{'native_array_arithmetic':True},{'native_public_loops':True}):
            with self.assertRaises(ValueError):
                make_request(payload,descriptor('ns-list'),old['compiler_profile_sha256'],construction_exercise='ns-list',**flags)


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_STAR_EXERCISE_GOLDENS'),'requires completed targeted star goldens')
class NativeStarExerciseEvidenceTests(unittest.TestCase):
    def test_manual_audit_cannot_be_relabelled_as_paid_agent(self):
        from audit_native_star_batch import audit
        root=Path(os.environ['POSEIDON_NATIVE_STAR_EXERCISE_GOLDENS'])
        result=audit(root,manual=True)
        self.assertEqual((result['status'],result['evidence_kind'],result['passed']),('covered','manual_golden',8))
        self.assertEqual((result['input_executions'],result['compared_values']),(32,128))
        self.assertEqual(len(result['feature_matrix']),9)
        self.assertEqual(sum(f['evidence_kind']=='trace_structural_only' for f in result['feature_matrix']),1)
        self.assertEqual((result['new_api_calls'],result['new_fhe_executions'],result['live_agent_cases']),(0,0,0))
        with self.assertRaisesRegex(ValueError,'terminal live starred'):audit(root)

    def test_real_trace_argument_expansion_and_independent_encrypted_comparison(self):
        import numpy as np
        from hecate_python_env import digest
        from seal_cpu_golden import compare
        from compiler_configuration import configuration
        from deepseek_provider import public_request
        root=Path(os.environ['POSEIDON_NATIVE_STAR_EXERCISE_GOLDENS'])
        batch=json.loads((root/'report.json').read_text())
        self.assertEqual((batch['status'],batch['generator'],batch['agent_calls']),('passed','manual_golden',0))
        self.assertEqual([r['case'] for r in batch['cases']],['exercise-'+n for n in EXERCISES]+['exercise-ns-reverse'])
        for index,row in enumerate(batch['cases']):
            run=Path(row['run']);r=json.loads((run/'report.json').read_text())
            self.assertTrue(row['matched_expected']);self.assertEqual(row['counterexample'],index==8)
            self.assertEqual((r['waterline'],r['agent_calls']),(45,0))
            self.assertEqual(r['compiler_configuration'],configuration('seal-cpu-eva-w45-v1'))
            self.assertFalse(r['llm_generation_validated'] or r['poseidon_gpu_validated'])
            self.assertEqual((r['parameters']['security_check'],r['parameters']['polynomial_degree']),('tc128',32768))
            self.assertEqual(r['parameters']['modulus_bits'],[60]*14)
            self.assertEqual(r['tolerance'],dict(atol=1e-5,rtol=1e-4))
            for file,h in r['frozen_hashes'].items():self.assertEqual(digest(run/file),h)
            request=json.loads((run/'request.json').read_text());self.assertEqual(request['task'],TASK)
            self.assertEqual(public_request(request),request)
            a=r['attempts'][0];out=run/'attempt-00/output'
            candidate=json.loads((run/'attempt-00/response.txt').read_text())
            self.assertEqual(canonical(validate_candidate(candidate,request)),canonical(a['static_check']))
            for file,h in a['artifact_hashes'].items():self.assertEqual(digest(out/file),h)
            observed=json.loads((out/'native-star-events.json').read_text())
            self.assertEqual(verify_trace_coverage(a['static_check']['construction_exercise'],observed),a['native_star_coverage'])
            self.assertTrue(a['compiled'] and a['executed'] and a['execution']['encrypted_execution'])
            self.assertFalse(a['trace']['candidate_python_executed'] or a['execution']['bootstrap_executed'])
            with np.load(run/'arrays.npz',allow_pickle=False) as data:inputs,reference=data['inputs'],data['reference']
            np.testing.assert_allclose(reference,inputs*1.5+.375,atol=1e-15,rtol=0)
            actual=np.load(out/'decrypted.npy',allow_pickle=False)
            comparison=compare(actual,reference,1e-5,1e-4);self.assertEqual(comparison,a['comparison'])
            self.assertEqual(comparison['passed'],index!=8)
            if index==8:
                self.assertEqual(a['failure_layer'],'numerical_comparison')
                self.assertGreater(comparison['max_absolute_error'],2.9)
            self.assertFalse((run/'private-keys').exists())
            self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
