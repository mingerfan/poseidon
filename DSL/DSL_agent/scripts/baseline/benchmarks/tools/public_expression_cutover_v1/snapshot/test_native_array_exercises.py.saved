"""No API: finite array influence, dead-code rejection and version boundaries."""
import ast
import copy
import hashlib
import unittest

from candidate_contract import make_request, validate_candidate, canonical
from native_array_exercises import (EXERCISES, TASK, descriptor, check_exercise,
    verify_trace_coverage, run_probe, samples, _analyze)
from native_function_core_fixtures import declaration
from test_native_array_core import request as old_request, source, CONSTANTS


def request(name):
    old = old_request()
    payload = {k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
    return make_request(payload, descriptor(name), old['compiler_profile_sha256'],
                        native_arrays=True, construction_exercise=name)


def candidate(text, req):
    return dict(schema=1,request_id=req['request_id'],hecate_source=text)


class NativeArrayExerciseTests(unittest.TestCase):
    def test_all_ten_goldens_have_finite_influence_or_explicit_empty_structure(self):
        for name,(fixture,required,_) in EXERCISES.items():
            with self.subTest(name=name):
                req = request(name)
                self.assertEqual(req['task'],TASK)
                checked = validate_candidate(candidate(source(fixture),req),req)
                result = checked['construction_exercise']
                self.assertEqual(result['required'],required)
                self.assertTrue(all(result['influence'].values()))
                self.assertFalse(result['encrypted_execution_verified'])
                self.assertFalse(result['real_native_trace_verified'])
                kinds = {w['status'] for ws in result['influence'].values() for w in ws}
                self.assertEqual(kinds, {'trace_structural_only'} if name == 'na-empty' else {'finite_cell_result_influence'})

    def test_probe_preserves_manual_math_and_storage_not_slot_semantics(self):
        for name,(fixture,_,_) in EXERCISES.items():
            nodes,plan = _analyze(source(fixture),CONSTANTS,1,('x',),arrays=True)
            for inputs in samples(('x',)):
                values,_ = run_probe(nodes,plan,CONSTANTS,inputs,[0])
                for actual,x in zip(values,inputs['x'].data):
                    self.assertAlmostEqual(actual,1.5*x+.375,delta=1e-15)

    def test_unused_helper_and_discarded_return_do_not_count(self):
        for name,(fixture,_,_) in EXERCISES.items():
            tree = ast.parse(source(fixture))
            golden = next(n for n in tree.body if n.name=='golden')
            golden.body = ast.parse('return x*c0+x+c1').body
            with self.subTest(name=name), self.assertRaisesRegex(ValueError,'no output influence'):
                check_exercise(ast.unparse(tree),request(name))
        text = declaration('helper','v','return np.array([[v,v]],dtype=object)')+declaration(
            'golden','x','unused=helper(x)\nreturn x*c0+x+c1')
        with self.assertRaisesRegex(ValueError,'no output influence'): check_exercise(text,request('na-transpose'))

    def test_cancelled_array_result_and_unused_mixed_plain_are_rejected(self):
        text = declaration('helper','v','return np.array([[v,v]],dtype=object)')+declaration(
            'golden','x','a=helper(x).T\nb=a[0,0]\nreturn x*c0+x+c1+b-b')
        with self.assertRaisesRegex(ValueError,'no output influence'): check_exercise(text,request('na-transpose'))
        text = source('mixed_plain').replace('r[0,0]*r[0,1]+r[1,1]','r[0,0]*c0+r[1,1]')
        with self.assertRaisesRegex(ValueError,'no output influence'): check_exercise(text,request('na-mixed'))

    def test_trace_witness_requires_same_operation_shape_and_site(self):
        result = check_exercise(source('rank_four'),request('na-rank-four'))
        observed = [w['event']['trace'] for ws in result['influence'].values() for w in ws]
        self.assertTrue(verify_trace_coverage(result,observed)['verified'])
        for field in ('shape','span','op','caller'):
            changed = copy.deepcopy(observed)
            for e in changed:
                e[field] = [0] if field=='shape' else [0]*4 if field=='span' else 'missing'
            with self.assertRaises(ValueError): verify_trace_coverage(result,changed)
        bad = copy.deepcopy(observed); bad[0]['span'][0]=True
        with self.assertRaises(ValueError): verify_trace_coverage(result,bad)

    def test_versioned_provider_gate_rejects_relabelled_old_request(self):
        from deepseek_provider import public_request
        from native_function_rules import ARRAY_RULES,ARRAY_TASK
        req = request('na-zero-item')
        self.assertEqual(public_request(req),req)
        wrong = copy.deepcopy(req);wrong['task']=ARRAY_TASK;wrong['rules']=ARRAY_RULES
        body={k:v for k,v in wrong.items() if k!='request_id'}
        wrong['request_id']=hashlib.sha256(canonical(body)).hexdigest()
        with self.assertRaises(ValueError): public_request(wrong)
        old=old_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
        for flags in ({},{'native_functions':True}):
            with self.assertRaises(ValueError):
                make_request(payload,descriptor('na-zero-item'),old['compiler_profile_sha256'],
                             construction_exercise='na-zero-item',**flags)
        with self.assertRaises(ValueError):
            make_request(payload,descriptor('na-zero-item'),old['compiler_profile_sha256'],
                         native_arrays=True,construction_exercise='nf-scalar')

    def test_batch_plan_preserves_frozen_requirements_without_api(self):
        import contextlib,io,json,sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from run_agent_batch import main,validate_construction_continuation
        from custom_batch_manifest import load_manifest
        manifest=Path(__file__).with_name('cases')/'native-array-exercises-10-manifest.json'
        rows,_,_=load_manifest(manifest)
        self.assertEqual([r['descriptor'] for r in rows],[descriptor(n) for n in EXERCISES])
        for name in EXERCISES:
            self.assertEqual(json.loads((manifest.parent/('exercise-'+name+'.json')).read_text()),descriptor(name))
        out=io.StringIO()
        with patch.object(sys,'argv',['run_agent_batch.py','--plan','--native-array-exercises']),contextlib.redirect_stdout(out):
            self.assertEqual(main(),0)
        plan=json.loads(out.getvalue())
        self.assertTrue(plan['native_array_exercises'] and plan['native_arrays'] and plan['native_functions'])
        self.assertEqual(plan['agent_calls'],0)
        self.assertEqual(len(plan['cases']),10)
        for flag in ('--native-function-exercises','--object-unary','--extended'):
            with patch.object(sys,'argv',['run_agent_batch.py','--plan','--native-array-exercises',flag]), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main()
        with self.assertRaises(ValueError):
            validate_construction_continuation({'native_arrays':True,'native_array_exercises':{}},
                                              SimpleNamespace(native_arrays=False))
        with self.assertRaisesRegex(ValueError,'requirements cannot be dropped'):
            validate_construction_continuation({'native_array_exercises':{'schema':1}},SimpleNamespace())

    def test_coverage_does_not_replace_numerical_equivalence(self):
        req=request('na-transpose')
        checked=validate_candidate(candidate(source('wrong_transpose'),req),req)
        self.assertTrue(checked['construction_exercise']['influence'])
        self.assertFalse(checked['construction_exercise']['encrypted_execution_verified'])


if __name__ == '__main__': unittest.main()
