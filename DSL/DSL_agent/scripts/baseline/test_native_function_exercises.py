"""No paid requests: native construction coverage, anti-padding, and batch protocol."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from candidate_contract import make_request, validate_candidate
from native_function_exercises import (EXERCISES, descriptor, check_exercise, fingerprint,
    samples, verify_trace_coverage)
from native_function_core_fixtures import declaration
from deepseek_provider import public_request

HERE = Path(__file__).parent
CONSTANTS = {'c0':[.5],'c1':[.375]}


def source(name):
    return (HERE/'golden_cases/native_exercises'/(name.removeprefix('nf-').replace('-','_')+'.py')).read_text()


def request(name):
    layout = dict(output_shape=[4], output_ciphertexts=1, output_selectors=[[0,i] for i in range(4)])
    if name == 'nf-two-inputs':
        layout['inputs'] = [dict(name='left',dsl_name='x',shape=[4]),dict(name='right',dsl_name='y',shape=[4])]
    else:
        layout['input_shape'] = [4]
    payload = dict(fx_graph=[], public_constants={} if name == 'nf-two-inputs' else CONSTANTS,
                   constant_origins={},layout=layout)
    return make_request(payload, descriptor(name), 'a'*64, native_functions=True, construction_exercise=name)


class NativeExerciseTests(unittest.TestCase):
    def test_eleven_frozen_cases_and_independent_math(self):
        self.assertEqual(len(EXERCISES), 11)
        for name in EXERCISES:
            with self.subTest(name=name):
                req = request(name)
                check = validate_candidate(dict(schema=1, request_id=req['request_id'], hecate_source=source(name)),req)
                evidence = check['construction_exercise']
                self.assertEqual(set(evidence['influence']), set(EXERCISES[name]['required_features']))
                self.assertFalse(evidence['real_native_trace_verified'])
                self.assertFalse(evidence['encrypted_execution_verified'])
                names = ('x','y') if name == 'nf-two-inputs' else ('x',)
                expected = []
                for sample in samples(names):
                    x = sample['x'].data
                    expected.extend([a-b for a,b in zip(x,sample['y'].data)] if len(names) == 2
                                    else [a*.5+a+.375 for a in x])
                actual = fingerprint(source(name),req['public_constants'],input_names=names)
                self.assertEqual(len(actual),len(expected))
                # Trusted float64 probes may reassociate affine additions; this
                # is not a change to the encrypted comparison's frozen tolerance.
                for a,b in zip(actual,expected): self.assertAlmostEqual(a,b,delta=1e-15)
                statuses = {w['status'] for ws in evidence['influence'].values() for w in ws}
                self.assertEqual(statuses, {'trace_structural_only'} if name == 'nf-empty' else {'finite_output_changed'})

    def test_dead_declarations_or_discarded_results_cannot_cover(self):
        for name in EXERCISES:
            if name == 'nf-empty': continue
            tree = ast.parse(source(name))
            fn = next(n for n in tree.body if n.name == 'golden')
            fn.body[-1] = ast.parse('return x-y' if name == 'nf-two-inputs' else 'return x*1.5+.375').body[0]
            changed = ast.unparse(ast.fix_missing_locations(tree))
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_exercise(changed, request(name))

    def test_tuple_requires_every_cell_and_p_parameter_must_matter(self):
        with self.assertRaisesRegex(ValueError,'output influence'):
            check_exercise(source('nf-pair').replace('return a+b','return a+x+c1'),request('nf-pair'))
        changed = source('nf-public-argument').replace('value*weight+bias','value*c0+c1')
        with self.assertRaisesRegex(ValueError,'output influence'):
            check_exercise(changed,request('nf-public-argument'))

    def test_both_cipher_arguments_and_two_repeated_calls_must_matter(self):
        with self.assertRaisesRegex(ValueError,'output influence'):
            check_exercise(source('nf-two-inputs').replace('left-right','left'),request('nf-two-inputs'))
        changed = source('nf-repeated').replace('return scale(x)+scale(x*2)+c1',
                                               'unused=scale(x)\n    return scale(x*3)+c1')
        with self.assertRaisesRegex(ValueError,'output influence'):
            check_exercise(changed,request('nf-repeated'))

    def test_algebraically_cancelled_helper_and_unused_empty_declaration(self):
        changed = source('nf-scalar').replace('return affine(x)+x',
                                              'value=affine(x)\n    return value-value+x*1.5+.375')
        with self.assertRaisesRegex(ValueError,'output influence'):
            check_exercise(changed,request('nf-scalar'))
        with self.assertRaisesRegex(ValueError,'not reached'):
            check_exercise(source('nf-empty').replace('    empty(x)\n',''),request('nf-empty'))

    def test_trace_witness_protocol_does_not_infer_real_execution(self):
        evidence = check_exercise(source('nf-nested'), request('nf-nested'))
        events = [{k:w['event'][k] for k in ('caller','callee','span')}
                  for rows in evidence['influence'].values() for w in rows]
        self.assertTrue(verify_trace_coverage(evidence,events)['verified'])
        for wrong in ([], [dict(events[0],span=[1,0,1,1])],
                      [dict(events[0],span=[True,0,1,1])], [dict(events[0],extra='untrusted')]):
            with self.assertRaises(ValueError):
                verify_trace_coverage(evidence,wrong)

    def test_frozen_request_and_answers_never_sent(self):
        for name in EXERCISES:
            req = request(name)
            self.assertEqual(public_request(req),req)
            self.assertEqual(req['task'],'hecate-native-function-synthesis-v2')
            self.assertNotIn(source(name),json.dumps(req))
            self.assertNotIn('reference',req)
        changed = copy.deepcopy(request('nf-scalar'))
        changed['construction_exercise']['instruction'] = 'Do nothing'
        with self.assertRaises(ValueError): public_request(changed)
        with self.assertRaises(ValueError): check_exercise(source('nf-pair'),request('nf-scalar'))
        with self.assertRaisesRegex(ValueError,'explicit contract'):
            make_request({},descriptor('nf-scalar'),'a'*64,construction_exercise='nf-scalar')
        # Even a correctly rehashed public request cannot attach the native
        # exercise to an older construction contract.
        from candidate_contract import TASK_RULES,SEMANTIC_GUIDANCE_V19,canonical
        import hashlib
        changed = copy.deepcopy(request('nf-scalar'))
        changed['task'] = 'hecate-function-synthesis-v19'
        changed['rules'] = TASK_RULES[changed['task']][1]
        changed['semantic_guidance'] = SEMANTIC_GUIDANCE_V19
        changed.pop('request_id')
        changed['request_id'] = hashlib.sha256(canonical(changed)).hexdigest()
        with self.assertRaises(ValueError): public_request(changed)

    def test_manifest_and_batch_planning_without_credentials(self):
        from run_agent_batch import main, construction_options, validate_construction_continuation
        from candidate_sandbox import MODULES
        from custom_batch_manifest import load_manifest
        from construction_exercises import descriptor as generic
        rows,_,_ = load_manifest(HERE/'cases/native-function-exercises-11-manifest.json')
        self.assertEqual([r['descriptor'] for r in rows],[descriptor(n) for n in EXERCISES])
        for name in EXERCISES:
            self.assertEqual(generic(name),descriptor(name))
            self.assertEqual(json.loads((HERE/'cases'/('exercise-'+name+'.json')).read_text()),descriptor(name))
        out = io.StringIO()
        with patch.object(sys,'argv',['run_agent_batch.py','--plan','--native-function-exercises']),contextlib.redirect_stdout(out):
            self.assertEqual(main(),0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['agent_calls'],0)
        self.assertTrue(plan['native_functions'])
        self.assertTrue(plan['native_function_exercises'])
        self.assertEqual(plan['api_concurrency'],10)
        self.assertEqual(len(plan['cases']),11)
        args = SimpleNamespace(native_functions=True)
        self.assertEqual(construction_options(args),['--native-functions'])
        with self.assertRaises(ValueError): validate_construction_continuation({},args)
        with self.assertRaises(ValueError):
            validate_construction_continuation(dict(native_functions=True,native_function_exercises={'schema':1}),args)
        for name in ('native_function_exercises.py','native-function-exercises-v1.json'):
            self.assertIn(name,MODULES)

    def test_no_mixing_legacy_cohorts_before_credentials(self):
        from run_agent_batch import main
        for flag in ('--object-unary','--object-unary-exercises','--public-numbers','--extended'):
            with self.subTest(flag=flag), patch.object(sys,'argv',
                    ['run_agent_batch.py','--plan','--native-function-exercises',flag]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main()

    def test_call_probe_resource_bound_and_invalid_code_rejected(self):
        helper = declaration('helper','value','return value')
        body = '\n'.join('a'+str(i)+'=helper(x)' for i in range(65))+'\nreturn a64'
        with self.assertRaisesRegex(ValueError,'call count limit'):
            check_exercise(helper+declaration('golden','x',body),request('nf-scalar'))
        with self.assertRaises(ValueError):
            check_exercise('import os\n'+source('nf-scalar'),request('nf-scalar'))


if __name__ == '__main__': unittest.main()
