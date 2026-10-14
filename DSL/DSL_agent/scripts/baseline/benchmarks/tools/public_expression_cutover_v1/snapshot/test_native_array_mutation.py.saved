"""Alias-aware native heap, atomic type updates, actual frontend and API boundary."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import native_array_alias as heap
from decorated_functions import validate,register
from native_function_core_fixtures import declaration
from run_native_array_mutation_goldens import CASES,LAYOUT_CASES
from test_native_array_core import CONSTANTS,request as array_request


def source(name):
    return (Path(__file__).parent/'golden_cases/native_array_mutation'/(name+'.py')).read_text()


class NativeArrayMutationTests(unittest.TestCase):
    def test_alias_view_copy_and_atomic_cell_kind_updates(self):
        def call(text):return ast.parse('a.'+text,mode='eval').body
        original=heap.fresh(('array',('p','p','p','p'),(2,2)))
        transposed=heap.result_type(original,heap.grid(original).T)
        view=heap.subscript_type(original,ast.parse('a[:,0]',mode='eval').body.slice)
        copied=heap.method_type(transposed,call('reshape(4)'))
        flat=heap.method_type(original,call('flatten()'))
        heap.inplace_type(view,'c')
        self.assertEqual(original[1],('c','p','c','p'))
        self.assertEqual(transposed[1],('c','c','p','p'))
        self.assertEqual(copied[1],('p',)*4);self.assertEqual(flat[1],('p',)*4)
        before=heap.freeze(original)
        with self.assertRaises(ValueError):heap.inplace_type(original,'p')
        self.assertEqual(heap.freeze(original),before) # no partial write after rejected p/p cell
        heap.inplace_type(original,'c');self.assertEqual(transposed[1],('c',)*4)

    def test_zero_dim_keeps_array_and_inplace_cannot_expand_shape(self):
        value=heap.fresh(('array',('p',),()))
        self.assertIs(heap.inplace_type(value,'c'),value)
        self.assertTrue(heap.is_array(value));self.assertEqual(value[2],())
        left=heap.fresh(('array',('c','c'),(2,1)))
        before=heap.freeze(left)
        with self.assertRaisesRegex(ValueError,'target shape'):
            heap.inplace_type(left,heap.fresh(('array',('p','p'),(1,2))))
        self.assertEqual(heap.freeze(left),before)
        empty=heap.fresh(('array',(),(0,2)))
        self.assertIs(heap.inplace_type(empty,heap.fresh(('array',('p','p'),(2,)))),empty)

    def test_constructor_and_ufunc_keep_order_affects_reshape_aliasing(self):
        def call(text):return ast.parse('a.'+text,mode='eval').body
        original=heap.fresh(('array',('p',)*4,(2,2)))
        transposed=heap.result_type(original,heap.grid(original).T)
        copied=heap.array_type(transposed)
        self.assertTrue(copied.indices.flags.f_contiguous)
        flattened=heap.method_type(copied,call('reshape(4)'))
        heap.inplace_type(flattened,'c')
        self.assertEqual(copied[1],('p',)*4)
        product=heap.arithmetic_type(transposed,'c')
        self.assertTrue(product.indices.flags.f_contiguous)
        flattened=heap.method_type(product,call('reshape(4)'))
        self.assertIsNot(flattened.kinds,product.kinds)

    def test_fixtures_and_independent_call_results_typecheck(self):
        for name in CASES+LAYOUT_CASES:
            with self.subTest(name=name):
                plan=validate(source(name),CONSTANTS,array_mutation=True)
                self.assertEqual(plan['contract'],'decorated-functions-core-v7')
                self.assertTrue(plan['array_mutation']['sites'])
                self.assertFalse(plan['array_mutation']['subscript_writes_enabled'])
                self.assertFalse(plan['encrypted_correctness_checked'] or plan['agent_generation_validated'])
                with self.assertRaises(ValueError):validate(source(name),CONSTANTS,scalar_augmented=True)
        sites=validate(source('callee_fresh'),CONSTANTS,array_mutation=True)['array_mutation']['sites']
        kinds={r['name']:r['kinds'] for r in sites[0]['after']}
        self.assertEqual(kinds,dict(first=['c'],second=['p']))

    def test_bad_operands_writes_and_old_contract_mix_rejected(self):
        bodies=[
          'items=np.array([c0],dtype=object)\nitems+=c1\nreturn x',
          'items=np.array([x,c0],dtype=object)\nitems*=np.array([x,c1],dtype=object)\nreturn x',
          'items=np.array([[x],[x]],dtype=object)\nitems+=np.array([x,x],dtype=object)\nreturn x',
          'items=np.array([x],dtype=object)\nitems[0]+=x\nreturn x',
          'items=np.array([x],dtype=object)\nitems.T+=x\nreturn x',
          'items=np.array([x],dtype=object)\nitems/=c0\nreturn x',
          'items=[x]\nitems+=x\nreturn x',
          'items=np.array([x],dtype=object)\nx+=items\nreturn x',
          'for i in range(0):\n    x[0]+=x\nreturn x',
        ]
        class Untouched:
            def __getattr__(self,name):raise AssertionError('frontend touched before validation')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError):
                register(declaration('golden','x',body),CONSTANTS,Untouched(),array_mutation=True)

    def test_request_provider_cli_and_fresh_batch_contract(self):
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        from native_function_rules import MUTATION_TASK,MUTATION_CONTRACT
        from run_candidate import parse_args,forward_options
        from run_agent_batch import construction_options,validate_construction_continuation,main
        old=array_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
        req=make_request(payload,old['model'],old['compiler_profile_sha256'],native_array_mutation=True)
        self.assertEqual(req['task'],MUTATION_TASK);self.assertEqual(public_request(req),req)
        checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source('callee_fresh')),req)
        self.assertEqual(checked['contract'],MUTATION_CONTRACT)
        for exercise in ('nf-scalar','na-reverse','ns-list'):
            with self.assertRaises(ValueError):make_request(payload,old['model'],old['compiler_profile_sha256'],
                native_array_mutation=True,construction_exercise=exercise)
        args=parse_args(['--case','cases/arithmetic-alias-chain.json','--prepare','--native-array-mutation'])
        self.assertIn('--native-array-mutation',forward_options(args))
        self.assertEqual(construction_options(args),['--native-array-mutation'])
        with self.assertRaisesRegex(ValueError,'fresh batch'):validate_construction_continuation({},args)
        with self.assertRaisesRegex(ValueError,'fresh batch'):
            validate_construction_continuation({'native_array_mutation':True},SimpleNamespace())
        output=io.StringIO()
        with (patch('sys.argv',['run_agent_batch.py','--plan','--native-array-mutation']),
              patch('agent_credentials.load_api_key',side_effect=AssertionError('credentials read')),
              contextlib.redirect_stdout(output)):
            self.assertEqual(main(),0)
        plan=json.loads(output.getvalue())
        self.assertTrue(plan['native_array_mutation'] and plan['native_scalar_augmented']);self.assertEqual(plan['agent_calls'],0)
        for flag in ('--native-function-exercises','--native-array-exercises','--native-star-exercises','--object-unary'):
            with (patch('sys.argv',['run_agent_batch.py','--plan','--native-array-mutation',flag]),
                  contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)):main()

    def test_alias_observation_budget_is_bounded_before_frontend(self):
        body='\n'.join('a'+str(i)+'=np.array([x],dtype=object)' for i in range(20))
        body+='\nfor i in range(100):\n    a0+=x\nreturn a0[0]'
        with self.assertRaisesRegex(ValueError,'observation resource bound'):
            validate(declaration('golden','x',body),{},array_mutation=True)


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_LIVE')=='1','requires real isolated frontend')
class NativeArrayMutationFrontendTests(unittest.TestCase):
    def setUp(self):
        from test_native_function_calls import NativeFunctionTests
        NativeFunctionTests.setUp(self)
    def tearDown(self):self.hc.lt.finalize(self.hc.ctxt)

    def test_predicted_alias_graph_matches_actual_frontend_for_all_fixtures(self):
        for index,name in enumerate(CASES+LAYOUT_CASES):
            if index:
                self.hc.lt.finalize(self.hc.ctxt);self.setUp()
            events=[]
            funcs,plan=register(source(name),CONSTANTS,self.hc,array_mutation=True,observe_mutation=events.append)
            funcs['golden'].eval()
            self.assertCountEqual(events,heap.trace_projection(plan['array_mutation']['sites']),name)
            self.assertEqual(funcs['golden'].outputlen,1)


if __name__=='__main__':unittest.main()
