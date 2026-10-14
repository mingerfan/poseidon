"""Native array shape/ABI safety tests. No model API or encrypted execution."""
import ast
from pathlib import Path
import unittest

from candidate_contract import make_request,validate_candidate
from decorated_functions import validate,register
from native_array_core import array_type,subscript_type,method_type
from native_function_core_fixtures import declaration
from native_function_rules import ARRAY_TASK
from run_native_array_goldens import CASES
from test_candidate_pipeline import request_fixture

HERE = Path(__file__).parent
CONSTANTS = {'c0':[.5],'c1':[.375]}


def source(name):
    return (HERE/'golden_cases/native_arrays'/(name+'.py')).read_text()


def request():
    old = request_fixture()
    payload = {k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
    payload['public_constants'] = CONSTANTS
    return make_request(payload,old['model'],old['compiler_profile_sha256'],native_arrays=True)


class NativeArrayTests(unittest.TestCase):
    def test_all_fixtures_typecheck_without_correctness_claim(self):
        for name in CASES:
            with self.subTest(name=name):
                plan = validate(source(name),CONSTANTS,arrays=True)
                self.assertEqual(plan['contract'],'decorated-functions-core-v2')
                self.assertEqual(plan['output_ciphertexts'],1)
                self.assertFalse(plan['encrypted_correctness_checked'])
                self.assertFalse(plan['agent_generation_validated'])
                # The wrong-order program is well-typed. Only actual numerical
                # validation may reject it as semantically different.
                with self.assertRaises(ValueError):
                    validate(source(name),CONSTANTS)

    def test_zero_dimension_and_shape_preserving_calls(self):
        p = validate(source('zero_return'),CONSTANTS,arrays=True)
        self.assertEqual(p['functions']['helper']['result'],('array',('c',),()))
        self.assertEqual(p['functions']['golden']['result'],('array',('c',),()))
        p = validate(source('rank_four'),CONSTANTS,arrays=True)
        self.assertEqual(p['functions']['helper']['result'][2],(2,2,1,1))
        p = validate(source('mixed_plain'),CONSTANTS,arrays=True)
        self.assertEqual(p['functions']['helper']['result'],('array',('c','p','c','c'),(2,2)))

    def test_type_permutations_follow_numpy_storage_not_slots(self):
        t = ('array',('c','p','c','p'),(2,2))
        def idx(text): return ast.parse('a['+text+']',mode='eval').body.slice
        def call(text): return ast.parse('a.'+text,mode='eval').body
        self.assertEqual(subscript_type(t,idx('1,0')),'c')
        self.assertEqual(subscript_type(t,idx('...,::-1')),('array',('p','c','p','c'),(2,2)))
        self.assertEqual(method_type(t,call('transpose()')),('array',('c','c','p','p'),(2,2)))
        self.assertEqual(method_type(t,call('item((0,1))')),'p')
        self.assertEqual(method_type(t,call('item(1)')),'p')
        self.assertEqual(method_type(t,call('reshape(1,-1)'))[2],(1,4))
        self.assertEqual(subscript_type(array_type('c'),idx('()')),'c')

    def test_array_unpacking_is_first_axis_and_zero_dim_not_iterable(self):
        valid = declaration('helper','x','return np.array([[x,x+.25],[x*.5,x*.75]],dtype=object)')+declaration(
            'golden','x','first,second=helper(x)\nreturn first')
        self.assertEqual(validate(valid,{},2,arrays=True)['functions']['golden']['result'],
                         ('array',('c','c'),(2,)))
        for src in (valid.replace('first,second=','a,b,c,d=').replace('return first','return a'),
                    source('zero_item').replace('return helper(x).item()','value,=helper(x)\n    return value')):
            with self.assertRaises(ValueError): validate(src,{},arrays=True)

    def test_ragged_size_rank_and_return_structure_rejected(self):
        for expr in ('np.array([[x],[x,x]],dtype=object)',
                     'np.array([[[[[x]]]]],dtype=object)',
                     'np.array(['+','.join(['x']*17)+'],dtype=object)',
                     'np.array([x,x],dtype=object).reshape(3)',
                     'np.array([x],dtype=object).reshape(-1,-1)',
                     'np.array(x,dtype=object).reshape()',
                     'np.array([x,x],dtype=object).item()',
                     'np.array([x],dtype=object).transpose((0,0))',
                     '[[x]]','np.array([.5],dtype=object)'):
            with self.subTest(expr=expr),self.assertRaises(ValueError):
                validate(declaration('golden','x','return '+expr),{},arrays=True)

    def test_no_dynamic_index_io_arbitrary_methods_or_array_arithmetic(self):
        base = 'np.array([x,x+.25],dtype=object)'
        for tail in ('[x]','[True]','[[0]]','[::0]','[999]','.item(x)','.item(True)',
                     '.reshape(x)','.reshape(2,order="F")','.flatten("F")',
                     '.view()','.__class__','.tofile("secret")','*2','+x'):
            with self.subTest(tail=tail),self.assertRaises(ValueError):
                validate(declaration('golden','x','return '+base+tail),{},arrays=True)
        for expr in ('np.asarray(x,dtype=object)','np.array(x)','np.array(x,dtype=float)',
                     'np.array(x,dtype=object,copy=False)','np.array(x,dtype=object).item(eval("0"))'):
            with self.assertRaises(ValueError):
                validate(declaration('golden','x','return '+expr),{},arrays=True)

    def test_reserved_names_array_args_and_writes_rejected_before_native(self):
        class Untouched:
            def __getattribute__(self,name): raise AssertionError('Native frontend accessed')
        for body in ('np=x\nreturn x','object=x\nreturn x',
                     'a=np.array([x],dtype=object)\na[0]=x+x\nreturn a[0]'):
            with self.assertRaises(ValueError): register(declaration('golden','x',body),{},Untouched(),arrays=True)
        src = declaration('helper','x','return x')+declaration(
            'golden','x','return helper(np.array(x,dtype=object))')
        with self.assertRaises(ValueError): register(src,{},Untouched(),arrays=True)
        for constants in ({'np':[1.]},{'object':[1.]}):
            with self.assertRaises(ValueError): validate(declaration('golden','x','return x'),constants,arrays=True)
        # Old names were not reserved by the array-free v1 contract.
        self.assertEqual(validate(declaration('golden','x','return x+np'),{'np':[1.]})['output_ciphertexts'],1)

    def test_versioned_request_cli_and_no_old_exercise_reinterpretation(self):
        from deepseek_provider import public_request
        from run_candidate import parse_args,forward_options
        req = request()
        self.assertEqual(req['task'],ARRAY_TASK)
        self.assertEqual(public_request(req),req)
        self.assertEqual(validate_candidate(dict(schema=1,request_id=req['request_id'],
                         hecate_source=source('zero_item')),req)['contract'],'hecate-native-functions-v2')
        args = parse_args(['--case','cases/arithmetic-alias-chain.json','--prepare','--native-arrays'])
        self.assertIn('--native-arrays',forward_options(args))
        with self.assertRaises(ValueError):
            make_request({}, {},'a'*64,native_arrays=True,construction_exercise='nf-scalar')
        with self.assertRaises(ValueError):
            make_request({}, {},'a'*64,native_arrays=True,object_unary=True)

    def test_batch_forwarding_and_no_prompt_contract_drift(self):
        from run_agent_batch import construction_options,validate_construction_continuation,main
        from types import SimpleNamespace
        from unittest.mock import patch
        import contextlib,io,sys
        args = SimpleNamespace(native_arrays=True,native_functions=True)
        self.assertEqual(construction_options(args),['--native-arrays'])
        with self.assertRaises(ValueError): validate_construction_continuation({'native_functions':True},args)
        validate_construction_continuation({'native_functions':True,'native_arrays':True},args)
        for flag in ('--native-function-exercises','--object-unary','--object-arrays'):
            with patch.object(sys,'argv',['run_agent_batch.py','--plan','--native-arrays',flag]), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main()


if __name__ == '__main__': unittest.main()
