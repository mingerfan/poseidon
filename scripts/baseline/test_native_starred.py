"""Native positional unpacking: typing, sandbox dispatch and actual frontend ABI."""
import os
from pathlib import Path
import unittest

from decorated_functions import validate,register
from native_function_core_fixtures import declaration
from run_native_starred_goldens import CASES
from test_native_array_core import CONSTANTS,request as array_request

HERE=Path(__file__).parent


def source(name):
    return (HERE/'golden_cases/native_starred'/(name+'.py')).read_text()


class NativeStarredTests(unittest.TestCase):
    def test_all_goldens_typed_and_old_array_contract_unchanged(self):
        for name in CASES:
            with self.subTest(name=name):
                p=validate(source(name),CONSTANTS,starred_calls=True)
                self.assertEqual(p['contract'],'decorated-functions-core-v3')
                self.assertEqual(p['output_ciphertexts'],1)
                self.assertFalse(p['encrypted_correctness_checked'])
                with self.assertRaises(ValueError): validate(source(name),CONSTANTS,arrays=True)

    def test_no_implicit_flatten_no_scalar_iteration_no_arity_or_kind_coercion(self):
        helper=declaration('helper','a,b','return a+b','c,c')
        for call in ('helper(*x)','helper(*np.array(x,dtype=object))',
                     'helper(*np.array([[x,x]],dtype=object))','helper(*[x,c0])',
                     'helper(*[])','helper(*[x,x,x])','helper(*[[x],[x]])',
                     'helper(*(v for v in [x,x]))','helper(**{})'):
            with self.subTest(call=call),self.assertRaises(ValueError):
                validate(helper+declaration('golden','x','return '+call),CONSTANTS,starred_calls=True)
        # Explicit first-axis expansion of (0,2) produces zero arguments,
        # consistent with Python; it is not a prohibition based only on rank.
        zero=declaration('helper','','return c0','')+declaration('golden','x',
            'empty=np.array([],dtype=object).reshape(0,2)\nreturn x*helper(*empty)+x+c1')
        self.assertEqual(validate(zero,CONSTANTS,starred_calls=True)['output_ciphertexts'],1)

    def test_expand_limit_and_no_new_star_syntax_elsewhere(self):
        helper=declaration('helper','a,b','return a+b','c,c')
        bad=[helper+declaration('golden','x','return helper(*['+','.join(['x']*16)+'],x)'),
             declaration('golden','x','return x.rotate(*[1])'),
             declaration('golden','x','a=np.array([x],dtype=object)\nreturn a.reshape(*[1])[0]'),
             declaration('golden','x','return [*np.array([x],dtype=object)]'),
             '@hc.func("c")\ndef golden(*x):\n    return x[0]\n']
        class Untouched:
            def __getattr__(self,name): raise AssertionError('Frontend reached before validation')
        for text in bad:
            with self.assertRaises(ValueError): register(text,CONSTANTS,Untouched(),starred_calls=True)

    def test_versioned_request_provider_and_cli(self):
        from candidate_contract import make_request,validate_candidate,request_input_names
        from deepseek_provider import public_request
        from native_function_rules import STAR_TASK
        from run_candidate import parse_args,forward_options
        old=array_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
        req=make_request(payload,old['model'],old['compiler_profile_sha256'],native_starred=True)
        self.assertEqual(req['task'],STAR_TASK)
        self.assertEqual(public_request(req),req)
        self.assertEqual(request_input_names(req),('x',))
        self.assertEqual(validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source('array')),req)
                         ['contract'],'hecate-native-functions-v3')
        for exercise in ('nf-scalar','na-reverse'):
            with self.assertRaises(ValueError):
                make_request(payload,old['model'],old['compiler_profile_sha256'],native_starred=True,construction_exercise=exercise)
        args=parse_args(['--case','cases/arithmetic-alias-chain.json','--prepare','--native-starred'])
        self.assertIn('--native-starred',forward_options(args))

    def test_batch_forwarding_and_frozen_exercise_rejection(self):
        import contextlib,io,sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from run_agent_batch import construction_options,validate_construction_continuation,main
        args=SimpleNamespace(native_starred=True,native_arrays=True,native_functions=True)
        self.assertEqual(construction_options(args),['--native-starred'])
        with self.assertRaises(ValueError): validate_construction_continuation({'native_arrays':True},args)
        for flag in ('--native-array-exercises','--native-function-exercises','--object-unary'):
            with patch.object(sys,'argv',['run_agent_batch.py','--plan','--native-starred',flag]), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main()


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_LIVE')=='1','requires real isolated native frontend')
class NativeStarredFrontendTests(unittest.TestCase):
    def setUp(self):
        from test_native_function_calls import NativeFunctionTests
        NativeFunctionTests.setUp(self)

    def tearDown(self):
        self.hc.lt.finalize(self.hc.ctxt)

    def test_upstream_python_star_call_uses_existing_scalar_abi(self):
        import numpy as np
        hc=self.hc;seen=[]
        @hc.func('c,p,c,p')
        def helper(a,weight,b,bias):
            seen.append('traced');return a*weight+b+bias
        @hc.func('c')
        def golden(x):
            items=np.array([x,hc.resolveType(.5),x,hc.resolveType(.375)],dtype=object)
            return helper(*items)
        golden.eval();self.assertEqual(golden.outputlen,1);self.assertEqual(seen,['traced'])

    def test_direct_expr_array_is_not_a_native_array_parameter(self):
        import numpy as np
        hc=self.hc
        @hc.func('c')
        def helper(a): return a
        @hc.func('c')
        def golden(x): return helper(np.array([x],dtype=object))
        with self.assertRaises((TypeError,ValueError)): golden.eval()

    def test_checked_dispatch_preserves_left_to_right_and_native_order(self):
        calls=[]
        functions,_=register(source('nested'),CONSTANTS,self.hc,starred_calls=True,observe=calls.append)
        functions['golden'].eval()
        self.assertEqual([e['callee'] for e in calls],['pack','combine'])
        self.assertEqual(functions['golden'].outputlen,1)


if __name__=='__main__': unittest.main()
