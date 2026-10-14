"""Type/shape and real Hecate dispatch tests for native object arithmetic."""
import os
from pathlib import Path
import unittest
from decorated_functions import validate,register
from native_function_core_fixtures import declaration
from native_array_core import arithmetic_type as infer
from run_native_array_arithmetic_goldens import CASES
from test_native_array_core import CONSTANTS,request as array_request

def source(name):
    return (Path(__file__).parent/'golden_cases/native_array_arithmetic'/(name+'.py')).read_text()

class NativeArrayArithmeticTests(unittest.TestCase):
    def test_goldens_versioned_and_legacy_unchanged(self):
        for name in CASES:
            with self.subTest(name=name):
                result=validate(source(name),CONSTANTS,array_arithmetic=True)
                self.assertEqual(result['contract'],'decorated-functions-core-v4')
                self.assertEqual(result['output_ciphertexts'],1)
                self.assertFalse(result['encrypted_correctness_checked'])
                with self.assertRaises(ValueError): validate(source(name),CONSTANTS,starred_calls=True)

    def test_broadcast_cell_kinds_rank_zero_empty_and_bounds(self):
        a=('array',('c','p'),(2,))
        self.assertEqual(infer(a,('array',('p','c'),(2,))),('array',('c','c'),(2,)))
        self.assertEqual(infer(('array',('c',),()),'p'),'c')
        self.assertEqual(infer(('array',('c',),())),'c')
        self.assertEqual(infer(('array',(),(0,2)),('array',('p','p'),(2,))),
                         ('array',(),(0,2)))
        self.assertEqual(infer(('array',('c','c'),(2,1)),('array',('p','p'),(1,2))),
                         ('array',('c',)*4,(2,2)))
        for left,right in ((a,a),('c',a),(a,('array',('c',)*3,(3,))),
                           (('array',('c',)*16,(16,1)),('array',('p',)*16,(1,16)))):
            with self.assertRaises(ValueError): infer(left,right)
        with self.assertRaises(ValueError): infer(a)

    def test_bad_dispatch_mutation_operator_and_plain_cells_rejected_before_frontend(self):
        body='a=np.array([x],dtype=object)\nb=np.array([c0],dtype=object)\n'
        expressions=['x+a','c0+a','a/b','a@a','a**2','b+b','-b',
                     'a+[[x]]','np.add(a,a)','(np.array(x,dtype=object)*c0).item()']
        sources=[declaration('golden','x',body+'return '+e) for e in expressions]
        sources += [declaration('golden','x',body+'a += b\nreturn a[0]'),
                    declaration('golden','x',body+'a[0]=x\nreturn a[0]')]
        class Untouched:
            def __getattr__(self,name): raise AssertionError('Frontend reached before validation')
        for text in sources:
            with self.subTest(source=text),self.assertRaises(ValueError):
                register(text,CONSTANTS,Untouched(),array_arithmetic=True)

    def test_request_provider_cli_frozen_exercises_and_batch(self):
        from candidate_contract import make_request,validate_candidate,request_input_names
        from deepseek_provider import public_request
        from native_function_rules import ARITHMETIC_TASK,ARITHMETIC_CONTRACT
        from run_candidate import parse_args,forward_options
        from run_agent_batch import construction_options,validate_construction_continuation,main
        from types import SimpleNamespace
        import contextlib,io,sys
        from unittest.mock import patch
        old=array_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
        req=make_request(payload,old['model'],old['compiler_profile_sha256'],native_array_arithmetic=True)
        self.assertEqual(req['task'],ARITHMETIC_TASK)
        self.assertEqual(public_request(req),req)
        self.assertEqual(request_input_names(req),('x',))
        self.assertEqual(validate_candidate(dict(schema=1,request_id=req['request_id'],
                         hecate_source=source('vector')),req)['contract'],ARITHMETIC_CONTRACT)
        for exercise in ('nf-scalar','na-reverse'):
            with self.assertRaises(ValueError):
                make_request(payload,old['model'],old['compiler_profile_sha256'],
                             native_array_arithmetic=True,construction_exercise=exercise)
        args=parse_args(['--case','cases/arithmetic-alias-chain.json','--prepare','--native-array-arithmetic'])
        self.assertIn('--native-array-arithmetic',forward_options(args))
        args=SimpleNamespace(native_array_arithmetic=True,native_starred=True,native_arrays=True,native_functions=True)
        self.assertEqual(construction_options(args),['--native-array-arithmetic'])
        with self.assertRaises(ValueError): validate_construction_continuation({'native_starred':True},args)
        for flag in ('--native-array-exercises','--native-function-exercises','--object-unary'):
            with patch.object(sys,'argv',['run_agent_batch.py','--plan','--native-array-arithmetic',flag]), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main()

@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_LIVE')=='1','requires real isolated native frontend')
class NativeArrayArithmeticFrontendTests(unittest.TestCase):
    def setUp(self):
        from test_native_function_calls import NativeFunctionTests
        NativeFunctionTests.setUp(self)
    def tearDown(self): self.hc.lt.finalize(self.hc.ctxt)

    def test_raw_numpy_broadcast_zero_dim_and_storage_identity(self):
        import numpy as np
        hc=self.hc
        @hc.func('c')
        def golden(x):
            a=np.array([[x],[x*2]],dtype=object)
            b=np.array([[hc.resolveType(.5),hc.resolveType(.25)]],dtype=object)
            saved=list(a.flat)
            result=a*b
            self.assertEqual(result.shape,(2,2))
            self.assertTrue(all(u is v for u,v in zip(saved,a.flat)))
            self.assertFalse(np.shares_memory(a,result))
            zero=np.array(x,dtype=object)
            for value in (zero*np.array(hc.resolveType(.5),dtype=object),zero*hc.resolveType(.5),-zero):
                self.assertIsInstance(value,hc.Expr)
                self.assertNotIsInstance(value,np.ndarray)
            empty=np.array([],dtype=object).reshape(0,2)+b
            self.assertEqual(empty.shape,(0,2))
            return result[0,0]+x+.375
        golden.eval();self.assertEqual(golden.outputlen,1)

    def test_scalar_left_array_is_not_rewritten_into_numpy(self):
        import numpy as np
        hc=self.hc
        @hc.func('c')
        def golden(x): return x+np.array([x],dtype=object)
        with self.assertRaises((TypeError,ValueError)): golden.eval()

    def test_checked_nested_call_and_arithmetic_dispatch(self):
        events=[]
        functions,plan=register(source('nested'),CONSTANTS,self.hc,
                                array_arithmetic=True,observe=events.append)
        functions['golden'].eval()
        self.assertEqual([e['callee'] for e in events],['pack','combine'])
        self.assertEqual(functions['golden'].outputlen,1)

if __name__=='__main__': unittest.main()
