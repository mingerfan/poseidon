"""Native scalar augmented assignment: alias semantics, contract and real evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from decorated_functions import validate,register
from native_function_core_fixtures import declaration
from run_native_augmented_goldens import CASES
from test_native_array_core import CONSTANTS,request as array_request


def source(name):
    return (Path(__file__).parent/'golden_cases/native_augmented'/(name+'.py')).read_text()


class NativeScalarAugmentedTests(unittest.TestCase):
    def test_typed_sites_and_old_contracts_unchanged(self):
        for name in CASES:
            with self.subTest(name=name):
                plan=validate(source(name),CONSTANTS,scalar_augmented=True)
                self.assertEqual(plan['contract'],'decorated-functions-core-v6')
                self.assertTrue(plan['scalar_augmented']['sites'])
                self.assertFalse(plan['encrypted_correctness_checked'] or plan['agent_generation_validated'])
                self.assertFalse(plan['scalar_augmented']['array_mutation_enabled'])
                for flags in ({},{'arrays':True},{'starred_calls':True},{'array_arithmetic':True},{'public_loops':True}):
                    with self.assertRaises(ValueError):validate(source(name),CONSTANTS,**flags)
        plan=validate(source('plain_left'),CONSTANTS,scalar_augmented=True)
        self.assertEqual([(s['left_kind'],s['right_kind']) for s in plan['scalar_augmented']['sites']],
                         [('p','c'),('c','c')])
        plan=validate(source('sum4'),{},scalar_augmented=True)
        self.assertEqual(plan['rotation_steps'],[1,2,3])
        self.assertEqual(len(plan['scalar_augmented']['sites']),3)

    def test_invalid_targets_and_unsupported_mutation_rejected_before_frontend(self):
        bodies=[
            'missing+=x\nreturn x', 'c0+=x\nreturn x', 'zero_ct+=x\nreturn x',
            'value=c0\nvalue+=c1\nreturn x', 'x/=c0\nreturn x', 'x**=2\nreturn x',
            'x.obj+=x\nreturn x', 'items=[x]\nitems[0]+=x\nreturn x',
            'items=np.array([x],dtype=object)\nitems+=x\nreturn items[0]',
            'items=np.array(x,dtype=object)\nitems*=c0\nreturn items.item()',
            'x+=np.array([x],dtype=object)\nreturn x',
            'for i in range(2):\n    i+=1\nreturn x',
            'for i in range(0):\n    c0+=x\nreturn x',
            'for i in range(0):\n    x.obj+=x\nreturn x',
            'for i in range(0):\n    eval("1")\nreturn x',
        ]
        class Untouched:
            def __getattr__(self,name):raise AssertionError('frontend touched before validation')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError):
                register(declaration('golden','x',body),CONSTANTS,Untouched(),scalar_augmented=True)

    def test_loop_expansion_cost_and_readonly_indices(self):
        text=declaration('golden','x','acc=x\nfor i in range(2):\n    for j in range(2):\n        acc+=x\nreturn acc')
        plan=validate(text,{},scalar_augmented=True)
        self.assertEqual(len(plan['scalar_augmented']['sites']),4)
        for body in ('for i in range(129):\n    x+=x\nreturn x',
                     'for i in range(128):\n    for j in range(128):\n        x+=x\nreturn x'):
            with self.assertRaises(ValueError):validate(declaration('golden','x',body),{},scalar_augmented=True)

    def test_request_provider_cli_and_batch_freeze(self):
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        from native_function_rules import AUGMENTED_TASK,AUGMENTED_CONTRACT
        from run_candidate import parse_args,forward_options
        from run_agent_batch import construction_options,validate_construction_continuation,main
        old=array_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
        req=make_request(payload,old['model'],old['compiler_profile_sha256'],native_scalar_augmented=True)
        self.assertEqual(req['task'],AUGMENTED_TASK)
        self.assertEqual(public_request(req),req)
        checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source('add')),req)
        self.assertEqual(checked['contract'],AUGMENTED_CONTRACT)
        for name in ('nf-scalar','na-reverse','ns-list'):
            with self.assertRaises(ValueError):make_request(payload,old['model'],old['compiler_profile_sha256'],
                native_scalar_augmented=True,construction_exercise=name)
        with self.assertRaises(ValueError):make_request(payload,old['model'],old['compiler_profile_sha256'],
                native_scalar_augmented=True,object_unary=True)
        args=parse_args(['--case','cases/arithmetic-alias-chain.json','--prepare','--native-scalar-augmented'])
        self.assertIn('--native-scalar-augmented',forward_options(args))
        self.assertEqual(construction_options(args),['--native-scalar-augmented'])
        with self.assertRaisesRegex(ValueError,'fresh batch'):validate_construction_continuation({},args)
        with self.assertRaisesRegex(ValueError,'fresh batch'):
            validate_construction_continuation({'native_scalar_augmented':True},SimpleNamespace())
        output=io.StringIO()
        with (patch('sys.argv',['run_agent_batch.py','--plan','--native-scalar-augmented']),
              patch('agent_credentials.load_api_key',side_effect=AssertionError('credentials read')),
              contextlib.redirect_stdout(output)):
            self.assertEqual(main(),0)
        plan=json.loads(output.getvalue())
        self.assertTrue(plan['native_scalar_augmented'] and plan['native_public_loops'])
        self.assertEqual(plan['agent_calls'],0)
        for flag in ('--native-function-exercises','--native-array-exercises','--native-star-exercises','--object-unary'):
            with (patch('sys.argv',['run_agent_batch.py','--plan','--native-scalar-augmented',flag]),
                  contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)):main()


@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_LIVE')=='1','requires real isolated Hecate frontend')
class NativeScalarAugmentedFrontendTests(unittest.TestCase):
    def setUp(self):
        from test_native_function_calls import NativeFunctionTests
        NativeFunctionTests.setUp(self)
    def tearDown(self):self.hc.lt.finalize(self.hc.ctxt)

    def test_actual_expr_rebinding_and_ndarray_mutation_are_different(self):
        hc=self.hc
        @hc.func('c')
        def golden(x):
            original=x;handle=x.obj
            storage=hc.np.array([x],dtype=object)
            x*=.5
            self.assertIsNot(x,original)
            self.assertEqual(original.obj,handle)
            self.assertIs(storage[0],original)
            alias=storage
            storage+=hc.resolveType(.25)
            self.assertIs(alias,storage)
            self.assertIsNot(alias[0],original)
            self.assertEqual(original.obj,handle)
            return x+original+.375
        golden.eval();self.assertEqual(golden.outputlen,1)

    def test_checked_loop_and_native_calls_emit_actual_augmented_events(self):
        calls=[];events=[]
        functions,plan=register(source('nested_loop'),CONSTANTS,self.hc,scalar_augmented=True,
                                observe=calls.append,observe_augmented=events.append)
        functions['golden'].eval()
        self.assertEqual(len(events),4) # helper body traced once, not three dynamic traces
        self.assertEqual([c['callee'] for c in calls],['scale']*3)
        self.assertEqual(sorted((r['function'],r['target'],r['operation'],r['span']) for r in events),
            sorted((r['function'],r['target'],r['operation'],r['span']) for r in plan['scalar_augmented']['sites']))
        self.assertTrue(all(r['fresh_expr'] and r['original_expr_unchanged'] for r in events))


if __name__=='__main__':unittest.main()
