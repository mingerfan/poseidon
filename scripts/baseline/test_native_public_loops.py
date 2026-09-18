"""Public-loop semantics, bounds, old contract separation and actual frontend."""
import ast
import os
from pathlib import Path
import unittest
from decorated_functions import validate,register
from native_public_loops import expand
from native_function_core_fixtures import declaration
from run_native_loop_goldens import CASES
from test_native_array_core import CONSTANTS,request as array_request

def source(name):
    return (Path(__file__).parent/'golden_cases/native_loops'/(name+'.py')).read_text()

class NativeLoopTests(unittest.TestCase):
    def test_fixture_plans_and_old_contracts(self):
        expected=(3,2,5,1,2,2,0,2,3)
        for name,count in zip(CASES,expected):
            with self.subTest(name=name):
                plan=validate(source(name),CONSTANTS,public_loops=True)
                self.assertEqual(plan['contract'],'decorated-functions-core-v5')
                self.assertEqual(plan['public_loops']['iterations'],count)
                self.assertFalse(plan['public_loops']['candidate_python_executed'])
                self.assertFalse(plan['encrypted_correctness_checked'])
                with self.assertRaises(ValueError): validate(source(name),CONSTANTS,array_arithmetic=True)
        self.assertEqual(validate(source('sum4'),CONSTANTS,public_loops=True)['rotation_steps'],[1,2,3])

    def test_python_final_binding_empty_ranges_and_integer_folding(self):
        tree,meta=expand(ast.parse(source('index_binding')),CONSTANTS)
        self.assertFalse(any(type(n) is ast.For for n in ast.walk(tree)))
        fn=tree.body[-1]
        self.assertFalse(any(type(n) is ast.Name and n.id=='i' for n in ast.walk(fn)))
        self.assertEqual([r['trip_count'] for r in meta['loops']],[1,0])
        text=declaration('golden','x','acc=x\nfor i in range(1,3):\n    acc=acc+x.rotate((i*2)//2)\nreturn acc')
        plan=validate(text,{},public_loops=True)
        self.assertEqual(plan['rotation_steps'],[1,2])
        text=declaration('golden','x','for i in range(0):\n    value=x\nreturn x*i')
        with self.assertRaises(ValueError): validate(text,{},public_loops=True)

    def test_invalid_control_and_induction_names_fail_before_frontend(self):
        bodies=[
          'for i in range(x):\n    x=x+x\nreturn x',
          'for i in range(c0):\n    x=x+x\nreturn x',
          'for i in range(1.0):\n    x=x+x\nreturn x',
          'for i in range(True):\n    x=x+x\nreturn x',
          'for i in range(stop=2):\n    x=x+x\nreturn x',
          'for i in range(1,3,0):\n    x=x+x\nreturn x',
          'for i in range(2):\n    i=1\nreturn x',
          'for i in range(2):\n    for i in range(2):\n        x=x+x\nreturn x',
          'for x in range(2):\n    value=x\nreturn value',
          'for c0 in range(2):\n    x=x+x\nreturn x',
          'for i in range(0):\n    import os\nreturn x',
          'for i in range(0):\n    eval("1")\nreturn x',
          'for i in range(0):\n    x.read_text()\nreturn x',
          'for i in range(0):\n    c0=x\nreturn x',
          'for i in range(0):\n    x[0]=x\nreturn x',
          'for i in range(2):\n    return x\nreturn x',
          'for i in range(2):\n    break\nreturn x',
          'for i in range(2):\n    continue\nreturn x',
          'for i in range(2):\n    x=x+x\nelse:\n    x=x+x\nreturn x',
          'while True:\n    x=x+x\nreturn x',
          'range=x\nreturn range',
        ]
        class Untouched:
            def __getattr__(self,name): raise AssertionError('Frontend reached before validation')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError):
                register(declaration('golden','x',body),CONSTANTS,Untouched(),public_loops=True)

    def test_range_expansion_and_integer_resource_bounds(self):
        bodies=['for i in range(129):\n    x=x+x\nreturn x',
                'for i in range(128):\n    for j in range(128):\n        x=x+x\nreturn x',
                'for i in range(1048577,1048578):\n    x=x+x\nreturn x',
                'for i in range(1//0):\n    x=x+x\nreturn x',
                'for i in range(1048576*1048576):\n    x=x+x\nreturn x']
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError):
                validate(declaration('golden','x',body),{},public_loops=True)

    def test_provider_cli_batch_and_frozen_cohorts(self):
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        from native_function_rules import LOOP_TASK,LOOP_CONTRACT
        from run_candidate import parse_args,forward_options
        from run_agent_batch import construction_options,validate_construction_continuation,main
        from types import SimpleNamespace
        from unittest.mock import patch
        import contextlib,io,sys
        old=array_request();payload={k:old[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
        request=make_request(payload,old['model'],old['compiler_profile_sha256'],native_public_loops=True)
        self.assertEqual(request['task'],LOOP_TASK)
        self.assertEqual(public_request(request),request)
        candidate=dict(schema=1,request_id=request['request_id'],hecate_source=source('nested'))
        self.assertEqual(validate_candidate(candidate,request)['contract'],LOOP_CONTRACT)
        for exercise in ('nf-scalar','na-reverse'):
            with self.assertRaises(ValueError):
                make_request(payload,old['model'],old['compiler_profile_sha256'],native_public_loops=True,
                             construction_exercise=exercise)
        args=parse_args(['--case','cases/arithmetic-alias-chain.json','--prepare','--native-public-loops'])
        self.assertIn('--native-public-loops',forward_options(args))
        args=SimpleNamespace(native_public_loops=True,native_array_arithmetic=True,native_starred=True,native_arrays=True)
        self.assertEqual(construction_options(args),['--native-public-loops'])
        with self.assertRaises(ValueError): validate_construction_continuation({},args)
        for flag in ('--native-array-exercises','--native-function-exercises','--object-unary'):
            with patch.object(sys,'argv',['run_agent_batch.py','--plan','--native-public-loops',flag]), \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit): main()

@unittest.skipUnless(os.environ.get('POSEIDON_NATIVE_CALLS_LIVE')=='1','requires real isolated native frontend')
class NativeLoopFrontendTests(unittest.TestCase):
    def setUp(self):
        from test_native_function_calls import NativeFunctionTests
        NativeFunctionTests.setUp(self)
    def tearDown(self): self.hc.lt.finalize(self.hc.ctxt)

    def test_upstream_python_range_traces_helper_once(self):
        hc=self.hc;seen=[]
        @hc.func('c')
        def helper(value):
            seen.append('trace')
            acc=value
            for step in range(1,4): acc=acc+value.rotate(step)
            return acc
        @hc.func('c')
        def golden(x): return helper(x)+helper(x)
        golden.eval()
        self.assertEqual(seen,['trace'])
        self.assertEqual(golden.outputlen,1)

    def test_checked_calls_repeat_without_retracing_helper(self):
        calls=[]
        funcs,plan=register(source('helper_repeated'),CONSTANTS,self.hc,
                            public_loops=True,observe=calls.append)
        funcs['golden'].eval()
        self.assertEqual([c['callee'] for c in calls],['half','half'])
        self.assertEqual(calls[0]['span'],calls[1]['span'])
        self.assertEqual(plan['public_loops']['iterations'],2)
        self.assertEqual(funcs['half'].evaluation_state,'done')
        self.assertEqual(funcs['golden'].outputlen,1)

if __name__=='__main__': unittest.main()
