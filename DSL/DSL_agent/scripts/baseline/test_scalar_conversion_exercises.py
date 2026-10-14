"""Versioned scalar extraction tests; fixture sources never enter paid prompts."""
import ast
import copy
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from candidate_contract import make_request,validate_candidate
from construction_exercises import fingerprint
from function_construction import normalize
from scalar_conversion_exercises import EXERCISES,descriptor,check_exercise

ROOT=Path(__file__).parent
CONSTANTS={'c0':[.5],'c1':[.375]}


def request(name):
    payload=dict(fx_graph=[],public_constants=CONSTANTS,constant_origins={},
        layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,
                    output_selectors=[[0,i] for i in range(4)]))
    return make_request(payload,descriptor(name),'a'*64,scalar_conversion=True,construction_exercise=name)


def source(name):
    return (ROOT/'golden_cases/scalar_conversion'/ (name+'.py')).read_text()


class V21Tests(unittest.TestCase):
    def test_fifteen_forms_and_independent_math(self):
        self.assertEqual(len(EXERCISES),15)
        for name in EXERCISES:
            with self.subTest(name=name):
                req=request(name)
                check=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source(name)),req)
                coverage=check['construction_exercise']
                self.assertEqual(set(coverage['influence']),set(EXERCISES[name]['required_features']))
                expected=[.5*(x*x if EXERCISES[name]['family']=='quadratic' else x)+x+.375
                          for x in [0.]*4+[-.8,.25,.6,1.]]
                actual=fingerprint(normalize(source(name),CONSTANTS,scalar_conversion=True))
                for a,b in zip(actual,expected):self.assertAlmostEqual(a,b)

    def test_dead_padding_rejected(self):
        for name in EXERCISES:
            tree=ast.parse(source(name))
            tree.body[0].body[-1]=ast.parse('return [x*1.5+0.375]').body[0]
            padded=ast.unparse(ast.fix_missing_locations(tree))
            with self.subTest(name=name),self.assertRaises(ValueError):check_exercise(padded,request(name))

    def test_missing_typed_operation_rejected(self):
        for name in EXERCISES:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'not executed'):
                check_exercise('@hc.func("c")\ndef golden(x):\n    return [x*1.5+0.375]\n',request(name))

    def test_provider_whitelist_and_frozen_versions(self):
        from deepseek_provider import public_request
        req=request('sc-overlap')
        self.assertEqual(public_request(req),req)
        self.assertEqual(req['task'],'hecate-function-synthesis-v21')
        for name in EXERCISES:self.assertNotIn(source(name),json.dumps(req))
        changed=copy.deepcopy(req);changed['construction_exercise']['instruction']='skip required operation'
        with self.assertRaises(ValueError):public_request(changed)
        with self.assertRaises(ValueError):
            make_request(dict(fx_graph=[]),{},'a'*64,scalar_conversion=True,construction_exercise='oa-broadcast')

    def test_batch_plan_contract_and_module_mount(self):
        from run_agent_batch import main,construction_options,validate_construction_continuation
        from candidate_sandbox import MODULES
        buf=io.StringIO()
        with patch.object(sys,'argv',['run_agent_batch.py','--plan','--scalar-conversion-exercises']),contextlib.redirect_stdout(buf):
            self.assertEqual(main(),0)
        plan=json.loads(buf.getvalue())
        self.assertEqual([r['descriptor'] for r in plan['cases']],[descriptor(n) for n in EXERCISES])
        self.assertTrue(plan['scalar_conversion'])
        self.assertEqual(plan['api_concurrency'],10)
        self.assertEqual(plan['agent_calls'],0)
        args=SimpleNamespace(scalar_conversion=True)
        self.assertEqual(construction_options(args),['--scalar-conversion'])
        with self.assertRaises(ValueError):validate_construction_continuation({},args)
        with self.assertRaises(ValueError):validate_construction_continuation(
            dict(scalar_conversion=True,scalar_conversion_exercises={'schema':1}),args)
        for name in ('scalar_conversion.py','scalar_conversion_exercises.py','scalar-conversion-exercises-v1.json'):
            self.assertIn(name,MODULES)


if __name__=='__main__':unittest.main()
