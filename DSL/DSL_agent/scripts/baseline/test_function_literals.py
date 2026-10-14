"""Anonymous construction functions and public sorting; no candidate execution."""
import hashlib
import os
from pathlib import Path
import unittest
from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


class FunctionLiteralTests(unittest.TestCase):
    def run_source(self, helpers, body, constants=None, outputs=1):
        constants={} if constants is None else constants
        result=normalize(program(helpers,body),constants,outputs,function_literals=True)
        value=evaluate_tree(result['source'],{k:Packed(v) for k,v in constants.items()},Packed([1,2,3,4]))
        return value,result

    def test_direct_returned_and_container_function_calls(self):
        helpers='def factory():\n    return lambda a: -a\n'
        value,_=self.run_source(helpers,'items = [factory()]\nreturn [(lambda a: -a)(x),factory()(x),items[0](x)]',outputs=3)
        self.assertEqual([v.values for v in value],[(-1,-2,-3,-4)]*3)

    def test_conditional_function_selection_and_parameters(self):
        value,_=self.run_source('', 'f = lambda a, /, *, flip=1: -a if flip else a\nreturn (f if 1 else f)(x,flip=0)')
        self.assertEqual(value.values,(1,2,3,4))

    def test_lambda_defaults_vs_late_bound_comprehension_cells(self):
        value,_=self.run_source('', 'frozen = [lambda a, w=w: a*w for w in [c0,c1]]\nlate = [lambda a: a*w for w in [c0,c1]]\nreturn [frozen[0](x),late[0](x)]',{'c0':.5,'c1':.25},outputs=2)
        self.assertEqual([v.values for v in value],[(.5,1,1.5,2),(.25,.5,.75,1)])

    def test_lambda_parameter_scope_does_not_modify_enclosing_input(self):
        value,_=self.run_source('', 'f = lambda x: -x\nunused = f(x)\nreturn x')
        self.assertEqual(value.values,(1,2,3,4))

    def test_nested_lambda_factory_has_independent_defaults_and_captures(self):
        value,_=self.run_source('', 'factory = lambda a: lambda: a\nf = factory(x)\ng = factory(-x)\nreturn [f(),g()]',outputs=2)
        self.assertEqual([v.values for v in value],[(1,2,3,4),(-1,-2,-3,-4)])

    def test_lambda_default_side_effect_occurs_once_before_calls(self):
        helpers='def change(items):\n    items[0] = -items[0]\n    return items[0]\n'
        value,result=self.run_source(helpers,'items = [x]\nf = lambda a=change(items): a\nfirst = f()\nreturn [first,f(),items[0]]',outputs=3)
        self.assertEqual([v.values for v in value],[(-1,-2,-3,-4)]*3)
        self.assertEqual(result['construction']['container_writes'],1)

    def test_stable_sort_equal_keys_in_both_directions(self):
        for reverse in ('False','True'):
            value,_=self.run_source('', 'items = [(0,x),(0,-x)]\nout = sorted(items,key=lambda item:item[0],reverse='+reverse+')\nreturn [out[0][1],out[1][1]]',outputs=2)
            self.assertEqual([v.values for v in value],[(1,2,3,4),(-1,-2,-3,-4)])

    def test_sort_key_evaluates_once_in_original_order(self):
        helpers='def key(log,item):\n    log.append(item[0])\n    return item[0]\n'
        value,result=self.run_source(helpers,'log = []\nout = sorted([(1,-x),(0,x)],key=lambda item:key(log,item))\nreturn out[0][1] if log[0] == 1 else -x')
        self.assertEqual(value.values,(1,2,3,4))
        self.assertEqual(result['construction']['sort_key_calls'],2)

    def test_sorted_materializes_input_before_key_mutation(self):
        helpers='def key(items,pair):\n    items.append((2,pair[1]))\n    return pair[0]\n'
        value,_=self.run_source(helpers,'items = [(1,-x),(0,x)]\nout = sorted(iter(items),key=lambda pair:key(items,pair))\nreturn out[0][1] if len(items) == 4 else -x')
        self.assertEqual(value.values,(1,2,3,4))

    def test_public_tuple_keys_and_default_public_sort(self):
        value,_=self.run_source('', 'out = sorted([((1,0),-x),((0,2),x)],key=lambda pair:pair[0])\norder = sorted([2,0,1])\nreturn out[0][1] if order[0] == 0 else -x')
        self.assertEqual(value.values,(1,2,3,4))

    def test_empty_sort_does_not_call_key(self):
        value,result=self.run_source('', 'out = sorted([],key=lambda item:missing)\nreturn x')
        self.assertEqual(value.values,(1,2,3,4))
        self.assertEqual(result['construction']['sort_key_calls'],0)

    def test_later_key_callback_cannot_hide_cipher_in_earlier_key(self):
        helpers='def key(keys,item,a):\n    if item == 1:\n        keys[0][0] = a\n    return keys[item]\n'
        with self.assertRaisesRegex(ValueError,'Sort keys must be public'):
            self.run_source(helpers,'keys = [[0],[1]]\nout = sorted([0,1],key=lambda item:key(keys,item,x))\nreturn x')

    def test_cipher_sort_and_external_callable_are_rejected(self):
        for body in ('out = sorted([x,-x])\nreturn x','out = sorted([(0,x)],key=lambda item:item[1])\nreturn x',
                     'out = sorted([0],reverse=x)\nreturn x','f = lambda a: open(a)\nreturn x',
                     'f = lambda a: a.__class__\nreturn x','return (lambda: x).__call__()',
                     'return [x][0]()','out = sorted([0],key=x)\nreturn x'):
            with self.subTest(body=body),self.assertRaises(ValueError):
                self.run_source('',body)

    def test_recursion_and_instance_budget(self):
        with self.assertRaisesRegex(ValueError,'limit'):
            self.run_source('', 'f = lambda: f()\nreturn f()')
        with self.assertRaisesRegex(ValueError,'limit'):
            self.run_source('', 'for i in range(2):\n    for j in range(65):\n        f = lambda: x\nreturn x')

    def test_version_cli_and_legacy_rejection(self):
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance
        from hecate_contract import validate_function
        from run_candidate import parse_args,forward_options
        from dsl_grammar_coverage import analyze_source
        payload=dict(fx_graph='x',public_constants={},constant_origins={},layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        req=make_request(payload,dict(schema=2,id='unit'),'a'*64,function_literals=True)
        source=program('', 'return (lambda a:-a)(x)')
        self.assertEqual(validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['contract'],'hecate-function-v11')
        self.assertTrue(valid_semantic_guidance(req))
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v11')['public_construction']['lambda_instances'],1)
        self.assertIn('--function-literals',forward_options(parse_args(['--case','case.json','--prepare','--function-literals'])))
        for version in range(11):
            with self.assertRaises(ValueError):
                validate_function(source,{},contract='hecate-function-v'+str(version))


class LiteralGoldenPlainTests(unittest.TestCase):
    def test_goldens_against_independent_formula(self):
        from run_function_literal_goldens import PLANS
        root=Path(__file__).resolve().parent
        weights=[[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            constants=dict(c0=weights[0],c1=weights[1]) if case=='construction-linear' else dict(c0=[.5],c1=[.375])
            source=(root/'golden_cases/function_literals'/(golden+'.py')).read_text()
            expanded=normalize(source,constants,2 if case=='construction-linear' else 1,function_literals=True)
            detected=False
            for x in ([0.]*4,[.5,-1.,.25,-.75],[-1.,1.,-1.,1.]):
                expected=[sum(a*b for a,b in zip(x,row)) for row in weights] if case=='construction-linear' else [1.5*v+.375 for v in x]
                value=evaluate_tree(expanded['source'],{k:Packed(v) for k,v in constants.items()},Packed(x))
                actual=[v.values[0] for v in value] if type(value) is list else value.values
                equal=len(actual)==len(expected) and all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
                detected |= not equal
                if not wrong:
                    self.assertTrue(equal,(golden,actual,expected))
            if wrong:
                self.assertTrue(detected,golden)


@unittest.skipUnless(os.environ.get('POSEIDON_LITERAL_REPORT'),'requires actual lambda/sort CPU evidence')
class LiteralEvidenceTests(unittest.TestCase):
    def test_real_artifacts_and_independent_reference(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_function_literal_goldens import PLANS
        root=Path(__file__).resolve().parent
        current={'function_construction.py':'cb6b1dde82b7a7011f1f5008ec902437685d2189df8ce893a3d71d3aee6d2606',
                 'lexical_scope.py':'5f80b879132ca2974a20a48610d13f2037597e1c4aa847fe29037f464f4a7c70',
                 'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539'}
        ClosureEvidenceTests.verify_evidence(self,os.environ['POSEIDON_LITERAL_REPORT'],PLANS,
            'function_literals','hecate-function-synthesis-v12',dict(function_literals=True),current)
