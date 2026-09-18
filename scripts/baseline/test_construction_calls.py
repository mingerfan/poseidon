"""Argument binding, default lifetime and true-FHE evidence are separate tests."""
import ast
import hashlib
import json
import os
from pathlib import Path
import unittest

from construction_calls import bind
from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


class CallBindingTests(unittest.TestCase):
    def run_source(self, helpers, body, constants=None, outputs=1):
        constants = {} if constants is None else constants
        source = program(helpers, body)
        expanded = normalize(source, constants, outputs, call_binding=True)
        value = evaluate_tree(expanded['source'], {k: Packed(v) for k,v in constants.items()}, Packed([1,2,3,4]))
        return value, expanded

    def test_binding_matrix_matches_real_python_function(self):
        # Trusted handwritten oracle, never execute a candidate source string.
        def oracle(a, /, b=2, *rest, c, d=4, **extra):
            return dict(a=a, b=b, rest=rest, c=c, d=d, extra=extra)
        args = ast.parse('def f(a, /, b=2, *rest, c, d=4, **extra):\n return a').body[0].args
        checked = 0
        for count in range(6):
            positional = list(range(count))
            for mask in range(32):
                keywords = {key: i+10 for i,key in enumerate(('a','b','c','d','other')) if mask & (1 << i)}
                try:
                    expected = oracle(*positional, **keywords)
                except TypeError:
                    with self.assertRaises(ValueError):
                        bind(args, (2,), (('d',4),), positional, keywords)
                else:
                    self.assertEqual(bind(args, (2,), (('d',4),), positional, keywords), expected)
                checked += 1
        self.assertEqual(checked, 192)

    def test_positional_keyword_and_default_overrides(self):
        value, _ = self.run_source('def f(a, b=weight, *, bias=offset):\n    return a*b+bias\n',
            'return f(b=weight, a=x)', {'weight':.5,'offset':.25})
        self.assertEqual(value.values, (.75,1.25,1.75,2.25))

    def test_default_captures_value_while_closure_observes_rebinding(self):
        value, expanded = self.run_source('', 'v = x\ndef f(a=v):\n    return [a, v]\nv = -x\nreturn f()', outputs=2)
        self.assertEqual([v.values for v in value], [(1,2,3,4),(-1,-2,-3,-4)])
        self.assertEqual(expanded['construction']['default_evaluations'], 1)

    def test_defaults_once_left_to_right_before_function_binding(self):
        helpers = ('def change(items):\n    items[0] = -items[0]\n    return items[0]\n')
        value, expanded = self.run_source(helpers,
            'items = [x]\ndef f(a=change(items), *, b=change(items)):\n    return [a,b]\nunused = f()\nreturn f()', outputs=2)
        self.assertEqual([v.values for v in value], [(-1,-2,-3,-4),(1,2,3,4)])
        self.assertEqual(expanded['construction']['container_writes'], 2)

    def test_previous_binding_visible_in_redefinition_default(self):
        value, _ = self.run_source('', 'def f(a):\n    return -a\ndef f(a=f(x)):\n    return a\nreturn f()')
        self.assertEqual(value.values, (-1,-2,-3,-4))

    def test_default_mutable_list_shared_per_function_instance(self):
        helpers = ('def factory():\n    def f(a, items=[]):\n        items.append(a)\n'
                   '        return items[0]\n    return f\n')
        value, _ = self.run_source(helpers, 'f = factory()\ng = factory()\nfirst = f(x)\nsecond = f(-x)\nthird = g(-x)\nreturn [first,second,third]', outputs=3)
        self.assertEqual([v.values for v in value], [(1,2,3,4),(1,2,3,4),(-1,-2,-3,-4)])

    def test_mutation_vs_rebinding_of_default_container(self):
        value, _ = self.run_source('', 'items = [x]\ndef f(a=items):\n    return a[0]\nitems[0] = -x\nitems = [x]\nreturn f()')
        self.assertEqual(value.values, (-1,-2,-3,-4))

    def test_keyword_evaluation_order_not_parameter_order(self):
        helpers = ('def change(items):\n    old = items[0]\n    items[0] = -old\n    return old\n'
                   'def subtract(a,b):\n    return a-b\n')
        value, _ = self.run_source(helpers, 'items = [x]\nreturn subtract(b=change(items), a=change(items))')
        self.assertEqual(value.values, (-2,-4,-6,-8))

    def test_star_arguments_evaluate_before_keywords_even_if_written_later(self):
        helpers = ('def change(items):\n    old = items[0]\n    items[0] = -old\n    return old\n'
                   'def subtract(a,b):\n    return a-b\n')
        value, _ = self.run_source(helpers, 'items = [x]\nreturn subtract(b=change(items), *[change(items)])')
        self.assertEqual(value.values, (2,4,6,8))

    def test_varargs_kwargs_forwarding_and_posonly_name_in_kwargs(self):
        helpers = ('def f(a, /, *rest, **options):\n    return a+rest[0]+options["a"]\n'
                   'def forward(*args, **kwargs):\n    return f(*args, **kwargs)\n')
        value, _ = self.run_source(helpers, 'return forward(x, -x, a=x)')
        self.assertEqual(value.values, (1,2,3,4))

    def test_kwargs_mapping_is_fresh_but_values_share_references(self):
        helpers = ('def update(**options):\n    options["items"][0] = -options["items"][0]\n'
                   '    options["items"] = []\n')
        value, _ = self.run_source(helpers, 'mapping = {"items": [x]}\nupdate(**mapping)\nreturn mapping["items"][0]')
        self.assertEqual(value.values, (-1,-2,-3,-4))

    def test_keyword_only_default_and_nonlocal_parameter(self):
        helpers = ('def factory(*, a):\n    def change():\n        nonlocal a\n        a = -a\n'
                   '    change()\n    return a\n')
        value, _ = self.run_source(helpers, 'return factory(a=x)')
        self.assertEqual(value.values, (-1,-2,-3,-4))

    def test_module_defaults_use_definition_order(self):
        helpers = 'def identity(a):\n    return a\ndef f(a, weight=identity(c0)):\n    return a*weight\n'
        value, _ = self.run_source(helpers, 'return f(x)', {'c0':.5})
        self.assertEqual(value.values, (.5,1,1.5,2))
        with self.assertRaises(ValueError):
            self.run_source('def f(a, weight=later(c0)):\n    return a*weight\ndef later(a):\n    return a\n', 'return f(x)', {'c0':.5})

    def test_required_duplicate_unexpected_and_wrong_parameter_kinds(self):
        helpers = 'def f(a, /, b, *, c):\n    return a+b+c\n'
        for call in ('f(x,x)', 'f(x,x,x)', 'f(a=x,b=x,c=x)', 'f(x,x,b=x,c=x)',
                     'f(x,x,c=x,z=x)', 'f(x,x,c=x,**{"c":x})'):
            with self.subTest(call=call), self.assertRaises(ValueError):
                self.run_source(helpers, 'return '+call)

    def test_invalid_star_and_keywords_never_become_cipher_control(self):
        helpers = 'def f(*a, **b):\n    return x\n'
        for body in ('return f(*x)', 'return f(**x)', 'return f(**[x])',
                     'return x.rotate(step=1)', 'return x.rotate(1, ignored=x)',
                     'return f(**{1:x})', 'mapping = {}\nmapping["cycle"] = mapping\nreturn x'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.run_source(helpers, body)

    def test_default_effects_are_checked_even_if_helper_unused(self):
        for helpers in ('def f(a=open(x)):\n    return a\n', 'def f(a=missing):\n    return a\n'):
            with self.assertRaises(ValueError):
                self.run_source(helpers, 'return x')

    def test_new_contract_cli_and_old_rejection(self):
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance
        from hecate_contract import validate_function
        from run_candidate import parse_args,forward_options
        from dsl_grammar_coverage import analyze_source
        payload=dict(fx_graph='x',public_constants={},constant_origins={},layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        req=make_request(payload,dict(schema=2,id='unit'),'a'*64,call_binding=True)
        source=program('def f(a, *, negate=1):\n    return -a if negate else a\n','return f(x,negate=1)')
        checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)
        self.assertEqual(checked['contract'],'hecate-function-v9')
        self.assertTrue(valid_semantic_guidance(req))
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v9')['public_construction']['keyword_arguments'],1)
        self.assertIn('--call-binding',forward_options(parse_args(['--case','case.json','--prepare','--call-binding'])))
        for version in range(9):
            with self.assertRaises(ValueError):
                validate_function(source,{},contract='hecate-function-v'+str(version))


class CallGoldenPlainTests(unittest.TestCase):
    def test_manual_goldens_against_independent_formulas(self):
        from run_call_binding_goldens import PLANS
        root=Path(__file__).resolve().parent
        weights=[[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            constants=dict(c0=weights[0],c1=weights[1]) if case=='construction-linear' else dict(c0=[.5],c1=[.375])
            source=(root/'golden_cases/call_binding'/(golden+'.py')).read_text()
            expanded=normalize(source,constants,2 if case=='construction-linear' else 1,call_binding=True)
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


@unittest.skipUnless(os.environ.get('POSEIDON_CALL_BINDING_REPORT'), 'requires actual parameter-binding CPU evidence')
class CallBindingEvidenceTests(unittest.TestCase):
    def test_real_artifacts_and_independent_reference(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_call_binding_goldens import PLANS
        # Exact pre-iteration producers; old-mode expansion is independently
        # reproduced by the shared auditor, not relabeled as new execution.
        current={'function_construction.py':'f209b563ca3032a602eeca9ab48f878506f3387cd5765ac8df83ba427cebf687',
                 'lexical_scope.py':'4883ebbc397b469377a9137b6adb49f0a457f7a8dd7dbf5917beb9c242f259f9',
                 'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539'}
        value=os.environ['POSEIDON_CALL_BINDING_REPORT']
        reports=json.loads(value) if value.lstrip().startswith('[') else value
        ClosureEvidenceTests.verify_evidence(self,reports,PLANS,
            'call_binding','hecate-function-synthesis-v10',dict(call_binding=True),current)
