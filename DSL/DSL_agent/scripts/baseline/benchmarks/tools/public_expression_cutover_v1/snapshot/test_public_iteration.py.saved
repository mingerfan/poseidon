"""Public iteration semantics; symbolic checks are separate from FHE evidence."""
import hashlib
import os
from pathlib import Path
import unittest

from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


class PublicIterationTests(unittest.TestCase):
    def run_source(self, helpers, body, constants=None, outputs=1):
        constants = {} if constants is None else constants
        result = normalize(program(helpers, body), constants, outputs, public_iteration=True)
        value = evaluate_tree(result['source'], {k:Packed(v) for k,v in constants.items()}, Packed([1,2,3,4]))
        return value, result

    def test_outer_iterable_uses_enclosing_name_and_target_does_not_leak(self):
        value, _ = self.run_source('', 'i = [x,-x]\nitems = [i for i in i]\nreturn [i[0],items[1]]', outputs=2)
        self.assertEqual([v.values for v in value], [(1,2,3,4),(-1,-2,-3,-4)])

    def test_target_can_shadow_cipher_input_without_rebinding_it(self):
        value, _ = self.run_source('', 'items = [-x for x in [x]]\nreturn [x,items[0]]', outputs=2)
        self.assertEqual([v.values for v in value], [(1,2,3,4),(-1,-2,-3,-4)])

    def test_nested_clauses_see_previous_bindings_and_keep_order(self):
        value, result = self.run_source('', 'items = [x if j == i else -x for i in range(2) for j in range(i+1)]\nreturn items', outputs=3)
        self.assertEqual([v.values for v in value], [(1,2,3,4),(-1,-2,-3,-4),(1,2,3,4)])
        self.assertEqual(result['construction']['comprehension_iterations'], 5)

    def test_nested_comprehensions_have_separate_frames(self):
        value, _ = self.run_source('', 'i = 3\nitems = [[x for i in range(2)] for i in range(1)]\nreturn items[0][0] if i == 3 else -x')
        self.assertEqual(value.values, (1,2,3,4))

    def test_comprehension_target_is_not_enclosing_local(self):
        value, _ = self.run_source('', 'items = [x for weight in range(1)]\nreturn x*weight', {'weight':.5})
        self.assertEqual(value.values, (.5,1,1.5,2))
        with self.assertRaisesRegex(ValueError, 'Undefined'):
            self.run_source('', 'items = [x for i in range(1)]\nreturn x if i == 0 else -x')

    def test_outer_iterable_evaluated_once_and_filters_short_circuit(self):
        helpers = ('def source(log):\n    log.append(1)\n    return range(3)\n'
                   'def accept(log,i):\n    log.append(i)\n    return i\n')
        value, _ = self.run_source(helpers, 'log = []\nitems = [x for i in source(log) if i > 0 if accept(log,i)]\nreturn items[0] if len(log) == 3 else -x')
        self.assertEqual(value.values, (1,2,3,4))

    def test_dict_comprehension_evaluates_key_before_value(self):
        helpers = ('def key(log):\n    log.append(1)\n    return "k"\n'
                   'def value(log,a):\n    return a if len(log) == 1 else -a\n')
        value, _ = self.run_source(helpers, 'log = []\ntable = {key(log):value(log,x) for i in range(1)}\nreturn table["k"]')
        self.assertEqual(value.values, (1,2,3,4))

    def test_dict_duplicate_keys_last_value_and_iteration_order(self):
        value, _ = self.run_source('', 'table = {key:value for key,value in [("a",x),("b",x),("a",-x)]}\nkeys = list(table)\nreturn [table[keys[0]],table[keys[1]]]', outputs=2)
        self.assertEqual([v.values for v in value], [(-1,-2,-3,-4),(1,2,3,4)])

    def test_enumerate_is_lazy_and_has_requested_start(self):
        value, _ = self.run_source('', 'items = [x]\nindices = enumerate(items,3)\nitems[0] = -x\ni,value = next(indices)\nreturn value if i == 3 else x')
        self.assertEqual(value.values, (-1,-2,-3,-4))

    def test_zip_shared_iterator_consumes_left_to_right_until_shortest(self):
        value, _ = self.run_source('', 'items = iter([x,-x,-x])\npairs = list(zip(items,items))\nremaining = next(items,x)\nreturn [pairs[0][0],pairs[0][1],remaining]', outputs=3)
        self.assertEqual([v.values for v in value], [(1,2,3,4),(-1,-2,-3,-4),(1,2,3,4)])

    def test_zip_observes_list_changes_after_creation(self):
        value, _ = self.run_source('', 'items = [x]\npairs = zip(items,[x])\nitems[0] = -x\na,b = next(pairs)\nreturn a-b')
        self.assertEqual(value.values, (-2,-4,-6,-8))

    def test_reversed_observes_mutation_without_reversing_original(self):
        value, _ = self.run_source('', 'items = [x,x]\nbackwards = reversed(items)\nitems[1] = -x\nreturn [next(backwards),items[0]]', outputs=2)
        self.assertEqual([v.values for v in value], [(-1,-2,-3,-4),(1,2,3,4)])

    def test_iter_aliases_share_consumption_and_next_default_is_eager(self):
        helpers = 'def change(items):\n    items[0] = -items[0]\n    return items[0]\n'
        value, _ = self.run_source(helpers, 'items = [x]\nit = iter([x,-x])\nother = iter(it)\na = next(it,change(items))\nb = next(other)\nreturn [a,b,items[0]]', outputs=3)
        self.assertEqual([v.values for v in value], [(1,2,3,4),(-1,-2,-3,-4),(-1,-2,-3,-4)])

    def test_for_loop_and_star_arguments_accept_public_iterators(self):
        helpers = 'def combine(a,b):\n    return a-b\n'
        value, _ = self.run_source(helpers, 'items = []\nfor a in iter([x,-x]):\n    items.append(a)\nreturn combine(*iter(items))')
        self.assertEqual(value.values, (2,4,6,8))

    def test_dict_iterator_invalidated_by_size_change(self):
        with self.assertRaisesRegex(ValueError, 'invalidated'):
            self.run_source('', 'table = {"a":x}\nit = iter(table)\ntable["b"] = x\nkey = next(it)\nreturn x')

    def test_exhaustion_without_default_and_cipher_iteration_fail(self):
        for body in ('return next(iter([]))', 'return [a for a in x]',
                     'return [x for i in range(1) if x]', 'return list(x)',
                     'return next(x)', 'return tuple(c0)', 'return reversed(iter([x]))'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.run_source('', body, {'c0':[.5]})

    def test_limits_count_filtered_work_and_materialized_results(self):
        for body in ('items = [x for i in range(2) for j in range(65)]\nreturn x',
                     'items = [x for i in range(128) for j in range(128) if 0]\nreturn x'):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, 'limit'):
                self.run_source('', body)

    def test_generator_async_set_and_external_calls_remain_rejected(self):
        for body in ('items = (x for i in range(1))\nreturn x',
                     'items = {x for i in range(1)}\nreturn x',
                     'items = [open(i) for i in []]\nreturn x',
                     'items = [x async for i in []]\nreturn x'):
            with self.assertRaises(ValueError):
                self.run_source('', body)

    def test_version_rules_cli_and_legacy_rejection(self):
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance
        from hecate_contract import validate_function
        from run_candidate import parse_args,forward_options
        from dsl_grammar_coverage import analyze_source
        payload=dict(fx_graph='x',public_constants={},constant_origins={},layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        req=make_request(payload,dict(schema=2,id='unit'),'a'*64,public_iteration=True)
        source=program('', 'return [a for a in [x]][0]')
        checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)
        self.assertEqual(checked['contract'],'hecate-function-v10')
        self.assertTrue(valid_semantic_guidance(req))
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v10')['public_construction']['comprehensions'],1)
        self.assertIn('--public-iteration',forward_options(parse_args(['--case','case.json','--prepare','--public-iteration'])))
        for version in range(10):
            with self.assertRaises(ValueError):
                validate_function(source,{},contract='hecate-function-v'+str(version))


class IterationGoldenPlainTests(unittest.TestCase):
    def test_manual_goldens_against_independent_formulas(self):
        from run_public_iteration_goldens import PLANS
        root=Path(__file__).resolve().parent
        weights=[[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            constants=dict(c0=weights[0],c1=weights[1]) if case=='construction-linear' else dict(c0=[.5],c1=[.375])
            source=(root/'golden_cases/public_iteration'/(golden+'.py')).read_text()
            expanded=normalize(source,constants,2 if case=='construction-linear' else 1,public_iteration=True)
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


@unittest.skipUnless(os.environ.get('POSEIDON_ITERATION_REPORT'),'requires actual public-iteration CPU evidence')
class IterationEvidenceTests(unittest.TestCase):
    def test_real_artifacts_and_independent_reference(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_public_iteration_goldens import PLANS
        root=Path(__file__).resolve().parent
        # Preserve actual pre-lambda producers; independently reproduce old mode.
        current={'function_construction.py':'ae0b8b42db89b046e920aae3754a0e2064702a575da3bf98f6cdc489435f3245',
                 'lexical_scope.py':'16f92b5b64e123e68db3c96918bba7eb492b7e30d21510385323f47774e2eba9',
                 'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539'}
        ClosureEvidenceTests.verify_evidence(self,os.environ['POSEIDON_ITERATION_REPORT'],PLANS,
            'public_iteration','hecate-function-synthesis-v11',dict(public_iteration=True),current)
