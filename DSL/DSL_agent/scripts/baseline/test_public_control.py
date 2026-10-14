"""Public control tests: symbolic evaluation is not encrypted execution."""
import copy
import itertools
import operator
import os
from pathlib import Path
import unittest
from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


class PublicControlTests(unittest.TestCase):
    def run_source(self, body, helpers='', constants=None):
        constants = {} if constants is None else constants
        before = copy.deepcopy(constants)
        result = normalize(program(helpers, body), constants, public_control=True)
        self.assertEqual(constants, before)
        actual = evaluate_tree(result['source'],
            {k: Packed(v) for k,v in result['constants'].items()}, Packed([1,2,3,4]))
        return actual.values, result

    def test_chebyshev_recurrence_integer_tuple_keys_while_else(self):
        helpers = '''def chebyshev(a):
    terms = {(0,0): 1, (1,0): a}
    degree = 2
    while degree <= 3:
        previous = (degree-1,0)
        before = (degree-2,0)
        if previous not in terms or before not in terms:
            return -a
        terms[(degree,0)] = 2*a*terms[previous] - terms[before]
        degree += 1
    else:
        return terms[(3,0)]
'''
        values, result = self.run_source('return chebyshev(x)', helpers)
        self.assertEqual(values, tuple(4*x**3-3*x for x in (1,2,3,4)))
        self.assertEqual(result['construction']['while_iterations'], 2)
        self.assertEqual(result['construction']['membership_tests'], 4)

    def test_mapping_collision_python_numeric_keys_and_comprehensions(self):
        values, _ = self.run_source('d = {1: 8, True: 7, 1.0: 2, None: 3, (2,"a"): 4}\n'
            'e = {i: d[i] for i in d}\nreturn x * (e[1]+e[None]+e[(2,"a")])')
        self.assertEqual(values, (9,18,27,36))

    def test_short_circuit_returns_operands_without_testing_last_cipher(self):
        for expression in ('False or x', 'True and x', '[] or x', '[x] and x',
                           '(0 and missing) or x', '(1 or missing) and x'):
            with self.subTest(expression=expression):
                self.assertEqual(self.run_source('return '+expression)[0], (1,2,3,4))
        helpers = 'def fail():\n    return unknown\n'
        self.assertEqual(self.run_source('return x if not (False and fail()) else -x', helpers)[0],(1,2,3,4))

    def test_truth_of_public_containers_functions_and_iterators(self):
        helpers = 'def identity(a):\n    return a\n'
        for condition in ('[x]', '{0:x}', 'identity', 'iter([])', '"a"', 'not None', 'not []'):
            with self.subTest(condition=condition):
                self.assertEqual(self.run_source('return x if '+condition+' else -x', helpers)[0], (1,2,3,4))

    def test_membership_consumes_iterator_to_match_not_to_end(self):
        values,_ = self.run_source('it = iter([1,2,3])\nfound = 2 in it\n'
            'return x * next(it) if found and ("bc" in "abcd") else -x')
        self.assertEqual(values, (3,6,9,12))

    def test_public_equality_order_and_chain_short_circuit(self):
        conditions = ('[1,2] < [1,3]', '(1,2) <= (1,2,0)', '"a" < "b"',
            '{0:[1,2]} == {0:[1,2]}', '[1] != (1,)', 'None == None', 'True == 1.0',
            'not (2 < 1 < missing)', 'None in [0,None]')
        for condition in conditions:
            with self.subTest(condition=condition):
                self.assertEqual(self.run_source('return x if '+condition+' else -x')[0], (1,2,3,4))

    def test_for_while_else_break_continue_and_pass(self):
        for loop in ('for n in range(3):', 'while n < 3:'):
            for stop in (False,True):
                body = 'n = 0\nw = 1\n'+loop+'\n    n += 1\n    pass\n'
                body += '    if n == 1:\n        continue\n'
                if stop: body += '    break\n'
                body += 'else:\n    w = 2\nreturn x*w'
                with self.subTest(loop=loop,stop=stop):
                    self.assertEqual(self.run_source(body)[0], tuple(x*(1 if stop else 2) for x in (1,2,3,4)))

    def test_loop_else_break_and_continue_belong_to_outer_loop(self):
        for inner in ('for j in []:', 'while False:'):
            body = 'w = 0\nfor i in range(3):\n    w += 1\n    '+inner+'\n        pass\n    else:\n        break\nelse:\n    w = 9\nreturn x*w'
            self.assertEqual(self.run_source(body)[0], (1,2,3,4))
            body = 'w = 0\nfor i in range(3):\n    w += 1\n    '+inner+'\n        pass\n    else:\n        continue\n    w = 9\nreturn x*w'
            self.assertEqual(self.run_source(body)[0], (3,6,9,12))

    def test_return_unwinds_loops_without_executing_else(self):
        helpers = 'def first(a):\n    while True:\n        for i in range(3):\n            return a\n        else:\n            return -a\n    else:\n        return -a\n'
        self.assertEqual(self.run_source('return first(x)',helpers)[0], (1,2,3,4))

    def test_comparisons_against_native_python_matrix(self):
        values = (None,False,True,-1,.5,'','a',[],[1],[1,2],(),(1,),{}, {0:1})
        operations = (('==',operator.eq),('!=',operator.ne),('<',operator.lt),
                      ('<=',operator.le),('>',operator.gt),('>=',operator.ge))
        for left,right,(symbol,operation) in itertools.product(values,values,operations):
            source = 'return x if '+repr(left)+' '+symbol+' '+repr(right)+' else -x'
            with self.subTest(left=left,right=right,operator=symbol):
                try:
                    expected = operation(left,right)
                except TypeError:
                    with self.assertRaises(ValueError): self.run_source(source)
                else:
                    self.assertEqual(self.run_source(source)[0],
                                     (1,2,3,4) if expected else (-1,-2,-3,-4))

    def test_loop_behavior_against_native_python_matrix(self):
        for count,stop,skip,kind in itertools.product(range(5),range(5),range(5),('for','while')):
            coefficient = 1
            for i in range(count):
                if i == skip: continue
                if i == stop: break
                coefficient += i
            else:
                coefficient += 10
            if kind == 'for':
                header = 'for i in range('+str(count)+'):\n'
                increment = ''
            else:
                header = 'i = -1\nwhile i < '+str(count-1)+':\n'
                increment = '    i += 1\n'
            body = 'w = 1\n'+header+increment+'    if i == '+str(skip)+':\n        continue\n'
            body += '    if i == '+str(stop)+':\n        break\n    w += i\nelse:\n    w += 10\nreturn x*w'
            with self.subTest(count=count,stop=stop,skip=skip,kind=kind):
                self.assertEqual(self.run_source(body)[0],tuple(coefficient*x for x in (1,2,3,4)))

    def test_cipher_dependence_invalid_keys_keyword_unpack_and_resources_rejected(self):
        bodies = ('return x if x else -x','return x and x','return x or x',
            'return x if x == x else -x','return x if x in [x] else -x',
            'd = {x:1}\nreturn x','d = {(1,x):1}\nreturn x','d = {[1]:2}\nreturn x',
            'return x if np.array([1]) else -x','while True:\n    pass\nreturn x',
            'd = {}\nd[0] = d\nreturn x', 'break\nreturn x',
            'if False:\n    continue\nreturn x')
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError): self.run_source(body)
        with self.assertRaises(ValueError):
            self.run_source('return f(**{0:x})','def f(**kw):\n    return kw[0]\n')
        with self.assertRaises(ValueError):
            self.run_source('for i in range(1):\n    f()\nreturn x','def f():\n    break\n')

    def test_versioned_request_cli_and_coverage(self):
        from candidate_contract import make_request, validate_candidate, valid_semantic_guidance
        from run_candidate import parse_args, forward_options
        from dsl_grammar_coverage import analyze_source
        payload = dict(fx_graph='x',public_constants={},constant_origins={},
            layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        request = make_request(payload,dict(schema=2,id='control-unit'),'a'*64,public_control=True)
        source = program('', 'return False or x')
        self.assertEqual(request['task'],'hecate-function-synthesis-v15')
        self.assertTrue(valid_semantic_guidance(request))
        checked = validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
        self.assertEqual(checked['contract'],'hecate-function-v14')
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v14')['public_construction']['schema'],9)
        self.assertIn('--public-control',forward_options(parse_args(['--case','case.json','--prepare','--public-control'])))

    def test_previous_versions_reject_control_and_numeric_mapping_keys(self):
        for body in ('while False:\n    pass\nreturn x','return False or x','d = {1:x}\nreturn d[1]'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                normalize(program('',body),{},public_numbers=True)


class ControlGoldenPlainTests(unittest.TestCase):
    def test_goldens_against_independent_formula(self):
        from run_public_control_goldens import PLANS
        root = Path(__file__).resolve().parent
        weights = [[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            constants = (dict(c0=weights[0],c1=weights[1]) if case == 'construction-linear' else
                         {} if case == 'construction-chebyshev3' else dict(c0=[.5],c1=[.375]))
            source = (root/'golden_cases/public_control'/(golden+'.py')).read_text()
            expanded = normalize(source,constants,2 if case == 'construction-linear' else 1,public_control=True)
            detected = False
            for x in ([0.]*4,[.5,-1.,.25,-.75],[-1.,1.,-1.,1.]):
                expected = ([sum(a*b for a,b in zip(x,row)) for row in weights] if case == 'construction-linear' else
                            [4*v**3-3*v for v in x] if case == 'construction-chebyshev3' else [1.5*v+.375 for v in x])
                value = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed(x))
                actual = [v.values[0] for v in value] if type(value) is list else value.values
                equal = len(actual) == len(expected) and all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
                detected |= not equal
                if not wrong: self.assertTrue(equal,(golden,actual,expected))
            if wrong: self.assertTrue(detected,golden)


@unittest.skipUnless(os.environ.get('POSEIDON_CONTROL_REPORT'),'requires actual public-control CPU evidence')
class ControlEvidenceTests(unittest.TestCase):
    def test_actual_artifacts_and_frozen_reference(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_public_control_goldens import PLANS
        # Actual producer identities, retained for future compatibility audits.
        # The shared auditor still reproduces exact source/metadata and checks
        # every original artifact hash; this is not a claim of new execution.
        producers = {
            'function_construction.py':'1dfd07d44eb566a78b1511f2a7f6bd15153133d6ffe25bc4046193cd48ab75d7',
            'public_numeric.py':'9cc7895f04ac8fd921e5899f8102a2767ccefe0c19a8dae8f558ad7452eff780',
            'lexical_scope.py':'5f80b879132ca2974a20a48610d13f2037597e1c4aa847fe29037f464f4a7c70',
            'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539'}
        ClosureEvidenceTests.verify_evidence(self,os.environ['POSEIDON_CONTROL_REPORT'],PLANS,
            'public_control','hecate-function-synthesis-v15',dict(public_control=True),producers)


if __name__ == '__main__':
    unittest.main()
