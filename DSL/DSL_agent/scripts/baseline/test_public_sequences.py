"""Public sequence semantics, independently checked against native Python data."""
import hashlib
import itertools
import os
from pathlib import Path
import unittest
from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


class PublicSequenceTests(unittest.TestCase):
    def run_source(self, helpers, body, constants=None, outputs=1):
        constants = {} if constants is None else constants
        result = normalize(program(helpers, body), constants, outputs, public_sequences=True)
        value = evaluate_tree(result['source'], {k: Packed(v) for k,v in constants.items()}, Packed([1,2,3,4]))
        return value, result

    def assert_public_result(self, body, expected):
        # Only public integer comparisons; an incorrect public result changes
        # the returned symbolic ciphertext, independently of the normalizer.
        choice = 'x'
        for index, value in reversed(list(enumerate(expected))):
            choice = '('+choice+' if items['+str(index)+'] == '+str(value)+' else -x)'
        choice = '('+choice+' if len(items) == '+str(len(expected))+' else -x)'
        actual,_ = self.run_source('', body+'\nreturn '+choice)
        self.assertEqual(actual.values, (1,2,3,4))

    def test_slice_bounds_against_native_python(self):
        bounds = (None, -7, -1, 0, 2, 7)
        for start,stop,step in itertools.product(bounds, bounds, (None,-3,-1,1,2,4)):
            with self.subTest(start=start,stop=stop,step=step):
                notation = ':'.join('' if v is None else str(v) for v in (start,stop,step))
                expected = [0,1,2,3][slice(start,stop,step)]
                self.assert_public_result('items = [0,1,2,3]['+notation+']', expected)

    def test_slice_assignment_against_native_python(self):
        for start,stop,step in itertools.product((None,-6,-1,0,3,6), (None,-6,-1,0,3,6), (None,-2,-1,1,2)):
            notation = ':'.join('' if v is None else str(v) for v in (start,stop,step))
            body = 'items = [0,1,2,3]\nitems['+notation+'] = [9,8]'
            expected = [0,1,2,3]
            try:
                expected[slice(start,stop,step)] = [9,8]
            except ValueError:
                with self.assertRaisesRegex(ValueError, 'slice assignment length'):
                    self.run_source('', body+'\nreturn x')
            else:
                self.assert_public_result(body, expected)

    def test_nested_aliases_preserved_by_slice_concat_repeat(self):
        for expression in ('items[:]', 'items + []', '2 * items', 'items * 2'):
            with self.subTest(expression=expression):
                actual,_ = self.run_source('', 'items = [[x]]\nother = '+expression+'\nother[0][0] = -x\nreturn items[0][0]')
                self.assertEqual(actual.values, (-1,-2,-3,-4))

    def test_slice_is_not_outer_list_alias(self):
        actual,_ = self.run_source('', 'items = [x]\nother = items[:]\nother[0] = -x\nreturn items[0]')
        self.assertEqual(actual.values, (1,2,3,4))

    def test_slice_self_assignment_and_iterator_snapshot(self):
        self.assert_public_result('items = [0,1,2,3]\nitems[::-1] = iter(items)', [3,2,1,0])
        self.assert_public_result('items = [0,1]\nitems[1:1] = items', [0,0,1,1])

    def test_rhs_target_bounds_evaluation_order(self):
        helpers = ('def rhs(log,a):\n    log.append(0)\n    return [a]\n'
                   'def target(log,items):\n    log.append(1)\n    return items\n'
                   'def bound(log,n):\n    log.append(n)\n    return 0\n')
        value,_ = self.run_source(helpers, 'log = []\nitems = [x]\ntarget(log,items)[bound(log,2):bound(log,3)] = rhs(log,-x)\nreturn items[log[0]] if log[1] == 1 else x')
        self.assertEqual(value.values, (-1,-2,-3,-4))
        value,_ = self.run_source(helpers, 'log = []\nitems = [x]\ntarget(log,items)[bound(log,2):bound(log,3)] = rhs(log,-x)\nreturn x if log[2] == 2 else -x')
        self.assertEqual(value.values, (1,2,3,4))

    def test_named_inplace_aliasing_vs_rebinding(self):
        for action,expected in (('items += [1]',[0,1]), ('items *= 2',[0,0]),
                                ('items = items + [1]',[0]), ('items = items * 2',[0])):
            self.assert_public_result('items = [0]\nalias = items\n'+action+'\nitems = alias', expected)

    def test_inplace_self_extension_and_iterator_alias(self):
        self.assert_public_result('items = [0,1]\nitems += items', [0,1,0,1])
        with self.assertRaisesRegex(ValueError, 'length limit'):
            self.run_source('', 'items = [x]\nitems += iter(items)\nreturn x')

    def test_tuple_string_and_bool_sequence_indices(self):
        self.assert_public_result('items = (0,1,2)[::-1] + (3,)', [2,1,0,3])
        value,_ = self.run_source('', 'key = ("abc" + "de")[::-2]\nmap = {"eca":x}\nreturn map[key]')
        self.assertEqual(value.values, (1,2,3,4))
        value,_ = self.run_source('', 'keys = list(reversed("ab"*True))\nmap = {"b":x,"a":-x}\nreturn map[keys[False]]')
        self.assertEqual(value.values, (1,2,3,4))

    def test_nonpositive_repeat(self):
        for count in (-3, 0, False):
            self.assert_public_result('items = [0,1] * '+repr(count), [])

    def test_invalid_types_private_indexing_and_budgets(self):
        bodies = ('items = [x][::0]', 'items = [x][x:]', 'items = x[:]', 'items = c0[:]',
                  'items = [x] * x', 'items = [x] * 129', 'items = [x]*128 + [x]',
                  'items = "a" * 129', 'items = [x] + (x,)', 'items = [x][9]',
                  'items = (x,)\nitems[:] = [x]', 'items = "a"\nitems[:] = "b"',
                  'items = [x]*128\nitems[0:0] = [x]', 'items = [x]\nitems[::2] = [x,x]',
                  'items = [x]\nitems[:] = [items]', 'items = [x]\nitems += [items]',
                  'items = [x]\nitems[0] += x')
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.run_source('', body+'\nreturn x', {'c0':.5})

    def test_version_cli_coverage_and_legacy_rejection(self):
        from candidate_contract import make_request, validate_candidate, valid_semantic_guidance
        from hecate_contract import validate_function
        from run_candidate import parse_args, forward_options
        from dsl_grammar_coverage import analyze_source
        payload = dict(fx_graph='x',public_constants={},constant_origins={},layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        req = make_request(payload,dict(schema=2,id='unit'),'a'*64,public_sequences=True)
        source = program('', 'return [x,-x][::-1][0]')
        self.assertEqual(validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['contract'], 'hecate-function-v12')
        self.assertTrue(valid_semantic_guidance(req))
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v12')['public_construction']['sequence_slices'],1)
        self.assertIn('--public-sequences',forward_options(parse_args(['--case','case.json','--prepare','--public-sequences'])))
        for version in range(12):
            with self.assertRaises(ValueError):
                validate_function(source,{},contract='hecate-function-v'+str(version))


class SequenceGoldenPlainTests(unittest.TestCase):
    def test_goldens_against_independent_formula(self):
        from run_public_sequence_goldens import PLANS
        root=Path(__file__).resolve().parent
        weights=[[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            constants=dict(c0=weights[0],c1=weights[1]) if case=='construction-linear' else dict(c0=[.5],c1=[.375])
            source=(root/'golden_cases/public_sequences'/(golden+'.py')).read_text()
            expanded=normalize(source,constants,2 if case=='construction-linear' else 1,public_sequences=True)
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


@unittest.skipUnless(os.environ.get('POSEIDON_SEQUENCE_REPORT'),'requires actual sequence CPU evidence')
class SequenceEvidenceTests(unittest.TestCase):
    def test_real_artifacts_and_independent_reference(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_public_sequence_goldens import PLANS
        root=Path(__file__).resolve().parent
        current={'function_construction.py':'147b1e25d47620746cf9e32be158fa8a1978c0d013ac38e5cf0010e602f0e7f0',
                 'lexical_scope.py':'5f80b879132ca2974a20a48610d13f2037597e1c4aa847fe29037f464f4a7c70',
                 'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539'}
        ClosureEvidenceTests.verify_evidence(self,os.environ['POSEIDON_SEQUENCE_REPORT'],PLANS,
            'public_sequences','hecate-function-synthesis-v13',dict(public_sequences=True),current)
