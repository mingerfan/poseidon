"""Public polynomial data and exact upstream GenPoly construction tests."""
import ast
import itertools
import importlib.util
import os
from pathlib import Path
import unittest
import public_numeric as numeric
import public_polynomial as poly
from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


def upstream_helpers():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root/'third_party/dacapo/python/poly/poly/MPCB.py').read_text())
    selected = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('fint','GenPoly')]
    if len(selected) != 2:
        raise ValueError('Pinned upstream helper definitions missing')
    return ast.unparse(ast.Module(body=selected,type_ignores=[]))


def evaluate_chebyshev(coefficients,x):
    # Independent recurrence; does not call NumPy polynomial evaluation.
    previous,current = 1.,x
    result = coefficients[0]
    if len(coefficients)>1: result += coefficients[1]*x
    for coefficient in coefficients[2:]:
        previous,current = current,2*x*current-previous
        result += coefficient*current
    return result


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class PolynomialTests(unittest.TestCase):
    def test_arithmetic_and_division_reconstruction(self):
        coefficients = ([0],[1],[-.5,.25],[.25,-.5,.125],[0,.25,0,.5],[1,0,0,0,.25])
        for a,b in itertools.product(coefficients,repeat=2):
            left,right = poly.create(a),poly.create(b)
            for op in (ast.Add,ast.Sub,ast.Mult):
                out = poly.binary(op(),left,right)
                for x in (-.75,-.125,0.,.5,1.):
                    av,bv = evaluate_chebyshev(a,x),evaluate_chebyshev(b,x)
                    expected = av+bv if op is ast.Add else av-bv if op is ast.Sub else av*bv
                    self.assertAlmostEqual(evaluate_chebyshev(out.coefficients,x),expected,places=12)
            if any(b):
                quotient = poly.binary(ast.FloorDiv(),left,right)
                remainder = poly.binary(ast.Mod(),left,right)
                degree = max(i for i,c in enumerate(b) if c)
                self.assertTrue(len(remainder.coefficients)-1 < degree or not any(remainder.coefficients))
                for x in (-.75,-.125,0.,.5,1.):
                    reconstructed = (evaluate_chebyshev(quotient.coefficients,x)*evaluate_chebyshev(b,x)
                                     +evaluate_chebyshev(remainder.coefficients,x))
                    self.assertAlmostEqual(reconstructed,evaluate_chebyshev(a,x),places=11)

    def test_preserved_constructor_metadata_and_trailing_coefficients(self):
        p = poly.create([1,0,0],domain=[0,2],window=[-2,2],symbol='t')
        self.assertEqual((p.coefficients,p.domain,p.window,p.symbol),((1.,0.,0.),(0.,2.),(-2.,2.),'t'))
        with self.assertRaises(ValueError): poly.binary(ast.Add(),p,poly.create([1]))
        self.assertEqual(poly.binary(ast.Add(),p,2).coefficients,(3.,))
        self.assertEqual(poly.binary(ast.Pow(),poly.create([0,1]),2).coefficients,(.5,0.,.5))

    def test_domains_resource_bounds_and_unsupported_operations(self):
        for data in ([],[[1,2]],[0]*129,[float('nan')],[1e100]):
            with self.assertRaises(ValueError): poly.create(data)
        for op,right in ((ast.FloorDiv,0),(ast.Mod,0),(ast.Div,2),(ast.Pow,-1),(ast.Pow,1.5),(ast.Pow,17)):
            with self.subTest(op=op),self.assertRaises(ValueError):
                poly.binary(op(),poly.create([0,1]),right)
        with self.assertRaises(ValueError):
            poly.binary(ast.Mult(),poly.create([1]*128),poly.create([1,1]))
        for value in (0,-1):
            with self.assertRaises(ValueError): poly.ufunc('log2',value)
        self.assertEqual(poly.ufunc('floor',-.25),-1.)
        self.assertEqual(poly.ufunc('ceil',-.25),0.)
        self.assertEqual(poly.ufunc('log2',8),3.)
        self.assertEqual(poly.ufunc('ceil',[[.5,-.5]]),numeric.Array((1,2),(1.,-0.),True))


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class GenPolyConstructionTests(unittest.TestCase):
    def run_source(self,body,helpers=''):
        expanded = normalize(program(helpers,body),{},public_polynomial=True)
        value = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},
                              Packed([.5,-1.,.25,-.75]))
        return value.values,expanded

    def test_exact_upstream_function_leaf_and_decomposed_tree(self):
        for tree in (['0'],['2','0 0']):
            body = 'fn = GenPoly('+repr(tree)+',["0","0.25","0","0.5"],4)\nreturn fn(x)'
            values,expanded = self.run_source(body,upstream_helpers())
            self.assertEqual(values,tuple(2*x**3-1.25*x for x in (.5,-1.,.25,-.75)))
            self.assertGreater(expanded['construction']['public_polynomial_operations'],0)
            self.assertGreater(expanded['construction']['public_numpy_functions'],0)

    def test_upstream_seventh_degree_and_scale(self):
        body = 'fn = GenPoly(["0"],["0","0.25","0","0.5","0","0.25","0","0.125"],8,scale=2)\nreturn fn(x)'
        values,_ = self.run_source(body,upstream_helpers())
        self.assertEqual(values,tuple(4*x**7-5*x**5+2*x**3-.4375*x for x in (.5,-1.,.25,-.75)))

    def test_upstream_odd_leaf_limitation_is_not_silently_repaired(self):
        body = 'fn = GenPoly(["0"],["0","0.25","0.25","0.5"],4)\nreturn fn(x)'
        values,_ = self.run_source(body,upstream_helpers())
        points = (.5,-1.,.25,-.75)
        odd_only = tuple(2*x**3-1.25*x for x in points)
        full_model = tuple(2*x**3-1.25*x+.25*(2*x*x-1) for x in points)
        self.assertEqual(values,odd_only)
        self.assertNotEqual(values,full_model)

    def test_upstream_builds_missing_giant_degree(self):
        body = 'fn = GenPoly(["4","0 0"],["0","0.25","0","0.5","0","0.125"],4)\nreturn fn(x)'
        values,_ = self.run_source(body,upstream_helpers())
        self.assertEqual(values,tuple(2*x**5-.5*x**3-.625*x for x in (.5,-1.,.25,-.75)))

    def test_dtype_constructor_keywords_and_coefficient_reads(self):
        body = 'p = np.polynomial.Chebyshev(coef=np.array([0,1],dtype=np.float64))\n'
        body += 'q = p*p\nw = q.coef[0]+q.coef[2]\nreturn x*w'
        self.assertEqual(self.run_source(body)[0],(.5,-1.,.25,-.75))

    def test_named_public_constant_keyword_matches_positional_constructor(self):
        constants = {'c0':[0.,1.,0.,0.]}
        sources = [program('','p = np.polynomial.Chebyshev('+arg+')\nreturn x*p.coef[1]')
                   for arg in ('c0','coef=c0')]
        expanded = [normalize(source,constants,public_polynomial=True) for source in sources]
        self.assertEqual(expanded[0]['source'],expanded[1]['source'])
        self.assertEqual(expanded[0]['derived_constants'],expanded[1]['derived_constants'])
        self.assertEqual(constants,{'c0':[0.,1.,0.,0.]})

    def test_polynomial_arithmetic_reads_named_public_operands(self):
        constants = {'c0':[.5]}
        for expression in ('p+c0','c0+p'):
            source = program('','p = np.polynomial.Chebyshev([0,1])\nq = '+expression+'\nreturn x*q.coef[0]+q.coef[1]')
            expanded = normalize(source,constants,public_polynomial=True)
            values = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},
                                   Packed([1,2,3,4])).values
            self.assertEqual(values,(1.5,2,2.5,3))

    def test_cipher_and_capability_boundaries(self):
        bodies = ('p = np.polynomial.Chebyshev([x])\nreturn x',
                  'p = np.polynomial.Chebyshev([0,1])\nreturn p*x',
                  'return x*np.log2(x)','return x*np.load("secret")',
                  'p = np.polynomial.Chebyshev([1])\np.coef[0] = 2\nreturn x',
                  'p = np.polynomial.Chebyshev([1])\nreturn p(x)',
                  'return x*np.polynomial.Polynomial([1])',
                  'p = np.polynomial.Chebyshev([1])\nreturn x if p else -x')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError): self.run_source(body)
        with self.assertRaises(ValueError):
            normalize(program('','return x*np.floor(.5)'),{},public_strings=True)

    def test_versioned_request_candidate_cli_and_batch(self):
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance
        from run_candidate import parse_args,forward_options
        from dsl_grammar_coverage import analyze_source
        from run_agent_batch import main,construction_options,validate_construction_continuation
        from types import SimpleNamespace
        from unittest.mock import patch
        import io,contextlib,json
        payload = dict(fx_graph='x',public_constants={},constant_origins={},
            layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        request = make_request(payload,dict(schema=2,id='poly-unit'),'a'*64,public_polynomial=True)
        source = program('','p = np.polynomial.Chebyshev([0,1])\nreturn x*p.coef[1]')
        self.assertEqual(request['task'],'hecate-function-synthesis-v17')
        self.assertTrue(valid_semantic_guidance(request))
        self.assertEqual(validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)['contract'],'hecate-function-v16')
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v16')['public_construction']['schema'],11)
        self.assertIn('--public-polynomial',forward_options(parse_args(['--case','case.json','--prepare','--public-polynomial'])))
        args = SimpleNamespace(public_polynomial=True)
        self.assertEqual(construction_options(args),['--public-polynomial'])
        validate_construction_continuation({'public_polynomial':True},args)
        with self.assertRaisesRegex(ValueError,'fresh batch'): validate_construction_continuation({},args)
        output = io.StringIO()
        with patch('sys.argv',['run_agent_batch.py','--plan','--public-polynomial']), \
             patch('agent_credentials.load_api_key',side_effect=AssertionError('credential access')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(),0)
        report = json.loads(output.getvalue())
        self.assertTrue(report['public_polynomial'])
        self.assertEqual(report['agent_calls'],0)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class PolynomialGoldenPlainTests(unittest.TestCase):
    def test_upstream_sources_and_independent_formulas(self):
        from run_public_polynomial_goldens import PLANS
        root = Path(__file__).resolve().parent
        original = ast.parse(upstream_helpers()).body
        for case,golden,wrong in PLANS:
            source = (root/'golden_cases/public_polynomial'/(golden+'.py')).read_text()
            functions = ast.parse(source).body
            self.assertEqual([ast.dump(n,include_attributes=False) for n in functions[:-1]],
                             [ast.dump(n,include_attributes=False) for n in original])
            expanded = normalize(source,{},public_polynomial=True)
            detected = False
            for x in ([0.]*4,[.5,-1.,.25,-.75],[-1.,1.,-1.,1.]):
                expected = [2*v**5-.5*v**3-.625*v if case == 'construction-genpoly5' else
                            4*v**7-5*v**5+2*v**3-.4375*v if case == 'construction-genpoly7' else
                            2*v**3-1.25*v+(.5*v*v-.25 if case == 'construction-genpoly-even' else 0.) for v in x]
                actual = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed(x)).values
                equal = all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
                if not wrong: self.assertTrue(equal,(golden,actual,expected))
                detected |= not equal
            if wrong: self.assertTrue(detected,golden)


@unittest.skipUnless(os.environ.get('POSEIDON_POLYNOMIAL_REPORT'),'requires actual GenPoly CPU evidence')
class PolynomialEvidenceTests(unittest.TestCase):
    def test_real_artifacts_upstream_functions_and_reference(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_public_polynomial_goldens import PLANS
        import numpy as np
        self.assertEqual(np.__version__,'1.25.2')
        # Actual batch producer precedes a named-keyword public-data fix.
        # Reproduce all original expansions exactly and retain historical hashes.
        producers = {
            'function_construction.py':(
                '83ef4e93abcef6a94c3edcacc6c7547bb052561d990dca270749d7add2a8aff0',
                '633799d8f7a3aab07efbd6e3d2749db726501d87b58ac7ad8ec954e2ddb22c36'),
            'public_polynomial.py':'0aa3e3d4bd5658205da48468eb29022b7c710effa0670519f9eaf650c44a1257',
            'public_strings.py':'244dee49f23786b48a43c8e27c9d629a6d5a55a3ba8be7e85410f5a96cc3445c',
            'public_numeric.py':'9cc7895f04ac8fd921e5899f8102a2767ccefe0c19a8dae8f558ad7452eff780',
            'lexical_scope.py':'5f80b879132ca2974a20a48610d13f2037597e1c4aa847fe29037f464f4a7c70',
            'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539'}
        import json
        evidence = os.environ['POSEIDON_POLYNOMIAL_REPORT']
        if evidence.startswith('['):
            evidence = json.loads(evidence)
        ClosureEvidenceTests.verify_evidence(self,evidence,PLANS,
            'public_polynomial','hecate-function-synthesis-v17',dict(public_polynomial=True),producers)


if __name__ == '__main__':
    unittest.main()
