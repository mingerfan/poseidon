"""Independent public-number/array tests; preflight is not FHE execution."""
import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import unittest
import public_numeric as numeric
from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


class NumericValueTests(unittest.TestCase):
    def test_scalar_operators(self):
        for a,b in itertools.product((-3,-.5,0,.25,2),repeat=2):
            for op,fn in numeric.OPS.items():
                if op in (ast.Div,ast.FloorDiv,ast.Mod) and b == 0: continue
                if op is ast.Pow and (a < 0 and type(b) is float or a == 0 and b < 0): continue
                expected=fn(a,b)
                if type(expected) in (int,float) and abs(expected) <= numeric.LIMIT:
                    self.assertEqual(numeric.scalar_binary(op(),a,b),expected)

    def test_broadcast_index_shape_and_scalar_array(self):
        result=numeric.binary(ast.Mult(),numeric.array([[1],[2]]),numeric.array([.5,1.5]))
        self.assertEqual((result.shape,result.values),((2,2),(.5,1.5,1.,3.)))
        self.assertEqual(numeric.index(result,(slice(None,None,-1),0)).values,(1.,.5))
        self.assertEqual(numeric.index(result,(1,1)),3.)
        self.assertEqual(numeric.reshape(result,(-1,)).shape,(4,))
        self.assertEqual(numeric.binary(ast.Add(),numeric.array([]),numeric.array([1])).shape,(0,))
        a=numeric.array(2,floating=True)
        self.assertIs(type(numeric.index(a,())),float)
        self.assertEqual(numeric.encoding(a),[2.])

    def test_empty_array_dtype_is_not_inferred_from_missing_values(self):
        floating=numeric.array([])
        self.assertTrue(floating.floating)
        self.assertEqual(numeric.binary(ast.Pow(),floating,-1).values,())
        integers=numeric.index(numeric.array([1]),slice(0,0))
        self.assertFalse(numeric.array(integers).floating)
        with self.assertRaises(ValueError): numeric.binary(ast.Pow(),integers,-1)
        promoted=numeric.binary(ast.Add(),integers,.5)
        self.assertTrue(numeric.reshape(promoted,(0,1)).floating)

    def test_invalid_domains_shapes_and_resource_budgets(self):
        for value in (float('inf'),float('nan'),1j,10**400):
            with self.assertRaises(ValueError): numeric.number(value)
        for op,a,b in ((ast.Div,1,0),(ast.Mod,1,0),(ast.Pow,-1,.5),(ast.Pow,2,17)):
            with self.assertRaises(ValueError): numeric.scalar_binary(op(),a,b)
        for data in ([[1],[2,3]],[0]*129,[[[[[1]]]]],[True],[[0]*128]*128):
            with self.assertRaises(ValueError): numeric.array(data)
        with self.assertRaises(ValueError): numeric.binary(ast.Add(),numeric.array([1,2]),numeric.array([1,2,3]))
        with self.assertRaises(ValueError): numeric.binary(ast.Pow(),numeric.array([2]),-1)
        with self.assertRaises(ValueError): numeric.encoding(numeric.array([[1,2],[3,4]]))
        with self.assertRaises(ValueError): numeric.encoding(1025)
        with self.assertRaises(ValueError): numeric.reshape(numeric.array([]),(0,-1))


class NumericConstructionTests(unittest.TestCase):
    def run_source(self, body, constants=None, helpers='', outputs=1):
        constants={} if constants is None else constants
        original=copy.deepcopy(constants)
        result=normalize(program(helpers,body),constants,outputs,public_numbers=True)
        self.assertEqual(constants,original)
        self.assertEqual({k:result['constants'][k] for k in constants},original)
        value=evaluate_tree(result['source'],{k:Packed(v) for k,v in result['constants'].items()},Packed([1,2,3,4]))
        return value,result

    def test_scalar_literals_and_derived_manifest(self):
        value,result=self.run_source('weight = float(" 1.5 ")\nbias = 3 / 8\nreturn weight * x + bias')
        self.assertEqual(value.values,(1.875,3.375,4.875,6.375))
        self.assertEqual(list(result['derived_constants'].values()),[1.5,.375])
        digest=hashlib.sha256(json.dumps(result['derived_constants'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.assertEqual(result['construction']['derived_constants_sha256'],digest)
        self.assertFalse(result['construction']['input_constants_changed'])

    def test_array_broadcast_explicit_flatten(self):
        value,_=self.run_source('a = np.array([[1.0],[2.0]])\nb = np.asarray([0.25,0.5])\nw = (a*b).flatten()\nreturn w * x')
        self.assertEqual(value.values,(.25,1.,1.5,4.))

    def test_original_constants_readonly_float64_arrays(self):
        value,_=self.run_source('w = c0 / 2\nb = c1[0]\nreturn x * w + b',{'c0':[1.,1.,1.,1.],'c1':.375})
        self.assertEqual(value.values,(.875,1.375,1.875,2.375))
        value,_=self.run_source('return x if c0.shape[0] == 1 else -x',{'c0':.5})
        self.assertEqual(value.values,(1,2,3,4))

    def test_arrays_and_sequences_distinct(self):
        value,_=self.run_source('a = np.array([1,2]) * [2,3]\nb = [1,2] * 2\nreturn x * [a[0],a[1],b[2],b[3]]')
        self.assertEqual(value.values,(2,12,3,8))

    def test_iteration_slice_reshape_dtype(self):
        value,_=self.run_source('a = np.array([[1,2],[3,4]],dtype="float64")\nb = [row[0] for row in reversed(a)]\nc = a[:,::-1].reshape(-1)\nreturn x*c')
        self.assertEqual(value.values,(2,2,12,12))

    def test_numeric_branch_pow_conversion(self):
        value,_=self.run_source('n = int("2")\nw = pow(2,n) / 8\nreturn x*w if -0.5 < 0 else -x')
        self.assertEqual(value.values,(.5,1,1.5,2))

    def test_derived_name_collision_and_dedup(self):
        value,result=self.run_source('derived0 = x\na = x * 0.5\nb = a * 0.5\nreturn b')
        self.assertEqual(value.values,(.25,.5,.75,1))
        self.assertEqual(result['derived_constants'],{'derived1':.5})

    def test_private_values_writes_external_capabilities_and_layout_rejected(self):
        bodies=('return x/2','return x**2','return x * np.array([[1,2],[3,4]])',
                'return x * np.array([1,2])','return x * np.array([x])','return x * float(x)',
                'return x * float("nan")','return x * 1e300','return x * (1/0)',
                'return x * np.load("secret")','return np.array([1]).__class__',
                'a = np.array([1])\na[0] = 2\nreturn x','a = c0\na[0] = 2\nreturn x',
                'a = np.array([1])\na += 1\nreturn x*a',
                'a = c0\na *= 2\nreturn x*a',
                'return x if np.array([1]) else -x','return x * np.array([1],dtype="object")',
                'return x * np.array([1]).reshape(2)','a = [[0]*128]*128\nreturn x * np.array(a)',
                'a = [[[[]]*128]*128]*128\nreturn x*np.array(a)')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError): self.run_source(body,{'c0':.5})

    def test_version_cli_coverage_old_rejection(self):
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance
        from run_candidate import parse_args,forward_options
        from hecate_contract import validate_function
        from dsl_grammar_coverage import analyze_source
        payload=dict(fx_graph='x',public_constants={},constant_origins={},layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        req=make_request(payload,dict(schema=2,id='unit'),'a'*64,public_numbers=True)
        source=program('','return x * (3/2)')
        self.assertEqual(validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['contract'],'hecate-function-v13')
        self.assertTrue(valid_semantic_guidance(req))
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v13')['public_construction']['derived_constant_count'],1)
        self.assertIn('--public-numbers',forward_options(parse_args(['--case','case.json','--prepare','--public-numbers'])))
        for version in range(13):
            with self.assertRaises(ValueError): validate_function(source,{},contract='hecate-function-v'+str(version))


class NumericNumpyTests(unittest.TestCase):
    def test_array_results_against_actual_numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('requires pinned NumPy environment')
        arrays=(np.array(2.),np.array([.25,2.]),np.array([[1.],[2.]]),np.array([[.5,1.],[2.,3.]]),np.empty((0,1)))
        for a,b in itertools.product(arrays,repeat=2):
            for op,fn in numeric.OPS.items():
                with self.subTest(a=a.shape,b=b.shape,op=op.__name__):
                    # tolist() discards trailing dimensions of empty arrays;
                    # both success and failure paths must preserve exact shapes.
                    aa=numeric.Array(a.shape,tuple(float(v) for v in a.flatten()),a.dtype.kind == 'f')
                    bb=numeric.Array(b.shape,tuple(float(v) for v in b.flatten()),b.dtype.kind == 'f')
                    try: expected=fn(a,b)
                    except ValueError:
                        with self.assertRaises(ValueError): numeric.binary(op(),aa,bb)
                        continue
                    actual=numeric.binary(op(),aa,bb)
                    self.assertEqual(actual.shape,expected.shape)
                    np.testing.assert_allclose(actual.values,expected.flatten(),rtol=1e-14,atol=1e-14)
        a=np.arange(24.).reshape(2,3,4)
        data=numeric.array(a.tolist())
        for key in ((1,slice(None),slice(None,None,-1)),(slice(None),1,2),(0,),()):
            actual=numeric.index(data,key)
            expected=a[key]
            self.assertEqual(actual.shape,expected.shape)
            np.testing.assert_array_equal(actual.values,expected.flatten())


class NumericGoldenPlainTests(unittest.TestCase):
    def test_goldens_against_independent_formula(self):
        from run_public_number_goldens import PLANS
        root=Path(__file__).resolve().parent
        weights=[[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            constants=dict(c0=weights[0],c1=weights[1]) if case=='construction-linear' else dict(c0=[.5],c1=[.375])
            source=(root/'golden_cases/public_numbers'/(golden+'.py')).read_text()
            expanded=normalize(source,constants,2 if case=='construction-linear' else 1,public_numbers=True)
            detected=False
            for x in ([0.]*4,[.5,-1.,.25,-.75],[-1.,1.,-1.,1.]):
                expected=[sum(a*b for a,b in zip(x,row)) for row in weights] if case=='construction-linear' else [1.5*v+.375 for v in x]
                value=evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed(x))
                actual=[v.values[0] for v in value] if type(value) is list else value.values
                equal=len(actual)==len(expected) and all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
                detected |= not equal
                if not wrong: self.assertTrue(equal,(golden,actual,expected))
            if wrong: self.assertTrue(detected,golden)


@unittest.skipUnless(os.environ.get('POSEIDON_NUMBER_REPORT'),'requires actual public-numeric CPU evidence')
class NumericEvidenceTests(unittest.TestCase):
    def test_real_artifacts_and_derived_manifest(self):
        from test_closure_construction import ClosureEvidenceTests
        from run_public_number_goldens import PLANS
        root=Path(__file__).resolve().parent
        # Actual producer before the separate augmented-array rejection guard.
        # Reproduce all original expansions exactly; never relabel past execution.
        current={'function_construction.py':'d2b6dd9102b40645784d3b1dc09eaaaa7d34d5600c578afe65483fa4e24c3490',
                 'lexical_scope.py':'5f80b879132ca2974a20a48610d13f2037597e1c4aa847fe29037f464f4a7c70',
                 'construction_calls.py':'5c9fc99056c364324be7796004a29e3e515c15991f95c656c589b5796d039539',
                 'public_numeric.py':'e1529a9f3db3a9ce38bb75107b7ad70a4e6d143b4f18a6d976ebc3c00085ef8c'}
        ClosureEvidenceTests.verify_evidence(self,os.environ['POSEIDON_NUMBER_REPORT'],PLANS,
            'public_numbers','hecate-function-synthesis-v14',dict(public_numbers=True),current)
