"""Object storage and trusted upstream dispatch oracles; no paid API or FHE claim."""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import object_arrays as objects
from function_construction import Value, normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed
from test_frontend_augmented_ops import frontend_double


ROOT = Path(__file__).resolve().parents[2]


def upstream_sumslots():
    tree = ast.parse((ROOT/'third_party/dacapo/python/poly/poly/MPCB.py').read_text())
    names = ('fint','roll','SumSlots')
    selected = [next(n for n in ast.walk(tree) if type(n) is ast.FunctionDef and n.name == name)
                for name in names]
    return ast.unparse(ast.Module(body=selected,type_ignores=[]))


def upstream_empty():
    """Execute ONLY trusted upstream definitions with a recording Plain boundary."""
    import numpy as np
    expr = frontend_double()
    namespace = expr.__add__.__globals__
    class Plain(expr):
        def __init__(self,data):
            self.data = data
            super().__init__(('plain',data))
    namespace.update(Expr=expr, Plain=Plain, np=np,
                     torch=SimpleNamespace(Tensor=type('Tensor',(),{})))
    tree = ast.parse((ROOT/'third_party/dacapo/python/hecate/hecate/expr.py').read_text())
    selected = [n for n in tree.body if
                type(n) is ast.ClassDef and n.name == 'Empty' or
                type(n) is ast.FunctionDef and n.name == 'resolveType']
    exec(compile(ast.Module(body=selected,type_ignores=[]),'trusted-upstream-empty','exec'),namespace)
    return namespace['Empty'], expr, Plain


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class ObjectStorageTests(unittest.TestCase):
    def test_full_identity_and_empty_none(self):
        sentinel = objects.Empty()
        array = objects.full((2,3),sentinel)
        self.assertEqual(array.shape,(2,3))
        self.assertTrue(all(v is sentinel for v in array.data.flat))
        self.assertTrue(all(v is None for v in objects.empty((2,3)).data.flat))
        self.assertEqual(objects.full((2,3),[1,2,3]).data.tolist(),[[1,2,3],[1,2,3]])

    def test_views_and_shallow_copies(self):
        x,y = Value('x','cipher'),Value('y','cipher')
        original = objects.array([[x,x],[x,x]])
        view = objects.get(original,(slice(None),0))
        duplicate = objects.copy(original)
        flattened = objects.flatten(original)
        same = objects.array(original,copy=False)
        reshaped = objects.reshape(original,(4,))
        objects.put(view,1,y)
        self.assertIs(objects.get(original,(1,0)),y)
        self.assertIs(objects.get(same,(1,0)),y)
        self.assertIs(objects.get(reshaped,2),y)
        self.assertIs(objects.get(duplicate,(1,0)),x)
        self.assertIs(objects.get(flattened,2),x)

    def test_numpy_index_and_overlapping_assignment_oracle(self):
        import numpy as np
        keys = (slice(None),slice(None,None,-1),slice(1,4),slice(0,4,2),slice(1,4,2))
        count = 0
        for destination in keys:
            for source in keys:
                native = np.array([1,2,3,4],dtype=object)
                checked = objects.array([1,2,3,4])
                if native[destination].size != native[source].size:
                    continue
                native[destination] = native[source]
                objects.put(checked,destination,objects.get(checked,source))
                self.assertEqual(checked.data.tolist(),native.tolist())
                count += 1
        self.assertEqual(count,9)

    def test_transpose_reshape_copy_follows_native_noncontiguous_layout(self):
        import numpy as np
        native = np.array([[1,2],[3,4]],dtype=object).T.reshape(4)
        original = objects.array([[1,2],[3,4]])
        checked = objects.reshape(objects.transpose(original),(4,))
        objects.put(checked,0,99)
        native[0] = 99
        self.assertEqual(checked.data.tolist(),native.tolist())
        self.assertEqual(original.data.tolist(),[[1,2],[3,4]])

    def test_iteration_observes_mutation_and_concatenate_copies(self):
        value = objects.array([1,2,3])
        iterator = objects.iterate(value)
        self.assertEqual(next(iterator),1)
        objects.put(value,1,9)
        self.assertEqual(next(iterator),9)
        combined = objects.concatenate([value,value])
        objects.put(value,0,8)
        self.assertEqual(combined.data.tolist(),[1,9,3,1,9,3])

    def test_resource_and_capability_boundaries(self):
        for dimensions in (129,(2,)*5,(-1,),True,(129,0)):
            with self.subTest(dimensions=dimensions),self.assertRaises(ValueError):
                objects.empty(dimensions)
        for value in ({},lambda:None,object(),[float('nan')],[[1],[2,3]],list(range(129))):
            with self.subTest(value=type(value)),self.assertRaises(ValueError):
                objects.array(value)
        cyclic = []; cyclic.append(cyclic)
        with self.assertRaises(ValueError): objects.array(cyclic)
        array = objects.array([1,2])
        for key in (True,[0],Value('x','cipher'),slice(None,None,0),10):
            with self.subTest(key=key),self.assertRaises(ValueError): objects.get(array,key)
        with self.assertRaises(ValueError): objects.put(array,0,lambda:None)
        with self.assertRaises(ValueError): objects.put(array,0,array)
        with self.assertRaises(ValueError): objects.put(array,0,[1,2])
        self.assertEqual(array.data.tolist(),[1,2])


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class EmptyOracleTests(unittest.TestCase):
    def test_upstream_identity_not_zero(self):
        Empty,Expr,Plain = upstream_empty()
        x = Expr('x')
        for op in (lambda e,x:e+x,lambda e,x:e-x):
            self.assertIs(op(Empty(),x),x)
        with contextlib.redirect_stdout(io.StringIO()):
            for op in (lambda e,x:x+e,lambda e,x:x-e,lambda e,x:e+e):
                with self.assertRaises(Exception): op(Empty(),x)
        for value in (2,-.5,True,[1,2,3,4]):
            for op in (lambda e,x:e+x,lambda e,x:e-x,lambda e,x:x+e,lambda e,x:x-e):
                self.assertIsInstance(op(Empty(),value),Plain)
        for op in (lambda e:e*2,lambda e:-e):
            with self.assertRaises(TypeError): op(Empty())

    def test_checked_empty_dispatch_matches_scalar_oracle(self):
        x = Value('x','cipher')
        for op in (ast.Add(),ast.Sub()):
            self.assertIs(objects.empty_binary(op,objects.Empty(),x,lambda v:v),x)
            with self.assertRaises(ValueError): objects.empty_binary(op,x,objects.Empty(),lambda v:v)
            with self.assertRaises(ValueError): objects.empty_binary(op,objects.Empty(),objects.Empty(),lambda v:v)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class ObjectConstructionTests(unittest.TestCase):
    def run_source(self,body,helpers=''):
        expanded = normalize(program(helpers,body),{},object_arrays=True)
        result = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},
                               Packed([.5,-1.,.25,-.75]))
        return result,expanded

    def test_actual_upstream_sumslots_power_and_nonpower_of_two(self):
        inputs = (.5,-1.,.25,-.75)
        for m,p in ((1,1),(2,1),(3,1),(4,1),(2,-1),(2,2)):
            with self.subTest(m=m,p=p):
                result,expanded = self.run_source('return SumSlots(x,'+str(m)+','+str(p)+')',upstream_sumslots())
                expected = tuple(sum(inputs[(i+j*p)%4] for j in range(m)) for i in range(4))
                self.assertEqual(result.values,expected)
                self.assertGreater(expanded['construction']['object_array_operations'],0)
                self.assertEqual(expanded['construction']['schema'],12)

    def test_mutation_alias_copy_and_empty_subtraction(self):
        body = '''a = np.full((2,2),hc.Empty(),dtype=object)
a[0,0] = x
b = a[:,0]
c = b.copy()
b[0] = b[0]*2
a[1,0] = a[1,0] - c[0]
return a[0,0] + a[1,0]'''
        result,_ = self.run_source(body)
        self.assertEqual(result.values,(1.5,-3.,.75,-2.25))

    def test_derived_plain_from_empty_is_still_public(self):
        result,_ = self.run_source('p = Empty()+0.5\nq = p+0.25\nreturn x*q')
        self.assertEqual(result.values,(.375,-.75,.1875,-.5625))
        for expression in ('Empty()+True','True-Empty()','Empty()+[True,True,True,True]'):
            result,_ = self.run_source('return x*('+expression+')')
            self.assertEqual(result.values,(.5,-1.,.25,-.75))

    def test_augmented_index_evaluates_target_once_before_rhs(self):
        body = '''a = np.full((2,),Empty(),dtype=object)
a[0] = x
a[1] = x
count = [0]
def index():
    count[0] = count[0]+1
    return 0
def rhs():
    a[0] = x*4
    return x
a[index()] += rhs()
return a[0]*count[0]'''
        result,_ = self.run_source(body)
        self.assertEqual(result.values,(1.,-2.,.5,-1.5))
        result,_ = self.run_source('a=np.full((1,),Empty(),dtype=object)\na[0]-=x\nreturn a[0]')
        self.assertEqual(result.values,(.5,-1.,.25,-.75))

    def test_object_metadata_len_and_iteration_views(self):
        body = '''a = np.array([[x,x],[x,x]],dtype=object)
for row in a:
    row[0] = row[0]*2
n = len(a)+a.ndim+a.size+a.shape[0]
return a[0,0]+x*n'''
        result,_ = self.run_source(body)
        self.assertEqual(result.values,(6.,-12.,3.,-9.))

    def test_constructor_asarray_reshape_flatten_and_return(self):
        body = '''a = np.array([x,x],dtype=np.object_)
b = np.asarray(a)
c = a.reshape(1,2).flatten()
b[0] = x*2
d = np.concatenate((a,c))
return d[0]+d[2]'''
        result,_ = self.run_source(body)
        self.assertEqual(result.values,(1.5,-3.,.75,-2.25))
        result,_ = self.run_source('a = np.array([x],dtype=object)\nreturn a')
        self.assertEqual(result[0].values,(.5,-1.,.25,-.75))

    def test_old_contract_and_unsafe_paths_still_reject(self):
        source = program('','a = np.full((1,),Empty(),dtype=object)\na[0] = x\nreturn a[0]')
        with self.assertRaises(ValueError): normalize(source,{},public_polynomial=True)
        bodies = ('return x+Empty()', 'return x-Empty()', 'return Empty()*x',
                  'a=np.empty((1,),dtype=object)\nreturn a[0]',
                  'a=np.array([[x]],dtype=object)\nreturn a',
                  'a=np.array([x],dtype=object)\nreturn a.data',
                  'a=np.array([x],dtype=object)\nreturn a[x]',
                  'a=np.array([x],dtype=object)\na[0]=a\nreturn x',
                  'a=np.array([x],dtype=object)\nreturn a+1')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError): self.run_source(body)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class ObjectContractTests(unittest.TestCase):
    def test_versioned_request_candidate_cli_and_batch(self):
        import json
        from unittest.mock import patch
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance
        from run_candidate import parse_args,forward_options
        from dsl_grammar_coverage import analyze_source
        from run_agent_batch import main,construction_options,validate_construction_continuation
        payload = dict(fx_graph='x',public_constants={},constant_origins={},
            layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        request = make_request(payload,dict(schema=2,id='objects-unit'),'a'*64,object_arrays=True)
        source = program('','a=np.full((1,),Empty(),dtype=object)\na[0]+=x\nreturn a[0]')
        self.assertEqual(request['task'],'hecate-function-synthesis-v18')
        self.assertTrue(valid_semantic_guidance(request))
        self.assertEqual(validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)['contract'],'hecate-function-v17')
        self.assertEqual(analyze_source(source,{},contract='hecate-function-v17')['public_construction']['schema'],12)
        self.assertIn('--object-arrays',forward_options(parse_args(['--case','case.json','--prepare','--object-arrays'])))
        args = SimpleNamespace(object_arrays=True)
        self.assertEqual(construction_options(args),['--object-arrays'])
        validate_construction_continuation({'object_arrays':True},args)
        with self.assertRaisesRegex(ValueError,'fresh batch'): validate_construction_continuation({},args)
        output = io.StringIO()
        with patch('sys.argv',['run_agent_batch.py','--plan','--object-arrays']), \
             patch('agent_credentials.load_api_key',side_effect=AssertionError('credential access')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main(),0)
        report = json.loads(output.getvalue())
        self.assertTrue(report['object_arrays'])
        self.assertEqual(report['agent_calls'],0)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class ObjectGoldenPlainTests(unittest.TestCase):
    def test_upstream_sources_and_independent_references(self):
        from run_object_array_goldens import PLANS
        upstream = ast.parse(upstream_sumslots()).body
        weights = [[.5,-.25,.125,.75],[-.375,.25,.5,-.125]]
        for case,golden,wrong in PLANS:
            source = (ROOT/'scripts/baseline/golden_cases/object_arrays'/(golden+'.py')).read_text()
            self.assertEqual([ast.dump(n,include_attributes=False) for n in ast.parse(source).body[:-1]],
                             [ast.dump(n,include_attributes=False) for n in upstream])
            constants = dict(c0=weights[0],c1=weights[1]) if case == 'construction-linear' else (
                {} if case.startswith('construction-sumslots') else dict(c0=[.5],c1=[.375]))
            expanded = normalize(source,constants,2 if case == 'construction-linear' else 1,object_arrays=True)
            detected = False
            for x in ([0.]*4,[.5,-1.,.25,-.75],[-1.,1.,-1.,1.]):
                if case.startswith('construction-sumslots'):
                    m = int(case[-1])
                    expected = [sum(x[(i+j)%4] for j in range(m)) for i in range(4)]
                elif case == 'construction-linear':
                    expected = [sum(a*b for a,b in zip(x,row)) for row in weights]
                else:
                    expected = [1.5*v+.375 for v in x]
                result = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed(x))
                actual = [v.values[0] for v in result] if type(result) is list else result.values
                equal = len(actual) == len(expected) and all(abs(a-b)<1e-12 for a,b in zip(actual,expected))
                if not wrong: self.assertTrue(equal,(golden,actual,expected))
                detected |= not equal
            if wrong: self.assertTrue(detected,golden)


@unittest.skipUnless(os.environ.get('POSEIDON_OBJECT_REPORT'),'requires actual object-array CPU evidence')
class ObjectEvidenceTests(unittest.TestCase):
    @staticmethod
    def producer_hashes():
        return {
          "object_arrays.py": "7c9cdc193e8c38dbf7dee04ffd5f5f9cb83789ca74c4f8aa1a94e2fce8a34a02",
          "function_construction.py": "8fa5a01f27c8a19294e5a69398f24e9302e05a7ebfc6f1d51fddd6e2c642f717",
          "candidate_contract.py": "7ecc3ad7115249c10cc8e5cde5a8fa7f6ff665be4d9fcd2799e8e21b0c9fa066",
          "candidate_trace.py": "33bb036695573784f2cbf58e83b0c879f5de659c4b225c19f42187c1696a95d2"
        }

    def test_real_artifacts_and_unchanged_reference(self):
        import numpy as np
        from test_closure_construction import ClosureEvidenceTests
        from run_object_array_goldens import PLANS
        self.assertEqual(np.__version__,'1.25.2')
        ClosureEvidenceTests.verify_evidence(self,os.environ['POSEIDON_OBJECT_REPORT'],PLANS,
            'object_arrays','hecate-function-synthesis-v18',dict(object_arrays=True),self.producer_hashes())

    @unittest.skipUnless(os.environ.get('POSEIDON_OBJECT_STABILITY_REPORT'),'requires independent-key stability evidence')
    def test_near_threshold_independent_key_repeats(self):
        import json
        from test_closure_construction import ClosureEvidenceTests
        filename = os.environ['POSEIDON_OBJECT_STABILITY_REPORT']
        report = json.loads(Path(filename).read_text())
        self.assertEqual(len({row['run'] for row in report['cases']}),2)
        ClosureEvidenceTests.verify_evidence(self,filename,[('construction-sumslots4','sumslots4',False)]*2,
            'object_arrays','hecate-function-synthesis-v18',dict(object_arrays=True),self.producer_hashes())


if __name__ == '__main__':
    unittest.main()
