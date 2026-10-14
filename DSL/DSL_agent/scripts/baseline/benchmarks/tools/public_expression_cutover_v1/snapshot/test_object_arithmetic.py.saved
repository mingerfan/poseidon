"""Opt-in object arithmetic: NumPy and trusted Expr dispatch, not FHE evidence."""
import ast
import contextlib
import importlib.util
import io
import os
import json
import hashlib
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
import operator
import unittest

import object_arrays as objects
from function_construction import normalize, Value
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed
from test_object_arrays import upstream_empty


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'requires pinned NumPy')
class ObjectArithmeticTests(unittest.TestCase):
    def evaluate(self, body, outputs=1):
        result=normalize(program('',body),{},outputs,object_arithmetic=True)
        values=evaluate_tree(result['source'],{k:Packed(v) for k,v in result['constants'].items()},
                             Packed([.5,-1.,.25,-.75]))
        self.assertEqual(result['construction']['schema'],14)
        return values,result

    def test_broadcast_numeric_native_matrix(self):
        import numpy as np
        shapes=[(),(1,),(2,),(2,1),(1,3),(2,3),(0,),(0,1)]
        count=0
        for sa in shapes:
            for sb in shapes:
                a=np.full(sa,3,dtype=object)
                b=np.full(sb,2,dtype=object)
                for op,fn in ((ast.Add(),operator.add),(ast.Sub(),operator.sub),(ast.Mult(),operator.mul)):
                    try:expected=fn(a,b)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            objects.elementwise(op,objects.wrap(a),objects.wrap(b),lambda o,x,y:fn(x,y))
                        continue
                    actual=objects.elementwise(op,objects.wrap(a),objects.wrap(b),lambda o,x,y:fn(x,y))
                    if type(expected) is np.ndarray:
                        self.assertEqual(actual.shape,expected.shape)
                        self.assertEqual(actual.data.tolist(),expected.tolist())
                    else:
                        self.assertIs(type(actual),type(expected))
                        self.assertEqual(actual,expected)
                    count+=1
        self.assertGreater(count,100)

    def test_numeric_broadcast_output_limit_precedes_callbacks(self):
        calls=[]
        with self.assertRaisesRegex(ValueError,'element limit'):
            objects.elementwise(ast.Add(),objects.full((128,1),1),objects.full((1,128),1),
                                lambda op,a,b:calls.append((a,b)) or a+b)
        self.assertEqual(calls,[])

    def test_overlap_inplace_and_alias_oracle(self):
        import numpy as np
        for op,fn in ((ast.Add(),operator.iadd),(ast.Sub(),operator.isub),(ast.Mult(),operator.imul)):
            for target,source in ((slice(1,None),slice(None,-1)),(slice(None,-1),slice(1,None)),
                                  (slice(None),slice(None,None,-1))):
                native=np.array([1,2,3,4],dtype=object)
                checked=objects.array([1,2,3,4])
                expected=fn(native[target],native[source])
                native[target]=expected
                view=objects.get(checked,target)
                result=objects.elementwise(op,view,objects.get(checked,source),
                    lambda o,a,b:{ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul}[type(o)](a,b),
                    inplace=True)
                self.assertIs(result,view)
                self.assertEqual(checked.data.tolist(),native.tolist())

    def test_inplace_cannot_expand_and_leaves_storage_unchanged(self):
        a=objects.array([[1,2]])
        with self.assertRaisesRegex(ValueError,'cannot expand'):
            objects.elementwise(ast.Add(),a,objects.array([[3],[4]]),lambda op,a,b:a+b,inplace=True)
        self.assertEqual(a.data.tolist(),[[1,2]])

    def test_zero_dim_result_and_inplace_identity(self):
        a=objects.array(2)
        self.assertEqual(objects.elementwise(ast.Add(),a,3,lambda op,a,b:a+b),5)
        self.assertIs(objects.elementwise(ast.Mult(),a,3,lambda op,a,b:a*b,inplace=True),a)
        self.assertEqual(a.data.item(),6)

    def test_real_expr_object_array_direction(self):
        import numpy as np
        Empty,Expr,Plain=upstream_empty()
        # Replace the recording Plain with the actual trusted constructor;
        # otherwise its missing float64 conversion would mask this failure.
        from pathlib import Path
        source=Path(__file__).resolve().parents[2]/'third_party/dacapo/python/hecate/hecate/expr.py'
        definition=next(n for n in ast.parse(source.read_text()).body
                        if type(n) is ast.ClassDef and n.name=='Plain')
        namespace=Expr.__add__.__globals__
        namespace['npcl']=np.ctypeslib
        namespace['getProperFrame']=lambda: (None,'unit',1,None,None,None)
        namespace['lt'].createConstant=lambda *args: ('constant',)
        exec(compile(ast.Module(body=[definition],type_ignores=[]),'trusted-plain','exec'),namespace)
        x=Expr('x')
        a=np.array([x,x],dtype=object)
        actual=a+x
        self.assertEqual([v.obj for v in actual],[('binary',6,'x','x')]*2)
        # Actual Expr resolves the whole ndarray into a numeric Plain and fails.
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises((TypeError,ValueError)):
            x+a
        result,_=self.evaluate('a=np.array([x,x],dtype=object)\nb=a+x\nreturn b[0]+b[1]')
        self.assertEqual(result.values,(2.,-4.,1.,-3.))
        with self.assertRaisesRegex(ValueError,'does not use'):
            self.evaluate('a=np.array([x],dtype=object)\nreturn (x+a)[0]')

    def test_storage_broadcast_not_slot_broadcast(self):
        result,_=self.evaluate('a=np.array([[x],[x*2]],dtype=object)\nb=a*np.array([2,3])\nreturn b[0,1]+b[1,0]')
        self.assertEqual(result.values,(3.5,-7.,1.75,-5.25))

    def test_fresh_binary_storage_and_inplace_alias(self):
        result,_=self.evaluate('a=np.array([x],dtype=object)\nalias=a\nb=a+1\na*=2\nreturn b[0]+alias[0]')
        self.assertEqual(result.values,(2.5,-2.,1.75,-1.25))

    def test_slice_overlap_symbolic_values(self):
        result,_=self.evaluate('a=np.array([x,x*2,x*3],dtype=object)\na[1:]+=a[:-1]\nreturn a[1]+a[2]')
        self.assertEqual(result.values,(4.,-8.,2.,-6.))

    def test_slice_target_evaluated_once_rhs_mutates_live_view(self):
        body='a=np.array([x,x],dtype=object)\ncount=[0]\ndef idx():\n    count[0]=count[0]+1\n    return 0\ndef rhs():\n    a[0]=x*3\n    return x\na[idx():]+=rhs()\nreturn (a[0]+a[1])*count[0]'
        result,_=self.evaluate(body)
        self.assertEqual(result.values,(3.,-6.,1.5,-4.5))

    def test_empty_broadcast_preserves_identity_not_zero(self):
        result,_=self.evaluate('a=np.full((2,),Empty(),dtype=object)\nb=np.array([x,x*2],dtype=object)\na-=b\nreturn a[0]+a[1]')
        self.assertEqual(result.values,(1.5,-3.,.75,-2.25))
        with self.assertRaises(ValueError):
            self.evaluate('a=np.array([x],dtype=object)\nb=np.full((1,),Empty(),dtype=object)\nreturn (a-b)[0]')

    def test_uninitialized_or_unsupported_operations_rejected(self):
        for body in ('a=np.empty((1,),dtype=object)\nreturn (a+x)[0]',
                     'a=np.array([x],dtype=object)\nreturn (a/2)[0]',
                     'a=np.array([x],dtype=object)\nreturn (a**2)[0]',
                     'a=np.array([x],dtype=object)\nreturn a.data',
                     'a=np.array([x],dtype=object)\na+=lambda: x\nreturn a[0]'):
            with self.subTest(body=body),self.assertRaises(ValueError):
                self.evaluate(body)

    def test_old_contract_rejection_and_metadata_unchanged(self):
        source=program('','a=np.array([x],dtype=object)\nreturn (a+1)[0]')
        with self.assertRaises(ValueError):normalize(source,{},public_mappings=True)
        plain=normalize(program('','return x'),{},public_mappings=True)
        self.assertEqual(plain['construction']['schema'],13)
        self.assertNotIn('object_elementwise_operations',plain['construction'])

    def test_v20_request_trace_cli_provider_and_old_rules(self):
        from candidate_contract import make_request,validate_candidate,valid_semantic_guidance,TASK_RULES
        from run_candidate import parse_args,forward_options
        from deepseek_provider import public_request
        from dsl_grammar_coverage import analyze_source
        payload=dict(fx_graph='x',public_constants={},constant_origins={},
            layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,
                        output_selectors=[[0,i] for i in range(4)]))
        req=make_request(payload,dict(schema=2,id='object-arithmetic-unit'),'a'*64,object_arithmetic=True)
        self.assertEqual(req['task'],'hecate-function-synthesis-v20')
        self.assertTrue(valid_semantic_guidance(req))
        source=program('','a=np.array([x],dtype=object)\na*=0.5\nreturn a[0]+x+0.375')
        candidate=dict(schema=1,request_id=req['request_id'],hecate_source=source)
        self.assertEqual(validate_candidate(candidate,req)['contract'],'hecate-function-v19')
        self.assertEqual(public_request(req)['task'],req['task'])
        analyze_source(source,{},contract='hecate-function-v19')
        args=parse_args(['--case','scripts/baseline/cases/arithmetic-alias-chain.json',
                         '--object-arithmetic','--prepare'])
        self.assertIn('--object-arithmetic',forward_options(args))
        old=make_request(payload,dict(schema=2,id='object-arithmetic-unit'),'a'*64,public_mappings=True)
        self.assertEqual(old['task'],'hecate-function-synthesis-v19')
        self.assertEqual(old['rules'],TASK_RULES[old['task']][1])
        with self.assertRaises(ValueError):
            validate_candidate(dict(candidate,request_id=old['request_id']),old)


@unittest.skipUnless(os.environ.get('POSEIDON_OBJECT_ARITHMETIC_REPORT'),
                     'requires real object arithmetic CPU evidence')
class ObjectArithmeticEvidenceTests(unittest.TestCase):
    def test_exact_sources_artifacts_security_and_numerical_counterexamples(self):
        import numpy as np
        from candidate_contract import validate_candidate, canonical
        from seal_cpu_golden import compare
        from seal_artifact_gate import inspect_artifacts
        root=Path(__file__).resolve().parents[2]
        batch=json.loads(Path(os.environ['POSEIDON_OBJECT_ARITHMETIC_REPORT']).read_text())
        self.assertEqual(batch['status'],'passed')
        self.assertEqual(batch['agent_calls'],0)
        names=('broadcast','alias','overlap','empty','wrong_alias','wrong_empty')
        self.assertEqual(len(batch['cases']),len(names))
        for name,row in zip(names,batch['cases']):
            with self.subTest(case=name):
                run=Path(row['run']).resolve()
                self.assertEqual(run.parent,Path(str(RESULTS)))
                data=json.loads((run/'report.json').read_text())
                self.assertEqual(data['agent_calls'],0)
                self.assertFalse(data['llm_generation_validated'])
                self.assertEqual(data['backend'],'upstream_SEAL_HEVM_CPU')
                self.assertFalse(data['poseidon_gpu_validated'])
                self.assertEqual(data['tolerance'],dict(atol=1e-5,rtol=1e-4))
                self.assertEqual(data['parameters']['security_check'],'tc128')
                self.assertEqual(data['parameters']['modulus_bits'],[60]*14)
                self.assertEqual(data['parameters']['polynomial_degree'],32768)
                for filename,digest in data['frozen_hashes'].items():
                    path=(run/filename).resolve()
                    self.assertTrue(path.is_relative_to(run))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                item=data['attempts'][0]
                output=run/'attempt-00/output'
                for filename,digest in item['artifact_hashes'].items():
                    path=(output/filename).resolve()
                    self.assertTrue(path.is_relative_to(output))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                req=json.loads((run/'request.json').read_text())
                self.assertEqual(req['task'],'hecate-function-synthesis-v20')
                self.assertEqual(req['request_id'],hashlib.sha256(canonical(
                    {k:v for k,v in req.items() if k!='request_id'})).hexdigest())
                source=(root/'scripts/baseline/golden_cases/object_arithmetic'/(name+'.py')).read_text()
                payload=json.loads((run/'attempt-00/trace-payload.json').read_text())
                self.assertEqual(payload['request'],req)
                self.assertEqual(payload['candidate']['hecate_source'],source)
                self.assertEqual(validate_candidate(payload['candidate'],req),item['static_check'])
                expanded=normalize(source,req['public_constants'],object_arithmetic=True)
                self.assertEqual(expanded['source'],(output/'normalized-source.py').read_text())
                self.assertEqual(expanded['construction'],json.loads((output/'construction.json').read_text()))
                self.assertEqual(expanded['derived_constants'],json.loads((output/'derived-constants.json').read_text()))
                self.assertEqual(item['artifact_gate'],inspect_artifacts(
                    (output/'lowered._hecate_golden.hevm').read_bytes(),
                    (output/'_hecate_golden.cst').read_bytes(),rotation_steps=(-3,-2,-1,1,2,3),expected_inputs=1))
                self.assertTrue(item['execution']['encrypted_execution'])
                self.assertFalse(item['execution']['bootstrap_executed'])
                self.assertEqual(item['execution']['input_batches'],4)
                self.assertFalse(item['trace']['candidate_python_executed'])
                with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
                    np.testing.assert_allclose(arrays['reference'],1.5*arrays['inputs']+.375,atol=1e-12,rtol=1e-12)
                    computed=compare(np.load(output/'decrypted.npy',allow_pickle=False),arrays['reference'],1e-5,1e-4)
                self.assertEqual(computed,item['comparison'])
                self.assertEqual(computed['passed'],not name.startswith('wrong_'))
                self.assertTrue(row['matched_expected'])
                if name.startswith('wrong_'):
                    self.assertEqual(item['failure_layer'],'numerical_comparison')
                    self.assertGreater(computed['max_absolute_error'],.1)
                self.assertFalse((run/'private-keys').exists())


if __name__=='__main__':
    unittest.main()
