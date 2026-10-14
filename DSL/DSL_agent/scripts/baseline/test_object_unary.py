"""Pinned NumPy/upstream unary oracle plus bounded symbolic construction tests."""
import ast
import importlib.util
import operator
import unittest
import numpy as np
import object_arrays as objects
from function_construction import normalize,Value
from test_object_arrays import upstream_empty
from test_function_construction import program
from test_closure_construction import Packed
from candidate_trace import evaluate_tree


class ObjectUnaryTests(unittest.TestCase):
    def test_numeric_numpy_matrix_and_fresh_storage(self):
        self.assertEqual(np.__version__,'1.25.2')
        for shape in [(),(1,),(3,),(2,3),(0,),(2,0),(2,1,2)]:
            for value in [False,True,-2,-.5,0.,.25]:
                for op,fn in [(ast.USub(),operator.neg),(ast.UAdd(),operator.pos)]:
                    with self.subTest(shape=shape,value=value,op=type(op).__name__):
                        native=np.full(shape,value,dtype=object);checked=objects.wrap(native.copy())
                        expected=fn(native)
                        actual=objects.unary(op,checked,lambda op,v:fn(v))
                        if type(expected) is np.ndarray:
                            self.assertEqual(actual.shape,expected.shape)
                            self.assertEqual(actual.data.tolist(),expected.tolist())
                            self.assertFalse(np.shares_memory(actual.data,checked.data))
                        else:self.assertEqual((type(actual),actual),(type(expected),expected))
                        self.assertEqual(checked.data.tolist(),native.tolist())

    def test_upstream_cipher_and_empty_dispatch(self):
        E,X,P=upstream_empty()
        for shape in [(),(1,),(2,1)]:
            x=X('x');native=np.full(shape,x,dtype=object)
            result=-native
            cells=list(result.flat) if type(result) is np.ndarray else [result]
            self.assertTrue(all(v.obj==('unary',13,'x') for v in cells))
            self.assertEqual(x.obj,'x')
            for bad in [native,np.full(shape,E(),dtype=object),np.full(shape,None,dtype=object)]:
                with self.assertRaises(TypeError):operator.pos(bad)
            for bad in [np.full(shape,E(),dtype=object),np.full(shape,None,dtype=object)]:
                with self.assertRaises(TypeError):operator.neg(bad)

    def evaluate(self,body):
        expanded=normalize(program('',body),{},object_unary=True)
        self.assertEqual(expanded['construction']['schema'],16)
        return evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},
                             Packed([.5,-1.,.25,-.75])),expanded

    def test_cipher_negation_and_zero_dim_result(self):
        for body in ['a=np.array([x],dtype=object)\nb=-a\nreturn b[0]',
                     'a=np.array(x,dtype=object)\nreturn -a',
                     'a=np.array([x],dtype=object)\nreturn np.negative(a)[0]']:
            result,expanded=self.evaluate(body)
            self.assertEqual(result.values,(-.5,1.,-.25,.75))
            self.assertEqual(expanded['construction']['object_unary_operations'],1)
            self.assertTrue(any(isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub)
                                for n in ast.walk(ast.parse(expanded['source']))))

    def test_result_is_fresh_source_and_view_unchanged(self):
        body='a=np.array([x,x*2],dtype=object)\nview=a[:1]\nb=-view\na[0]=x*3\nreturn b[0]+view[0]+a[1]'
        result,_=self.evaluate(body)
        self.assertEqual(result.values,(2.,-4.,1.,-3.))
        for call in ['+a','np.positive(a)']:
            result,_=self.evaluate('a=np.array([0.5],dtype=object)\nb='+call+'\na[0]=1.0\nreturn x*(b[0]+a[0])')
            self.assertEqual(result.values,(.75,-1.5,.375,-1.125))

    def test_plain_negation_preserves_plain_not_numeric_cast(self):
        body='p=Empty()+0.5\na=np.array([p],dtype=object)\nb=-a\nreturn x*b[0]'
        result,_=self.evaluate(body)
        self.assertEqual(result.values,(-.25,.5,-.125,.375))
        for tail in ['return x*float(b[0])','return x*float(b.item())','return x*float(b)']:
            with self.assertRaises(ValueError):self.evaluate(body.rsplit('return ',1)[0]+tail)
        with self.assertRaises(ValueError):self.evaluate(body.replace('b=-a','b=+a'))

    def test_reject_unsupported_cells_and_ufunc_options(self):
        for body in ['return +np.array(x,dtype=object)',
                     'return np.positive(np.array([x],dtype=object))[0]',
                     'a=np.full((1,),Empty(),dtype=object)\nreturn (-a)[0]',
                     'a=np.empty((1,),dtype=object)\nreturn (-a)[0]',
                     'a=np.array([x],dtype=object)\nreturn np.negative(a,out=a)[0]',
                     'a=np.array([x],dtype=object)\nreturn (a/2)[0]',
                     'a=np.array([x],dtype=object)\nreturn (a**2)[0]',
                     'return np.negative(x)',
                     'return np.positive(np.array([1.0]))']:
            with self.subTest(body=body),self.assertRaises(ValueError):self.evaluate(body)

    def test_unary_operand_evaluated_once(self):
        body='count=[0]\ndef get():\n    count[0]=count[0]+1\n    return np.array([x],dtype=object)\na=-get()\nreturn a[0]*count[0]'
        result,_=self.evaluate(body)
        self.assertEqual(result.values,(-.5,1.,-.25,.75))

    def test_old_contract_remains_closed(self):
        for body in ['a=np.array([x],dtype=object)\nreturn (-a)[0]',
                     'a=np.array([x],dtype=object)\nreturn np.negative(a)[0]']:
            with self.assertRaises(ValueError):normalize(program('',body),{},scalar_conversion=True)



    def test_formal_contract_cli_batch_and_old_request(self):
        import contextlib,io,json,sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from candidate_contract import make_request,validate_candidate
        from deepseek_provider import public_request
        from run_candidate import parse_args,forward_options
        from run_agent_batch import main,construction_options,validate_construction_continuation
        from dsl_grammar_coverage import analyze_source
        payload=dict(fx_graph=[],public_constants={},constant_origins={},
            layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,output_selectors=[[0,i] for i in range(4)]))
        model=dict(schema=2,id='object-unary-unit')
        req=make_request(payload,model,'a'*64,object_unary=True)
        source=program('','a=np.array([x],dtype=object)\nreturn (-a)[0]')
        checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)
        self.assertEqual(checked['contract'],'hecate-function-v21')
        self.assertEqual(public_request(req),req)
        analyze_source(source,{},contract='hecate-function-v21')
        args=parse_args(['--case','scripts/baseline/cases/arithmetic-alias-chain.json','--object-unary','--prepare'])
        self.assertIn('--object-unary',forward_options(args))
        with self.assertRaises(ValueError):
            make_request(payload,model,'a'*64,object_unary=True,construction_exercise='sc-defaults')
        buf=io.StringIO()
        with patch.object(sys,'argv',['run_agent_batch.py','--plan','--object-unary']),contextlib.redirect_stdout(buf):
            self.assertEqual(main(),0)
        self.assertTrue(json.loads(buf.getvalue())['object_unary'])
        ns=SimpleNamespace(object_unary=True)
        self.assertEqual(construction_options(ns),['--object-unary'])
        with self.assertRaises(ValueError):validate_construction_continuation({},ns)
        validate_construction_continuation(dict(object_unary=True),ns)
        old=make_request(payload,model,'a'*64,scalar_conversion=True)
        with self.assertRaises(ValueError):
            validate_candidate(dict(schema=1,request_id=old['request_id'],hecate_source=source),old)

if __name__=='__main__':unittest.main()
