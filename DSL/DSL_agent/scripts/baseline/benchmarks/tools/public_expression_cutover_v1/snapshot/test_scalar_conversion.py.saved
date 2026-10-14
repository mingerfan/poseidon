"""Pinned NumPy scalar extraction oracle; no paid calls or decryption in casts."""
import importlib.util
import unittest
import warnings
import json
from pathlib import Path
import scalar_conversion as sc
import public_numeric as numeric
import object_arrays as objects
from function_construction import normalize,Value
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires pinned NumPy')
class ScalarConversionTests(unittest.TestCase):
    def test_cast_numpy_shape_and_warning_matrix(self):
        import numpy as np
        self.assertEqual(np.__version__,'1.25.2')
        for value in [-3,-1.75,-0.0,.5,2.75]:
            for shape in [(),(1,),(1,1),(1,1,1),(1,1,1,1)]:
                for kind in ['float','int']:
                    native=np.full(shape,value)
                    checked=numeric.array(native.tolist())
                    for obj in [checked,sc.object_from_numeric(checked)]:
                        with warnings.catch_warnings(record=True) as seen:
                            warnings.simplefilter('always')
                            expected=(float if kind=='float' else int)(native if type(obj) is numeric.Array else native.astype(object))
                        actual,deprecated=sc.cast(kind,[obj])
                        self.assertIs(type(actual),type(expected))
                        self.assertEqual(actual,expected)
                        self.assertEqual(deprecated,bool(shape))
                        self.assertEqual(bool(seen),bool(shape))

    def test_non_singletons_are_not_first_element_conversions(self):
        for data in [[],[1,2],[[1,2]]]:
            for kind in ['float','int']:
                with self.assertRaisesRegex(ValueError,'exactly one'):sc.cast(kind,[numeric.array(data)])

    def test_defaults_booleans_strings_and_base(self):
        for kind,args in [('float',[]),('int',[]),('float',[True]),('int',[False]),
                          ('int',['ff',16]),('int',['0b101',0]),('int',['-10',2]),
                          ('float',[' -1.25 ']),('int',['1_000'])]:
            expected=(float if kind=='float' else int)(*args)
            actual,legacy=sc.cast(kind,args)
            self.assertEqual(actual,expected);self.assertIs(type(actual),type(expected))
            self.assertFalse(legacy)
        for kind,args in [('float',[None]),('float',['nan']),('float',['inf']),
                          ('int',['-1.2']),('int',[3,2]),('int',['10',1]),
                          ('float',[[1]]),('float',[float('inf')]),('int',['1'*129])]:
            with self.subTest(kind=kind,args=str(args)[:50]),self.assertRaises(ValueError):sc.cast(kind,args)

    def test_item_all_index_forms_numpy_oracle(self):
        import numpy as np
        for native in [np.array(.5),np.array([.5]),np.array([[1.,2.],[3.,4.]]),np.array([])]:
            checked=numeric.array(native.tolist())
            for args in [[],[0],[-1],[(1,0)],[1,0],[True],[(True,0)],[1.0],[()],[-5],[0,0,0]]:
                for value in [checked,sc.object_from_numeric(checked)]:
                    try:expected=native.item(*args)
                    except (TypeError,ValueError,IndexError):
                        with self.assertRaises(ValueError):sc.item(value,args)
                    else:
                        actual=sc.item(value,args)
                        self.assertEqual(actual,expected)
                        self.assertIs(type(actual),type(expected))

    def test_no_secret_conversion_but_item_preserves_symbol(self):
        x=Value('x','cipher')
        a=objects.array(x)
        self.assertIs(sc.item(a,[]),x)
        for v in [x,a,objects.array(Value('p','plain')),objects.array(None),objects.array(objects.Empty())]:
            with self.assertRaises(ValueError):sc.cast('float',[v])

    def test_source_and_old_contract(self):
        source=program('','w=float(c0)\nb=np.asarray(c1).item()\nreturn x*w+x+b')
        expanded=normalize(source,{'c0':[.5],'c1':[.375]},scalar_conversion=True)
        self.assertEqual(expanded['construction']['schema'],15)
        self.assertEqual(expanded['construction']['legacy_array_scalar_conversions'],1)
        result=evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed([.5,-1,.25,-.75]))
        self.assertEqual(result.values,(1.125,-1.125,.75,-.75))
        with self.assertRaises(ValueError):normalize(source,{'c0':[.5],'c1':[.375]},object_arithmetic=True)

    def test_object_constructor_preserves_named_numeric_array_shape(self):
        source=program('','a=np.array(c0,dtype=object)\nif a.shape[0]!=4:\n    return -x\nreturn x*float(a.item(0))+x*float(a.item(1))')
        expanded=normalize(source,{'c0':[.5,1.,2.,3.]},scalar_conversion=True)
        result=evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},Packed([.5,-1,.25,-.75]))
        self.assertEqual(result.values,(.75,-1.5,.375,-1.125))

    def test_symbolic_plain_is_not_public_array(self):
        for body in ['p=Empty()+0.5\nreturn x*float(p)',
                     'p=Empty()+0.5\nreturn x*p.item()',
                     'return x*float(np.array(x,dtype=object))']:
            with self.subTest(body=body),self.assertRaises(ValueError):
                normalize(program('',body),{},scalar_conversion=True)
        expanded=normalize(program('','a=np.array(x,dtype=object)\nreturn a.item()'),{},scalar_conversion=True)
        self.assertTrue(expanded['check'])

    def test_object_input_conversion_bounded_before_materialization(self):
        body='a=[x]\nfor i in range(20):\n    a=[a,a]\nb=np.array(a,dtype=object)\nreturn x'
        with self.assertRaisesRegex(ValueError,'resource limit'):
            normalize(program('',body),{},scalar_conversion=True)


if __name__=='__main__':unittest.main()
