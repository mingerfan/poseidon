"""Bounded dict semantics and actual upstream shape-helper construction oracles."""
import ast
import importlib.util
import itertools
from pathlib import Path
import unittest

from function_construction import normalize
from candidate_trace import evaluate_tree
from test_function_construction import program
from test_closure_construction import Packed


ROOT = Path(__file__).resolve().parents[2]


def upstream_shapes():
    tree = ast.parse((ROOT/'third_party/dacapo/python/poly/poly/MPCB.py').read_text())
    names = ('cint','fint','InferShapes','CascadeDS','CascadePool')
    selected = [next(n for n in tree.body if type(n) is ast.FunctionDef and n.name == name)
                for name in names]
    return ast.unparse(ast.Module(body=selected,type_ignores=[]))


def shape_reference(data):
    """Independent integer arithmetic, no upstream helper or NumPy formula."""
    result = dict(data)
    def ceildiv(a,b): return (a+b-1)//b
    result.update(ho=data['hi']//data['s'],wo=data['wi']//data['s'],ko=data['s']*data['ki'])
    result['ti'] = ceildiv(data['ci'],data['ki']**2)
    result['to'] = ceildiv(data['co'],result['ko']**2)
    input_size = data['ki']**2*data['hi']*data['wi']*result['ti']
    output_size = result['ko']**2*result['ho']*result['wo']*result['to']
    result['ni'],result['no'] = ceildiv(input_size,data['nt']),ceildiv(output_size,data['nt'])
    for name,size in (('pi',input_size),('po',output_size)):
        copies = data['nt']//size
        result[name] = 1 if copies < 1 else 1 << (copies.bit_length()-1)
    result['q'] = ceildiv(data['co'],result['pi'])
    return result


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class MappingConstructionTests(unittest.TestCase):
    def result(self,body,helpers=''):
        expanded = normalize(program(helpers,body),{},public_mappings=True)
        value = evaluate_tree(expanded['source'],{k:Packed(v) for k,v in expanded['constants'].items()},
                              Packed([.5,-1.,.25,-.75]))
        return value.values,expanded

    def test_copy_is_shallow_without_aliasing_outer_mapping(self):
        body = '''d = dict(a=[1],b=x)
e = d.copy()
e['b'] = x*2
e['a'][0] = 3
return e['b']+d['b']*d['a'][0]'''
        self.assertEqual(self.result(body)[0],(2.5,-5.,1.25,-3.75))

    def test_defaults_insert_and_pop_preserve_exact_object(self):
        body = '''d = {}
a = [x]
b = d.setdefault('a',a)
b[0] = x*2
c = d.get('missing',x)
d.setdefault('a',d.values())
pair = d.popitem()
return pair[1][0]+d.pop('none',c)'''
        self.assertEqual(self.result(body)[0],(1.5,-3.,.75,-2.25))

    def test_ordered_update_duplicate_keys_and_last_inserted_popitem(self):
        body = '''d = dict([('a',x),('b',x),('a',x*2)],c=x*3)
d.update([('b',x*4)],a=x*5)
k,v = d.popitem()
if k != 'c':
    return -x
d.update(d)
return v+d['a']+d['b']'''
        self.assertEqual(self.result(body)[0],(6.,-12.,3.,-9.))

    def test_views_remain_dynamic_after_value_write_and_clear(self):
        body = '''d = dict(a=x,b=x)
v = d.values()
i = iter(v)
first = next(i)
d['b'] = x*2
second = next(i)
keys = d.keys()
items = d.items()
if len(v) != 2 or not keys:
    return -x
d.clear()
if keys or items or len(v):
    return -x
return first+second'''
        self.assertEqual(self.result(body)[0],(1.5,-3.,.75,-2.25))

    def test_reverse_views_and_iterable_pair_update(self):
        body = '''d = dict(a=1,b=2)
e = dict(d.items())
e.update(['cd'])
keys = list(reversed(e.keys()))
items = list(reversed(d.items()))
values = list(reversed(d.values()))
if keys != ['c','b','a'] or items != [('b',2),('a',1)] or values != [2,1]:
    return -x
return x'''
        self.assertEqual(self.result(body)[0],(.5,-1.,.25,-.75))

    def test_view_membership_native_matrix(self):
        native = {1:2,'b':3}
        for kind,value in itertools.product(('keys','values','items'),(0,1,2,'b',(1,2),[1,2],('b',4))):
            try:
                expected = value in getattr(native,kind)()
            except TypeError:
                continue
            body = 'd = {1:2,"b":3}\nreturn x if '+repr(value)+' in d.'+kind+'() else -x'
            actual,_ = self.result(body)
            self.assertEqual(actual,tuple(v if expected else -v for v in (.5,-1.,.25,-.75)),(kind,value))

    def test_method_receiver_and_arguments_evaluate_once_in_order(self):
        body = '''count = [0]
d = dict(a=x)
def receiver():
    count[0] = count[0]+1
    return d
def arg():
    count[0] = count[0]*2
    return 'a'
def default():
    count[0] = count[0]+3
    return x*8
y = receiver().get(arg(),default())
return y*count[0]'''
        self.assertEqual(self.result(body)[0],(2.5,-5.,1.25,-3.75))

    def test_self_view_cycles_and_invalidated_iterator_rejected(self):
        bodies = ('d={}\nd.update(self=d.values())\nreturn x',
                  'd={}\nv=d.items()\nd.setdefault("self",v)\nreturn x',
                  'd={"a":1}\ni=iter(d.keys())\nd["b"]=2\na=next(i)\nreturn x',
                  'd={}\na=d.pop("missing")\nreturn x',
                  'd={}\na=d.popitem()\nreturn x',
                  'd={}\nd.update([("a",1,2)])\nreturn x',
                  'd={}\na=d.get(key="a")\nreturn x',
                  'd={}\nd.update({x:1})\nreturn x',
                  'd={"a":x}\nreturn x if x in d.values() else -x',
                  'd={i:i for i in range(128)}\nd.setdefault(128,x)\nreturn x',
                  'd={}\na=d.keys().__class__\nreturn x')
        for body in bodies:
            with self.subTest(body=body),self.assertRaises(ValueError): self.result(body)

    def test_existing_contract_stays_closed_and_array_copy_survives(self):
        source = program('','d={"a":x}\ne=d.copy()\nreturn e["a"]')
        with self.assertRaises(ValueError): normalize(source,{},object_arrays=True)
        body = 'a=np.array([x],dtype=object)\nb=a.copy()\nb[0]=x*2\nreturn a[0]+b[0]'
        self.assertEqual(self.result(body)[0],(1.5,-3.,.75,-2.25))


@unittest.skipUnless(importlib.util.find_spec('numpy'),'requires existing pinned NumPy environment')
class UpstreamShapeTests(unittest.TestCase):
    def test_actual_helpers_against_independent_integer_shapes(self):
        import numpy as np
        namespace = {'np':np}
        helpers = upstream_shapes()
        exec(compile(helpers,'trusted-upstream-shapes','exec'),namespace)
        variants = itertools.product((4,16,64),(2,4),(1,2),(1,3,8))
        count = 0
        for nt,hi,ki,co in variants:
            initial = dict(nt=nt,hi=hi,wi=hi,s=1,ki=ki,ci=3,co=co,fh=1,fw=1)
            base = shape_reference(initial)
            self.assertEqual(namespace['InferShapes'](dict(initial)),base)
            expected_all = {'InferShapes':base}
            for name in ('CascadeDS','CascadePool'):
                next_shape = dict(base,fh=1,fw=1,s=2 if name == 'CascadeDS' else 1,
                                  ci=base['co'],co=base['co']*(2 if name == 'CascadeDS' else 1),
                                  hi=base['ho'],wi=base['wo'],ki=base['ko'])
                expected = shape_reference(next_shape)
                native_input = dict(base)
                self.assertEqual(namespace[name](native_input),expected)
                self.assertEqual(native_input,base)
                expected_all[name] = expected
            for name,expected in expected_all.items():
                original = initial if name == 'InferShapes' else base
                body = 'data = '+repr(original)+'\nresult = '+name+'(data)\n'
                for key,value in expected.items():
                    body += 'if result['+repr(key)+'] != '+repr(value)+':\n    return -x\n'
                if name != 'InferShapes':
                    body += 'if data != '+repr(original)+':\n    return -x\n'
                body += 'return x'
                expanded = normalize(program(helpers,body),{},public_mappings=True)
                self.assertEqual(evaluate_tree(expanded['source'],{},Packed([1,2,3,4])).values,(1,2,3,4))
                self.assertEqual(expanded['construction']['schema'],13)
                count += 1
        self.assertEqual(count,108)


if __name__ == '__main__':
    unittest.main()
