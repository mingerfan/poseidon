"""Typed arithmetic/alias/overlap probes with independent concrete storage oracles."""
import ast
import copy
import unittest
import numpy as np
import object_arrays as objects
from function_construction import Value
from benchmark_suite import Builder
from compiler_configuration import configuration,PROFILE_SHA256
from unified_graph_contract import prepare,validate_candidate
from unified_public_contract import CONTRACT,normalize,event_record
from unified_public_exercises import golden_variant,ARITHMETIC_RECIPES
from unified_public_coverage import verify_trace_coverage,fingerprint
from unified_graph_lowering import lower


class ArithmeticCoverageTests(unittest.TestCase):
    def request(self,feature):
        b=Builder([(2,),(2,)]);g=b.finish(b.node('add',['input0','input1']))
        return prepare(g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),
                       'unified-public-'+feature.replace('.','-'),construction_profile=CONTRACT)

    def source(self,body):
        return '@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+line+'\n' for line in body.splitlines())

    def checked(self,feature,body):
        req=self.request(feature)
        return validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=self.source(body)),req)

    def test_broadcast_pair_facts_require_matching_cell_positions(self):
        x=Value('x','cipher')
        a=objects.array([x,.5]);b=objects.array([.5,x])
        facts=objects.arithmetic_facts(a,b)
        self.assertFalse(facts['cipher_pair']);self.assertEqual(facts['cipher_pair_cells'],0)
        scalar_rhs=objects.arithmetic_facts(a,x)
        self.assertFalse(scalar_rhs['cipher_pair']);self.assertEqual(scalar_rhs['cipher_pair_cells'],0)
        zero_array_rhs=objects.arithmetic_facts(a,objects.array(x))
        self.assertTrue(zero_array_rhs['cipher_pair']);self.assertEqual(zero_array_rhs['cipher_pair_cells'],1)
        facts=objects.arithmetic_facts(objects.array([[x],[.5]]),objects.array([.5,x]))
        self.assertTrue(facts['cipher_pair']);self.assertEqual(facts['cipher_pair_cells'],1)
        from object_arithmetic_exercises import features
        facts['result_shape']=[2,2]
        self.assertNotIn('cipher_pair',features('object_binary',ast.parse('a+b',mode='eval').body,facts))
        self.assertIn('cipher_pair',features('object_binary',ast.parse('a*b',mode='eval').body,facts))

    def test_real_snapshot_vs_counterfactual_sequential_oracle(self):
        for op,apply in ((ast.Add(),lambda op,x,y:x+y),(ast.Mult(),lambda op,x,y:x*y)):
            original=np.array([.5,1.,2.],dtype=object)
            expected=original.copy()
            if type(op) is ast.Add:expected[1:]+=expected[:-1]
            else:expected[1:]*=expected[:-1]
            a=objects.array(original.tolist())
            objects.elementwise(op,objects.get(a,slice(1,None)),objects.get(a,slice(None,-1)),apply,inplace=True)
            self.assertEqual(a.data.tolist(),expected.tolist())
            sequential=original.copy()
            for i in range(1,3):sequential[i]=apply(op,sequential[i],sequential[i-1])
            a=objects.array(original.tolist())
            objects.elementwise(op,objects.get(a,slice(1,None)),objects.get(a,slice(None,-1)),apply,inplace=True,_probe_sequential=True)
            self.assertEqual(a.data.tolist(),sequential.tolist())
            self.assertNotEqual(sequential.tolist(),expected.tolist())

    def test_all_arithmetic_partitions_have_real_event_binding(self):
        for feature in ARITHMETIC_RECIPES:
            with self.subTest(feature=feature):
                req=self.request(feature);source=golden_variant(lower(req),req['construction_exercise']['id'],req)
                checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['construction_exercise']
                events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
                record=dict(events=events,normalized_sha256=expanded['construction']['normalized_sha256'],candidate_python_executed=False)
                result=verify_trace_coverage(checked,record)
                self.assertEqual(result['numeric_features'],[feature])
                self.assertTrue(result['finite_influence_checked'])
                bad=copy.deepcopy(record);bad['events']=[e for e in bad['events'] if not set(e['features'])&{'event.object_binary','event.object_inplace'}]
                with self.assertRaises(ValueError):verify_trace_coverage(checked,bad)

    def test_cipher_pairs_must_multiply_and_their_cells_must_contribute(self):
        bodies=[
            'a=np.array([x,0.5],dtype=object)\nb=np.array([0.5,x],dtype=object)\nc=a*b\nreturn [c[0]+c[1]]',
            'a=np.array([x,0.5],dtype=object)\nb=np.array([x,0.5],dtype=object)\nc=a*b\nreturn [x+c[1]]',
            'a=np.array([x],dtype=object)\nb=np.array([x],dtype=object)\nc=a+b\nreturn [c[0]]',
            'a=np.array([x],dtype=object)\nc=a*0.5\nreturn [c[0]]',
            'a=np.array([x],dtype=object)\nc=a*x\nreturn [c[0]]',
        ]
        for body in bodies:
            with self.subTest(body=body),self.assertRaisesRegex(ValueError,'Missing contributing typed array'):
                self.checked('cipher_pair',body)

    def test_empty_cells_unobserved_or_zero_substitution_equivalent_rejected(self):
        for body in [
            'a=np.array([Empty(),x],dtype=object)\na-=np.array([x,0.5],dtype=object)\nreturn [a[1]]',
            'a=np.array([Empty()],dtype=object)\na+=np.array([x],dtype=object)\nreturn [a[0]]',
            'a=np.array([0],dtype=object)\na-=np.array([x],dtype=object)\nreturn [a[0]]',
        ]:
            with self.subTest(body=body),self.assertRaisesRegex(ValueError,'Missing contributing typed array'):
                self.checked('empty_left',body)

    def test_alias_and_overlap_must_be_observable(self):
        cases={
            'inplace.Mult':['a=np.array([x],dtype=object)\nalias=a.copy()\na*=0.5\nreturn [a[0]]',
                            'a=np.array([x],dtype=object)\nalias=a\na*=0.5\nreturn [a[0]]'],
            'overlap':['a=np.array([x,x+1,x+2],dtype=object)\na[1:]+=a[:-1].copy()\nreturn [a[2]]',
                       'a=np.array([x,x+1,x+2],dtype=object)\na[1:]+=a[:-1]\nreturn [a[0]]',
                       'a=np.array([x,x+1],dtype=object)\na+=a\nreturn [a[0]]'],
            'object.overlap_mult':['a=np.array([x,x+1,x+2],dtype=object)\na[1:]*=a[:-1].copy()\nreturn [a[2]]'],
            'rank_broadcast':['a=np.array([[x,x]],dtype=object)\nc=a+np.array([0.25,0.75])\nreturn [c[0,0]]'],
            'zero_dim':['a=np.array([x],dtype=object)\nc=a+0.5\nreturn [c[0]]'],
        }
        for feature,bodies in cases.items():
            for body in bodies:
                with self.subTest(feature=feature,body=body),self.assertRaisesRegex(ValueError,'Missing contributing typed array'):
                    self.checked(feature,body)

    def test_unobserved_probe_does_not_change_normal_execution(self):
        req=self.request('cipher_pair')
        source=self.source('a=np.array([x,y],dtype=object)\nb=np.array([y,x],dtype=object)\nc=a*b\nreturn [c[0]+c[1]]')
        base=normalize(source,req);events=[]
        observed=normalize(source,req,lambda n:events.append(event_record(n)))
        self.assertEqual(base,observed)
        untouched=normalize(source,req,_object_probe=dict(span=[999,0,999,1],kind='cipher_pair'))
        self.assertEqual(base,untouched)
        for bad in ({'span':[1,2,3,4],'kind':'arbitrary'},{'span':[1],'kind':'all_results'}):
            with self.assertRaises(ValueError):normalize(source,req,_object_probe=bad)


if __name__=='__main__':unittest.main()
