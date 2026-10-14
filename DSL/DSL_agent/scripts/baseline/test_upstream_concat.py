"""HE_Concat binding and actual public-vector tests; not FHE tests."""
import copy,unittest
import numpy as np
from benchmark_graph import digest,samples
from benchmark_math import evaluate
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare,validate_request,validate_candidate
from upstream_candidate_helpers import PROFILE,BN_PROFILE,CONCAT_PROFILE,manifest
from upstream_concat_cases import cases
from upstream_adapters.concat_node import bind_node,PeriodicRotationInput,apply

class ConcatContractTests(unittest.TestCase):
    def request(self,row=None):
        return prepare((row or cases()[1])['model'],PROFILE_SHA256,helper_profile=CONCAT_PROFILE,helper_exercise=['HE_Concat0'])
    def test_valid_bindings_and_directed_contexts(self):
        rows=cases();self.assertEqual(len(rows),16)
        for row in rows:
            if row['name']=='concat_frozen_7':continue
            with self.subTest(name=row['name']):
                r=self.request(row);c=validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=row['source']),r)
                self.assertIn('HE_Concat0',c['upstream_exercise']['witnesses'])
                expected=r['upstream_helpers']['helpers']['HE_Concat0']['rotations']
                self.assertTrue(set(expected)<=set(c['functions']['golden']['rotation_steps']))
    def test_original_profiles_do_not_gain_concat(self):
        g=cases()[0]['model']
        for profile in [PROFILE,BN_PROFILE]:self.assertNotIn('HE_Concat0',manifest(g,profile)['helpers'])
    def test_oversize_intermediate_remains_explicitly_blocked(self):
        row=cases()[7];r=prepare(row['model'],PROFILE_SHA256,helper_profile=CONCAT_PROFILE)
        self.assertEqual(r['model'],row['model']);self.assertNotIn('HE_Concat0',r['upstream_helpers']['helpers'])
        self.assertIn('does not fit',r['upstream_helpers']['unavailable_bindings'][0]['reason'])
    def test_incompatible_shapes_and_axis_do_not_disappear(self):
        for shapes,axis in [([(2,),(3,)],0),([(2,3),(2,3)],1),([(1,2,2),(1,2,2)],2)]:
            b=Builder(shapes);g=b.finish(b.node('concat',['input0','input1'],axis=axis))
            r=prepare(g,PROFILE_SHA256,helper_profile=CONCAT_PROFILE)
            self.assertEqual(r['model'],g);self.assertEqual(len(r['upstream_helpers']['unavailable_bindings']),1)
    def test_rehashed_geometry_rotation_shape_forgery(self):
        r=self.request()
        for field in ['shape','step','geometry','arity','cost']:
            q=copy.deepcopy(r);spec=q['upstream_helpers']['helpers']['HE_Concat0'];b=spec['binding']
            if field=='shape':b['input_shape']=[2]
            elif field=='step':b['rotation']['steps']=[1]
            elif field=='geometry':b['expected_geometry']['nt']=65536
            elif field=='arity':spec['parameters']=['c']
            else:spec['work']=1
            q['request_id']=digest({k:v for k,v in q.items() if k!='request_id'})
            with self.subTest(field=field),self.assertRaises(ValueError):validate_request(q)
    def test_arity_and_direct_signed_rotation_are_rejected(self):
        r=self.request();header='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'
        for source in [header+'    return HE_Concat0(x)\n',header+'    return HE_Concat0(x,1.0)\n',header+'    return HE_Concat0(x.rotate(-3),y)\n']:
            with self.assertRaises(ValueError):validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=source),r)
    def test_discarded_cancelled_and_unused_helper_rejected(self):
        r=self.request();header='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'
        for body in ['    h = HE_Concat0(x,y)\n    return x\n','    h = HE_Concat0(x,y)\n    return h-h+x\n']:
            with self.assertRaisesRegex(ValueError,'contribution'):validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=header+body),r)
    def test_modulo_binary_rotation_is_full_vector_equivalence(self):
        class V:
            def __init__(self,x):self.x=x
            def rotate(self,k):return V(np.roll(self.x,-k))
        for p in [4,8,16,32,64,128,256]:
            for n in range(1,p//2+1):
                k=(-n)%p;steps=[1<<j for j in range(p.bit_length()-1) if k&(1<<j)]
                b={'rotation':dict(requested=-n,normalized=k,steps=steps)};events=[]
                values=np.tile(np.arange(p,dtype=float)**2,16384//p)
                out=PeriodicRotationInput(V(values),b,events).rotate(-n)
                np.testing.assert_array_equal(out.x,np.roll(values,n));self.assertEqual(events,[b['rotation']])
        with self.assertRaises(ValueError):PeriodicRotationInput(V(values),b,[]).rotate(999)

class V:
    def __init__(self,x):self.x=np.asarray(x,dtype=np.float64)
    def __mul__(self,other):return V(self.x*(other.x if type(other) is V else np.asarray(other,dtype=float)))
    def __add__(self,other):return V(self.x+(other.x if type(other) is V else np.asarray(other,dtype=float)))
    def rotate(self,k):return V(np.roll(self.x,-k))

class ConcatUpstreamVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from upstream_adapters.test_batch_norm import BatchNormAdapterTests
        BatchNormAdapterTests.setUpClass.__func__(cls)
    def test_actual_upstream_vectors_match_math_and_period(self):
        from upstream_helper_coverage import helper_evaluator
        from packed_native_exercises import Cell
        for row in cases()[:12]:
            if row['name']=='concat_frozen_7':continue
            model=row['model'];r=prepare(model,PROFILE_SHA256,helper_profile=CONCAT_PROFILE)
            binding=r['upstream_helpers']['helpers']['HE_Concat0']['binding'];period=r['layout']['input_slot_period']
            for probe in samples(model,4):
                arrays=[np.pad(probe[n['name']].reshape(-1),(0,period-probe[n['name']].size)) for n in model['inputs']]
                values=[V(np.tile(x,16384//period)) for x in arrays]
                out,record=apply(binding,*values,self.helpers,self.mpcb)
                expected=next(iter(evaluate(model,probe).values())).reshape(-1)
                np.testing.assert_allclose(out.x[:len(expected)],expected,atol=0,rtol=0)
                np.testing.assert_array_equal(out.x,np.tile(out.x[:period],16384//period))
                finite=helper_evaluator(r)('HE_Concat0',[Cell('c',tuple(x)) for x in arrays])
                np.testing.assert_array_equal(out.x[:period],finite.values)
                self.assertEqual(record['helper'],'HE_Concat')

if __name__=='__main__':unittest.main()
