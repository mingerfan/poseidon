"""Immutable BN node binding and profile-separation/rejection tests, not FHE."""
import copy,json,unittest
from benchmark_suite import Builder,apply
from benchmark_graph import digest
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_request,validate_candidate
from upstream_candidate_helpers import PROFILE,BN_PROFILE,manifest
from upstream_adapters.batch_norm_node import bind_node

class BoundBatchNormTests(unittest.TestCase):
    def graph(self,shape=(1,3,2),intermediate=False):
        b=Builder([shape]);x='input0'
        if intermediate:x=b.node('square',[x])
        z=apply(b,x,'batch_norm');return b.finish(z)
    def request(self,g=None,**kwargs):
        return prepare(self.graph() if g is None else g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),helper_profile=BN_PROFILE,**kwargs)
    def test_node_in_composition_uses_original_parameters(self):
        g=self.graph(intermediate=True);r=self.request(g);spec=r['upstream_helpers']['helpers']['HE_BN0']['binding']
        self.assertEqual(spec['input_value'],g['nodes'][0]['outputs'][0]);self.assertEqual(spec['input_shape'],[1,3,2])
        self.assertEqual(spec['mathematical_view']['constants'],g['constants'])
        self.assertEqual(spec['slot_period'],8);self.assertEqual(spec['closure_period'],8)
        source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return HE_BN0(x*x)\n'
        checked=validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=source),r)
        self.assertEqual(checked['upstream_calls'][0]['callee'],'HE_BN0')
    def test_previous_profile_manifest_and_requests_unchanged(self):
        self.assertEqual(manifest(),manifest(self.graph(),PROFILE))
        self.assertNotIn('HE_BN0',manifest(self.graph(),PROFILE)['helpers'])
        r=prepare(self.graph(),PROFILE_SHA256,helper_profile=PROFILE)
        with self.assertRaises(ValueError):validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return HE_BN0(x)\n'),r)
    def test_unsupported_nodes_remain_in_model_and_queue(self):
        r=self.request(self.graph((2,3)))
        self.assertNotIn('HE_BN0',r['upstream_helpers']['helpers'])
        self.assertEqual(len(r['model']['nodes']),1);self.assertEqual(len(r['upstream_helpers']['unavailable_bindings']),1)
        self.assertIn('batch one',r['upstream_helpers']['unavailable_bindings'][0]['reason'])
    def test_multiple_bindings_follow_node_identity_not_model_id(self):
        b=Builder([(1,2),(1,3,2)])
        x=apply(b,'input0','batch_norm');y=apply(b,'input1','batch_norm');g=b.finish(x,y);r=self.request(g)
        h=r['upstream_helpers']['helpers'];self.assertEqual(h['HE_BN0']['binding']['input_shape'],[1,2]);self.assertEqual(h['HE_BN1']['binding']['input_shape'],[1,3,2])
        self.assertEqual(h['HE_BN0']['binding']['slot_period'],8);self.assertEqual(h['HE_BN0']['binding']['closure_period'],2)
        g['id']='an_unseen_graph_name';self.assertEqual(manifest(g,BN_PROFILE),r['upstream_helpers'])
    def test_intermediate_larger_than_period_is_explicit_blocker(self):
        b=Builder([(1,2)]);x=b.node('concat',['input0']*4,axis=1);x=apply(b,x,'batch_norm');x=b.node('sum',[x],axes=[1],keepdims=True);r=self.request(b.finish(x))
        self.assertEqual(r['layout']['input_slot_period'],4)
        self.assertIn('does not fit',r['upstream_helpers']['unavailable_bindings'][0]['reason'])
    def test_rehashed_binding_tampering_rejected(self):
        original=self.request()
        for change in ['parameters','shape','period','node','work','bool']:
            r=copy.deepcopy(original);h=r['upstream_helpers']['helpers']['HE_BN0'];v=h['binding']
            if change=='parameters':next(iter(v['mathematical_view']['constants'].values()))[0]=.999
            elif change=='shape':v['input_shape']=[1,6]
            elif change=='period':v['slot_period']=16
            elif change=='node':v['node_id']='other'
            elif change=='work':h['work']=1
            else:h['bootstrap']=0
            r['request_id']=digest({k:v for k,v in r.items() if k!='request_id'})
            with self.subTest(change=change),self.assertRaises(ValueError):validate_request(r)
    def test_bound_event_geometry_and_parameters_are_exact(self):
        import hashlib
        from upstream_candidate_helpers import verify_events
        from upstream_adapters.batch_norm_node import expected_record
        r=self.request();source='@hc.func("c,c")\n'+'def golden(x,zero_ct):\n    return HE_BN0(x)\n'
        c=validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=source),r)
        binding=r['upstream_helpers']['helpers']['HE_BN0']['binding']
        event=dict(schema=1,profile=BN_PROFILE,request_id=r['request_id'],source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   calls=[dict(x,actual_upstream=True) for x in c['upstream_calls']],candidate_python_executed=False,output_contribution_proven=False,
                   bound_calls=[dict(callee='HE_BN0',binding_sha256=digest(binding),actual=expected_record(binding))])
        self.assertEqual(verify_events(event,r,source)['actual_upstream_calls'],1)
        for field in ['ci','hi','pi','q']:
            bad=copy.deepcopy(event);bad['bound_calls'][0]['actual']['geometry'][field]+=1
            with self.assertRaises(ValueError):verify_events(bad,r,source)
        bad=copy.deepcopy(event);bad['bound_calls'][0]['binding_sha256']='0'*64
        with self.assertRaises(ValueError):verify_events(bad,r,source)
    def test_bounded_shapes_and_profiles(self):
        for shape in [(1,1),(1,3),(1,2,3),(1,2,2,3),(1,4,8,8)]:
            r=self.request(self.graph(shape));validate_request(r);self.assertIn('HE_BN0',r['upstream_helpers']['helpers'])
        with self.assertRaises(ValueError):self.request(construction_profile='hecate-unified-public-v1')

if __name__=='__main__':unittest.main()
