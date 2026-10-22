"""Typed storage identity must be observable, not merely a copy/view spelling."""
import copy
import unittest
import numpy as np
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from unified_public_contract import CONTRACT,normalize,event_record
from unified_public_exercises import SPECS,COMPOSITES,VIEW_RECIPES,golden_variant
from unified_public_coverage import verify_trace_coverage
from audit_semantic_benchmark import verify_directed_coverage

class PublicViewTests(unittest.TestCase):
    def request(self,feature):
        b=Builder([(2,),(2,)]);g=b.finish(b.node('add',['input0','input1']))
        name=next(k for k,v in SPECS.items() if v[0]==feature)
        return prepare(g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),name,construction_profile=CONTRACT)
    def source(self,body):
        return '@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+s+'\n' for s in body.splitlines())
    def checked(self,req,source):
        return validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['construction_exercise']
    def test_storage_oracle_matches_numpy_including_noncontiguous_reshape(self):
        import object_arrays as objects
        a=objects.array([[1.,2.],[3.,4.]])
        x=np.array([[1.,2.],[3.,4.]],dtype=object)
        for name,left,right in [('copy',objects.copy(a),x.copy()),('reshape',objects.reshape(a,(4,)),x.reshape(4)),
                                ('transpose',objects.transpose(a),x.T)]:
            f=objects.storage_facts(a,left,name)
            self.assertEqual(f['shares_storage'],np.shares_memory(x,right))
            objects.put(a,(0,0),5.);x[0,0]=5.
            np.testing.assert_array_equal(left.data,right)
        view=objects.transpose(a);flat=objects.reshape(view,(4,))
        self.assertFalse(objects.storage_facts(view,flat,'reshape')['shares_storage'])
    def test_typed_witnesses_and_scope_are_trace_bound(self):
        for feature in VIEW_RECIPES:
            req=self.request(feature);source=golden_variant(lower(req),req['construction_exercise']['id'],req)
            checked=self.checked(req,source);events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
            trace=dict(events=events,normalized_sha256=expanded['construction']['normalized_sha256'],candidate_python_executed=False)
            coverage=verify_trace_coverage(checked,trace)
            self.assertEqual(coverage['evidence_scopes'][feature],'object_storage_alias_distinction')
            fact=checked['witnesses'][feature]['typed_context']
            self.assertEqual(fact['shares_storage'],feature!='storage.copy_independence')
            bad=copy.deepcopy(trace);bad['events']=[e for e in bad['events'] if 'event.object_storage' not in e['features']]
            with self.assertRaises(ValueError):verify_trace_coverage(checked,bad)
    def test_equal_values_unused_or_unobserved_storage_does_not_count(self):
        for feature,operation,tail in [
            ('storage.copy_independence','a.copy()','b[0,0]'),
            ('storage.reshape_view','a.reshape(2)','b[0]'),
            ('storage.transpose_view','a.transpose()','b[0,0]')]:
            req=self.request(feature)
            for rest in ['return ['+tail+']','a[0,0]=a[0,0]+0.5\nreturn [a[0,0]]',
                         'a[0,1]=a[0,1]+0.5\nreturn ['+tail+']']:
                body='a=np.array([[x,y]],dtype=object)\nb='+operation+'\n'+rest
                with self.subTest(feature=feature,body=body),self.assertRaisesRegex(ValueError,'Missing observable'):
                    self.checked(req,self.source(body))
    def test_detached_reshape_is_not_a_view(self):
        req=self.request('storage.reshape_view')
        body='a=np.array([[x,y],[zero_ct+0.25,zero_ct+0.5]],dtype=object)\nv=a.T\nb=v.reshape(4)\na[0,0]=a[0,0]+0.5\nreturn [b[0]]'
        with self.assertRaisesRegex(ValueError,'Missing observable'):self.checked(req,self.source(body))
    def test_transpose_attribute_and_copy_receiver_effects(self):
        req=self.request('storage.transpose_view')
        body='a=np.array([[x,y]],dtype=object)\nb=a.T\na[0,0]=a[0,0]+0.5\nreturn [b[0,0]]'
        self.checked(req,self.source(body))
        req=self.request('storage.copy_independence')
        body='value=x\ndef make():\n nonlocal value\n value=value+0.5\n return np.array([y],dtype=object)\nb=make().copy()\nreturn [value]'
        with self.assertRaisesRegex(ValueError,'Missing observable'):self.checked(req,self.source(body))
    def test_composites_require_every_child_and_each_scope(self):
        for feature,children in COMPOSITES.items():
            req=self.request(feature);source=golden_variant(lower(req),req['construction_exercise']['id'],req)
            checked=self.checked(req,source);self.assertEqual(list(checked['witnesses']),list(children))
            for child in children:
                name=next(k for k,v in SPECS.items() if v[0]==child)
                with self.subTest(feature=feature,child=child),self.assertRaises(ValueError):
                    self.checked(req,golden_variant(lower(req),name,req))
            events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
            trace=dict(events=events,normalized_sha256=expanded['construction']['normalized_sha256'],candidate_python_executed=False)
            coverage=verify_trace_coverage(checked,trace)
            task=dict(requirement=feature,required_features=list(children),structural_features=[],acceptance_kind='numeric_influence')
            verify_directed_coverage(coverage,task,req['construction_exercise'])
            bad=copy.deepcopy(coverage);bad['numeric_features'].pop()
            with self.assertRaises(ValueError):verify_directed_coverage(bad,task,req['construction_exercise'])
            if feature=='array.view_copy':
                bad=copy.deepcopy(coverage);bad['evidence_scopes'].pop(children[0])
                with self.assertRaisesRegex(ValueError,'scope'):verify_directed_coverage(bad,task,req['construction_exercise'])
    def test_unmatched_probe_and_observation_leave_normal_program_unchanged(self):
        req=self.request('storage.copy_independence');source=self.source('a=np.array([x],dtype=object)\nb=a.copy()\nreturn [b[0]]')
        base=normalize(source,req);events=[]
        self.assertEqual(base,normalize(source,req,lambda n:events.append(event_record(n))))
        self.assertEqual(base,normalize(source,req,_storage_probe=dict(span=[999,0,999,1],kind='copy_alias')))

if __name__=='__main__':unittest.main()
