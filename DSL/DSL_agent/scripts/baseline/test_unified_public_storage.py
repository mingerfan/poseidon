"""Do not claim numeric dtype-alias effects or values for uninitialized storage."""
import copy
import unittest
from benchmark_suite import Builder
from compiler_configuration import configuration,PROFILE_SHA256
from unified_graph_contract import prepare,validate_candidate
from unified_public_contract import CONTRACT,normalize,event_record
from unified_public_exercises import golden_variant,STORAGE_RECIPES,EVIDENCE_SCOPES
from unified_public_coverage import verify_trace_coverage
from unified_graph_lowering import lower


class StorageCoverageTests(unittest.TestCase):
    def request(self,feature):
        b=Builder([(2,),(2,)]);g=b.finish(b.node('add',['input0','input1']))
        return prepare(g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),
                       'unified-public-'+feature.replace('.','-'),construction_profile=CONTRACT)

    def source(self,body):
        return '@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+line+'\n' for line in body.splitlines())

    def checked(self,feature,body):
        req=self.request(feature)
        return validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=self.source(body)),req)

    def test_each_context_and_output_scope_is_trace_bound(self):
        for feature in STORAGE_RECIPES:
            with self.subTest(feature=feature):
                req=self.request(feature);source=golden_variant(lower(req),req['construction_exercise']['id'],req)
                checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['construction_exercise']
                events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
                trace=dict(events=events,normalized_sha256=expanded['construction']['normalized_sha256'],candidate_python_executed=False)
                got=verify_trace_coverage(checked,trace)
                self.assertEqual(got['evidence_scopes'][feature],EVIDENCE_SCOPES.get(feature,'executed_operation'))
                self.assertTrue(got['finite_influence_checked'])
                tampered=copy.deepcopy(trace);tampered['events']=[]
                with self.assertRaises(ValueError):verify_trace_coverage(checked,tampered)
                if feature.startswith('attr.'):
                    witness=checked['witnesses'][feature]
                    self.assertEqual(witness['typed_context']['direct_marker'],feature[5:])
                    self.assertTrue(witness['typed_context']['floating'])
                    self.assertIn('not_dtype_alias_numeric_effect',witness['evidence'])

    def test_unused_wrong_or_indirect_dtype_markers_do_not_supply_credit(self):
        for marker,other in (('double','float64'),('float64','double')):
            bodies=[
                'unused=np.'+marker+'\nreturn [x]',
                'unused=np.array([0.5],dtype=np.'+marker+')\nreturn [x]',
                'a=np.array([0.5],dtype=np.'+other+')\nreturn [x+a[0]]',
                'd=np.'+marker+'\na=np.array([0.5],dtype=d)\nreturn [x+a[0]]',
                'a=np.array([0.5],dtype="float64")\nreturn [x+a[0]]',
            ]
            for body in bodies:
                with self.subTest(marker=marker,body=body),self.assertRaisesRegex(ValueError,'Missing contributing typed storage'):
                    self.checked('attr.'+marker,body)

    def test_dtype_argument_side_effect_is_not_constructor_result_influence(self):
        for marker in ('double','float64'):
            body='value=x\ndef produce():\n nonlocal value\n value=value+0.5\n return [0.25]\nunused=np.array(produce(),dtype=np.'+marker+')\nreturn [value]'
            with self.subTest(marker=marker),self.assertRaisesRegex(ValueError,'Missing contributing typed storage'):
                self.checked('attr.'+marker,body)

    def test_dtype_aliases_have_equal_normal_numeric_output(self):
        req=self.request('attr.double')
        a=normalize(self.source('v=np.array([1,2],dtype=np.double)\nreturn [x*v[0]+v[1]]'),req)
        b=normalize(self.source('v=np.array([1,2],dtype=np.float64)\nreturn [x*v[0]+v[1]]'),req)
        self.assertNotEqual(a["construction"]["source_sha256"],b["construction"]["source_sha256"])
        a["construction"].pop("source_sha256");b["construction"].pop("source_sha256")
        self.assertEqual(a,b)

    def test_allocation_requires_initialization_and_observable_shape(self):
        for body in [
            'unused=np.empty(2,dtype=object)\nreturn [x]',
            'a=np.empty(2,dtype=object)\na[0]=x\na[1]=y\nreturn [a[0]+a[1]]',
            'a=np.empty(0,dtype=object)\nreturn [x+len(a)]',
        ]:
            with self.subTest(body=body),self.assertRaisesRegex(ValueError,'Missing contributing typed storage'):
                self.checked('call.empty',body)
        with self.assertRaises(ValueError):self.checked('call.empty','a=np.empty(2,dtype=object)\nreturn [x+a[0]]')

    def test_for_loop_requires_actual_iterations_and_body_influence(self):
        for body in [
            'for i in range(0):\n x=x+0.5\nreturn [x]',
            'for i in range(2):\n unused=i+1\nreturn [x]',
            'value=x\ndef produce():\n nonlocal value\n value=value+0.5\n return [0,1]\nfor i in produce():\n unused=i\nreturn [value]',
        ]:
            with self.subTest(body=body),self.assertRaisesRegex(ValueError,'Missing contributing'):
                self.checked('public.loop',body)

    def test_normal_observation_and_unmatched_probe_leave_output_unchanged(self):
        req=self.request('attr.double');source=self.source('a=np.array([0.5],dtype=np.double)\nreturn [x+a[0]]')
        base=normalize(source,req);events=[]
        self.assertEqual(base,normalize(source,req,lambda n:events.append(event_record(n))))
        self.assertEqual(base,normalize(source,req,_storage_probe=dict(span=[999,0,999,1],kind='numeric_constructor_result')))
        with self.assertRaises(ValueError):normalize(source,req,_storage_probe=dict(span=[1,0,1,1],kind='arbitrary'))


if __name__=='__main__':unittest.main()
