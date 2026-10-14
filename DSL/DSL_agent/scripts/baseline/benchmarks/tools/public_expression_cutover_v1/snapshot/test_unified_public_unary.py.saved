"""Typed unary contribution and storage probes; plaintext tests are not FHE."""
import copy
import unittest
from benchmark_suite import Builder
from compiler_configuration import configuration,PROFILE_SHA256
from unified_graph_contract import prepare,validate_candidate
from unified_public_contract import CONTRACT,normalize,event_record
from unified_public_exercises import golden_variant,UNARY_RECIPES
from unified_public_coverage import verify_trace_coverage,fingerprint
from unified_graph_lowering import lower


class UnaryCoverageTests(unittest.TestCase):
    def request(self,feature):
        b=Builder([(2,),(2,)])
        g=b.finish(b.node('add',['input0','input1']))
        return prepare(g,PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),
                       'unified-public-'+feature.replace('.','-'),construction_profile=CONTRACT)

    def source(self,body):
        return '@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'+''.join(' '+line+'\n' for line in body.splitlines())

    def validate(self,feature,body):
        req=self.request(feature)
        return validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=self.source(body)),req)

    def test_each_typed_partition_has_trace_bound_finite_witness(self):
        for feature in UNARY_RECIPES:
            with self.subTest(feature=feature):
                req=self.request(feature);source=golden_variant(lower(req),req['construction_exercise']['id'],req)
                checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)['construction_exercise']
                events=[];expanded=normalize(source,req,lambda n:events.append(event_record(n)))
                trace=dict(events=events,normalized_sha256=expanded['construction']['normalized_sha256'],candidate_python_executed=False)
                got=verify_trace_coverage(checked,trace)
                self.assertEqual(got['numeric_features'],[feature])
                self.assertTrue(got['finite_influence_checked'])
                witness=checked['witnesses'][feature]
                self.assertIn('intervention_sha256',witness)
                if feature.endswith('_cells'):
                    self.assertEqual(witness['intervention']['kind'],'typed_unary_result')
                elif feature.startswith('fresh.'):
                    self.assertEqual(witness['intervention']['kind'],'fresh_storage_vs_alias')
                forged=copy.deepcopy(trace)
                forged['events']=[e for e in forged['events'] if 'event.object_unary' not in e['features']]
                with self.assertRaises(ValueError):verify_trace_coverage(checked,forged)

    def test_unused_cells_of_required_type_are_not_credited(self):
        cases={
            'cipher_cells':'a=np.array([x,0.5],dtype=object)\nb=-a\nreturn [x+b[1]]',
            'plain_cells':'p=Empty()+0.5\na=np.array([p,x],dtype=object)\nb=-a\nreturn [-b[1]]',
            'public_cells':'a=np.array([x,0.5],dtype=object)\nb=-a\nreturn [-b[0]]',
            'boolean_cells':'a=np.array([x,True],dtype=object)\nb=-a\nreturn [-b[0]]',
        }
        for feature,body in cases.items():
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,'Missing contributing typed unary'):
                self.validate(feature,body)

    def test_wrong_types_and_syntax_are_rejected(self):
        cases={
            'cipher_cells':'a=np.array([0.5],dtype=object)\nb=-a\nreturn [x+b[0]]',
            'plain_cells':'a=np.array([0.5],dtype=object)\nb=-a\nreturn [x+b[0]]',
            'boolean_cells':'a=np.array([1],dtype=object)\nb=-a\nreturn [x+b[0]]',
            'public_cells':'a=np.array([True],dtype=object)\nb=-a\nreturn [x+b[0]]',
            'call.USub':'a=np.array([x],dtype=object)\nb=-a\nreturn [b[0]]',
            'operator.USub':'a=np.array([x],dtype=object)\nb=np.negative(a)\nreturn [b[0]]',
            'input_view':'a=np.array([x],dtype=object)\nb=-a\nreturn [b[0]]',
        }
        for feature,body in cases.items():
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,'Missing contributing typed unary'):
                self.validate(feature,body)

    def test_storage_distinction_must_be_observed(self):
        cases={
            'fresh.UAdd':['a=np.array([0.5],dtype=object)\nb=+a\nreturn [x+b[0]]',
                          'a=np.array([0.5],dtype=object)\nb=+a\na[0]=0.5\nreturn [x+b[0]]'],
            'fresh.USub':['a=np.array([x],dtype=object)\nview=a[:1]\nb=-view\nreturn [b[0]]'],
        }
        for feature,bodies in cases.items():
            for body in bodies:
                with self.subTest(feature=feature,body=body),self.assertRaisesRegex(ValueError,'Missing contributing typed unary'):
                    self.validate(feature,body)

    def test_typed_probe_only_changes_requested_result_kind(self):
        req=self.request('boolean_cells')
        source=self.source('a=np.array([x,0.5,True],dtype=object)\nb=-a\nreturn [b[0]+b[1]+b[2]]')
        events=[];base=normalize(source,req,lambda n:events.append(event_record(n)))
        event=next(e for e in events if 'event.object_unary' in e['features'])
        changed=normalize(source,req,_unary_probe=dict(span=event['span'],kind='bool'))
        before=fingerprint(base,req);after=fingerprint(changed,req)
        self.assertTrue(all(abs(a-b-1)<1e-12 for a,b in zip(after,before)))
        self.assertEqual(event['facts']['cipher_cells'],1)
        self.assertEqual(event['facts']['public_real_cells'],1)
        self.assertEqual(event['facts']['boolean_cells'],1)
        with self.assertRaises(ValueError):normalize(source,req,_unary_probe=dict(span=event['span'],kind='any'))

    def test_unused_operand_side_effect_does_not_credit_unary(self):
        body='value=x\ndef produce():\n nonlocal value\n value=value+0.5\n return np.array([x],dtype=object)\nunused=-produce()\nreturn [value]'
        for feature in ('cipher_cells','operator.USub'):
            with self.subTest(feature=feature),self.assertRaisesRegex(ValueError,'Missing contributing typed unary'):
                self.validate(feature,body)


if __name__=='__main__':unittest.main()
