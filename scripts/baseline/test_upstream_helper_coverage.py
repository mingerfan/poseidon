"""Bounded helper influence checks; these tests are not encrypted execution."""
import copy,hashlib,unittest
import numpy as np
from benchmark_graph import digest
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_request,validate_candidate
from upstream_candidate_helpers import BN_PROFILE
from upstream_candidate_cases import cases as silu_cases
from upstream_bn_candidate_cases import cases as bn_cases
from upstream_helper_coverage import helper_evaluator,verify_trace
from packed_native_exercises import Cell

class HelperInfluenceTests(unittest.TestCase):
    def request(self,row,names=None):
        return prepare(row['model'],PROFILE_SHA256,configuration('seal-cpu-eva-w40-v1'),
                       helper_profile=BN_PROFILE,helper_exercise=names or ['HE_BN0'])
    def checked(self,row,source=None,names=None):
        r=self.request(row,names)
        return r,validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=source or row['source']),r)
    def test_three_contexts_each_and_named_outputs(self):
        for row,names in [(r,['HE_SiLU']) for r in silu_cases() if r['name'] in ('silu_frozen_0','silu_nested_native','silu_shared_rotation')]+[(r,['HE_BN0']) for r in bn_cases() if r['name'] in ('bn_asymmetric_0','bn_then_linear','bn_nested_native')]:
            with self.subTest(row=row['name']):
                r,c=self.checked(row,names=names);w=c['upstream_exercise']
                self.assertEqual(set(w['witnesses']),set(names));self.assertFalse(w['plaintext_reference_used'])
                self.assertTrue(all(v['outputs'] for v in w['witnesses'].values()))
    def test_unused_definition_does_not_count(self):
        for row,names in [(silu_cases()[-1],['HE_SiLU']),(bn_cases()[-1],['HE_BN0'])]:
            with self.subTest(row=row['name']),self.assertRaisesRegex(ValueError,'No reachable'):self.checked(row,names=names)
    def test_discarded_cancelled_and_masked_returns_rejected(self):
        row=bn_cases()[0];header='@hc.func("c,c")\ndef golden(x,zero_ct):\n'
        for body in ['    h = HE_BN0(x)\n    return x\n',
                     '    h = HE_BN0(x)\n    return h-h+x\n',
                     '    a = HE_BN0(x)\n    b = HE_BN0(x)\n    return a-b+x\n',
                     '    h = HE_BN0(x)\n    return h*0.0+x\n']:
            with self.subTest(body=body),self.assertRaisesRegex(ValueError,'contribution'):self.checked(row,header+body)
    def test_every_requested_binding_must_contribute(self):
        row=next(r for r in bn_cases() if r['name']=='bn_multi_residual')
        r,c=self.checked(row,names=['HE_BN1','HE_BN0'])
        self.assertEqual(r['upstream_exercise']['required_helpers'],['HE_BN0','HE_BN1'])
        source=row['source'].replace('HE_BN1(y)','y')
        with self.assertRaisesRegex(ValueError,'No reachable'):self.checked(row,source,names=['HE_BN0','HE_BN1'])
    def test_unknown_duplicate_and_unconfigured_exercises_rejected(self):
        row=bn_cases()[0]
        for names in [['HE_ReLU'],['HE_BN0','HE_BN0'],[True],[]]:
            with self.subTest(names=names),self.assertRaises(ValueError):
                prepare(row['model'],PROFILE_SHA256,helper_profile=BN_PROFILE,helper_exercise=names)
        with self.assertRaises(ValueError):prepare(row['model'],PROFILE_SHA256,helper_exercise=['HE_SiLU'])
    def test_rehashed_contract_forgery_rejected(self):
        r=self.request(bn_cases()[0])
        for key,value in [('probes',1),('individual_return',False),('schema',True),('extra','x')]:
            q=copy.deepcopy(r);q['upstream_exercise'][key]=value;q['request_id']=digest({k:v for k,v in q.items() if k!='request_id'})
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(q)
    def test_missing_or_rebound_real_trace_rejected(self):
        row=bn_cases()[0];r,c=self.checked(row);w=c['upstream_exercise'];site=w['witnesses']['HE_BN0']['trace']
        records=dict(request_id=r['request_id'],source_sha256=w['source_sha256'],calls=[{k:v for k,v in site.items() if k!='kind'}|dict(actual_upstream=True)])
        self.assertTrue(verify_trace(w,records)['finite_return_influence_checked'])
        for change in ['missing','span','source','actual']:
            e=copy.deepcopy(records)
            if change=='missing':e['calls']=[]
            elif change=='span':e['calls'][0]['span'][0]+=1
            elif change=='source':e['source_sha256']='0'*64
            else:e['calls'][0]['actual_upstream']=False
            with self.subTest(change=change),self.assertRaises(ValueError):verify_trace(w,e)
    def test_full_bn_period_padding_and_repetition(self):
        from benchmark_suite import Builder
        from upstream_bn_candidate_cases import bn
        b=Builder([(1,3),(8,)]);h=bn(b,'input0');r=prepare(b.finish(h,'input1'),PROFILE_SHA256,helper_profile=BN_PROFILE)
        spec=r['upstream_helpers']['helpers']['HE_BN0']['binding'];self.assertEqual(spec['closure_period'],4)
        y=helper_evaluator(r)('HE_BN0',[Cell('c',(1.,)*8)]).values
        self.assertEqual(y[3],0.);self.assertEqual(y[7],0.);self.assertEqual(y[:4],y[4:])
    def test_probe_polynomial_matches_independent_math_prefix(self):
        from benchmark_math import evaluate
        row=silu_cases()[0];r=self.request(row,['HE_SiLU']);p=r['layout']['input_slot_period']
        x=np.linspace(-.4,.4,p);actual=helper_evaluator(r)('HE_SiLU',[Cell('c',tuple(x))]).values
        shape=row['model']['inputs'][0]['shape'];n=np.prod(shape)
        ref=evaluate(row['model'],{'input0':x[:n].reshape(shape)})
        np.testing.assert_allclose(actual[:n],next(iter(ref.values())).reshape(-1),atol=1e-13,rtol=1e-13)
    def test_frozen_helper_task_contexts_and_no_answers(self):
        from upstream_helper_directed_cases import tasks
        rows=tasks();self.assertEqual(len(rows),60)
        for name in ('HE_SiLU','HE_BN0','HE_Concat0','HE_Conv0','HE_Avg0','HE_Pool0','HE_ConvBN0','HE_DwConv0','HE_DS0','HE_MPBN0','HE_Linear0','HE_ReshapeLinear0'):
            self.assertGreaterEqual(len({r['topology'] for r in rows if name in r['required_helpers']}),3)
        for family in ("HE_MPBN","HE_Linear","HE_ReshapeLinear"):
            contexts={r["topology"] for r in rows if r.get("chunk_period") is not None
                      and any(n.startswith(family) for n in r["required_helpers"])}
            self.assertGreaterEqual(len(contexts),3)
        for r in rows:
            self.assertNotIn('source',r);self.assertEqual(r['execution_status'],'not_run')
            self.assertEqual(r['task_sha256'],digest({k:v for k,v in r.items() if k!='task_sha256'}))
    def test_no_exercise_preserves_free_request(self):
        row=bn_cases()[0];a=prepare(row['model'],PROFILE_SHA256,helper_profile=BN_PROFILE)
        b=prepare(row['model'],PROFILE_SHA256,helper_profile=BN_PROFILE,helper_exercise=None)
        self.assertEqual(a,b);self.assertNotIn('upstream_exercise',a)

if __name__=='__main__':unittest.main()
