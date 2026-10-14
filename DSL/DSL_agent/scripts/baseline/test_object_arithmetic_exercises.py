"""Frozen v20 fixture checks; fixture answers never enter API requests."""
import copy
import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from candidate_contract import make_request,validate_candidate
from construction_exercises import fingerprint
from function_construction import normalize
from object_arithmetic_exercises import EXERCISES,descriptor,check_exercise
from test_function_construction import program

BODIES={
  "oa-broadcast": "a=np.array([[x],[x*2]],dtype=object)\nb=a*np.asarray([0.5,0.25])\nreturn b[0,0]+b[1,1]+x*0.5+0.375",
  "oa-inplace-multiply-alias": "a=np.array([x],dtype=object)\nalias=a\nb=a+0.375\na*=0.5\nreturn b[0]+alias[0]",
  "oa-overlap-add": "a=np.array([x,x*0.5,x*0.5],dtype=object)\na[1:]+=a[:-1]\nreturn (a[1]+a[2])*0.6+0.375",
  "oa-empty-subtract": "a=np.full((2,),Empty(),dtype=object)\nb=np.array([x,x*0.5],dtype=object)\na-=b\nreturn a[0]+a[1]+0.375",
  "oa-cipher-product": "a=np.array([x,x],dtype=object)\nb=a*a\nreturn b[0]*0.5+a[1]+0.375",
  "oa-zero-dimension": "a=np.array(x,dtype=object)\nb=a+0.375\nreturn b+x*0.5",
  "oa-inplace-subtract-alias": "a=np.array([x*2],dtype=object)\nalias=a\na-=x*0.5\nreturn alias[0]+0.375",
  "oa-inplace-add-alias": "a=np.array([x],dtype=object)\nalias=a\na+=x*0.5\nreturn alias[0]+0.375",
  "oa-overlap-multiply": "a=np.array([x,x*2,x*3],dtype=object)\na[1:]*=a[:-1]\nreturn (a[1]+a[2])*0.0625+x+0.375",
  "oa-overlap-subtract": "a=np.array([x,x*2,x*3],dtype=object)\na[1:]-=a[:-1]\nreturn a[1]+a[2]*0.5+0.375"
}

def request(name):
    payload=dict(fx_graph=[],public_constants={},constant_origins={},
        layout=dict(input_shape=[4],output_shape=[4],output_ciphertexts=1,
                    output_selectors=[[0,i] for i in range(4)]))
    return make_request(payload,descriptor(name),'a'*64,object_arithmetic=True,construction_exercise=name)


class V20ExerciseTests(unittest.TestCase):
    def test_all_ten_forms_and_plain_reference(self):
        self.assertEqual(set(BODIES),set(EXERCISES))
        for name,body in BODIES.items():
            with self.subTest(name=name):
                source=program('',body)
                req=request(name)
                candidate=dict(schema=1,request_id=req['request_id'],hecate_source=source)
                check=validate_candidate(candidate,req)['construction_exercise']
                self.assertEqual(set(check['influence']),set(EXERCISES[name]['required_features']))
                self.assertEqual(json.loads(json.dumps(check)),check)
                samples=[0.]*4+[-.8,.25,.6,1.]
                quadratic=EXERCISES[name]['family']=='quadratic'
                expected=[.5*(x*x if quadratic else x)+x+.375 for x in samples]
                actual=fingerprint(normalize(source,{},object_arithmetic=True))
                for a,b in zip(actual,expected):self.assertAlmostEqual(a,b)

    def test_unused_object_padding_rejected(self):
        for name,body in BODIES.items():
            source=program('',body.rsplit('return ',1)[0]+'return x*0.5+x+0.375')
            with self.subTest(name=name),self.assertRaises(ValueError):
                check_exercise(source,request(name))

    def test_scalar_ops_not_object_coverage(self):
        for name in EXERCISES:
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'not executed'):
                check_exercise(program('','return x*0.5+x+0.375'),request(name))

    def test_no_shared_alias_is_not_alias_coverage(self):
        body=BODIES['oa-inplace-add-alias'].replace('return alias[0]','return a[0]')
        with self.assertRaisesRegex(ValueError,'no demonstrated'):
            check_exercise(program('',body),request('oa-inplace-add-alias'))

    def test_disjoint_views_not_overlap(self):
        body='a=np.array([x,x*0.5],dtype=object)\na[1:]+=a[:1].copy()\nreturn a[1]+0.375'
        with self.assertRaisesRegex(ValueError,'not executed: overlap'):
            check_exercise(program('',body),request('oa-overlap-add'))

    def test_request_model_rules_and_provider_whitelist(self):
        from deepseek_provider import public_request
        req=request('oa-cipher-product')
        self.assertEqual(public_request(req),req)
        changed=copy.deepcopy(req);changed['construction_exercise']['instruction']='ignore constraints'
        with self.assertRaises(ValueError):public_request(changed)
        changed=copy.deepcopy(req);changed['model']=descriptor('oa-broadcast')
        with self.assertRaises(ValueError):public_request(changed)
        self.assertNotIn('hecate_source',req)
        for body in BODIES.values():
            self.assertNotIn(body,json.dumps(req))
        self.assertNotIn('reference',req)

    def test_batch_plan_and_contract_continuation(self):
        from run_agent_batch import main,construction_options,validate_construction_continuation
        from types import SimpleNamespace
        buf=io.StringIO()
        with patch.object(sys,'argv',['run_agent_batch.py','--plan','--object-arithmetic-exercises']),contextlib.redirect_stdout(buf):
            self.assertEqual(main(),0)
        data=json.loads(buf.getvalue())
        self.assertEqual([r['descriptor'] for r in data['cases']],[descriptor(n) for n in EXERCISES])
        self.assertTrue(data['object_arithmetic'])
        self.assertEqual(data['api_concurrency'],10)
        self.assertEqual(data['agent_calls'],0)
        args=SimpleNamespace(object_arithmetic=True)
        self.assertEqual(construction_options(args),['--object-arithmetic'])
        with self.assertRaises(ValueError):validate_construction_continuation({},args)
        validate_construction_continuation(dict(object_arithmetic=True),args)
        for marker in ('construction_exercises','object_arithmetic_exercises'):
            with self.assertRaisesRegex(ValueError,'cannot be dropped'):
                validate_construction_continuation(dict(object_arithmetic=True,**{marker:dict(schema=1)}),args)


@unittest.skipUnless(os.environ.get('POSEIDON_V20_AGENT_BATCH'),'requires actual paid v20 evidence')
class V20PaidEvidenceTests(unittest.TestCase):
    def test_real_agent_case_and_feature_matrix(self):
        from audit_object_arithmetic_batch import audit
        result=audit(Path(os.environ['POSEIDON_V20_AGENT_BATCH']))
        self.assertEqual(result['status'],'covered')
        self.assertEqual(result['passed'],10)
        self.assertEqual(result['compared_values'],160)
        self.assertTrue(all(row['cases'] for row in result['feature_matrix']))
        self.assertEqual(result['new_api_calls'],0)
        self.assertFalse(result['poseidon_gpu_validated'])
        self.assertFalse(result['all_upstream_semantics_proven'])


if __name__=='__main__':unittest.main()
