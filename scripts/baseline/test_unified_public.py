"""Public construction ABI tests; numerical probes here are plaintext, not FHE."""
import ast
import copy
import unittest
import numpy as np
from benchmark_suite import Builder
from benchmark_graph import samples,digest
from benchmark_math import evaluate
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_candidate,validate_request
from unified_graph_lowering import lower
from unified_public_contract import CONTRACT,normalize,event_record
from test_unified_graph import probe

RECIPES={
"closure":('def public_helper(a):\n    value = a\n    def advance():\n        nonlocal value\n        value = value * 0.5\n    advance()\n    return value\n',
           'covered_out = public_helper(covered_out) * 2.0\n'),
"binding":('def public_helper(a, /, factor=0.5, *, bias=0.25):\n    stash = {"v": a * factor}\n    return stash["v"] + bias\n',
           'covered_out = (public_helper(covered_out, bias=0.25) - 0.25) * 2.0\n'),
"control":('', 'i = 0\ncurrent = covered_out * 0.5\nwhile i < 2:\n    i += 1\n    if i == 1:\n        continue\n    current = current + covered_out * 0.5\nelse:\n    current = current + 0.25\ncovered_out = current - 0.25\n'),
"numeric":('', 'factor = float(" 0.5 ".strip())\nweights = np.array([factor]*8, dtype="float64")\ncovered_out = covered_out * weights * 2.0\n'),
"object":('', 'storage = np.array([covered_out * 0.5, covered_out * 0.5], dtype=object)\nview = storage[:1]\nstorage += covered_out * 0.25\ncovered_out = view[0] + storage[1] - covered_out * 0.5\n')}

def variant(source,name):
    tree=ast.parse(source);g=tree.body[0];ret=g.body[-1]
    helper,body=RECIPES[name]
    body='covered_out = '+ast.unparse(ret.value.elts[0])+'\n'+body
    g.body[-1:-1]=ast.parse(body).body
    ret.value.elts[0]=ast.Name(id='covered_out',ctx=ast.Load())
    tree.body[:0]=ast.parse(helper).body
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'

def model():
    b=Builder([(2,3),(6,)])
    y=b.node('reshape',['input1'],shape=[2,3]);z=b.node('subtract',['input0',y])
    avg=b.node('mean',[z],axes=[1],keepdims=True)
    return b.finish(z,avg)

class UnifiedPublicTests(unittest.TestCase):
    def request(self,g=None):
        return prepare(g or model(),PROFILE_SHA256,configuration('seal-cpu-eva-w45-v1'),construction_profile=CONTRACT)
    def test_public_profiles_preserve_multishape_multioutput_math(self):
        req=self.request()
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                source=variant(lower(req),recipe)
                checked=validate_candidate(dict(schema=1,request_id=req['request_id'],hecate_source=source),req)
                self.assertEqual(checked['construction_profile'],CONTRACT)
                events=[];expanded=normalize(source,req,lambda node:events.append(event_record(node)))
                self.assertTrue(events)
                import json
                json.dumps(events,allow_nan=False)
                if recipe=='numeric':self.assertTrue(any('event.scalar_cast' in e['features'] and e['facts'] for e in events))
                if recipe=='object':self.assertTrue(any('event.object_inplace' in e['features'] and e['facts'] for e in events))
                self.assertFalse(expanded['construction']['candidate_python_executed'])
                self.assertEqual(expanded['construction']['slot_period'],8)
                for inputs in samples(req['model'],4):
                    expected=np.concatenate([v.reshape(-1) for v in evaluate(req['model'],inputs).values()])
                    bound=dict(req,public_constants=expanded['constants'])
                    np.testing.assert_allclose(probe(bound,expanded['source'],inputs),expected,atol=1e-12,rtol=1e-12)
    def test_old_native_requests_and_contracts_unchanged(self):
        g=model();native=prepare(g,PROFILE_SHA256)
        self.assertNotIn('construction_profile',native)
        with self.assertRaises(ValueError):
            validate_candidate(dict(schema=1,request_id=native['request_id'],hecate_source=variant(lower(native),'closure')),native)
        from function_construction import normalize as expand
        source='@hc.func("c")\ndef golden(x):\n return x.rotate(4)\n'
        with self.assertRaises(ValueError):expand(source,{},public_mappings=True)
    def test_public_free_preparation_does_not_require_rule_program(self):
        from unittest.mock import patch
        from semantic_benchmark_execution import preflight
        g=model();row=dict(model=g,model_sha256=digest(g),category='test')
        with patch('unified_graph_lowering.lower',side_effect=ValueError('rule unavailable')):
            good=preflight([row],need_rule=False,construction_profile=CONTRACT)[0]
            bad=preflight([row],need_rule=True,construction_profile=CONTRACT)[0]
        self.assertEqual(good['status'],'ready');self.assertEqual(bad['status'],'blocked_rule_preflight')

    def test_periods_and_wrong_rotation_rejected(self):
        from function_construction import normalize as expand
        from hecate_contract import UNIFIED_FLAT_CONTRACT
        for p in (4,8,16,32,64,128,256):
            source='@hc.func("c,c")\ndef golden(x,zero_ct):\n return x.rotate('+str(p//2)+')\n'
            result=expand(source,{},input_names=('x','zero_ct'),public_mappings=True,flat_contract=UNIFIED_FLAT_CONTRACT,slot_period=p)
            self.assertEqual(result['construction']['slot_period'],p)
        req=self.request()
        for step in (-1,3,8):
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n return [x.rotate('+str(step)+'),y]\n'
            with self.subTest(step=step),self.assertRaises(ValueError):normalize(source,req)
    def test_profile_tampering_and_native_exercise_mixing_rejected(self):
        req=self.request()
        for field,value in [('construction_profile','anything'),('rules','native')]:
            bad=copy.deepcopy(req);bad[field]=value;bad['request_id']=digest({k:v for k,v in bad.items() if k!='request_id'})
            with self.assertRaises(ValueError):validate_request(bad)
        with self.assertRaises(ValueError):
            prepare(model(),PROFILE_SHA256,construction='unified-view',construction_profile=CONTRACT)
    def test_encrypted_control_io_and_resource_overflow_rejected(self):
        req=self.request()
        for body in ['return [x if x else y,y]','import os\n return [x,y]',
                     'while True:\n  x = x + y\n return [x,y]',
                     'a = [x]*129\n return [a[0],y]',
                     'return [x.rotate(3),y]']:
            source='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n '+body.replace('\n','\n ')+'\n'
            with self.subTest(body=body),self.assertRaises(ValueError):normalize(source,req)

if __name__=='__main__':unittest.main()
