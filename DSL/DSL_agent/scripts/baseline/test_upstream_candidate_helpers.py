"""Capability contract/rejection tests; these are not real FHE evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_suite import Builder
from benchmark_graph import digest
from compiler_configuration import PROFILE_SHA256,configuration
from unified_graph_contract import prepare,validate_request,validate_candidate
from upstream_candidate_helpers import PROFILE,manifest,verify_events,verify_sources,LOCK

SOURCE='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return HE_SiLU(x)\n'
class HelperCapabilityTests(unittest.TestCase):
    def graph(self):
        b=Builder([(2,)])
        return b.finish(b.node('square',['input0']))
    def request(self,**kw):
        return prepare(self.graph(),PROFILE_SHA256,configuration('seal-cpu-eva-w40-v1'),**kw)
    def check(self,source,request=None):
        request=self.request(helper_profile=PROFILE) if request is None else request
        return validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
    def test_opt_in_and_public_request(self):
        old=self.request();new=self.request(helper_profile=PROFILE)
        self.assertNotIn('upstream_helpers',old)
        with self.assertRaises(ValueError): self.check(SOURCE,old)
        self.assertEqual(validate_request(new),4)
        from deepseek_provider import public_request
        self.assertEqual(public_request(new),new)
        self.assertEqual(self.check(SOURCE)['functions']['golden']['result'],'c')
        self.assertGreaterEqual(self.check(SOURCE)['functions']['golden']['expanded_cost'],256)
    def test_legacy_request_identity(self):
        r=self.request()
        self.assertEqual(r,self.request(helper_profile=None))
        self.assertNotIn('upstream_calls',self.check(SOURCE.replace('HE_SiLU(x)','x*x'),r))
    def test_contract_combinations_rejected(self):
        for kw in [dict(construction_profile='hecate-unified-public-v1'),dict(construction='unified-native-call-nested')]:
            with self.assertRaises(ValueError):self.request(helper_profile=PROFILE,**kw)
        with self.assertRaises(ValueError):self.request(helper_profile='arbitrary')
    def test_forged_manifest_rejected_after_rehash(self):
        for field,value in [('profile','anything'),('source_lock_sha256','0'*64),('helpers',{})]:
            r=self.request(helper_profile=PROFILE);r['upstream_helpers'][field]=value
            r['request_id']=digest({k:v for k,v in r.items() if k!='request_id'})
            with self.assertRaises(ValueError):validate_request(r)
    def test_unknown_import_types_keywords_and_collision(self):
        bad=['HE_ReLU(x)','HE_BN(x)','HE_SiLU(1)','HE_SiLU([x])','HE_SiLU(x,zero_ct)','HE_SiLU(v=x)','__import__("os")']
        for expr in bad:
            with self.subTest(expr=expr),self.assertRaises(ValueError):self.check(SOURCE.replace('HE_SiLU(x)',expr))
        for source in ['import poly.Func\n'+SOURCE,SOURCE.replace('    return','    HE_SiLU = x\n    return'),SOURCE.replace('golden(x,zero_ct)','golden(HE_SiLU,zero_ct)')]:
            with self.assertRaises(ValueError):self.check(source)
    def test_work_cannot_hide_in_candidate_helpers(self):
        source='@hc.func("c")\ndef f(v):\n    return HE_SiLU(v)\n'+SOURCE.replace('HE_SiLU(x)','f(f(f(f(x))))')
        with self.assertRaisesRegex(ValueError,'work budget'):self.check(source)
    def test_actual_trace_requires_complete_bound_calls(self):
        r=self.request(helper_profile=PROFILE);checked=self.check(SOURCE,r)
        e=dict(schema=1,profile=PROFILE,request_id=r['request_id'],source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),calls=[dict(c,actual_upstream=True) for c in checked['upstream_calls']],candidate_python_executed=False,output_contribution_proven=False)
        self.assertEqual(verify_events(e,r,SOURCE)['actual_upstream_calls'],1)
        for calls in [[],e['calls']*2,[dict(e['calls'][0],actual_upstream=False)],[dict(e['calls'][0],callee='HE_BN')]]:
            with self.assertRaises(ValueError):verify_events(dict(e,calls=calls),r,SOURCE)
        with self.assertRaises(ValueError):verify_events(dict(e,output_contribution_proven=True),r,SOURCE)
    def test_source_lock_rejects_missing_changed_extra_sources(self):
        from workspace_paths import ROOT
        self.assertEqual(verify_sources()['source_lock_sha256'],manifest()['source_lock_sha256'])
        lock=json.loads(LOCK.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in lock['sources']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
            name='src/poseidon/tools/dacapo/poly-python-wheels.lock.json'
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
            verify_sources(root)
            extra=root/'third_party/dacapo/python/poly/poly/extra.py';extra.write_text('# injected')
            with self.assertRaises(ValueError):verify_sources(root)
            extra.unlink()
            extra=root/'third_party/dacapo/python/poly/poly/Func.pyc';extra.write_bytes(b'not a source-checked cache')
            with self.assertRaises(ValueError):verify_sources(root)
            extra.unlink();p.write_text('{}')
            with self.assertRaises(ValueError):verify_sources(root)
    def test_helper_events_saved_after_lazy_frontend_trace(self):
        import candidate_trace
        text=Path(candidate_trace.__file__).read_text()
        self.assertLess(text.index("save_trace(hc, request, 'validated_native_AST_to_Hecate_functions')"),
                        text.index("Path('/out/upstream-calls.json').write_text"))
    def test_public_loops_keep_call_spans_and_budget(self):
        source=SOURCE.replace('    return HE_SiLU(x)','    y = x\n    for i in range(2):\n        y = HE_SiLU(y)\n    return y')
        checked=self.check(source)
        self.assertEqual(len(checked['upstream_calls']),2)
        self.assertEqual(checked['upstream_calls'][0]['span'],checked['upstream_calls'][1]['span'])

if __name__=='__main__':unittest.main()
