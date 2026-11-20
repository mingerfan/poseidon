"""Syntax-only request plumbing must not change native program authority."""
import copy
import unittest
from benchmark_graph import canonical,digest
from benchmark_suite import Builder
from compiler_configuration import PROFILE_SHA256
from unified_graph_contract import prepare,validate_request,validate_candidate,SYNTAX_GUIDANCE

class SyntaxGuidanceTests(unittest.TestCase):
    def model(self):
        b=Builder([(2,)])
        return b.finish(b.node('square',['input0']))

    def request(self,version='syntax-explicit-v1',**kwargs):
        return prepare(self.model(),PROFILE_SHA256,generation_guidance=version,**kwargs)

    def test_only_opt_in_metadata_and_request_identity_change(self):
        old=self.request('explicit-v1');new=self.request()
        for r in (old,new):validate_request(r)
        self.assertNotEqual(old['request_id'],new['request_id'])
        self.assertEqual({k:v for k,v in old.items()if k not in ('request_id','generation_guidance')},
                         {k:v for k,v in new.items()if k not in ('request_id','generation_guidance')})
        self.assertEqual({k:v for k,v in old['generation_guidance'].items()if k!='version'},
                         {k:v for k,v in new['generation_guidance'].items()if k not in ('version','native_syntax_rules')})
        self.assertNotIn('logical_model_semantics',new['generation_guidance'])
        self.assertNotIn('typed_request_bindings',new['generation_guidance'])

    def test_rehashed_permission_and_scope_tampering_still_rejected(self):
        for field in ('authority','comprehensions','mutation','calls'):
            r=self.request();r['generation_guidance']['native_syntax_rules'][field]='arbitrary Python is allowed'
            r['request_id']=digest({k:v for k,v in r.items()if k!='request_id'})
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Changed unified generation guidance'):
                validate_request(r)

    def test_original_checker_rejects_comprehension_and_writes_in_both_versions(self):
        bad=['a = [x for i in range(2)]','a = np.array([x],dtype=object)\n a[0] = x',
             'a = x ** 2','a = sum([x,x])','a = x if x else zero_ct']
        for statement in bad:
            source='@hc.func("c,c")\ndef golden(x,zero_ct):\n '+statement+'\n return x\n'
            for version in ('explicit-v1',SYNTAX_GUIDANCE):
                r=self.request(version)
                with self.subTest(statement=statement,version=version),self.assertRaises(ValueError):
                    validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=source),r)

    def test_scalar_and_array_name_augmentation_remain_admitted(self):
        bodies=['a=x\n a+=x\n return a',
                'a=np.array([x,x],dtype=object)\n a+=x\n return a[0]',
                'a=x\n for i in range(2):\n  a=a+x\n return a',
                'a,b=(x,zero_ct)\n return a+b']
        for body in bodies:
            source='@hc.func("c,c")\ndef golden(x,zero_ct):\n '+body+'\n'
            checks=[]
            for version in ('explicit-v1',SYNTAX_GUIDANCE):
                r=self.request(version)
                checks.append(validate_candidate(dict(schema=1,request_id=r['request_id'],hecate_source=source),r))
            self.assertEqual(checks[0],checks[1])

    def test_source_options_provider_and_cli_round_trip(self):
        from component_contract import reconstruct_request,request_options,runner_options
        from deepseek_provider import public_request
        from run_candidate import parse_args,forward_options
        r=self.request();self.assertEqual(reconstruct_request(r),r)
        self.assertEqual(public_request(r),r)
        self.assertEqual(request_options(r)['generation_guidance'],SYNTAX_GUIDANCE)
        flags=runner_options(r);self.assertEqual(flags[flags.index('--unified-guidance')+1],SYNTAX_GUIDANCE)
        a=parse_args(['--case','scripts/baseline/cases/unified-two-input-two-output.json','--prepare','--unified-guidance',SYNTAX_GUIDANCE])
        self.assertIn(SYNTAX_GUIDANCE,forward_options(a))

    def test_other_construction_and_native_execution_profiles_require_own_guidance(self):
        with self.assertRaisesRegex(ValueError,'ordinary native DSL contract'):
            self.request(construction_profile='hecate-unified-public-v1')
        from native_execution_profile import configuration
        cfg=configuration()
        with self.assertRaisesRegex(ValueError,'ordinary native DSL contract'):
            prepare(self.model(),cfg['ckks_config_sha256'],cfg,generation_guidance=SYNTAX_GUIDANCE)

    def test_no_model_specific_algorithm_is_added_for_another_graph(self):
        b=Builder([(3,),(3,)])
        m=b.finish(b.node('add',['input0','input1']))
        other=prepare(m,PROFILE_SHA256,generation_guidance=SYNTAX_GUIDANCE)
        self.assertEqual(self.request()['generation_guidance']['native_syntax_rules'],
                         other['generation_guidance']['native_syntax_rules'])
        self.assertNotIn(m['id'],canonical(other['generation_guidance']).decode())

if __name__=='__main__':unittest.main()
