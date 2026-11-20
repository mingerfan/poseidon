import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery_component import bound,canonical,sha
import completed_delivery_startup as startup
import public_loop_constraints as loops
from shared_execution import KINDS


class StartupControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.folder=root/'delivery';self.folder.mkdir();st=root.stat()
        self.config=dict(model='deepseek-flash',reasoning_effort='none',max_tokens=32768,timeout_seconds=1200,service_provider='deepseek',provider_retries=0,max_calls=1)
        self.mechanism=bound(dict(source_sha256={},provider_configuration=self.config))
        self.source=b'original source, no compile claim\n'
        metrics=dict(self.config,agent_calls=1,generation_attempts=1,request_attempts=1,transport_retries=0,
            calls=[dict(status='response_received',finish_reason='stop',transport_diagnostics=dict(stage='complete',status=200))])
        self.receipt=bound(dict(mode='delivery_generation',origin='provider_first_output',real_Agent=True,first_output_eligible=True,
            first_generation_index=0,post_output_repairs=0,external_evaluation_performed=False,request_id='r',mechanism_binding=self.mechanism['binding'],
            source_sha256=sha(self.source),source_bytes=len(self.source),status='dsl_delivered',response_sha256='a'*64,provider_metrics=metrics))
        plan=dict(branch='lhy-generalization',source_binding=self.mechanism['binding'],reserve_bytes=1024,wall_seconds=10,stages={k:int(k=='api')for k in KINDS})
        self.owner=dict(pid=123,start_ticks=456,boot_id='retained-boot')
        self.row=dict(plan=plan,plan_binding=sha(canonical(plan)),status='interrupted',active=None,finished=12345,owner=self.owner,
            used={k:int(k=='api')for k in KINDS},artifacts=[dict(path=str(root),device=st.st_dev,inode=st.st_ino)],uncertain_stage=None,
            last_stage=dict(kind='api',number=1,outcome='completed',execution_uncertain=False))

    def check(self,row=None,receipt=None,source=None,observed=None,retained=False):
        return startup._state(self.folder,receipt or self.receipt,self.mechanism,self.source if source is None else source,row or self.row,observed,test_only=True,retained=retained)

    def rebound(self,value):return bound({k:v for k,v in value.items()if k!='binding'})

    def test_completed_terminal_and_dead_interrupted_states_preserve_all_bytes(self):
        before=copy.deepcopy((self.row,self.receipt,self.mechanism,self.source))
        for status in ('succeeded','failed','interrupted'):
            with self.subTest(status=status):
                row=copy.deepcopy(self.row);row['status']=status;result=self.check(row=row)
                self.assertFalse(result['eligible_for_external_assessment']);self.assertTrue(result['test_only']);self.assertEqual(result['API'],0)
        self.assertEqual(before,(self.row,self.receipt,self.mechanism,self.source))

    def test_live_owner_and_uncertain_or_uncompleted_API_are_rejected(self):
        with self.assertRaises(ValueError):self.check(observed=self.owner)
        mutations=[('status','running'),('active',{'kind':'api'}),('uncertain_stage',{'kind':'api'})]
        for key,value in mutations:
            row=copy.deepcopy(self.row);row[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(row=row)
        for key,value in [('outcome','failed'),('execution_uncertain',True),('number',2),('kind','qualify')]:
            row=copy.deepcopy(self.row);row['last_stage'][key]=value
            with self.subTest(stage=key),self.assertRaises(ValueError):self.check(row=row)

    def test_consumed_allowance_and_plan_binding_cannot_be_reset(self):
        for key,value in [('api',0),('api',2),('qualify',1),('api',True)]:
            row=copy.deepcopy(self.row);row['used'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.check(row=row)
        row=copy.deepcopy(self.row);row['plan']['source_binding']='b'*64
        with self.assertRaises(ValueError):self.check(row=row)
        row['plan_binding']=sha(canonical(row['plan']))
        with self.assertRaises(ValueError):self.check(row=row)

    def test_repaired_or_changed_source_counts_and_unaccounted_path_rejected(self):
        for key,value in [('post_output_repairs',1),('origin','retained_response'),('first_generation_index',1),('source_sha256','c'*64)]:
            receipt=self.rebound(dict(self.receipt,**{key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(receipt=receipt)
        receipt=copy.deepcopy(self.receipt);receipt['provider_metrics']['request_attempts']=2;receipt=self.rebound(receipt)
        with self.assertRaises(ValueError):self.check(receipt=receipt)
        row=copy.deepcopy(self.row);row['artifacts'][0]['inode']+=1
        with self.assertRaises(ValueError):self.check(row=row)

    def test_retained_reader_cannot_retroactively_select_future_startup(self):
        result=self.check(retained=True);self.assertEqual(result['mode'],'retained_development_read')
        self.assertFalse(result['eligible_for_external_assessment']);self.assertFalse(result['first_output_success_certified'])
        manifest=startup.source_manifest();mechanism=self.rebound(dict(self.mechanism,source_sha256=manifest))
        with patch.object(startup.external,'verify_selection',lambda v:v):
            selected=startup.bind_startup(mechanism);self.assertNotEqual(selected['binding'],self.mechanism['binding'])
            startup.verify_selection(selected)
            with self.assertRaises(ValueError):startup.verify_selection(mechanism)
            with self.assertRaises(ValueError):startup.bind_startup(selected)
            with patch.object(startup,'source_manifest',lambda:dict(manifest,other='e'*64)),self.assertRaises(ValueError):startup.verify_selection(selected)

    def test_incomplete_HTTP_and_changed_transport_configuration_rejected(self):
        for update in [dict(status='in_flight'),dict(finish_reason=None),dict(transport_diagnostics=dict(stage='headers',status=200)),dict(transport_diagnostics=dict(stage='complete',status=503))]:
            receipt=copy.deepcopy(self.receipt);receipt['provider_metrics']['calls'][0].update(update);receipt=self.rebound(receipt)
            with self.subTest(update=update),self.assertRaises(ValueError):self.check(receipt=receipt)
        receipt=copy.deepcopy(self.receipt);receipt['provider_metrics']['max_tokens']=8192;receipt=self.rebound(receipt)
        with self.assertRaises(ValueError):self.check(receipt=receipt)

    def test_loop_facts_are_original_literals_and_do_not_modify_a_request(self):
        descriptor=loops.describe();self.assertEqual(descriptor['limits']['MAX_RANGE'],128)
        self.assertIn('128 iterations',loops.prompt_text(descriptor));self.assertFalse(descriptor['limits_changed'])
        with patch.object(loops.native_public_loops,'MAX_RANGE',129),self.assertRaises(ValueError):loops.describe()

    def test_disclosure_audit_ignores_weights_and_references_and_preserves_wire(self):
        wire=dict(request_id='r',rules='Public bounded range loops are permitted.',model=dict(constants=dict(weight=[128,4096])),
            generation_guidance=dict(native_syntax_rules=dict(loops='Only bounded public loops.')))
        before=copy.deepcopy(wire);self.assertFalse(loops.audit(wire)['exact_per_range_limit_disclosed']);self.assertEqual(before,wire)
        wire['generation_guidance']['native_syntax_rules']['loops']='Each static range has at most 128 iterations.'
        self.assertTrue(loops.audit(wire)['exact_per_range_limit_disclosed'])


if __name__=='__main__':unittest.main()
