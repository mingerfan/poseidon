"""Synthetic lineage and orchestration only; no array/model/compiler execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import bert_budget_fork_compile as entry
from pt_agent.tests import test_bert_retained_budget_confirm as fixture_module


class ForkCompileTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture_module.RecoveryTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.root
        self.info_path=self.f.case/'model/model_info.json'
        info=entry.base.read(self.info_path);info.update(inputs={'inputs_embeds':[1,40,2]},sequence=dict(length=40))
        info['source']['type']='hf';fixture_module.put(self.info_path,info)
        self.profile_dir=self.root/'profiles';self.dacapo=self.root/'dacapo'
        self.patchers=[mock.patch.object(entry.paths,'PROFILES_DIR',self.profile_dir),
            mock.patch.object(entry.paths,'DACAPO_ROOT',self.dacapo),
            mock.patch.object(entry.base.original,'compiler_argv',return_value=['synthetic-native-command'])]
        for p in self.patchers:p.start();self.addCleanup(p.stop)
        profile=self.profile_dir/'qwen48/compiler-config.json';fixture_module.put(profile,dict(synthetic=True))
        frontend=self.dacapo/'python/hecate/hecate/expr.py';frontend.parent.mkdir(parents=True);frontend.write_text('# synthetic\n')
        def finish(*a,**k):
            observation=self.f.finish(selected=getattr(self,'selected',True))
            state=entry.base.read(self.f.out/'session/SESSION.json')
            state['confirmation']=entry.base.read(self.f.out/'session/AGENT.json')['confirm']
            fixture_module.put(self.f.out/'session/SESSION.json',state)
            return observation
        self.finish=finish
        self.fork_result=self.f.execute(runner=finish)
        self.assertEqual(self.fork_result['status'],'ok',self.fork_result.get('error'))
        self.audit=self.root/'checkpoint/result.json'
        from pt_agent.experiments import bert_checkpoint_audit as checkpoint
        worker=dict(status='ok',source=dict(model_info=str(self.info_path),model=self.fork_result['source']['model']),
            state_coverage=dict(passed=True,classifier=dict(both_original_checkpoint_keys=True)),
            **dict.fromkeys(checkpoint.ZERO_FIELDS,0),**dict.fromkeys(('checkpoint_load_attempts','checkpoint_load_completed',
                'from_pretrained_attempts','from_pretrained_completed'),1))
        fixture_module.put(self.audit,dict(status='ok',round=checkpoint.ROUND,model=entry.MODEL,
            checkpoint_semantic_coverage_passed=True,worker=worker,process=fixture_module.clean()))
        self.args=SimpleNamespace(source_case=self.f.out,checkpoint_audit=self.audit,
            original_source_case=self.f.case,continuation_case=self.f.run,boundary_snapshot=self.f.args.boundary_snapshot,
            out=self.root/'compile-observation',compile_out=self.root/'compiled',
            metadata_template=self.f.args.metadata_template,intake_python=self.f.args.intake_python,
            profile=profile,frontend=frontend,remaining_seconds=11160,worker_phase=None,delegate_compile=False,
            work_dir=None,math_result=None,model_info=None,program=None)

    def source(self):return entry.metadata_source(self.args.source_case,self.audit,self.args)

    def test_two_distinct_model_info_paths_and_original_prepare_audit_are_honest(self):
        source=self.source();p=source['fork_provenance']
        self.assertEqual(source['model_info'],str(self.f.out/'model/model_info.json'))
        self.assertEqual(source['original_checkpoint']['model_info'],str(self.info_path))
        self.assertNotEqual(p['original_model_info'],p['copied_fork_model_info'])
        self.assertEqual(p['original_prepare_result'],str(self.f.case/'phases/001-prepare/worker_result.json'))
        self.assertFalse((self.f.out/'phases/001-prepare').exists())
        self.assertEqual(p['snapshot_calls'],3);self.assertEqual(p['completed_search_rounds'],3)
        self.assertEqual(p['original_actual_session_calls'],4);self.assertTrue(p['original_fourth_call_charged'])
        self.assertFalse(p['original_search_and_partial_work_discarded']);self.assertTrue(source['report_deferred'])

    def test_foreign_original_case_boundary_and_continuation_reject(self):
        for key in entry.BRIDGE_FIELDS:
            with self.subTest(key=key):
                args=copy.copy(self.args);setattr(args,key,self.root/'foreign')
                with self.assertRaises(ValueError):entry.metadata_source(args.source_case,self.audit,args)

    def test_copied_metadata_graph_or_calibration_heldout_destination_drift_reject(self):
        for name in ('model_info.json','calibration.json','samples/source.json'):
            p=self.f.out/'model'/name;old=entry.base.read(p);fixture_module.put(p,dict(old,foreign=True))
            with self.subTest(metadata=name),self.assertRaises(ValueError):self.source()
            fixture_module.put(p,old)
        for name in ('graph','samples/calibration.npz','samples/heldout.npz'):
            p=self.f.out/'model'/name;old=p.readlink();p.unlink();p.symlink_to(self.root/'foreign')
            with self.subTest(reference=name),self.assertRaises(ValueError):self.source()
            p.unlink();p.symlink_to(old,target_is_directory=name=='graph')

    def test_semantic_checkpoint_foreign_model_info_or_incomplete_coverage_reject(self):
        with self.assertRaisesRegex(ValueError,'result.json'):
            entry.metadata_source(self.args.source_case,self.audit.with_name('foreign.json'),self.args)
        old=entry.base.read(self.audit)
        for change in ('model_info','model','coverage'):
            record=copy.deepcopy(old)
            if change=='coverage':record['worker']['state_coverage']['passed']=False
            else:record['worker']['source'][change]='/foreign'
            fixture_module.put(self.audit,record)
            with self.subTest(change=change),self.assertRaises(ValueError):self.source()
        fixture_module.put(self.audit,old)

    def test_original_weight_or_cache_file_replaced_by_foreign_alias_refuses(self):
        for name in ('graph/weights.npz','samples/calibration.npz','samples/heldout.npz'):
            target=self.f.case/'model'/name;saved=target.with_name(target.name+'.kept');target.rename(saved)
            foreign=self.root/'foreign-payload';foreign.write_text('synthetic only');target.symlink_to(foreign)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'foreign alias'):self.source()
            target.unlink();saved.rename(target)

    def test_quality_or_frozen_program_drift_rejects_original_c2(self):
        p=self.f.out/'session/AGENT.json';old=entry.base.read(p)
        bad=copy.deepcopy(old);bad['confirm']['evals'][0]['max_u_calibrated']=.91;fixture_module.put(p,bad)
        with self.assertRaises(ValueError):self.source()
        fixture_module.put(p,old)
        source=self.f.out/'session/program.ptdsl';saved=source.read_text();source.write_text('foreign program')
        with self.assertRaises(ValueError):self.source()
        source.write_text(saved)

    def test_source_claim_worker_process_registry_and_completion_fail_closed(self):
        targets=[(self.f.out/'completion.json',lambda d:d.update(registry_persisted=False)),
            (self.f.out/'result.json',lambda d:d.update(completed_confirm=0)),
            (self.root/'bert-budget-stop-c2-claims/source.json',lambda d:d.update(branch='/foreign')),
            (self.f.out/'phases/001-confirm/worker_result.json',lambda d:d.update(llm_calls=1))]
        for p,change in targets:
            old=entry.base.read(p);bad=copy.deepcopy(old);change(bad);fixture_module.put(p,bad)
            with self.subTest(path=p.name),self.assertRaises(ValueError):self.source()
            fixture_module.put(p,old)
        registry=self.root/'registry.jsonl';old=registry.read_text();registry.write_text(old+old)
        with self.assertRaises(ValueError):self.source()

    def test_session_confirmation_cannot_disagree_with_accepted_agent(self):
        p=self.f.out/'session/SESSION.json';state=entry.base.read(p);state['confirmation']['selected']=None
        fixture_module.put(p,state)
        with self.assertRaisesRegex(ValueError,'SESSION'):self.source()

    def test_future_report_marker_is_not_read_and_blocks_compilation(self):
        p=self.f.out/'session/REPORT_ACCESS.json';p.write_text('deliberately unreadable JSON')
        with self.assertRaisesRegex(ValueError,'report access'):self.source()

    def test_plain_namespace_preserves_numeric_bodies_and_original_globals(self):
        original_globals=entry.base.metadata_source.__globals__
        ns=entry.namespace(self.args)
        self.assertIs(ns['math_worker'].__code__,entry.base.math_worker.__code__)
        self.assertIs(ns['_math_api'].__code__,entry.base._math_api.__code__)
        self.assertIs(ns['execute'].__code__,entry.base.execute.__code__)
        self.assertIs(entry.base.metadata_source.__globals__,original_globals)
        self.assertIs(entry.base.serial.serial_lease,entry.serial.serial_lease)
        compiled=ns['compile_namespace']()
        self.assertIs(compiled['_emit'].__code__,entry.base.original._emit.__code__)
        self.assertEqual(compiled['ROUND'],entry.ROUND);self.assertEqual(compiled['MODE'],entry.MODE)

    def test_every_master_math_delegate_and_native_phase_propagates_same_source_bridge(self):
        ns=entry.namespace(self.args)
        for phase in ('math','compile'):
            argv=ns['_argv'](self.args,phase,self.args.out/'math/math-result.json',10000)
            self.assertIn(entry.MODULE,argv)
            for name in entry.BRIDGE_FIELDS:
                flag='--'+name.replace('_','-');self.assertEqual(argv.count(flag),1)
                self.assertEqual(argv[argv.index(flag)+1],str(getattr(self.args,name)))
        args=copy.copy(self.args);args.model_info=self.f.out/'model/model_info.json';args.program=self.f.out/'session/program.ptdsl'
        args.math_result=self.args.out/'math/math-result.json'
        for phase in ('emit','trace','compile'):
            argv=ns['compile_namespace']()['phase_command'](args,phase,self.root/'work')
            self.assertIn(entry.MODULE,argv);self.assertNotIn('--report-result',argv)
            self.assertEqual(argv[argv.index('--boundary-snapshot')+1],str(self.args.boundary_snapshot))
            self.assertEqual(argv[argv.index('--math-result')+1],str(args.math_result))

    def proof(self,ns):
        source=ns['metadata_source'](self.args.source_case,self.audit)
        pairs=[('graph_reference','tensor_exact'),('graph_reference','mock_exact'),('tensor_program','mock_program')]
        return dict(round=entry.ROUND,model=entry.MODEL,mode=entry.MODE,phase='math',status='ok',math_passed=True,
            source=source,source_recheck=copy.deepcopy(source),probe_source=source['calibration'],
            compiler_inputs=ns['compiler_metadata'](self.args),attempted=dict.fromkeys(entry.base.COUNTERS,3),
            completed=dict.fromkeys(entry.base.COUNTERS,3),strict_checks_attempted=9,strict_checks_completed=9,
            calibration_probes=3,frozen_math=[dict(sample=i,sample_index=i,checks=[dict(reference=a,observed=b,
                passed=True,criterion=entry.base.original.CRITERION,max_absolute_error=0.,max_tolerance_ratio=0.) for a,b in pairs]) for i in range(3)],
            **dict.fromkeys(entry.base.ZERO,0))

    def test_original_three_by_three_math_source_counter_and_criterion_gate_still_rejects(self):
        ns=entry.namespace(self.args);path=self.root/'math-proof.json';good=self.proof(ns)
        for change in ('valid','counter','source','criterion','coverage'):
            record=copy.deepcopy(good)
            if change=='counter':record['strict_checks_completed']=8
            elif change=='source':record['source']['fork_provenance']['original_model_info']='/foreign'
            elif change=='criterion':record['frozen_math'][0]['checks'][0]['criterion']='loose'
            elif change=='coverage':record['completed']['tensor_exact']=2
            fixture_module.put(path,record)
            if change=='valid':self.assertEqual(ns['compile_source'](self.args.source_case,self.audit,path)['mathematical_groups_passed'],3)
            else:
                with self.subTest(change=change),self.assertRaises(ValueError):ns['compile_source'](self.args.source_case,self.audit,path)

    def test_original_template_path_and_profile_are_preserved(self):
        ns=entry.namespace(self.args);original=self.args.metadata_template;self.args.metadata_template=self.root/'foreign-template.json'
        with self.assertRaisesRegex(ValueError,'template'):ns['compiler_metadata'](self.args)
        self.args.metadata_template=original;self.args.profile=self.root/'foreign-profile.json'
        with self.assertRaisesRegex(ValueError,'profile'):ns['compiler_metadata'](self.args)

    def test_fake_full_compile_protocol_retains_two_phases_registry_and_completion(self):
        ns=entry.namespace(self.args);calls=[]
        def child(argv,**kwargs):
            calls.append(argv);self.assertEqual(kwargs['mem_gib'],24);self.assertEqual(kwargs['fsize_gib'],16)
            self.assertLessEqual(kwargs['timeout'],12000)
            proof=self.args.out/'math/math-result.json'
            if '--worker-phase' in argv:fixture_module.put(proof,self.proof(ns))
            else:
                source=ns['compile_source'](self.args.source_case,self.audit,proof)
                fixture_module.put(self.args.compile_out/'result.json',dict(status='ok',round=entry.ROUND,mode=entry.MODE,
                    model=entry.MODEL,source=source,completed=dict(emit=1,trace=1,compile=1),compiled_bootstraps=7,runtime_plan='/synthetic-only'))
                fixture_module.put(self.args.compile_out/'completion.json',dict(status='ok',registry_persisted=True,required_artifacts_persisted=True))
            return fixture_module.clean()
        ns['run_process']=child
        with mock.patch.object(entry.serial,'foreign_heavy_processes',return_value=[]), \
             mock.patch.object(entry.base.original,'_available_ram_bytes',return_value=50*entry.base.GIB), \
             mock.patch.object(entry.base.shutil,'disk_usage',return_value=SimpleNamespace(free=50*entry.base.GIB)), \
             mock.patch('pt_agent.registry.git_rev',return_value='synthetic'):
            result=ns['execute'](self.args)
        self.assertEqual(result['status'],'ok',result.get('error'));self.assertEqual(len(calls),2)
        self.assertTrue(result['compile_accepted']);self.assertEqual(result['completed'],dict(math=1,compile=1))
        self.assertTrue((self.args.out/'original-compile-first-source.py').is_file())
        self.assertTrue((self.args.out/'original-budget-confirm-source.py').is_file())
        self.assertEqual(entry.base.read(self.args.out/'completion.json')['status'],'ok')
        rows=[json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        own=[row for row in rows if row['run_id']==self.args.out.name];self.assertEqual(len(own),1)
        self.assertEqual(set(own[0]),entry.serial.REGISTRY_FIELDS)
        self.assertFalse((self.f.out/'session/REPORT_ACCESS.json').exists())

    def test_main_admission_error_saves_zero_processes(self):
        args=copy.copy(self.args);args.boundary_snapshot=self.root/'foreign'
        argv=self.cli(args)
        with mock.patch('pt_agent.registry.git_rev',return_value='synthetic'),mock.patch('builtins.print'):
            self.assertEqual(entry.main(argv),1)
        result=entry.base.read(args.out/'result.json');self.assertEqual(result['attempted'],dict(math=0,compile=0))
        self.assertFalse(result['compile_accepted']);self.assertEqual(result['processes'],[])

    def cli(self,args):
        argv=[]
        for name in ('source_case','checkpoint_audit','out','compile_out','intake_python','metadata_template','profile','frontend',*entry.BRIDGE_FIELDS):
            argv+=['--'+name.replace('_','-'),str(getattr(args,name))]
        return argv

    def test_normal_rejected_fork_observation_never_starts_math_or_compiler(self):
        # Same accepted lineage/claim, with the original C2 returning no selection.
        case=self.f.out;state=entry.base.read(case/'session/SESSION.json');state['confirmation']=dict(selected=None,evals=[],confirm_rule='c2',applicable=True)
        fixture_module.put(case/'session/SESSION.json',state)
        fixture_module.put(case/'session/AGENT.json',dict(status='confirm_failed',confirm=state['confirmation']))
        for name in ('FROZEN.json','program.ptdsl'):(case/'session'/name).unlink()
        result=entry.base.read(case/'result.json');outcome=dict(outcome='quality_rejected',frozen=False)
        result.update(status='quality_rejected',outcome='quality_rejected',selection_outcome=outcome,frozen=False,frozen_accepted=False)
        result['worker']['frozen']=False;result['metrics'].update(status='quality_rejected',outcome='quality_rejected',frozen=False,frozen_accepted=False)
        fixture_module.put(case/'phases/001-confirm/worker_result.json',result['worker']);fixture_module.put(case/'result.json',result)
        completion=entry.base.read(case/'completion.json');completion['status']='quality_rejected';fixture_module.put(case/'completion.json',completion)
        registry=self.root/'registry.jsonl';row=json.loads(registry.read_text());row['metrics']=result['metrics'];registry.write_text(json.dumps(row)+'\n')
        with mock.patch.object(entry.base,'run_process',side_effect=AssertionError('no actual child')), \
             mock.patch('pt_agent.registry.git_rev',return_value='synthetic'),mock.patch('builtins.print'):
            self.assertEqual(entry.main(self.cli(self.args)),0)
        observed=entry.base.read(self.args.out/'result.json');self.assertEqual(observed['status'],'quality_rejected')
        self.assertEqual(observed['attempted'],dict(math=0,compile=0));self.assertFalse(self.args.compile_out.exists())


if __name__=='__main__':unittest.main()
