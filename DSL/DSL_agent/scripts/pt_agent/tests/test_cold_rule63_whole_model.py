"""New cold Rule63 contract and synthetic supervision tests; no model/compiler."""
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import cold_rule63_whole_model as d


def observation(**changes):
    row=dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[],
             started_ns=100,ended_ns=100000100,peak_rss_kib_by_pid={})
    row.update(changes);return row


class ColdRuleContract(unittest.TestCase):
    def args(self,root,**changes):
        values=dict(model='resnet20_silu',placement='depth',out=root/'case',paper_runs=root/'runs.jsonl',
            inputs=root/'inputs',selections=root/'selections.json',frontend=root/'frontend.py',
            profile=root/'profile.json',metadata_template=root/'template.json',seconds=12000,
            case_gib=64,diagnostic_elide_final_ir=True)
        values.update(changes);return SimpleNamespace(**values)

    def fake_execute(self,root,*,native=None,boot=230,watcher=None,fail_phase=None):
        from pt_agent import paths
        from pt_agent import registry
        from pt_agent.experiments import dacapo_compile_continuation as resource
        args=self.args(root);calls=[]
        def process(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];calls.append(phase)
            row=dict(status='ok',model=args.model,phase=phase)
            if phase==fail_phase:row.update(status='error',error='Synthetic generation exception')
            if phase=='compile':
                row.update(native_process=native or observation(),compiled_bootstraps=boot,
                    predicted_latency_us=None,execution_steps=1234,execution_ops=dict(boot=boot,relinearize=2),
                    prediction=dict(table_known_sum_us=23,missing=[dict(op='relinearize')]),
                    runtime_plan='synthetic-small-runtime-plan-metadata')
                if not d.clean(row['native_process']):row.update(status='error',error='Synthetic compiler failure')
            work=args.out/'work'/phase
            (work/'phase_result.json').write_text(json.dumps(row))
            (work/'resources.jsonl').write_text(json.dumps(dict(processes=[dict(vmrss_kib=1024)],
                outer_process=dict(vmrss_kib=1024)))+'\n')
            self.assertEqual(kwargs['mem_gib'],24);self.assertEqual(kwargs['fsize_gib'],16)
            return observation()
        with patch.object(paths,'RESULTS_DIR',root),patch.object(registry,'git_rev',return_value='synthetic'),\
                patch.object(d,'admission',return_value=dict(factory='synthetic-source')),\
                patch.object(d,'snapshot'),patch.object(d,'run_process',side_effect=process),\
                patch.object(resource,'available_ram',return_value=20*d.GIB),\
                patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=100*d.GIB)):
            if watcher is None:result=d.execute(args)
            else:
                with patch.object(d.threading,'Thread',return_value=watcher):result=d.execute(args)
        return args,result,calls

    def test_fresh_commands_pass_every_current_path_and_finite_native_limits(self):
        with tempfile.TemporaryDirectory()as tmp:
            args=self.args(Path(tmp))
            for phase in d.PHASES:
                argv,wall=d.phase_command(args,phase,12000)
                inner=argv[argv.index('--')+1:]
                self.assertEqual(inner[:3],['-B','-m','pt_agent.experiments.cold_rule63_whole_model'])
                self.assertEqual(inner[inner.index('--worker-phase')+1],phase)
                self.assertIn(str(args.out/'work/intake/model_info/model_info.json'),inner)
                self.assertIn(str(args.out/'work/generation/program.ptdsl'),inner)
                self.assertIn(str(args.out/'work/emit'),inner)
                self.assertEqual(int(inner[inner.index('--native-seconds')+1]),7200 if phase=='compile'else 1800)
                self.assertEqual(wall,7320 if phase=='compile'else 1920)
                self.assertNotIn('--ledger',inner)

    def test_admission_rejects_same_ring_with_changed_boot_or_scale(self):
        from pt_agent import paths,toolchain
        from pt_agent.experiments import r13_common_models as originals,r29_local_compile_matrix as native
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=self.args(root);args.inputs.mkdir()
            required=[args.inputs/'dataset.json',args.selections,args.frontend,args.metadata_template,
                      root/'source.py',root/'checkpoint',root/'preprocess.py',root/'compiler']
            for p in required:p.write_text('synthetic metadata prerequisite')
            args.metadata_template.write_text('[]')
            config=dict(bootstrapLevelLowerBound=1,bootstrapLevelUpperBound=5,levelLowerBound=1,
                levelUpperBound=8,polynomialDegree=65536,rescalingFactor=43,runtime='HEAAN-HEVM')
            original=dict(source=root/'source.py',checkpoint=root/'checkpoint',inference_preprocessing_source=root/'preprocess.py')
            with patch.object(paths,'RESULTS_DIR',root),patch.dict(originals.FACTORIES,
                    {'pt_agent.experiments.r13_common_models:'+args.model:original}),\
                    patch.object(toolchain,'host_paths',return_value=dict(hecate_opt=root/'compiler')),\
                    patch.object(native,'compiler_argv',return_value=['synthetic']),\
                    patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=100*d.GIB)):
                args.profile.write_text(json.dumps(config));self.assertEqual(d.admission(args)['checkpoint_values_read'],0)
                for field,value in [('bootstrapLevelUpperBound',16),('rescalingFactor',40)]:
                    args.profile.write_text(json.dumps(dict(config,**{field:value})))
                    with self.subTest(field=field),self.assertRaises(ValueError):d.admission(args)

    def test_fake_complete_case_persists_cold_clock_and_one_standard_row(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args,result,calls=self.fake_execute(root)
            self.assertEqual(calls,list(d.PHASES));self.assertEqual(result['status'],'ok')
            self.assertTrue(result['monitor_stopped']);self.assertTrue(result['expected_bootstraps_match'])
            rows=[json.loads(l)for l in(root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(len(rows[0]),8)
            completion=d.read(args.out/'completion.json');paper=d.read(args.out/'paper-record.json')
            self.assertTrue(completion['registry_persisted']);self.assertEqual(paper['status'],'ok')
            self.assertEqual(paper['boots_compiled'],230);self.assertEqual(paper['exec_ops'],1234)
            self.assertEqual(paper['compile_s'],.1);self.assertIsNone(paper['pred_latency_s'])
            self.assertEqual(paper['known_cost_partial_s'],.000023)
            self.assertIsNone(paper['stages_s']['placement']);self.assertTrue(paper['cold'])
            self.assertEqual(paper['generation_e2e_s'],completion['complete_case_elapsed_seconds'])
            self.assertAlmostEqual(sum(v for v in paper['stages_s'].values()if v is not None),paper['generation_e2e_s'])
            self.assertEqual(len((root/'runs.jsonl').read_text().splitlines()),1)

    def test_compiler_timeout_retains_native_clock_and_terminal_failure_record(self):
        with tempfile.TemporaryDirectory()as tmp:
            args,result,calls=self.fake_execute(Path(tmp),native=observation(exit_code=-15,timed_out=True))
            row=d.read(args.out/'paper-record.json')
            self.assertEqual(result['status'],'error');self.assertEqual(row['status'],'timeout')
            self.assertEqual(row['compile_s'],.1);self.assertEqual(calls,list(d.PHASES))
            self.assertEqual(d.read(args.out/'completion.json')['status'],'error')

    def test_depth_mismatch_preserves_actual_compile_and_fails_acceptance(self):
        with tempfile.TemporaryDirectory()as tmp:
            args,result,_=self.fake_execute(Path(tmp),boot=229)
            row=d.read(args.out/'paper-record.json')
            self.assertFalse(result['expected_bootstraps_match']);self.assertEqual(row['boots_compiled'],229)
            self.assertEqual(row['status'],'driver_error');self.assertEqual(row['acceptance_status'],'error')

    def test_generation_failure_has_no_compile_or_fake_unobserved_stage(self):
        with tempfile.TemporaryDirectory()as tmp:
            args,result,calls=self.fake_execute(Path(tmp),fail_phase='generation')
            row=d.read(args.out/'paper-record.json')
            self.assertEqual(calls,['intake','calibration','generation'])
            self.assertIsNone(row['compile_s']);self.assertIsNone(row['stages_s']['compile'])
            self.assertIsNone(row['boots_compiled']);self.assertEqual(row['api_calls'],0)
            self.assertEqual(result['completed'],dict(intake=1,calibration=1))

    def test_monitor_start_failure_still_closes_registry_completion_and_paper(self):
        class StartFailure:
            def start(self):raise RuntimeError('Synthetic monitor start failure')
            def is_alive(self):return False
        with tempfile.TemporaryDirectory()as tmp:
            args,result,calls=self.fake_execute(Path(tmp),watcher=StartFailure())
            self.assertEqual(calls,[]);self.assertEqual(result['status'],'error')
            self.assertTrue((args.out/'completion.json').is_file())
            self.assertEqual(d.read(args.out/'paper-record.json')['status'],'driver_error')

    def test_cleanup_masks_real_join_and_persistence_signals(self):
        class FakeWatcher:
            alive=True
            def start(self):pass
            def is_alive(self):return self.alive
            def join(self,timeout):
                signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGALRM)
                self.alive=False
        original=d.persist_case_completion
        def persist(*args,**kwargs):
            self.assertTrue(all(signal.getsignal(sig)==signal.SIG_IGN for sig in d.SIGNALS))
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)
            signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGINT)
            return original(*args,**kwargs)
        with tempfile.TemporaryDirectory()as tmp,patch.object(d,'persist_case_completion',side_effect=persist):
            args,result,_=self.fake_execute(Path(tmp),watcher=FakeWatcher())
            self.assertEqual(result['status'],'ok');self.assertTrue(result['monitor_stopped'])
            self.assertTrue((args.out/'completion.json').is_file())

    def test_incomplete_monitor_keeps_signals_masked_and_marks_error(self):
        class Stuck:
            def start(self):pass
            def is_alive(self):return True
            def join(self,timeout):signal.raise_signal(signal.SIGTERM)
        saved={s:signal.getsignal(s)for s in d.SIGNALS}
        try:
            with tempfile.TemporaryDirectory()as tmp:
                args,result,_=self.fake_execute(Path(tmp),watcher=Stuck())
                self.assertEqual(result['status'],'error');self.assertFalse(result['monitor_stopped'])
                self.assertTrue(all(signal.getsignal(s)==signal.SIG_IGN for s in d.SIGNALS))
                self.assertEqual(d.read(args.out/'completion.json')['status'],'error')
                signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGALRM)
        finally:
            for s,h in saved.items():signal.signal(s,h)

    def test_late_guard_between_last_phase_and_monitor_join_rejects_success(self):
        class LateGuard:
            alive=True
            def __init__(self,*,target,daemon):
                cells=dict(zip(target.__code__.co_freevars,[cell.cell_contents for cell in target.__closure__]))
                self.result=cells['result']
            def start(self):pass
            def is_alive(self):return self.alive
            def join(self,timeout):
                self.result['guard_error']='resource_limit: synthetic last-phase guard'
                self.alive=False
        with tempfile.TemporaryDirectory()as tmp,patch.object(d.threading,'Thread',LateGuard):
            args,result,_=self.fake_execute(Path(tmp))
            self.assertEqual(result['status'],'error')
            self.assertEqual(d.read(args.out/'paper-record.json')['status'],'resource_limit')
            self.assertEqual(d.read(args.out/'completion.json')['status'],'error')


if __name__=='__main__':unittest.main(verbosity=2)
