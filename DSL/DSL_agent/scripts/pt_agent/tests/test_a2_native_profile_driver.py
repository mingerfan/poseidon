"""New A2 header/command/native metadata and fake outer execution tests only."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a2_native_profile_driver as d
from pt_agent.tests.test_a2_native_profile_planner import config


TEXT='''import fhe
@fhe.program(slots=8,l_max=5,boot_out_level=5,refresh="agent",placement="program-plan")
def tiny(x):
    y=x*x
    return y+x
'''


def observation():
    return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[],signals_sent=[],started_ns=100,ended_ns=300,peak_rss_kib_by_pid={})


class NativeDriverCases(unittest.TestCase):
    def test_header_rewrite_changes_only_two_system_literals(self):
        original=ast.parse(TEXT);changed=ast.parse(d.reprofile_source(TEXT))
        header=changed.body[1].decorator_list[0]
        values={kw.arg:kw for kw in header.keywords}
        self.assertEqual(ast.literal_eval(values['l_max'].value),29)
        self.assertEqual(ast.literal_eval(values['boot_out_level'].value),16)
        values['l_max'].value=ast.Constant(5);values['boot_out_level'].value=ast.Constant(5)
        self.assertEqual(ast.dump(original),ast.dump(changed))

    def test_pars_command_keeps_original_profile_identifiers_and_waterline(self):
        template=['/existing/compiler','/old.mlir','--dacapo','--boot-placement=optimized','--waterline=40','--ckks-config=/old.json','--runtime-plan-target-id=native','--runtime-plan-operator-spec-sha256=existing-identifier','--runtime-plan-context-id=n17-r51','--runtime-plan-boot-profile=l3-to16','-o','/old-output.mlir']
        changed=d.compiler_argv(template,Path('/new.mlir'),Path('/native.json'),Path('/new-output.mlir'))
        self.assertEqual(changed[:3],['/existing/compiler','/new.mlir','--pars'])
        self.assertNotIn('--dacapo',changed);self.assertNotIn('--boot-placement=optimized',changed)
        self.assertIn('--waterline=40',changed);self.assertIn('--ckks-config=/native.json',changed)
        self.assertEqual([v for v in changed if v.startswith('--runtime-plan-')],[v for v in template if v.startswith('--runtime-plan-')])
        self.assertEqual(changed[changed.index('-o')+1],'/new-output.mlir')
        with self.assertRaises(ValueError):d.compiler_argv(template+['--waterline=40'],'a','b','c')

    def test_actual_level_metadata_and_missing_relinearize_keep_full_prediction_null(self):
        plan=dict(values=[dict(id='x',level=16),dict(id='lo',level=3),dict(id='booted',level=16),dict(id='out',level=15)],external_inputs=['x'],execution=[dict(id='b',op='boot',kind='compute',inputs=['lo'],output='booted'),dict(id='r',op='relinearize',kind='compute',inputs=['booted'],output='out')])
        found,audit=d.observe_runtime(plan,config(),1)
        self.assertEqual(audit,dict(input_levels=[16],max_level=16,profile_upper_bound=29))
        self.assertIsNone(found['predicted_latency_us'])
        self.assertGreater(found['prediction']['table_known_sum_us'],0)
        self.assertEqual(found['prediction']['missing'],[dict(index=1,op='relinearize',reason='operation has no profile entry')])
        for idx,level in((0,29),(1,4),(2,15),(3,30)):
            bad=copy.deepcopy(plan);bad['values'][idx]['level']=level
            with self.subTest(idx=idx),self.assertRaises(ValueError):d.observe_runtime(bad,config(),1)
        with self.assertRaises(ValueError):d.observe_runtime(plan,config(),2)

    def _fake_execute(self,root,compile_exit=0,available_ram=None):
        args=SimpleNamespace(run=True,out=root/'case',model='resnet20_silu',source_case=root/'source',native_case=root/'native',frontend=root/'frontend',planning_seconds=10,state_limit=16,case_gib=4)
        source=root/'original.ptdsl';source.write_text(TEXT)
        admission=dict(program=str(source),model_info='saved-info',profile='saved-profile',native_plan='saved-plan')
        calls=[]
        def fake_process(argv,**kwargs):
            phase=argv[argv.index('--worker')+1];calls.append(phase);out=args.out/'phases'/phase
            row=dict(phase=phase,status='ok')
            if phase=='trace':(args.out/'emit/candidate_trace.mlir').write_text('fake retained path marker')
            if phase=='compile':
                row['native_process']=dict(observation(),exit_code=compile_exit)
                if compile_exit:row.update(status='error',error='native compiler failed')
            if phase=='observe':row.update(compiled=dict(compiled_bootstraps=2,predicted_latency_us=None),native_reference=dict(predicted_latency_us=None))
            (out/'result.json').write_text(json.dumps(row));return observation()
        from pt_agent import paths
        from pt_agent.experiments import r27_orbit_lazy_backend as resource
        with patch.object(paths,'RESULTS_DIR',root),patch.object(d,'admission',return_value=admission),patch.object(d,'run_process',side_effect=fake_process),patch.object(resource,'_available_ram_bytes',return_value=20*d.GIB if available_ram is None else available_ram),patch.object(d.shutil,'disk_usage',return_value=SimpleNamespace(free=30*d.GIB)):
            result=d.execute(args)
        return args,result,calls

    def test_fake_five_phase_execution_persists_exact_registry_and_completion(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args,result,calls=self._fake_execute(root)
            self.assertEqual(calls,list(d.PHASES));self.assertEqual(result['status'],'ok')
            self.assertTrue(result['monitor_stopped']);self.assertEqual(result['compiled_bootstraps'],2)
            self.assertIsNone(result['generation_e2e_s']);self.assertIsNone(result['cold_end_to_end_s']);self.assertIsNone(result['pred_latency_s'])
            self.assertEqual(result['compile_s'],2e-7)
            saved=json.loads((args.out/'completion.json').read_text())
            self.assertTrue(saved['registry_persisted']);self.assertEqual(saved['status'],'ok')
            registry=[json.loads(line)for line in(root/'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(registry),1);self.assertEqual(len(registry[0]),8)

    def test_direct_resource_guard_has_resource_class_without_starting_worker(self):
        with tempfile.TemporaryDirectory()as tmp:
            args,result,calls=self._fake_execute(Path(tmp),available_ram=0)
            self.assertEqual(result['failure_class'],'resource_limit')
            self.assertEqual(calls,[])

    def test_fake_native_compiler_failure_keeps_inner_error_and_skips_observe(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args,result,calls=self._fake_execute(root,compile_exit=1)
            self.assertEqual(calls,list(d.PHASES[:-1]));self.assertEqual(result['status'],'error')
            self.assertEqual(result['failure_class'],'compiler_crash')
            self.assertEqual(result['phases'][-1]['worker']['native_process']['exit_code'],1)
            self.assertIsNone(result['compiled_bootstraps']);self.assertEqual(json.loads((args.out/'completion.json').read_text())['status'],'error')


if __name__=='__main__':unittest.main(verbosity=2)
