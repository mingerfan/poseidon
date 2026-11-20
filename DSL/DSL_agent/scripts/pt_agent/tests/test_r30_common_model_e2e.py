"""Small synthetic orchestration checks, without model/numeric/API execution."""
from argparse import Namespace
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import r28_common_agent, r30_common_model_e2e as entry
from pt_agent.experiments.completion_metrics import persist_case_completion


class FreshE2EChecks(unittest.TestCase):
    def test_explicit_r28_sources_never_resolve_a_historical_intake(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);info=root/'model_info.json';cal=root/'calibration.json'
            info.write_text(json.dumps(dict(source=dict(factory=entry.FACTORY_PREFIX+'resnet20_silu'))));cal.write_text('{}')
            with patch.object(paths,'RESULTS_DIR',root/'absent-old-results'):
                source=r28_common_agent.sources('resnet20_silu',model_info=info,calibration=cal)
            self.assertEqual(source,dict(model_info=info,calibration=cal))
            with self.assertRaises(ValueError):r28_common_agent.sources('vgg16_silu_avgpool',model_info=info,calibration=cal)
            with self.assertRaises(ValueError):r28_common_agent.sources('resnet20_silu',model_info=info)

    def test_phase_commands_connect_only_current_model_and_calibration(self):
        args=Namespace(out=Path('/fresh/run'),model='resnet20_silu',inputs=Path('/original/records'),
            selections=Path('/original/selections'),agent_seconds=7200,trace_seconds=1800,compile_seconds=7200,
            ledger=Path('/budget/calls'),allowance=Path('/budget/allowance'),profile=Path('/profile'),
            frontend=Path('/frontend'),metadata_template=Path('/native-metadata'),calls=6,tuner_seconds=600,
            memory_gib=6,case_gib=8)
        for phase in entry.PHASES:
            argv,seconds=entry.phase_command(args,phase,args.out/'work'/phase,remaining_seconds=28000)
            self.assertEqual(argv[argv.index('--model-info')+1],'/fresh/run/work/intake/model_info/model_info.json')
            self.assertEqual(argv[argv.index('--calibration')+1],'/fresh/run/work/calibration/calibration.json')
            self.assertNotIn('r13-common-',str(argv));self.assertNotIn('r14-common-',str(argv))
            self.assertNotIn('--enable-debug-printer',argv)
            if phase=='compile':self.assertEqual(seconds,7320)

    def _fixture(self,root,*,reject=False,stop_after='compile'):
        out=root/'new-e2e';original=root/'original';original.mkdir();inputs=original/'records';inputs.mkdir()
        (inputs/'dataset.json').write_text('{}')
        files={key:original/key for key in ('source','checkpoint','inference_preprocessing_source')}
        for path in files.values():path.write_text('original metadata fixture')
        options=dict(out=out,model='resnet20_silu',inputs=inputs,selections=original/'selections',
            frontend=original/'frontend',profile=original/'profile',metadata_template=original/'metadata',
            ledger=root/'llm_calls.jsonl',allowance=root/'allowance.json',seconds=28800,agent_seconds=7200,
            trace_seconds=1800,compile_seconds=7200,memory_gib=6,case_gib=8,calls=6,tuner_seconds=600,
            stop_after=stop_after)
        for key in ('selections','frontend','profile','metadata_template'):options[key].write_text('{}')
        args=Namespace(**options);order=[]
        def process(argv,**kwargs):
            phase=argv[argv.index('--worker-phase')+1];work=Path(argv[argv.index('--work-dir')+1]);order.append(phase)
            body=dict(status='ok',phase=phase)
            if phase=='intake':
                model=work/'model_info';model.mkdir()
                (model/'model_info.json').write_text(json.dumps(dict(source=dict(factory=entry.FACTORY_PREFIX+args.model),
                    threshold=dict(acc_drop_pp=.5))))
            if phase=='calibration':(work/'calibration.json').write_text('{}')
            if phase=='agent':
                program='def candidate(x):\n    return x\n';(work/'program.ptdsl').write_text(program)
                (work/'SESSION.json').write_text(json.dumps(dict(phase='done',failed=None,calls=1)))
                api=work/'api';api.mkdir();(api/'call.request.json').write_text('{}');(api/'call.response.json').write_text('{}')
                row=dict(iteration=1,origin='llm',passed=True,diverged=False,n_diverged=0,total=1000,
                         acc_drop_pp=0.0,max_u_calibrated=.5,max_u_analytic=0.0)
                selected=dict(iteration=1,origin='llm',source=program)
                check=dict(passed=True,finite=True,dtype_ok=True,shape_ok=True,
                           criterion='1e-12+1e-12*abs(reference)',max_error_ratio=0.)
                worker=dict(status='quality_rejected'if reject else'ok',frozen=not reject,model=args.model,
                    source_model_info=str(out/'work/intake/model_info/model_info.json'),
                    source_calibration=str(out/'work/calibration/calibration.json'),quality_acceptance=dict(passed=not reject),
                    actual_api_calls=1,report_samples=0,compiler_runs=0,real_ckks_runs=0,
                    selected_program=str(work/'program.ptdsl'),agent=dict(best_program=program,llm_calls=1,
                    confirm=dict(selected=selected,evals=[row])),strict_math_probes=[dict(sample_id='test:'+str(9000+i),
                    exact=check,approximate=check)for i in range(3)])
                (work/'worker-result.json').write_text(json.dumps(worker))
                args.ledger.write_text(json.dumps(dict(ts='fixture',purpose='agent-iter-1',status=200,error=None))+'\n')
            if phase=='compile':body.update(compiled_bootstraps=2,predicted_latency_us=123)
            (work/'phase_result.json').write_text(json.dumps(body))
            return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[])
        return args,files,order,process

    def test_actual_e2e_boundary_encloses_all_fresh_stages_and_registry(self):
        from pt_agent.experiments import r13_common_models
        with TemporaryDirectory()as directory:
            root=Path(directory);args,files,order,process=self._fixture(root)
            endpoint=[]
            def completion(out,**kwargs):
                def clock():
                    self.assertEqual(order,list(entry.PHASES))
                    self.assertTrue((out/'result.json').is_file())
                    rows=[json.loads(line)for line in (root/'registry.jsonl').read_text().splitlines()]
                    self.assertEqual(rows[-1]['mode'],entry.MODE)
                    self.assertEqual(rows[-1]['run_id'],out.name)
                    self.assertEqual(set(rows[-1]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
                    endpoint.append(True)
                    return kwargs['started_ns']+1000000000
                return persist_case_completion(out,clock=clock,**kwargs)
            with ExitStack()as stack:
                stack.enter_context(patch.object(paths,'RESULTS_DIR',root))
                stack.enter_context(patch.dict(r13_common_models.FACTORIES,{entry.FACTORY_PREFIX+args.model:files}))
                stack.enter_context(patch('pt_agent.registry.git_rev',return_value='fixture'))
                stack.enter_context(patch.object(entry,'run_process',side_effect=process))
                stack.enter_context(patch.object(entry,'persist_case_completion',side_effect=completion))
                result=entry.execute(args)
            self.assertEqual(result['status'],'ok');self.assertEqual(endpoint,[True])
            self.assertEqual(result['compiled_bootstraps'],2);self.assertEqual(result['predicted_latency_us'],123)
            self.assertFalse(result['historical_elapsed_reused']);self.assertFalse(result['original_intake_reused'])
            self.assertEqual(result['rule_initialization_runs'],0);self.assertEqual(result['variant'],entry.VARIANT)
            self.assertIsNone(result['CKKS_latency']);self.assertIsNone(result['accuracy_acceptance'])
            self.assertEqual(json.loads((args.out/'completion.json').read_text())['complete_case_elapsed_seconds'],1.)

    def test_quality_reject_stops_before_placement_emit_or_compile(self):
        from pt_agent.experiments import r13_common_models
        with TemporaryDirectory()as directory:
            root=Path(directory);args,files,order,process=self._fixture(root,reject=True)
            with patch.object(paths,'RESULTS_DIR',root),patch.dict(r13_common_models.FACTORIES,{entry.FACTORY_PREFIX+args.model:files}), \
                 patch('pt_agent.registry.git_rev',return_value='fixture'),patch.object(entry,'run_process',side_effect=process):
                result=entry.execute(args)
            self.assertEqual(result['status'],'quality_rejected')
            self.assertEqual(order,['intake','calibration','agent'])
            self.assertNotIn('emit',result['attempted']);self.assertIsNone(result['compiled_bootstraps'])
            self.assertIsNone(result['quality_audit'])


if __name__=='__main__':unittest.main()
