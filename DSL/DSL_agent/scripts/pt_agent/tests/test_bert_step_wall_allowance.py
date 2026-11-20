"""Finite step-wall plumbing only; every process/transport is blocked or fake."""
from argparse import Namespace
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from types import FunctionType
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_sst2 as original
from pt_agent.experiments import bert_sst2_recorded as recorded
from pt_agent.experiments import bert_serial_continuation as serial
from pt_agent.experiments.r10_paper_metrics import StageRecorder
from pt_agent.tests.test_bert_serial_continuation import fixture,checkpoint


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def clean():
    return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])


class BertStepWallChecks(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        # Includes Provider subprocess transports: no accidental real child can
        # escape these metadata tests. Scoped fake launch mocks override this.
        blocked=patch.object(subprocess,'Popen',side_effect=AssertionError('Real child/Provider transport forbidden'))
        blocked.start();self.addCleanup(blocked.stop)

    def args(self):
        return Namespace(out=self.root/'source',intake_python=Path('/original/python'),
            model=Path('/original/model'),inputs=Path('/original/inputs'),ledger=self.root/'ledger.jsonl',
            allowance=self.root/'allowance.json',data_python=Path('/original/data-python'),mem_gib=4)

    def test_validator_accepts_finite_integer_range_and_rejects_invalid_values(self):
        for value in (7200,10000,21600):self.assertEqual(recorded.validate_step_seconds(value),value)
        for value in (True,False,float('nan'),float('inf'),-float('inf'),7200.,7199,21601,0,None,'21600'):
            with self.subTest(value=value),self.assertRaises(ValueError):recorded.validate_step_seconds(value)

    def test_recorded_cli_default_and_explicit_are_stripped_before_original_entry(self):
        for suffix,want in (([],7200),(['--step-seconds','21600'],21600)):
            with patch.object(recorded,'outer_delegate',return_value=0)as outer:
                self.assertEqual(recorded.main(['--phase','step']+suffix),0)
            self.assertEqual(outer.call_args.args[0],['--phase','step'])
            self.assertEqual(outer.call_args.args[2],want)

    def test_invalid_recorded_cli_precedes_import_worker_or_output(self):
        with patch.object(recorded,'outer_delegate')as outer,patch.object(recorded,'worker_delegate')as worker,redirect_stderr(StringIO()):
            for value in ('nan','inf','true','7199','21601'):
                with self.subTest(value=value),self.assertRaises(SystemExit):
                    recorded.main(['--step-seconds',value,'--phase','step'])
            outer.assert_not_called();worker.assert_not_called()
        with patch.object(recorded.importlib,'import_module')as importing:
            for value in (True,float('nan'),21601):
                with self.assertRaises(ValueError):recorded.outer_delegate([],self.root/'unused',value)
                with self.assertRaises(ValueError):recorded.worker_delegate([],self.root/'unused',value)
            importing.assert_not_called()

    def test_private_namespace_uses_original_commands_with_only_step_clock_changed(self):
        args=self.args();before=dict(original.PHASE_SECONDS);command=original.command
        namespace=recorded.private_outer_namespace(original,21600)
        self.assertIs(namespace['command'].__globals__,namespace)
        self.assertEqual(original.PHASE_SECONDS,before);self.assertIs(original.command,command)
        for phase in before:
            actual=namespace['command'](args,phase,self.root/'work')
            old=original.command(args,phase,self.root/'work')
            if phase=='step':
                expected=list(old);expected[expected.index('--seconds')+1]='21660'
                self.assertEqual(actual,expected)
            else:self.assertEqual(actual,old)

    def test_explicit_outer_delegate_keeps_shared_globals_and_real_command_contract(self):
        args=self.args();before=dict(original.PHASE_SECONDS);old_command=original.command
        def main(argv):return command(TEST_ARGS,'step',TEST_WORK)
        private_main=FunctionType(main.__code__,vars(original),'main')
        with patch.object(original,'TEST_ARGS',args,create=True),patch.object(original,'TEST_WORK',self.root/'work',create=True),\
             patch.object(original,'main',private_main),patch.object(recorded.importlib,'import_module',return_value=original):
            argv=recorded.outer_delegate(['--phase','step'],self.root/'recipe.json',21600)
        self.assertEqual(argv[argv.index('--seconds')+1],'21660')
        self.assertEqual(argv[argv.index('--step-seconds')+1],'21600')
        self.assertNotIn(recorded.WORKER,argv);self.assertIn(recorded.__spec__.name,argv)
        self.assertIs(original.command,old_command);self.assertEqual(original.PHASE_SECONDS,before)

    def test_default_delegate_keeps_original7260_command_and_restores_original_function(self):
        args=self.args();old_command=original.command
        expected=old_command(args,'step',self.root/'work')
        expected[expected.index(recorded.WORKER)]=recorded.__spec__.name
        expected+=['--delegate-worker','--metadata-template',str(self.root/'recipe.json')]
        def main(argv):return command(TEST_ARGS,'step',TEST_WORK)
        private_main=FunctionType(main.__code__,vars(original),'main')
        with patch.object(original,'TEST_ARGS',args,create=True),patch.object(original,'TEST_WORK',self.root/'work',create=True),\
             patch.object(original,'main',private_main),patch.object(recorded.importlib,'import_module',return_value=original):
            argv=recorded.outer_delegate(['--phase','step'],self.root/'recipe.json')
        self.assertEqual(argv,expected);self.assertNotIn('--step-seconds',argv)
        self.assertEqual(argv[argv.index('--seconds')+1],'7260');self.assertIs(original.command,old_command)

    def test_worker_wrapper_records21600_and_preserves_original_worker_argv(self):
        from pt_agent import toolchain
        template=['{hecate_opt}','{input}','--pars','--waterline=43',
            '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64,
            '--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
            '--ckks-config={package}/config/compiler-config.json','-o','{output_dir}/lowered.mlir']
        recipe=self.root/'recipe.json';save(recipe,template)
        source=self.root/'original-worker.py';source.write_text('# synthetic trusted worker only\n')
        argv=['--work-dir',str(self.root/'worker'),'--phase','step'];received=[]
        def worker(args):
            received.append(args)
            self.assertEqual(toolchain.compile_argv('pars','qwen48',Path('/metadata-only'))[:2],
                             ['/hecate-opt','/out/candidate_trace.mlir'])
            return 0
        with patch.object(recorded.importlib,'import_module',return_value=Namespace(__file__=str(source),main=worker)),\
             patch.object(toolchain,'compile_argv',side_effect=AssertionError('Legacy identifier builder forbidden'))as legacy:
            self.assertEqual(recorded.main(['--delegate-worker','--metadata-template',str(recipe),
                '--step-seconds','21600',*argv]),0)
            legacy.assert_not_called()
        self.assertEqual(received,[argv])
        note=json.loads((self.root/'worker/recorded-command-delegation.json').read_text())
        self.assertEqual(note['configured_step_seconds'],21600)
        self.assertTrue(note['early_context_installed_before_worker_import'])
        self.assertEqual(note['new_digest_computations'],0)

    def test_real_original_execute_phase_receives21720_seconds_and_preserves_AS4(self):
        from pt_agent import autonomous_budget,registry
        args=self.args();args.phase='step';captured={}
        @contextmanager
        def guard(out):yield None
        def launch(argv,**kwargs):
            captured.update(argv=argv,options=kwargs)
            work=Path(argv[argv.index('--work-dir')+1])
            save(work/'worker_result.json',dict(status='ok',phase='step',next_step='tune'))
            return clean()
        def terminal(out,work,result):result['case_artifact_bytes']=0
        def main(argv):return execute_phase(TEST_ARGS,'step')
        private_main=FunctionType(main.__code__,vars(original),'main')
        with patch.object(original,'TEST_ARGS',args,create=True),patch.object(original,'main',private_main),\
             patch.object(original,'source_preflight',return_value={}),patch.object(original,'check_budget'),\
             patch.object(original,'budget_guard',guard),patch.object(original,'child_environment',return_value={}),\
             patch.object(original,'run_process',side_effect=launch),patch.object(original,'check_worker'),\
             patch.object(original,'terminal_result',side_effect=terminal),patch.object(original,'append_registry'),\
             patch.object(autonomous_budget,'call_records',return_value=[]),patch.object(registry,'git_rev',return_value='fixture'),\
             patch.object(recorded.importlib,'import_module',return_value=original):
            result=recorded.outer_delegate(['--phase','step'],self.root/'recipe.json',21600)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(captured['options']['timeout'],21720)
        self.assertEqual(captured['options']['mem_gib'],4)
        self.assertEqual(captured['argv'][captured['argv'].index('--seconds')+1],'21660')
        self.assertEqual(original.PHASE_SECONDS['step'],7200)

    def test_serial_default_and_explicit_layer_bounds_leave_other_phases_unchanged(self):
        legacy=Namespace();selected=Namespace(step_seconds=21600)
        self.assertEqual(serial.phase_limits(legacy,'step'),(7440,4))
        self.assertEqual(serial.phase_limits(selected,'step'),(21840,4))
        for phase in ('checkpoint','report','compile'):
            self.assertEqual(serial.phase_limits(selected,phase),serial.PHASE_LIMITS[phase])
        self.assertEqual(serial.STATE_POLICY['tuner_budget_s'],2400)
        self.assertEqual(serial.STATE_POLICY['max_iters'],10)

    def test_serial_command_changes_only_step_option_and_retains_original_source(self):
        args=fixture(self.root);source=dict(model='/model',inputs='/inputs',source_case=str(args.source_case))
        default=serial.command(args,source,'step');args.step_seconds=21600
        expanded=serial.command(args,source,'step')
        self.assertEqual(expanded,default+['--step-seconds','21600'])
        self.assertEqual(expanded[expanded.index('--out')+1],str(args.source_case))
        self.assertEqual(expanded[expanded.index('--mem-gib')+1],'4')
        report=serial.command(args,source,'report');self.assertNotIn('--step-seconds',report)

    def test_serial_launch_21840_with_retained_tune3_history_and_zero_replayed_calls(self):
        args=fixture(self.root);args.step_seconds=21600
        state_path=args.source_case/'session/SESSION.json';state=json.loads(state_path.read_text())
        state.update(phase='tune',iteration=3,calls=3,pending={'source':'saved synthetic candidate'},
            messages=[{'role':'user','content':'synthetic only'}],history=[{'iteration':i}for i in (1,2)])
        save(state_path,state);save(args.ledger,{'retained_successful_calls':3})
        ledger_before=json.loads(args.ledger.read_text());args.out.mkdir();(args.out/'phases').mkdir()
        with patch.object(serial.paths,'RESULTS_DIR',args.out.parent):source=serial.source_gate(args)
        self.assertEqual(source['initial_session']['calls'],3);self.assertEqual(source['initial_session']['phase'],'tune')
        checkpoint(args,source);serial.accepted_checkpoint(args,source)
        result=dict(termination=None,processes=[],attempted=dict.fromkeys(serial.PHASE_LIMITS,0),completed=dict.fromkeys(serial.PHASE_LIMITS,0))
        with patch.object(serial,'foreign_heavy_processes',return_value=[]),patch.object(serial,'check_budget'),\
             patch.object(serial,'phase_result'),patch.object(serial,'run_process',return_value=clean())as child:
            serial._launch(args,source,'step',1,StageRecorder(args.out/'events.jsonl'),result,time.monotonic()+50000)
        self.assertEqual(child.call_args.kwargs['timeout'],21840)
        self.assertEqual(child.call_args.kwargs['mem_gib'],4)
        self.assertEqual(json.loads(state_path.read_text()),state)
        self.assertEqual(json.loads(args.ledger.read_text()),ledger_before)
        self.assertEqual(result['processes'][0]['source_state_before']['calls'],3)
        self.assertEqual(result['processes'][0]['source_state_after']['calls'],3)

    def test_serial_invalid_direct_value_precedes_source_or_output(self):
        args=fixture(self.root)
        with patch.object(serial,'session_state')as state:
            for value in (True,float('nan'),float('inf'),7199,21601):
                args.step_seconds=value
                with self.assertRaises(ValueError):serial.execute(args)
            state.assert_not_called()
        self.assertFalse(args.out.exists())

    def test_serial_cli_validates_and_passes_default_or_explicit_without_execution(self):
        for suffix,want in (([],7200),(['--step-seconds','21600'],21600)):
            with patch.object(serial,'execute',return_value=dict(status='ok'))as execute,redirect_stdout(StringIO()):
                self.assertEqual(serial.main(['--source-case',str(self.root/'source'),'--out',str(self.root/'driver')]+suffix),0)
            self.assertEqual(execute.call_args.args[0].step_seconds,want)
        with patch.object(serial,'execute')as execute,redirect_stderr(StringIO()):
            for value in ('nan','inf','true','7199','21601'):
                with self.assertRaises(SystemExit):serial.main(['--source-case','/source','--out','/out','--step-seconds',value])
            execute.assert_not_called()

    def test_detached_launch_forwards_explicit_allowance_with_existing_checkpoint(self):
        args=fixture(self.root);args.step_seconds=21600
        with patch.object(serial.paths,'RESULTS_DIR',args.out.parent):source=serial.source_gate(args)
        checkpoint(args,source)
        with patch.object(serial.paths,'RESULTS_DIR',args.out.parent),\
             patch.object(serial,'foreign_heavy_processes',return_value=[]),\
             patch.object(serial.subprocess,'Popen',return_value=Namespace(pid=123))as child:
            record=serial.start_detached(args)
        argv=child.call_args.args[0]
        self.assertEqual(argv[argv.index('--step-seconds')+1],'21600')
        self.assertEqual(argv[argv.index('--checkpoint-out')+1],str(args.checkpoint_out))
        self.assertEqual(record['status'],'detached_started')


if __name__=='__main__':unittest.main()
