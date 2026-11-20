"""Finite adapter-memory plumbing; all child processes and sources are fake."""
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_cnn_math_check as math_entry
from pt_agent.experiments import heir_common_model_pipeline as pipeline
from pt_agent.experiments import r10_paper_metrics as supervision


class AdapterMemoryChecks(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.source={name:self.root/name for name in
            ('graph_dir','program','mlir','probes','probe_source')}

    def math_args(self,**changes):
        values=dict(self.source,out=self.root/'math',common_model='alexnet_silu_avgpool',
            exact_optiona_selector=False,exact_spatial_mean_matvec=False)
        values.update(changes);return Namespace(**values)

    def pipeline_args(self,**changes):
        values=dict(model='alexnet_silu_avgpool',out=self.root/'pipeline',run=False,
            case_gib=32,registry=self.root/'registry.jsonl',export_python=Path('/fixture/python'),
            heir_opt=Path('/fixture/heir-opt'),translate=Path('/fixture/heir-translate'),
            go=Path('/fixture/go'),cache_source=self.root/'cache')
        values.update(changes);return Namespace(**values)

    def math_cli(self):
        command=['--supervise']
        for name,value in dict(self.source,out=self.root/'cli-math').items():
            command+=['--'+name.replace('_','-'),str(value)]
        return command

    @staticmethod
    def clean_observation(exit_code=0):
        return dict(exit_code=exit_code,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])

    def run_fake_math(self,**changes):
        captured={}
        def process(command,**options):
            captured.update(argv=command,options=options)
            target=Path(command[command.index('--out')+1]);target.mkdir(parents=True)
            (target/'result.json').write_text(json.dumps(dict(status='ok'))+'\n')
            Path(options['log']).write_text('fake supervised math\n')
            Path(options['resources']).write_text('')
            return self.clean_observation()
        with patch.object(math_entry,'__name__','pt_agent.experiments.heir_cnn_math_check'), \
                patch.object(pipeline.paths,'RESULTS_DIR',self.root), \
                patch.object(supervision,'run_process',side_effect=process),redirect_stdout(StringIO()):
            self.assertEqual(math_entry.supervised(self.math_args(**changes)),0)
        record=json.loads((self.root/'math/result.json').read_text())
        return captured,record

    def test_memory_validator_accepts_bounded_numbers_and_rejects_bool_nonfinite_or_outside(self):
        for value in (2,2.5,12):
            self.assertEqual(math_entry.validate_memory_gib(value),value)
        for value in (True,False,float('nan'),float('inf'),-float('inf'),1.999,12.001,0,'12',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                math_entry.validate_memory_gib(value)

    def test_supervised_default_retains_two_gib_and_records_actual_limit(self):
        captured,record=self.run_fake_math()
        self.assertEqual(captured['options']['mem_gib'],2)
        self.assertEqual(captured['options']['timeout'],360)
        argv=captured['argv']
        self.assertEqual(argv[argv.index('--memory-gib')+1],'2')
        self.assertEqual(argv[argv.index('--seconds')+1],'300')
        self.assertEqual(record['limits'],dict(AS_gib=2,nix_seconds=300,supervisor_seconds=360))
        rows=[json.loads(line) for line in (self.root/'registry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})

    def test_supervised_explicit_twelve_gib_reaches_real_process_interface_and_saved_metadata(self):
        captured,record=self.run_fake_math(memory_gib=12)
        self.assertEqual(captured['options']['mem_gib'],12)
        self.assertEqual(record['limits']['AS_gib'],12)
        command=record['process']['argv']
        self.assertEqual(command[command.index('--memory-gib')+1],'12')
        for name,path in self.source.items():
            self.assertEqual(command[command.index('--'+name.replace('_','-'))+1],str(path))
        self.assertIsNone(record['compiled_bootstraps']);self.assertIsNone(record['CKKS_latency'])

    def test_direct_supervised_invalid_memory_precedes_directory_or_child_creation(self):
        with patch.object(supervision,'run_process',side_effect=AssertionError('No child allowed')):
            for value in (True,float('nan'),1,13):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    math_entry.supervised(self.math_args(memory_gib=value))
        self.assertFalse((self.root/'math').exists())

    def test_math_cli_validates_before_delegation_and_preserves_default(self):
        for suffix,want in (([],2),(['--memory-gib','12'],12)):
            with patch.object(math_entry,'supervised',return_value=0) as call:
                self.assertEqual(math_entry.main(self.math_cli()+suffix),0)
            self.assertEqual(call.call_args.args[0].memory_gib,want)
        with patch.object(math_entry,'supervised') as call,redirect_stderr(StringIO()):
            for value in ('nan','inf','1','13','true'):
                with self.subTest(value=value),self.assertRaises(SystemExit) as error:
                    math_entry.main(self.math_cli()+['--memory-gib',value])
                self.assertEqual(error.exception.code,2)
            call.assert_not_called()
        self.assertFalse((self.root/'cli-math').exists())

    def planned(self,**changes):
        args=self.pipeline_args(**changes)
        return pipeline.commands(args,self.source,pipeline.phase_directories(args.out),self.root/'native.pipeline')

    def test_pipeline_default_passes_two_only_to_adapter_and_keeps_other_phase_limits(self):
        planned=self.planned()
        adapter=planned['adapter_math']
        self.assertEqual(adapter[adapter.index('--memory-gib')+1],'2')
        self.assertNotIn('--memory-gib',planned['export'])
        for stage,want in (('native_compile','12'),('go_codegen','4'),('clear_math','12')):
            argv=planned[stage];self.assertEqual(argv[argv.index('--memory-gib')+1],want)

    def test_pipeline_explicit_twelve_changes_only_adapter_memory_argument(self):
        original=self.planned();selected=self.planned(adapter_memory_gib=12)
        for stage in original:
            if stage!='adapter_math':self.assertEqual(original[stage],selected[stage])
        index=original['adapter_math'].index('--memory-gib')+1
        expected=list(original['adapter_math']);expected[index]='12'
        self.assertEqual(selected['adapter_math'],expected)
        self.assertIn('--supervise',expected)

    def test_pipeline_command_builder_rejects_invalid_direct_values(self):
        for value in (True,False,float('nan'),float('inf'),1,13):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.planned(adapter_memory_gib=value)

    def test_pipeline_execute_rejects_invalid_memory_before_lock_or_output(self):
        with patch.object(pipeline,'serial_lock',side_effect=AssertionError('No lock/output allowed')):
            for value in (True,float('nan'),1,13):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    pipeline.execute(self.pipeline_args(adapter_memory_gib=value))
        self.assertFalse((self.root/'pipeline').exists())

    def test_pipeline_cli_explicit_and_default_reach_existing_executor(self):
        for suffix,want in (([],2),(['--adapter-memory-gib','12'],12)):
            record=dict(status='prepared',model='alexnet_silu_avgpool',failed_stage=None,
                attempted={},completed={},static_native_bootstraps=None)
            with patch.object(pipeline,'execute',return_value=record) as call,redirect_stdout(StringIO()):
                self.assertEqual(pipeline.main(['--model','alexnet_silu_avgpool','--out',str(self.root/'cli')]+suffix),0)
            self.assertEqual(call.call_args.args[0].adapter_memory_gib,want)

    def test_pipeline_cli_rejects_nonfinite_outside_and_boolean_text_before_execute(self):
        with patch.object(pipeline,'execute') as call,redirect_stderr(StringIO()):
            for value in ('nan','inf','1','13','true'):
                with self.subTest(value=value),self.assertRaises(SystemExit) as error:
                    pipeline.main(['--model','alexnet_silu_avgpool','--out',str(self.root/'cli'),
                        '--adapter-memory-gib',value])
                self.assertEqual(error.exception.code,2)
            call.assert_not_called()

    def fake_pipeline_context(self):
        checked=dict(contract={},pipeline='builtin.module()',native_parameters={},source_samples=['train:0','train:1','train:2'])
        patches=(patch.object(pipeline.paths,'RESULTS_DIR',self.root),
            patch.object(pipeline,'sources',return_value=self.source),
            patch.object(pipeline,'source_gate',return_value=checked),
            patch.object(pipeline,'_available_ram_bytes',return_value=16*pipeline.GIB),
            patch.object(pipeline.shutil,'disk_usage',return_value=SimpleNamespace(free=100*pipeline.GIB)))
        for item in patches:item.start();self.addCleanup(item.stop)

    def test_prepared_record_preserves_outer_physical_guards_and_records_inner_twelve(self):
        self.fake_pipeline_context()
        with patch.object(pipeline,'run_process',side_effect=AssertionError('Preparation only')):
            record=pipeline.execute(self.pipeline_args(adapter_memory_gib=12))
        self.assertEqual(record['status'],'prepared')
        self.assertEqual(record['limits']['AS_gib'],12)
        self.assertEqual(record['limits']['adapter_math_child_AS_gib'],12)
        self.assertEqual(record['limits']['subtree_RSS_gib'],14)
        self.assertEqual(record['limits']['minimum_host_available_gib'],1.25)
        self.assertEqual(record['attempted'],dict.fromkeys(pipeline.PHASE_SECONDS,0))
        self.assertEqual(json.loads((self.root/'pipeline/result.json').read_text())['limits'],record['limits'])

    def test_fake_adapter_failure_still_stops_before_native_compile(self):
        self.fake_pipeline_context();calls=[]
        def process(command,**options):
            calls.append(command)
            Path(options['log']).write_text('fake phase\n');Path(options['resources']).write_text('')
            self.assertEqual(options['mem_gib'],12)
            return self.clean_observation(0 if len(calls)==1 else 1)
        with patch.object(pipeline,'run_process',side_effect=process), \
                patch.object(pipeline,'check_phase',return_value=dict(status='ok')):
            record=pipeline.execute(self.pipeline_args(run=True,adapter_memory_gib=12))
        self.assertEqual(record['status'],'error');self.assertEqual(record['failed_stage'],'adapter_math')
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1][calls[1].index('--memory-gib')+1],'12')
        self.assertEqual(record['completed']['export'],1);self.assertEqual(record['completed']['adapter_math'],0)
        self.assertEqual(record['attempted']['native_compile'],0)


if __name__=='__main__':unittest.main()
