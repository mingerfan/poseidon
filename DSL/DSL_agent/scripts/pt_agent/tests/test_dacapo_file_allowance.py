"""Finite per-file policy transport only; all native commands are mocked."""
from contextlib import nullcontext,redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_original_compile as entry
from pt_agent.tests import test_dacapo_diagnostic_output as prior


class DaCapoFileAllowanceChecks(unittest.TestCase):
    def test_cli_default_eight_explicit_sixteen_and_invalid_values(self):
        for options,want in (([],8),(['--file-gib','16'],16)):
            captured=[]
            with patch.object(entry,'run',side_effect=lambda args:captured.append(args) or 0):
                self.assertEqual(entry.main(['--model','vgg16','--out','/new','--case-gib','13','--planning-gib','12',*options]),0)
            self.assertEqual(captured[0].file_gib,want);self.assertEqual(captured[0].case_gib,13)
        for value in ('0','-1','16.1','nan','inf'):
            with self.subTest(value=value),redirect_stderr(StringIO()),patch.object(entry,'run',side_effect=AssertionError('Invalid policy cannot execute')):
                with self.assertRaises(SystemExit):entry.main(['--out','/new','--file-gib',value])

    def test_both_nix_workers_forward_sixteen_and_legacy_default_argv_stays_unchanged(self):
        args=SimpleNamespace(model='vgg16',trace_seconds=1800,compile_seconds=7200,compiler=Path('/compiler'),memory_gib=12,
            intake_python=Path('/python'),source=Path('/official/VGG16.py'),frontend=Path('/frontend'),poly_root=Path('/poly'),einops=Path('/einops'))
        for phase in ('trace','compile'):
            before=entry._nix_worker(args,phase,Path('/work'),command_file=Path('/command.json'))
            self.assertNotIn('--file-gib',before)
            args.file_gib=16
            changed=entry._nix_worker(args,phase,Path('/work'),command_file=Path('/command.json'))
            index=changed.index('--file-gib');self.assertEqual(changed[index+1],'16')
            self.assertEqual(changed[:index]+changed[index+2:],before)
            del args.file_gib

    def test_native_compile_worker_passes_sixteen_to_the_existing_resource_supervisor(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);command=entry.compiler_argv(prior.template(),binary='/compiler',source='/input',profile='/profile',output='/lowered')
            (root/'command.json').write_text(json.dumps(command))
            args=SimpleNamespace(worker_dir=root/'worker',command_file=root/'command.json',compiler=Path('/compiler'),
                worker_seconds=7200,memory_gib=12,file_gib=16)
            with patch.dict(os.environ,{'HECATE_PYTHON_LIBRARY_PATH':'/lib'}),\
                    patch.object(entry,'run_process',return_value=dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])) as child:
                self.assertEqual(entry.compile_worker(args),0)
            self.assertEqual(child.call_args.kwargs['fsize_gib'],16)
            self.assertEqual(json.loads((args.worker_dir/'worker_result.json').read_text())['file_gib'],16)

    def test_outer_trace_and_compile_both_enforce_sixteen_and_keep_case_thirteen_reserve_ten(self):
        with TemporaryDirectory() as temp:
            args=prior.DaCapoDiagnosticChecks().fixture(Path(temp));args.file_gib=16;seen=[]
            def process(argv,**kwargs):
                Path(kwargs['log']).write_text('synthetic-revision\n');Path(kwargs['resources']).write_text('')
                if argv[0]=='git':return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
                seen.append(argv);self.assertEqual(kwargs['fsize_gib'],16)
                self.assertEqual(argv[argv.index('--file-gib')+1],'16')
                work=Path(argv[argv.index('--worker-dir')+1])
                if '--worker-trace' in argv:
                    trace=work/'traced/VGG16.mlir';trace.parent.mkdir();trace.write_text('synthetic trace')
                    result=dict(status='ok',trace_completed=1,file_gib=16)
                else:prior.exported(work);result=dict(status='ok',compile_attempts=1,compile_completed=1,file_gib=16)
                (work/'worker_result.json').write_text(json.dumps(result))
                return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])
            with patch.object(entry,'run_process',side_effect=process),\
                    patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=23*1024**3)):
                self.assertEqual(entry.run(args),0)
            self.assertEqual(len(seen),2)
            result=json.loads((args.out/'result.json').read_text())
            self.assertEqual(result['limits'],dict(file_gib=16,memory_gib=12,case_gib=13,minimum_free_disk_gib=10))
            self.assertEqual((result['trace_completed'],result['compile_completed']),(1,1))

    def test_trace_worker_records_selected_inherited_file_policy_without_running_original_model(self):
        from pt_agent.models import dacapo_builders
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'VGG16.py';source.write_text('# unexecuted original source fixture\n')
            args=SimpleNamespace(worker_dir=root/'worker',source=source,frontend=root/'frontend',compiler=root/'build/bin/compiler',
                poly_root=root/'poly',einops=root/'einops',worker_seconds=1,file_gib=16)
            def trace(*unused,**kwargs):Path('traced/VGG16.mlir').write_text('synthetic trace')
            with patch.object(dacapo_builders,'_torchvision_imports',return_value=nullcontext()),\
                    patch.object(entry.runpy,'run_path',side_effect=trace):
                self.assertEqual(entry.trace_worker(args),0)
            result=json.loads((args.worker_dir/'worker_result.json').read_text())
            self.assertEqual(result['file_gib'],16);self.assertEqual(result['trace_completed'],1)
            self.assertEqual(result['quality_model_forwards'],0)


if __name__=='__main__':unittest.main()
