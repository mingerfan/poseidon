"""Only synthetic artifacts/fake commands exercise original DaCapo print policy."""
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_original_compile as entry
from pt_agent.tests.test_dacapo_original_compile import fixtures,plan_fixture


def template():
    return ['--runtime-plan-format=json','--runtime-plan-bundle-format=files',
            '--runtime-plan-operator-spec-sha256=sha256:'+'0'*64]


def exported(work, *, elided=True):
    plan=plan_fixture()
    plan.update(plaintext_bundle=dict(id='synthetic-bundle',version=1),
        initialization=[dict(kind='encode',output='2',payload=dict(kind='bundle',content='fixture:constant'))])
    plan['values'].append(dict(id='2',context=entry.CONTEXT,level=3,scale_log2=48,components=1))
    path=work/'lowered._hecate_VGG16.runtime-plan.json';path.write_text(json.dumps(plan))
    (work/'lowered.mlir').write_text('module { // '+('dense_resource<__elided__>' if elided else 'full synthetic text')+'\n}\n')
    bundle=work/'lowered._hecate_VGG16.bundle';(bundle/'data').mkdir(parents=True)
    (bundle/'manifest.json').write_text(json.dumps(dict(bundle_format_version=1,bundle_id='synthetic-bundle',version=1,
        blobs=[dict(content='fixture:constant',byte_length=16)])))
    (bundle/'data/constant.bin').write_bytes(b'synthetic-input!')
    return path,plan,bundle


class DaCapoDiagnosticChecks(unittest.TestCase):
    def clean(self):return dict(exit_code=0,timed_out=False,signal=None,signals_sent=[],remaining_pids=[])

    def command(self,**changes):
        args=dict(binary='/compiler',source='/saved/trace.mlir',profile='/original/profile.json',output='/new/lowered.mlir')
        args.update(changes);return entry.compiler_argv(template(),**args)

    def test_default_command_is_preserved_and_enabled_option_changes_only_printing(self):
        default=self.command();explicit=self.command(diagnostic_elide_final_ir=False)
        changed=self.command(diagnostic_elide_final_ir=True)
        self.assertEqual(default,explicit);self.assertNotIn(entry.DIAGNOSTIC_ELIDE_OPTION,default)
        self.assertEqual(changed.count(entry.DIAGNOSTIC_ELIDE_OPTION),1)
        self.assertEqual([v for v in changed if v!=entry.DIAGNOSTIC_ELIDE_OPTION],default)
        self.assertEqual(changed[changed.index('-o')+1],'/new/lowered.mlir')
        for value in ('--dacapo','--boot-placement=optimized','--waterline=40',
                      '--runtime-plan-format=json','--runtime-plan-bundle-format=files'):
            self.assertIn(value,changed)
        self.assertNotIn('--pars',changed);self.assertNotIn('--enable-debug-printer',changed)

    def test_enabled_option_rejects_discarded_output_and_nonboolean_choice(self):
        for out in ('-','/dev/null'):
            with self.subTest(out=out),self.assertRaises(ValueError):self.command(output=out,diagnostic_elide_final_ir=True)
        with self.assertRaises(ValueError):self.command(diagnostic_elide_final_ir=1)

    def test_default_cli_keeps_original_model_allowances_and_accepts_explicit_small_disk_case(self):
        for model in ('vgg16','alexnet'):
            for flags,chosen in (([],False),(['--diagnostic-elide-final-ir','--planning-gib','12','--case-gib','13'],True)):
                with self.subTest(model=model,chosen=chosen):
                    captured=[]
                    with patch.object(entry,'run',side_effect=lambda args:captured.append(args) or 0):
                        self.assertEqual(entry.main(['--model',model,'--out','/new-case',*flags]),0)
                    args=captured[0];self.assertIs(args.diagnostic_elide_final_ir,chosen)
                    self.assertEqual(args.planning_gib,12 if chosen else 24)
                    self.assertEqual(args.case_gib,13 if chosen else 32)
                    self.assertEqual(args.waterline,40);self.assertEqual(args.compile_seconds,7200)

    def test_only_compile_worker_receives_explicit_choice(self):
        args=SimpleNamespace(trace_seconds=1800,compile_seconds=7200,compiler=Path('/compiler'),memory_gib=12,
            intake_python=Path('/python'),source=Path('/official/VGG16.py'),frontend=Path('/frontend'),
            poly_root=Path('/poly'),einops=Path('/einops'),diagnostic_elide_final_ir=True)
        trace=entry._nix_worker(args,'trace',Path('/trace'))
        compiled=entry._nix_worker(args,'compile',Path('/compile'),command_file=Path('/argv.json'))
        self.assertNotIn('--diagnostic-elide-final-ir',trace)
        self.assertEqual(compiled.count('--diagnostic-elide-final-ir'),1)
        self.assertEqual(compiled[compiled.index('--worker-seconds')+1],'7200')

    def test_compile_worker_validates_saved_choice_and_runs_one_fake_command(self):
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch),TemporaryDirectory() as directory:
                root=Path(directory);command=self.command(diagnostic_elide_final_ir=True)
                path=root/'argv.json';path.write_text(json.dumps(command))
                args=SimpleNamespace(worker_dir=root/'worker',command_file=path,compiler=Path('/compiler'),worker_seconds=7200,
                    memory_gib=12,diagnostic_elide_final_ir=not mismatch)
                with patch.dict(os.environ,{'HECATE_PYTHON_LIBRARY_PATH':'/trusted/lib'}),\
                        patch.object(entry,'run_process',return_value=self.clean()) as process:
                    self.assertEqual(entry.compile_worker(args),1 if mismatch else 0)
                self.assertEqual(process.call_count,0 if mismatch else 1)
                record=json.loads((args.worker_dir/'worker_result.json').read_text())
                self.assertEqual(record['compile_completed'],0 if mismatch else 1)
                if not mismatch:self.assertEqual(process.call_args.args[0],command)

    def fixture(self,root,*,elided=True):
        upstream=root/'official';selected=entry.model_paths(upstream,'vgg16')
        for path in (selected['script'],selected['checkpoint'],selected['model_source']):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('unexecuted original source fixture')
        compiler=root/'build/bin/compiler';frontend=root/'frontend';poly=upstream/'python/poly';einops=root/'einops'
        files=[compiler,compiler.parent.parent/'lib/libHecateFrontend.so',frontend/'python/hecate/hecate/expr.py',
            poly/'poly/Poly.py',poly/'poly/Func.py',poly/'poly/MPCB.py',einops/'einops/__init__.py']
        files += [poly/'poly/data'/name for name in entry.POLY_FILES]
        for path in files:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('unexecuted fixture')
        compiler_profile,imported=fixtures()
        for name,value in (('compiler-profile',compiler_profile),('imported-spec',imported),('template',template()),
            ('r13',dict(round='R13',status='ok',model=selected['factory'],worker=dict(model_source=dict(
                factory=selected['factory'],checkpoint=str(selected['checkpoint']),original_source=str(selected['model_source'])))))):
            (root/(name+'.json')).write_text(json.dumps(value))
        return SimpleNamespace(out=root/'new-case',model='vgg16',source=selected['script'],checkpoint=selected['checkpoint'],
            upstream=upstream,compiler=compiler,frontend=frontend,poly_root=poly,einops=einops,intake_python=Path('/python'),
            compiler_profile=root/'compiler-profile.json',imported_spec=root/'imported-spec.json',metadata_template=root/'template.json',
            source_r13_result=root/'r13.json',registry=root/'registry.jsonl',waterline=40,memory_gib=12,case_gib=13,planning_gib=12,
            trace_seconds=1800,compile_seconds=7200,diagnostic_elide_final_ir=elided)

    def test_full_outer_fake_run_preserves_boot_metrics_full_artifacts_and_disk_reserve(self):
        with TemporaryDirectory() as directory:
            args=self.fixture(Path(directory));seen=[]
            def process(argv,**kwargs):
                Path(kwargs['log']).write_text('synthetic-revision\n');Path(kwargs['resources']).write_text('')
                if argv[0]=='git':return self.clean()
                seen.append(argv);work=Path(argv[argv.index('--worker-dir')+1])
                if '--worker-trace' in argv:
                    trace=work/'traced/VGG16.mlir';trace.parent.mkdir();trace.write_text('synthetic source')
                    value=dict(status='ok',trace_completed=1)
                else:
                    command=json.loads(Path(argv[argv.index('--command-file')+1]).read_text())
                    self.assertEqual(command.count(entry.DIAGNOSTIC_ELIDE_OPTION),1)
                    self.assertEqual(command[command.index('-o')+1],str(work/'lowered.mlir'))
                    exported(work);value=dict(status='ok',compile_attempts=1,compile_completed=1)
                (work/'worker_result.json').write_text(json.dumps(value));return self.clean()
            with patch.object(entry,'run_process',side_effect=process),\
                    patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=23*1024**3)):
                self.assertEqual(entry.run(args),0)
            self.assertEqual(len(seen),2);result=json.loads((args.out/'result.json').read_text())
            self.assertEqual(result['bootstrap_count'],1);self.assertEqual(result['compile_completed'],1)
            self.assertEqual(result['native_compile_completed'],1)
            self.assertEqual(result['compiler_output']['plaintext_bundle']['blob_bytes'],16)
            self.assertFalse(result['compiler_output']['executable_payloads_elided'])
            self.assertEqual(result['compiler_output_variant'],'diagnostic-elide-nonsplat-elements-over16')
            for key in ('keygen_runs','real_ckks_attempts','report_samples','api_calls'):self.assertEqual(result[key],0)
            rows=[json.loads(line) for line in args.registry.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['metrics']['bootstrap_count'],1)
            self.assertEqual(set(rows[0]),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            self.assertEqual(json.loads((args.out/'completion.json').read_text())['status'],'ok')

    def test_diagnostic_output_still_rejects_missing_manifest_or_wrong_blob_length(self):
        for defect in ('manifest','length','reference'):
            with self.subTest(defect=defect),TemporaryDirectory() as directory:
                work=Path(directory);path,plan,bundle=exported(work)
                if defect=='manifest':(bundle/'manifest.json').unlink()
                elif defect=='length':(bundle/'data/constant.bin').write_bytes(b'truncated')
                else:plan['initialization'][0]['payload']['content']='fixture:missing'
                with self.assertRaises(ValueError):entry.compile_artifact_metadata(work,path,plan,elided=True)

    def test_ten_gib_free_reserve_still_stops_before_any_child(self):
        with TemporaryDirectory() as directory:
            with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3-1)),\
                    patch.object(entry,'run_process',side_effect=AssertionError('No child may enter')):
                with self.assertRaisesRegex(RuntimeError,'allowance exhausted'):
                    with entry._budget_guard(Path(directory),case_gib=13,minimum_free_gib=10):self.fail('Reserve must hold')


if __name__=='__main__':unittest.main()
