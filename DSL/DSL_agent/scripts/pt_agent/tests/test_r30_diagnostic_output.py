"""Synthetic printer-policy and executable-artifact checks; no compiler runs."""
from argparse import Namespace
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import r30_common_model_e2e as entry
from pt_agent.experiments import r29_local_compile_matrix as commands


def template():
    return ['{hecate_opt}', '{input}', '--pars',
            '--ckks-config={package}/config/compiler-config.json', '--waterline=43',
            '--runtime-plan-context-id=qwen-bootstrap48',
            '--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5',
            '--runtime-plan-format=json', '--runtime-plan-bundle-format=files',
            '--runtime-plan-operator-spec-sha256=sha256:' + '0' * 64,
            '-o', '{output_dir}/lowered.mlir']


def artifacts(emit):
    emit.mkdir(parents=True, exist_ok=True)
    plan_path = emit / 'lowered._hecate_golden.runtime-plan.json'
    plan = dict(plaintext_bundle=dict(id='synthetic', version=1),
                initialization=[dict(kind='encode', output='p0',
                                     payload=dict(kind='bundle', content='fixture:constant'))])
    plan_path.write_text(json.dumps(plan))
    (emit / 'lowered.mlir').write_text('module { // diagnostic dense_resource<__elided__>\n}\n')
    bundle = emit / 'lowered._hecate_golden.bundle'
    (bundle / 'data').mkdir(parents=True)
    (bundle / 'manifest.json').write_text(json.dumps(dict(bundle_format_version=1, version=1,
        bundle_id='synthetic', blobs=[dict(content='fixture:constant', byte_length=16)])))
    (bundle / 'data/constant.bin').write_bytes(b'synthetic-input!')
    return plan_path, plan, bundle


class DiagnosticOutputChecks(unittest.TestCase):
    def test_default_argv_and_agent_variant_are_preserved(self):
        argv=commands.compiler_argv(template(),'pars',binary='/compiler',source='/fresh/input.mlir',
            profile='/original/profile.json',output='/fresh/lowered.mlir')
        self.assertEqual(entry.diagnostic_compile_argv(argv),argv)
        self.assertIsNot(entry.diagnostic_compile_argv(argv),argv)
        args=Namespace(tuner=True)
        self.assertEqual(entry.compiler_output_variant(args),'full-final-mlir')
        self.assertEqual(entry.agent_variant(args),entry.VARIANT)

    def test_elision_changes_only_one_print_option_and_keeps_export_prefix(self):
        argv=commands.compiler_argv(template(),'pars',binary='/compiler',source='/fresh/input.mlir',
            profile='/original/profile.json',output='/fresh/lowered.mlir')
        changed=entry.diagnostic_compile_argv(argv,elide_final_ir=True)
        self.assertEqual(changed.count(entry.DIAGNOSTIC_ELIDE_OPTION),1)
        self.assertEqual([item for item in changed if item!=entry.DIAGNOSTIC_ELIDE_OPTION],argv)
        self.assertEqual(changed[changed.index('-o')+1],'/fresh/lowered.mlir')
        self.assertIn('--runtime-plan-format=json',changed)
        self.assertIn('--runtime-plan-bundle-format=files',changed)
        self.assertNotIn('/dev/null',changed)

    def test_conflicting_flags_or_discarded_output_are_rejected(self):
        base=['/compiler','/input','-o','/fresh/lowered.mlir']
        for argv in (base+[entry.DIAGNOSTIC_ELIDE_OPTION],
                     base+['--enable-debug-printer'],
                     ['/compiler','/input','-o','/dev/null'],
                     ['/compiler','/input','-o','-'],['/compiler','/input']):
            with self.subTest(argv=argv),self.assertRaises(ValueError):
                entry.diagnostic_compile_argv(argv,elide_final_ir=True)

    def test_outer_command_forwards_only_the_explicit_diagnostic_choice(self):
        args=Namespace(out=Path('/fresh'),model='alexnet_silu_avgpool',inputs=Path('/records'),
            selections=Path('/selections'),agent_seconds=7200,trace_seconds=1800,compile_seconds=7200,
            ledger=Path('/ledger'),allowance=Path('/allowance'),profile=Path('/profile'),
            frontend=Path('/frontend'),metadata_template=Path('/template'),calls=6,tuner_seconds=600,
            memory_gib=12,case_gib=32)
        default,_=entry.phase_command(args,'compile',Path('/fresh/work/compile'),remaining_seconds=28000)
        self.assertNotIn('--diagnostic-elide-final-ir',default)
        args.diagnostic_elide_final_ir=True
        changed,_=entry.phase_command(args,'compile',Path('/fresh/work/compile'),remaining_seconds=28000)
        self.assertEqual(changed.count('--diagnostic-elide-final-ir'),1)
        self.assertEqual([item for item in changed if item!='--diagnostic-elide-final-ir'],default)

    def test_cli_default_false_and_explicit_choice_are_separate_from_tuning(self):
        for flags,expected in (([],False),(['--diagnostic-elide-final-ir'],True)):
            captured=[]
            with patch.object(entry,'execute',side_effect=lambda args:captured.append(args) or dict(status='ok')):
                self.assertEqual(entry.main(['--model','alexnet_silu_avgpool','--out','/fresh',*flags]),0)
            self.assertIs(captured[0].diagnostic_elide_final_ir,expected)
            self.assertIs(captured[0].tuner,True)

    def test_elided_text_still_requires_complete_plan_and_payload_inventory(self):
        with TemporaryDirectory() as directory:
            emit=Path(directory)/'emit';plan_path,plan,bundle=artifacts(emit)
            result=entry.compile_artifact_metadata(emit,plan_path,plan,elided=True)
            self.assertIs(result['final_ir']['constant_preserving_roundtrip'],False)
            self.assertIs(result['executable_payloads_elided'],False)
            self.assertEqual(result['plaintext_bundle']['blob_count'],1)
            self.assertEqual(result['plaintext_bundle']['referenced_blob_count'],1)
            self.assertEqual(result['plaintext_bundle']['blob_bytes'],16)
            self.assertEqual(result['plaintext_bundle']['payload_values_read'],0)

    def test_missing_final_text_manifest_or_payload_never_passes(self):
        for missing in ('lowered.mlir','lowered._hecate_golden.bundle/manifest.json',
                        'lowered._hecate_golden.bundle/data/constant.bin'):
            with self.subTest(missing=missing),TemporaryDirectory() as directory:
                emit=Path(directory)/'emit';plan_path,plan,_=artifacts(emit)
                (emit/missing).unlink()
                with self.assertRaises(ValueError):
                    entry.compile_artifact_metadata(emit,plan_path,plan,elided=True)

    def test_foreign_manifest_id_and_wrong_payload_length_are_rejected(self):
        for changed in (dict(bundle_id='foreign'),dict(blobs=[dict(content='fixture:constant',byte_length=17)])):
            with self.subTest(changed=changed),TemporaryDirectory() as directory:
                emit=Path(directory)/'emit';plan_path,plan,bundle=artifacts(emit)
                path=bundle/'manifest.json';manifest=json.loads(path.read_text());manifest.update(changed)
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):entry.compile_artifact_metadata(emit,plan_path,plan,elided=True)

    def test_real_compile_orchestration_forwards_flag_and_observes_full_artifacts(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);emit=root/'post/emit';work=root/'work';work.mkdir()
            recipe=root/'template.json';recipe.write_text(json.dumps(template()))
            profile=root/'profile.json';profile.write_text('{}')
            args=Namespace(worker_phase='compile',post_dir=root/'post',work_dir=work,
                metadata_template=recipe,profile=profile,compile_seconds=7200,memory_gib=12,
                diagnostic_elide_final_ir=True)
            (root/'post').mkdir();(root/'post/refresh-plan.json').write_text(json.dumps(dict(boots=1)))
            seen=[]
            def process(argv,**kwargs):
                seen.append(argv);artifacts(emit)
                return dict(exit_code=0,timed_out=False,signal=None,remaining_pids=[])
            from pt_agent import toolchain
            recorder=SimpleNamespace(span=lambda *args,**kwargs:nullcontext())
            paths=dict(hecate='/hecate',hecate_build='/build',hecate_opt='/compiler')
            result={}
            # The original dispatcher imports numeric parser dependencies even
            # for its compile-only branch. These unused imports stay isolated;
            # the real branch, command builder and artifact checks still run.
            parser=ModuleType('pt_agent.dsl.parser');parser.parse=None
            checker=ModuleType('pt_agent.dsl.typecheck');checker.check=None
            with patch.object(toolchain,'host_paths',return_value=paths),patch.dict('os.environ',{'HECATE_PYTHON_LIBRARY_PATH':'/library'}),\
                    patch.dict(sys.modules,{'pt_agent.dsl.parser':parser,'pt_agent.dsl.typecheck':checker}),\
                    patch.object(entry,'run_process',side_effect=process),\
                    patch.object(commands,'plan_observation',return_value=dict(compiled_bootstraps=1,predicted_latency_us=None)):
                entry._post_freeze(args,recorder,result)
            self.assertEqual(len(seen),1)
            self.assertIn(entry.DIAGNOSTIC_ELIDE_OPTION,seen[0])
            self.assertEqual(result['compiled_bootstraps'],1)
            self.assertEqual(result['compiler_output_variant'],'diagnostic-elide-nonsplat-elements-over16')
            self.assertEqual(result['compiler_output']['plaintext_bundle']['blob_bytes'],16)
            self.assertIs(result['compiler_output']['executable_payloads_elided'],False)


if __name__=='__main__':unittest.main()
