"""Synthetic selection/trace checks for original VGG16 and AlexNet baselines."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_original_compile as subject


class OriginalModelSelectionChecks(unittest.TestCase):
    def test_three_original_scripts_and_checkpoint_names_remain_explicit(self):
        root=Path('/official/dacapo')
        expected={
            'resnet20':('ResNet','resnet20.silu.model','resnet20_silu'),
            'vgg16':('VGG16','vgg16_silu_avgpool_model','vgg16_silu_avgpool'),
            'alexnet':('AlexNet','alexNet_silu_avgpool_model','alexnet_silu_avgpool')}
        for model,(entry,checkpoint,factory) in expected.items():
            with self.subTest(model=model):
                record=subject.model_paths(root,model)
                self.assertEqual(record['script'],root/'examples/benchmarks'/(entry+'.py'))
                self.assertEqual(record['checkpoint'],root/'examples/data'/checkpoint)
                self.assertEqual(record['model_source'],root/'python/poly/poly/models'/(entry+'.py'))
                self.assertEqual(record['trace_relative'],Path('traced')/(entry+'.mlir'))
                self.assertEqual(record['factory'],'pt_agent.experiments.r13_common_models:'+factory)

    def test_r13_identity_matches_each_selected_source_without_checkpoint_read(self):
        root=Path('/official/dacapo')
        for model in subject.MODEL_SOURCES:
            selected=subject.model_paths(root,model)
            record=dict(round='R13',status='ok',model=selected['factory'],worker=dict(model_source=dict(
                factory=selected['factory'],checkpoint=str(selected['checkpoint']),
                original_source=str(selected['model_source']),training_preprocessing_verified=False)))
            observed=subject.validate_source_record(record,upstream=root,checkpoint=selected['checkpoint'],model=model)
            self.assertEqual(observed['factory'],selected['factory'])
            self.assertFalse(observed['training_preprocessing_verified'])
            record['model']='different:factory'
            with self.assertRaisesRegex(ValueError,'Saved original ModelInfo'):
                subject.validate_source_record(record,upstream=root,checkpoint=selected['checkpoint'],model=model)

    def test_unknown_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown original'):
            subject.model_paths('/official','custom_tiny')

    def test_large_model_cli_keeps_original_profile_and_finite_default_allowances(self):
        for model in ('vgg16','alexnet'):
            with self.subTest(model=model):
                seen=[]
                with patch.object(subject,'run',side_effect=lambda args:seen.append(args) or 0):
                    self.assertEqual(subject.main(['--model',model,'--out','/fresh/case']),0)
                args=seen[0]
                source=subject.model_paths(args.upstream,model)
                self.assertEqual(args.source,source['script'])
                self.assertEqual(args.checkpoint,source['checkpoint'])
                self.assertEqual(args.source_r13_result.parent.name,subject.model_source(model)['r13_run'])
                self.assertEqual(args.compiler_profile.name,'profiled_HEAAN_CPU.json')
                self.assertEqual((args.memory_gib,args.case_gib,args.planning_gib),(12,32,24))
                self.assertEqual((args.waterline,args.trace_seconds,args.compile_seconds),(40,1800,7200))

    def test_resnet_existing_defaults_are_preserved(self):
        seen=[]
        with patch.object(subject,'run',side_effect=lambda args:seen.append(args) or 0):
            self.assertEqual(subject.main(['--out','/fresh/case']),0)
        args=seen[0]
        self.assertEqual(args.model,'resnet20')
        self.assertEqual((args.case_gib,args.planning_gib),(16,6))
        self.assertEqual(args.source.name,'ResNet.py')

    def test_model_selector_is_explicit_inside_nix_argv(self):
        args=SimpleNamespace(model='vgg16',trace_seconds=1800,compile_seconds=7200,
            compiler=Path('/existing/compiler'),memory_gib=12,intake_python=Path('/venv/python'),
            source=Path('/official/VGG16.py'),frontend=Path('/frontend'),poly_root=Path('/poly'),einops=Path('/einops'))
        for phase in ('trace','compile'):
            argv=subject._nix_worker(args,phase,Path('/case/work'),command_file=Path('/case/command.json'))
            self.assertEqual(argv[argv.index('--model')+1],'vgg16')
            self.assertEqual(argv[argv.index('--worker-seconds')+1],str(1800 if phase=='trace' else 7200))

    def test_original_save_filename_is_selected_for_vgg_and_alex(self):
        from pt_agent.models import dacapo_builders
        for entry in ('VGG16','AlexNet'):
            with self.subTest(entry=entry),tempfile.TemporaryDirectory() as directory:
                base=Path(directory)
                source=base/(entry+'.py');source.write_text('# trusted synthetic file only\n')
                args=SimpleNamespace(worker_dir=base/'work',source=source,frontend=base/'frontend',
                    compiler=base/'build/bin/compiler',poly_root=base/'poly',einops=base/'einops',worker_seconds=5)
                finalized=[]
                hc=SimpleNamespace(ctxt=1,lt=SimpleNamespace(finalize=lambda ptr:finalized.append(ptr)))
                def trusted_trace(path,run_name):
                    self.assertEqual(Path(path),source)
                    self.assertEqual(sys.argv,[str(source)])
                    self.assertEqual(run_name,'__main__')
                    Path('traced',entry+'.mlir').write_text('module {}')
                    sys.modules['hecate']=hc
                with patch.object(dacapo_builders,'_torchvision_imports',return_value=nullcontext()):
                    with patch.object(subject.runpy,'run_path',side_effect=trusted_trace),patch.dict(sys.modules):
                        self.assertEqual(subject.trace_worker(args),0)
                result=json.loads((args.worker_dir/'worker_result.json').read_text())
                self.assertEqual(Path(result['mlir']['path']).name,entry+'.mlir')
                self.assertEqual(finalized,[1])
                self.assertEqual(result['quality_model_forwards'],0)

    def test_registry_uses_selected_model_and_unique_case_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);out=base/'fresh-vgg';registry=base/'registry.jsonl'
            result=dict(model=subject.model_source('vgg16')['result_model'],status='ok')
            subject._register(registry,result,out,'captured-revision')
            row=json.loads(registry.read_text())
            self.assertEqual(row['model'],'dacapo-original-vgg16-silu-avgpool')
            self.assertEqual(set(row),{'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'})
            with self.assertRaisesRegex(RuntimeError,'already exists'):
                subject._register(registry,result,out,'captured-revision')


if __name__=='__main__':
    unittest.main()
