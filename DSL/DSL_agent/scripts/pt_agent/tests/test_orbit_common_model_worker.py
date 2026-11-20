"""Small fake/AST checks of official common Orbit frontend dispatch."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import orbit_common_model_worker as subject
from pt_agent.experiments import r16_orbit_models
from pt_agent.experiments.r10_paper_metrics import StageRecorder


PREFIX='pt_agent.experiments.r13_common_models:'


class OrbitCommonModelWorkerChecks(unittest.TestCase):
    def tearDown(self):subject.configure_model(PREFIX+'resnet20_silu')

    def test_original_model_entries_and_activation_coverage(self):
        expected={'resnet20_silu':('ResNetSiLU64k','ResNetSiLU.py',19,True),
            'vgg16_silu_avgpool':('VGG16SiLU64k','VGG16SiLU.py',15,False),
            'alexnet_silu_avgpool':('AlexNetSiLU64k','AlexNetSiLU.py',7,False)}
        for suffix,values in expected.items():
            record=subject.configure_model(PREFIX+suffix)
            self.assertEqual((record['benchmark'],record['source'],record['activations'],record['checkpoint_envelope']),values)
            self.assertEqual(subject.MODEL,PREFIX+suffix)
            self.assertEqual(subject.BENCHMARK_NAME,values[0])
            self.assertEqual(subject.EXPECTED_ACTIVATIONS,values[2])

    def test_vgg_and_alex_required_helpers_are_explicit(self):
        vgg=subject.configure_model(PREFIX+'vgg16_silu_avgpool')
        self.assertIn('CascadeMax',vgg['mpcb']);self.assertIn('HE_Avg',vgg['functions']);self.assertIn('HE_MPBN',vgg['functions'])
        alex=subject.configure_model(PREFIX+'alexnet_silu_avgpool')
        self.assertIn('HE_ReshapeLinear',alex['functions']);self.assertIn('HE_Avg',alex['functions'])
        with self.assertRaisesRegex(ValueError,'Unknown original'):
            subject.configure_model('custom:model')

    def test_only_selected_synchronous_official_function_is_executed(self):
        for suffix in ('vgg16_silu_avgpool','alexnet_silu_avgpool'):
            name=subject.configure_model(PREFIX+suffix)['benchmark'];registered=[]
            def decorator(value):
                self.assertEqual(value,'c')
                def install(fn):registered.append(fn);return fn
                return install
            namespace=dict(hc=SimpleNamespace(func=decorator))
            with tempfile.TemporaryDirectory() as directory:
                source=Path(directory)/'trusted.py'
                source.write_text("raise RuntimeError('top-level must stay unexecuted')\n@hc.func('c')\ndef "+name+"(ctxt):\n    return ctxt\n")
                function=subject.load_benchmark_function(source,namespace)
                sentinel=object();self.assertIs(function(sentinel),sentinel)
                self.assertEqual(registered,[function])

    def test_decorator_and_async_mutations_are_rejected_before_execution(self):
        name=subject.configure_model(PREFIX+'vgg16_silu_avgpool')['benchmark']
        for prefix in ("@hc.func('p')\ndef ","@unapproved\ndef ","@hc.func('c')\nasync def "):
            with self.subTest(prefix=prefix),tempfile.TemporaryDirectory() as directory:
                source=Path(directory)/'fixture.py';source.write_text(prefix+name+'(ctxt):\n    return ctxt\n')
                with self.assertRaises(ValueError):subject.load_benchmark_function(source,{})

    def test_direct_checkpoint_state_and_data_parallel_wrapper_preserve_original_module(self):
        class Wrapper:
            def __init__(self,module):self.module=module
            def eval(self):return self
        state={'original_weight':object()};module=object();seen=[]
        report=dict(passed=True,strict_load=True,structure=dict(passed=True),extra_state=[])
        for suffix in ('vgg16_silu_avgpool','alexnet_silu_avgpool'):
            subject.configure_model(PREFIX+suffix)
            with tempfile.TemporaryDirectory() as directory:
                work=Path(directory);args=SimpleNamespace(work_dir=work,case_gib=4,checkpoint=work/'unused.model')
                result=subject._initial_result('trace')
                fake_torch=SimpleNamespace(nn=SimpleNamespace(DataParallel=Wrapper),load=lambda *a,**k:state)
                def build(model,source,incoming):
                    self.assertEqual(model,PREFIX+suffix);self.assertIs(incoming,state)
                    seen.append(incoming);return module,report
                args.model_source=work/'never-executed.py'
                with patch.object(subject,'_numeric_modules',return_value=(None,fake_torch)),patch.object(subject,'_check_sources',return_value={}),patch.object(r16_orbit_models,'build_model',side_effect=build),patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                    actual=subject._load_model(args,StageRecorder(work/'events.jsonl'),result)
                self.assertIs(actual.module,module)
                self.assertTrue(result['benchmark_wrapper']['added'])
                self.assertFalse(result['benchmark_wrapper']['original_state_values_changed'])
                self.assertEqual((result['checkpoint_load_completed'],result['model_build_completed']),(1,1))
        self.assertEqual(len(seen),2)

    def test_resnet_state_envelope_and_existing_parallel_wrapper_are_preserved(self):
        class Wrapper:
            def __init__(self,module):self.module=module
            def eval(self):return self
        state={'original':object()};module=Wrapper(object())
        subject.configure_model(PREFIX+'resnet20_silu')
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory);args=SimpleNamespace(work_dir=work,case_gib=4,checkpoint=work/'unused',model_source=work/'unused.py')
            result=subject._initial_result('trace');fake_torch=SimpleNamespace(nn=SimpleNamespace(DataParallel=Wrapper),load=lambda *a,**k:{'state_dict':state})
            report=dict(passed=True,strict_load=True,structure=dict(passed=True),extra_state=[])
            def build(model,source,incoming):self.assertIs(incoming,state);return module,report
            with patch.object(subject,'_numeric_modules',return_value=(None,fake_torch)),patch.object(subject,'_check_sources',return_value={}),patch.object(r16_orbit_models,'build_model',side_effect=build),patch.object(subject.shutil,'disk_usage',return_value=SimpleNamespace(free=1<<40)):
                actual=subject._load_model(args,StageRecorder(work/'events.jsonl'),result)
            self.assertIs(actual,module);self.assertFalse(result['benchmark_wrapper']['added'])

    def test_original_64k_profile_alias_remains_metadata_only(self):
        class Params:
            def __init__(self,path,sysname,mode,threads):
                self.poly_deg=0;self.max_slot=0;self.mode=mode;self.sysname=sysname;self.threads=threads
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json';path.write_text(json.dumps(dict(polynomialDegree=131072)))
            actual,metadata=subject._profile_params(path,Params)
            self.assertEqual((actual.poly_deg,actual.max_slot),(131072,65536))
            self.assertEqual(metadata['profile_use'],'parser_metadata_only')
            self.assertFalse(metadata['security_validated'])

    def test_source_gate_uses_the_official_patch_benchmark_path_before_reading_records(self):
        args=SimpleNamespace(model=PREFIX+'vgg16_silu_avgpool',orbit_root=Path('/official/orbit'),
            model_source=Path('/official/orbit/frontend/dacapo/python/poly/poly/models/VGG16SiLU.py'),
            benchmark=Path('/wrong/VGG16SiLU64k.py'))
        with patch.object(subject,'_read_record',side_effect=AssertionError('No source JSON read before path gate')):
            with self.assertRaisesRegex(subject.WorkerFailure,'exact official'):
                subject._check_sources(args)

    def test_new_round_has_no_native_optimizer_or_quality_claims(self):
        subject.configure_model(PREFIX+'alexnet_silu_avgpool')
        result=subject._initial_result('trace')
        self.assertEqual(result['round'],'ORBIT-COMMON-FRONTEND')
        self.assertEqual(result['model'],PREFIX+'alexnet_silu_avgpool')
        self.assertTrue(all(result[key]==0 for key in subject.ZERO_FIELDS))
        self.assertTrue(all(result[key] is None for key in subject.UNMEASURED_FIELDS))
        self.assertFalse(result['backend_ready']);self.assertFalse(result['formal_matrix_complete'])


if __name__=='__main__':unittest.main()
