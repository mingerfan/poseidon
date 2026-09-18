"""Immutable profile IDs, legacy compatibility, API allowlist and run evidence."""
import copy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
from workspace_paths import ROOT, WORK, RESULTS
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compiler_configuration as cfg
from candidate_contract import make_request, validate_candidate, canonical
from deepseek_provider import public_request, ProviderError
from test_candidate_pipeline import request_fixture


def request(name=None):
    legacy=request_fixture()
    translation={k:legacy[k] for k in ('fx_graph','public_constants','constant_origins','layout')}
    return make_request(translation,legacy['model'],cfg.PROFILE_SHA256,
                        compiler_configuration=cfg.configuration(name) if name else None)


def answer(r):
    return dict(schema=1,request_id=r['request_id'],
                hecate_source='@hc.func("c")\ndef golden(x):\n    return x * w + b\n')


class CompilerConfigurationTests(unittest.TestCase):
    def test_named_identity_binds_profile_pipeline_and_waterline(self):
        a,b=[cfg.configuration(n) for n in cfg.CONFIGURATIONS]
        self.assertEqual((a['waterline'],b['waterline']),(40,45))
        self.assertNotEqual(a['identity_sha256'],b['identity_sha256'])
        self.assertEqual(a['ckks_config_sha256'],b['ckks_config_sha256'])
        for name,value in (('waterline',50),('waterline',45.0),('schema',True),
                           ('pipeline','dacapo'),('backend','PoseidonGPU'),
                           ('ckks_config_sha256','0'*64),('identity_sha256','0'*64),('extra','data')):
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                cfg.validate_configuration(dict(b,**{name:value}),cfg.PROFILE_SHA256)
        with self.assertRaises(ValueError): cfg.configuration('seal-cpu-eva-w45-v1','0'*64)
        with self.assertRaises(ValueError): cfg.configuration('unregistered')

    def test_legacy_request_has_no_extension_and_retains_original_body_hash(self):
        r=request(); expected=request_fixture()
        expected['compiler_profile_sha256']=cfg.PROFILE_SHA256
        expected['request_id']=hashlib.sha256(canonical({k:v for k,v in expected.items() if k!='request_id'})).hexdigest()
        self.assertEqual(r,expected)
        self.assertNotIn('compiler_configuration',r)
        self.assertIsNone(cfg.request_configuration(r))
        self.assertIsNone(cfg.verify_execution_configuration(r,cfg.PROFILE_SHA256,40))
        with self.assertRaises(ValueError): cfg.verify_execution_configuration(r,cfg.PROFILE_SHA256,45)

    def test_configured_request_id_changes_and_provider_accepts_only_checked_extension(self):
        legacy,a,b=request(),request('seal-cpu-eva-w40-v1'),request('seal-cpu-eva-w45-v1')
        self.assertEqual(len({x['request_id'] for x in (legacy,a,b)}),3)
        for r in (legacy,a,b):
            self.assertEqual(public_request(r),r)
            validate_candidate(answer(r),r)
        changed=copy.deepcopy(b); changed['compiler_configuration']['waterline']=40
        with self.assertRaises(ProviderError): public_request(changed)
        with self.assertRaises(ValueError): validate_candidate(answer(changed),changed)
        changed=copy.deepcopy(b); changed['compiler_configuration']=a['compiler_configuration']
        with self.assertRaises(ProviderError): public_request(changed)
        with self.assertRaises(ValueError): validate_candidate(answer(a),b)
        for field in ('compiler_configuration','waterline','compiler_profile_sha256'):
            with self.assertRaises(ValueError): validate_candidate(dict(answer(b),**{field:45}),b)

    def test_approved_config_is_copied_and_actual_gate_must_match(self):
        r=request('seal-cpu-eva-w45-v1'); value=cfg.request_configuration(r)
        value['waterline']=40
        self.assertEqual(r['compiler_configuration']['waterline'],45)
        gate=dict(initial_level=13,arg_level=[13],arg_scale=[45])
        self.assertEqual(cfg.verify_artifact_configuration(r,gate,cfg.PROFILE_SHA256),r['compiler_configuration'])
        for g in (dict(gate,arg_scale=[40]),dict(gate,arg_level=[12]),dict(gate,arg_scale=[45,45]),
                  dict(gate,initial_level=12)):
            with self.assertRaises(ValueError): cfg.verify_artifact_configuration(r,g,cfg.PROFILE_SHA256)
        with self.assertRaises(ValueError): cfg.verify_artifact_configuration(r,gate,'0'*64)
        with self.assertRaises(ValueError): cfg.verify_execution_configuration(r,cfg.PROFILE_SHA256,40)
        with self.assertRaises(ValueError): cfg.verify_execution_configuration(r,cfg.PROFILE_SHA256,45.0)

    def test_candidate_cli_forwards_opt_in_without_default_change(self):
        from run_candidate import parse_args,forward_options
        legacy=parse_args(['--case','example.json','--prepare'])
        self.assertIsNone(legacy.compiler_configuration)
        self.assertNotIn('--compiler-configuration',shlex.split(forward_options(legacy)))
        for name in cfg.CONFIGURATIONS:
            args=parse_args(['--case','example.json','--prepare','--compiler-configuration',name])
            forwarded=parse_args(shlex.split(forward_options(args)))
            self.assertEqual(forwarded.compiler_configuration,name)

    def test_batch_plan_is_offline_and_continuation_cannot_change_configuration(self):
        from run_agent_batch import main,validate_construction_continuation
        for name in (None,*cfg.CONFIGURATIONS):
            argv=['run_agent_batch.py','--plan']+(['--compiler-configuration',name] if name else [])
            output=io.StringIO()
            with (patch('sys.argv',argv),
                  patch('agent_credentials.load_api_key',side_effect=AssertionError('credentials')),
                  contextlib.redirect_stdout(output)):
                self.assertEqual(main(),0)
            r=json.loads(output.getvalue())
            self.assertEqual(r['compiler_configuration'],cfg.configuration(name) if name else None)
            self.assertEqual(r['agent_calls'],0)
        a=SimpleNamespace(compiler_configuration='seal-cpu-eva-w45-v1',allow_config_change=True)
        validate_construction_continuation({'compiler_configuration':cfg.configuration(a.compiler_configuration)},a)
        for prior in ({},{'compiler_configuration':cfg.configuration('seal-cpu-eva-w40-v1')}):
            with self.assertRaisesRegex(ValueError,'fresh batch'): validate_construction_continuation(prior,a)


@unittest.skipUnless(os.environ.get('POSEIDON_LOOP_CONFIG45_GOLDENS'),'requires explicit45 native-loop batch')
class CompilerConfigurationEvidenceTests(unittest.TestCase):
    def test_new_requests_real_trace_compile_execute_and_old_failure_preserved(self):
        import numpy as np
        from hecate_python_env import digest
        from native_function_rules import LOOP_TASK
        from run_native_loop_goldens import CASES
        from seal_cpu_golden import compare,WATERLINE
        root=Path(os.environ['POSEIDON_LOOP_CONFIG45_GOLDENS'])
        batch=json.loads((root/'report.json').read_text())
        self.assertEqual((batch['status'],batch['generator'],batch['agent_calls']),('passed','manual_golden',0))
        self.assertEqual(len(batch['cases']),9)
        self.assertEqual(WATERLINE,40)
        for name,row in zip(CASES,batch['cases']):
            self.assertEqual(Path(row['command'][row['command'].index('--golden-file')+1]).stem,name)
            self.assertEqual(row['command'][row['command'].index('--compiler-configuration')+1],'seal-cpu-eva-w45-v1')
            self.assertTrue(row['matched_expected'])
            run=Path(row['run']);r=json.loads((run/'report.json').read_text())
            self.assertEqual((r['waterline'],r['agent_calls']),(45,0))
            self.assertEqual(r['compiler_configuration'],cfg.configuration('seal-cpu-eva-w45-v1'))
            self.assertFalse(r['llm_generation_validated'] or r['poseidon_gpu_validated'])
            self.assertEqual(r['tolerance'],dict(atol=1e-5,rtol=1e-4))
            self.assertEqual((r['parameters']['security_check'],r['parameters']['polynomial_degree']),('tc128',32768))
            self.assertEqual(r['parameters']['modulus_bits'],[60]*14)
            for file,value in r['frozen_hashes'].items(): self.assertEqual(digest(run/file),value)
            request=json.loads((run/'request.json').read_text())
            self.assertEqual(public_request(request),request)
            self.assertEqual(request['task'],LOOP_TASK)
            self.assertEqual(request['compiler_configuration'],r['compiler_configuration'])
            attempt=r['attempts'][0];out=run/'attempt-00/output'
            candidate=json.loads((run/'attempt-00/response.txt').read_text())
            self.assertEqual(canonical(validate_candidate(candidate,request)),canonical(attempt['static_check']))
            self.assertTrue(attempt['compiled'] and attempt['executed'])
            self.assertFalse(attempt['trace']['candidate_python_executed'])
            self.assertEqual(attempt['trace']['request_id'],request['request_id'])
            self.assertIn('--waterline=45',attempt['compile_command'])
            cfg.verify_artifact_configuration(request,attempt['artifact_gate'],cfg.PROFILE_SHA256)
            for file,value in attempt['artifact_hashes'].items(): self.assertEqual(digest(out/file),value)
            with np.load(run/'arrays.npz',allow_pickle=False) as data: inputs,reference=data['inputs'],data['reference']
            independent=sum(np.roll(inputs,-i,axis=-1) for i in range(4 if name=='sum4' else 3)) \
                if name in ('sum4','descending','wrong_bound') else inputs*1.5+.375
            np.testing.assert_allclose(reference,independent,atol=1e-15,rtol=0)
            actual=np.load(out/'decrypted.npy',allow_pickle=False)
            comparison=compare(actual,reference,1e-5,1e-4)
            self.assertEqual(comparison,attempt['comparison'])
            self.assertEqual(comparison['passed'],name!='wrong_bound')
            if name=='wrong_bound':
                self.assertEqual(attempt['failure_layer'],'numerical_comparison')
                self.assertGreater(comparison['max_absolute_error'],.9)
                self.assertTrue(compare(actual,sum(np.roll(inputs,-i,axis=-1) for i in range(4)),1e-5,1e-4)['passed'])
            execution=attempt['execution']
            self.assertTrue(execution['encrypted_execution']);self.assertFalse(execution['bootstrap_executed'])
            for values in execution['ciphertext_metadata']: self.assertEqual(values['input']['log2_scale'],45)
            self.assertFalse((run/'private-keys').exists())
            self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])
        old=Path(str(RESULTS / 'candidate-replay-t9aglvbt/report.json'))
        self.assertEqual(digest(old),'4ea525b5baa833cf40df36050a00b01f683a86c5a6e307fc28d3c5e05f1f00e0')
        self.assertFalse(json.loads(old.read_text())['attempts'][0]['comparison']['passed'])


@unittest.skipUnless(os.environ.get('POSEIDON_COMPILER_CONFIG_GUARD'),'requires real sandbox rejection probes')
class CompilerConfigurationGuardEvidenceTests(unittest.TestCase):
    def test_worker_rejected_before_keys_or_hevm_execution(self):
        from hecate_python_env import digest
        root=Path(os.environ['POSEIDON_COMPILER_CONFIG_GUARD'])
        r=json.loads((root/'report.json').read_text())
        self.assertEqual(r['status'],'passed')
        self.assertEqual((r['api_calls'],r['fhe_executions'],r['keys_mounted']),(0,0,False))
        self.assertEqual(digest(Path(r['source_run'])/'report.json'),r['source_report_sha256'])
        for name,value in r['source_hashes'].items(): self.assertEqual(digest(Path(name)),value)
        self.assertEqual([c['name'] for c in r['cases']],
                         ['different_valid_waterline','changed_configuration','changed_json_hash'])
        for case in r['cases']:
            self.assertTrue(case['matched_expected']);self.assertNotEqual(case['exit_code'],0)
            for name,value in case['hashes'].items(): self.assertEqual(digest(root/name),value)
            folder=root/case['name'];log=(folder/'worker.log').read_text()
            self.assertIn(case['expected_diagnostic'],log)
            self.assertIn('verify_artifact_configuration',log)
            self.assertNotIn('execute_artifact(',log)
            self.assertEqual({p.name for p in (folder/'output').iterdir()},
                             {'lowered._hecate_golden.hevm','_hecate_golden.cst'})


@unittest.skipUnless(os.environ.get('POSEIDON_CONFIG40_LEGACY_RUN') and os.environ.get('POSEIDON_CONFIG40_EXPLICIT_RUN'),
                     'requires fresh legacy40 and explicit40 real executions')
class CompilerLegacyCompatibilityEvidenceTests(unittest.TestCase):
    def test_legacy_request_unchanged_and_explicit40_has_distinct_identity(self):
        import numpy as np
        from hecate_python_env import digest
        from seal_cpu_golden import compare
        old=Path(str(RESULTS / 'candidate-replay-k61oyid5'))
        before=json.loads((old/'request.json').read_text())
        fresh=Path(os.environ['POSEIDON_CONFIG40_LEGACY_RUN'])
        explicit=Path(os.environ['POSEIDON_CONFIG40_EXPLICIT_RUN'])
        self.assertEqual(json.loads((fresh/'request.json').read_text()),before)
        updated=json.loads((explicit/'request.json').read_text())
        self.assertEqual(updated['compiler_configuration'],cfg.configuration('seal-cpu-eva-w40-v1'))
        self.assertNotEqual(updated['request_id'],before['request_id'])
        self.assertEqual({k:v for k,v in updated.items() if k not in ('compiler_configuration','request_id')},
                         {k:v for k,v in before.items() if k!='request_id'})
        for run in (fresh,explicit):
            r=json.loads((run/'report.json').read_text())
            self.assertEqual((r['status'],r['waterline'],r['agent_calls']),('passed',40,0))
            self.assertFalse(r['llm_generation_validated'] or r['poseidon_gpu_validated'])
            self.assertEqual(r.get('compiler_configuration'),cfg.configuration('seal-cpu-eva-w40-v1') if run==explicit else None)
            for file,value in r['frozen_hashes'].items(): self.assertEqual(digest(run/file),value)
            for file in ('weights.npz','arrays.npz'):
                with np.load(old/file,allow_pickle=False) as a,np.load(run/file,allow_pickle=False) as b:
                    self.assertEqual(set(a.files),set(b.files))
                    for key in a.files: np.testing.assert_array_equal(a[key],b[key])
            attempt=r['attempts'][0];out=run/'attempt-00/output'
            for file,value in attempt['artifact_hashes'].items(): self.assertEqual(digest(out/file),value)
            self.assertIn('--waterline=40',attempt['compile_command'])
            with np.load(run/'arrays.npz',allow_pickle=False) as data:
                inputs,reference=data['inputs'],data['reference']
            np.testing.assert_allclose(reference,inputs*1.5+.375,atol=1e-15,rtol=0)
            actual=np.load(out/'decrypted.npy',allow_pickle=False)
            self.assertEqual(compare(actual,reference,1e-5,1e-4),attempt['comparison'])
            self.assertTrue(attempt['comparison']['passed'] and attempt['execution']['encrypted_execution'])
            for item in attempt['execution']['ciphertext_metadata']: self.assertEqual(item['input']['log2_scale'],40)
            self.assertFalse((run/'private-keys').exists())
            self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])


if __name__=='__main__': unittest.main()
