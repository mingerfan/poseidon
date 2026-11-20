"""Twelve bounded controls; fixtures are never Agent/compiler/CKKS success."""
import copy,json,os
from pathlib import Path
import unittest

from delivery_component import canonical,bound,sha,read_bytes,verify_delivery
import compiler_configuration as cfg
import poseidon_runtime_compile as compile_commands
import compiler_managed_evaluation as evaluation
import compiler_managed_delivery as managed
from benchmark_suite import Builder
from generation_protocol import Config
from deepseek_provider import DeepSeekProvider,system_prompt,SYSTEM,COMPILER_MANAGED_SYSTEM

R=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results')


class Transport:
    is_live=False
    def __init__(self,answer):self.answer=answer;self.calls=[]
    def post(self,body,timeout):
        self.calls.append(json.loads(body))
        return 200,canonical(dict(object='chat.completion',model='deepseek-flash',
            choices=[dict(finish_reason='stop',message=dict(role='assistant',content=self.answer))],
            usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)))


class CompilerManagedControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        admission=json.loads((R/'agent-normal-cpu-host-proof-publication-r2692/NORMAL_CPU_TEST_ADMISSION.json').read_bytes())
        cls.profile=evaluation.freeze_profile(
            evaluation.descriptor(R/'agent-normal-cpu-approved-overlay-staging-r2680/sdk/ordinary-cpu-selector.json'),
            admission['normal_CPU_canary'],
            evaluation.descriptor(R/'agent-native-compiler-help-snapshot-publication-r2854/ACTUAL_HELP_SNAPSHOT_EVIDENCE.json'),
            evaluation.descriptor(R/'agent-native-fast-compiler-observer-r2851/run-r2853/help.log'))
        b=Builder([(2,)])
        cls.model=b.finish(b.node('square',['input0']))
        cls.request=managed.prepare(cls.model)
        cls.root=Path(os.environ['POSEIDON_COMPILER_CONTROL_OUTPUT'])

    def test_01_new_pipeline_preserves_all_security_parameters(self):
        old=cfg.configuration(cfg.CURRENT_CONFIGURATION);new=cfg.configuration(cfg.COMPILER_MANAGED_CONFIGURATION)
        for key in ['waterline','ckks_config_sha256','operator_spec_sha256','source_lock_sha256','parameter_identity']:
            self.assertEqual(new[key],old[key])
        self.assertEqual(new['pipeline'],'dacapo-runtime-plan')
        self.assertEqual(new['ciphertext_state_owner'],'compiler')
        self.assertNotEqual(new['identity_sha256'],old['identity_sha256'])

    def test_02_changed_policy_or_strategy_refused(self):
        value=cfg.configuration(cfg.COMPILER_MANAGED_CONFIGURATION)
        for key,new in [('bootstrap_placement','greedy'),('ciphertext_state_owner','Agent'),('waterline',60),('schema',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                cfg.validate_configuration(dict(value,**{key:new}),value['ckks_config_sha256'])

    def test_03_legacy_configuration_and_command_bytes_preserved(self):
        root=R/'candidate-replay-h5iolm7l'
        request=json.loads((root/'request.json').read_bytes())
        report=json.loads((root/'report.json').read_bytes())
        self.assertEqual(cfg.configuration(cfg.CURRENT_CONFIGURATION),request['compiler_configuration'])
        self.assertEqual(compile_commands.command_for_configuration(request['compiler_configuration']),report['attempts'][0]['compile_command'])

    def test_04_compiler_owns_automatic_pipeline_selection(self):
        command=self.profile['command']
        self.assertEqual(command.count('--dacapo'),1)
        self.assertNotIn('--eva',command)
        self.assertIn('--boot-placement=optimized',command)
        self.assertIn('--runtime-plan-boot-implementation=native',command)
        self.assertNotIn('decrypt_reencrypt',' '.join(command))
        with self.assertRaises(ValueError):compile_commands.command(state_management_policy='Agent-position-list')

    def test_05_new_schema_retains_original_entry_body(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=self.root)as tmp:
            output=Path(tmp)
            source='"builtin.module"() ({\n  "func.func"() <{sym_name = "_hecate_golden"}>\n  "func.func"() <{sym_name = "helper"}>\n})\n'
            (output/'candidate_trace.mlir').write_text(source)
            receipt=compile_commands.prepare_entrypoint(output,self.profile['configuration'])
            selected=(output/'candidate_trace.entry.mlir').read_text()
            self.assertEqual(selected.replace(', sym_visibility = "private"',''),source)
            compile_commands.audit_entrypoint(output,self.request,receipt,self.profile['command'])

    def test_06_default_computational_request_has_no_Agent_refresh_schedule(self):
        original=copy.deepcopy(self.model)
        request=managed.prepare(self.model)
        self.assertEqual(self.model,original)
        self.assertEqual(request['compiler_configuration'],self.profile['configuration'])
        self.assertNotIn('refresh_schedule',request)
        self.assertNotIn('native_Boot_invocations_per_process',request['compiler_configuration'])
        evaluation.backend(self.profile).check_request(request,'compiled')

    def test_07_new_prompt_delegates_all_state_operations(self):
        text=system_prompt(self.request)
        self.assertEqual(text,COMPILER_MANAGED_SYSTEM)
        for token in ['rescale','modswitch','bootstrap insertion','ciphertext state','do not choose refresh locations']:
            self.assertIn(token,text)
        self.assertIn('No compiler, execution, numerical feedback or post-output repair',text)

    def test_08_historical_prompt_and_bodies_keep_legacy_selection(self):
        request=json.loads((R/'candidate-replay-h5iolm7l/request.json').read_bytes())
        self.assertEqual(system_prompt(request),SYSTEM)
        self.assertIn('No tools, network, filesystem operations or bootstrap',SYSTEM)

    def test_09_interface_configuration_does_not_claim_numeric_bootstrap(self):
        evaluation.verify_profile(self.profile)
        self.assertTrue(self.profile['compiler_interface_observed'])
        self.assertFalse(self.profile['bootstrap_numerically_certified'])
        self.assertEqual(self.profile['mathematical_tolerance'],dict(atol=1e-12,rtol=1e-12))
        self.assertEqual(self.profile['CKKS_tolerance'],dict(atol=1e-5,rtol=1e-4))
        bad=dict(self.profile);bad.pop('binding');bad['bootstrap_numerically_certified']=True
        with self.assertRaises(ValueError):evaluation.verify_profile(bound(bad))

    def mechanism(self):
        return managed.generator_manifest('compiler-owned-controls-only-v1',[],profile=self.profile,
            baseline_binding='0'*64,capabilities_sha256='1'*64,evaluation_policy_sha256='2'*64,
            provider_config=Config(max_calls=1,max_tokens=128))

    def test_10_new_generation_does_not_wait_for_numeric_refresh_certificate(self):
        mechanism=self.mechanism()
        self.assertFalse(mechanism['compiler_evaluation_profile']['bootstrap_numerically_certified'])
        self.assertEqual(mechanism['compiler_evaluation_profile'],self.profile)

    def test_11_single_fixture_answer_is_immediately_frozen_unchanged(self):
        source='@hc.func("c,c")\r\ndef golden(x, zero_ct):\r\n    return x*x\r\n\r\n'
        transport=Transport(json.dumps(dict(schema=1,request_id=self.request['request_id'],hecate_source=source)))
        provider=DeepSeekProvider(Config(max_calls=1,max_tokens=128),transport=transport)
        output=self.root/'fixture-delivery'
        receipt=managed.generate_once(self.request,provider,output,profile=self.profile,mechanism=self.mechanism())
        self.assertEqual(read_bytes(output/'candidate.py'),source.encode())
        self.assertEqual(receipt['source_sha256'],sha(source.encode()))
        self.assertFalse(receipt['first_output_eligible'])
        self.assertFalse(receipt['external_evaluation_performed'])
        self.assertEqual(len(transport.calls),1)
        self.assertEqual(transport.calls[0]['messages'][0]['content'],COMPILER_MANAGED_SYSTEM)
        verify_delivery(output,receipt['binding'])

    def test_12_failure_without_dependency_evidence_is_not_backend_blocked(self):
        result=evaluation.classify_dependency(dict(attempts=[dict(checked=True,executed=True,failure_layer='numerical_comparison')]),
            self.root,self.profile)
        self.assertEqual(result['status'],'validation_failed')
        self.assertFalse(result['dependency_confirmed'])
        original=bound(dict(status='backend_blocked',failure_category='backend_blocked',
                            raw_joint_passed=False,first_joint_passed=False))
        current=evaluation._dependency_outcome(original,{})
        self.assertEqual(current['status'],'evaluation_failed')
        self.assertEqual(original['status'],'backend_blocked')
        self.assertFalse(current['dependency_failure_confirmed'])
        self.assertFalse(current['first_joint_passed'])


if __name__=='__main__':unittest.main()
