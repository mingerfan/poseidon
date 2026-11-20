"""Offline planning/admission controls. Fixture readiness is never live authority."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmark_suite import Builder
from delivery_component import bound, canonical, delivery_plan, sha, verify_delivery
from deepseek_provider import DeepSeekProvider
from generation_protocol import Config
from first_output_assessment import assess_once, evaluation_policy, summarize_requests
import first_output_planning as planning
import planned_delivery_component as planned
from test_first_output_delivery import Transport


def graph(name='planning_fixture'):
    builder = Builder([(2,)])
    model = builder.finish(builder.node('square', ['input0']))
    model['id'] = name
    return model


def fixture_constraints(ready=True):
    return bound(dict(format='poseidon-frozen-generation-backend-constraints-v1', evidence_sha256={},
        normal_Application_ready=ready, certified_refresh_profiles=[], test_only=True))


class Backend:
    def __init__(self, reject=False, drift=False): self.reject, self.drift, self.checks = reject, drift, 0
    def identity_manifest(self): return dict(fixture=True, version=2 if self.drift and self.checks else 1)
    def check_request(self, request, level):
        self.checks += 1
        if self.reject: raise ValueError('SDK not ready')


class PreGenerationPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.authority = planning.source_authority()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = mock.patch.object(planning, 'source_authority', return_value=copy.deepcopy(self.authority))
        self.cache.start(); self.addCleanup(self.cache.stop)
        self.request, self.planning = planning.prepare_plan(graph(), fixture_constraints(), test_only=True)
        self.policy = evaluation_policy(['a'*64], domain_sha256='d'*64,
            approximation_requirements=[], application_requirements=[])

    def freeze(self, planning_record=None, request=None, backend=None):
        self.request = self.request if request is None else request
        self.planning = self.planning if planning_record is None else planning_record
        self.backend = Backend() if backend is None else backend
        self.slots = delivery_plan([dict(request=self.request, policy=self.policy,
            delivery_path=str(self.root/'delivery'), split='regression')])
        self.mechanism = planned.generator_manifest('pre-generation-fixture-v1', [], plans=[self.planning],
            backend_identity=dict(fixture=True, version=1), baseline_binding='0'*64,
            evaluation_policy_sha256=sha(canonical(self.slots)), provider_config=Config(max_calls=1,max_tokens=128))
        return self.mechanism

    def provider(self):
        answer=json.dumps(dict(schema=1,request_id=self.request['request_id'],
            hecate_source='@hc.func("c,c")\r\ndef golden(x,zero_ct):\r\n    return x*x\r\n'))
        self.transport=Transport(answer)
        return DeepSeekProvider(Config(max_calls=1,max_tokens=128),transport=self.transport)

    def deliver(self, provider=None, **kwargs):
        return planned.generate_once(self.request,provider,self.root/'delivery',mechanism=self.mechanism,
            plan=self.slots,backend=self.backend,backend_identity=dict(fixture=True,version=1),test_only=True,**kwargs)

    def test_registered_generic_cost_selects_a_bound_request(self):
        planning.verify_plan(self.planning,self.request)
        self.assertEqual(self.planning['selection']['score'], min(r['score'] for r in self.planning['candidates'] if r['status']=='admitted'))
        self.assertEqual(self.planning['Provider_calls'],0)

    def test_renaming_model_does_not_select_a_different_route(self):
        other, plan = planning.prepare_plan(graph('resnet20_not_a_route_key'),fixture_constraints(),test_only=True)
        self.assertEqual(plan['selection']['options'],self.planning['selection']['options'])
        self.assertEqual(plan['selection']['score'],self.planning['selection']['score'])
        self.assertEqual(plan['features'],self.planning['features'])
        self.assertNotEqual(other['request_id'],self.request['request_id'])

    def test_explicit_period_is_preserved(self):
        request, plan = planning.prepare_plan(graph(),fixture_constraints(),options=dict(chunk_period=8),test_only=True)
        self.assertEqual(request['layout']['input_slot_period'],8)
        self.assertEqual(len(plan['candidate_options']),1)

    def test_logical_depth_never_becomes_a_CKKS_level_or_refresh_estimate(self):
        self.assertEqual(self.planning['features']['cipher_multiply_path']['value'],1)
        for key in ('CKKS_level_consumption','refresh_schedule','actual_runtime_peak_bytes'):
            self.assertIsNone(self.planning[key])

    def test_fixture_capability_cannot_authorize_a_live_plan(self):
        with self.assertRaises(ValueError): planning.prepare_plan(graph(),fixture_constraints())

    def test_native_refresh_requirement_is_blocked_without_numeric_certificate(self):
        request, plan = planning.prepare_plan(graph(),fixture_constraints(),
            options=dict(compiler_configuration='poseidon-cpu-eva-w44-native-epoch-v1'),test_only=True)
        self.assertGreater(plan['required_refresh_calls_per_process'],0)
        self.assertEqual(plan['status'],'backend_blocked')
        self.assertIn('required_refresh_has_no_numeric_certificate',plan['failure_reasons'])

    def test_memory_ceiling_is_unchanged_and_payload_lower_bound_can_reject(self):
        with self.assertRaises(ValueError): planning.prepare_plan(graph(),fixture_constraints(),memory_budget_bytes=4*1024**3+1,test_only=True)
        _, plan = planning.prepare_plan(graph(),fixture_constraints(),memory_budget_bytes=1,test_only=True)
        self.assertEqual(plan['status'],'resource_blocked')

    def test_blocked_request_needs_no_Provider_or_loader_and_remains_a_denominator(self):
        request, plan = planning.prepare_plan(graph(),fixture_constraints(False),test_only=True)
        self.freeze(plan,request)
        receipt=self.deliver()
        self.assertEqual(receipt['status'],'backend_blocked')
        self.assertEqual(verify_delivery(self.root/'delivery',receipt['binding'])[2],None)
        self.assertEqual(json.loads((self.root/'delivery/attempt.json').read_bytes())['generations'],0)
        result=assess_once(self.root/'delivery',receipt['binding'],self.slots,self.root/'assessment',lambda *_:self.fail('Evaluator must not run'))
        cohort=bound(dict(format='poseidon-first-output-cohort-v1',mechanism_binding=self.mechanism['binding'],requests=[dict(request_id=self.request['request_id'],model_sha256=sha(canonical(self.request['model'])),input_sha256s=self.policy['inputs'],split='regression',tier='operators',unseen=False)]))
        score=summarize_requests(cohort,{self.request['request_id']:result})['overall_descriptive_only']
        self.assertEqual(score['requests'],1);self.assertEqual(score['first_joint_rate'],0.)
        self.assertEqual(score['statuses']['backend_blocked'],1)

    def test_backend_preflight_rejection_consumes_no_generation(self):
        self.freeze(backend=Backend(reject=True));provider=self.provider();receipt=self.deliver(provider)
        self.assertEqual(receipt['status'],'backend_blocked')
        self.assertEqual(provider.generation_attempts,0);self.assertFalse(self.transport.calls)

    def test_backend_identity_change_is_detected_before_Provider(self):
        self.freeze(backend=Backend(drift=True));provider=self.provider()
        with self.assertRaises(ValueError):self.deliver(provider)
        self.assertFalse(self.transport.calls)

    def test_single_fixture_output_is_unchanged_and_preceded_by_admission(self):
        self.freeze();provider=self.provider();receipt=self.deliver(provider)
        self.assertEqual(provider.generation_attempts,1)
        self.assertEqual(len(self.transport.calls[0]['messages']),2)
        self.assertEqual(json.loads(self.transport.calls[0]['messages'][1]['content']),self.request)
        admission=planned.verify_planned_delivery(self.root/'delivery',receipt,self.mechanism)
        self.assertEqual(admission['Provider_calls'],0)
        self.assertFalse(receipt['first_output_eligible'])
        self.assertEqual((self.root/'delivery/candidate.py').read_bytes(),b'@hc.func("c,c")\r\ndef golden(x,zero_ct):\r\n    return x*x\r\n')

    def test_original_delivery_entry_cannot_bypass_current_assessment_admission(self):
        import delivery_component
        self.freeze();provider=self.provider()
        receipt=delivery_component.generate_once(self.request,provider,self.root/'delivery',mechanism=self.mechanism,plan=self.slots)
        result=assess_once(self.root/'delivery',receipt['binding'],self.slots,self.root/'assessment',lambda *_:self.fail('Missing admission must stop evaluator'))
        self.assertEqual(result['status'],'evaluation_failed')
        self.assertFalse(result['first_compile_passed']);self.assertFalse(result['first_joint_passed'])

    def test_changed_frozen_layout_or_status_is_rejected(self):
        for key,value in [('request_id','0'*64),('status','passed'),('CKKS_level_consumption',1)]:
            record=copy.deepcopy(self.planning);record.pop('binding');record[key]=value
            with self.assertRaises(ValueError):planning.verify_plan(bound(record),self.request)

    def test_changed_backend_evidence_after_freeze_is_rejected(self):
        path=self.root/'capability-fixture.json';path.write_bytes(b'{}')
        c=fixture_constraints();c.pop('binding');c['evidence_sha256']={str(path):sha(path.read_bytes())};c=bound(c)
        request,record=planning.prepare_plan(graph(),c,test_only=True)
        path.write_bytes(b'{"changed":true}')
        with self.assertRaises(ValueError):planning.verify_plan(record,request)

    def test_model_shape_changes_are_rejected_before_generation(self):
        request=copy.deepcopy(self.request);request['model']['inputs'][0]['shape']=[1,2]
        with self.assertRaises(ValueError):planning.verify_plan(self.planning,request)

    def test_changed_capability_digest_cannot_be_substituted_in_generator(self):
        with self.assertRaises(ValueError):
            planned.generator_manifest('bad-caps',[],plans=[self.planning],baseline_binding='0'*64,
                evaluation_policy_sha256='1'*64,capabilities_sha256='2'*64)

    def test_blocked_output_cannot_be_replaced_by_a_later_attempt(self):
        request,record=planning.prepare_plan(graph(),fixture_constraints(False),test_only=True);self.freeze(record,request);self.deliver()
        with self.assertRaises(FileExistsError):self.deliver()

    def test_invalid_candidate_and_wall_limits_are_rejected(self):
        for kwargs in [dict(candidate_limit=9),dict(wall_seconds=121)]:
            with self.assertRaises(ValueError):planning.prepare_plan(graph(),fixture_constraints(),test_only=True,**kwargs)

    def test_mutated_admission_cannot_be_used_for_current_assessment(self):
        self.freeze();receipt=self.deliver(self.provider())
        p=self.root/'delivery.admission/admission.json';d=json.loads(p.read_bytes());d.pop('binding');d['planning_binding']='0'*64;p.chmod(0o600);p.write_bytes(canonical(bound(d)))
        with self.assertRaises(ValueError):planned.verify_planned_delivery(self.root/'delivery',receipt,self.mechanism)

    def test_v5_metadata_completion_never_opens_numeric_or_normal_gate(self):
        sdk=bound(dict(format='poseidon-runtime-toolchain-native-v5',native=dict(readiness=dict(normal_Application_profile=True,compiler_and_origin_attestation=True,standalone_native_runtime_measured=True),numeric_policy='metrics-only'),artifacts=dict(poseidon_library='a'*64)))
        path=self.root/'sdk-fixture.json';path.write_bytes(canonical(sdk))
        constraints=planning.backend_constraints(path,sdk_sha256=sha(path.read_bytes()))
        self.assertFalse(constraints['normal_Application_ready'])
        self.assertEqual(constraints['certified_refresh_profiles'],[])

    def test_original_SDK_string_library_digest_is_bound_to_observed_failure(self):
        library=self.root/'libposeidon_fixture.so';library.write_bytes(b'inert fixture; never executed')
        library_hash=sha(library.read_bytes())
        sdk=bound(dict(format='poseidon-runtime-toolchain-native-v5',native=dict(readiness=dict(normal_Application_profile=False,compiler_and_origin_attestation=True,standalone_native_runtime_measured=True),numeric_policy='metrics-only'),artifacts=dict(poseidon_library=library_hash),library_overlay=dict(candidate=dict(path=str(library),sha256=library_hash))))
        path=self.root/'sdk-path-fixture.json';path.write_bytes(canonical(sdk))
        negative=bound(dict(profile=dict(numeric_gate='closed',numerically_certified_refresh_profiles=0,library_sha256=sha(library.read_bytes()))))
        boundary=self.root/'negative-boundary-fixture.json';boundary.write_bytes(canonical(negative))
        constraints=planning.backend_constraints(path,sdk_sha256=sha(path.read_bytes()),negative_boundary=dict(path=str(boundary),sha256=sha(boundary.read_bytes())))
        planning.verify_constraints(constraints)
        library.write_bytes(b'changed fixture')
        with self.assertRaises(ValueError):planning.verify_constraints(constraints)


if __name__=='__main__':unittest.main()
