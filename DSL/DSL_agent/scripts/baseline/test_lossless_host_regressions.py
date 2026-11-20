"""Offline host/wire boundary controls; no Agent, compiler, or FHE certification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import lossless_host_bound_delivery as host_bound
import first_output_planning as planning
from deepseek_provider import DeepSeekProvider, public_request
from generation_protocol import Config
from delivery_component import bound, canonical, delivery_plan, sha
from first_output_assessment import evaluation_policy, summarize_requests
from test_first_output_delivery import Transport


def constraints(ready=True):
    return bound(dict(format='poseidon-frozen-generation-backend-constraints-v1', evidence_sha256={},
        normal_Application_ready=ready, certified_refresh_profiles=[], test_only=True))


class Backend:
    def __init__(self, reject=False, drift=False):
        self.reject, self.drift, self.seen = reject, drift, []
    def identity_manifest(self):
        return dict(fixture=True, version=2 if self.drift and self.seen else 1)
    def check_request(self, request, level):
        self.seen.append(copy.deepcopy(request))
        host_bound._full_request(request)
        if self.reject: raise ValueError('fixture SDK blocked')


class HostBoundDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='fixture',
            inputs=[dict(name='input0',shape=[2])],constants=dict(w=[[.125,.25],[-.25,.5]],b=[.03125,-.0625]),
            nodes=[dict(id='n0',op='linear',inputs=['input0','w','b'],attrs={},outputs=['v0']),
                   dict(id='n1',op='square',inputs=['v0'],attrs={},outputs=['v1'])],
            outputs=[dict(name='output0',value='v1')])
        cls.authority=planning.source_authority()
        opts=dict(chunk_period=8,public_weight_policy='descriptors-compact-origins-v3')
        cls.full_template,cls.planning_template=planning.prepare_plan(model,constraints(),options=opts,test_only=True)
        _,cls.blocked_template=planning.prepare_plan(model,constraints(False),options=opts,test_only=True)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        # This cache is confined to fixture tests; no live admission is claimed.
        patch=mock.patch.object(planning,'source_authority',return_value=copy.deepcopy(self.authority))
        patch.start();self.addCleanup(patch.stop)
        self.full=copy.deepcopy(self.full_template);self.planning=copy.deepcopy(self.planning_template)
        self.backend=Backend();self.identity=dict(fixture=True,version=1)
        self.config=Config(max_calls=1,max_tokens=128)
        self.source='@hc.func("c,c")\r\ndef golden(x,zero_ct):\r\n    return x*x\r\n\r\n'
        self.host_path=self.root/'host.json';self.host_path.write_bytes(canonical(self.full))
        self.binding=host_bound.freeze_host_request(self.host_path)
        self.policy=evaluation_policy(['a'*64],domain_sha256='d'*64,
            approximation_requirements=[dict(id='max_absolute_error',comparison='<=',limit=0.)],application_requirements=[])
        self.output=self.root/'delivery';self.assessment=self.root/'assessment'
        self.plan=delivery_plan([dict(request=self.full,policy=self.policy,delivery_path=str(self.output),split='development')])
        self.freeze()

    def freeze(self):
        self.mechanism=host_bound.generator_manifest('host-wire-fixture-v1',[],host_bindings=[self.binding],plans=[self.planning],
            assessment_paths={self.full['request_id']:str(self.assessment)},backend_identity=self.identity,
            baseline_binding='0'*64,evaluation_policy_sha256=sha(canonical(self.plan)),provider_config=self.config)

    def provider(self,config=None):
        answer=json.dumps(dict(schema=1,request_id=self.full['request_id'],hecate_source=self.source))
        self.transport=Transport(answer)
        return DeepSeekProvider(self.config if config is None else config,transport=self.transport)

    def deliver(self,provider=None):
        return host_bound.generate_once(self.full,self.provider()if provider is None else provider,self.output,
            mechanism=self.mechanism,plan=self.plan,backend=self.backend,backend_identity=self.identity,test_only=True)

    def witness(self,source,test_only=True,wrong_math=False):
        digest=sha(source)
        return dict(source_sha256=digest,compiler_source_sha256=digest,compiled_validated=True,real_compiler=True,
            encrypted_execution=True,real_CKKS=True,test_only=test_only,intermediate_decryptions=0,
            reference_or_tolerance_changed=False,inputs=[dict(input_sha256='a'*64,source_sha256=digest,
                original_reference=[1.,2.],declared_reference=[1.,2.],
                dsl_mathematical_output=[1.01,2.]if wrong_math else[1.,2.],CKKS_real=[1.,2.],CKKS_imag=None,
                inside_frozen_domain=True,domain_sha256='d'*64,application_metrics={},whole_model_output=True)])

    def rewrite(self,path,value):
        path.chmod(0o600);path.write_bytes(canonical(value))

    def test_host_and_public_projection_have_distinct_content_and_one_id(self):
        wire=public_request(self.full)
        self.assertNotEqual(canonical(wire),canonical(self.full))
        self.assertEqual(wire['request_id'],self.full['request_id'])
        self.assertEqual(host_bound.resolve_host(self.binding,wire,mechanism=self.mechanism),self.full)

    def test_backend_receives_full_request_provider_receives_exact_public_context(self):
        receipt=self.deliver()
        self.assertEqual(self.backend.seen,[self.full])
        self.assertEqual(self.transport.calls[0]['messages'][1]['content'],canonical(public_request(self.full)).decode())
        self.assertEqual((self.output/'candidate.py').read_bytes(),self.source.encode())
        self.assertFalse(receipt['first_output_eligible'])
        self.assertEqual(host_bound.verify_host_delivery(self.output,receipt['binding'])[1],self.full)

    def test_mutating_host_values_stops_before_provider(self):
        altered=copy.deepcopy(self.full);altered['model']['constants']['w'][0][0]+=.01
        self.rewrite(self.host_path,altered);provider=self.provider()
        with self.assertRaises(ValueError):self.deliver(provider)
        self.assertEqual(provider.generation_attempts,0);self.assertFalse(self.transport.calls)

    def test_mutating_binding_stops_before_host_read(self):
        altered=copy.deepcopy(self.binding);altered['sha256']='f'*64
        with self.assertRaises(ValueError):host_bound.resolve_host(altered)

    def test_another_public_projection_cannot_use_this_host(self):
        wire=public_request(self.full);wire['request_id']='f'*64
        with self.assertRaises(ValueError):host_bound.resolve_host(self.binding,wire,mechanism=self.mechanism)

    def test_missing_host_asset_is_not_reconstructed_from_provider_context(self):
        self.host_path.unlink()
        with self.assertRaises((ValueError,OSError)):host_bound.resolve_host(self.binding)

    def test_host_asset_must_belong_to_frozen_generator(self):
        mechanism=copy.deepcopy(self.mechanism);mechanism['source_sha256'].pop(str(self.host_path))
        with self.assertRaises(ValueError):host_bound.resolve_host(self.binding,mechanism=mechanism)

    def test_missing_pre_provider_admission_invalidates_delivered_receipt(self):
        receipt=self.deliver();(host_bound._admission_path(self.output)/'admission.json').unlink()
        with self.assertRaises((ValueError,OSError)):host_bound.verify_host_delivery(self.output,receipt['binding'])

    def test_late_admission_cannot_relabel_existing_response(self):
        receipt=self.deliver();path=host_bound._admission_path(self.output)/'admission.json'
        admission=json.loads(path.read_bytes());admission.pop('binding');admission['recorded_at_ns']=10**30
        self.rewrite(path,bound(admission))
        with self.assertRaises(ValueError):host_bound.verify_host_delivery(self.output,receipt['binding'])

    def test_backend_rejection_consumes_no_generation_and_remains_in_denominator(self):
        self.backend=Backend(reject=True);provider=self.provider();receipt=self.deliver(provider)
        self.assertEqual(receipt['status'],'backend_blocked');self.assertEqual(provider.generation_attempts,0)
        result=host_bound.assess_once(self.output,receipt['binding'],self.plan,self.assessment,lambda *_:self.fail('Evaluator invoked'))
        cohort=bound(dict(format='poseidon-first-output-cohort-v1',mechanism_binding=self.mechanism['binding'],requests=[dict(
            request_id=self.full['request_id'],model_sha256=sha(canonical(self.full['model'])),input_sha256s=self.policy['inputs'],
            split='development',tier='medium',unseen=False)]))
        score=summarize_requests(cohort,{self.full['request_id']:result})['overall_descriptive_only']
        self.assertEqual(score['requests'],1);self.assertEqual(score['first_joint_rate'],0.)

    def test_unready_plan_needs_no_provider(self):
        self.planning=copy.deepcopy(self.blocked_template);self.freeze()
        receipt=host_bound.generate_once(self.full,None,self.output,mechanism=self.mechanism,plan=self.plan,test_only=True)
        self.assertEqual(receipt['status'],'backend_blocked')
        self.assertEqual(host_bound.verify_host_delivery(self.output,receipt['binding'])[2],None)

    def test_backend_drift_stops_before_provider(self):
        self.backend=Backend(drift=True);provider=self.provider()
        with self.assertRaises(ValueError):self.deliver(provider)
        self.assertFalse(self.transport.calls)

    def test_second_output_slot_attempt_cannot_select_a_repaired_version(self):
        self.deliver();provider=self.provider()
        with self.assertRaises((ValueError,OSError)):self.deliver(provider)
        self.assertFalse(self.transport.calls)

    def test_external_evaluator_gets_exact_host_and_source_once(self):
        receipt=self.deliver();seen=[]
        def evaluator(request,source):
            seen.append((request,source));return self.witness(source)
        result=host_bound.assess_once(self.output,receipt['binding'],self.plan,self.assessment,evaluator)
        self.assertEqual(seen,[(self.full,self.source.encode())]);self.assertTrue(result['raw_joint_passed'])
        self.assertFalse(result['first_joint_passed']);self.assertTrue(result['test_only'])
        with self.assertRaises((ValueError,OSError)):host_bound.assess_once(self.output,receipt['binding'],self.plan,self.assessment,evaluator)
        with self.assertRaises(ValueError):host_bound.assess_once(self.output,receipt['binding'],self.plan,self.root/'another',evaluator)
        self.assertEqual(len(seen),1)

    def test_changed_source_does_not_reach_external_evaluator(self):
        receipt=self.deliver();self.rewrite(self.output/'candidate.py',dict(replacement=True))
        with self.assertRaises(ValueError):host_bound.assess_once(self.output,receipt['binding'],self.plan,self.assessment,lambda *_:self.fail('Evaluator invoked'))

    def test_host_change_after_delivery_does_not_reach_external_evaluator(self):
        receipt=self.deliver();self.rewrite(self.host_path,dict(replacement=True))
        with self.assertRaises(ValueError):host_bound.assess_once(self.output,receipt['binding'],self.plan,self.assessment,lambda *_:self.fail('Evaluator invoked'))

    def test_json_claims_without_native_proof_never_count_as_first_success(self):
        receipt=self.deliver()
        result=host_bound.assess_once(self.output,receipt['binding'],self.plan,self.assessment,
            lambda _,source:self.witness(source,test_only=False))
        self.assertEqual(result['status'],'backend_blocked');self.assertFalse(result['raw_joint_passed'])
        self.assertFalse(result['first_joint_passed']);self.assertEqual(result['compared_values'],0)

    def test_math_error_cannot_be_replaced_by_matching_ckks_cells(self):
        receipt=self.deliver()
        result=host_bound.assess_once(self.output,receipt['binding'],self.plan,self.assessment,
            lambda _,source:self.witness(source,wrong_math=True))
        self.assertEqual(result['status'],'numeric_failed');self.assertFalse(result['raw_joint_passed'])

    def test_more_than_one_provider_generation_is_rejected(self):
        provider=self.provider(Config(max_calls=2,max_tokens=128))
        with self.assertRaises(ValueError):self.deliver(provider)
        self.assertEqual(provider.generation_attempts,0);self.assertFalse(self.transport.calls)

    def test_existing_large_host_bound_is_separate_from_response_limit(self):
        path=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-spatial-channel-normalization-controls-r2703/M02/prepared-request.json')
        self.assertGreater(path.stat().st_size,128*1024)
        self.assertLess(path.stat().st_size,4*1024**2)
        binding=host_bound.freeze_host_request(path)
        host=host_bound.resolve_host(binding)
        self.assertEqual(host['request_id'],'29763d28cccb57448aaea16470b0a9fc0d68afb601988dca5bedc695a57a746a')
        self.assertNotEqual(binding['host_canonical_sha256'],binding['public_canonical_sha256'])

    def test_fixture_plan_cannot_authorize_a_live_delivery(self):
        provider=self.provider()
        with self.assertRaises(ValueError):host_bound.generate_once(self.full,provider,self.output,
            mechanism=self.mechanism,plan=self.plan,backend=self.backend,backend_identity=self.identity,test_only=False)
        self.assertFalse(self.transport.calls)


if __name__=='__main__':unittest.main()
