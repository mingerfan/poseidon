"""Boundary/metric controls with offline fixtures; never real Agent or FHE evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from benchmark_suite import Builder
from component_contract import prepare_task
from deepseek_provider import DeepSeekProvider
from generation_protocol import Config
from delivery_component import (bound, canonical, capture_response, delivery_plan, generate_once,
    generator_manifest, read_bytes, sha, verify_delivery)
from first_output_assessment import (CKKS, assess_once, evaluation_policy,
    evaluate_witness, summarize_requests)
from development_assessment import run_development


class Transport:
    is_live = False
    def __init__(self, content, *, refused=False):
        self.content = content
        self.calls = []
        self.refused = refused
    def post(self, body, timeout):
        self.calls.append(json.loads(body))
        return 200, canonical(dict(object='chat.completion',model='deepseek-flash',
            choices=[dict(finish_reason='stop',message=dict(role='assistant',content=self.content,refusal='no' if self.refused else None))],
            usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2)))


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        b = Builder([(2,)])
        self.request = prepare_task(b.finish(b.node('square', ['input0'])))
        self.source = '@hc.func("c,c")\r\ndef golden(x, zero_ct):\r\n    return x*x\r\n\r\n'
        self.policy = evaluation_policy(['a'*64, 'b'*64, 'c'*64],domain_sha256='d'*64,
            approximation_requirements=[dict(id='max_absolute_error',comparison='<=',limit=.01)],
            application_requirements=[dict(id='accuracy',comparison='>=',limit=.9)],whole_model_required=True)
        self.mechanism = generator_manifest('boundary-controls-only-v1', [], baseline_binding='0'*64,
            capabilities_sha256='1'*64, evaluation_policy_sha256=sha(canonical(self.policy)),
            provider_config=Config(max_calls=1,max_tokens=128))
    def tearDown(self):
        self.tmp.cleanup()
    def answer(self, source=None):
        return json.dumps(dict(schema=1,request_id=self.request['request_id'],hecate_source=self.source if source is None else source))
    def deliver(self, raw=None, *, name='delivery', **config):
        self.transport = Transport(self.answer() if raw is None else raw)
        self.provider = DeepSeekProvider(Config(max_calls=1, max_tokens=128, **config),transport=self.transport)
        return generate_once(self.request,self.provider,self.root/name,mechanism=self.mechanism)
    def witness(self, source=None):
        source = self.source.encode() if source is None else source
        digest = sha(source)
        return dict(source_sha256=digest, compiler_source_sha256=digest,compiled_validated=True,
            real_compiler=True,encrypted_execution=True,real_CKKS=True,test_only=True,
            intermediate_decryptions=0,reference_or_tolerance_changed=False,
            inputs=[dict(input_sha256=h,source_sha256=digest,original_reference=[1.,4.],
                declared_reference=[1.,4.],dsl_mathematical_output=[1.,4.],CKKS_real=[1.000001,4.000001],
                inside_frozen_domain=True,domain_sha256='d'*64,whole_model_output=True,
                application_metrics=dict(accuracy=.95)) for h in self.policy['inputs']])
    def assess(self, witness, *, delivery=None, name='assessment'):
        receipt = self.deliver() if delivery is None else delivery
        return assess_once(self.root/'delivery',receipt['binding'],self.policy,self.root/name,
                           lambda request, source: witness)

    def test_exact_CRLF_and_trailing_lines_are_frozen(self):
        receipt = self.deliver()
        self.assertEqual(read_bytes(self.root/'delivery/candidate.py'),self.source.encode())
        self.assertEqual(receipt['source_sha256'],sha(self.source.encode()))
        self.assertEqual(receipt['status'],'dsl_delivered')
        self.assertFalse(receipt['external_evaluation_performed'])
        self.assertFalse(receipt['first_output_eligible'])
        self.assertEqual(len(self.transport.calls),1)
        self.assertEqual(len(self.transport.calls[0]['messages']),2)
        self.assertEqual(verify_delivery(self.root/'delivery',receipt['binding'])[2],self.source.encode())

    def test_invalid_DSL_is_preserved_before_any_static_check(self):
        receipt = self.deliver(self.answer('not valid python {{{\n'))
        self.assertEqual(receipt['status'],'dsl_delivered')
        self.assertEqual(read_bytes(self.root/'delivery/candidate.py'),b'not valid python {{{\n')

    def test_contract_rejection_has_retained_raw_answer_and_no_repair(self):
        for i,raw in enumerate(['{}',self.answer(''),'{"schema":1,"schema":1}',self.answer().replace(self.request['request_id'],'0'*64)]):
            with self.subTest(raw=raw):
                receipt = self.deliver(raw,name='bad-'+str(i))
                self.assertEqual(receipt['status'],'contract_rejected')
                self.assertEqual(read_bytes(self.root/('bad-'+str(i))/'response.txt'),raw.encode())
                self.assertEqual(self.provider.generation_attempts,1)

    def test_provider_refusal_and_consumed_attempt_are_retained(self):
        transport=Transport('unused',refused=True)
        provider=DeepSeekProvider(Config(max_calls=1,max_tokens=128),transport=transport)
        receipt=generate_once(self.request,provider,self.root/'refusal',mechanism=self.mechanism)
        self.assertEqual(receipt['status'],'refused')
        self.assertEqual(provider.request_attempts,1)
        self.assertEqual(len(transport.calls),1)

    def test_existing_attempt_cannot_be_replaced(self):
        self.deliver()
        with self.assertRaises(FileExistsError):self.deliver()
        self.assertEqual(self.transport.calls,[])

    def test_retry_or_multi_generation_config_rejected_before_calls(self):
        for config in [Config(max_calls=2),Config(max_calls=1,provider_retries=1)]:
            transport=Transport(self.answer());provider=DeepSeekProvider(config,transport=transport)
            with self.assertRaises(ValueError):generate_once(self.request,provider,self.root/'not-created',mechanism=self.mechanism)
            self.assertFalse(transport.calls)

    def test_no_evaluator_or_repair_callback_in_delivery_API(self):
        provider=DeepSeekProvider(Config(max_calls=1),transport=Transport(self.answer()))
        with self.assertRaises(TypeError):
            generate_once(self.request,provider,self.root/'bad',mechanism=self.mechanism,evaluator=lambda *_:True)
        self.assertEqual(provider.generation_attempts,0)

    def test_fake_live_transport_cannot_be_counted_as_Agent(self):
        transport=Transport(self.answer());transport.is_live=True
        provider=DeepSeekProvider(Config(max_calls=1),transport=transport)
        with self.assertRaises(ValueError):generate_once(self.request,provider,self.root/'bad',mechanism=self.mechanism)
        self.assertFalse(transport.calls)

    def test_retrospective_capture_never_creates_first_output_eligibility(self):
        receipt=capture_response(self.request,self.answer(),self.root/'retro',mechanism=self.mechanism)
        self.assertEqual(receipt['mode'],'retrospective_capture')
        self.assertFalse(receipt['first_output_eligible'])

    def test_tampered_source_or_receipt_is_rejected(self):
        receipt=self.deliver();p=self.root/'delivery/candidate.py';p.chmod(0o600);p.write_bytes(b'replacement\n')
        with self.assertRaises(ValueError):verify_delivery(p.parent,receipt['binding'])
        changed=copy.deepcopy(receipt);changed['source_sha256']=sha(b'replacement\n');changed=bound({k:v for k,v in changed.items() if k!='binding'})
        rp=p.parent/'delivery.json';rp.chmod(0o600);rp.write_bytes(canonical(changed))
        with self.assertRaises(ValueError):verify_delivery(p.parent,receipt['binding'])

    def test_external_evaluator_receives_exact_bytes_once(self):
        receipt=self.deliver();seen=[]
        def evaluator(request,source):
            seen.append(source);return self.witness(source)
        result=assess_once(self.root/'delivery',receipt['binding'],self.policy,self.root/'assessment',evaluator)
        self.assertEqual(seen,[self.source.encode()]);self.assertTrue(result['raw_joint_passed'])
        self.assertFalse(result['first_joint_passed']);self.assertTrue(result['test_only'])
        self.assertEqual(result['input_groups_evaluated'],3)
        with self.assertRaises(FileExistsError):
            assess_once(self.root/'delivery',receipt['binding'],self.policy,self.root/'assessment',evaluator)
        self.assertEqual(len(seen),1)

    def test_post_evaluation_source_mutation_invalidates_result(self):
        receipt=self.deliver()
        def evaluator(request,source):
            p=self.root/'delivery/candidate.py';p.chmod(0o600);p.write_bytes(b'fixed')
            return self.witness(source)
        result=assess_once(self.root/'delivery',receipt['binding'],self.policy,self.root/'assessment',evaluator)
        self.assertEqual(result['status'],'integrity_rejected');self.assertFalse(result['source_unchanged'])

    def test_evaluator_failure_never_reenters_generation(self):
        receipt=self.deliver();calls=[]
        def failed(*_):calls.append(1);raise RuntimeError('compiler failed')
        result=assess_once(self.root/'delivery',receipt['binding'],self.policy,self.root/'assessment',failed)
        self.assertEqual(result['status'],'evaluation_failed');self.assertEqual(len(calls),1)
        self.assertEqual(self.provider.generation_attempts,1)

    def test_compile_pass_does_not_replace_CKKS_failure(self):
        witness=self.witness();witness['inputs'][0]['CKKS_real'][0]=10.
        result=self.assess(witness)
        self.assertTrue(result['raw_compile_passed']);self.assertFalse(result['raw_numeric_passed'])
        self.assertEqual(result['inputs'][0]['CKKS']['failed_values'],1)

    def test_CKKS_imaginary_error_is_included(self):
        witness=self.witness();witness['inputs'][1]['CKKS_imag']=[.1,0.]
        result=self.assess(witness);self.assertFalse(result['raw_numeric_passed'])
        self.assertGreater(result['inputs'][1]['CKKS']['max_absolute_error'],.09)

    def test_math_approximation_application_domain_each_remain_required(self):
        for i,field in enumerate(('dsl_mathematical_output','original_reference','application_metrics','inside_frozen_domain','whole_model_output')):
            witness=self.witness()
            witness['inputs'][0][field]={'dsl_mathematical_output':[1.01,4.], 'original_reference':[2.,4.],
                'application_metrics':dict(accuracy=.5),'inside_frozen_domain':False,'whole_model_output':False}[field]
            # No additional generation is needed to audit independent witness variants.
            result=evaluate_witness(witness,self.policy,sha(self.source.encode()))
            self.assertFalse(result['raw_numeric_passed'],field)

    def test_missing_inputs_simulation_and_intermediate_decryption_cannot_pass(self):
        for mutation in ('missing','simulation','reencrypt','reference'):
            witness=self.witness()
            if mutation=='missing':witness['inputs'].pop()
            elif mutation=='simulation':witness['real_CKKS']=False
            elif mutation=='reencrypt':witness['intermediate_decryptions']=1
            else:witness['reference_or_tolerance_changed']=True
            self.assertFalse(evaluate_witness(witness,self.policy,sha(self.source.encode()))['raw_numeric_passed'])

    def test_threshold_relaxation_or_different_compiler_source_is_rejected(self):
        policy=copy.deepcopy(self.policy);policy.pop('binding');policy['CKKS']['atol']=1.;policy=bound(policy)
        with self.assertRaises(ValueError):evaluate_witness(self.witness(),policy,sha(self.source.encode()))
        witness=self.witness();witness['compiler_source_sha256']='0'*64
        self.assertFalse(evaluate_witness(witness,self.policy,sha(self.source.encode()))['raw_compile_passed'])

    def cohort(self):
        rows=[dict(request_id=str(i),model_sha256='same-model' if i<2 else 'other-model',
            input_sha256s=['a'*64,'b'*64,'c'*64],split='regression' if i<2 else 'acceptance',
            tier='operators' if i<2 else 'complete',unseen=i==2) for i in range(3)]
        return bound(dict(format='poseidon-first-output-cohort-v1',mechanism_binding=self.mechanism['binding'],
            independent_seal_verified=True,requests=rows))

    def outcome(self, request_id, status):
        return bound(dict(mode='external_first_output_assessment',request_id=request_id,status=status,
            mechanism_binding=self.mechanism['binding'],first_output_eligible=True,source_unchanged=True,
            post_output_repairs=0,first_compile_passed=status=='passed',first_numeric_passed=status=='passed',
            first_joint_passed=status=='passed',input_groups_evaluated=3 if status=='passed' else 0))

    def test_request_denominator_keeps_blockers_and_distinguishes_models_inputs(self):
        result=summarize_requests(self.cohort(),{'0':self.outcome('0','passed'),'2':self.outcome('2','backend_blocked')})
        totals=result['overall_descriptive_only'];self.assertEqual(totals['requests'],3);self.assertEqual(totals['models'],2)
        self.assertEqual(totals['planned_input_groups'],9);self.assertEqual(totals['evaluated_input_groups'],3)
        self.assertEqual(totals['first_joint_rate'],1/3);self.assertEqual(totals['statuses']['not_started'],1)
        self.assertEqual(len(result['strata']),2)

    def test_refused_unsupported_resource_and_backend_blocked_stay_in_denominator(self):
        for status in ('refused','unsupported','resource_blocked','backend_blocked','provider_failed'):
            totals=summarize_requests(self.cohort(),{'0':self.outcome('0',status)})['overall_descriptive_only']
            self.assertEqual(totals['requests'],3);self.assertEqual(totals['first_joint_rate'],0.)
            self.assertEqual(totals['statuses'][status],1)

    def test_repaired_development_or_changed_version_cannot_replace_first_result(self):
        for field,value in [('mode','development_assessment'),('mechanism_binding','0'*64)]:
            outcome=self.outcome('0','passed');outcome.pop('binding');outcome[field]=value;outcome=bound(outcome)
            with self.assertRaises(ValueError):summarize_requests(self.cohort(),{'0':outcome})
        outcome=self.outcome('0','passed');outcome.pop('binding');outcome['post_output_repairs']=1;outcome=bound(outcome)
        self.assertEqual(summarize_requests(self.cohort(),{'0':outcome})['overall_descriptive_only']['first_joint_rate'],0.)

    def test_unseen_requires_independent_seal(self):
        cohort=self.cohort();cohort.pop('binding');cohort['independent_seal_verified']=False;cohort=bound(cohort)
        with self.assertRaises(ValueError):summarize_requests(cohort,{})

    def test_legacy_feedback_workflow_is_explicitly_development(self):
        class Application:
            def run(self,task,provider):return dict(status='succeeded',generations=2)
        result=run_development(Application(),'task',None)
        self.assertEqual(result['mode'],'development_assessment');self.assertFalse(result['first_output_acceptance_eligible'])

    def test_second_output_location_cannot_replace_first_for_same_request(self):
        plan=delivery_plan([dict(request=self.request,policy=self.policy,delivery_path=str(self.root/'only-slot'),split='regression')])
        mechanism=generator_manifest('sealed-slots-v2',[],baseline_binding='0'*64,capabilities_sha256='1'*64,
            evaluation_policy_sha256=sha(canonical(plan)),provider_config=Config(max_calls=1,max_tokens=128))
        first=DeepSeekProvider(Config(max_calls=1,max_tokens=128),transport=Transport(self.answer()))
        generate_once(self.request,first,self.root/'only-slot',mechanism=mechanism,plan=plan)
        second_transport=Transport(self.answer());second=DeepSeekProvider(Config(max_calls=1,max_tokens=128),transport=second_transport)
        with self.assertRaises(ValueError):
            generate_once(self.request,second,self.root/'picked-second-answer',mechanism=mechanism,plan=plan)
        with self.assertRaises(FileExistsError):
            generate_once(self.request,second,self.root/'only-slot',mechanism=mechanism,plan=plan)
        self.assertFalse(second_transport.calls)

    def test_changed_Provider_settings_are_rejected_before_any_attempt(self):
        provider=DeepSeekProvider(Config(max_calls=1,max_tokens=256),transport=Transport(self.answer()))
        with self.assertRaises(ValueError):generate_once(self.request,provider,self.root/'bad',mechanism=self.mechanism)
        self.assertEqual(provider.request_attempts,0)

    def test_multi_model_policies_share_frozen_mechanism_without_mixing_inputs(self):
        b=Builder([(2,)]);other=prepare_task(b.finish(b.node('add',['input0','input0'])))
        other_policy=evaluation_policy(['f'*64],domain_sha256='d'*64,approximation_requirements=[],application_requirements=[])
        plan=delivery_plan([dict(request=self.request,policy=self.policy,delivery_path=str(self.root/'one'),split='regression'),
            dict(request=other,policy=other_policy,delivery_path=str(self.root/'two'),split='regression')])
        mechanism=generator_manifest('multi-model-plan-v2',[],baseline_binding='0'*64,capabilities_sha256='1'*64,
            evaluation_policy_sha256=sha(canonical(plan)),provider_config=Config(max_calls=1,max_tokens=128))
        for i,(request,policy) in enumerate([(self.request,self.policy),(other,other_policy)]):
            raw=json.dumps(dict(schema=1,request_id=request['request_id'],hecate_source=self.source))
            provider=DeepSeekProvider(Config(max_calls=1,max_tokens=128),transport=Transport(raw))
            path=self.root/('one' if i==0 else 'two')
            receipt=generate_once(request,provider,path,mechanism=mechanism,plan=plan)
            witness=self.witness();witness['inputs']=witness['inputs'] if i==0 else [dict(witness['inputs'][0],input_sha256='f'*64)]
            result=assess_once(path,receipt['binding'],plan,self.root/('evaluated-'+str(i)),lambda request,source:witness)
            self.assertTrue(result['raw_joint_passed']);self.assertEqual(result['input_groups_evaluated'],len(policy['inputs']))
            self.assertEqual(result['mechanism_binding'],mechanism['binding'])


if __name__=='__main__':
    unittest.main()
