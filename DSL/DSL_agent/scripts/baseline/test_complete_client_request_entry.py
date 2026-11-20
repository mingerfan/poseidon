"""Six new durable admission controls; no Provider or evaluator is invoked."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import complete_client_request_entry as entry
import complete_client_capability as capability
from component_contract import prepare_task
from delivery_component import bound,canonical,delivery_plan,generator_manifest,sha
from first_output_assessment import evaluation_policy,summarize_requests
from generation_protocol import Config

CONTROL_ROOT=None
CALLS=0
SCORECARD=None
def rebound(value):return bound({k:copy.deepcopy(v)for k,v in value.items()if k!='binding'})
def invoke(function,*args,**kwargs):
    global CALLS
    CALLS+=1
    return function(*args,**kwargs)


def fixture(root,kinds):
    cap=json.loads((Path(__file__).parent/'SOURCE_CAPABILITY.json').read_bytes())
    routes=[];slots=[];requests=[]
    for index,kind in enumerate(kinds):
        model=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='admission_probe_'+str(index),
            inputs=[dict(name='x',shape=[index+2])],constants={},
            nodes=[dict(id='square',op='square',inputs=['x'],attrs={},outputs=['squared'])],
            outputs=[dict(name='out',value='squared')])
        config='poseidon-cpu-eva-w45-current-v1'if kind=='unsupported'else'poseidon-cpu-eva-w44-native-epoch-v1'
        request=prepare_task(model,dict(compiler_configuration=config,chunk_period=8));requests.append(request)
        requirements=capability.freeze_requirements(request,rotations=[1],input_level=1,input_scale_log2=44,
            maximum_units=2,maximum_instructions=32,maximum_values=64,
            maximum_work_memory_bytes=4*1024**3+(1 if kind=='resource_blocked'else 0))
        policy=evaluation_policy(['a'*64,'b'*64],domain_sha256='c'*64,
            approximation_requirements=[],application_requirements=[],whole_model_required=False)
        routes.append(dict(request=request,requirements=requirements,capability=copy.deepcopy(cap)))
        slots.append(dict(request=request,policy=policy,delivery_path=str(root/('delivery-'+str(index))),split='development',unseen=False))
    plan=delivery_plan(slots)
    mechanism=generator_manifest('complete_client_admission_controls_r2849',list(entry.source_manifest()),
        baseline_binding='0'*64,capabilities_sha256=sha(canonical(cap)),evaluation_policy_sha256=sha(canonical(plan)),
        provider_config=Config(max_calls=1,max_tokens=128,provider_retries=0))
    mechanism=entry.freeze(mechanism,plan,routes,test_only=True)
    return requests,plan,mechanism


class CompleteClientRequestEntryControls(unittest.TestCase):
    def setUp(self):
        self.root=CONTROL_ROOT/self._testMethodName;self.root.mkdir(mode=0o700)
        self.patches=[patch('deepseek_provider.DeepSeekProvider.__init__',side_effect=AssertionError('Provider must not be constructed')),
            patch('deepseek_provider.HTTPSTransport.post',side_effect=AssertionError('No HTTP in admission'))]
        for value in self.patches:value.start();self.addCleanup(value.stop)

    def record(self,request,plan,mechanism,index=0):
        return invoke(entry.record_once,request,self.root/('delivery-'+str(index)),mechanism=mechanism,plan=plan,test_only=True)

    def test_01_unsupported_request_saved_without_Provider(self):
        requests,plan,mechanism=fixture(self.root,['unsupported']);receipt=self.record(requests[0],plan,mechanism)
        self.assertEqual(receipt['status'],'unsupported');self.assertFalse(receipt['first_output_eligible'])
        self.assertFalse((self.root/'delivery-0/candidate.py').exists())
        self.assertEqual(json.loads((self.root/'delivery-0/attempt.json').read_bytes())['Provider_constructions'],0)

    def test_02_resource_blocked_has_immutable_original_request(self):
        requests,plan,mechanism=fixture(self.root,['resource_blocked']);receipt=self.record(requests[0],plan,mechanism)
        self.assertEqual(receipt['status'],'resource_blocked')
        self.assertEqual((self.root/'delivery-0/request.json').read_bytes(),canonical(requests[0]))
        self.assertEqual(receipt['post_output_repairs'],0)

    def test_03_source_interface_match_stays_backend_blocked(self):
        requests,plan,mechanism=fixture(self.root,['backend_blocked']);receipt=self.record(requests[0],plan,mechanism)
        preserved,request,saved=entry.verify_record(self.root/'delivery-0',receipt['binding'])
        self.assertEqual(preserved['status'],'backend_blocked');self.assertEqual(request,requests[0]);self.assertEqual(saved,mechanism)
        self.assertEqual(preserved['mode'],'development_admission_probe');self.assertFalse(preserved['real_Agent'])
        self.assertFalse((self.root/'delivery-0/response.txt').exists())

    def test_04_rebound_decision_rejected_before_output_folder(self):
        requests,plan,mechanism=fixture(self.root,['backend_blocked']);key=requests[0]['request_id']
        decision=mechanism['complete_client_routes'][key]['frozen_admission'];decision['status']='unsupported'
        mechanism['complete_client_routes'][key]['frozen_admission']=rebound(decision);mechanism=rebound(mechanism)
        with self.assertRaisesRegex(ValueError,'Rebound or stale admission classification'):
            self.record(requests[0],plan,mechanism)
        self.assertFalse((self.root/'delivery-0').exists())

    def test_05_frozen_slot_is_exclusive_and_old_bytes_remain(self):
        requests,plan,mechanism=fixture(self.root,['backend_blocked']);receipt=self.record(requests[0],plan,mechanism)
        folder=self.root/'delivery-0';before={p.name:p.read_bytes()for p in folder.iterdir()}
        with self.assertRaises(FileExistsError):self.record(requests[0],plan,mechanism)
        self.assertEqual({p.name:p.read_bytes()for p in folder.iterdir()},before)
        self.assertEqual(entry.verify_record(folder,receipt['binding'])[0],receipt)

    def test_06_all_three_statuses_remain_in_standard_denominators(self):
        global SCORECARD
        kinds=['unsupported','resource_blocked','backend_blocked'];requests,plan,mechanism=fixture(self.root,kinds)
        assessments={};rows=[]
        for index,(request,kind)in enumerate(zip(requests,kinds)):
            receipt=self.record(request,plan,mechanism,index)
            result=invoke(entry.assess_blocked_once,self.root/('delivery-'+str(index)),receipt['binding'],plan,self.root/('assessment-'+str(index)))
            self.assertEqual(result['status'],kind);self.assertFalse(result['first_joint_passed'])
            self.assertEqual(result['input_groups_evaluated'],0);assessments[request['request_id']]=result
            rows.append(dict(request_id=request['request_id'],model_sha256=sha(canonical(request['model'])),
                input_sha256s=['a'*64,'b'*64],split='development',tier='operators',unseen=False))
        cohort=bound(dict(format='poseidon-first-output-cohort-v1',mechanism_binding=mechanism['binding'],requests=rows,
            independent_seal_verified=False,test_only=True,scope='inert admission accounting controls; not a real benchmark cohort'))
        SCORECARD=summarize_requests(cohort,assessments);summary=SCORECARD['overall_descriptive_only']
        self.assertEqual((summary['requests'],summary['planned_input_groups'],summary['evaluated_input_groups']),(3,6,0))
        self.assertEqual(summary['statuses'],{kind:1 for kind in kinds})
        self.assertEqual((summary['first_compile_rate'],summary['first_numeric_rate'],summary['first_joint_rate']),(0.,0.,0.))
        self.assertEqual(len(SCORECARD['strata']),1)
