"""Ten new controls; old TestCases are not executed. All plans remain inert."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import complete_client_capability as target
from delivery_component import bound, canonical, sha
from component_contract import prepare_task
from test_derived_compiler_program_adapter import runtime_fixture
import derived_compiler_program_adapter as adapter

CALLS=0
def rebound(value): return bound({k:copy.deepcopy(v)for k,v in value.items()if k!='binding'})
def fingerprint(value):
    if type(value)is bytes:return dict(bytes_sha256=sha(value),length=len(value))
    if type(value)is dict:return {k:fingerprint(v)for k,v in value.items()}
    if type(value)in(list,tuple):return [fingerprint(v)for v in value]
    return value
def invoke(function,*args,**kwargs):
    global CALLS
    CALLS+=1
    return function(*args,**kwargs)


def fixture():
    original,data=runtime_fixture(); request=original['request']
    cap=json.loads((Path(__file__).parent/'SOURCE_CAPABILITY.json').read_bytes())
    cap.update(test_only=True,compiler_configuration=copy.deepcopy(request['compiler_configuration']),runtime_context='fixture',
        input_state=dict(level=3,scale_log2=45),level_min=0,level_max=12)
    cap['context'].update(parameter_identity=request['compiler_configuration']['parameter_identity'],
        compiler_profile_sha256=request['compiler_profile_sha256'],q=cap['context']['q'][:13],p=cap['context']['p'][:1])
    cap=rebound(cap)
    requirements=target.freeze_requirements(request,rotations=[1,-1],input_level=3,input_scale_log2=45,
        maximum_units=16,maximum_instructions=4096,maximum_values=8192)
    expected=dict(request_id=request['request_id'],requirements_binding=requirements['binding'],capability_binding=cap['binding'])
    return original,data,request,requirements,cap,expected


def public_native_request():
    model=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='public_native_shape_only',
        inputs=[dict(name='x',shape=[2])],constants={},nodes=[dict(id='sq',op='square',inputs=['x'],attrs={},outputs=['squared'])],
        outputs=[dict(name='out',value='squared')])
    return prepare_task(model,dict(compiler_configuration='poseidon-cpu-eva-w44-native-epoch-v1',chunk_period=8))


class CompleteClientCapabilityControls(unittest.TestCase):
    def test_01_original_public_native_config_only_interface_compatible(self):
        request=public_native_request();cap=json.loads((Path(__file__).parent/'SOURCE_CAPABILITY.json').read_bytes())
        requirements=target.freeze_requirements(request,rotations=[1],input_level=1,input_scale_log2=44,
            maximum_units=2,maximum_instructions=32,maximum_values=64)
        result=invoke(target.assess,request,requirements,cap,dict(request_id=request['request_id'],requirements_binding=requirements['binding'],capability_binding=cap['binding']))
        self.assertEqual(result['status'],'backend_blocked');self.assertTrue(result['interface_compatible'])
        self.assertFalse(result['Provider_or_execution_authority'])

    def test_02_ordinary_profile_rejects_before_any_Provider(self):
        original,data,request,requirements,cap,expected=fixture()
        actual=json.loads((Path(__file__).parent/'SOURCE_CAPABILITY.json').read_bytes());expected['capability_binding']=actual['binding']
        result=invoke(target.assess,request,requirements,actual,expected)
        self.assertEqual(result['status'],'unsupported');self.assertIn('exact_compiler_profile_bytes_mismatch',result['reasons'])
        self.assertEqual(result['Provider_calls'],0)

    def test_03_retagged_request_and_stale_independent_pin_rejected(self):
        _,_,request,requirements,cap,expected=fixture()
        request['compiler_configuration']['parameter_identity']='poseidon-agent-n15-q16-p2-tc128-original-epoch-v1'
        with self.assertRaises(ValueError):invoke(target.assess,request,requirements,cap,expected,test_only=True)
        _,_,request,requirements,cap,expected=fixture();expected['requirements_binding']='0'*64
        with self.assertRaises(ValueError):invoke(target.assess,request,requirements,cap,expected,test_only=True)

    def test_04_missing_rotation_and_conjugation_rejected(self):
        for key,value in [('rotations',[3]),('conjugation',True),('required_refreshes',1)]:
            _,_,request,requirements,cap,expected=fixture();requirements[key]=value;requirements=rebound(requirements);expected['requirements_binding']=requirements['binding']
            cap['conjugation']=False;cap=rebound(cap);expected['capability_binding']=cap['binding']
            result=invoke(target.assess,request,requirements,cap,expected,test_only=True)
            if key=='required_refreshes':
                self.assertEqual(result['status'],'backend_blocked');self.assertIn('required_bootstrap_capability_not_certified',result['reasons'])
            else:
                self.assertEqual(result['status'],'unsupported');self.assertIn('required_rotation_or_conjugation_not_available',result['reasons'])

    def test_05_input_scale_and_input_level_mismatch_rejected(self):
        for key in ['input_scale_log2','input_level']:
            _,_,request,requirements,cap,expected=fixture();requirements[key]+=1;requirements=rebound(requirements);expected['requirements_binding']=requirements['binding']
            result=invoke(target.assess,request,requirements,cap,expected,test_only=True)
            self.assertIn('original_input_level_or_scale_mismatch',result['reasons'])

    def test_06_aggregate_bounds_and_boolean_resource_not_coerced(self):
        for key,value in [('maximum_units',17),('maximum_units',True),('maximum_work_memory_bytes',4*1024**3+1)]:
            _,_,request,requirements,cap,expected=fixture();requirements[key]=value;requirements=rebound(requirements);expected['requirements_binding']=requirements['binding']
            result=invoke(target.assess,request,requirements,cap,expected,test_only=True)
            self.assertIn(key+'_outside_capability',result['reasons']);self.assertEqual(result['status'],'resource_blocked')

    def test_07_exact_original_program_and_all_periodic_aliases(self):
        original,data,request,requirements,cap,expected=fixture();before=canonical(fingerprint([original,data,request,requirements,cap]))
        result=invoke(target.audit_external,request,requirements,cap,expected,original,data,test_only=True)
        self.assertEqual((result['actual_unit_plans_checked'],result['physical_finals']),(3,2))
        self.assertEqual((result['logical_values'],result['periodic_alias_values'],result['complete_real_imag_components']),(6,12282,24576))
        self.assertEqual(canonical(fingerprint([original,data,request,requirements,cap])),before)
        self.assertFalse(result['compiler_host_input_samekey_terminal_math_CKKS_or_first_qualification'])

    def test_08_compact_helper_cannot_drop_copies_or_change_alias_owner(self):
        for kind in ['periodic','owner']:
            original,data,request,requirements,cap,expected=fixture();runtime=adapter.runtime_origins(**data)
            fragment=adapter.compact_target(data['compiled'],runtime,data['compiled']['binding'],runtime['binding'])
            if kind=='periodic':fragment['decoder_outputs'][0]['segments'][0]['periodic']['count']=16
            else:fragment['target_outputs'][2]['physical_owner']='same_final'
            fragment=rebound(fragment)
            with patch.object(adapter,'compact_target',return_value=fragment):
                with self.assertRaisesRegex(ValueError,'Program (target bridge|complex decoder):'):invoke(target.audit_external,request,requirements,cap,expected,original,data,test_only=True)

    def test_09_rebound_unit_scale_and_false_whole_derivation_rejected(self):
        for kind in ['plan','compiled']:
            original,data,request,requirements,cap,expected=fixture()
            if kind=='plan':
                old=next(iter(data['plan_bytes']));plan=json.loads(data['plan_bytes'][old]);plan['values'][0]['scale_log2']+=1
                data['plan_bytes'][old]=canonical(plan)
            else:
                data['compiled']['physical_inputs'][0]['descriptor']['level']+=1;data['compiled']=rebound(data['compiled']);data['expected']['compiled_binding']=data['compiled']['binding']
            with self.assertRaises(ValueError):invoke(target.audit_external,request,requirements,cap,expected,original,data,test_only=True)

    def test_10_fixture_or_source_only_flag_cannot_grant_execution(self):
        _,_,request,requirements,cap,expected=fixture()
        with self.assertRaises(ValueError):invoke(target.assess,request,requirements,cap,expected)
        cap['execution_qualified']=True;cap=rebound(cap);expected['capability_binding']=cap['binding']
        with self.assertRaises(ValueError):invoke(target.assess,request,requirements,cap,expected,test_only=True)


if __name__=='__main__':unittest.main()
