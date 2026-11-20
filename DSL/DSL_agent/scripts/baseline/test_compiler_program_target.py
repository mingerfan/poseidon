"""Twelve new synthetic program-target controls; no model/CKKS execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

from component_contract import prepare_task
from delivery_component import bound,canonical,read_bytes,sha
import compiler_target_coordinates as coords
import compiler_program_target as target

BRIDGE=Path('/home/lhohy/Code/Poseidon/DSL/DSL_agent/validation/backend_certification/program_complex_target_bridge_v2.py')
BRIDGE_SHA='e6c6b83aa62751f34fee0c64d83cbf8bbfc70b040014e0f9c7dc31311158f3cf'


def rebound(value):return bound({k:copy.deepcopy(v)for k,v in value.items()if k!='binding'})


def nominal(index):
    return dict(id=str(index),kind='ciphertext',place=dict(kind='host',rank=0),context='fixture',
        level=3,scale_log2=45,components=2,ntt=True)


def fixture(period=1024):
    size=period+2
    model=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='program_target_fixture',
        inputs=[dict(name='input0',shape=[size]),dict(name='input1',shape=[size])],constants={},
        nodes=[dict(id='a',op='add',inputs=['input0','input1'],attrs={},outputs=['v0']),
            dict(id='b',op='add',inputs=['v0','input1'],attrs={},outputs=['v1'])],
        outputs=[dict(name='final',value='v1'),dict(name='middle',value='v0'),dict(name='same_final',value='v1')])
    request=prepare_task(model,dict(chunk_period=period,output_binding_policy='exact-values-v1',
        compiler_configuration='poseidon-cpu-eva-w45-current-v1'))
    source=b'# Synthetic program mapping fixture; no compiler or HE execution.\n'
    origin_inputs=[dict(logical_value=n,chunk_index=c,local_value_id=str(i))for i,(n,c)in enumerate([
        ('input0',0),('input0',1),('input1',0),('input1',1),('zero_ct',0)])]
    source_units=[];units=[];receipts=[];plans={}
    for index,value in enumerate(['v0','v1']):
        local_inputs=copy.deepcopy(origin_inputs)
        if index:
            for i in range(2):local_inputs[i]['logical_value']='v0'
        local_outputs=[dict(logical_value=value,chunk_index=c,local_value_id=str(5+c))for c in range(2)]
        plan=dict(format_version=1,plan_id='fixture',target=dict(operator_spec=dict(source_sha256='sha256:'+'a'*64)),
            values=[nominal(i)for i in range(7)],external_inputs=[str(i)for i in range(5)],initialization=[],
            execution=[dict(kind='compute',op='add_cc',inputs=[str(i),str(i+2)],output=str(5+i),ordinal=i,
                place=dict(kind='host',rank=0))for i in range(2)],finalization=[],final_outputs=['5','6'])
        raw=canonical(plan);plan_sha=sha(raw);plans[plan_sha]=raw
        typ=dict(kind='ciphertext',components=2,level=3,scale_log2=45)
        dag=bound(dict(format=coords.DAG,parameters=[dict(ssa=f'%arg{i}',value_id=str(i),type=typ)for i in range(5)],
            returns=[dict(ssa=f'%r{i}',value_id=str(5+i),type=typ)for i in range(2)],value_count=7,instruction_count=2))
        selectors=[[ct,slot]for ct,n in enumerate([period,2])for slot in range(n)]
        local_request=sha(('unit-request%d'%index).encode());local_source=sha(('unit-source%d'%index).encode())
        io=bound(dict(format='poseidon-generic-compiler-logical-I-O-map-v1',request_id=local_request,source_sha256=local_source,
            compiler_DAG_binding=dag['binding'],inputs=[dict(abi_index=i,ssa=r['ssa'],value_id=r['value_id'],role=dict(dsl_name='x%d'%i))
                for i,r in enumerate(dag['parameters'])],outputs=[dict(name='chunk%d'%i,model_value=value+'_chunk%d'%i,
                shape=[n],alias_of=None,segments=[dict(ciphertext=i,value_id=str(5+i),logical_start=0,slot_start=0,elements=n)])
                for i,n in enumerate([period,2])],output_selectors_sha256=sha(canonical(selectors)),logical_values=size,
            unique_physical_cells=size,alias_cells=0))
        receipt=bound(dict(format=coords.COMPILER,request_id=local_request,source_sha256=local_source,DAG=dag,logical_IO=io,
            periodic_binding=bound(dict(format='poseidon-original-periodic-payload-correspondence-v1',input_slot_period=period,
                all_non_payload_fields_unchanged=True,plaintext_values_changed=False)),
            original_proof_binding=sha(('proof%d'%index).encode()),original_backend_identity_sha256='b'*64,original_compiler_sha256='c'*64,
            files_read={'attempt-00/output/lowered._hecate_golden.runtime-plan.json':dict(sha256=plan_sha),
                'attempt-00/output/bound/plan.json':dict(sha256=plan_sha)}))
        receipts.append(receipt)
        row=bound(dict(unit_index=index,source_plan_sha256=plan_sha,derived_plan_sha256=plan_sha,inputs=local_inputs,outputs=local_outputs))
        source_units.append(row)
        units.append(dict(unit_id='u%03d'%index,raw_plan_sha256=plan_sha,plan_sha256=plan_sha,source_origin_map_sha256=sha(canonical(row)),
            periodic_binding_sha256='d'*64,boundary_binding_sha256='e'*64,continuation_binding='f'*64,spec_sha256='a'*64,
            **{side:[dict(local_value_id=r['local_value_id'],logical=dict(value=r['logical_value'],chunk=r['chunk_index']))for r in entries]
                for side,entries in [('inputs',local_inputs),('outputs',local_outputs)]}))
    key=dict(epoch='1'*32,origin='2'*64,public_manifest_sha256='3'*64,library_sha256='4'*64)
    finals=[dict(value=v,chunk=c)for v in ['v1','v0']for c in range(2)]
    container=bound(dict(format='poseidon-bound-tensor-multiunit-source-container-v1',scope='synthetic-multiunit-control',
        execution_authority=False,key_epoch=key['epoch'],key_origin=key['origin'],
        source_model_sha256=sha(canonical(model)),parent_layout_sha256=sha(canonical(request['layout'])),
        units=source_units,inputs=origin_inputs,outputs=[dict(logical_value=r['value'],chunk_index=r['chunk'])for r in finals]))
    container_raw=canonical(container)
    program=bound(dict(format='poseidon-ciphertext-program-v1',scope=container['scope'],binding_stage='physical_bound',
        source=dict(container_sha256=sha(container_raw),origins_sha256=sha(container_raw),model_sha256=container['source_model_sha256'],
            layout_sha256=container['parent_layout_sha256']),key=key,units=units,
        inputs=[dict(logical=dict(value=r['logical_value'],chunk=r['chunk_index']),role='encrypted_zero'if r['logical_value']=='zero_ct'else'data',
            descriptor=nominal(i))for i,r in enumerate(origin_inputs)],final_outputs=finals))
    terminals=[bound(dict(unit_id=u['unit_id'],unit_plan_sha256=u['plan_sha256'],program_binding=program['binding'],
        case_binding='5'*64,input_bank_binding='6'*64))for u in units]
    selection=bound(dict(format='poseidon-program-final-selection-v1',program_binding=program['binding'],case_binding='5'*64,
        input_bank_binding='6'*64,key=key,terminals=terminals,outputs=[dict(logical=r,unit_id='u001'if r['value']=='v1'else'u000',
            unit_plan_sha256=units[1 if r['value']=='v1'else 0]['plan_sha256'],local_value_id=str(5+r['chunk']),decoded_file='final-%d.bin'%i)
            for i,r in enumerate(finals)]))
    expected=dict(request_id=request['request_id'],source_sha256=sha(source),container_sha256=sha(container_raw),container_binding=container['binding'],
        program_binding=program['binding'],selection_binding=selection['binding'],unit_receipt_bindings=[r['binding']for r in receipts])
    return dict(request=request,source=source,container_raw=container_raw,program=program,unit_receipts=receipts,plan_bytes=plans,selection=selection,expected=expected)


def reset_program(fixture):
    fixture['program']=rebound(fixture['program']);fixture['expected']['program_binding']=fixture['program']['binding']


def reset_selection(fixture):
    fixture['selection']=rebound(fixture['selection']);fixture['expected']['selection_binding']=fixture['selection']['binding']


class ProgramTargetControls(unittest.TestCase):
    def test_01_multiunit_local_ID_collisions_are_qualified(self):
        value=target.build(**fixture());returns=value['compiler_program_map']['returns']
        self.assertEqual(len({r['value_id']for r in returns}),4)
        self.assertEqual({r['local_value_id']for r in returns},{'5','6'})
        self.assertEqual(value['original_unit_count'],2)
        self.assertFalse(value['compiler_program_map']['one_compiler_global_DAG_or_whole_source_decomposition_claimed'])

    def test_02_tails_logical_and_periodic_aliases_all_counted(self):
        value=target.build(**fixture())
        self.assertEqual((value['logical_values'],value['unique_physical_cells'],value['logical_alias_cells']),(3078,2052,1026))
        self.assertEqual(value['periodic_alias_values'],46170);self.assertEqual(value['complex_components_with_periodic_aliases'],98496)
        self.assertEqual(value['decoder_outputs'][0]['segments'][1]['count'],2)
        self.assertEqual(value['target_outputs'][2]['physical_owner'],'final')

    def test_03_source_unit_origin_and_local_return_order_rejected(self):
        for side in ['inputs','outputs']:
            data=fixture();data['program']['units'][1][side].reverse();reset_program(data)
            with self.assertRaises(ValueError):target.build(**data)

    def test_04_whole_model_layout_and_encrypted_zero_roles_rejected(self):
        for mutation in ['model','layout','zero']:
            data=fixture()
            if mutation=='zero':data['program']['inputs'][-1]['role']='data'
            else:data['program']['source'][mutation+'_sha256']='0'*64
            reset_program(data)
            with self.assertRaises(ValueError):target.build(**data)

    def test_05_forward_input_or_overwritten_logical_origin_rejected(self):
        for mutation in ['forward','overwrite']:
            data=fixture()
            if mutation=='forward':data['program']['units'][0]['inputs'][0]['logical']=dict(value='v1',chunk=0)
            else:data['program']['units'][1]['outputs'][0]['logical']=dict(value='v0',chunk=0)
            reset_program(data)
            with self.assertRaises(ValueError):target.build(**data)

    def test_06_frozen_compiler_receipts_and_raw_bound_plan_hashes(self):
        data=fixture();data['expected']['unit_receipt_bindings'][0]='0'*64
        with self.assertRaises(ValueError):target.build(**data)
        data=fixture();data['program']['units'][0]['raw_plan_sha256']='0'*64;reset_program(data)
        with self.assertRaises(ValueError):target.build(**data)

    def test_07_final_selection_identity_order_and_missing_cells(self):
        for mutation in ['logical','id','missing','foreigncase']:
            data=fixture()
            if mutation=='logical':data['selection']['outputs'].reverse()
            elif mutation=='id':data['selection']['outputs'][0]['local_value_id']='0'
            elif mutation=='missing':data['selection']['outputs'].pop()
            else:data['selection']['case_binding']='7'*64
            reset_selection(data)
            with self.assertRaises(ValueError):target.build(**data)

    def test_08_missing_unit_metadata_or_same_compiler_identity_rejected(self):
        data=fixture();data['unit_receipts'].pop()
        with self.assertRaises(ValueError):target.build(**data)
        data=fixture();receipt=data['unit_receipts'][1];receipt['original_backend_identity_sha256']='0'*64
        data['unit_receipts'][1]=rebound(receipt);data['expected']['unit_receipt_bindings'][1]=data['unit_receipts'][1]['binding']
        with self.assertRaises(ValueError):target.build(**data)

    def test_09_actual_client_metadata_and_aggregate_limits_preserved(self):
        data=fixture();key=next(iter(data['plan_bytes']));data['plan_bytes'][key]+=b' '*(1024**2)
        with self.assertRaises(ValueError):target.build(**data)
        data=fixture();data['program']['units']*=9;reset_program(data)
        with self.assertRaises(ValueError):target.build(**data)
        transport=target.transport();data=fixture();unit=data['program']['units'][0]
        plan=__import__('json').loads(data['plan_bytes'][unit['plan_sha256']]);plan['execution']*=2049
        raw=canonical(plan);unit['plan_sha256']=sha(raw);data['plan_bytes'][sha(raw)]=raw
        data['program']=rebound(data['program'])
        with self.assertRaises(ValueError):transport.validate_program(data['program'],data['program']['binding'],data['plan_bytes'])

    def test_10_small_period_cannot_truncate_required_aliases(self):
        requirements=target.decoder_period_requirements(8)
        self.assertEqual(requirements['required_explicit_aliases_per_chunk'],2047);self.assertFalse(requirements['fits_original_decoder'])
        self.assertFalse(requirements['aliases_shortened_or_sampled'])
        with self.assertRaises(target.ProgramTargetBlocked)as caught:target.build(**fixture(period=8))
        self.assertEqual(caught.exception.reason,'all_periodic_aliases_exceed_original_decoder_limit:2047>16')

    def test_11_output_rows_interoperate_with_unchanged_B_v2(self):
        data=fixture();fragment=target.build(**data)
        code=read_bytes(BRIDGE);self.assertEqual(sha(code),BRIDGE_SHA)
        spec=importlib.util.spec_from_file_location('unchanged_B_target_bridge_for_control',str(BRIDGE));module=importlib.util.module_from_spec(spec)
        exec(compile(code,str(BRIDGE),'exec'),module.__dict__)
        rows=module.output_alignment(data['request']['model'],fragment['target_outputs'],dict(outputs=fragment['decoder_outputs']),
            data['selection'],fragment['logical_IO'],fragment['compiler_program_map'])
        self.assertEqual(sum(r['complex_components_with_periodic_aliases']for r in rows),98496)
        self.assertEqual(sum(r['periodic_alias_values']for r in rows),46170)

    def test_12_preserved_source_and_incomplete_native_claims(self):
        data=fixture();before=canonical({k:v for k,v in data.items()if k not in ['source','container_raw','plan_bytes']})
        fragment=target.build(**data)
        self.assertEqual(canonical({k:v for k,v in data.items()if k not in ['source','container_raw','plan_bytes']}),before)
        self.assertFalse(fragment['complete_decoder_contract_or_native_eligibility_created'])
        self.assertFalse(fragment['first_original_or_complex_whole_model_or_CKKS_success_certified'])
        self.assertEqual(fragment['numerical_rows_decoded'],0);self.assertGreaterEqual(len(fragment['missing_external_witnesses']),5)
        data['source']+=b'changed'
        with self.assertRaises(ValueError):target.build(**data)


if __name__=='__main__':unittest.main()
