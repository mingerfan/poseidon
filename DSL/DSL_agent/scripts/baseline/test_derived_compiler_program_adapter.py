"""Twelve new opt-in adapter controls, using synthetic metadata only."""
import copy
import json
import unittest

from candidate_contract import request_input_names
from component_contract import prepare_task
from delivery_component import bound,canonical,sha
from test_compiler_partition_derivation import fixture as whole_fixture
import derived_compiler_program_adapter as adapter


def rebound(value):return bound({k:copy.deepcopy(v)for k,v in value.items()if k!='binding'})


def compiler_fixture():
    model=dict(format='poseidon-model-graph-v2',resource_profile='general-tensors-v1',id='derived_program_fixture',
        inputs=[dict(name='input0',shape=[2]),dict(name='input1',shape=[2])],constants=dict(c=[.25,-0.0]),
        nodes=[dict(id='n0',op='add',inputs=['input0','c'],attrs={},outputs=['v4']),
            dict(id='n1',op='negate',inputs=['v4'],attrs={},outputs=['v5']),
            dict(id='n2',op='add',inputs=['v5','input1'],attrs={},outputs=['v6']),
            dict(id='n3',op='add',inputs=['v6','c'],attrs={},outputs=['v7']),
            dict(id='n4',op='add',inputs=['v7','v4'],attrs={},outputs=['v8'])],
        outputs=[dict(name='final',value='v8'),dict(name='branch',value='v6'),dict(name='same_final',value='v8')])
    request=prepare_task(model,dict(chunk_period=8,output_binding_policy='exact-values-v1',compiler_configuration='poseidon-cpu-eva-w45-current-v1'))
    receipt,raw,_=whole_fixture();source=b'# Synthetic compiler receipt fixture: no real compiler or HE was executed.\n'
    receipt['request_id']=request['request_id'];receipt['source_sha256']=sha(source)
    receipt['periodic_binding'].update(input_slot_period=8);receipt['periodic_binding']=rebound(receipt['periodic_binding'])
    names=list(request_input_names(request));roles=request['layout']['inputs']+request['layout']['auxiliary_ciphertexts']
    returns=receipt['DAG']['returns'];outputs=[]
    for model_row,layout in zip(model['outputs'],request['layout']['outputs']):
        ct=layout['chunks'][0]['ciphertext']
        outputs.append(dict(name=model_row['name'],model_value=model_row['value'],shape=[2],alias_of=layout.get('alias_of'),
            segments=[dict(ciphertext=ct,value_id=returns[ct]['value_id'],logical_start=0,slot_start=0,elements=2)]))
    io=bound(dict(format='poseidon-generic-compiler-logical-I-O-map-v1',compiler_DAG_binding=receipt['DAG']['binding'],
        request_id=request['request_id'],source_sha256=sha(source),
        inputs=[dict(abi_index=i,ssa=receipt['DAG']['parameters'][i]['ssa'],value_id=receipt['DAG']['parameters'][i]['value_id'],role=role)for i,role in enumerate(roles)],
        outputs=outputs,output_selectors_sha256=sha(canonical(request['layout']['output_selectors'])),logical_values=6,unique_physical_cells=4,alias_cells=2))
    receipt['logical_IO']=io;receipt['original_compiler_sha256']='c'*64;receipt['original_backend_identity_sha256']='d'*64
    receipt['original_proof_binding']='e'*64;receipt=rebound(receipt)
    receipt['files_read']['attempt-00/output/lowered._hecate_golden.runtime-plan.json']=dict(sha256=sha(raw),bytes=len(raw));receipt=rebound(receipt)
    policy=adapter.freeze_policy(2)
    expected=dict(receipt_binding=receipt['binding'],request_id=request['request_id'],source_sha256=sha(source),
        whole_plan_sha256=sha(raw),policy_binding=policy['binding'])
    return dict(request=request,source=source,receipt=receipt,whole_raw=raw,policy=policy,expected=expected)


def runtime_fixture():
    original=compiler_fixture();compiled,partition=adapter.derive(**original)
    plans={u['plan_sha256']:canonical(u['plan'])for u in partition['units']}
    source_units=[];units=[]
    for origin in compiled['units']:
        def sources(side):
            return [dict(local_value_id=r['local_value_id'],whole_value_id=r['whole_value_id'],
                logical_value=r['logical']['value'],chunk_index=r['logical']['chunk'])for r in origin[side]]
        source=bound(dict(unit_index=origin['unit_index'],source_plan_sha256=origin['plan_sha256'],derived_plan_sha256=origin['plan_sha256'],
            origin_role='exact_partition_of_one_whole_compiler',actually_compiled_as_separate_unit=False,
            whole_compiler_receipt_binding=compiled['whole_compiler_receipt_binding'],partition_unit_binding=origin['partition_unit_binding'],
            inputs=sources('inputs'),outputs=sources('outputs')))
        source_units.append(source)
        plan=json.loads(plans[origin['plan_sha256']])
        units.append(dict(unit_id=origin['unit_id'],raw_plan_sha256=origin['plan_sha256'],plan_sha256=origin['plan_sha256'],
            source_origin_map_sha256=sha(canonical(source)),periodic_binding_sha256='1'*64,boundary_binding_sha256='2'*64,
            continuation_binding='3'*64,spec_sha256=plan['target']['operator_spec']['source_sha256'][7:],
            **{side:[dict(local_value_id=r['local_value_id'],logical=r['logical'])for r in origin[side]]for side in ('inputs','outputs')}))
    key=dict(epoch='4'*32,origin='5'*64,public_manifest_sha256='6'*64,library_sha256='7'*64)
    container=bound(dict(format='poseidon-bound-tensor-multiunit-source-container-v1',execution_authority=False,
        origin_role='exact_partition_of_one_whole_compiler',whole_compiler_receipt_binding=compiled['whole_compiler_receipt_binding'],
        partition_audit_binding=compiled['partition_audit']['binding'],scope='synthetic-derived-program',
        key_epoch=key['epoch'],key_origin=key['origin'],source_model_sha256=compiled['source_model_sha256'],
        parent_layout_sha256=compiled['layout_sha256'],units=source_units,
        inputs=[dict(logical_value=r['logical']['value'],chunk_index=r['logical']['chunk'])for r in compiled['physical_inputs']],
        outputs=[dict(logical_value=r['logical']['value'],chunk_index=r['logical']['chunk'])for r in compiled['finals']]))
    container_raw=canonical(container)
    program=bound(dict(format='poseidon-ciphertext-program-v1',binding_stage='physical_bound',scope=container['scope'],key=key,
        source=dict(container_sha256=sha(container_raw),origins_sha256=sha(container_raw),model_sha256=compiled['source_model_sha256'],layout_sha256=compiled['layout_sha256']),
        units=units,inputs=[dict(logical=r['logical'],role=r['role'],descriptor=r['descriptor'])for r in compiled['physical_inputs']],
        final_outputs=[r['logical']for r in compiled['finals']]))
    terminals=[bound(dict(unit_id=u['unit_id'],unit_plan_sha256=u['plan_sha256'],program_binding=program['binding'],
        case_binding='8'*64,input_bank_binding='9'*64))for u in units]
    by_id={u['unit_id']:u for u in units}
    selection=bound(dict(format='poseidon-program-final-selection-v1',program_binding=program['binding'],case_binding='8'*64,input_bank_binding='9'*64,
        key=key,terminals=terminals,outputs=[dict(logical=r['logical'],unit_id=r['unit_id'],local_value_id=r['local_value_id'],
            unit_plan_sha256=by_id[r['unit_id']]['plan_sha256'],decoded_file='derived-final-%d.bin'%i)for i,r in enumerate(compiled['finals'])]))
    expected=dict(compiled_binding=compiled['binding'],program_binding=program['binding'],source_container_binding=container['binding'],
        source_container_sha256=sha(container_raw),selection_binding=selection['binding'])
    return original,dict(compiled=compiled,program=program,source_container_raw=container_raw,selection=selection,plan_bytes=plans,expected=expected)


def fragment(data):
    runtime=adapter.runtime_origins(**data)
    return adapter.compact_target(data['compiled'],runtime,data['compiled']['binding'],runtime['binding']),runtime


class DerivedProgramControls(unittest.TestCase):
    def test_01_whole_compiler_proof_and_source_only_policy(self):
        original=compiler_fixture();compiled,artifact=adapter.derive(**original)
        self.assertEqual(len(compiled['units']),3);self.assertEqual(artifact['cuts'],[2,4])
        self.assertEqual(compiled['whole_compiler_receipt_binding'],original['receipt']['binding'])
        self.assertTrue(all(not u['actually_compiled_as_separate_unit']for u in compiled['units']))
        self.assertFalse(compiled['separate_unit_real_compilation_receipts_created'])

    def test_02_runtime_origins_preserve_all_source_unit_edges(self):
        original,data=runtime_fixture();result=adapter.runtime_origins(**data)
        self.assertEqual((result['value_count'],result['instruction_count']),(14,7))
        self.assertEqual([r['whole_value_id']for r in result['returns']],['8','6'])
        self.assertEqual(result['compiled_binding'],data['compiled']['binding'])
        self.assertFalse(result['mathematical_or_CKKS_or_first_delivery_eligibility_granted'])

    def test_03_complete_period8_and_exact_logical_aliases(self):
        original,data=runtime_fixture();value,runtime=fragment(data);bridge,decoder=adapter.pinned_B_modules()
        rows=bridge.output_alignment(original['request']['model'],value['target_outputs'],dict(outputs=value['decoder_outputs']),
            data['selection'],value['logical_IO'],value['compiler_program_map'],expected_period=8)
        self.assertEqual((value['logical_values'],value['unique_physical_cells'],value['logical_alias_cells']),(6,4,2))
        self.assertEqual(value['periodic_alias_values'],12282)
        self.assertEqual(sum(r['complex_components_with_periodic_aliases']for r in rows),24576)
        self.assertEqual(value['decoder_outputs'][0]['segments'][0]['periodic'],dict(period=8,stride=8,count=2048))

    def test_04_complete_periodic_slots_and_byte_offsets_without_numeric_read(self):
        original,data=runtime_fixture();value,_=fragment(data);_,decoder=adapter.pinned_B_modules()
        files={r['decoded_file']for r in data['selection']['outputs']}
        self.assertEqual(decoder.validate_outputs(value['decoder_outputs'],files,8),(6,12282))
        cells=list(decoder.selector_cells(value['decoder_outputs']));self.assertEqual(len(cells),12288)
        self.assertEqual(max(r['alias_ordinal']for r in cells),2047)
        self.assertTrue(all(r['imaginary_byte_offset']==r['real_byte_offset']+8 for r in cells))
        self.assertTrue(all('real'not in r and 'imaginary'not in r for r in cells))

    def test_05_period1_is_not_capped_to_sixteen_copies(self):
        _,decoder=adapter.pinned_B_modules()
        rows=[dict(name='single',shape=[1],segments=[dict(file='one.bin',logical_start=0,count=1,slot_start=0,
            periodic=dict(period=1,stride=1,count=16384))])]
        self.assertEqual(decoder.validate_outputs(rows,{'one.bin'},1),(1,16383))
        cells=list(decoder.selector_cells(rows));self.assertEqual(len(cells),16384)
        self.assertEqual(cells[-1]['slot'],16383)

    def test_06_changed_compiler_rule_source_or_output_frozen_bytes_rejected(self):
        for kind in ['policy','source','plan']:
            original=compiler_fixture()
            if kind=='policy':original['policy']['maximum_execution_operations']=3;original['policy']=rebound(original['policy'])
            elif kind=='source':original['source']+=b'changed'
            else:original['whole_raw']+=b' '
            with self.assertRaises(ValueError):adapter.derive(**original)

    def test_07_no_unit_compilation_forgery_or_runtime_retaging(self):
        original,data=runtime_fixture();container=json.loads(data['source_container_raw'])
        container['units'][0]['actually_compiled_as_separate_unit']=True;container['units'][0]=rebound(container['units'][0]);container=rebound(container)
        data['program']['units'][0]['source_origin_map_sha256']=sha(canonical(container['units'][0]))
        raw=canonical(container);data['source_container_raw']=raw;data['expected']['source_container_binding']=container['binding'];data['expected']['source_container_sha256']=sha(raw)
        data['program']['source']['container_sha256']=data['program']['source']['origins_sha256']=sha(raw)
        data['program']=rebound(data['program']);data['expected']['program_binding']=data['program']['binding']
        with self.assertRaises(ValueError):adapter.runtime_origins(**data)
        original,data=runtime_fixture();data['program']['inputs'][0]['descriptor']['scale_log2']+=1
        data['program']=rebound(data['program']);data['expected']['program_binding']=data['program']['binding']
        with self.assertRaises(ValueError):adapter.runtime_origins(**data)

    def test_08_missing_branch_local_ID_or_final_selection_rejected(self):
        for kind in ['origin','selector','missing']:
            original,data=runtime_fixture()
            if kind=='origin':
                data['program']['units'][1]['inputs'].reverse();data['program']=rebound(data['program']);data['expected']['program_binding']=data['program']['binding']
            elif kind=='selector':
                data['selection']['outputs'].reverse();data['selection']=rebound(data['selection']);data['expected']['selection_binding']=data['selection']['binding']
            else:
                data['program']['units'].pop();data['program']=rebound(data['program']);data['expected']['program_binding']=data['program']['binding']
            with self.assertRaises(ValueError):adapter.runtime_origins(**data)

    def test_09_period_stride_or_copies_cannot_sample_overlap_or_skip(self):
        original,data=runtime_fixture();value,_=fragment(data);_,decoder=adapter.pinned_B_modules()
        for key,number in [('count',16),('stride',4),('period',16)]:
            rows=copy.deepcopy(value['decoder_outputs']);rows[0]['segments'][0]['periodic'][key]=number
            with self.assertRaises(ValueError):decoder.validate_outputs(rows,{r['decoded_file']for r in data['selection']['outputs']},8)

    def test_10_aliases_must_keep_original_model_value_and_owner(self):
        original,data=runtime_fixture();value,_=fragment(data);bridge,_=adapter.pinned_B_modules()
        for kind in ['expression','owner','file']:
            targets=copy.deepcopy(value['target_outputs']);rows=copy.deepcopy(value['decoder_outputs'])
            if kind=='expression':targets[1]['model_value']='v8'
            elif kind=='owner':targets[2]['physical_owner']='same_final'
            else:rows[2]['segments'][0]['file']=rows[1]['segments'][0]['file']
            with self.assertRaises(ValueError):bridge.output_alignment(original['request']['model'],targets,dict(outputs=rows),
                data['selection'],value['logical_IO'],value['compiler_program_map'],expected_period=8)

    def test_11_original_program_file_and_foreign_case_identity_rejected(self):
        original,data=runtime_fixture();container=json.loads(data['source_container_raw']);container['extra']='x'*(1024**2);container=rebound(container)
        raw=canonical(container);data['source_container_raw']=raw;data['expected']['source_container_binding']=container['binding'];data['expected']['source_container_sha256']=sha(raw)
        data['program']['source']['container_sha256']=data['program']['source']['origins_sha256']=sha(raw)
        data['program']=rebound(data['program']);data['expected']['program_binding']=data['program']['binding']
        with self.assertRaises(ValueError):adapter.runtime_origins(**data)
        original,data=runtime_fixture();data['selection']['case_binding']='a'*64;data['selection']=rebound(data['selection']);data['expected']['selection_binding']=data['selection']['binding']
        with self.assertRaises(ValueError):adapter.runtime_origins(**data)

    def test_12_no_mutation_feedback_native_or_first_output_credit(self):
        original,data=runtime_fixture();before=canonical(data['compiled']);value,runtime=fragment(data)
        self.assertEqual(canonical(data['compiled']),before);self.assertEqual(value['numerical_values_decoded'],0)
        self.assertFalse(value['full_decoder_contract_or_actual_host_input_math_CKKS_or_first_delivery_qualification'])
        self.assertTrue(value['every_periodic_copy_included'])
        runtime['mathematical_or_CKKS_or_first_delivery_eligibility_granted']=True;runtime=rebound(runtime)
        with self.assertRaises(ValueError):adapter.compact_target(data['compiled'],runtime,data['compiled']['binding'],runtime['binding'])


if __name__=='__main__':unittest.main()
