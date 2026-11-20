"""Twelve new source-partition controls; every plan here is synthetic."""
import copy
import unittest
from unittest.mock import patch

from delivery_component import bound,canonical,sha
import compiler_partition_derivation as derivation


def rebound(value):return bound({k:copy.deepcopy(v)for k,v in value.items()if k!='binding'})


def descriptor(index,kind='ciphertext'):
    return dict(id=str(index),kind=kind,place=dict(kind='host',rank=0),context='fixture',level=3,
        scale_log2=45,components=1 if kind=='plaintext'else 2,ntt=True)


def source(plan):
    raw=canonical(plan)
    typ=dict(kind='ciphertext',level=3,scale_log2=45,components=2)
    dag=bound(dict(format=derivation.DAG,value_count=len(plan['values']),
        instruction_count=len(plan['initialization'])+len(plan['execution']),
        parameters=[dict(ssa=f'%arg{i}',value_id=v,type=typ)for i,v in enumerate(plan['external_inputs'])],
        returns=[dict(ssa=f'%result{i}',value_id=v,type=typ)for i,v in enumerate(plan['final_outputs'])]))
    receipt=bound(dict(format=derivation.COMPILER,request_id='a'*64,source_sha256='b'*64,DAG=dag,
        periodic_binding=bound(dict(format='poseidon-original-periodic-payload-correspondence-v1',
            all_non_payload_fields_unchanged=True,plaintext_values_changed=False)),
        files_read={'attempt-00/output/bound/plan.json':dict(sha256=sha(raw),bytes=len(raw))},
        metadata_origin='synthetic_fixture_not_real_compilation'))
    expected=dict(receipt_binding=receipt['binding'],request_id=receipt['request_id'],source_sha256=receipt['source_sha256'],whole_plan_sha256=sha(raw))
    return receipt,raw,expected


def fixture(*,streaming=False,chain=None):
    public=dict(kind='encode',output='3',ordinal=0,place=dict(kind='host',rank=0),payload=dict(kind='inline',values=[.25,-0.0]))
    if chain is None:
        operations=[('add_cp',['0','3'],'4'),('negate',['4'],'5'),('add_cc',['5','1'],'6'),
            ('add_cp',['6','3'],'7'),('add_cc',['7','4'],'8')]
        finals=['8','6']
    else:
        operations=[('add_cp',[str(0 if i==0 else i+3),'3'],str(i+4))for i in range(chain)]
        finals=[str(chain+3)]
    execution=[dict(kind='compute',op=op,inputs=args,output=out,ordinal=i+1,place=dict(kind='host',rank=0))
        for i,(op,args,out)in enumerate(operations)]
    init=[public]
    if streaming:
        init=[];execution=[public,*execution]
        for i,op in enumerate(execution):op['ordinal']=i
    max_id=max(int(op['output'])for op in execution)
    plan=dict(format_version=1,plan_id='fixture',target=dict(target_id='fixture',operator_spec=dict(source_sha256='sha256:'+'c'*64)),
        values=[descriptor(i,'plaintext'if i==3 else'ciphertext')for i in range(max_id+1)],
        external_inputs=['0','1','2'],initialization=init,execution=execution,finalization=[],final_outputs=finals)
    return source(plan)


def built(**kwargs):
    receipt,raw,expected=fixture(**kwargs)
    value=derivation.partition(receipt,raw,[2],expected)
    return receipt,raw,expected,value


def resign_unit(value,index):
    unit=value['units'][index];unit['plan_sha256']=sha(canonical(unit['plan']));value['units'][index]=rebound(unit)
    return rebound(value)


def check(receipt,raw,expected,value):return derivation.audit(receipt,raw,value,expected,value['binding'])


class PartitionDerivationControls(unittest.TestCase):
    def test_01_complete_source_and_residual_liveness(self):
        receipt,raw,expected,value=built();proof=check(receipt,raw,expected,value)
        self.assertEqual(proof['units'][0]['inputs'],['0','1','2'])
        self.assertEqual(proof['units'][0]['outputs'],['4','5'])
        self.assertEqual(proof['units'][1]['inputs'],['1','4','5'])
        self.assertEqual(proof['whole_final_values'],['8','6'])
        self.assertEqual(proof['derived_values'],13)
        self.assertTrue(proof['whole_compute_and_streaming_operations_preserved'])
        self.assertFalse(proof['whole_DSL_mathematical_equivalence_or_live_compiler_host_certified'])

    def test_02_only_exact_public_initialization_can_repeat(self):
        receipt,raw,expected,value=built();proof=check(receipt,raw,expected,value)
        self.assertEqual(proof['original_instructions'],6);self.assertEqual(proof['derived_instructions'],7)
        self.assertEqual(proof['public_encode_materializations'],2)
        self.assertEqual([u['replicated_initial_values']for u in value['units']],[['3'],['3']])
        self.assertTrue(all(u['actually_compiled_as_separate_unit']is False for u in value['units']))

    def test_03_auditor_does_not_call_partition_constructor(self):
        receipt,raw,expected,value=built()
        with patch.object(derivation,'partition',side_effect=AssertionError('Auditor must inspect source directly')):
            proof=check(receipt,raw,expected,value)
        self.assertFalse(proof['actual_input_same_key_final_output_or_CKKS_or_first_delivery_certified'])

    def test_04_wrong_arithmetic_operand_or_attribute_rejected(self):
        for kind in ['opcode','operand','attribute','order']:
            receipt,raw,expected,value=built();execution=value['units'][1]['plan']['execution']
            if kind=='opcode':execution[0]['op']='mul_cc'
            elif kind=='operand':execution[0]['inputs'].reverse()
            elif kind=='attribute':execution[0]['attrs']=dict(reuse_input0=True)
            else:execution.reverse()
            changed=resign_unit(value,1)
            with self.assertRaises(ValueError):check(receipt,raw,expected,changed)

    def test_05_nominal_scale_level_context_and_header_preserved(self):
        for kind in ['level','scale_log2','context','header']:
            receipt,raw,expected,value=built();plan=value['units'][0]['plan']
            if kind=='header':plan['target']['target_id']='retagged'
            elif kind=='context':plan['values'][0][kind]='other'
            else:plan['values'][0][kind]+=1
            changed=resign_unit(value,0)
            with self.assertRaises(ValueError):check(receipt,raw,expected,changed)

    def test_06_missing_residual_and_extra_final_outputs_rejected(self):
        for kind in ['missing','extra']:
            receipt,raw,expected,value=built();plan=value['units'][0]['plan']
            if kind=='missing':plan['final_outputs'].pop(0)
            else:plan['final_outputs'].append(plan['external_inputs'][0])
            changed=resign_unit(value,0)
            with self.assertRaises(ValueError):check(receipt,raw,expected,changed)

    def test_07_bijection_and_frozen_original_source_required(self):
        receipt,raw,expected,value=built();unit=value['units'][0];keys=list(unit['local_to_whole']);unit['local_to_whole'][keys[1]]=unit['local_to_whole'][keys[0]]
        changed=resign_unit(value,0)
        with self.assertRaises(ValueError):check(receipt,raw,expected,changed)
        receipt,raw,expected,value=built();unit=value['units'][0];new_id=str(len(unit['plan']['values']))
        unit['plan']['values'].append(dict(descriptor(8),id=new_id));unit['local_to_whole'][new_id]='8'
        changed=resign_unit(value,0)
        with self.assertRaises(ValueError):check(receipt,raw,expected,changed)
        receipt,raw,expected,value=built();expected['source_sha256']='0'*64
        with self.assertRaises(ValueError):check(receipt,raw,expected,value)

    def test_08_public_payload_signed_zero_and_foreign_encode_rejected(self):
        receipt,raw,expected,value=built();value['units'][1]['plan']['initialization'][0]['payload']['values'][1]=0.0
        changed=resign_unit(value,1)
        with self.assertRaises(ValueError):check(receipt,raw,expected,changed)
        receipt,raw,expected,value=built();value['units'][1]['plan']['initialization'][0]['payload']['values'][0]=.5
        changed=resign_unit(value,1)
        with self.assertRaises(ValueError):check(receipt,raw,expected,changed)

    def test_09_streaming_encode_stays_in_execution_and_cannot_cross_cut(self):
        receipt,raw,expected=fixture(streaming=True)
        value=derivation.partition(receipt,raw,[],expected)
        self.assertEqual(value['units'][0]['plan']['initialization'],[])
        self.assertEqual(value['units'][0]['plan']['execution'][0]['kind'],'encode')
        with self.assertRaises(derivation.PartitionUnsupported):derivation.partition(receipt,raw,[2],expected)

    def test_10_no_omitted_or_repeated_original_instruction_spans(self):
        for kind in ['omit','duplicate','cut']:
            receipt,raw,expected,value=built()
            if kind=='omit':value['units'].pop()
            elif kind=='duplicate':value['units'].append(copy.deepcopy(value['units'][0]))
            else:value['cuts']=[1,2]
            value=rebound(value)
            with self.assertRaises(ValueError):check(receipt,raw,expected,value)

    def test_11_original_native_limits_include_replicated_encodes(self):
        receipt,raw,expected=fixture(chain=4095)
        with self.assertRaises(ValueError):derivation.partition(receipt,raw,[2047],expected)
        receipt,raw,expected=fixture(chain=20)
        with self.assertRaises(ValueError):derivation.partition(receipt,raw,list(range(1,18)),expected)

    def test_12_no_first_output_or_fake_unit_compilation_credit(self):
        receipt,raw,expected,value=built();before=raw
        for key in ['numerical_feedback_used','first_delivery_or_mathematical_or_CKKS_success_certified']:
            changed=copy.deepcopy(value);changed[key]=True;changed=rebound(changed)
            with self.assertRaises(ValueError):check(receipt,raw,expected,changed)
        changed=copy.deepcopy(value);changed['units'][0]['actually_compiled_as_separate_unit']=True;changed=resign_unit(changed,0)
        with self.assertRaises(ValueError):check(receipt,raw,expected,changed)
        self.assertEqual(raw,before)


if __name__=='__main__':unittest.main()
