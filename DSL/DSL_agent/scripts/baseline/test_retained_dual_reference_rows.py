import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from delivery_component import bound,canonical,sha
import retained_dual_reference_rows as rows


class Fixture:
    """Synthetic data/byte fixtures only; no model, DSL or backend execution."""
    def __init__(self,root,*,width=2,outputs=2,offset_delta=0,bad_row_hash=False,swap_roles=False,
                 bad_record_hash=False,nonfinite_reference=False,nonfinite_input=False,source_dtype='<f4',fortran=False):
        self.root=root;root.mkdir();self.width=width;self.output_count=outputs
        def file(name,data):
            p=root/name;p.write_bytes(data);return dict(path=str(p),sha256=sha(data),bytes=len(data))
        def document(name,data):
            value=bound(data);descriptor=file(name,canonical(value)+b'\n');descriptor['binding']=value['binding'];return descriptor,value
        self.file=file;self.document=document
        self.provenance={
            'model_source':file('model.py',b'raise AssertionError("model source must stay opaque")\n'),
            'full_ModelSpec':file('declared-model.json',b'{"opaque_model":true}\n'),
            'original_weights':file('weights.bin',b'opaque-public-weights'),
            'approximation_definition':file('approximation.json',b'{"opaque_approximation":true}\n')}
        self.decoder=file('decoder.py',b'raise AssertionError("decoder campaign must not execute")\n')
        source=np.asarray([[1/3]+[-0.0]*(outputs-1),[1e-20]+[2.]*(outputs-1)],dtype=source_dtype)
        declared=np.asarray([[0.35]+[-0.0]*(outputs-1),[5.]+[7.]*(outputs-1)],dtype='<f8')
        if nonfinite_reference:declared[0,0]=float('nan')
        def array(name,value):
            p=root/name;np.save(p,np.asfortranarray(value)if fortran else value,allow_pickle=False)
            raw=p.read_bytes();start=10 if tuple(raw[6:8])==(1,0)else 12;offset=start+int.from_bytes(raw[8:start],'little')
            return dict(path=str(p),sha256=sha(raw),bytes=len(raw),dtype=value.dtype.str,item_bytes=value.dtype.itemsize,shape=list(value.shape),data_offset=offset)
        arrays={'source32':array('source32.npy',source),'declared64':array('declared64.npy',declared)}
        self.arrays=arrays;cases=[];links=[];base_receipts=[];self.inputs={}
        for number in range(2):
            case_id='case-'+str(number);raw=bytes(range(1+number*width,1+(number+1)*width))
            values=[(.25+i*.125 if number==0 else -.5+i*.125)for i in range(width)]
            if nonfinite_input and number==0:values[0]=float('nan')
            wire=b''.join(struct.pack('<d',v)for v in values);self.inputs[case_id]=(raw,wire)
            refs={}
            role_keys={'original_float32':'source32','approximation':'declared64'}
            if swap_roles:role_keys={k:('declared64'if v=='source32'else'source32')for k,v in role_keys.items()}
            for role,key in role_keys.items():
                desc=arrays[key];count=outputs*desc['item_bytes'];offset=desc['data_offset']+number*count
                payload=Path(desc['path']).read_bytes()[offset:offset+count]
                refs[role]=dict(source_key=key,dtype=desc['dtype'],row=number,file_byte_offset=offset+offset_delta,bytes=count,
                    row_sha256='f'*64 if bad_row_hash else sha(payload))
            record=sha(bytes([number])+raw)
            case=dict(case_id=case_id,kind='labelled',label=number,source_index=number,
                logical_input_shape=[1,1,1,width],normalized_logical_f64_bytes=8*width,normalized_logical_f64_sha256=sha(wire),
                input=dict(record_sha256='d'*64 if bad_record_hash else record),references=refs)
            source_input=sha(b'fixture-only-reconstructed-input'+raw)
            link=dict(case_id=case_id,kind='labelled',label=number,original_case_source_index=number,input_positions=width,
                normalized_logical_float64_sha256=sha(wire),raw_CHW_uint8_sha256=sha(raw),
                reconstructed_original_preprocessing_float32_sha256=source_input,retained_original_and_declared_reference_rows=refs)
            receipt=dict(format='poseidon-retained-original-declared-row-receipt-v1',case_id=case_id,kind='labelled',label=number,source_index=number,
                declared_input_sha256=sha(wire),raw_input_sha256=sha(raw),original_source_input_sha256=source_input,references=refs,
                original_source_input_is_reconstruction_not_historical_tensor=True,historical_producers_verified_as_bytes=True,
                reader_executes_model_or_preprocessing=False,accepted_A_native_source_report=False,first_output_or_application_certified=False)
            cases.append(case);links.append(link);base_receipts.append(receipt)
        public=dict(self.provenance,**arrays)
        self.original,self.original_value=document('original.json',dict(format='synthetic-source-manifest-v1',case_count=2,cases=cases,public_files=public))
        producers={k:file(k+'.json',b'{"fixture_only":true}\n')for k in ('original_accuracy_producer','original_probe_producer','original_frozen_accuracy_plan','original_frozen_probe_plan')}
        self.relation,self.relation_value=document('relation.json',dict(format='synthetic-input-relation-v1',cases=links,
            source_graph_weights_and_approximation=self.provenance,original_historical_input_tensor_buffers_or_complete_model_not_reobserved=True,
            native_source_full_model_adapter_certified=False,**producers))
        policy=dict(reference_roles=copy.deepcopy(rows.ROLES),input_shape=[1,1,1,width],output_values=outputs,
            sample_scope='retained_fixed_cases_only_not_a_new_application_report')
        self.contract,self.contract_value=document('contract.json',dict(format='poseidon-retained-original-declared-row-reader-v1',original_contract=self.original,input_relation=self.relation,policy=policy))
        receipts=[bound(dict(value,reader_contract_binding=self.contract['binding'],original_contract=self.original,input_relation=self.relation))for value in base_receipts]
        self.index,self.index_value=document('index.json',dict(format='synthetic-retained-index-v1',reader_contract=self.contract,decoder_source_sha256=self.decoder['sha256'],
            accepted_A_native_source_report=False,actual_first_DSL_present=False,first_original_passes=0,reference_bytes_roundtrip_mismatches=0,
            case_count=2,reference_rows=4,cases=receipts))
        self.target=dict(request_id='a'*64,kind='metadata_integration',declared_model_file_sha256=self.provenance['full_ModelSpec']['sha256'],
            input_shape=[1,1,1,width],output_values=outputs)
    def freeze(self):return rows.freeze(self.contract,self.index,self.decoder,self.target)
    def provider(self):return rows.Provider(self.freeze())
    def resolve(self,provider,case='case-0',target=None):
        raw,declared=self.inputs[case];return provider.resolve(case,raw,declared,target=target or self.target)


class RetainedRowControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.count=0
    def fixture(self,**kwargs):
        self.count+=1;return Fixture(self.root/str(self.count),**kwargs)

    def test_distinct_reference_roles_and_exact_bytes_survive_roundtrip(self):
        f=self.fixture();provider=f.provider();result=f.resolve(provider)
        self.assertNotEqual(result['original_reference'],result['declared_reference'])
        self.assertEqual(result['reference_byte_identities']['original_float32']['dtype'],'<f4')
        self.assertEqual(result['reference_byte_identities']['approximation']['dtype'],'<f8')
        self.assertEqual(struct.pack('<f',result['original_reference'][1]),struct.pack('<f',-0.0))
        self.assertEqual(provider.reference_rows_decoded,2);self.assertFalse(result['native_source_worker_report'])

    def test_raw_and_declared_inputs_must_belong_to_the_same_case(self):
        f=self.fixture();provider=f.provider();a,b=f.inputs['case-0'],f.inputs['case-1']
        for raw,declared in [(a[0],b[1]),(b[0],a[1]),(bytearray(a[0]),a[1])]:
            with self.subTest(raw_type=type(raw)),self.assertRaises(ValueError):provider.resolve('case-0',raw,declared,target=f.target)
        self.assertEqual(provider.reference_rows_decoded,0)

    def test_other_target_and_unknown_case_are_rejected(self):
        f=self.fixture();provider=f.provider()
        for key,value in [('request_id','b'*64),('declared_model_file_sha256','c'*64),('output_values',3)]:
            with self.subTest(key=key),self.assertRaises(ValueError):f.resolve(provider,target=dict(f.target,**{key:value}))
        with self.assertRaises(ValueError):provider.resolve('unknown',b'',b'',target=f.target)

    def test_original_label_record_identity_is_required(self):
        f=self.fixture(bad_record_hash=True);provider=f.provider()
        with self.assertRaises(ValueError):f.resolve(provider)
        self.assertEqual(provider.reference_rows_decoded,0)

    def test_original_and_declared_dtypes_cannot_be_swapped(self):
        f=self.fixture(swap_roles=True)
        with self.assertRaises(ValueError):f.freeze()

    def test_row_extent_and_row_digest_are_both_checked(self):
        f=self.fixture(offset_delta=1)
        with self.assertRaises(ValueError):f.freeze()
        f=self.fixture(bad_row_hash=True);provider=f.provider()
        with self.assertRaises(ValueError):f.resolve(provider)

    def test_snapshot_tampering_after_freeze_or_after_open_is_rejected(self):
        f=self.fixture();selection=f.freeze();p=Path(f.arrays['declared64']['path']);data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
        with self.assertRaises(ValueError):rows.Provider(selection)
        f=self.fixture();provider=f.provider();p=Path(f.arrays['source32']['path']);p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaises(ValueError):f.resolve(provider)

    def test_selection_cannot_promote_historical_rows_or_change_assets(self):
        f=self.fixture();selection=f.freeze()
        for key,value in [('native_source_worker_report',True),('source_input_reconstruction_is_historical_tensor_observation',True)]:
            changed={k:v for k,v in selection.items()if k!='binding'};changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):rows.Provider(bound(changed))
        changed=copy.deepcopy(selection);changed.pop('binding');next(iter(changed['assets'].values()))['sha256']='e'*64
        with self.assertRaises(ValueError):rows.Provider(bound(changed))

    def test_numpy_endianness_and_fortran_storage_are_rejected(self):
        for options in [dict(source_dtype='>f4'),dict(fortran=True)]:
            f=self.fixture(**options)
            with self.subTest(options=options),self.assertRaises(ValueError):f.freeze()

    def test_nonfinite_reference_and_declared_input_are_rejected(self):
        for options in [dict(nonfinite_reference=True),dict(nonfinite_input=True)]:
            f=self.fixture(**options);provider=f.provider()
            with self.subTest(options=options),self.assertRaises(ValueError):f.resolve(provider)

    def test_different_shapes_and_output_widths_keep_distinct_roles(self):
        for width,outputs in [(2,2),(3,3),(4,1)]:
            f=self.fixture(width=width,outputs=outputs);provider=f.provider();result=f.resolve(provider)
            self.assertEqual(len(result['original_reference']),outputs);self.assertEqual(len(result['declared_reference']),outputs)
            self.assertFalse(result['independent_DSL_mathematics_verified']);self.assertFalse(result['actual_CKKS_verified'])

    def test_source_is_opaque_and_private_credentials_are_never_read(self):
        f=self.fixture();provider=f.provider();result=f.resolve(provider)
        self.assertFalse(result['model_or_preprocessing_or_DSL_executed']);self.assertFalse(result['first_output_success_certified'])
        p=Path(f.provenance['original_weights']['path']);p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaises(ValueError):provider.verify_sources()
        forbidden=self.root/'.env'
        with patch.object(rows,'read_bytes',side_effect=AssertionError('credential read forbidden'))as read:
            with self.assertRaises(ValueError):rows._Files().read(dict(path=str(forbidden),sha256='a'*64))
            read.assert_not_called()


if __name__=='__main__':unittest.main()
