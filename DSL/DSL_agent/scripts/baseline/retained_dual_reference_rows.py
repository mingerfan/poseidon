"""Generic, private evaluation provider for frozen source/declared reference rows.

Reuse an audited retained index without running its full case campaign again.
Model source and weights are opaque bytes. No decoder, model, preprocessing or
candidate code is executed. This is neither a native_source worker report nor
independent DSL mathematics or a numerical/whole-model certificate.
"""
import ast
import copy
import json
import math
from pathlib import Path
import struct

from delivery_component import bound,canonical,checked_bound,read_bytes,require,sha
import delivery_component

FORMAT='poseidon-retained-source-declared-row-provider-v1'
RECORD='poseidon-retained-source-declared-reference-row-v1'
ROLES={'original_float32':'<f4','approximation':'<f8'}
PROVENANCE={'model_source','full_ModelSpec','original_weights','approximation_definition'}
MAX_FILE=16*1024**2
MAX_TOTAL=32*1024**2


def _json(raw):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'Duplicate retained metadata key');value[key]=item
        return value
    def invalid(_):raise ValueError('Nonfinite retained metadata')
    result=json.loads(raw.decode(),object_pairs_hook=pairs,parse_constant=invalid)
    require(type(result)is dict,'Retained metadata object required')
    return result


def _hex(value):
    return type(value)is str and len(value)==64 and all(c in'0123456789abcdef'for c in value)


class _Files:
    def __init__(self):self.data={};self.assets={};self.signatures={};self.total=0
    def read(self,descriptor):
        require(type(descriptor)is dict and type(descriptor.get('path'))is str and _hex(descriptor.get('sha256')),'Frozen retained file descriptor')
        path=Path(descriptor['path']);require(path.is_absolute()and path.suffix in('.json','.py','.npy','.th','.pth','.bin','.records','.txt')and path.name!='.env','Retained public artifact path')
        require(path.resolve()==path and path.is_file()and not path.is_symlink(),'Indirect retained artifact')
        key=str(path)
        if key not in self.data:
            before=path.stat();signature=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
            value=read_bytes(path,MAX_FILE);after=path.stat()
            require(signature==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Retained artifact changed during read')
            require(sha(value)==descriptor['sha256'],'Retained artifact bytes changed')
            self.total+=len(value);require(self.total<=MAX_TOTAL,'Retained snapshot byte limit')
            self.data[key]=value;self.assets[key]=dict(sha256=sha(value),bytes=len(value));self.signatures[key]=signature
        value=self.data[key];require(sha(value)==descriptor['sha256'],'Conflicting retained file descriptor')
        if 'bytes'in descriptor:require(type(descriptor['bytes'])is int and descriptor['bytes']==len(value),'Retained file size')
        return value
    def document(self,descriptor):
        value=_json(self.read(descriptor));checked_bound(value)
        if 'binding'in descriptor:require(value['binding']==descriptor['binding'],'Retained parent binding changed')
        return value
    def verify(self):
        for path,info in self.assets.items():
            st=Path(path).stat();require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==self.signatures[path]
                and sha(read_bytes(path,MAX_FILE))==info['sha256'],'Retained snapshot changed after read')


def _numpy_layout(raw,descriptor):
    require(raw[:6]==b'\x93NUMPY'and len(raw)>=10,'Retained NPY magic')
    version=tuple(raw[6:8]);require(version in((1,0),(2,0)),'Retained NPY version')
    start=10 if version==(1,0)else 12
    size=int.from_bytes(raw[8:start],'little');require(0<size<=16384 and start+size<=len(raw),'Retained NPY header extent')
    try:header=ast.literal_eval(raw[start:start+size].decode('latin1'))
    except (ValueError,TypeError,SyntaxError):raise ValueError('Retained NPY header parse')from None
    require(type(header)is dict and set(header)=={'descr','fortran_order','shape'}and header['fortran_order']is False,'Retained NPY storage')
    shape=header['shape'];dtype=header['descr']
    require(type(shape)is tuple and len(shape)==2 and all(type(n)is int and n>0 for n in shape)and math.prod(shape)<=2**24,'Retained reference shape')
    require(dtype in ROLES.values(),'Retained floating dtype')
    item=4 if dtype=='<f4'else 8;offset=start+size
    require(descriptor.get('dtype')==dtype and descriptor.get('shape')==list(shape)and descriptor.get('item_bytes')==item
        and descriptor.get('data_offset')==offset and len(raw)==offset+math.prod(shape)*item,'Retained NPY descriptor binding')
    return dict(shape=shape,dtype=dtype,item_bytes=item,data_offset=offset)


def _metadata(reader_contract,index,decoder_source,target):
    files=_Files();contract=files.document(reader_contract);lookup=files.document(index)
    require(contract.get('format')=='poseidon-retained-original-declared-row-reader-v1','Retained reader contract')
    require(lookup.get('reader_contract')==reader_contract and lookup.get('decoder_source_sha256')==decoder_source['sha256'],'Retained index/decoder identity')
    require(lookup.get('accepted_A_native_source_report')is False and lookup.get('actual_first_DSL_present')is False
        and lookup.get('first_original_passes')==0 and lookup.get('reference_bytes_roundtrip_mismatches')==0,'Historical row scope required')
    files.read(decoder_source)  # Opaque, never imported or executed.
    original=files.document(contract['original_contract']);relation=files.document(contract['input_relation'])
    policy=contract['policy'];require(policy.get('reference_roles')==ROLES,'Original/declared dtype roles')
    require(policy.get('sample_scope')=='retained_fixed_cases_only_not_a_new_application_report','Retained sample scope')
    require(type(target)is dict and set(target)=={'request_id','kind','declared_model_file_sha256','input_shape','output_values'}
        and _hex(target['request_id'])and target['kind']in('metadata_integration','frozen_generation_request')
        and _hex(target['declared_model_file_sha256']),'Explicit target identity')
    require(type(target['input_shape'])is list and all(type(n)is int and n>0 for n in target['input_shape'])
        and type(target['output_values'])is int and target['output_values']>0,'Target numeric types')
    require(target['input_shape']==policy['input_shape']and target['output_values']==policy['output_values'],'Reference target shape')
    require(type(policy['input_shape'])is list and bool(policy['input_shape'])and all(type(n)is int and n>0 for n in policy['input_shape'])
        and math.prod(policy['input_shape'])<=2**20 and type(policy['output_values'])is int and 1<=policy['output_values']<=2**20,'Reference shape budget')
    provenance=relation['source_graph_weights_and_approximation'];require(set(provenance)==PROVENANCE,'Complete source/declared provenance roles')
    for role,descriptor in provenance.items():
        parent=original['public_files'][role]
        require(all(parent[k]==descriptor[k]for k in('path','bytes','sha256')),'Source provenance parent differs');files.read(descriptor)
    require(target['declared_model_file_sha256']==provenance['full_ModelSpec']['sha256'],'Target declared model file differs')
    for role in('original_accuracy_producer','original_probe_producer','original_frozen_accuracy_plan','original_frozen_probe_plan'):
        files.read(relation[role])
    require(relation['original_historical_input_tensor_buffers_or_complete_model_not_reobserved']is True
        and relation['native_source_full_model_adapter_certified']is False,'Retained source observation scope')
    cases=original['cases'];indexed=lookup['cases'];links=relation['cases']
    count=original['case_count'];require(type(count)is int and 1<=count<=1024 and len(cases)==len(indexed)==len(links)==count
        and lookup['case_count']==count and lookup['reference_rows']==2*count,'Complete retained case index')
    rows={};arrays={}
    for case,receipt,link in zip(cases,indexed,links):
        checked_bound(receipt);case_id=case['case_id']
        require(type(case_id)is str and bool(case_id)and case_id not in rows,'Unique retained cases')
        require(receipt.get('format')=='poseidon-retained-original-declared-row-receipt-v1'
            and receipt['reader_contract_binding']==contract['binding']and receipt['original_contract']==contract['original_contract']
            and receipt['input_relation']==contract['input_relation'],'Retained row parent identities')
        require(all(receipt[k]==case[k]==link[k]for k in('case_id','kind','label'))
            and receipt['source_index']==case['source_index']==link['original_case_source_index'],'Retained case association')
        require(type(case['source_index'])is int and case['source_index']>=0 and case['logical_input_shape']==policy['input_shape'],'Retained input signature')
        require(receipt['declared_input_sha256']==case['normalized_logical_f64_sha256']==link['normalized_logical_float64_sha256']
            and receipt['raw_input_sha256']==link['raw_CHW_uint8_sha256']
            and receipt['original_source_input_sha256']==link['reconstructed_original_preprocessing_float32_sha256'],'Both input identities and source reconstruction')
        require(all(_hex(receipt[k])for k in('declared_input_sha256','raw_input_sha256','original_source_input_sha256')),'Input digest format')
        require(receipt['original_source_input_is_reconstruction_not_historical_tensor']is True
            and receipt['historical_producers_verified_as_bytes']is True and receipt['reader_executes_model_or_preprocessing']is False
            and receipt['accepted_A_native_source_report']is False and receipt['first_output_or_application_certified']is False,'Retained receipt scope')
        require(case['normalized_logical_f64_bytes']==8*math.prod(policy['input_shape'])and link['input_positions']==math.prod(policy['input_shape']),'Retained input byte count')
        require(case['references']==receipt['references']==link['retained_original_and_declared_reference_rows']and set(case['references'])==set(ROLES),'Retained reference role links')
        for role,row in case['references'].items():
            descriptor=original['public_files'][row['source_key']]
            if row['source_key']not in arrays:
                raw=files.read(descriptor);arrays[row['source_key']]=(raw,_numpy_layout(raw,descriptor))
            _,layout=arrays[row['source_key']];n=policy['output_values']
            require(layout['dtype']==ROLES[role]==row['dtype']and layout['shape'][1]==n,'Reference role cannot be substituted')
            require(type(row['row'])is int and row['row']==case['source_index']and 0<=row['row']<layout['shape'][0]
                and row['bytes']==n*layout['item_bytes']and row['file_byte_offset']==layout['data_offset']+row['row']*row['bytes']and _hex(row['row_sha256']),'Reference row extent')
        rows[case_id]=(case,receipt)
    return files,contract,lookup,rows,arrays


def freeze(reader_contract,index,decoder_source,target):
    """Freeze metadata/parents, with zero row decodes or reference forwards."""
    files,contract,lookup,rows,_=_metadata(reader_contract,index,decoder_source,target)
    files.verify()
    return bound(dict(format=FORMAT,reader_contract=copy.deepcopy(reader_contract),retained_index=copy.deepcopy(index),decoder_source=copy.deepcopy(decoder_source),
        target=copy.deepcopy(target),case_ids=list(rows),assets=files.assets,
        provider_source_sha256={str(p):sha(read_bytes(p))for p in [Path(__file__).absolute(),Path(delivery_component.__file__).absolute()]},reference_roles=copy.deepcopy(ROLES),
        private_evaluation_only=True,Agent_prompt_may_include_reference_rows=False,reference_origin='retained_source_and_declared_rows',
        native_source_worker_report=False,source_input_reconstruction_is_historical_tensor_observation=False,
        declared_model_function_binding_still_requires_external_policy=True,first_output_success_certified=False))


class Provider:
    def __init__(self,selection):
        checked_bound(selection);require(selection.get('format')==FORMAT,'Retained provider format')
        actual=freeze(selection['reader_contract'],selection['retained_index'],selection['decoder_source'],selection['target'])
        require(actual==selection,'Frozen retained provider/source/parent changed')
        self.selection=copy.deepcopy(selection)
        self.files,self.contract,self.index,self.rows,self.arrays=_metadata(selection['reader_contract'],selection['retained_index'],selection['decoder_source'],selection['target'])
        self.reference_rows_decoded=0

    def resolve(self,case_id,raw_input,declared_input,*,target):
        self.files.verify()
        require(target==self.selection['target'],'Retained row used for another target')
        require(type(case_id)is str and case_id in self.rows,'Unknown retained case')
        require(type(raw_input)is bytes and type(declared_input)is bytes,'Raw and declared input bytes required')
        case,receipt=self.rows[case_id];elements=math.prod(target['input_shape'])
        require(len(raw_input)==elements and sha(raw_input)==receipt['raw_input_sha256'],'Raw input differs from frozen case')
        require(len(declared_input)==8*elements and sha(declared_input)==receipt['declared_input_sha256'],'Declared input differs from frozen case')
        require(all(math.isfinite(v[0])for v in struct.iter_unpack('<d',declared_input)),'Nonfinite declared input')
        if case['label']is not None:
            require(type(case['label'])is int and 0<=case['label']<=255
                and sha(bytes([case['label']])+raw_input)==case['input']['record_sha256'],'Original labelled input record differs')
        decoded={};row_descriptors={}
        for role,row in case['references'].items():
            raw,_=self.arrays[row['source_key']];payload=raw[row['file_byte_offset']:row['file_byte_offset']+row['bytes']]
            require(len(payload)==row['bytes']and sha(payload)==row['row_sha256'],'Original reference row bytes changed')
            code='<f'if row['dtype']=='<f4'else'<d';values=[v[0]for v in struct.iter_unpack(code,payload)]
            require(all(math.isfinite(v)for v in values),'Nonfinite retained reference')
            require(b''.join(struct.pack(code,v)for v in values)==payload,'Reference dtype/bytes not preserved')
            decoded[role]=values;row_descriptors[role]=dict(dtype=row['dtype'],bytes=len(payload),sha256=sha(payload))
        self.files.verify();self.reference_rows_decoded+=2
        return bound(dict(format=RECORD,provider_selection_binding=self.selection['binding'],target=copy.deepcopy(target),case_id=case_id,
            raw_input_sha256=sha(raw_input),declared_input_sha256=sha(declared_input),original_reference=decoded['original_float32'],
            declared_reference=decoded['approximation'],reference_byte_identities=row_descriptors,retained_row_receipt=copy.deepcopy(receipt),
            reference_origin='retained_source_and_declared_rows',original_source_input_is_reconstruction_not_historical_tensor=True,
            model_or_preprocessing_or_DSL_executed=False,native_source_worker_report=False,independent_DSL_mathematics_verified=False,
            actual_CKKS_verified=False,application_accuracy_certified=False,first_output_success_certified=False))

    def verify_sources(self):
        self.files.verify();return bound(dict(provider_selection_binding=self.selection['binding'],files=len(self.files.assets),
            bytes=self.files.total,reference_rows_decoded=self.reference_rows_decoded,source_or_reference_values_changed=False))
