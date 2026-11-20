"""Read one frozen real compiler package without opening numerical output data.

The returned receipt proves data/SSA/layout correspondence only. Live compiler,
crypto input/output, host/resource and numerical eligibility remain external.
"""
import copy
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from candidate_contract import strict_json
import compiler_plan_provenance as lineage

FORMAT = 'poseidon-retained-compiler-program-provenance-v1'
MAX_FILE = 16*1024**2
MAX_TOTAL = 4*1024**3
EXPORTER_SOURCES = {
    '/home/lhohy/Code/Poseidon/third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/EmitRuntimePlan.cpp':
        '45949e654973933eb6f1b2888d46162380cdd04e1a71e08d01a3780885c689a6',
    '/home/lhohy/Code/Poseidon/third_party/ckks-runtime/third_party/dacapo/lib/Dialect/CKKS/Transforms/RuntimePlanUtils.cpp':
        'e767de710760b0740bee58965d3bded3ecac85b5db17d16d1ae033d5e90bdc6d'}


def source_manifest():
    import poseidon_runtime_compile
    import native_execution_profile
    import compiler_configuration
    import lossless_host_bound_delivery
    import candidate_contract
    import component_contract
    import benchmark_graph
    import deepseek_provider
    import unified_output_aliases
    import delivery_component
    paths = [Path(__file__).absolute(), *[Path(m.__file__).absolute() for m in
        (lineage,poseidon_runtime_compile,native_execution_profile,compiler_configuration,lossless_host_bound_delivery,
         candidate_contract,component_contract,benchmark_graph,deepseek_provider,unified_output_aliases,delivery_component)]]
    return {str(p):sha(read_bytes(p))for p in paths}


def exporter_sources():
    require(all(sha(read_bytes(p))==value for p,value in EXPORTER_SOURCES.items()),'Audited current exporter source changed; versioned review required')
    return dict(EXPORTER_SOURCES)


def project_json(raw, paths):
    """Select metadata paths while lexically skipping all other numeric bodies.

    This validates JSON delimiters, keys and number syntax; it never converts
    ignored numerical cells into Python int/float objects.
    """
    require(type(raw)is bytes and len(raw)<=MAX_FILE,'Original JSON metadata bound')
    text = raw.decode('utf-8'); wanted = {tuple(p)for p in paths}; selected = {}; length = len(text)
    number = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
    def ws(i):
        while i<length and text[i]in ' \t\r\n':i+=1
        return i
    def string(i):
        require(i<length and text[i]=='"','JSON string')
        return json.decoder.scanstring(text,i+1,True)
    def visit(i,path,depth):
        require(depth<=128,'Finite JSON metadata depth');i=ws(i);start=i;require(i<length,'Complete JSON value')
        char=text[i]
        if char=='{':
            keys=set();i=ws(i+1)
            if i<length and text[i]=='}':i+=1
            else:
                while True:
                    key,i=string(i);require(key not in keys,'Unique JSON metadata keys');keys.add(key);i=ws(i)
                    require(i<length and text[i]==':','JSON object colon');i=visit(i+1,path+(key,),depth+1);i=ws(i)
                    require(i<length and text[i]in ',}','JSON object delimiter')
                    if text[i]=='}':i+=1;break
                    i=ws(i+1)
        elif char=='[':
            i=ws(i+1);index=0
            if i<length and text[i]==']':i+=1
            else:
                while True:
                    i=visit(i,path+(index,),depth+1);index+=1;i=ws(i)
                    require(i<length and text[i]in ',]','JSON array delimiter')
                    if text[i]==']':i+=1;break
                    i=ws(i+1)
        elif char=='"':_,i=string(i)
        else:
            literal=next((v for v in ('true','false','null')if text.startswith(v,i)),None)
            if literal is not None:i+=len(literal)
            else:
                match=number.match(text,i);require(match is not None,'Original finite JSON literal syntax');i=match.end()
        if path in wanted:selected[path]=strict_json(text[start:i])
        return i
    end=visit(0,(),0);require(ws(end)==length,'No trailing JSON metadata')
    return selected


class Files:
    def __init__(self,root,inventory,*,maximum_total=MAX_TOTAL):
        require(type(maximum_total)is int and 1<=maximum_total<=MAX_TOTAL,'Original finite case byte ceiling')
        self.root=Path(root).absolute();require(self.root.resolve()==self.root and self.root.is_dir(),'Direct compiler evidence root')
        self.inventory=inventory;self.seen={};self.total=0;self.maximum_total=maximum_total
    def read(self,name):
        require(type(name)is str and '\\'not in name,'Relative compiler artifact')
        relative=PurePosixPath(name)
        require(str(relative)==name and not relative.is_absolute()and '..'not in relative.parts and name in self.inventory,'Original artifact inventory member')
        path=self.root/name;require(path.suffix in ('.py','.json','.mlir','.bin')and path.name!='.env','Compiler data only; no numerical execution files')
        require(all(not p.is_symlink()for p in (path,*path.parents)),'No artifact symlink')
        info=self.inventory[name]
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        signature_of=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        try:
            before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_size<=MAX_FILE,'Bounded regular compiler file')
            with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(MAX_FILE+1)
            after=os.fstat(fd)
        finally:os.close(fd)
        signature=signature_of(before)
        require(signature==signature_of(after)==signature_of(path.stat())and len(raw)==before.st_size,'Stable original compiler artifact read')
        require(type(info)is dict and sha(raw)==info['sha256']and len(raw)==info['bytes'],'Original compiler file bytes')
        if name not in self.seen:
            require(all(old[0][:2]!=signature[:2]for old in self.seen.values()),'No compiler artifact hardlink alias')
            self.total+=len(raw);require(self.total<=self.maximum_total,'Finite total compiler metadata/payload read')
        else:require(self.seen[name]==(signature,sha(raw)),'Compiler data changed during read')
        self.seen[name]=(signature,sha(raw));return raw
    def verify(self):
        for name,(old,digest)in self.seen.items():
            path=self.root/name;st=path.stat();require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==old,'Compiler artifact replaced during read')
            require(sha(read_bytes(path,MAX_FILE))==digest,'Compiler artifact bytes changed during read')


def payload_reader(files):
    def get(plan,op,which):
        payload=op['payload'];require(type(payload)is dict,'Original Encode payload')
        if payload.get('kind')=='inline':
            require(set(payload)=={'kind','values'}and type(payload['values'])is list and 1<=len(payload['values'])<=16384,'Finite inline public payload')
            require(all(type(v)in(int,float)and math.isfinite(v)for v in payload['values']),'Finite inline public values')
            return struct.pack('<'+'d'*len(payload['values']),*payload['values'])
        require(set(payload)=={'kind','content'}and payload['kind']=='bundle'and re.fullmatch(r'sha256:[0-9a-f]{64}',payload['content']),'Original files-bundle reference')
        base='attempt-00/output/'+('lowered._hecate_golden.bundle/'if which=='raw'else'bound/bundle/')
        manifest_raw=files.read(base+'manifest.json');manifest=strict_json(manifest_raw.decode())
        pointer=plan['plaintext_bundle'];require('sha256:'+sha(manifest_raw)==pointer['manifest_sha256']and manifest['bundle_format_version']==1
            and manifest['bundle_id']==pointer['id']and manifest['version']==pointer['version'],'Original bundle identity')
        rows=[r for r in manifest['blobs']if r['content']==payload['content']];require(len(rows)==1,'One exact original bundle entry')
        raw=files.read(base+'data/'+payload['content'][7:]+'.bin')
        require(len(raw)==rows[0]['byte_length']and len(raw)%8==0 and 0<len(raw)<=16384*8 and 'sha256:'+sha(raw)==payload['content'],'Original public bundle byte content')
        return raw
    return get


def _entry(files,request,observed_command,reported):
    import native_execution_profile
    import poseidon_runtime_compile as original
    config=request['compiler_configuration'];frontend=files.read('attempt-00/output/candidate_trace.mlir').decode('utf-8')
    if config.get('schema')==6:expected=native_execution_profile.command(config)
    else:expected=original.command(config['waterline'],entry_policy=config.get('runtime_plan_entry_policy'),serialization_policy=config.get('runtime_plan_serialization_policy'))
    require(observed_command==expected,'Observed compiler command differs from original frozen request configuration')
    if config.get('schema')not in (3,4,5,6):
        require(observed_command[1]=='/out/candidate_trace.mlir'and reported is None,'Historical direct frontend input');return frontend
    selected_source=frontend;serialization=None
    if config['schema']in(4,5,6):
        selected_source,serialization=original.compact_f64_trace(frontend)
        require(files.read('attempt-00/output/candidate_trace.compact.mlir')==selected_source.encode(),'Original bit-preserving compact frontend')
    selected,record=original.entrypoint_trace(selected_source)
    if serialization is not None:record['trace_serialization']=serialization
    if config['schema']==6:record['native_compiler_configuration_sha256']=config['identity_sha256']
    actual=strict_json(files.read('attempt-00/output/entry-selection.json').decode())
    require(actual==record and reported==record and files.read('attempt-00/output/candidate_trace.entry.mlir')==selected.encode(),
            'Only the original generic entry/serialization transformation is allowed')
    require(observed_command[1]=='/out/candidate_trace.entry.mlir','Actual selected compiler input path')
    return selected


def inspect(proof_description, request, source, *, maximum_total_read_bytes=MAX_TOTAL):
    from first_output_backend import VERSION as PROOF
    from deepseek_provider import public_request
    import lossless_host_bound_delivery
    require(type(source)is bytes and bool(source),'Original frozen DSL bytes')
    require(type(proof_description)is dict and set(proof_description)=={'path','sha256'},'Independently frozen compiler proof descriptor')
    proof_raw=read_bytes(proof_description['path'],MAX_FILE);require(sha(proof_raw)==proof_description['sha256'],'Original compiler proof bytes')
    proof=checked_bound(strict_json(proof_raw.decode()))
    require(proof.get('format')==PROOF and proof.get('request_id')==request['request_id']and proof.get('source_sha256')==sha(source)
        and proof.get('candidate_preserved')is True and proof.get('compiler_input_bound')is True and proof.get('real_compilation_observed')is True,'Original source/compiler proof role')
    files=Files(proof['evidence'],proof['files'],maximum_total=maximum_total_read_bytes)
    exporter=exporter_sources()
    stored=lossless_host_bound_delivery._decode_json(files.read('request.json'))
    require(canonical(public_request(stored))==canonical(public_request(request))and files.read('attempt-00/candidate.py')==source,'Exact original public request and unchanged candidate')
    payload=lossless_host_bound_delivery._decode_json(files.read('attempt-00/trace-payload.json'))
    require(type(payload)is dict and set(payload)=={'request','candidate'}and payload['candidate']==dict(schema=1,request_id=request['request_id'],hecate_source=source.decode())
        and canonical(public_request(payload['request']))==canonical(public_request(request)),'Exact DSL at the original frontend entrance')
    trace=strict_json(files.read('attempt-00/output/trace-evidence.json').decode())
    require(trace.get('request_id')==request['request_id']and trace.get('frontend')=='real_Hecate'and trace.get('candidate_python_executed')is False,'Original trusted Hecate frontend data')
    paths=[('attempts',0,k)for k in ('index','checked','source_parsed','compiled','compile_command','runtime_plan_entry')]
    paths.extend([('attempts',1),('backend',),('request_id',)])
    metadata=project_json(files.read('report.json'),paths)
    require(('attempts',1)not in metadata and metadata.get(('backend',))=='poseidon_cpu'and metadata.get(('request_id',))==request['request_id']
        and metadata.get(('attempts',0,'index'))==0 and all(metadata.get(('attempts',0,k))is True for k in ('checked','source_parsed','compiled')),'One original actual compiler attempt')
    command=metadata[('attempts',0,'compile_command')];require(proof['compiler_input']==command[1],'Original proof/compiler input path')
    selected=_entry(files,stored,command,metadata.get(('attempts',0,'runtime_plan_entry')))
    lowered=files.read('attempt-00/output/lowered.ckks.mlir').decode()
    raw=strict_json(files.read('attempt-00/output/lowered._hecate_golden.runtime-plan.json').decode())
    packed=strict_json(files.read('attempt-00/output/bound/plan.json').decode())
    get_payload=payload_reader(files);dag=lineage.audit_dag(lowered,raw,command,get_payload)
    periodic=lineage.audit_periodic_binding(raw,packed,stored['layout']['input_slot_period'],get_payload)
    layout=lineage.layout_mapping(stored,source.decode(),selected,dag)
    files.verify()
    require(sha(read_bytes(proof_description['path'],MAX_FILE))==proof_description['sha256'],'Original compiler descriptor changed during audit')
    return bound(dict(format=FORMAT,original_proof_binding=proof['binding'],request_id=request['request_id'],source_sha256=sha(source),
        compiler_input=proof['compiler_input'],original_compiler_sha256=proof['compiler_sha256'],
        original_backend_identity_sha256=proof['backend_identity_sha256'],DAG=dag,periodic_binding=periodic,logical_IO=layout,
        audited_current_exporter_source=exporter,retained_compiler_binary_source_origin_recertified=False,
        files_read={name:dict(sha256=v[1],bytes=v[0][2])for name,v in files.seen.items()},unique_bytes_read=files.total,
        numerical_output_or_reference_files_opened=False,numerical_output_or_reference_values_decoded=0,
        source_or_reference_or_program_modified=False,candidate_Python_executed=False,compiler_executed=False,
        source_model_math_equivalence_or_live_compiler_host_execution_certified=False,
        actual_raw_to_encrypted_input_or_same_key_final_output_or_CKKS_certified=False,
        complex_program_eligibility_or_first_delivery_success_granted=False))
