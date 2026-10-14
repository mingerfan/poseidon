"""Read-only HEVM scale/level lineage and compiler/runtime evidence.

A nominal scale budget is not a CKKS error bound or an acceptance policy. No
constant, artifact, request, threshold, compiler option or runtime is modified.
"""
import hashlib,json,math,struct
from pathlib import Path
from benchmark_graph import require,digest
from candidate_contract import request_input_names,request_rotations
from cipher_abi import artifact_options
from seal_artifact_gate import inspect_artifacts
from unified_graph_contract import validate_request

NAMES={0:'encode',1:'rotate',2:'negate',3:'rescale',4:'modswitch',6:'addcc',7:'addcp',8:'mulcc_relinearize',9:'mulcp',65535:'empty'}

def lineage(raw,cst,request):
    validate_request(request)
    from compiler_configuration import PROFILE_SHA256
    require(request['compiler_profile_sha256']==PROFILE_SHA256,'Nominal lineage requires fixed 60-bit profile')
    gate=inspect_artifacts(raw,cst,rotation_steps=request_rotations(request),expected_inputs=len(request_input_names(request)),**artifact_options(request['layout']))
    body_size=struct.unpack_from('<Q',raw,24)[0]
    operations=list(struct.iter_unpack('<4H',raw[24+body_size:]))
    cipher={i:dict(level=level,scale=scale,producer='arg'+str(i)) for i,(level,scale) in enumerate(zip(gate['arg_level'],gate['arg_scale']))}
    plains={};events=[];scale_overrides=[]
    for index,(opcode,dst,lhs,rhs) in enumerate(operations):
        if opcode==65535:continue
        event=dict(index=index,opcode=opcode,operation=NAMES[opcode],destination=dst)
        if opcode==0:
            out=dict(level=rhs>>10,scale=rhs&1023,producer=index)
            plains[dst]=out;event.update(constant_index=None if lhs==65535 else lhs,output=out);events.append(event);continue
        left=dict(cipher[lhs]);inputs=[left];out=dict(left,producer=index)
        if opcode in (6,7,8,9):
            right=dict(cipher[rhs] if opcode in (6,8) else plains[rhs]);inputs.append(right)
            if opcode in (6,7):
                # Stock runtime aligns the LHS scale metadata to RHS before add.
                if left['scale']!=right['scale']:scale_overrides.append(dict(index=index,before=left['scale'],after=right['scale']))
                cipher[lhs]=dict(cipher[lhs],scale=right['scale'])
                out['scale']=right['scale']
            else:out['scale']=left['scale']+right['scale']
        elif opcode==3:out.update(level=left['level']-1,scale=left['scale']-60)
        elif opcode==4:out['level']-=rhs;event['moduli_dropped']=rhs
        elif opcode==1:event['rotation_step']=rhs if rhs<32768 else rhs-65536
        require(out['level']>=1 and out['scale']>0,'Invalid nominal scale/level propagation')
        out['nominal_modulus_minus_scale_bits']=60*out['level']-out['scale']
        require(out['nominal_modulus_minus_scale_bits']>0,'Nominal scale exceeds modulus capacity')
        cipher[dst]=out;event.update(inputs=inputs,output=out);events.append(event)
    outputs=[dict(cipher[dst],ciphertext=i,register=dst) for i,dst in enumerate(gate['res_dst'])]
    require([x['level'] for x in outputs]==gate['res_level'] and [x['scale'] for x in outputs]==gate['res_scale'],'HEVM result scale lineage mismatch')
    return dict(schema=1,contract='nominal-hevm-scale-lineage-v1',hevm_sha256=hashlib.sha256(raw).hexdigest(),cst_sha256=hashlib.sha256(cst).hexdigest(),
                gate=gate,events=events,outputs=outputs,scale_metadata_overrides=scale_overrides,
                nominal_rescale_bits=60,exact_prime_rounding_modeled=False,error_bound_proven=False,
                automatically_changes_acceptance=False)


def audit(folder):
    from audit_unified_candidate import read,sha,verify_files
    from hecate_python_env import ROOT
    from python_compiler_smoke import BUILD
    from seal_cpu_golden import KEY_BUILD,PROFILE
    from compiler_configuration import PROFILE_SHA256,verify_artifact_configuration
    folder=Path(folder);report=read(folder/'report.json');request=read(folder/'request.json')
    require(report['agent_calls']==0 and not report['llm_generation_validated'],'Manual evidence required')
    verify_files(ROOT/'scripts/baseline',report['source_hashes']);verify_files(folder,report['frozen_hashes'])
    require(sha(PROFILE)==PROFILE_SHA256==request['compiler_profile_sha256'],'Compiler profile source')
    require(sha(BUILD/'lib/libSEAL_HEVM.so')==report['runtime_sha256'],'Runtime binary changed')
    require(sha(KEY_BUILD/'libseal_golden_metadata.so')==report['metadata_observer_sha256'],'Metadata observer changed')
    require(sha(KEY_BUILD/'libseal_packed_metadata.so')==report['packed_observer_sha256'],'Key observer changed')
    require(sha(KEY_BUILD/'seal_packed_keys')==report['packed_key_helper_sha256'],'Key generator changed')
    require(read(folder/'key-cleanup-outcome.json')['complete'],'Keys not cleaned')
    attempts=[a for a in report['attempts'] if a.get('compiled') and a.get('executed')]
    require(len(attempts)==1,'Exactly one executed attempt required')
    attempt=attempts[0];out=folder/('attempt-%02d'%attempt['index'])/'output'
    verify_files(out,attempt['artifact_hashes'])
    trace=lineage((out/'lowered._hecate_golden.hevm').read_bytes(),(out/'_hecate_golden.cst').read_bytes(),request)
    require(trace['gate']==attempt['artifact_gate'],'Stored artifact gate differs')
    verify_artifact_configuration(request,trace['gate'],sha(PROFILE))
    execution=read(out/'execution.json')
    require(execution==attempt['execution'] and execution['encrypted_execution'] is True and execution['bootstrap_executed'] is False,'Real encrypted execution required')
    require(str(BUILD/'lib/libSEAL_HEVM.so') in execution['mapped_libraries'],'Runtime was not loaded')
    params=report['parameters']
    require(params['parameters_set'] is True and params['security_check']=='tc128' and params['polynomial_degree']==32768 and params['slots']==16384 and params['modulus_bits']==[60]*14 and params['data_modulus_count']==13,'Frozen security parameters differ')
    require(execution['rotation_key_check']==dict(actual_key_file_verified=True,required_steps=trace['gate']['rotation_steps']),'Actual rotation key verification differs')
    observations=execution['ciphertext_metadata'];require(len(observations)==4,'Four runtime input batches required')
    for observation in observations:
        actual=observation['outputs'];require(len(actual)==len(trace['outputs']),'Output metadata arity')
        for item,expected in zip(actual,trace['outputs']):
            require(item['data_modulus_count']==expected['level'] and abs(item['log2_scale']-expected['scale'])<=1e-6 and item['polynomials']==2,'Runtime scale/level/size mismatch')
    cpp=ROOT/'third_party/dacapo/lib/Runtime/SEAL_HEVM.cpp'
    # Bound the explanatory source to the pinned gitlink, independent of unrelated submodule edits.
    import subprocess
    pinned=subprocess.check_output(['git','-C',str(ROOT/'third_party/dacapo'),'show','4616402710f39df3e5f5bd7930a6c036025aaac3:lib/Runtime/SEAL_HEVM.cpp'],timeout=10)
    require(cpp.read_bytes()==pinned,'Runtime explanatory source differs from pinned version')
    require('evaluator->relinearize_inplace(ciphers[dst], *relin_key);' in pinned.decode(),'Pinned fused relinearization missing')
    counts=trace['gate']['opcode_counts']
    compiler_evidence=dict(rescale=dict(artifact_operations=int(counts.get('3',0)),level_transitions=[e for e in trace['events'] if e['opcode']==3]),
        modswitch=dict(artifact_operations=int(counts.get('4',0)),level_transitions=[e for e in trace['events'] if e['opcode']==4]),
        relinearization=dict(fused_mulcc_operations=int(counts.get('8',0)),explicit_standalone_opcode=False,
            pinned_runtime_source_sha256=hashlib.sha256(pinned).hexdigest(),runtime_binary_sha256=report['runtime_sha256'],output_ciphertext_polynomials=2,
            per_operation_runtime_observed=False,evidence_scope='fused operation in artifact and pinned runtime source plus observed compact outputs'),
        security_parameters=dict(actual_generated_parameters=params,profile_sha256=sha(PROFILE)),
        rotation_keys=execution['rotation_key_check'])
    return dict(schema=1,request_id=request['request_id'],evidence=str(folder),report_sha256=sha(folder/'report.json'),
        numerical_passed=attempt['comparison']['passed'],comparison=attempt['comparison'],lineage=trace,compiler_evidence=compiler_evidence,
        platform_identity=report['platform_identity'],agent_calls=0,all_input_proof=False)
