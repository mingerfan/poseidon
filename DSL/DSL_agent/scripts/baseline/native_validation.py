"""Original Hecate compilation and real Native client/evaluator comparison."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from benchmark_graph import require
from candidate_contract import strict_json,validate_candidate,trace_payload_json,request_input_names
from numerical_results import dump,compare
from hecate_python_env import VENV,digest
from native_sandbox import NativeSandbox,sha,read
import native_execution_profile as profile
import native_boot_plan_link_v3 as linker
from native_argument_binding import prepare_inputs


def _runtime_inputs(output,request,sdk):
    source=(output/'lowered._hecate_golden.runtime-plan.json').read_bytes()
    capability=Path(sdk['native']['capability']['path']).read_bytes()
    spec=Path(sdk['native']['operator_spec']['path']).read_bytes()
    derived,record=linker.link_fragment(source,capability,spec)
    parsed=linker.parse(source)
    require(len(parsed['external_inputs'])==len(request_input_names(request))and
            len(derived['final_outputs'])==request['layout']['output_ciphertexts'],'Compiled Native parameter/output ABI')
    bound=output/'native-bound';bound.mkdir(mode=0o700)
    (bound/'source-plan.json').write_bytes(source);(bound/'plan.json').write_bytes(linker.encode(derived))
    (bound/'operator-spec.json').write_bytes(spec);dump(bound/'link-record.json',record)
    return bound,derived,record


def execute_samples(root,runtime_input,request,sdk,sdk_file,samples,verify_frozen):
    """Each sample gets fresh public-key encryption and final-only decryption."""
    require(type(samples)is list and 1<=len(samples)<=4,'Finite Native sample count')
    root=Path(root);root.mkdir(mode=0o700)
    sandbox=NativeSandbox(sdk_file,root,max_processes=3*len(samples),verify_frozen=verify_frozen)
    bound=Path(runtime_input)
    local=root/'runtime-input';local.mkdir(mode=0o700)
    for name in ['source-plan.json','plan.json','operator-spec.json','link-record.json']:
        (local/name).write_bytes((bound/name).read_bytes())
    plan=linker.parse((local/'plan.json').read_bytes());record=read(local/'link-record.json')
    rows=[];outputs=[]
    for index,sample in enumerate(samples):
        verify_frozen()
        identity=dict(public_manifest_sha256=sdk['native']['assets']['public_manifest_sha256'],
            runtime_plan_sha256=sha(local/'plan.json'),operator_spec_sha256=sha(local/'operator-spec.json'),
            source_plan_sha256=sha(local/'source-plan.json'),link_record_sha256=sha(local/'link-record.json'),
            task_binding=hashlib.sha256(linker.encode(dict(request=request['request_id'],
                root=str(root),sample=index,sdk=sandbox.sdk['binding']))).hexdigest())
        payload,parameter_receipt=prepare_inputs(request,(local/'source-plan.json').read_bytes(),
            (local/'plan.json').read_bytes(),(local/'link-record.json').read_bytes(),
            Path(sdk['native']['capability']['path']).read_bytes(),(local/'operator-spec.json').read_bytes(),
            identity['task_binding'],sample)
        group=root/('sample-'+str(index));group.mkdir(mode=0o700)
        plain=group/'plain-input';plain.mkdir(mode=0o700)
        incoming=group/'inputs';incoming.mkdir(mode=0o700)
        evaluated=group/'evaluated';evaluated.mkdir(mode=0o700)
        validation=group/'validation';validation.mkdir(mode=0o700)
        dump(plain/'input.json',payload);dump(group/'parameter-binding.json',parameter_receipt)
        encrypted=sandbox.run('encryptor',local,identity,sha(plain/'input.json'),plain_input=plain,incoming=incoming)
        require(encrypted['new_keys']==0 and encrypted['secret_key_access']is False and
                encrypted['input_count']==len(plan['external_inputs']),'Real public-only fresh input encryption')
        execution=sandbox.run('runtime',local,identity,sha(incoming/'manifest.json'),incoming=incoming,evaluated=evaluated)
        require(execution['runtime_boot_calls']==record['derived_Boot_calls']and
                execution['engine_native_boot_calls']==record['derived_Boot_calls']and
                execution['runtime_compute_calls']==len(plan['execution'])and
                execution['evaluator_decryptions']==execution['reencryptions']==execution['new_keys']==0 and
                execution['actual_scale_not_reassigned']is True,'Real original native operations/key boundary')
        decoded=sandbox.run('validator',local,identity,sha(evaluated/'manifest.json'),
                            evaluated=evaluated,validation=validation)
        require(decoded['intermediate_decryptions']==decoded['reencryptions']==decoded['new_keys']==0 and
                decoded['no_reference_computation_in_active_client_path']is True and
                decoded['client_final_decryptions']==len(plan['final_outputs']),'Final-only original client')
        values=[]
        require([row['value_id']for row in decoded['files']]==plan['final_outputs'],'Declared final output order')
        for row in decoded['files']:
            filename='output-'+row['value_id']+'.actual-complex.f64'
            require(row['name']==filename and row['bytes']==16384*16 and row['slots']==16384,
                    'Fixed finite actual output encoding')
            path=validation/filename
            require(not path.is_symlink()and path.stat().st_size==16384*16 and sha(path)==row['sha256'],
                    'Actual decoded bytes')
            data=np.frombuffer(path.read_bytes(),dtype='<f8').reshape(16384,2)
            require(np.isfinite(data).all(),'Finite actual final ciphertext decode')
            values.append(data[:,0].copy())
        selected=np.asarray([values[cipher][slot]for cipher,slot in request['layout']['output_selectors']],
                            dtype=np.float64)
        outputs.append(selected)
        rows.append(dict(index=index,identity=identity,parameter_receipt=parameter_receipt,
            encryptor=encrypted,runtime=execution,final_client=decoded))
        verify_frozen()
    return np.stack(outputs),dict(format='poseidon-native-sample-execution-v1',samples=rows,
        physical_native_processes=sandbox.used,native_Boot_calls=sum(r['runtime']['runtime_boot_calls']for r in rows),
        new_keys=0,intermediate_decryptions=0,reencryptions=0,only_final_outputs_decrypted=True)


class NativeValidationAdapter:
    def __init__(self,session):self.session=session

    def evaluate(self,raw,index):
        session=self.session;request=session.request;report=session.report;root=session.result
        item=dict(index=index,status='failed',parsed=False,checked=False,compiled=False,executed=False,
                  numerically_correct=False)
        report['attempts'].append(item);attempt=root/('attempt-'+format(index,'02d'));attempt.mkdir(mode=0o700)
        output=attempt/'output';output.mkdir(mode=0o700);phase='response_parse';artifacts={}
        try:
            session.verify_frozen();candidate=strict_json(raw);item['parsed']=True
            phase='static_check';item['static_check']=validate_candidate(candidate,request);item['checked']=True
            dump(attempt/'candidate.json',candidate);(attempt/'candidate.py').write_text(candidate['hecate_source'])
            payload=attempt/'trace-payload.json';payload.write_text(trace_payload_json(request,candidate))
            session.frozen[str(payload.relative_to(root))]=digest(payload)
            phase='dsl_trace'
            require(session.native_run(payload,output,[str(VENV/'bin/python'),'/app/candidate_trace.py'],
                attempt/'trace.log')==0,'Original isolated AST/Hecate tracing failed; inspect trace.log')
            item['trace']=read(output/'trace-evidence.json')
            phase='compiler'
            item['runtime_plan_entry']=profile.prepare_entrypoint(output,request['compiler_configuration'])
            command=profile.command(request['compiler_configuration']);item['compile_command']=command
            require(session.native_run(payload,output,command,attempt/'compile.log',120)==0,
                    'Original isolated Native-profile compiler failed; inspect compile.log')
            item['compiled']=True;phase='artifact_gate'
            bound,derived,record=_runtime_inputs(output,request,session.sdk)
            identity=dict(public_manifest_sha256=session.sdk['native']['assets']['public_manifest_sha256'],
                runtime_plan_sha256=sha(bound/'plan.json'),operator_spec_sha256=sha(bound/'operator-spec.json'),
                source_plan_sha256=sha(bound/'source-plan.json'),link_record_sha256=sha(bound/'link-record.json'),
                task_binding=hashlib.sha256(linker.encode(dict(request=request['request_id'],attempt=index))).hexdigest())
            checker=NativeSandbox(session.descriptor,output,max_processes=1,verify_frozen=session.verify_frozen)
            metadata=checker.verify_plan(bound,identity)
            item['artifact_gate']=dict(source_plan_sha256=sha(bound/'source-plan.json'),
                runtime_plan_sha256=sha(bound/'plan.json'),link_record_sha256=sha(bound/'link-record.json'),
                native_Boot_calls=record['derived_Boot_calls'],retained_inputs=record['retained_source_external_input_ids'],
                removed_inputs=record['removed_unreferenced_source_external_input_ids'],original_Runtime_metadata=metadata,
                metadata_processes=1)
            artifacts={str(p.relative_to(output)):digest(p)for p in output.rglob('*')if p.is_file()}
            if session.validation_level=='numerical':
                phase='poseidon_runtime';samples=profile.samples(request['model'])
                actual,execution=execute_samples(attempt/'native-execution',bound,request,session.sdk,
                    session.descriptor,samples,session.verify_frozen)
                np.save(output/'decrypted.npy',actual,allow_pickle=False)
                dump(attempt/'native-execution-report.json',execution);item.update(executed=True,execution=execution)
                phase='numerical_comparison'
                with np.load(root/'arrays.npz',allow_pickle=False)as data:
                    item['comparison']=compare(actual,data['reference'],1e-5,1e-4)
                require(item['comparison']['passed'],'Frozen Native numerical tolerance failed')
                item['numerically_correct']=True
            item.update(status='passed',compiled_validated=True,validation_level=session.validation_level)
        except Exception as error:
            item.update(failure_layer=phase,diagnostic=str(error)[:2000],
                category='candidate'if phase in ('response_parse','static_check','numerical_comparison')else'pipeline_unclassified')
        finally:
            try:
                session.verify_frozen()
                require(all(digest(output/name)==value for name,value in artifacts.items()),
                        'Compiled Native artifact changed during execution')
            except Exception as error:
                item.update(status='failed',failure_layer='integrity',category='integrity',
                            diagnostic=str(error)[:2000],numerically_correct=False)
            item['artifact_hashes']=artifacts;dump(attempt/'report.json',item)
        return dict(status=item['status'],layer=item.get('failure_layer','complete'),
            category=item.get('category'),diagnostic=item.get('diagnostic','All original Native numerical checks passed'))


def audit_candidate(folder,*,validation_level='numerical'):
    """Recompute the derivation and actual logical comparison from retained bytes."""
    from component_contract import reconstruct_request
    folder=Path(folder);report=read(folder/'report.json');request=reconstruct_request(read(folder/'request.json'))
    profile.validate_request(request)
    require(report['status']=='passed'and report['validation_level']==validation_level,'Passed Native qualification')
    require(all(type(name)is str and not Path(name).is_absolute()and '..'not in Path(name).parts and
                (folder/name).resolve().is_relative_to(folder.resolve())and
                not(folder/name).is_symlink()and digest(folder/name)==value
                for name,value in report['frozen_hashes'].items()),'Original frozen qualification evidence')
    require(len(report['attempts'])==1,'Single fixed candidate qualification')
    attempt=folder/'attempt-00';item=read(attempt/'report.json');candidate=read(attempt/'candidate.json')
    validate_candidate(candidate,request)
    require(item==report['attempts'][0]and item['compiled']is True and item['status']=='passed','Native attempt binding')
    output=attempt/'output';bound=output/'native-bound'
    require(all(type(name)is str and not Path(name).is_absolute()and '..'not in Path(name).parts and
                (output/name).resolve().is_relative_to(output.resolve())and
                not(output/name).is_symlink()and digest(output/name)==value
                for name,value in item['artifact_hashes'].items()),
            'Native compiled artifact evidence')
    from poseidon_runtime_paths import checked_sdk
    sdk=checked_sdk()
    derived,record=linker.link_fragment((bound/'source-plan.json').read_bytes(),
        Path(sdk['native']['capability']['path']).read_bytes(),(bound/'operator-spec.json').read_bytes())
    require(linker.encode(derived)==(bound/'plan.json').read_bytes()and record==read(bound/'link-record.json'),
            'Native complete source and ordered final-output derivation')
    require(item['compile_command']==profile.command(request['compiler_configuration']),'Exact Native compiler flags')
    result=dict(request_id=request['request_id'],compiled=True,native_sdk=True)
    if validation_level=='numerical':
        from benchmark_math import evaluate
        from benchmark_torch import evaluate as torch_reference
        references=[]
        for sample in profile.samples(request['model']):
            expected=evaluate(request['model'],sample);torch_values=torch_reference(request['model'],sample)
            for name,value in expected.items():np.testing.assert_allclose(torch_values[name],value,atol=1e-12,rtol=1e-12)
            references.append(np.concatenate([expected[o['name']].reshape(-1)for o in request['model']['outputs']]))
        reference=np.stack(references)
        with np.load(folder/'arrays.npz',allow_pickle=False)as arrays:
            require(np.array_equal(reference,arrays['reference']),'Original Native mathematical reference')
        decoded=[]
        execution=read(attempt/'native-execution-report.json')
        require(len(execution['samples'])==4 and execution['physical_native_processes']==12 and
                execution['native_Boot_calls']==4*record['derived_Boot_calls'],
                'Exactly four fixed Native encrypted probe executions')
        for index,sample in enumerate(execution['samples']):
            directory=attempt/'native-execution'/('sample-'+str(index))/'validation'
            rows=[]
            require([row['value_id']for row in sample['final_client']['files']]==derived['final_outputs'],
                    'Audited actual final ciphertext order')
            for value in sample['final_client']['files']:
                require(value['name']=='output-'+value['value_id']+'.actual-complex.f64'and
                        value['bytes']==16384*16 and value['slots']==16384,'Bounded final Native decode record')
                filename=directory/value['name']
                require(not filename.is_symlink()and sha(filename)==value['sha256'],'Retained actual final decode bytes')
                values=np.frombuffer(filename.read_bytes(),dtype='<f8').reshape(16384,2)
                require(np.isfinite(values).all(),'Finite retained Native decode');rows.append(values[:,0])
            decoded.append([rows[c][s]for c,s in request['layout']['output_selectors']])
        actual=np.asarray(decoded,dtype=np.float64)
        require(np.array_equal(actual,np.load(output/'decrypted.npy',allow_pickle=False)),'Original actual logical decode')
        comparison=compare(actual,reference,1e-5,1e-4)
        require(comparison==item['comparison']and comparison['passed'],'Recomputed original Native numerical predicate')
        result['comparison']=comparison
    return result
