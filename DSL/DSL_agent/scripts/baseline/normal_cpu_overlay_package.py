"""Checked ordinary-CPU package envelope around the unchanged RuntimePlan bundle.

The inner exporter/checker and compiler artifacts remain original. The envelope
binds the approved DSO adapter and its current execution identity. Reload uses
that same read-only DSO sandbox, without generation, compilation or references.
"""
import copy
import json
import shutil
from pathlib import Path

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
from first_output_backend import _json

FORMAT='poseidon-normal-CPU-overlay-program-v1'
MANIFEST='normal-cpu-program.json'


def inventory(folder):
    folder=Path(folder).absolute()
    require(folder.is_dir() and folder.resolve()==folder,'Direct package directory')
    files={}
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(),'No indirect package entries')
        if not path.is_file():continue
        name=str(path.relative_to(folder))
        if name==MANIFEST:continue
        require(path.stat().st_size<=16*1024**2,'Original package per-file ceiling')
        files[name]=sha(read_bytes(path,16*1024**2))
    return files


def verify_files(folder,value):
    checked_bound(value)
    require(value['format']==FORMAT and value['files']==inventory(folder),'Package complete byte inventory changed')
    require(all(name.startswith('program/')or name in {'adapter-identity.json','generation-provenance.json','README.txt'}
                for name in value['files']),'Only public program/adapter provenance belongs in a package')
    require(value['inner_manifest']==dict(path='program/manifest.json',sha256=value['files']['program/manifest.json']), 'Original RuntimePlan package descriptor')
    require(value['source_sha256']==value['files']['program/candidate.py']
            and value['request_sha256']==value['files']['program/request.json'], 'Frozen DSL/request package bytes')
    require(value['qualification_level']in('compiled','numerical') and value['full_SDK_or_network_qualified'] is False,'Scoped program qualification only')
    return value


def _backend_descriptor(backend):
    return dict(overlay=copy.deepcopy(backend.overlay),canary_plan=copy.deepcopy(backend.canary_plan),canary_result=copy.deepcopy(backend.canary_result))


def export(backend,result,destination,level,*,model_provenance=None,generation_provenance=None,model_source=None,artifact_observer=None):
    from candidate_bundle import export_bundle
    from application_identity import current_identity_manifest
    from generation_receipt import verify as verify_generation
    root=Path(destination).absolute();require(not root.exists() and root.parent.resolve()==root.parent,'New direct package export only')
    require(level in('compiled','numerical') and result.get('status')=='passed','Passed fixed-candidate qualification required')
    evidence=Path(result['evidence']).absolute();report=_json(evidence/'report.json');request=_json(evidence/'request.json')
    require(len(report['attempts'])==1 and report['attempts'][0]['index']==0 and report['status']=='passed','Exactly one original qualifier source; no repaired selection')
    source=read_bytes(evidence/'attempt-00/candidate.py')
    backend.check_request(request,level);identity=backend.identity_manifest()
    if generation_provenance is not None:
        verify_generation(generation_provenance,request,source.decode('utf-8'),generation_identity_sha256=identity['generation']['sha256'])
    root.mkdir(mode=0o700)
    if artifact_observer is not None:artifact_observer(root)
    # The inner record retains the unchanged original application identity.
    # The outer record separately binds this adapter; no old record is re-signed.
    # The original exporter retains an audit workspace beside its destination.
    # Keep that private validation evidence outside the delivered program tree.
    staging=root.with_name(root.name+'.export-evidence');require(not staging.exists(),'Preserve prior export evidence')
    staging.mkdir(mode=0o700)
    if artifact_observer is not None:artifact_observer(staging)
    inner=export_bundle(evidence,staging/'program',model_provenance=model_provenance,validation_level=level,
        application_identities=current_identity_manifest(),model_source=model_source)
    shutil.copytree(staging/'program',root/'program',symlinks=False)
    require(read_bytes(root/'program/candidate.py')==source,'Original exported DSL source changed')
    write_once(root/'adapter-identity.json',canonical(identity))
    generation=None
    if generation_provenance is not None:
        write_once(root/'generation-provenance.json',canonical(generation_provenance))
        generation=dict(path='generation-provenance.json',sha256=sha(read_bytes(root/'generation-provenance.json')))
    note=('Poseidon ordinary CPU program with an approved repaired-library adapter.\n'
          'The DSL and compiled RuntimePlan are frozen. Reload uses normal_cpu_overlay_package.execute.\n'
          'The recorded qualification covers its fixed inputs; new supplied inputs need independent numerical validation.\n'
          'No key material, private test inputs or references are included. Bootstrap and complete-network qualification are separate.\n')
    write_once(root/'README.txt',note.encode())
    value=bound(dict(format=FORMAT,inner_manifest=dict(path='program/manifest.json',sha256=sha(read_bytes(root/'program/manifest.json'))),
        files=inventory(root),backend=_backend_descriptor(backend),qualification_level=level,
        original_qualification=dict(evidence=str(evidence),report_sha256=sha(read_bytes(evidence/'report.json')),
            source_sha256=sha(source),original_adapter_identity_sha256=report.get('normal_CPU_approved_overlay',{}).get('identity',{}).get('manifest_sha256')),
        inner_binding=inner['binding'],adapter_identity_sha256=sha(read_bytes(root/'adapter-identity.json')),
        source_sha256=sha(source),request_sha256=sha(read_bytes(root/'program/request.json')),
        generation_provenance=generation,agent_generated=generation is not None,
        full_SDK_or_network_qualified=False))
    write_once(root/MANIFEST,canonical(value))
    require(backend.identity_manifest()==identity,'Adapter changed during program export')
    check(backend,root,level)
    return value


def check(backend,folder,level=None):
    from candidate_bundle import check_execution_dependencies,package_identities
    from application_identity import check_manifest,compatible
    from generation_receipt import verify as verify_generation
    folder=Path(folder).absolute();value=verify_files(folder,_json(folder/MANIFEST))
    require(value['backend']==_backend_descriptor(backend),'Program selected another SDK or canary')
    require(level is None or value['qualification_level']==level,'Program qualification grade mismatch')
    saved=check_manifest(_json(folder/'adapter-identity.json'));current=backend.identity_manifest()
    require(sha(read_bytes(folder/'adapter-identity.json'))==value['adapter_identity_sha256']
            and compatible(saved,current),'Program execution adapter/stage identity changed')
    inner,request=check_execution_dependencies(folder/'program')
    actual_inner={str(p.relative_to(folder/'program'))for p in(folder/'program').rglob('*')if p.is_file()}
    require(actual_inner==set(inner['files'])|{'manifest.json','README.md'},'Original inner package includes unlisted evidence')
    require(inner['binding']==value['inner_binding'] and inner['validation']['level']==value['qualification_level']
            and inner['agent_generated'] is False,'Original inner package and separate generation provenance')
    backend.check_request(request,value['qualification_level'])
    generation=value['generation_provenance']
    if generation is not None:
        require(value['agent_generated'] is True and generation==dict(path='generation-provenance.json',sha256=value['files']['generation-provenance.json']), 'Outer generation descriptor')
        verify_generation(_json(folder/'generation-provenance.json'),request,read_bytes(folder/'program/candidate.py').decode(),generation_identity_sha256=saved['generation']['sha256'])
    else:require(value['agent_generated'] is False and 'generation-provenance.json'not in value['files'],'Unreferenced Agent provenance')
    return value,inner,request


def load(folder):
    """Reconstruct the explicit trusted adapter in a new already selected worker."""
    from normal_cpu_overlay_backend import NormalCpuOverlayBackend
    folder=Path(folder).absolute();value=verify_files(folder,_json(folder/MANIFEST));record=value['backend']
    require(type(record)is dict and set(record)=={'overlay','canary_plan','canary_result'},'Explicit approved-overlay backend descriptor')
    backend=NormalCpuOverlayBackend(record['overlay'],record['canary_plan'],record['canary_result'])
    check(backend,folder)
    return backend


def execute(backend,folder,samples,*,artifact_observer=None,sample_format='named'):
    import os,sys,tempfile
    import numpy as np
    from hecate_python_env import WORK,VENV,digest
    from program_execution import normalize_samples,_pack_logical_samples,named_outputs,package_execution_metadata
    from execution_input_domain import check_approximation_domains
    from poseidon_package import copy_checked_artifacts
    from poseidon_runtime_compile import audit_entrypoint
    from poseidon_validation import audit_gate,execute as original_execute,input_payload
    from candidate_contract import trace_payload_json
    from native_execution_slots import native_slot
    from normal_cpu_overlay_sandbox import run,verify_execution_libraries
    require(os.environ.get('IN_NIX_SHELL')=='pure' and Path(sys.prefix)==VENV,'Pinned original SDK worker required')
    folder=Path(folder).absolute();outer,inner,request=check(backend,folder);identity=backend.identity_manifest()
    metadata=package_execution_metadata(folder/'program',inner,request['model']);structure=metadata['output_structure']
    require(sample_format in('named','forward_call'),'Original input calling convention')
    if sample_format=='forward_call':
        from pytorch_input_binding import flatten_samples
        require('input_call_plan'in metadata,'Program has no structured input calling contract')
        samples=flatten_samples(metadata['input_call_plan'],samples)
    logical=normalize_samples(request,samples);domain=check_approximation_domains(metadata['original_model'],logical)
    packed=_pack_logical_samples(request,logical)
    evidence=Path(tempfile.mkdtemp(prefix='normal-cpu-program-execution-',dir=WORK/'results'))
    if artifact_observer is not None:artifact_observer(evidence)
    output=evidence/'output';output.mkdir(mode=0o700);copy_checked_artifacts(folder/'program',inner,output)
    record=_json(output/'validation.json');audit_entrypoint(output,request,record.get('runtime_plan_entry'),record.get('compile_command'));audit_gate(output,request,record['gate'])
    from application_planning.compiled_resources import execution_storage_precheck
    resources=execution_storage_precheck(output,request)
    source=read_bytes(folder/'program/candidate.py');candidate=dict(schema=1,request_id=request['request_id'],hecate_source=source.decode())
    payload=evidence/'payload.json';write_once(payload,trace_payload_json(request,candidate).encode())
    write_once(output/'inputs.json',canonical(input_payload(packed,request['layout'])))
    frozen={str(p.relative_to(output)):digest(p)for p in output.rglob('*')if p.is_file()};payload_hash=digest(payload)
    context,_=backend._evidence()
    def native_run(*args,**kwargs):return run(*args,**kwargs,selected_library=context['library'])
    with native_slot(WORK/'cache/agent-native-slots'):
        execution=original_execute(payload,output,request,native_run,evidence/'execute.log')
    receipt=verify_execution_libraries(output,evidence/'execute.log',context['library'])
    require(all(digest(output/p)==h for p,h in frozen.items()) and digest(payload)==payload_hash,'Compiled program/input changed during reload')
    final,_,_=check(backend,folder);require(final==outer and backend.identity_manifest()==identity,'Reloaded package/adapter changed')
    outputs=named_outputs(request,np.load(output/'decrypted.npy',allow_pickle=False),len(logical))
    report=dict(format='poseidon-normal-CPU-overlay-program-execution-v1',status='executed',backend='poseidon_cpu',bundle_binding=outer['binding'],
        inner_binding=inner['binding'],request_id=request['request_id'],source_sha256=sha(source),package_validation=copy.deepcopy(inner['validation']),
        encrypted_execution=True,executor_secret_key=False,numerically_validated_on_supplied_inputs=False,application_accuracy_certified=False,
        validation_scope='Supplied inputs executed with reused exact compiler artifacts; no numerical reference comparison requested',
        samples=len(logical),outputs=outputs,execution=execution,resource_precheck=resources,input_domain_check=copy.deepcopy(domain),
        actual_DSO_binding=receipt['binding'],agent_calls=0,compiler_calls=0,compiled_artifacts_reused=True,evidence=str(evidence))
    if structure is not None:
        from pytorch_source_contract import restore_outputs
        report.update(output_structure=structure,structured_outputs=[restore_outputs(v,structure,**({'allow_none':True}if metadata.get('output_structure_policy')=='poseidon-mha-ordered-source-v1'else{}))for v in outputs])
    write_once(evidence/'report.json',canonical(report));return report
