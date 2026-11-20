"""Explicit Native program format; original CPU bundle formats stay unchanged."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from benchmark_graph import canonical,digest,require
from component_contract import reconstruct_request
from strict_json_files import strict_file,dump
from candidate_contract import request_file_bytes,validate_candidate
import native_execution_profile as profile

FORMAT='poseidon-native-dsl-bundle-v1'
MANDATORY={'model.json','request.json','candidate.py',
    'artifacts/source-plan.json','artifacts/plan.json','artifacts/operator-spec.json',
    'artifacts/link-record.json','artifacts/compiler-profile.json','artifacts/validation.json',
    'artifacts/application-identities.json'}


def file(folder,name):
    require(type(name)is str and name and not Path(name).is_absolute()and
            name!='.env'and 'client-private'not in Path(name).parts and '..'not in Path(name).parts,
            'Native package relative file')
    path=Path(folder)/name;root=Path(folder).resolve()
    require(path.resolve().is_relative_to(root)and path.is_file()and
            all(not p.is_symlink()for p in (path,*path.parents))and path.stat().st_size<=16*1024**2,
            'Native package direct bounded file')
    return path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_native(folder):
    manifest=Path(folder)/'manifest.json'
    if not manifest.exists():return False
    return strict_file(manifest,131072).get('format')==FORMAT


def check_bundle(folder):
    folder=Path(folder);manifest=strict_file(file(folder,'manifest.json'),131072)
    require(type(manifest)is dict and set(manifest)=={'format','files','compiler_configuration','origin',
        'dependencies','validation','model_provenance','agent_generated','binding'}and
        manifest['format']==FORMAT,'Explicit Native package format')
    require(manifest['binding']==digest({k:v for k,v in manifest.items()if k!='binding'}),'Native package binding')
    require(type(manifest['files'])is dict and type(manifest['agent_generated'])is bool,'Native package fields')
    from generation_receipt import FILE as GENERATION,MAX_BYTES as GENERATION_LIMIT,verify as verify_generation
    from model_source_provenance import FILE as MODEL_SOURCE,LIMIT as SOURCE_LIMIT,verify as verify_source
    allowed=set(MANDATORY)
    if manifest['model_provenance']=='included':allowed.add('model-provenance.json')
    else:require(manifest['model_provenance']=='not_supplied','Native model provenance status')
    if manifest['agent_generated']:allowed.add(GENERATION)
    if 'model_source'in manifest['origin']:allowed.add(MODEL_SOURCE)
    require(set(manifest['files'])==allowed,'Exact Native package executable file inventory')
    actual={str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()}
    require(actual==allowed|{'manifest.json'}and not any(p.is_symlink()for p in folder.rglob('*')),
            'No unreferenced/indirect Native package files')
    require(all(sha(file(folder,name))==value for name,value in manifest['files'].items()),
            'Native package file bytes')
    from public_weight_context import read_request
    request=reconstruct_request(read_request(file(folder,'request.json').read_text()))
    profile.validate_request(request)
    require(strict_file(file(folder,'model.json'),131072)==request['model']and
            manifest['compiler_configuration']==request['compiler_configuration']and
            manifest['origin']['request_id']==request['request_id'],'Native package request/configuration')
    source=file(folder,'candidate.py').read_text()
    validate_candidate(dict(schema=1,request_id=request['request_id'],hecate_source=source),request)
    level=manifest['validation']['level']
    require(level in ('compiled','numerical')and manifest['validation']==dict(level=level,
        encrypted_execution=level=='numerical',numerically_validated=level=='numerical'),
        'Original Native package acceptance level')
    from application_identity import check_manifest
    identities=check_manifest(strict_file(file(folder,'artifacts/application-identities.json'),16*1024**2))
    require(manifest['dependencies']==dict(backend_identity=identities['backend']),
            'Native package/stage dependencies')
    provenance=None
    if manifest['model_provenance']=='included':
        from candidate_bundle import check_provenance
        provenance=strict_file(file(folder,'model-provenance.json'),4*1024**2)
        check_provenance(provenance,request['model'])
    if manifest['agent_generated']:
        require(manifest['origin'].get('generation_provenance')==
            dict(file=GENERATION,sha256=manifest['files'][GENERATION]),'Native generation receipt descriptor')
        verify_generation(strict_file(file(folder,GENERATION),GENERATION_LIMIT),request,source,
            generation_identity_sha256=identities['generation']['sha256'])
    else:
        require('generation_provenance'not in manifest['origin'],'No fabricated Native Agent origin')
    if MODEL_SOURCE in allowed:
        from model_source_provenance import FORMAT as SOURCE_FORMAT
        require(manifest['origin']['model_source']==dict(format=SOURCE_FORMAT,file=MODEL_SOURCE,
                sha256=manifest['files'][MODEL_SOURCE]),'Native model-source descriptor')
        verify_source(strict_file(file(folder,MODEL_SOURCE),SOURCE_LIMIT),request,provenance,
            controller_sha256=identities['generation']['sources'].get('application_model_source.py'))
    return manifest


def check_execution_dependencies(folder):
    manifest=check_bundle(folder)
    request=reconstruct_request(strict_file(file(folder,'request.json'),4*1024**2))
    from poseidon_backend_identity import identity
    from application_identity import current_identity_manifest,compatible
    require(identity()==manifest['dependencies']['backend_identity'],'Exact current Native context/toolchain')
    recorded=strict_file(file(folder,'artifacts/application-identities.json'),16*1024**2)
    require(compatible(recorded,current_identity_manifest()),'Native stage sources changed; requalification required')
    from poseidon_runtime_paths import checked_sdk
    import native_boot_plan_link_v3 as linker
    sdk=checked_sdk()
    require(type(sdk)is dict and 'native'in sdk,'Explicit Native SDK required')
    derived,link=linker.link_fragment(file(folder,'artifacts/source-plan.json').read_bytes(),
        Path(sdk['native']['capability']['path']).read_bytes(),file(folder,'artifacts/operator-spec.json').read_bytes())
    require(linker.encode(derived)==file(folder,'artifacts/plan.json').read_bytes()and
            link==strict_file(file(folder,'artifacts/link-record.json'),256*1024),'Recomputed Native package derivation')
    require(sha(file(folder,'artifacts/compiler-profile.json'))==profile.PROFILE and
            sha(file(folder,'artifacts/operator-spec.json'))==profile.SPEC,'Native package physical profile/spec')
    return manifest,request


def export_bundle(evidence,output,*,model_provenance=None,validation_level='numerical',
                  application_identities=None,generation_provenance=None,model_source=None):
    from native_validation import audit_candidate
    from hecate_python_env import WORK
    from application_identity import check_manifest,current_identity_manifest
    from poseidon_runtime_paths import checked_sdk
    from candidate_bundle import audit_provenance,serialize_model_provenance
    evidence=Path(evidence);output=Path(output)
    require(evidence.resolve().is_relative_to((WORK/'results').resolve())and not output.exists(),
            'New Native package from original evidence')
    audit=audit_candidate(evidence,validation_level=validation_level)
    request=reconstruct_request(strict_file(evidence/'request.json',4*1024**2));report=strict_file(evidence/'report.json',8*1024**2)
    require(application_identities is not None and
            check_manifest(application_identities)==current_identity_manifest(),'Current Native Application identities')
    require(report['backend_identity']==application_identities['backend'],'Qualified actual Native toolchain')
    sdk=checked_sdk();attempt=evidence/'attempt-00'
    source=(attempt/'candidate.py').read_text()
    output.mkdir(parents=True,mode=0o700);artifacts=output/'artifacts';artifacts.mkdir(mode=0o700)
    dump(output/'model.json',request['model']);(output/'request.json').write_bytes(request_file_bytes(request))
    (output/'candidate.py').write_text(source)
    for name in ['source-plan.json','plan.json','operator-spec.json','link-record.json']:
        shutil.copyfile(attempt/'output/native-bound'/name,artifacts/name)
    shutil.copyfile(sdk['native']['compiler_profile']['path'],artifacts/'compiler-profile.json')
    dump(artifacts/'application-identities.json',application_identities)
    dump(artifacts/'validation.json',dict(level=validation_level,scope='four explicit Native bounded probes',
        report_sha256=sha(evidence/'report.json'),audit=audit,gate=report['attempts'][0]['artifact_gate']))
    names=set(MANDATORY)
    manifest=dict(format=FORMAT,files={},compiler_configuration=request['compiler_configuration'],
        origin=dict(request_id=request['request_id'],report_sha256=sha(evidence/'report.json')),
        dependencies=dict(backend_identity=application_identities['backend']),
        validation=dict(level=validation_level,encrypted_execution=validation_level=='numerical',
                        numerically_validated=validation_level=='numerical'),
        model_provenance='included'if model_provenance is not None else'not_supplied',
        agent_generated=generation_provenance is not None)
    if model_provenance is not None:
        audit_provenance(model_provenance,request['model'])
        (output/'model-provenance.json').write_bytes(serialize_model_provenance(model_provenance,4*1024**2))
        names.add('model-provenance.json')
    if generation_provenance is not None:
        from generation_receipt import FILE,verify
        verify(generation_provenance,request,source,generation_identity_sha256=application_identities['generation']['sha256'])
        (output/FILE).write_bytes(canonical(generation_provenance));names.add(FILE)
        manifest['origin']['generation_provenance']=dict(file=FILE,sha256=sha(output/FILE))
    if model_source is not None:
        from model_source_provenance import FILE,FORMAT as SOURCE_FORMAT,verify
        verify(model_source,request,model_provenance,
            controller_sha256=application_identities['generation']['sources'].get('application_model_source.py'))
        (output/FILE).write_bytes(canonical(model_source));names.add(FILE)
        manifest['origin']['model_source']=dict(format=SOURCE_FORMAT,file=FILE,sha256=sha(output/FILE))
    manifest['files']={name:sha(output/name)for name in sorted(names)}
    manifest['binding']=digest(manifest);dump(output/'manifest.json',manifest)
    check_bundle(output)
    return manifest


def execute_program(folder,samples,*,artifact_observer=None,sample_format='named'):
    from hecate_python_env import WORK
    from program_execution import (normalize_samples,named_outputs,package_execution_metadata,FORMAT as EXECUTION_FORMAT)
    from native_argument_binding import _logical_array
    from native_validation import execute_samples
    from poseidon_runtime_paths import checked_sdk,MANIFEST
    manifest,request=check_execution_dependencies(folder)
    # A checked compiled package may execute under the explicit metrics-only policy.
    metadata=package_execution_metadata(folder,manifest,request['model'])
    require(sample_format in ('named','forward_call'),'Native execution sample format')
    if sample_format=='forward_call':
        from pytorch_input_binding import flatten_samples
        require('input_call_plan'in metadata,'Native package structured input contract')
        samples=flatten_samples(metadata['input_call_plan'],samples)
    require(type(samples)is list and 1<=len(samples)<=4,'Native per-operation physical process budget')
    # Check the original data before generic float64 normalization can erase
    # a boolean or wider-precision dtype.
    specification=request['model']['inputs'][0]
    for sample in samples:
        require(type(sample)is dict and set(sample)=={specification['name']},'Native named input declaration')
        _logical_array(sample[specification['name']],specification['shape'])
    logical=normalize_samples(request,samples)
    from execution_input_domain import check_approximation_domains
    domain=check_approximation_domains(metadata['original_model'],logical)
    evidence=Path(tempfile.mkdtemp(prefix='poseidon-program-execution-',dir=WORK/'results'))
    if artifact_observer is not None:artifact_observer(evidence)
    initial=copy.deepcopy(manifest)
    def frozen():
        now,_=check_execution_dependencies(folder)
        require(now==initial,'Native package changed during new-input execution')
    sdk=checked_sdk()
    actual,execution=execute_samples(evidence/'native-execution',Path(folder)/'artifacts',
        request,sdk,MANIFEST,logical,frozen)
    frozen()
    outputs=named_outputs(request,actual,len(logical))
    report=dict(format=EXECUTION_FORMAT,status='executed',backend='poseidon_cpu',native_sdk=True,
        bundle_binding=manifest['binding'],request_id=request['request_id'],package_validation=manifest['validation'],
        encrypted_execution=True,executor_secret_key=False,numerically_validated_on_supplied_inputs=False,
        application_accuracy_certified=False,validation_scope='New supplied inputs; no reference comparison requested',
        samples=len(logical),outputs=outputs,execution=execution,input_domain_check=domain,
        agent_calls=0,compiler_calls=0,evidence=str(evidence))
    if metadata['output_structure']is not None:
        from pytorch_source_contract import restore_outputs
        structure=metadata['output_structure']
        report.update(output_structure=structure,structured_outputs=[restore_outputs(value,structure)for value in outputs])
    dump(evidence/'report.json',report)
    return report
