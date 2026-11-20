"""Data-only prototype for an explicit Native SDK v3; old SDK v2 stays immutable.

Validation of a staging descriptor is distinct from fresh-input and genuine
Agent acceptance. This module never opens credentials or private key files,
loads candidate code, creates keys, runs a compiler or starts a native process.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import poseidon_sdk_manifest as base_sdk

FORMAT='poseidon-runtime-toolchain-native-v4'
POLICY='poseidon-native-certified-unused-input-link-prototype-v3'
CAPABILITY='e4e8fe348b7f875d7af1283ca4c51da23090c17f591f42f677a5b4d56d914c9e'
ENGINE_CERTIFICATE='0d71b0c3154ad98344a555895d1cad5bd482c22f6cd0a0e417cd733e3c2456ff'
PROFILE='273e3a20a734a3beb71c4eafc5c29a21e49db8bd5c64576e30ef13c159c2c621'
SPEC='a1e74e356276cda1141e3b651e72ba8a5456dabeed03120830f6f0b371c55841'
EPOCH='ad7439a7013ff2cf8f215d97daad155a'
ORIGIN='a594528b70cbff43f1b62ec6e4e0c23a4568aa4ffacf76a4031d9d97b2e60e8e'
BASE_SDK='6d2b5f6afc33e4b9d7cc3e6ec71291eba4ad9def5b2e41c52e0197bb823f66d6'
SOURCE_LOCK='243946ab90446743f0063b7b71bdbfd6ba46f0bab63e0ca6eecf57af9eef0526'
CONTROLS='bfa29f109340b4a0956bd66c4cec9a11655df387e6bfa297fd2a2051d056aff4'
PUBLIC_MANIFEST='4b4478ee035661525c8e6690199af756a9ef40d682435e42ba51156c8e78ba65'
PRIVATE_MANIFEST='60f04ad753cb929995b8e5c6d924fc17c7335faef9aebf2cf25bb8d97a4c9e98'
FRESH_INPUT='e7e8f6277779471c8a950d0bf6d30e91542a9e60c95570f4122765b04c1c5bdd'
FRESH_CERTIFICATE='af0bb1f09a90d9ff88f259ff6189176f25429977a764b691e8826ae3d37ea59a'
HASH=re.compile(r'[0-9a-f]{64}\Z')
DOMAIN=dict(format='poseidon-native-bounded-input-profile-v1',minimum=-.5,maximum=.5,
            max_slot_period=256,max_logical_inputs=1,
            validation_sampler='native-bounded-four-groups-v1')
NATIVE_FIELDS={'format','adapter_policy','base_sdk','capability','engine_certificate',
    'binding_controls','compiler_profile','operator_spec','runtime','client','modules',
    'assets','domain','fresh_input_acceptance','agent_acceptance'}
MODULES={'native_boot_plan_link_v2.py','native_boot_plan_link_v3.py','native_sdk_protocol.py',
         'native_sdk_manifest.py','native_execution_profile.py','native_argument_binding.py',
         'native_sandbox.py','native_qualification.py','native_validation.py','native_program_package.py','native_release_evidence.py'}


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),
                      ensure_ascii=False,allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition,reason):
    if not condition:raise ValueError('Native SDK binding: '+reason)


def _hash(value):return type(value)is str and HASH.fullmatch(value)is not None


def _public_file(value,work=None):
    path=base_sdk.path(value,within=work)
    # Reject disallowed paths BEFORE hashing. In particular this validator is
    # never a credential or private-key reader, even for a malformed descriptor.
    require(path.name!='.env'and 'client-private'not in path.parts and
            path.suffix in('','.json','.py'),'public metadata/program artifact only')
    return path


def _record(value,work=None):
    require(type(value)is dict and set(value)=={'path','sha256'},'exact artifact record')
    require(_hash(value['sha256']),'artifact sha256')
    path=_public_file(value['path'],work)
    require(path.stat().st_size<=16*1024**2 and base_sdk.file_hash(path)==value['sha256'],
            'bounded unchanged artifact')
    return path


def _bound(value,work=None,document_digest=digest):
    require(type(value)is dict and set(value)=={'path','sha256','binding'},'exact bound JSON record')
    require(_hash(value['binding']),'JSON binding sha256')
    require(_public_file(value['path'],work).suffix=='.json','bound document must be JSON')
    path=_record({k:value[k]for k in ['path','sha256']},work)
    document=base_sdk.read_json(path)
    require(type(document)is dict and document.get('binding')==value['binding'] and
            document['binding']==document_digest({k:v for k,v in document.items()if k!='binding'}),
            'bound JSON contents')
    return document


def validate(value,work,platform,agent_root):
    if type(value)is dict and value.get('format')=='poseidon-runtime-toolchain-native-v5':
        from native_sdk_manifest_v5 import validate as validate_v5
        return validate_v5(value,work,platform,agent_root)
    require(type(value)is dict and set(value)==base_sdk.FIELDS|{'native'},'descriptor fields')
    require(value['format']==FORMAT and value['platform']==platform,'format/platform')
    require(value['binding']==digest({k:v for k,v in value.items()if k!='binding'}),'descriptor binding')
    require(value['qualification']in ('pending_agent_acceptance','minimal_native_agent_verified'),
            'explicit Native qualification state')
    native=value['native']
    require(type(native)is dict and set(native)==NATIVE_FIELDS,'native fields')
    require(native['format']=='poseidon-native-public-runtime-profile-v1'and
            native['adapter_policy']==POLICY,'native protocol policy')
    original=_bound(native['base_sdk'],work,document_digest=base_sdk.digest)
    require(original['binding']==BASE_SDK and original['source_lock_sha256']==SOURCE_LOCK,
            'exact existing source/compiler/library baseline')
    # Let the existing loader validate all v2 path/hash/field rules itself.
    checked=base_sdk.load(native['base_sdk']['path'],work,platform)
    require(checked==original,'base SDK unchanged')
    for field in base_sdk.FIELDS-{'format','qualification','binding'}:
        require(canonical(value[field])==canonical(original[field]),'base toolchain field changed: '+field)
    capability=_bound(native['capability'])
    require(capability['binding']==CAPABILITY and capability['key_epoch']==EPOCH and
            capability['public_origin']==ORIGIN,'original certified context/epoch')
    certificate=_bound(native['engine_certificate'])
    require(certificate['binding']==ENGINE_CERTIFICATE and certificate['prototype_four_group_backend_certified']is True and
            certificate['capability_contract_binding']==CAPABILITY and certificate['failed_values']==0 and
            certificate['Agent_generation_claim']is False,'original engine certificate scope')
    controls=_bound(native['binding_controls'],work)
    require(controls['binding']==CONTROLS and controls['status']=='passed'and
            controls['format']=='poseidon-A5-native-v3-complete-output-binding-build-controls-v1'and
            controls['explicit_no_scale_or_level_translation']is True and
            controls['complete_deterministic_derived_plan_and_ordered_final_output_binding']is True and
            controls['API_calls']==controls['encrypted_calls']==controls['new_keys']==0,
            'v3 metadata controls do not assert native acceptance')
    _record(native['compiler_profile'],work);_record(native['operator_spec'],work)
    require(native['compiler_profile']['sha256']==PROFILE and native['operator_spec']['sha256']==SPEC,
            'approved physical compiler profile/spec')
    profile=base_sdk.read_json(native['compiler_profile']['path'])
    require(type(profile.get('levelUpperBound'))is int and profile['levelUpperBound']==1 and
            type(profile.get('bootstrapLevelUpperBound'))is int and profile['bootstrapLevelUpperBound']==1 and
            profile['rescalingFactor']==48 and profile['polynomialDegree']==32768,
            'exact physical state availability')
    for role,key in [('runtime','runtime_sha256'),('client','client_sha256')]:
        _record(native[role],work)
        require(native[role]['sha256']==controls[key],'new artifact/control receipt mismatch')
    require(type(native['modules'])is dict and set(native['modules'])==MODULES,'static registered modules only')
    root=Path(agent_root).resolve()
    for name,record in native['modules'].items():
        require(Path(record['path']).name==name,'exact static module filename')
        path=_record(record)
        require(path.name==name and path.is_relative_to(root),'module outside selected Agent source')
        if name=='native_sdk_manifest.py':
            require(path==Path(__file__).resolve(),'actual descriptor validator source must be bound')
    assets=native['assets']
    require(type(assets)is dict and set(assets)=={'public_directory','public_manifest_sha256',
            'client_private_directory','client_private_manifest_sha256'},'exact original asset fields')
    public=base_sdk.path(assets['public_directory'],within=work,directory=True)
    private=base_sdk.path(assets['client_private_directory'],within=work,directory=True)
    expected=Path(work)/'results/agent-bootstrap-piece-bridge-r1900'
    require(public==expected/'public'and private==expected/'client-private','original asset locations')
    require(assets['public_manifest_sha256']==PUBLIC_MANIFEST==certificate['public_manifest_sha256']and
            assets['client_private_manifest_sha256']==PRIVATE_MANIFEST,
            'original public manifest binding')
    # Private manifest and secret bytes are read only by the original final client.
    public_manifest=base_sdk.path(str(public/'manifest.json'),within=work)
    require(base_sdk.file_hash(public_manifest)==assets['public_manifest_sha256'],'public manifest bytes')
    public_metadata=base_sdk.read_json(public_manifest)
    require(public_metadata['epoch']==EPOCH and public_metadata['origin']==ORIGIN,
            'original public metadata epoch')
    require(canonical(native['domain'])==canonical(DOMAIN),'explicit fixed bounded input profile')
    fresh=native['fresh_input_acceptance']
    if fresh is not None:
        attestation=_bound(fresh,work)
        require(attestation['binding']==FRESH_INPUT and
                attestation.get('format')=='poseidon-native-fresh-input-acceptance-v1'and
                attestation.get('passed')is True and attestation.get('new_keys')==0 and
                attestation.get('runtime_sha256')==native['runtime']['sha256']and
                attestation.get('client_sha256')==native['client']['sha256']and
                attestation.get('capability_binding')==CAPABILITY and
                attestation.get('original_epoch')==EPOCH,
                'fresh-input attestation for actual v3 binaries')
        measured=_bound(dict(path=attestation['measured_certificate'],
            sha256=attestation['measured_certificate_sha256'],
            binding=attestation['measured_certificate_binding']))
        require(measured['binding']==FRESH_CERTIFICATE and measured['new_keys']==0 and
                measured['actual_native_processes']==3 and measured['actual_native_Boots']==2 and
                measured['complete_original_source_final_output_relation_verified']is True and
                measured['host_owner_verified']is True and measured['failed_values']==0 and
                measured['compared_values']==16384 and measured['Agent_generated']is False and
                measured['normal_Application_SDK_acceptance']is False and
                measured['program_package_reload']is False,
                'measured backend scope cannot substitute Agent/program acceptance')
        _record(dict(path=attestation['report'],sha256=attestation['report_sha256']),work)
    acceptance=native['agent_acceptance']
    if acceptance is not None:
        _agent_acceptance(value,acceptance,work)
    require(value['qualification']=='pending_agent_acceptance'or
            fresh is not None and acceptance is not None,'real Agent/package acceptance required for promotion')
    return copy.deepcopy(value)


def load(filename,work,platform,agent_root):
    filename=_public_file(str(filename),work)
    return validate(base_sdk.read_json(filename),work,platform,agent_root)


def attest(value,work,agent_root):
    if type(value)is dict and value.get('format')=='poseidon-runtime-toolchain-native-v5':
        from native_sdk_manifest_v5 import attest as attest_v5
        return attest_v5(value,work,agent_root)
    value=validate(value,work,value['platform'],agent_root)
    original=base_sdk.load(value['native']['base_sdk']['path'],work,value['platform'])
    base_sdk.attest(original,work,agent_root)
    capability=_bound(value['native']['capability'])
    launch=Path(capability['source_launch'])
    for name,expected in capability['kernel_source_sha256'].items():
        source=base_sdk.path(str(launch/name))
        require(source.is_relative_to(launch)and
                (source.name=='CMakeLists.txt'or source.suffix in('.h','.hpp','.cpp','.json','.py','.nix'))and
                'client-private'not in source.parts and base_sdk.file_hash(source)==expected,
                'actual original certified Native source unchanged')
    for record in capability['library_static_archives']:
        archive=base_sdk.path(record['path'],within=work)
        require(archive.suffix=='.a'and base_sdk.file_hash(archive)==record['sha256'],
                'actual original Native archive unchanged')
    return value


def ready_for_agent_qualification(value):
    """A real fresh-input acceptance is required; staging validation alone is insufficient."""
    if value.get('format')=='poseidon-runtime-toolchain-native-v5':
        # This first v5 explicitly attests the new compiler/library/key origin.
        # Fresh-input and normal Application profiles remain separate work.
        return False
    return value['native']['fresh_input_acceptance']is not None


def execution_fingerprint(value):
    """Semantic resources, independent of publication status and descriptor filename."""
    if value.get('format')=='poseidon-runtime-toolchain-native-v5':
        from native_sdk_manifest_v5 import execution_fingerprint as fingerprint_v5
        return fingerprint_v5(value)
    native=value['native']
    content={k:copy.deepcopy(native[k])for k in NATIVE_FIELDS-{'fresh_input_acceptance','agent_acceptance'}}
    return digest(dict(format=FORMAT,base_source_lock=value['source_lock_sha256'],native=content))


def _agent_acceptance(value,record,work):
    """Accept only complete measured Native lifecycle chains, never self-reported counts."""
    document=_bound(record,work)
    from native_release_evidence import verify_release
    verify_release(value,document,work)
