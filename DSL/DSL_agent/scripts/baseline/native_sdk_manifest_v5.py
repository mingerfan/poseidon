"""New, explicitly selected Native SDK with an attested library overlay.

The historical SDK is used only as compiler ancestry. Its rejected executor
and library are never made executable by this reader. Current runtime/key
evidence has its own identity; numerical measurements do not confer accuracy.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import poseidon_sdk_manifest as base
from native_key_origin_intake import verify as verify_key_origin

FORMAT='poseidon-runtime-toolchain-native-v5'
ANCESTOR='6d2b5f6afc33e4b9d7cc3e6ec71291eba4ad9def5b2e41c52e0197bb823f66d6'
SOURCE_LOCK='243946ab90446743f0063b7b71bdbfd6ba46f0bab63e0ca6eecf57af9eef0526'
POLICY='poseidon-valid-epoch-public-continuation-v1'
KEY_ORIGIN='a540f8446f9d83182250d165a5764310374f48ce338da21f55f79ccb011b587e'
PHYSICAL_CONTRACT='b5dc3ba541cb8eecf6bfb057c7ee6b100cb61e0f406836f2b5b77b3ec8d08837'
BACKEND_CERTIFICATE='67c6cc9b2f293f9ed4ff2d78a4ad4cca52c276d61fb20f02ff4f9c7c666b37cd'
COMPILER_PROFILE='273e3a20a734a3beb71c4eafc5c29a21e49db8bd5c64576e30ef13c159c2c621'
OPERATOR_SPEC='a50483378f90c50960e82d4f5b4856c1c285b28740efb4c11904fe43063c1b8f'
FIELDS=base.FIELDS|{'native','library_overlay'}
NATIVE_FIELDS={'format','adapter_policy','base_sdk','key_origin_evidence','physical_io_contract',
    'backend_certificate','runtime','client','compiler_profile','operator_spec','assets','modules',
    'readiness','numeric_policy'}
SHA=re.compile(r'[0-9a-f]{64}\Z')


def need(ok,reason):
    if not ok:raise ValueError('Native SDK v5: '+reason)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def _hash(value):return type(value)is str and SHA.fullmatch(value)is not None


def public_path(value,*,within=None,directory=False):
    need(type(value)is str and '.env'not in Path(value).parts and
         'client-private'not in Path(value).parts,'public artifact path required')
    return base.path(value,within=within,directory=directory)


def record(value,*,within=None,maximum=16*1024**2):
    need(type(value)is dict and set(value)in({'path','sha256'},{'path','sha256','binding'})and
         _hash(value.get('sha256')),'artifact record fields')
    p=public_path(value['path'],within=within)
    need(p.stat().st_size<=maximum and base.file_hash(p)==value['sha256'],'actual artifact bytes')
    return p


def bound(value,*,within=None,maximum=256*1024):
    need(type(value)is dict and set(value)=={'path','sha256','binding'}and _hash(value['binding']),
         'bound public metadata fields')
    p=record(value,within=within,maximum=maximum);d=base.read_json(p,maximum)
    need(type(d)is dict and d.get('binding')==value['binding']==
         digest({k:v for k,v in d.items()if k!='binding'}),'public metadata binding')
    return d


def _compiler_ancestor(native,work):
    a=bound(native['base_sdk'],within=work)
    need(set(a)==base.FIELDS and a.get('format')==base.FORMAT and
         a['binding']==ANCESTOR and a['source_lock_sha256']==SOURCE_LOCK,
         'registered historical compiler ancestry')
    # This does not call base.load/attest or select the unsafe historical SDK.
    # Verify only the compiler/frontend and their original source/build chain.
    need(a['artifacts']['poseidon_library']==base.REJECTED_KEYGEN_LIBRARY_SHA256,
         'ancestry is historical, never the current library')
    return a


def _key_origin(native,work):
    rows=native['key_origin_evidence']
    from native_key_origin_intake import DOCUMENTS
    need(type(rows)is dict and set(rows)==DOCUMENTS,'complete approved L/K public evidence')
    data={};pins={}
    for name,row in rows.items():
        p=record(row,maximum=1024**2)
        data[name]=p.read_bytes();pins[name]=row['sha256']
    result=verify_key_origin(data,expected_sha256=pins)
    need(result['binding']==KEY_ORIGIN,'registered actual approved key provenance')
    return result


def validate(value,work,platform,agent_root):
    need(type(value)is dict and set(value)==FIELDS and value.get('format')==FORMAT and
         value.get('platform')==platform,'descriptor version/platform/fields')
    need(value.get('binding')==digest({k:v for k,v in value.items()if k!='binding'}),'descriptor binding')
    need(value.get('qualification')=='pending_agent_acceptance','staging qualification only')
    base.reject_known_unsafe_key_generator(value)
    native=value['native'];need(type(native)is dict and set(native)==NATIVE_FIELDS,'Native fields')
    need(native['format']=='poseidon-valid-epoch-native-binding-v1'and
         native['adapter_policy']==POLICY and native['numeric_policy']=='metrics-only','explicit current Native policy')
    ancestor=_compiler_ancestor(native,work)
    preserved=base.FIELDS-{'format','qualification','binding','artifacts','poseidon_build'}
    for field in preserved:
        need(canonical(value[field])==canonical(ancestor[field]),'compiler ancestry field changed: '+field)
    need(set(value['artifacts'])==set(ancestor['artifacts'])and
         all(value['artifacts'][k]==ancestor['artifacts'][k]for k in ('compiler','frontend','executor')),
         'unchanged compiler/frontend and explicit inherited executor identity')
    origin=_key_origin(native,work)
    overlay=value['library_overlay']
    need(type(overlay)is dict and set(overlay)=={'format','candidate','build_report','build_terminal',
         'original_library_sha256','patched_source','source_plan_binding'},'explicit library overlay fields')
    need(overlay['format']=='poseidon-approved-isolated-library-overlay-v1'and
         overlay['original_library_sha256']==ancestor['artifacts']['poseidon_library'],
         'original library is ancestry only')
    candidate=record(overlay['candidate'],within=work)
    need(candidate==Path(value['poseidon_build'])/'libposeidon_shared.so'and
         overlay['candidate']['sha256']==value['artifacts']['poseidon_library']==origin['candidate_library_sha256'],
         'actual repaired library selection')
    build_path=record(overlay['build_report'],within=work)
    terminal_path=record(overlay['build_terminal'],within=work)
    need(overlay['build_report']['sha256']==native['key_origin_evidence']['library_report']['sha256']and
         overlay['build_terminal']['sha256']==native['key_origin_evidence']['library_terminal']['sha256'],
         'same actual successful L evidence')
    build=base.read_json(build_path)
    need(build['source_plan_binding']==overlay['source_plan_binding']==origin['library_plan_binding'],
         'original approved source plan')
    record(overlay['patched_source'])
    need(overlay['patched_source']['sha256']==build['patched_draft_source_sha256'],
         'approved effective source overlay')
    contract=bound(native['physical_io_contract']);certificate=bound(native['backend_certificate'])
    need(contract['binding']==PHYSICAL_CONTRACT and certificate['binding']==BACKEND_CERTIFICATE,
         'registered actual backend evidence')
    need(contract.get('format')=='poseidon-fresh-dense-key-actual-native-physical-IO-contract-v1'and
         contract.get('actual_execution_pending')is False and
         contract.get('epoch')==origin['epoch']and contract.get('origin')==origin['public_origin']and
         contract.get('candidate_library_sha256')==origin['candidate_library_sha256'],
         'actual current context/runtime provenance')
    need(contract['context']==origin['physical_context']and contract['physical_slots']==16384 and
         contract['physical_degree']==32768,'original N/Q/P/scales')
    need(certificate.get('generic_physical_IO_contract_binding')==contract['binding']and
         certificate.get('epoch')==origin['epoch']and certificate.get('origin')==origin['public_origin']and
         certificate.get('candidate_library_sha256')==origin['candidate_library_sha256']and
         certificate.get('actual_Native_Boots')==8 and
         certificate.get('actual_ciphertext_only_intermediate_boundaries')==6 and
         certificate.get('all_native_processes_with_host_mapped_DSO_and_resource_proof')is True and
         certificate.get('all_shared_batches_terminal_at_closeout')is True,
         'actual four-group standalone native execution scope')
    need(certificate.get('numeric_policy')=='metrics-only'and
         certificate.get('numerical_correctness_claimed')is False and
         certificate.get('full_ResNet20_encrypted')is False and
         all(certificate.get(k)==0 for k in ('new_keys','evaluator_decryptions',
             'intermediate_client_decryptions','reencryptions','host_private_file_content_reads','host_private_file_hashes')),
         'numeric observations and private/evaluator boundaries')
    for role,key in (('runtime','runtime_artifact'),('client','client_artifact')):
        need(native[role]==contract[key],'actual independent runtime/client artifacts')
        record(native[role],within=work)
    record(native['compiler_profile'],within=work);record(native['operator_spec'])
    need(native['compiler_profile']['sha256']==COMPILER_PROFILE and
         native['operator_spec']['sha256']==OPERATOR_SPEC,'registered compiler profile and operator specification')
    need(native['operator_spec']['path']==contract['operator_spec_source'],'actual operator source')
    spec=base.read_json(native['operator_spec']['path'])
    declared=sorted(name for name,row in spec['operators'].items()if row['supported'])
    need(declared==sorted(contract['actual_covered_ops']+contract['source_implemented_but_not_covered_ops'])and
         spec['operators']['sub_cp']['supported']is False and
         spec['context']['poly_degree']==32768 and spec['context']['rns_moduli_log2']==[48]*16 and
         spec['context']['default_scale_log2']==33,'exact supported operator surface')
    assets=native['assets'];need(type(assets)is dict and set(assets)=={'public_directory',
        'public_manifest_sha256','client_private_directory'},'private contents have no public host hash')
    public=public_path(assets['public_directory'],within=work,directory=True)
    need(base.file_hash(public/'manifest.json')==assets['public_manifest_sha256']==
         origin['public_manifest_sha256'],'actual public origin manifest')
    # Existence/path checks only. Private file names, contents and hashes are
    # exclusively the independent client's responsibility.
    private=base.path(assets['client_private_directory'],within=work,directory=True)
    need(private.name=='client-private'and public.name=='public'and private.parent==public.parent,
         'separate client-private/public sibling directories')
    modules=native['modules'];root=Path(agent_root).resolve()
    need(type(modules)is dict and set(modules)=={'native_sdk_manifest_v5.py','native_key_origin_intake.py','poseidon_sdk_manifest.py',
         'native_sdk_manifest.py','poseidon_runtime_paths.py','poseidon_backend_identity.py'},'registered SDK source closure')
    for name,row in modules.items():
        p=record(row);need(p.name==name and p.is_relative_to(root),'actual registered Agent source')
        if name=='native_sdk_manifest_v5.py':need(p==Path(__file__).resolve(),'actual v5 reader source')
    need(native['readiness']==dict(compiler_and_origin_attestation=True,
         standalone_native_runtime_measured=True,normal_Application_profile=False,
         fresh_public_input_client=False,full_model=False),'explicit staged capability boundaries')
    return copy.deepcopy(value)


def load(filename,work,platform,agent_root):
    p=public_path(str(filename),within=work)
    return validate(base.read_json(p),work,platform,agent_root)


def attest(value,work,agent_root):
    value=validate(value,work,value['platform'],agent_root)
    a=_compiler_ancestor(value['native'],work);source=Path(value['backend_source_root'])
    roots=dict(runtime=Path(value['runtime_source']),api=source/'src/poseidon/runtime_api',
               runner=source/'src/poseidon/tools/agent_runtime')
    need({k:base.tree(p)for k,p in roots.items()}==a['source_trees'],'unchanged original compiler/API sources')
    build=Path(value['compiler_build'])
    for role,path in (('compiler',build/'bin/hecate-opt'),('frontend',build/'lib/libHecateFrontend.so')):
        need(base.file_hash(base.path(str(path),within=work))==a['artifacts'][role],
             'actual original compiler/frontend bytes')
    need(base.file_hash(build/'CMakeCache.txt')==a['cmake_caches']['compiler'],'original compiler build configuration')
    receipt=base.read_json(record(a['build_receipts']['compiler'],within=work))
    need(receipt['status']=='succeeded'and
         receipt['artifacts']['bin/hecate-opt']['sha256']==a['artifacts']['compiler']and
         receipt['artifacts']['lib/libHecateFrontend.so']['sha256']==a['artifacts']['frontend'],'original compiler build receipt')
    for name,expected in a['environment_files'].items():
        rel=Path(name);need(not rel.is_absolute()and '..'not in rel.parts,'environment source name')
        p=(Path(agent_root)/rel).resolve();need(p.is_relative_to(Path(agent_root).resolve())and
             base.file_hash(p)==expected,'unchanged original build environment')
    need(base.file_hash(value['source_lock'])==SOURCE_LOCK and
         base.file_hash(value['resource_policy'])==a['resource_policy_sha256'],'original source/resource ancestry')
    lock=base.read_json(value['source_lock'])
    need(lock['source_trees']==a['source_trees']and
         lock['compiler_source_binding']==receipt['source_binding'],'original compiler/source lock chain')
    return value


def execution_fingerprint(value):
    return digest(dict(format=FORMAT,source_lock=value['source_lock_sha256'],
         library_overlay=value['library_overlay'],native=value['native']))


def backend_identity(value,platform):
    native=value['native'];contract=bound(native['physical_io_contract'])
    return dict(format='poseidon-backend-identity-v3',platform=platform,
        build_configuration=dict(environment_files=value['environment_files'],
            compiler_cmake_cache_sha256=value['cmake_caches']['compiler'],
            library_overlay_report_sha256=value['library_overlay']['build_report']['sha256']),
        source_lock_sha256=value['source_lock_sha256'],source_lock=base.read_json(value['source_lock']),
        runtime_dacapo_tree_sha256=value['source_trees']['runtime'],api_tree_sha256=value['source_trees']['api'],
        runner_tree_sha256=value['source_trees']['runner'],compiler_sha256=value['artifacts']['compiler'],
        frontend_sha256=value['artifacts']['frontend'],executor_sha256=native['runtime']['sha256'],
        poseidon_library_sha256=value['artifacts']['poseidon_library'],
        compiler_profile_sha256=native['compiler_profile']['sha256'],operator_spec_sha256=native['operator_spec']['sha256'],
        execution_profile=POLICY,native_sdk_execution_fingerprint=execution_fingerprint(value),
        native_runtime_sha256=native['runtime']['sha256'],native_client_sha256=native['client']['sha256'],
        native_capability_binding=contract['binding'],native_key_epoch=contract['epoch'],
        actual_backend_numeric_policy='metrics-only',full_model_capability=False,
        normal_Application_profile_ready=False)
