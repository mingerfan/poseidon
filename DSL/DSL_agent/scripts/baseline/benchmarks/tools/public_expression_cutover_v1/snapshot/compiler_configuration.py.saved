"""Versioned immutable compiler identities; no paths, credentials or network.

Legacy requests omit this extension and retain their original IDs/waterline40.
An explicit configuration binds exact profile bytes, EVA pipeline and waterline.
Only the trusted launcher selects a configuration; candidates cannot override it.
"""
import copy
import hashlib
import json

PROFILE_SHA256 = 'ab53faeac14298e846a4ea8e9b2cbfaef4469a6b03ba25b80f37c79387fd246e'
LEGACY_CONFIGURATIONS = {'seal-cpu-eva-w40-v1': 40, 'seal-cpu-eva-w45-v1': 45}
COMPACT_CONFIGURATION = 'poseidon-cpu-eva-w45-compact-v1'
TRACE_SERIALIZATION_POLICY = 'hecate-dense-f64-shortest-v1'
CONFIGURATIONS = {COMPACT_CONFIGURATION:45, **LEGACY_CONFIGURATIONS, 'poseidon-cpu-eva-w40-v1':40, 'poseidon-cpu-eva-w45-v1':45, 'poseidon-cuda-eva-w45-v1':45}
ACTIVE_CONFIGURATIONS = (COMPACT_CONFIGURATION,'poseidon-cpu-eva-w40-v1','poseidon-cpu-eva-w45-v1','poseidon-cuda-eva-w45-v1')
CPU_CONFIGURATIONS = tuple(name for name in ACTIVE_CONFIGURATIONS if name.startswith("poseidon-cpu-"))
POSEIDON_PROFILE_SHA256 = 'acf39bf80a0049c3344c147ae4eaac5ee03665b8f5f76cab6c91195aa1429037'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configuration(name, profile_sha256=None, *, schema=None):
    require(type(name) is str and name in CONFIGURATIONS, 'Unknown compiler configuration')
    if name in ACTIVE_CONFIGURATIONS:
        profile_sha256 = POSEIDON_PROFILE_SHA256 if profile_sha256 is None else profile_sha256
        require(profile_sha256 == POSEIDON_PROFILE_SHA256, 'Poseidon compiler profile hash mismatch')
        version = (4 if name == COMPACT_CONFIGURATION else 3 if name.startswith('poseidon-cpu') else 2) if schema is None else schema
        require(type(version) is int and version in ((4,) if name == COMPACT_CONFIGURATION else (2,3)),
                'Unsupported Poseidon compiler configuration version')
        require(version != 3 or name.startswith('poseidon-cpu'), 'CPU entry policy only')
        body = dict(schema=version,name=name,backend='poseidon_cpu' if name.startswith('poseidon-cpu') else 'poseidon_cuda',
                    pipeline='eva-runtime-plan',ckks_config_sha256=profile_sha256,waterline=CONFIGURATIONS[name],
                    runtime_commit='23c8b94202a14d3e322f87c50434253d56acb9f7',
                    dacapo_commit='edcebf5de955dd2a5cd1b87726b92355276e4390',
                    api_commit='62dbe22559667199487edb8ecdcb06624f66e6d7',
                    parameter_identity='poseidon-agent-n15-q13-p1-tc128-v1',
                    source_lock_sha256='5e8d7adebd0cac8e9748bcddf0dc5ea408b7123095ff79199cdf47738de84647',
                    operator_spec_sha256='a16b1ae98958370731a1753db2ea023a7c0d6658f1246c330b44d27f161e790e')
        if version in (3,4):
            body['runtime_plan_entry_policy'] = 'hecate-private-symbol-dce-v1'
        if version == 4:
            body['trace_serialization_policy'] = TRACE_SERIALIZATION_POLICY
        prefix = ('poseidon-compiler-configuration-v'+str(version)+'\0').encode()
        return dict(body,identity_sha256=hashlib.sha256(prefix+canonical(body)).hexdigest())
    require(schema is None or type(schema) is int and schema == 1, 'Unsupported legacy compiler configuration version')
    profile_sha256 = PROFILE_SHA256 if profile_sha256 is None else profile_sha256
    require(profile_sha256 == PROFILE_SHA256, 'Compiler configuration profile hash mismatch')
    body = dict(schema=1, name=name, backend='upstream_SEAL_HEVM_CPU', pipeline='eva',
                ckks_config_sha256=profile_sha256, waterline=CONFIGURATIONS[name])
    return dict(body, identity_sha256=hashlib.sha256(b'poseidon-compiler-configuration-v1\0'+canonical(body)).hexdigest())


def validate_configuration(value, profile_sha256):
    require(type(value) is dict, 'Compiler configuration must be an object')
    expected = configuration(value.get('name'), profile_sha256, schema=value.get('schema'))
    # Canonical bytes also distinguish 1 from True and 45 from 45.0.
    require(canonical(value) == canonical(expected), 'Changed compiler configuration')
    return copy.deepcopy(expected)


def request_configuration(request):
    if 'compiler_configuration' not in request:
        return None
    result = validate_configuration(request['compiler_configuration'], request.get('compiler_profile_sha256'))
    body = {k: v for k,v in request.items() if k != 'request_id'}
    if "public_weight_binding" in request:
        from public_weight_context import request_digest
        expected_hash=request_digest(request)
    else:
        expected_hash=hashlib.sha256(canonical(body)).hexdigest()
    require(expected_hash == request.get('request_id'),'Compiler configuration request hash mismatch')
    return result


def verify_execution_configuration(request, profile_sha256, waterline):
    require(type(waterline) is int, 'Compiler waterline must be an integer')
    require(request.get('compiler_profile_sha256') == profile_sha256, 'Compiler profile changed')
    value = request_configuration(request)
    expected = value['waterline'] if value is not None else 40
    require(waterline == expected, 'Compiler waterline differs from immutable request')
    return value


def verify_artifact_configuration(request, gate, profile_sha256):
    value = request_configuration(request)
    if value is None:
        # Preserve old diagnostic workers that explicitly reuse legacy I/O only.
        # The normal legacy candidate launcher still compiles at fixed40.
        return None
    verify_execution_configuration(request, profile_sha256, value['waterline'])
    require(gate['initial_level'] == 13 and gate['arg_level'] == [13]*len(gate['arg_scale']) and
            gate['arg_scale'] == [value['waterline']]*len(gate['arg_level']),
            'Compiled artifact input precision differs from immutable request')
    return value


def options(args):
    name = getattr(args, 'compiler_configuration', None)
    if name is None:
        return []
    configuration(name)
    return ['--compiler-configuration', name]


def validate_continuation(prior, args):
    before = prior.get('compiler_configuration')
    if before is not None:
        before = validate_configuration(before, before["ckks_config_sha256"])
    name = getattr(args, 'compiler_configuration', None)
    after = configuration(name) if name is not None else None
    require(before == after, 'Changed compiler configuration requires a fresh batch')
