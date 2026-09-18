"""Versioned immutable compiler identities; no paths, credentials or network.

Legacy requests omit this extension and retain their original IDs/waterline40.
An explicit configuration binds exact profile bytes, EVA pipeline and waterline.
Only the trusted launcher selects a configuration; candidates cannot override it.
"""
import copy
import hashlib
import json

PROFILE_SHA256 = 'ab53faeac14298e846a4ea8e9b2cbfaef4469a6b03ba25b80f37c79387fd246e'
CONFIGURATIONS = {'seal-cpu-eva-w40-v1': 40, 'seal-cpu-eva-w45-v1': 45}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configuration(name, profile_sha256=PROFILE_SHA256):
    require(type(name) is str and name in CONFIGURATIONS, 'Unknown compiler configuration')
    require(profile_sha256 == PROFILE_SHA256, 'Compiler configuration profile hash mismatch')
    body = dict(schema=1, name=name, backend='upstream_SEAL_HEVM_CPU', pipeline='eva',
                ckks_config_sha256=profile_sha256, waterline=CONFIGURATIONS[name])
    return dict(body, identity_sha256=hashlib.sha256(b'poseidon-compiler-configuration-v1\0'+canonical(body)).hexdigest())


def validate_configuration(value, profile_sha256):
    require(type(value) is dict, 'Compiler configuration must be an object')
    expected = configuration(value.get('name'), profile_sha256)
    # Canonical bytes also distinguish 1 from True and 45 from 45.0.
    require(canonical(value) == canonical(expected), 'Changed compiler configuration')
    return copy.deepcopy(expected)


def request_configuration(request):
    if 'compiler_configuration' not in request:
        return None
    result = validate_configuration(request['compiler_configuration'], request.get('compiler_profile_sha256'))
    body = {k: v for k,v in request.items() if k != 'request_id'}
    require(hashlib.sha256(canonical(body)).hexdigest() == request.get('request_id'),
            'Compiler configuration request hash mismatch')
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
        before = validate_configuration(before, PROFILE_SHA256)
    name = getattr(args, 'compiler_configuration', None)
    after = configuration(name) if name is not None else None
    require(before == after, 'Changed compiler configuration requires a fresh batch')
