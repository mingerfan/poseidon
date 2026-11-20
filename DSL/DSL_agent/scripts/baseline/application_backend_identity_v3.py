"""Data-only v3 identity projection; SDK attestation and execution gates stay separate.

Bind the complete original v3 record in every stage. The compiler cache and
library overlay retain their actual names; no executor/legacy cache is invented.
"""
import copy
import json
import math
import re

FORMAT = 'poseidon-application-backend-projection-v3'
POLICY = 'complete-actual-backend-record-v3'
STAGES = ('generation', 'compilation', 'execution', 'validation')
PROFILE = 'poseidon-valid-epoch-public-continuation-v1'
HASH_FIELDS = frozenset({
    'source_lock_sha256', 'runtime_dacapo_tree_sha256', 'api_tree_sha256',
    'runner_tree_sha256', 'compiler_sha256', 'frontend_sha256',
    'executor_sha256', 'poseidon_library_sha256', 'compiler_profile_sha256',
    'operator_spec_sha256', 'native_sdk_execution_fingerprint',
    'native_runtime_sha256', 'native_client_sha256', 'native_capability_binding',
})
REQUIRED = HASH_FIELDS | {
    'format', 'platform', 'build_configuration', 'source_lock', 'execution_profile',
    'native_key_epoch', 'actual_backend_numeric_policy', 'full_model_capability',
    'normal_Application_profile_ready',
}


def need(ok, reason):
    if not ok:
        raise ValueError('Backend identity v3: ' + reason)


def hash_value(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def json_data(value):
    """Bound depth, size and types before copying or encoding caller data."""
    count = 0
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        need(count <= 65536 and depth <= 32, 'bounded JSON structure')
        kind = type(item)
        if kind is dict:
            need(all(type(k) is str and len(k) <= 1024 for k in item), 'string JSON keys')
            stack.extend((v, depth+1) for v in item.values())
        elif kind is list:
            stack.extend((v, depth+1) for v in item)
        elif kind is str:
            need(len(item) <= 65536, 'bounded string')
        elif kind is int:
            need(item.bit_length() <= 128, 'bounded integer')
        elif kind is float:
            need(math.isfinite(item), 'finite JSON number')
        else:
            need(item is None or kind is bool, 'plain JSON types')
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    need(len(raw) <= 2*1024**2, 'bounded JSON bytes')


def project(backend, stage):
    need(type(stage) is str and stage in STAGES, 'known stage')
    need(type(backend) is dict and REQUIRED <= set(backend), 'complete v3 fields')
    json_data(backend)
    need(backend['format'] == 'poseidon-backend-identity-v3', 'explicit v3 format')
    need(all(hash_value(backend[k]) for k in HASH_FIELDS), 'actual artifact hashes')
    need(type(backend['platform']) is dict and backend['platform'] and
         type(backend['source_lock']) is dict and backend['source_lock'], 'platform/source records')
    need(backend['execution_profile'] == PROFILE and
         backend['actual_backend_numeric_policy'] == 'metrics-only', 'registered current profile')
    need(type(backend['native_key_epoch']) is str and
         re.fullmatch('[0-9a-f]{32}', backend['native_key_epoch']) is not None, 'epoch identity')
    need(type(backend['full_model_capability']) is bool and
         type(backend['normal_Application_profile_ready']) is bool, 'explicit capability booleans')
    need(backend['executor_sha256'] == backend['native_runtime_sha256'], 'actual native executor relation')
    config = backend['build_configuration']
    need(type(config) is dict and {'environment_files', 'compiler_cmake_cache_sha256',
         'library_overlay_report_sha256'} <= set(config), 'actual v3 build fields')
    need(hash_value(config['compiler_cmake_cache_sha256']) and
         hash_value(config['library_overlay_report_sha256']), 'actual compiler/overlay hashes')
    files = config['environment_files']
    need(type(files) is dict and 1 <= len(files) <= 32 and
         all(type(k) is str and k and not k.startswith('/') and
             all(part not in ('', '.', '..', '.env', 'client-private') for part in k.split('/')) and
             hash_value(v) for k, v in files.items()), 'public environment hash records')
    # Include unknown future fields and all readiness values as data. This
    # projection grants no runtime/Provider/normal-Application authority.
    return dict(format=FORMAT, policy=POLICY, stage=stage, backend=copy.deepcopy(backend))
