"""Data-only preflight for fresh original-epoch ciphers; no keys or ciphertext loading."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

FORMAT = 'poseidon-native-sdk-cipher-handoff-prototype-v1'
EPOCH = 'ad7439a7013ff2cf8f215d97daad155a'
ORIGIN = 'a594528b70cbff43f1b62ec6e4e0c23a4568aa4ffacf76a4031d9d97b2e60e8e'
CAPABILITY = 'e4e8fe348b7f875d7af1283ca4c51da23090c17f591f42f677a5b4d56d914c9e'
INPUT_SCALE = 274877947904.0026
OUTPUT_SCALE = 17592186019840.062
HASH_FIELDS = {'public_manifest_sha256', 'runtime_plan_sha256', 'operator_spec_sha256',
               'source_plan_sha256', 'link_record_sha256', 'task_binding'}
FIELDS = {'name', 'type', 'group', 'index', 'piece', 'polys', 'degree', 'q_count',
          'ntt', 'parms_id', 'scale', 'correction', 'level', 'bytes', 'sha256'}
HEX = re.compile('[0-9a-f]{64}\\Z')
VALUE = re.compile('(?:0|[1-9][0-9]{0,8})\\Z')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def strict(raw):
    need(type(raw) is bytes and 0 < len(raw) <= 256*1024, 'sdk_bounded_manifest_json')
    def pairs(items):
        value = {}
        for name, item in items:
            need(name not in value, 'sdk_duplicate_json_key')
            value[name] = item
        return value
    def nonfinite(_):
        raise ValueError('sdk_nonfinite_json')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)
    def depth(item, n=0):
        need(n <= 64, 'sdk_json_depth')
        if type(item) is dict:
            for v in item.values(): depth(v, n+1)
        elif type(item) is list:
            for v in item: depth(v, n+1)
    depth(value)
    return value


def schema(manifest, stage, binding, expected_ids, parameter_ids):
    need(stage in ('input', 'evaluated'), 'sdk_cipher_stage')
    need(type(binding) is dict and set(binding) == HASH_FIELDS and
         all(type(v) is str and HEX.fullmatch(v) for v in binding.values()), 'sdk_bound_identity_fields')
    expected = dict(format=FORMAT, stage=stage, epoch=EPOCH, origin=ORIGIN,
                    capability_binding=CAPABILITY, **binding)
    need(type(manifest) is dict and set(manifest) == set(expected) | {'files'}, 'sdk_exact_handoff_fields')
    need(all(manifest[k] == v for k, v in expected.items()), 'sdk_original_epoch_task_artifact_handoff')
    need(type(expected_ids) is list and 1 <= len(expected_ids) <= 16 and
         all(type(v) is str and VALUE.fullmatch(v) for v in expected_ids) and
         len(set(expected_ids)) == len(expected_ids), 'sdk_expected_value_ids')
    rows = manifest['files']
    need(type(rows) is list and len(rows) == len(expected_ids), 'sdk_exact_handoff_count')
    count, scale, kind, prefix = (14, INPUT_SCALE, 'input', 'input-') if stage == 'input' else (
                                  2, OUTPUT_SCALE, 'odd_refreshed', 'output-')
    need(type(parameter_ids) is dict and str(count-1) in parameter_ids, 'sdk_original_parameter_ids')
    pid = parameter_ids[str(count-1)]
    need(type(pid) is list and len(pid) == 4 and all(type(v) is int and 0 <= v < 2**64 for v in pid),
         'sdk_original_parameter_id_types')
    checked = {}
    for identifier, row in zip(expected_ids, rows):
        need(type(row) is dict and set(row) == {'value_id', 'record'} and
             type(row['value_id']) is str and row['value_id'] == identifier, 'sdk_declared_input_output_order')
        r = row['record']
        need(type(r) is dict and set(r) == FIELDS, 'sdk_exact_cipher_record_fields')
        name = prefix + identifier + '.bin'
        need(r['name'] == name and r['type'] == kind, 'sdk_exact_cipher_name_and_type')
        integer_fields = dict(group=-1, index=0, piece=0, polys=2, degree=32768,
                              q_count=count, correction=1, level=count-1,
                              bytes=113+2*32768*count*8)
        need(all(type(r[k]) is int and r[k] == v for k, v in integer_fields.items()),
             'sdk_exact_cipher_integer_metadata')
        need(r['ntt'] is True and type(r['parms_id']) is list and
             all(type(v) is int for v in r['parms_id']) and r['parms_id'] == pid,
             'sdk_exact_cipher_context')
        need(type(r['scale']) in (float, int) and math.isfinite(r['scale']) and
             struct.pack('<d', r['scale']) == struct.pack('<d', scale), 'sdk_exact_actual_scale_no_snap')
        need(type(r['sha256']) is str and HEX.fullmatch(r['sha256']), 'sdk_cipher_sha256')
        checked[name] = r
    return checked


def preflight(directory, expected_sha, stage, binding, expected_ids, parameter_ids):
    directory = Path(directory)
    need(directory.is_dir() and not directory.is_symlink(), 'sdk_real_owned_handoff_directory')
    manifest_path = directory/'manifest.json'
    need(manifest_path.is_file() and not manifest_path.is_symlink() and
         manifest_path.stat().st_size <= 256*1024 and sha(manifest_path) == expected_sha,
         'sdk_bound_manifest_bytes')
    manifest = strict(manifest_path.read_bytes())
    rows = schema(manifest, stage, binding, expected_ids, parameter_ids)
    need({p.name for p in directory.iterdir()} == set(rows) | {'manifest.json'}, 'sdk_exact_directory_inventory')
    for name, r in rows.items():
        path = directory/name
        need(path.is_file() and not path.is_symlink() and
             path.stat().st_uid == directory.stat().st_uid and not path.stat().st_mode & 0o022 and
             path.stat().st_size == r['bytes'] and sha(path) == r['sha256'], 'sdk_owned_complete_cipher_bytes')
        with path.open('rb') as stream:
            h = stream.read(113)
        need(len(h) == 113, 'sdk_complete_cipher_header')
        magic, size, major, minor, compressed, reserved, total = struct.unpack_from('<HBBBBHQ', h)
        pid = list(struct.unpack_from('<4Q', h, 16))
        ntt, polys, degree, count, scale, correction = struct.unpack_from('<BQQQdQ', h, 48)
        need(magic == 0xA15E and size == 16 and compressed == reserved == 0 and total == r['bytes'] and
             pid == r['parms_id'] and ntt == 1 and polys == 2 and degree == 32768 and
             count == r['q_count'] and struct.pack('<d', scale) == struct.pack('<d', r['scale']) and
             correction == 1, 'sdk_original_serialized_cipher_state')
        need(struct.unpack_from('<HBBBBHQ', h, 89) ==
             (magic, 16, major, minor, 0, 0, 24+2*32768*count*8) and
             struct.unpack_from('<Q', h, 105)[0] == 2*32768*count, 'sdk_original_unseeded_buffer_header')
    return manifest
