"""Opt-in bounded lossless wire for public-basis requests and model data."""
import base64
import copy
import hashlib
import json
import math
import zlib
from benchmark_graph import require, canonical, is_graph
from provenance_json_transport import WIRE_BYTES, LOGICAL_BYTES, measure

VERSION = 'capacity-public-tensors-v11'
REQUEST_FORMAT = 'poseidon-public-basis-request-wire-v1'
MODEL_FORMAT = 'poseidon-public-basis-model-wire-v1'
VIEW = '__poseidon_public_basis_weight_view_v1__'


def selected(request):
    if type(request) is not dict or type(request.get('constant_origins')) is not dict:
        return False
    descriptor=request['constant_origins'].get('capacity_tensor_packing')
    return type(descriptor) is dict and descriptor.get('version') == VERSION


def is_packet(value):
    return type(value) is dict and value.get('format') in (REQUEST_FORMAT, MODEL_FORMAT)


def loads(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate public basis wire key')
            value[key] = item
        return value
    def invalid(value):
        raise ValueError('Nonfinite public basis wire value')
    def real(token):
        value = float(token)
        require(math.isfinite(value), "Nonfinite public basis wire number")
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=real)


def _encode(body, original, format_):
    logical_bytes = measure(original, LOGICAL_BYTES)
    logical_raw = canonical(original)
    assert len(logical_raw) == logical_bytes
    body_raw = canonical(body)
    require(len(body_raw) <= LOGICAL_BYTES, 'Public basis compact logical bytes')
    packet = dict(format=format_, codec='zlib-base64-v1', logical_bytes=logical_bytes,
                  logical_sha256=hashlib.sha256(logical_raw).hexdigest(),
                  body=base64.b64encode(zlib.compress(body_raw, 6)).decode('ascii'))
    require(len(canonical(packet)) <= WIRE_BYTES, 'Original public basis wire byte bound')
    return packet


def _all_float(value):
    if type(value) is list:
        return bool(value) and all(_all_float(v) for v in value)
    return type(value) is float


def encode_request(compacted, original):
    require(selected(original) and selected(compacted), 'Versioned public basis request wire')
    value = dict(compacted)
    value['model'] = dict(compacted['model'])
    segments = original['constant_origins']['capacity_tensor_packing']['scalar_bank']['segments']
    # Float leaves are represented once, through the exact original coefficient bank.
    # Integer/mixed containers retain their original JSON representation.
    value['model']['constants'] = {name: {VIEW: name} if name in segments and _all_float(data) else data
                                    for name, data in compacted['model']['constants'].items()}
    return _encode(value, original, REQUEST_FORMAT)


def encode_model(model):
    require(is_graph(model), 'Public basis model wire')
    return _encode(model, model, MODEL_FORMAT)


def decode_compact(packet):
    require(type(packet) is dict and set(packet) == {'format', 'codec', 'logical_bytes', 'logical_sha256', 'body'}
            and packet['format'] in (REQUEST_FORMAT, MODEL_FORMAT) and packet['codec'] == 'zlib-base64-v1',
            'Public basis wire fields')
    require(len(canonical(packet)) <= WIRE_BYTES and type(packet['logical_bytes']) is int
            and 0 < packet['logical_bytes'] <= LOGICAL_BYTES, 'Original public basis wire/logical limits')
    require(type(packet['body']) is str and len(packet['body']) <= WIRE_BYTES, 'Public basis encoded body')
    try:
        compressed = base64.b64decode(packet['body'], validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, LOGICAL_BYTES + 1)
    except (ValueError, zlib.error) as error:
        raise ValueError('Invalid public basis compressed body') from error
    require(len(raw) <= LOGICAL_BYTES and decoder.eof and not decoder.unused_data
            and not decoder.unconsumed_tail, 'Bounded complete public basis decompression')
    value = loads(raw)
    require(type(value) is dict, 'Public basis wire document')
    if packet['format'] == REQUEST_FORMAT:
        require(selected(value), 'Public basis request version changed in wire')
    else:
        require(is_graph(value), 'Public basis model format')
    return value


def restore_model_views(request):
    require(selected(request), 'Public basis weight views version')
    bank = request['constant_origins']['capacity_tensor_packing']['scalar_bank']
    pages = bank['pages']
    period = request['layout']['input_slot_period']
    public = request['public_constants']
    require(type(period) is int and 4 <= period <= 4096 and type(bank['elements']) is int
            and 0 <= bank['elements'] <= 256 * period, 'Public basis bank extent')
    require(type(pages) is list and len(pages) <= 256 and all(type(public.get(k)) is list and
            len(public[k]) == period for k in pages), 'Public basis coefficient bank pages')

    def rebuild(shape, values):
        if not shape:
            return next(values)
        return [rebuild(shape[1:], values) for _ in range(shape[0])]

    constants = request['model']['constants']
    for name, value in list(constants.items()):
        if type(value) is not dict or set(value) != {VIEW}:
            continue
        require(value[VIEW] == name and name in bank['segments'], 'Public basis original tensor view')
        segment = bank['segments'][name]
        start, count = segment['offset'], segment['elements']
        require(type(start) is int and type(count) is int and start >= 0 and count > 0 and
                start + count <= bank['elements'], 'Public basis view extent')
        shape = segment['shape']
        require(type(shape) is list and len(shape) <= 32 and all(type(n) is int and n > 0 for n in shape)
                and math.prod(shape) == count, 'Public basis original tensor dimensions')
        numbers = [public[pages[i // period]][i % period] for i in range(start, start + count)]
        require(all(type(v) is float and math.isfinite(v) and abs(v) <= 1024 for v in numbers),
                'Public basis original binary64 values')
        values = iter(numbers)
        constants[name] = rebuild(segment['shape'], values)
        try:
            next(values)
        except StopIteration:
            pass
        else:
            raise ValueError('Public basis original tensor shape')


def verify_logical(packet, value):
    require(measure(value, LOGICAL_BYTES) == packet['logical_bytes'] and
            hashlib.sha256(canonical(value)).hexdigest() == packet['logical_sha256'],
            'Public basis original logical bytes/hash')


def decode_data_file(value, name):
    if not is_packet(value):
        return value
    if name == 'model.json' and value['format'] == MODEL_FORMAT:
        model = decode_compact(value)
        verify_logical(value, model)
        return model
    if name == 'request.json' and value['format'] == REQUEST_FORMAT:
        from unified_request_storage import restore_public_constants
        restore_public_constants(value)
        return value
    return value
