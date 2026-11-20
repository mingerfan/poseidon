"""Read a bounded, trusted evaluation dataset without model execution."""
import copy
import json
import math
import struct
import zlib
from pathlib import Path

FORMAT = 'poseidon-fixed-evaluation-inputs-v1'
MAX_CASES = 256
MAX_VALUES = 524288
MAX_MANIFEST_BYTES = 4 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def _path(value, root):
    require(type(value) is str and Path(value).is_absolute(), 'Absolute evaluation data path')
    path = Path(value)
    require(path.resolve() == path and path.is_relative_to(root) and path.is_file(),
            'Evaluation data must stay inside the approved results directory')
    return path


def _vector(source, count, root):
    require(type(source) is dict and source.get('encoding') in ('f64', 'zlib-f64'),
            'Evaluation data encoding')
    offset = source.get('offset', 0)
    require(type(offset) is int and offset >= 0 and type(count) is int and count > 0,
            'Evaluation data extent')
    path = _path(source.get('path'), root)
    if source['encoding'] == 'f64':
        with path.open('rb') as stream:
            stream.seek(offset)
            raw = stream.read(count * 8)
    else:
        require(offset == 0 and path.stat().st_size <= MAX_MANIFEST_BYTES,
                'Compressed evaluation data extent')
        decoder = zlib.decompressobj()
        raw = decoder.decompress(path.read_bytes(), count * 8 + 1)
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                'Complete bounded evaluation data required')
    try:
        values = list(struct.unpack('<' + str(count) + 'd', raw))
    except struct.error as error:
        raise ValueError('Evaluation data does not match its declared shape') from error
    require(all(math.isfinite(v) for v in values), 'Finite evaluation values required')
    return values


def _reshape(shape, values):
    iterator = iter(values)
    def build(axis):
        return next(iterator) if axis == len(shape) else [build(axis + 1) for _ in range(shape[axis])]
    return build(0)


def read_manifest(value, *, allowed_root):
    root = Path(allowed_root).resolve()
    if type(value) is dict:
        require(len(json.dumps(value, allow_nan=False).encode()) <= MAX_MANIFEST_BYTES,
                'Evaluation manifest size')
        return copy.deepcopy(value)
    path = _path(str(Path(value).absolute()), root)
    require(path.stat().st_size <= MAX_MANIFEST_BYTES, 'Evaluation manifest size')
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, 'Duplicate evaluation manifest field')
            result[key] = item
        return result
    def constant(value):
        raise ValueError('Nonfinite evaluation manifest field')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)


def load(value, input_specs, output_elements, *, allowed_root):
    """Resolve ordered inputs and retained references supplied by the evaluator."""
    root = Path(allowed_root).resolve()
    manifest = read_manifest(value, allowed_root=root)
    require(type(manifest) is dict and manifest.get('format') == FORMAT and
            manifest.get('inputs') == input_specs, 'Evaluation input declarations')
    require(type(output_elements) is int and output_elements > 0 and
            manifest.get('output_elements') == output_elements, 'Evaluation output declaration')
    mode = manifest.get('reference_mode')
    require(mode in ('retained', 'compute'), 'Evaluation reference mode')
    cases = manifest.get('cases')
    require(type(cases) is list and 1 <= len(cases) <= MAX_CASES, 'Evaluation case count')
    names = [s['name'] for s in input_specs]
    require(len(set(names)) == len(names), 'Distinct model input names')
    counts = {}
    for spec in input_specs:
        shape = spec['shape']
        require(type(shape) is list and len(shape) <= 32 and
                all(type(n) is int and n > 0 for n in shape), 'Evaluation input dimensions')
        counts[spec['name']] = math.prod(shape)
    require(len(cases) * (sum(counts.values()) + (output_elements if mode == 'retained' else 0)) <= MAX_VALUES,
            'Evaluation data value budget')
    seen = set()
    resolved = []
    for case in cases:
        require(type(case) is dict and type(case.get('id')) is str and
                0 < len(case['id']) <= 128 and case['id'] not in seen, 'Distinct evaluation case identity')
        seen.add(case['id'])
        sources = case.get('values')
        require(type(sources) is dict and set(sources) == set(names), 'Evaluation input names')
        inputs = {s['name']: _reshape(s['shape'], _vector(sources[s['name']], counts[s['name']], root))
                  for s in input_specs}
        reference = None
        if mode == 'retained':
            require(type(case.get('reference')) is dict and
                    case['reference'].get('role') == 'declared_model_float64', 'Retained reference role')
            reference = _vector(case['reference'], output_elements, root)
        else:
            require(case.get('reference') is None, 'Computed reference mode')
        resolved.append(dict(id=case['id'], values=inputs, reference=reference))
    return dict(manifest=manifest, cases=resolved, reference_mode=mode)
