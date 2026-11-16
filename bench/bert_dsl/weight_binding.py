"""Bind generated parameters to an existing, SHA-checked dense plan asset.

A seed alone does not guarantee bitwise float32 normal samples across numeric
library/libm environments. Dense parameters are loaded; the remaining LN
parameters must independently match the frozen fixture hashes or fail closed.
"""
from dataclasses import asdict
from functools import wraps
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from backend import digest


_DENSE = {'q_weight', 'k_weight', 'v_weight', 'o_weight', 'ffn_up_weight', 'ffn_down_weight'}


def _file_digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            hasher.update(block)
    return 'sha256:' + hasher.hexdigest()


def _schema_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError(f'{function.__name__}: invalid frozen data schema: {exc}') from exc
    return checked


def _compatible_config(source, config, allow_prefix_layers):
    source = dict(source)
    expected = asdict(config)
    layers = source.get('layers')
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        return False
    if allow_prefix_layers:
        if config.layers > layers:
            return False
        source['layers'] = config.layers
    return source == expected


@_schema_errors
def bind_frozen_weights(regenerated, config, seed, directory, *, allow_prefix_layers=False):
    """Return independent layer dictionaries with exact frozen dense arrays.

    This does not import changed approximations, geometry, scale or scheduling.
    Golden fixtures without all-parameter hashes are deliberately unsupported.
    """
    directory = Path(directory)
    fixture_path, manifest_path = directory / 'fixture.json', directory / 'compact-linear.json'
    fixture_raw, manifest_raw = fixture_path.read_bytes(), manifest_path.read_bytes()
    fixture, manifest = json.loads(fixture_raw), json.loads(manifest_raw)
    model = fixture['model']
    if not _compatible_config(model['config'], config, allow_prefix_layers) or model['seed'] != seed:
        raise ValueError('frozen weights require identical model config and seed (or explicit shorter layer prefix)')
    source_layers = model['config']['layers']
    if manifest.get('format_version') != 1 or manifest.get('dtype') != '<f4':
        raise ValueError('unsupported frozen dense manifest')
    weight_path = directory / manifest['weight_file']
    weight_digest = _file_digest(weight_path)
    if weight_digest != manifest['weight_sha256']:
        raise ValueError('frozen dense weight file SHA256 mismatch')
    if len(regenerated) != config.layers:
        raise ValueError('regenerated layer count disagrees with frozen model')
    result = [dict(layer) for layer in regenerated]
    seen = set()
    size = weight_path.stat().st_size
    with weight_path.open('rb') as source:
        for row in manifest['matrices']:
            match = re.fullmatch(r'layer(\d+)\.(.+)', row['matrix_id'])
            if not match:
                raise ValueError('invalid frozen matrix id')
            layer, name = int(match.group(1)), match.group(2)
            if layer >= source_layers or name not in _DENSE or (layer, name) in seen:
                raise ValueError('duplicate/out-of-range/unsupported frozen matrix')
            seen.add((layer, name))
            bias_name = name[:-7] + '_bias'
            expected_shape = ((config.intermediate, config.hidden) if name == 'ffn_up_weight' else
                              (config.hidden, config.intermediate) if name == 'ffn_down_weight' else
                              (config.hidden, config.hidden))
            dimensions = (row['output_width'], row['input_width'])
            if (any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in dimensions) or
                    dimensions != expected_shape):
                raise ValueError('frozen dense shape disagrees with regenerated schema')
            for field, offset_key, shape in (
                (name, 'weight_offset', (row['output_width'], row['input_width'])),
                (bias_name, 'bias_offset', (row['output_width'],))):
                if layer < config.layers and shape != regenerated[layer][field].shape:
                    raise ValueError('frozen dense shape disagrees with regenerated schema')
                offset = row[offset_key]
                count = int(np.prod(shape))
                if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or offset > size or count > (size - offset) // 4:
                    raise ValueError('frozen dense range exceeds weight asset')
                if layer >= config.layers:
                    continue  # explicit prefix: validate descriptor, do not load unused arrays
                source.seek(offset)
                payload = source.read(count * 4)
                if len(payload) != count * 4:
                    raise ValueError('truncated frozen dense weight asset')
                value = np.frombuffer(payload, dtype='<f4').reshape(shape).copy()
                if not np.isfinite(value).all():
                    raise ValueError('nonfinite frozen dense parameter')
                result[layer][field] = value
    if seen != {(layer, name) for layer in range(source_layers) for name in _DENSE}:
        raise ValueError('frozen asset must contain all six dense matrices per encoder')
    hashes = model['weight_sha256']
    if set(hashes) != {str(i) for i in range(source_layers)}:
        raise ValueError('frozen all-parameter hash layer set mismatch')
    for layer, values in enumerate(result):
        expected = hashes[str(layer)]
        if set(expected) != set(values):
            raise ValueError('frozen all-parameter hash schema mismatch')
        for name, value in values.items():
            if digest(value.tobytes(order='C')) != expected[name]:
                raise ValueError(f'frozen parameter SHA256 mismatch: layer{layer}.{name}; '
                                 'non-dense LN parameters require the original numeric environment')
    provenance = dict(mode='frozen-plan-dense-and-all-parameter-sha256',
        fixture_sha256=digest(fixture_raw), manifest_sha256=digest(manifest_raw),
        weight_file_sha256=weight_digest, loaded_dense_matrices=6 * config.layers,
        all_parameter_hashes_verified=True,
        note='dense arrays loaded exactly; regenerated LN arrays must match frozen hashes or fail')
    if allow_prefix_layers:
        provenance.update(source_layers=source_layers, selected_prefix_layers=config.layers,
            all_source_parameter_hashes_verified=(source_layers == config.layers),
            note='explicit prefix: all selected parameters verified; unused source arrays are not revalidated individually')
    return result, provenance


@_schema_errors
def load_frozen_input(config, directory, *, expected_fixture_sha256=None, allow_prefix_layers=False):
    """Load only the ORIGINAL clear client input, never a decrypted HE output."""
    try:
        from .cerium_packing import unpack_outputs
    except ImportError:
        from cerium_packing import unpack_outputs
    raw = (Path(directory) / 'fixture.json').read_bytes()
    if expected_fixture_sha256 is not None and digest(raw) != expected_fixture_sha256:
        raise ValueError('frozen fixture SHA256 changed between parameter and input binding')
    fixture = json.loads(raw)
    if not _compatible_config(fixture['model']['config'], config, allow_prefix_layers):
        raise ValueError('frozen original client input config mismatch')
    packed = np.asarray([row['values'] for row in fixture['inputs']], dtype='<f8')
    hidden = unpack_outputs(packed, config)
    actual = digest(hidden.astype('<f8').tobytes(order='C'))
    if actual != fixture['model']['input_sha256']:
        raise ValueError('frozen original client input SHA256 mismatch')
    return hidden, actual
