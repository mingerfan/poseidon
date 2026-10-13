#!/usr/bin/env python3
"""Export the complete, verified two-token Qwen3 reference for the GPU runner.

The qwen3-0.6b-dsl-v1 signature stores vectors in a real slot prefix. The
runner verifies every remaining slot against zero, including imaginary parts.
This export certifies fixture shape/provenance, not encrypted execution.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from prepare_checkpoint import REVISION, WEIGHT_SHA256
from qwen3 import spec
from qwen3.weights import file_hash


TARGET_CONFIG = dict(num_layers=28, hidden_size=1024, intermediate_size=3072,
                     query_heads=16, kv_heads=8, head_dim=128, vocab_size=151936,
                     max_sequence_length=40960, rope_theta=1000000.0,
                     rms_norm_eps=0.000001)
TARGET_HF_CONFIG = dict(model_type='qwen3', num_hidden_layers=28,
                        hidden_size=1024, intermediate_size=3072,
                        num_attention_heads=16, num_key_value_heads=8,
                        head_dim=128, vocab_size=151936,
                        max_position_embeddings=40960,
                        rope_theta=1000000.0, rms_norm_eps=0.000001,
                        attention_bias=False, tie_word_embeddings=True)
ATOL = 0.001
RTOL = 0.0001


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def verify_target_config(configuration):
    if configuration != TARGET_CONFIG:
        raise ValueError('Signature/model config does not match complete official Qwen3-0.6B')


def verify_signature(signature, configuration, positions):
    verify_target_config(configuration)
    if signature['format'] != 'qwen3-0.6b-dsl-v1' or signature['config'] != configuration:
        raise ValueError('Signature format/config does not match fixture model')
    if signature['mode'] not in ('prefill', 'decode-pair'):
        raise ValueError('This fixture provides two embeddings, not external cached inputs')
    if (signature['new_positions'] != positions or signature['cached_positions'] or
            len(positions) != 2):
        raise ValueError('Signature positions do not match the two-token reference')
    if signature['logit_chunk_size'] != 1024:
        raise ValueError('Complete target uses 1024-logit ciphertext chunks')
    canonical = spec.signature(configuration, signature['mode'], positions,
                               slots=signature['slots'], chunk=1024)
    if signature['function'] != canonical['function']:
        raise ValueError('Unexpected Qwen function in signature')
    if signature['logit_ciphertexts'] != 149:
        raise ValueError('Complete 151936-token vocabulary requires 149 logit outputs')
    for field in ('inputs', 'outputs'):
        rows = signature[field]
        if not isinstance(rows, list) or len(rows) != len(canonical[field]):
            raise ValueError(f'{field} count differs from complete Qwen interface')
        for index, (row, expected) in enumerate(zip(rows, canonical[field])):
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f'{field}[{index}] name, prefix size or ciphertext order mismatch')
            if row.get('layout', 'prefix') != 'prefix':
                raise ValueError(f'{field}[{index}] uses an unsupported slot layout')
    if len(signature['inputs']) != 2 or len(signature['outputs']) != 261:
        raise ValueError('Complete target requires two inputs and 261 outputs')


def verify_provenance(provenance, configuration):
    if (provenance['model'] != 'Qwen/Qwen3-0.6B' or
            provenance['revision'] != REVISION or
            provenance['weight_sha256'] != WEIGHT_SHA256):
        raise ValueError('Checkpoint provenance is not the pinned official pretrained model')
    for key, expected in TARGET_HF_CONFIG.items():
        if provenance['config'].get(key) != expected:
            raise ValueError('Checkpoint configuration mismatch: ' + key)
    if (provenance['tensor_count'] != len(spec.weight_shapes(configuration)) or
            provenance['tensor_count'] != 311 or
            provenance['stored_parameters'] != 751632384 or
            provenance['tied_weights_identical'] is not True):
        raise ValueError('Checkpoint tensor/parameter/tied-weight verification is incomplete')
    if provenance['ckks_tolerance'] != {'absolute': ATOL, 'relative': RTOL}:
        raise ValueError('Reference tolerance must remain atol=0.001 and rtol=0.0001')
    if (len(provenance['tokens']) != 2 or len(provenance['token_strings']) != 2 or
            any(type(token) is not int or not 0 <= token < configuration['vocab_size']
                for token in provenance['tokens'])):
        raise ValueError('Provenance must identify both valid input token IDs')
    comparison = provenance['approximate_vs_exact']
    for field in ('logit_max_abs', 'logit_rms'):
        if not math.isfinite(comparison[field]) or comparison[field] < 0:
            raise ValueError('Invalid exact/approximate reference metric: ' + field)
    if not math.isfinite(comparison['cosine']) or not -1.0000000001 <= comparison['cosine'] <= 1.0000000001:
        raise ValueError('Invalid exact/approximate reference cosine')
    for field in ('exact_top1', 'approximate_top1'):
        if type(provenance[field]) is not int or not 0 <= provenance[field] < configuration['vocab_size']:
            raise ValueError('Invalid reference top-1 token: ' + field)


def validate_array(array, shape, name):
    if array.shape != shape or array.dtype.kind != 'f' or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite floating-point array with shape {shape}')


def map_inputs(rows, embeddings, positions):
    position_index = {position: index for index, position in enumerate(positions)}
    result = []
    for row in rows:
        match = re.fullmatch(r'embedding\.position_(\d+)', row['name'])
        if not match or int(match[1]) not in position_index:
            raise ValueError('Unsupported embedding input: ' + row['name'])
        values = embeddings[position_index[int(match[1])]]
        if values.shape != (row['prefix_size'],):
            raise ValueError('Embedding prefix size mismatch: ' + row['name'])
        result.append(dict(name=row['name'], layout='prefix', values=values.tolist()))
    return result


def map_outputs(rows, reference, positions):
    position_index = {position: index for index, position in enumerate(positions)}
    result = []
    for row in rows:
        logits = re.fullmatch(r'logits\.(\d+):(\d+)', row['name'])
        cache = re.fullmatch(r'layer_(\d+)\.(key|value)\.position_(\d+)', row['name'])
        if logits:
            start, end = map(int, logits.groups())
            if not 0 <= start < end <= len(reference['logits']):
                raise ValueError('Invalid logit range: ' + row['name'])
            values = reference['logits'][start:end]
        elif cache:
            layer, kind, position = cache.groups()
            layer, position = int(layer), int(position)
            array = reference['keys' if kind == 'key' else 'values']
            if not 0 <= layer < len(array) or position not in position_index:
                raise ValueError('Invalid KV position/layer: ' + row['name'])
            values = array[layer, position_index[position]]
        else:
            raise ValueError('Unsupported output name: ' + row['name'])
        if values.shape != (row['prefix_size'],):
            raise ValueError('Output prefix size mismatch: ' + row['name'])
        result.append(dict(name=row['name'], layout='prefix', values=values.tolist()))
    return result


def ckks_context(operator_spec, p_bits, slots):
    source = operator_spec['context']
    degree, q_bits = source['poly_degree'], source['rns_moduli_log2']
    scale = source['default_scale_log2']
    if type(degree) is not int or degree < 2 or degree & (degree - 1) or degree // 2 != slots:
        raise ValueError('Operator spec degree and signature slot capacity disagree')
    if (not isinstance(q_bits, list) or not q_bits or not p_bits or
            any(type(bit) is not int or not 2 <= bit <= 32 for bit in q_bits + p_bits)):
        raise ValueError('Explicit GPU Q/P modulus bit sizes must be integers in [2,32]')
    if type(scale) is not int or not 0 < scale <= 1023:
        raise ValueError('Operator spec default scale must be a positive integer')
    if operator_spec['levels']['upper_bound'] != len(q_bits) - 1:
        raise ValueError('Operator spec upper level does not match its complete Q chain')
    return dict(poly_degree=degree, q_moduli_bits=q_bits, p_moduli_bits=p_bits,
                default_scale_log2=scale)


def waterline_configuration(source):
    fields = {'device_capacity_bytes', 'device_reserve_bytes', 'device_fractions',
              'host_budget_bytes', 'pinned_budget_bytes', 'raw_budget_bytes', 'lookahead_groups'}
    if set(source) not in (fields, fields | {'backend'}):
        raise ValueError('Waterline JSON must provide every runtime budget field')
    capacities, reserves, fractions = (source[field] for field in (
        'device_capacity_bytes', 'device_reserve_bytes', 'device_fractions'))
    if (not all(isinstance(items, list) for items in (capacities, reserves, fractions)) or
            not capacities or len(capacities) != len(reserves) or len(capacities) != len(fractions)):
        raise ValueError('Waterline requires matching nonempty per-device budget arrays')
    for capacity, reserve, fraction in zip(capacities, reserves, fractions):
        if (type(capacity) is not int or type(reserve) is not int or
                not 0 <= reserve < capacity <= 2**53 - 1 or
                type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction < 1):
            raise ValueError('Invalid Waterline per-device budget')
    for field in ('host_budget_bytes', 'pinned_budget_bytes', 'raw_budget_bytes', 'lookahead_groups'):
        if type(source[field]) is not int or not 0 < source[field] <= 2**53 - 1:
            raise ValueError('Invalid positive Waterline budget: ' + field)
    if 'backend' in source and source['backend'] not in ('cpu', 'gpu', 'auto'):
        raise ValueError('Invalid Waterline backend')
    return dict(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signature', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--p-moduli-bits', type=int, nargs='+', required=True)
    parser.add_argument('--waterline-json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    signature = read_json(arguments.signature)
    directory = arguments.fixture
    configuration = read_json(directory / 'model.json')
    provenance = read_json(directory / 'provenance.json')
    verify_provenance(provenance, configuration)
    verify_signature(signature, configuration, provenance['positions'])
    manifest = directory / 'weights' / 'weights.json'
    weight_manifest = read_json(manifest)
    if (weight_manifest['config'] != configuration or
            set(weight_manifest['arrays']) != set(spec.weight_shapes(configuration))):
        raise ValueError('Verified full weight manifest disagrees with signature configuration')
    binding = file_hash(manifest)
    if signature.get('weights_binding') != binding:
        raise ValueError('Traced signature is not bound to this verified weight manifest')
    operator_spec = read_json(arguments.spec)
    context = ckks_context(operator_spec, arguments.p_moduli_bits, signature['slots'])
    embeddings = np.load(directory / 'embeddings.npy', allow_pickle=False)
    validate_array(embeddings, (2, 1024), 'embeddings')
    with np.load(directory / 'reference-approximate.npz', allow_pickle=False) as archive:
        if set(archive.files) != {'logits', 'keys', 'values'}:
            raise ValueError('Approximate reference must provide exact logits/keys/values array names')
        reference = {field: archive[field] for field in archive.files}
    validate_array(reference['logits'], (151936,), 'reference logits')
    validate_array(reference['keys'], (28, 2, 1024), 'reference keys')
    validate_array(reference['values'], (28, 2, 1024), 'reference values')
    with np.load(directory / 'reference-exact.npz', allow_pickle=False) as archive:
        exact_logits = archive['logits']
    validate_array(exact_logits, (151936,), 'exact reference logits')
    error = reference['logits'] - exact_logits
    metrics = dict(logit_max_abs=float(np.max(np.abs(error))),
                   logit_rms=float(np.sqrt(np.mean(error**2))),
                   cosine=float(np.dot(reference['logits'], exact_logits) /
                                (np.linalg.norm(reference['logits']) * np.linalg.norm(exact_logits))))
    for field, value in metrics.items():
        if not math.isclose(value, provenance['approximate_vs_exact'][field], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('Stored reference metrics disagree with reference arrays: ' + field)
    if (int(np.argmax(reference['logits'])) != provenance['approximate_top1'] or
            int(np.argmax(exact_logits)) != provenance['exact_top1']):
        raise ValueError('Stored reference top-1 token disagrees with reference arrays')
    model = dict(provenance)
    model.update(dsl_source_sha256=signature.get('dsl_source_sha256'),
                 trace_script_sha256=signature.get('trace_script_sha256'),
                 complete_qwen=True, full_model_ckks_validated=False,
                 native_tracing_completed=signature.get('native_tracing_completed', False),
                 signature_sha256=file_hash(arguments.signature),
                 weights_manifest_sha256=binding,
                 reference_approximate_sha256=file_hash(directory / 'reference-approximate.npz'),
                 reference_exact_sha256=file_hash(directory / 'reference-exact.npz'),
                 embeddings_sha256=file_hash(directory / 'embeddings.npy'),
                 reference_checked='polynomial dense FP64 reference; all real-prefix outputs and zero tails',
                 mode=signature['mode'], output_count=261, logit_output_count=149,
                 kv_output_count=112, slot_layout='prefix',
                 model_completeness='28 layers, full 151936 vocabulary; numerical CKKS acceptance is performed by the runner')
    result = dict(format_version=1, model=model, context=context,
                  tolerances=dict(atol=ATOL, rtol=RTOL),
                  inputs=map_inputs(signature['inputs'], embeddings, provenance['positions']),
                  expected_outputs=map_outputs(signature['outputs'], reference, provenance['positions']))
    if arguments.waterline_json:
        result['waterline'] = waterline_configuration(read_json(arguments.waterline_json))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(arguments.output), inputs=len(result['inputs']),
                          outputs=len(result['expected_outputs']), complete_qwen=True,
                          full_model_ckks_validated=False, context=context,
                          weight_sha256=provenance['weight_sha256'],
                          approximate_vs_exact=provenance['approximate_vs_exact']), indent=2))


if __name__ == '__main__':
    main()
