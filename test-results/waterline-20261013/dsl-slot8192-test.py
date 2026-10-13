#!/usr/bin/env python3
"""FP64 physical-slot adapter checks of the production Qwen DSL operations.

No CKKS encryption, compiler or GPU is used. All seven real model matrix
shapes are checked against dense NumPy multiplication; the model signature
retains all 28 layers and 151936 logits (149 chunks, final chunk 384).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import types

import numpy as np


SLOTS = 8192
COUNTS = dict(rotations=0, mul_cp=0, mul_cc=0, adds=0)


class Expr:
    def __init__(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) > SLOTS:
            raise ValueError('too many physical slots')
        self.values = np.pad(values, (0, SLOTS - len(values)))

    def rotate(self, steps):
        COUNTS['rotations'] += 1
        # Existing Hecate/GPU convention: output[i] = input[(i+s) % slots].
        return Expr(np.roll(self.values, -steps))

    def __add__(self, other):
        COUNTS['adds'] += 1
        return Expr(self.values + other.values)

    def __mul__(self, other):
        key = 'mul_cp' if isinstance(self, Plain) or isinstance(other, Plain) else 'mul_cc'
        COUNTS[key] += 1
        return Expr(self.values * other.values)


class Plain(Expr):
    """Short plaintext arrays are zero-padded, never broadcast."""


def load_dsl():
    hecate = types.ModuleType('hecate')
    expression = types.ModuleType('hecate.expr')
    expression.Expr, expression.Plain = Expr, Plain
    hecate.expr = expression
    sys.modules['hecate'], sys.modules['hecate.expr'] = hecate, expression
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts/qwen3_dsl'))
    from qwen3.kernels import qwen_layers, ops
    from qwen3.spec import signature
    return qwen_layers, ops, signature


def reset_counts():
    for key in COUNTS:
        COUNTS[key] = 0


def fullslot_error(actual, prefix, *, label):
    reference = np.zeros(SLOTS)
    reference[:len(prefix)] = prefix
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise AssertionError(f'{label}: wrong shape or nonfinite physical slots')
    error = float(np.max(np.abs(actual - reference)))
    tail = float(np.max(np.abs(actual[len(prefix):])))
    if error >= 1e-8 or tail != 0:
        raise AssertionError(f'{label}: fullslot error={error}, tail={tail}')
    return error, tail


def test_linear(ql, rng, expected_rotation_count=None):
    shapes = [('q_proj', 2048, 1024), ('k_proj', 1024, 1024),
              ('v_proj', 1024, 1024), ('o_proj', 1024, 2048),
              ('gate_proj', 3072, 1024), ('up_proj', 3072, 1024),
              ('down_proj', 1024, 3072)]
    results = []
    for name, output_size, input_size in shapes:
        # Synthetic FP64 weights with the full unmodified model dimensions.
        weight = rng.standard_normal((output_size, input_size)) / math.sqrt(input_size)
        dense = rng.standard_normal(input_size)
        for case in ('dense', 'short_logical_tail'):
            data = dense.copy()
            if case == 'short_logical_tail':
                data[-37:] = 0
            expected = weight @ data
            reset_counts()
            clean = ql.linear(ql.PackedVector(Expr(data), input_size, SLOTS), weight).expr.values
            counts = dict(COUNTS)
            physical = np.concatenate((data, rng.standard_normal(SLOTS - input_size) * 1e5))
            dirty = ql.linear(ql.PackedVector(Expr(physical), input_size, SLOTS), weight).expr.values
            clean_error, _ = fullslot_error(clean, expected, label=f'{name}/{case}/clean')
            dirty_error, tail = fullslot_error(dirty, expected, label=f'{name}/{case}/dirty')
            contamination = float(np.max(np.abs(clean - dirty)))
            if contamination != 0:
                raise AssertionError(f'{name}: physical input tail contaminated output')
            if expected_rotation_count is None:
                block = min(64, input_size)
                first_giant = (1 - output_size) // block * block
                last_giant = (input_size - 1) // block * block
                # Every baby occurs among shifts 0..input_size-1. The signed
                # giant range contains zero, whose rotation is omitted.
                expected_rotations = block - 1 + (last_giant - first_giant) // block
            else:
                expected_rotations = expected_rotation_count(output_size, input_size)
            if counts['rotations'] != expected_rotations:
                raise AssertionError(f'{name}: unexpected linear rotation count')
            row = dict(name=name, shape=[output_size, input_size], case=case,
                       max_abs=max(clean_error, dirty_error), output_tail_abs=tail,
                       physical_input_tail_contamination=contamination, operations=counts)
            results.append(row)
            print(f'PASS linear {name} {output_size}x{input_size} {case}: '
                  f'max_abs={row["max_abs"]:.17g} output_tail={tail} '
                  f'input_tail_contamination={contamination}', flush=True)
    return results


def test_grouped_norm(ql, ops, rng):
    dim, eps = 128, 1e-6
    lo, hi = .25, 4.
    coefficients = np.polynomial.chebyshev.chebinterpolate(
        lambda t: 1 / np.sqrt((t + 1) * (hi - lo) / 2 + lo), 80)
    cheby = ql.ChebyshevApproximation('rsqrt', tuple(coefficients), (lo, hi),
                                     evaluation='blocked', block_size=8)
    results = []
    for heads in (16, 8):
        for case in ('dense', 'short_last_head', 'zero_final_heads', 'wide_head_scales'):
            size = heads * dim
            data = rng.standard_normal((heads, dim)).reshape(-1)
            if case == 'short_last_head':
                data[-37:] = 0
            if case == 'zero_final_heads':
                data[-2 * dim:] = 0
            if case == 'wide_head_scales':
                data *= np.repeat(np.logspace(-2, 2, heads), dim)
            gain = rng.uniform(.25, 1.75, dim)
            variance = np.mean(data.reshape(heads, dim) ** 2, axis=1) + eps
            if case in ('zero_final_heads', 'wide_head_scales'):
                domain = (max(float(variance.min()) * .5, 1e-12), float(variance.max()) * 2)
                iterations = math.ceil(math.log(domain[1] / domain[0]) / math.log(2.25)) + 8
                profile = ql.InverseSqrtApproximation(domain, iterations,
                                                     evaluation='residual', refinement_steps=2)
            else:
                profile = cheby
            physical = np.concatenate((data, rng.standard_normal(SLOTS - size) * 1e5))
            vector = ql.PackedVector(Expr(physical), size, SLOTS)
            sums = ops.block_sum_repeated(vector.expr, block_size=dim, blocks=heads, slots=SLOTS).values
            sum_reference = np.repeat(data.reshape(heads, dim).sum(axis=1), dim)
            sum_error, _ = fullslot_error(sums, sum_reference, label=f'block_sum/{heads}/{case}')
            reset_counts()
            actual = ql.grouped_head_rms_norm(vector, gain, profiles=(profile,) * heads,
                                             heads=heads, head_dim=dim, eps=eps).expr.values
            counts = dict(COUNTS)
            expected = (data.reshape(heads, dim) / np.sqrt(variance[:, None]) * gain).reshape(-1)
            error, tail = fullslot_error(actual, expected, label=f'grouped_norm/{heads}/{case}')
            if counts['rotations'] != 14:
                raise AssertionError('grouped norm did not use seven masked butterfly stages')
            results.append(dict(heads=heads, head_dim=dim, case=case,
                                profile=type(profile).__name__, max_abs=error,
                                output_tail_abs=tail, block_sum_max_abs=sum_error, operations=counts))
            print(f'PASS grouped_norm heads={heads} {case}: max_abs={error:.17g} '
                  f'output_tail={tail} rotations={counts["rotations"]}', flush=True)
    vector = ql.PackedVector(Expr(np.zeros(16 * dim)), 16 * dim, SLOTS)
    negatives = [('unequal_profile', (cheby,) * 15 + (ql.InverseSqrtApproximation((.1, 2), 10),), np.ones(dim), 16),
                 ('missing_profile', (cheby,) * 15, np.ones(dim), 16),
                 ('wrong_gain', (cheby,) * 16, np.ones(dim - 1), 16),
                 ('wrong_head_count', (cheby,) * 8, np.ones(dim), 8)]
    for label, profiles, gain, heads in negatives:
        try:
            ql.grouped_head_rms_norm(vector, gain, profiles=profiles, heads=heads, head_dim=dim)
        except (TypeError, ValueError):
            print('PASS grouped_norm negative', label)
        else:
            raise AssertionError(label + ' unexpectedly accepted')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    ql, ops, signature = load_dsl()
    config = dict(num_layers=28, hidden_size=1024, intermediate_size=3072,
                  query_heads=16, kv_heads=8, head_dim=128, vocab_size=151936,
                  max_sequence_length=40960, rope_theta=1_000_000., rms_norm_eps=1e-6)
    interface = signature(config, 'decode-pair', (0, 1), slots=SLOTS, chunk=1024)
    assert len(interface['inputs']) == 2 and len(interface['outputs']) == 261
    assert interface['logit_ciphertexts'] == 149
    assert interface['outputs'][148]['prefix_size'] == 384
    assert sum(row['prefix_size'] for row in interface['outputs'][:149]) == 151936
    print('PASS complete interface: N=16384 slots=8192 layers=28 inputs=2 '
          'outputs=261 logits=151936 chunks=149 last_chunk=384', flush=True)
    rng = np.random.default_rng(20261013)
    linear = test_linear(ql, rng)
    grouped = test_grouped_norm(ql, ops, rng)
    report = dict(passed=True,
                  scope='FP64 NumPy physical-slot adapter; synthetic weights; not CKKS/full-model execution',
                  poly_degree=16384, slots=SLOTS, numerical_tolerance=1e-8,
                  rotation_convention='rotate(s)[i] = input[(i+s) % slots]',
                  interface=interface, linear_cases=linear, grouped_norm_cases=grouped,
                  grouped_norm_negative_cases=4,
                  linear_max_abs=max(row['max_abs'] for row in linear),
                  grouped_norm_max_abs=max(row['max_abs'] for row in grouped),
                  full_model_ckks_validated=False)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS all slot8192 checks', flush=True)


if __name__ == '__main__':
    main()
