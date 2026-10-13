#!/usr/bin/env python3
"""Independent dense FP64 oracle for the production BSGS linear DSL.

For R_s(x)[i]=x[(i+s)%S], s=g+b,
R_g(R_b(x)*R_-g(d_s)) = R_s(x)*d_s exactly in slot indexing.
This invokes the actual production linear, checking dense matmul and dirty
physical tails at all seven unchanged model dimensions. It is not a CKKS test.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_adapter():
    source = Path(__file__).with_name('dsl-slot8192-test.py')
    module_spec = importlib.util.spec_from_file_location('slot8192_adapter', source)
    adapter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(adapter)
    return adapter


def rotation_count(output_size, input_size):
    block = min(64, input_size)
    shifts = range(1 - output_size, input_size)
    babies = {s % block for s in shifts}
    giants = {s - s % block for s in shifts}
    return sum(b != 0 for b in babies) + sum(g != 0 for g in giants)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    adapter = load_adapter()
    ql, _, signature = adapter.load_dsl()
    rng = np.random.default_rng(20261013)
    results = adapter.test_linear(ql, rng, rotation_count)
    for row in results:
        row['baby_step_size'] = min(64, row['shape'][1])
        row['old_diagonal_rotations'] = sum(row['shape']) - 2
        expected_terms = sum(row['shape']) - 1
        if row['operations']['mul_cp'] != expected_terms:
            raise AssertionError('production linear dropped a diagonal')
    small = []
    for output_size, input_size in ((1, 1), (2, 1), (1, 2), (3, 5), (17, 9), (70, 35), (129, 71)):
        weight = rng.standard_normal((output_size, input_size))
        data = rng.standard_normal(input_size)
        bias = rng.standard_normal(output_size)
        physical = np.concatenate((data, rng.standard_normal(adapter.SLOTS - input_size) * 1e5))
        vector = ql.PackedVector(adapter.Expr(physical), input_size, adapter.SLOTS)
        adapter.reset_counts()
        actual = ql.linear(vector, weight, bias).expr.values
        counts = dict(adapter.COUNTS)
        error, tail = adapter.fullslot_error(actual, weight @ data + bias, label='small BSGS')
        if counts['rotations'] != rotation_count(output_size, input_size):
            raise AssertionError('unexpected small-case rotation count')
        row = dict(shape=[output_size, input_size], baby_step_size=min(64, input_size),
                   max_abs=error, output_tail_abs=tail, operations=counts)
        small.append(row)
        print(f'PASS small production BSGS block={row["baby_step_size"]} '
              f'shape={output_size}x{input_size} bias max_abs={error:.17g} tail={tail}', flush=True)
    config = dict(num_layers=28, hidden_size=1024, intermediate_size=3072,
                  query_heads=16, kv_heads=8, head_dim=128, vocab_size=151936,
                  max_sequence_length=40960, rope_theta=1_000_000., rms_norm_eps=1e-6)
    interface = signature(config, 'decode-pair', (0, 1), slots=adapter.SLOTS, chunk=1024)
    if len(interface['inputs']) != 2 or len(interface['outputs']) != 261:
        raise AssertionError('full model interface changed')
    if sum(row['prefix_size'] for row in interface['outputs'][:149]) != 151936:
        raise AssertionError('full vocabulary output changed')
    report = dict(passed=True, scope='Production linear FP64 physical-slot adapter vs independent dense oracle; synthetic weights; not CKKS',
                  implementation='scripts/qwen3_dsl/qwen3/kernels/qwen_layers.py:linear',
                  slots=adapter.SLOTS, baby_step_rule='min(64, input_size)', cases=results,
                  max_abs=max(row['max_abs'] for row in results + small),
                  small_bias_cases=small, full_model_interface=interface,
                  full_model_ckks_validated=False)
    args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
