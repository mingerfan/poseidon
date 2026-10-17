#!/usr/bin/env python3
"""Standalone FP64 slot oracle; it does not change the production Qwen DSL.

Checks prefix replication + cyclic BSGS, packed partial down-projection, and a
full-width LM head whose outputs are sliced back to the original 149 chunks.
Synthetic full-dimension weights and dense NumPy matmul are the independent
reference. No native tracing, CKKS, GPU, or performance claim is involved.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

SLOTS = 8192
COST = dict(mul_cp=0, rotations=0, adds=0)
SHAPES = [('q_proj', 2048, 1024), ('k_proj', 1024, 1024),
          ('v_proj', 1024, 1024), ('o_proj', 1024, 2048),
          ('gate_proj', 3072, 1024), ('up_proj', 3072, 1024),
          ('down_proj', 1024, 3072)]


def reset_cost():
    for key in COST:
        COST[key] = 0


def physical(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not 0 < len(values) <= SLOTS or not np.isfinite(values).all():
        raise ValueError('finite nonempty values must fit the physical slots')
    return np.pad(values, (0, SLOTS - len(values)))


class Cipher:
    def __init__(self, values, depth=0):
        self.values = physical(values)
        self.depth = depth

    def rotate(self, step):
        COST['rotations'] += 1
        return Cipher(np.roll(self.values, -step), self.depth)

    def mul_plain(self, values):
        COST['mul_cp'] += 1
        return Cipher(self.values * physical(values), self.depth + 1)

    def add(self, other):
        COST['adds'] += 1
        return Cipher(self.values + other.values, max(self.depth, other.depth))

    def add_plain(self, values):
        COST['adds'] += 1
        return Cipher(self.values + physical(values), self.depth)


def replicate_prefix(vector, input_size):
    """Mask dirty physical tails, pad a power-of-two period, and tile all slots."""
    period = 1 << (input_size - 1).bit_length()
    result = vector.mul_plain(np.ones(input_size))
    shift = period
    while shift < SLOTS:
        result = result.add(result.rotate(-shift))
        shift *= 2
    return result, period


def cyclic_from_replicated(vector, weight, period, bias=None):
    """Every valid column uses (row+s) modulo the replicated input period."""
    output_size, input_size = weight.shape
    if not 0 < output_size <= SLOTS or not input_size <= period <= SLOTS:
        raise ValueError('matrix and period must fit physical slots')
    block = min(64, period)
    babies = [vector.rotate(b) if b else vector for b in range(block)]
    rows = np.arange(output_size)
    result = None
    for giant in range(0, period, block):
        group = None
        for baby in range(block):
            shift = giant + baby
            columns = (rows + shift) % period
            valid = columns < input_size
            # Omit only a structurally empty diagonal, never inspect weight
            # magnitudes or replace small/zero checkpoint values approximately.
            if not valid.any():
                continue
            diagonal = np.zeros(SLOTS)
            diagonal[rows[valid]] = weight[rows[valid], columns[valid]]
            term = babies[baby].mul_plain(np.roll(diagonal, giant))
            group = term if group is None else group.add(term)
        if group is None:
            continue
        term = group.rotate(giant) if giant else group
        result = term if result is None else result.add(term)
    return result if bias is None else result.add_plain(bias)


def cyclic_linear(vector, weight, bias=None):
    tiled, period = replicate_prefix(vector, weight.shape[1])
    return cyclic_from_replicated(tiled, weight, period, bias)


def packed_down_projection(vector, weight):
    """Pack the three 1024-column partial products in one ciphertext.

    One signed diagonal encodes all three weight blocks. Its mask excludes
    reads crossing block boundaries, including reads into dirty input tails.
    Fold the partial outputs into the prefix and mask every output tail.
    """
    output_size, input_size = weight.shape
    if input_size != 3 * output_size or output_size != 1024:
        raise ValueError('prototype requires the actual 1024x3072 down projection')
    block = 64
    first = (1 - output_size) // block * block
    last = (output_size - 1) // block * block
    babies = [vector.rotate(b) if b else vector for b in range(block)]
    rows = np.arange(output_size)
    result = None
    for giant in range(first, last + 1, block):
        group = None
        for baby in range(block):
            shift = giant + baby
            if not 1 - output_size <= shift < output_size:
                continue
            columns = rows + shift
            valid = (columns >= 0) & (columns < output_size)
            diagonal = np.zeros(SLOTS)
            for part in range(3):
                diagonal[part * output_size + rows[valid]] = weight[
                    rows[valid], part * output_size + columns[valid]]
            term = babies[baby].mul_plain(np.roll(diagonal, giant))
            group = term if group is None else group.add(term)
        term = group.rotate(giant) if giant else group
        result = term if result is None else result.add(term)
    folded = result.add(result.rotate(output_size)).add(result.rotate(2 * output_size))
    return folded.mul_plain(np.ones(output_size))


def extract(vector, start, size):
    source = vector.rotate(start) if start else vector
    return source.mul_plain(np.ones(size))


def fullslot_check(actual, reference, label):
    expected = physical(reference)
    error = float(np.max(np.abs(actual.values - expected)))
    tail = float(np.max(np.abs(actual.values[len(reference):]), initial=0.0))
    if error >= 1e-8 or tail != 0.0:
        raise AssertionError(f'{label}: max_abs={error}, tail={tail}')
    return error, tail


def bsgs_count(output_size, input_size):
    block = min(64, input_size)
    first = (1 - output_size) // block * block
    last = (input_size - 1) // block * block
    return dict(mul_cp=output_size + input_size - 1,
                rotations=block - 1 + (last - first) // block,
                adds=output_size + input_size - 2, relative_cp_depth=1)


def verify_current_lm_counts():
    # Invoke unchanged production linear for representative full and short LM
    # chunks, proving counts used in the full-vocabulary arithmetic below.
    path = Path(__file__).with_name('dsl-slot8192-test.py')
    spec = importlib.util.spec_from_file_location('current_slot_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    ql, _, _ = adapter.load_dsl()
    results = []
    for output_size in (1024, 384):
        weight = np.ones((output_size, 1024)) / 1024
        adapter.reset_counts()
        actual = ql.linear(ql.PackedVector(adapter.Expr(np.ones(1024)), 1024, SLOTS), weight)
        measured = dict(adapter.COUNTS)
        expected = bsgs_count(output_size, 1024)
        for name in ('mul_cp', 'rotations', 'adds'):
            if measured[name] != expected[name]:
                raise AssertionError('unchanged production LM count mismatch')
        if float(np.max(np.abs(actual.expr.values[:output_size] - 1))) != 0:
            raise AssertionError('unchanged production LM numeric check failed')
        results.append(dict(output_size=output_size, measured=measured))
    return results


def test_shapes(rng):
    results = []
    for name, output_size, input_size in SHAPES:
        weight = rng.standard_normal((output_size, input_size)) / math.sqrt(input_size)
        base = rng.standard_normal(input_size)
        for case in ('dense', 'short_logical_tail'):
            data = base.copy()
            if case == 'short_logical_tail':
                data[-37:] = 0
            dirty = np.concatenate((data, rng.standard_normal(SLOTS - input_size) * 1e5))
            expected = weight @ data
            for implementation in ('cyclic', 'packed_down'):
                if implementation == 'packed_down' and name != 'down_proj':
                    continue
                fn = cyclic_linear if implementation == 'cyclic' else packed_down_projection
                reset_cost()
                actual = fn(Cipher(dirty), weight)
                cost = dict(COST)
                error, tail = fullslot_check(actual, expected, name+'/'+case+'/'+implementation)
                clean = fn(Cipher(data), weight)
                contamination = float(np.max(np.abs(actual.values - clean.values)))
                if contamination != 0:
                    raise AssertionError('dirty physical tails contaminated output')
                row = dict(name=name, shape=[output_size, input_size], case=case,
                           implementation=implementation, max_abs=error, output_tail_abs=tail,
                           dirty_input_contamination=contamination, operations=cost,
                           relative_cp_depth=actual.depth)
                results.append(row)
                print('PASS shape '+json.dumps(row), flush=True)
    for output_size, input_size in ((1, 1), (2, 1), (1, 2), (3, 5), (17, 9), (70, 35), (129, 71)):
        weight = rng.standard_normal((output_size, input_size))
        data = rng.standard_normal(input_size)
        bias = rng.standard_normal(output_size)
        dirty = np.concatenate((data, rng.standard_normal(SLOTS-input_size)*1e5))
        actual = cyclic_linear(Cipher(dirty), weight, bias)
        error, _ = fullslot_check(actual, weight @ data+bias, 'small bias')
        print(f'PASS small bias {output_size}x{input_size} max_abs={error}', flush=True)
    return results


def test_full_lm(rng):
    vocab, hidden = 151936, 1024
    data = rng.standard_normal(hidden)
    dirty = np.concatenate((data, rng.standard_normal(SLOTS-hidden)*1e5))
    reset_cost()
    tiled, period = replicate_prefix(Cipher(dirty), hidden)
    replication_cost = dict(COST)
    reference = np.empty(vocab)
    assembled = np.empty(vocab)
    cases = []
    # Stream the full synthetic matrix, about 1.16 GiB of weights in total,
    # instead of keeping it all resident. All rows are still evaluated.
    for begin in range(0, vocab, SLOTS):
        count = min(SLOTS, vocab-begin)
        weight = rng.standard_normal((count, hidden)) / math.sqrt(hidden)
        dense = weight @ data
        reset_cost()
        wide = cyclic_from_replicated(tiled, weight, period)
        projection_cost = dict(COST)
        wide_error, _ = fullslot_check(wide, dense, 'wide LM')
        slices = []
        reset_cost()
        for offset in range(0, count, 1024):
            size = min(1024, count-offset)
            result = extract(wide, offset, size)
            error, tail = fullslot_check(result, dense[offset:offset+size], 'LM interface slice')
            global_begin = begin+offset
            assembled[global_begin:global_begin+size] = result.values[:size]
            slices.append(dict(begin=global_begin, size=size, max_abs=error,
                               output_tail_abs=tail, relative_cp_depth=result.depth))
        slice_cost = dict(COST)
        reference[begin:begin+count] = dense
        row = dict(begin=begin, size=count, wide_max_abs=wide_error,
                   projection_operations=projection_cost, extraction_operations=slice_cost,
                   slices=slices)
        cases.append(row)
        print(f'PASS full LM rows {begin}:{begin+count} {len(slices)} output slices '
              f'max_abs={wide_error:.17g}', flush=True)
    all_slices = [r for case in cases for r in case['slices']]
    if len(cases) != 19 or len(all_slices) != 149 or all_slices[-1]['size'] != 384:
        raise AssertionError('complete LM layout changed')
    if sum(r['size'] for r in all_slices) != vocab:
        raise AssertionError('vocabulary reduced')
    operations = dict(replication_cost)
    for case in cases:
        for counts in (case['projection_operations'], case['extraction_operations']):
            for key in operations:
                operations[key] += counts[key]
    return dict(vocabulary=vocab, hidden=hidden, wide_blocks=19, output_ciphertexts=149,
                final_prefix_size=384, max_abs=float(np.max(np.abs(assembled-reference))),
                operations=operations, relative_cp_depth=3,
                replication_operations=replication_cost, blocks=cases)


def complexity_report(shapes, lm):
    current = []
    for name, output_size, input_size in SHAPES:
        row = dict(name=name, output_size=output_size, input_size=input_size,
                   operations=bsgs_count(output_size, input_size))
        row['cyclic'] = next(r for r in shapes if r['name']==name and r['implementation']=='cyclic')
        if name == 'down_proj':
            row['packed_down'] = next(r for r in shapes if r['implementation']=='packed_down')
        current.append(row)
    old_lm = {key: 148*bsgs_count(1024,1024)[key]+bsgs_count(384,1024)[key]
              for key in ('mul_cp','rotations','adds')}
    old_total = {key:56*sum(r['operations'][key] for r in current)+old_lm[key]
                 for key in old_lm}
    periodic_total = {key:56*sum(r['cyclic']['operations'][key] for r in current)+lm['operations'][key]
                      for key in old_lm}
    partial_total = {key:56*sum((r['packed_down'] if r['name']=='down_proj' else r['cyclic'])['operations'][key]
                              for r in current)+lm['operations'][key]
                     for key in old_lm}
    wide_bsgs = {key:18*bsgs_count(8192,1024)[key]+bsgs_count(4480,1024)[key]
                for key in old_lm}
    wide_bsgs['mul_cp'] += 149
    wide_bsgs['rotations'] += 130
    return dict(per_matrix=current, current_lm=old_lm, current_linear_total=old_total,
                full_lm_wide_without_cyclic=wide_bsgs,
                prototype_cyclic_linear_total=periodic_total,
                prototype_with_packed_down_linear_total=partial_total,
                invocation_count='56 per transformer matrix: 28 full layers times 2 tokens; one complete LM head',
                production_prepare_count_not_cp_count=1418689,
                production_total_groups_not_linear_count=8273326)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('report must be a new path; preserve earlier evidence')
    rng = np.random.default_rng(20261013)
    current_lm = verify_current_lm_counts()
    shapes = test_shapes(rng)
    lm = test_full_lm(rng)
    report = dict(passed=True, scope='Independent FP64 physical-slot prototype vs dense NumPy; full model dimensions with synthetic weights; not CKKS or full-model execution',
                  production_dsl_modified=False, fhe_validated=False, gpu_speedup_claimed=False,
                  slots=SLOTS, full_model_interface=dict(layers=28, inputs=2, outputs=261,
                                                       logits=151936, logit_ciphertexts=149, kv_ciphertexts=112),
                  current_production_lm_counts=current_lm, shape_cases=shapes, small_bias_cases=7,
                  full_lm=lm, complexity=complexity_report(shapes,lm),
                  limits=['Input prefix masks and output extraction add CP/rescale depth; existing compiler may add more Boot.',
                          'One CP depth is one plaintext multiplication stage, not a measured compiler schedule; 120-bit logical rescale usually drops four 30-bit Q primes.',
                          'Current normalization already emits prefix-zero masks, but this prototype deliberately proves generic dirty-tail safety instead of assuming or rewriting that contract.',
                          'Preparation count differs from source MulCP count due to CSE/sharing, placement, type/scale choices and other operators.',
                          '19 wider LM temporaries do not replace the final 149 ciphertexts; their retained lifetime and all extraction outputs need real resource validation.',
                          'No HEFT, placement algorithm, runtime scheduler, production DSL, pretrained weights or interface is changed by this oracle.'])
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS complexity '+json.dumps(report['complexity']['current_linear_total'])+
          ' -> '+json.dumps(report['complexity']['prototype_with_packed_down_linear_total']),flush=True)


if __name__ == '__main__':
    main()
