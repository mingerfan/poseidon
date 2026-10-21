#!/usr/bin/env python3
"""Model-level ReLU ResNet20 frontend; CKKS planning belongs to Dacapo.

Requires the framework's Hecate/poly Python packages. No native execution
trace, CKKS instructions, physical levels or HEVM files are consumed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SLOTS = 32768
BOUNDARY = 40.0


def vector(items):
    result = np.empty(len(items), dtype=object)
    for i, value in enumerate(items):
        result[i] = value
    return result


def load_model(weights):
    from poly.models.ResNet import resnet20
    # Convert before loading: the native text weights must never pass through
    # the default float32 model state.
    model = resnet20().double().eval()
    state = model.state_dict()
    expected = {key.replace('.', '_') + '.txt' for key in state}
    available = {p.name for p in weights.glob('*.txt')}
    if available != expected or len(expected) != 116:
        raise ValueError(f'expected all 116 original weights; missing={expected-available}, extra={available-expected}')
    for key, old in state.items():
        path = weights / (key.replace('.', '_') + '.txt')
        values = np.asarray(np.loadtxt(path, dtype=np.float64))
        if values.size != old.numel() or not np.isfinite(values).all():
            raise ValueError(f'invalid weight: {path}')
        state[key] = torch.from_numpy(values.copy()).reshape(old.shape).to(old.dtype)
    model.load_state_dict(state, strict=True)
    return model


def component(tree_rows, coefficients, degree, baby_length, divisor, scalar):
    """Evaluate the d13 parameter file's decomposed Chebyshev polynomial.

    These are mathematical coefficients and a polynomial expression tree,
    not a ciphertext execution schedule or scale/rescale annotations.
    """
    rows = [[int(v) for v in row.split()] for row in tree_rows]
    depth = len(rows) - 1
    nodes = [-1] + [v for row in rows for v in row]
    size = 1 << (depth + 1)
    nodes += [-1] * (size - len(nodes))
    degrees, starts = [-1] * size, [-1] * size
    degrees[1] = degree
    for level in range(1, depth + 1):
        for node in range(1 << level, 1 << (level + 1)):
            degrees[node] = (nodes[node // 2] - 1 if node % 2 == 0
                             else degrees[node // 2] - nodes[node // 2])
    cursor = 1
    for node in range(1, size):
        if nodes[node] == 0:
            starts[node] = cursor
            cursor += degrees[node] + 1
    if cursor - 1 != len(coefficients):
        raise ValueError('d13 coefficient count does not match polynomial tree')
    coeff = [scalar(float(c) / divisor) for c in coefficients]
    two, minus_one = scalar(2.0), scalar(-1.0)

    def evaluate(x):
        giant = {1: x}

        def power(n):
            if n not in giant:
                half = power(n // 2)
                giant[n] = two * half * half + minus_one
            return giant[n]

        baby = [x]
        for level in range(1, baby_length.bit_length() - 1):
            products = [two * value * power(1 << level) for value in baby]
            baby += [a - b for a, b in zip(products, reversed(baby))]

        def subtree(node):
            if nodes[node] == 0:
                start = starts[node]
                terms = [coeff[start + 2*k] * baby[k]
                         for k in range((degrees[node] + 1) // 2)]
                result = terms[0]
                for term in terms[1:]:
                    result = result + term
                return result
            return subtree(2*node + 1) * power(nodes[node]) + subtree(2*node)

        return subtree(1)
    return evaluate


def make_relu(coefficients, scalar):
    from poly import Poly
    values = np.loadtxt(coefficients, dtype=np.float64).reshape(-1)
    if values.size != 60 or not np.isfinite(values).all():
        raise ValueError('expected all 60 finite original d13 coefficients')
    p1 = component(Poly.treeStr1, values[:16], 15, 4, 2.0, scalar)
    p2 = component(Poly.treeStr2, values[16:32], 15, 4, 1.7, scalar)
    p3 = component(Poly.treeStr3, values[32:], 27, 8, 2.0, scalar)
    half = scalar(0.5)
    return lambda x: x * (p3(p2(p1(x))) + half)


def network(model, image, relu, observe=lambda *_: None):
    """The same ResNet20 equations for Hecate expressions and clear slots."""
    from poly.MPCB import CascadeConv, CascadeDS, CascadePool, shapeClosure
    from poly.Func import HE_ConvBN, HE_DS, HE_Pool, HE_Linear

    shape = dict(nt=SLOTS, bb=BOUNDARY, ko=1, ho=32, wo=32)
    shape = CascadeConv(shape, model.conv1)
    out = HE_ConvBN(shapeClosure(**shape), vector([image]), model.conv1, model.bn1)
    observe('stem.conv_bn', out, shape)
    out = vector([relu(x) for x in out])
    observe('stem.relu', out, shape)
    for stage in range(1, 4):
        for i, block in enumerate(getattr(model, f'layer{stage}')):
            shortcut = out
            if stage > 1 and i == 0:
                shortcut = HE_DS(shapeClosure(**CascadeDS(shape)), out)
            first = CascadeConv(shape, block.conv1)
            out = HE_ConvBN(shapeClosure(**first), out, block.conv1, block.bn1)
            out = vector([relu(x) for x in out])
            second = CascadeConv(first, block.conv2)
            out = HE_ConvBN(shapeClosure(**second), out, block.conv2, block.bn2)
            out = vector([relu(x+y) for x, y in zip(out, shortcut)])
            shape = second
            observe(f'layer{stage}.{i}.output', out, shape)
    pool = CascadePool(shape)
    out = HE_Pool(shapeClosure(**pool), out)
    # Return actual logits. No legacy runner's external multiplication by 32.
    return HE_Linear(shapeClosure(**pool)['OP'], out, model.linear,
                     p=BOUNDARY, scale=1.0, nt=SLOTS)


def pack_input(image, model):
    from poly.MPCB import CascadeConv, shapeClosure
    shape = CascadeConv(dict(nt=SLOTS, bb=BOUNDARY, ko=1, ho=32, wo=32), model.conv1)
    packed = shapeClosure(**shape)['MPP'](image)
    if tuple(packed.shape) != (1, SLOTS):
        raise ValueError('expected one 32768-slot input')
    return packed[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--relu-coefficients', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bytecode', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output directory')
    import hecate as hc
    model = load_model(args.weights)
    constants = {}

    def scalar(value):
        # Current Runtime zero-pads short payloads; length-one constants must
        # not silently become masks when a uniform slot constant is intended.
        if value not in constants:
            constants[value] = hc.Plain(np.full(SLOTS, value, dtype=np.float64))
        return constants[value]

    @hc.func('c')
    def ResNet20(image):
        return network(model, image, make_relu(args.relu_coefficients, scalar))

    args.output.mkdir(parents=True)
    print(hc.save(str(args.output), str(args.output), bytecode=args.bytecode))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.weights.glob('*.txt'))}
    (args.output/'frontend.json').write_text(json.dumps(dict(
        model='ResNet20', frontend='Hecate Python/Earth; compiler plans CKKS',
        weights=hashes, relu_sha256=hashlib.sha256(args.relu_coefficients.read_bytes()).hexdigest(),
        slots=SLOTS, normalization=BOUNDARY, output='actual logits in first 10 slots',
        bootstrap_requirement='native FHE, NoHoistBsgs; no decrypt/re-encrypt refresh',
        manual_bootstraps=False, encrypted_execution_validated=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
