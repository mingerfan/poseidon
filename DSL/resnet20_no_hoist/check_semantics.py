#!/usr/bin/env python3
"""Check the real frontend's slot equations against independent dense CHW ops.

This is clear FP64 validation, not compilation or encrypted execution.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ResNet20 import BOUNDARY, SLOTS, load_model, make_relu, network, pack_input


def dense(model, image, relu):
    stages = {}

    def conv(x, layer, bn):
        y = F.conv2d(x, layer.weight, stride=layer.stride, padding=layer.padding)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = (bn.bias - bn.running_mean * scale) / BOUNDARY
        return y * scale[None, :, None, None] + bias[None, :, None, None]

    out = conv(image / BOUNDARY, model.conv1, model.bn1)
    stages['stem.conv_bn'] = out
    out = relu(out)
    stages['stem.relu'] = out
    for stage in range(1, 4):
        for i, block in enumerate(getattr(model, f'layer{stage}')):
            shortcut = out
            if stage > 1 and i == 0:
                c = out.shape[1]
                shortcut = F.pad(out[:, :, ::2, ::2], (0, 0, 0, 0, c//2, c//2))
            out = relu(conv(out, block.conv1, block.bn1))
            out = relu(conv(out, block.conv2, block.bn2) + shortcut)
            stages[f'layer{stage}.{i}.output'] = out
    features = out.mean(dim=(2, 3)) * BOUNDARY
    return F.linear(features, model.linear.weight, model.linear.bias)[0], stages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--relu-coefficients', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--native-oracle', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--runtime-fixture', type=Path,
                   help='write this frontend\'s packed input and native reference logits')
    args = p.parse_args()
    if args.report.exists():
        p.error('use a new report path')
    if args.runtime_fixture and args.runtime_fixture.exists():
        p.error('use a new runtime fixture path')
    import poly.MPCB as mpcb
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    model = load_model(args.weights)
    relu = make_relu(args.relu_coefficients, float)
    with args.images.open() as f:
        values = list(itertools.islice((float(v) for line in f for v in line.split()), 3072))
    if len(values) != 3072:
        raise ValueError('missing image0')
    image = torch.tensor(values).reshape(1, 3, 32, 32)
    original_empty, original_roll = mpcb.Empty, mpcb.roll
    # Exercise the existing frontend packing/kernels on clear Torch slots.
    # Hecate compilation and CKKS execution are tested separately, not mocked.
    mpcb.Empty = lambda: 0.0
    mpcb.roll = lambda value, shift: torch.roll(value, int(shift))
    errors = {}
    try:
        with torch.no_grad():
            reference, expected = dense(model, image, relu)

            def observe(name, values, shape):
                packed = mpcb.shapeClosure(**shape)['OP'](expected[name] * BOUNDARY)
                assert len(values) == len(packed), name
                error = max(float((a-b).abs().max()) for a, b in zip(values, packed))
                errors[name] = error
                print(name, 'max_error', error, flush=True)
                if error > 1e-10:
                    raise AssertionError(f'{name}: clear slot/dense mismatch {error}')

            packed_input = pack_input(image, model)
            result = network(model, packed_input, relu, observe)[0][:10]
    finally:
        mpcb.Empty, mpcb.roll = original_empty, original_roll
    native = json.loads(args.native_oracle.read_text())['reference_logits']
    native_error = float((result-torch.tensor(native)).abs().max())
    dense_error = float((result-reference).abs().max())
    assert native_error < 1e-9 and dense_error < 1e-10, (native_error, dense_error)
    source_sha256 = hashlib.sha256(Path(__file__).with_name('ResNet20.py').read_bytes()).hexdigest()
    model_sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(args.weights.glob('*.txt'))}
    model_sources['d13'] = hashlib.sha256(args.relu_coefficients.read_bytes()).hexdigest()
    model_sources['frontend'] = source_sha256
    model_sha256 = hashlib.sha256(json.dumps(model_sources,sort_keys=True).encode()).hexdigest()
    report = dict(status='PASS',scope='clear FP64 slot/dense/original C++ oracle; not CKKS execution',
        frontend_sha256=source_sha256,
        model_sha256=model_sha256, model_sources=model_sources,
        weights=116,image=0,prediction=int(result.argmax()),native_prediction=max(range(10),key=native.__getitem__),
        stage_max_errors=errors,dense_max_error=dense_error,native_cpp_max_error=native_error,
        logits=result.tolist(),keys_generated=False,gpu_executed=False)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if args.runtime_fixture:
        fixture = dict(format_version=1, model='resnet20', seed=0,
            model_sha256=model_sha256, input=image.reshape(-1).tolist(),
            packed_input=packed_input.tolist(), input_encoding='frontend-packed-slots',
            python_output=result.tolist(), native_output=native, label=3,
            output_encoding='actual-logits-first-10-slots',
            bootstrap_requirement='native-no-hoist',
            semantic_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest())
        args.runtime_fixture.write_text(json.dumps(fixture,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
