#!/usr/bin/env python3
"""Model-level ImageNet ResNet18 frontend; CKKS planning belongs to Dacapo."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SLOTS = 32768
BOUNDARY = 20.0
IMAGE_SIZE = 224
IMAGE_VALUES = 3 * IMAGE_SIZE * IMAGE_SIZE
INPUT_PACKS = 8


def vector(items):
    result = np.empty(len(items), dtype=object)
    for i, value in enumerate(items):
        result[i] = value
    return result


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, 0, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(F.relu(out)))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + shortcut)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool2d(3, 2, 1)
        self.layer1 = self._stage(64, 64, 2, 1)
        self.layer2 = self._stage(64, 128, 2, 2)
        self.layer3 = self._stage(128, 256, 2, 2)
        self.layer4 = self._stage(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(inplanes, planes, blocks, stride):
        result = []
        for block in range(blocks):
            result.append(BasicBlock(inplanes, planes, stride if block == 0 else 1))
            inplanes = planes
        return nn.Sequential(*result)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def resnet18():
    return ResNet18()


def load_model(weights):
    model = resnet18().double().eval()
    files = {path.name for path in weights.glob("*.txt")}
    matched = set()
    state = model.state_dict()
    for key, old in state.items():
        name = key.replace(".", "_") + ".txt"
        if name not in files:
            if key.endswith("num_batches_tracked"):
                continue
            raise ValueError(f"missing ResNet18 weight: {name}")
        values = np.asarray(np.loadtxt(weights / name, dtype=np.float64))
        if values.size != old.numel() or not np.isfinite(values).all():
            raise ValueError(f"invalid weight: {weights / name}")
        state[key] = torch.from_numpy(values.copy()).reshape(old.shape).to(old.dtype)
        matched.add(name)
    missing = files - matched
    if missing:
        raise ValueError(f"unmapped ResNet18 weights: {sorted(missing)}")
    if len(files) != 102:
        raise ValueError(f"expected 102 ResNet18 weight files, got {len(files)}")
    model.load_state_dict(state, strict=True)
    return model


def load_image(values, image_id=0):
    need = (image_id + 1) * IMAGE_VALUES
    with values.open() as stream:
        data = np.fromiter(
            (float(token) for line in stream for token in line.split()),
            dtype=np.float64,
            count=need,
        )
    if data.size != need or not np.isfinite(data).all():
        raise ValueError("missing ImageNet image values")
    return torch.from_numpy(data[-IMAGE_VALUES:].reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE))


def load_label(labels, image_id=0):
    with labels.open() as stream:
        values = [int(token) for line in stream for token in line.split()]
    return values[image_id]


def component(tree_rows, coefficients, degree, baby_length, divisor, scalar):
    rows = [[int(v) for v in row.split()] for row in tree_rows]
    depth = len(rows) - 1
    nodes = [-1] + [v for row in rows for v in row]
    size = 1 << (depth + 1)
    nodes += [-1] * (size - len(nodes))
    degrees, starts = [-1] * size, [-1] * size
    degrees[1] = degree
    for level in range(1, depth + 1):
        for node in range(1 << level, 1 << (level + 1)):
            degrees[node] = (
                nodes[node // 2] - 1
                if node % 2 == 0
                else degrees[node // 2] - nodes[node // 2]
            )
    cursor = 1
    for node in range(1, size):
        if nodes[node] == 0:
            starts[node] = cursor
            cursor += degrees[node] + 1
    if cursor - 1 != len(coefficients):
        raise ValueError("d13 coefficient count does not match polynomial tree")
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
                terms = [
                    coeff[start + 2 * k] * baby[k]
                    for k in range((degrees[node] + 1) // 2)
                ]
                result = terms[0]
                for term in terms[1:]:
                    result = result + term
                return result
            return subtree(2 * node + 1) * power(nodes[node]) + subtree(2 * node)

        return subtree(1)

    return evaluate


def make_relu(coefficients, scalar):
    from poly import Poly

    values = np.loadtxt(coefficients, dtype=np.float64).reshape(-1)
    if values.size != 60 or not np.isfinite(values).all():
        raise ValueError("expected all 60 finite original d13 coefficients")
    p1 = component(Poly.treeStr1, values[:16], 15, 4, 2.0, scalar)
    p2 = component(Poly.treeStr2, values[16:32], 15, 4, 1.7, scalar)
    p3 = component(Poly.treeStr3, values[32:], 27, 8, 2.0, scalar)
    half = scalar(0.5)
    return lambda x: x * (p3(p2(p1(x))) + half)


def dense(model, image, relu):
    def conv(x, layer, bn):
        y = F.conv2d(x, layer.weight, stride=layer.stride, padding=layer.padding)
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = (bn.bias - bn.running_mean * scale) / BOUNDARY
        return y * scale[None, :, None, None] + bias[None, :, None, None]

    out = conv(image / BOUNDARY, model.conv1, model.bn1)
    stages = {"stem.conv": out}
    out = relu(out)
    stages["stem.relu"] = out
    out = F.avg_pool2d(out, 3, 2, 1)
    stages["stem.avgpool"] = out
    for stage_index in range(1, 5):
        stage = getattr(model, f"layer{stage_index}")
        for block_index, block in enumerate(stage):
            shortcut = out
            if block.downsample is not None:
                shortcut = conv(out, block.downsample[0], block.downsample[1])
            out = relu(conv(out, block.conv1, block.bn1))
            out = relu(conv(out, block.conv2, block.bn2) + shortcut)
            stages[f"layer{stage_index}.{block_index}"] = out
    stages["head.before_pool"] = out
    stages["head.pool"] = out.mean(dim=(2, 3))
    features = out.mean(dim=(2, 3)) * BOUNDARY
    return F.linear(features, model.fc.weight, model.fc.bias)[0], stages


def pack_input(image, model):
    import imagenet_layout as layout

    result = []
    for py in range(2):
        for px in range(2):
            phase = F.pad(image[0, :, py::2, px::2] / BOUNDARY, (0, 16, 0, 16))
            result.extend(layout.pack(phase, layout.shape(3, 128)))
    return result


def network(model, inputs, relu, observe=lambda *_: None, clear=False):
    import imagenet_layout as layout

    def folded(layer, bn):
        gain = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
        weight = (layer.weight.detach() * gain[:, None, None, None]).numpy()
        bias = ((bn.bias.detach() - bn.running_mean * gain) / BOUNDARY).numpy()
        return weight, bias

    def convolution(values, source, layer, bn, valid):
        weight, bias = folded(layer, bn)
        return layout.conv(values, source, weight, bias, layer.stride[0], valid, clear)

    weight, bias = folded(model.conv1, model.bn1)
    stem = []
    for py in range(2):
        for px in range(2):
            kernel = np.zeros((64, 3, 5, 5))
            for kh in range(7):
                for kw in range(7):
                    dy, dx = kh-3, kw-3
                    if dy % 2 == py and dx % 2 == px:
                        kernel[:, :, (dy-py)//2+2, (dx-px)//2+2] = weight[:, :, kh, kw]
            phase = py*2+px
            values, shape = layout.conv(inputs[phase*2:phase*2+2], layout.shape(3, 128),
                                        kernel, bias if phase == 0 else np.zeros(64),
                                        1, 112, clear)
            stem.append(values)
    out = [layout.add(phase[i] for phase in stem) for i in range(shape['no'])]
    observe('stem.conv', out, shape)
    out = [relu(value) for value in out]
    observe('stem.relu', out, shape)
    out, shape = layout.avg_pool(out, shape, 56, clear)
    observe('stem.avgpool', out, shape)
    valid = 56
    for stage_index in range(1, 5):
        for block_index, block in enumerate(getattr(model, f'layer{stage_index}')):
            shortcut = out
            next_valid = valid // block.conv1.stride[0]
            if block.downsample is not None:
                shortcut, _ = convolution(out, shape, block.downsample[0],
                                          block.downsample[1], next_valid)
            out, first = convolution(out, shape, block.conv1, block.bn1, next_valid)
            out = [relu(value) for value in out]
            out, shape = convolution(out, first, block.conv2, block.bn2, next_valid)
            out = [relu(x+y) for x, y in zip(out, shortcut)]
            valid = next_valid
            observe(f'layer{stage_index}.{block_index}', out, shape)
    observe('head.before_pool', out, shape)
    out = layout.global_pool(out, shape, valid, clear)
    observe('head.pool', out, layout.shape(512, 1))
    return layout.linear(out, model.fc.weight.detach().numpy() * BOUNDARY,
                         model.fc.bias.detach().numpy(), SLOTS, clear)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--relu-coefficients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bytecode", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    import hecate as hc

    torch.set_num_threads(2)
    model = load_model(args.weights)
    constants = {}

    def scalar(value):
        if value not in constants:
            constants[value] = hc.Plain(np.full(SLOTS, value, dtype=np.float64))
        return constants[value]

    @hc.func("c,c,c,c,c,c,c,c")
    def ResNet18Frontend(i0, i1, i2, i3, i4, i5, i6, i7):
        return network(model, [i0, i1, i2, i3, i4, i5, i6, i7], make_relu(args.relu_coefficients, scalar))

    args.output.mkdir(parents=True)
    print(hc.save(str(args.output), str(args.output), bytecode=args.bytecode))
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.weights.glob("*.txt"))
    }
    (args.output / "frontend.json").write_text(
        json.dumps(
            dict(
                model="ResNet18",
                frontend="Hecate Python/Earth; compiler plans CKKS",
                weights=hashes,
                relu_sha256=hashlib.sha256(args.relu_coefficients.read_bytes()).hexdigest(),
                slots=SLOTS,
                input_packs=INPUT_PACKS,
                image=[3, IMAGE_SIZE, IMAGE_SIZE],
                normalization=BOUNDARY,
                output="actual logits in first 1000 slots",
                bootstrap_requirement="native FHE, NoHoistBsgs; no decrypt/re-encrypt refresh",
                manual_bootstraps=False,
                encrypted_execution_validated=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
