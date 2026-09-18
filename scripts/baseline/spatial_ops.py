"""Bounded spatial semantics and explicit public linear lowering.

Unbatched channel-first Conv1D/2D are cross-correlations, not flipped kernels.
Stride/padding/dilation and groups are explicit; zero padding, ceil_mode=False.
The scalar reference loops over windows independently of the lowering matrix.
"""
import itertools
import math

from seal_artifact_gate import require

OPS = frozenset(("conv1d", "conv2d", "avg_pool1d", "avg_pool2d"))
INPUT_SHAPES = ((4,), (2, 2), (1, 4), (1, 2, 2), (2, 1, 2), (1, 1, 4))


def geometry(op, shape, weight_shape, kernel, stride, padding, include_pad=True, *, dilation=None, groups=1,
             max_input_elements=4,max_output_elements=4):
    require(type(max_input_elements) is int and type(max_output_elements) is int and
            ((max_input_elements,max_output_elements)==(4,4) or
             max_input_elements==256 and 1<=max_output_elements<=16),'Invalid spatial resource profile')
    require(op in OPS, "Unknown spatial operator")
    dims = 1 if op.endswith("1d") else 2
    require(len(shape) == dims+1 and all(type(n) is int and 1 <= n <= max_input_elements for n in shape)
            and math.prod(shape) <= max_input_elements, "Spatial input must be bounded unbatched channel-first tensor")
    dilation = (1,)*dims if dilation is None else dilation
    for name, values in (("kernel", kernel), ("stride", stride), ("padding", padding), ("dilation", dilation)):
        require(type(values) in (tuple, list) and len(values) == dims and
                all(type(n) is int and (0 <= n <= max_input_elements//2 if name == "padding" else 1 <= n <= max_input_elements) for n in values),
                "Invalid spatial " + name)
    require(type(include_pad) is bool, "Invalid count_include_pad")
    require(type(groups) is int and 1 <= groups <= max_input_elements, "Invalid Conv groups")
    if op.startswith("conv"):
        require(shape[0] % groups == 0 and weight_shape is not None and
                len(weight_shape) == dims+2 and weight_shape[0] % groups == 0 and
                tuple(weight_shape[1:]) == (shape[0]//groups, *kernel) and
                1 <= weight_shape[0] <= max_output_elements, "Conv weight/channel/kernel mismatch")
        channels = weight_shape[0]
    else:
        require(groups == 1 and all(d == 1 for d in dilation), "Pooling does not support groups/dilation")
        require(all(p <= k//2 for p, k in zip(padding, kernel)), "Pool padding exceeds half kernel")
        channels = shape[0]
    spatial = tuple((n+2*p-(k-1)*d-1)//s+1 for n, p, k, s, d in zip(shape[1:], padding, kernel, stride, dilation))
    require(all(n >= 1 for n in spatial) and math.prod((channels, *spatial)) <= max_output_elements,
            'Spatial scalar ciphertext output budget exceeded')
    return (channels, *spatial)


def index(coords, shape):
    result = 0
    for c, n in zip(coords, shape):
        result = result*n+c
    return result


def get(values, coords):
    for c in coords:
        values = values[c]
    return float(values)


def nested(values, shape):
    if len(shape) == 1:
        return list(values)
    width = math.prod(shape[1:])
    return [nested(values[i*width:(i+1)*width], shape[1:]) for i in range(shape[0])]


def lowering(op, shape, weight, weight_shape, bias, kernel, stride, padding, include_pad=True, *, dilation=None, groups=1,
             max_input_elements=4,max_output_elements=4):
    """Public coefficient matrix for compiler use only, not a numerical oracle."""
    dilation = (1,)*len(kernel) if dilation is None else dilation
    output_shape = geometry(op, shape, weight_shape, kernel, stride, padding, include_pad,
                            dilation=dilation, groups=groups,max_input_elements=max_input_elements,
                            max_output_elements=max_output_elements)
    matrix, biases = [], []
    for channel in range(output_shape[0]):
        for location in itertools.product(*(range(n) for n in output_shape[1:])):
            row = [0.] * math.prod(shape)
            valid = []
            for offset in itertools.product(*(range(n) for n in kernel)):
                position = tuple(o*s-p+k*d for o, s, p, k, d in zip(location, stride, padding, offset, dilation))
                if all(0 <= p < n for p, n in zip(position, shape[1:])):
                    valid.append((offset, position))
            if op.startswith("conv"):
                group_start = (channel // (output_shape[0]//groups)) * (shape[0]//groups)
                for local_channel in range(shape[0]//groups):
                    cin = group_start + local_channel
                    for offset, position in valid:
                        row[index((cin, *position), shape)] = get(weight, (channel, local_channel, *offset))
                biases.append(0. if bias is None else float(bias[channel]))
            else:
                divisor = math.prod(kernel) if include_pad else len(valid)
                require(divisor > 0, "Pool window has no valid elements")
                for _, position in valid:
                    row[index((channel, *position), shape)] = 1./divisor
                biases.append(0.)
            matrix.append(row)
    return matrix, biases, output_shape


def reference(op, value, shape, weight, bias, kernel, stride, padding, include_pad=True, *, dilation=None, groups=1):
    """Direct nested-loop scalar reference; never consumes a lowering matrix."""
    is_conv = op.startswith("conv")
    dilation = (1,)*len(kernel) if dilation is None else dilation
    channels = len(weight) if is_conv else shape[0]
    output_spatial = tuple((shape[i+1]+2*padding[i]-((kernel[i]-1)*dilation[i]+1))//stride[i]+1
                           for i in range(len(kernel)))
    outputs = []
    for cout in range(channels):
        for out in itertools.product(*(range(n) for n in output_spatial)):
            terms, valid_count = [], 0
            first_channel = (cout*groups//channels)*(shape[0]//groups) if is_conv else cout
            last_channel = first_channel + (shape[0]//groups if is_conv else 1)
            for cin in range(first_channel, last_channel):
                for offsets in itertools.product(*(range(k) for k in kernel)):
                    src = tuple(out[d]*stride[d]-padding[d]+offsets[d]*dilation[d] for d in range(len(kernel)))
                    if any(src[d] < 0 or src[d] >= shape[d+1] for d in range(len(kernel))):
                        continue
                    sample = get(value, (cin, *src))
                    terms.append(sample*get(weight, (cout, cin-first_channel, *offsets)) if is_conv else sample)
                    valid_count += 1
            total = math.fsum(terms)
            if is_conv:
                total += 0. if bias is None else bias[cout]
            else:
                total /= math.prod(kernel) if include_pad else valid_count
            outputs.append(total)
    return nested(outputs, (channels, *output_spatial))
