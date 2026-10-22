"""Power-of-two, channel-interleaved ImageNet slots using only DSL arithmetic.

Physical spatial padding is masked at every layer. Tensor values are divided by
the model boundary; padding and input parity packing do not change the model.
"""
import math
import numpy as np
import torch


def shape(channels, height, k=1, slots=32768):
    tiles = math.ceil(channels / (k * k))
    return dict(co=channels, ho=height, wo=height, ko=k, to=tiles,
                nt=slots, no=math.ceil(tiles * height * height * k * k / slots), po=1)


def indices(s):
    c, y, x = np.indices((s['co'], s['ho'], s['wo']))
    k, h = s['ko'], s['ho']
    return (c // (k*k)) * h*h*k*k + y*k*k*h + (c//k % k)*h*k + x*k + c % k


def pack(tensor, s):
    result = np.zeros(s['no'] * s['nt'], dtype=np.float64)
    result[indices(s).reshape(-1)] = np.asarray(tensor).reshape(-1)
    return [torch.from_numpy(row.copy()) for row in result.reshape(s['no'], s['nt'])]


def unpack(values, s):
    return torch.cat(list(values))[torch.from_numpy(indices(s).reshape(-1))].reshape(
        s['co'], s['ho'], s['wo'])


def plain(value, clear):
    if clear:
        return torch.from_numpy(np.asarray(value, dtype=np.float64).copy())
    import hecate as hc
    return hc.Plain(np.asarray(value, dtype=np.float64))


def roll(value, shift, clear):
    return torch.roll(value, int(shift)) if clear else value.rotate(-int(shift))


def add(terms):
    result = None
    for term in terms:
        result = term if result is None else result + term
    if result is None:
        raise ValueError('empty slot sum')
    return result


def conv(values, source, weight, bias, stride, valid, clear):
    """Channel-parallel convolution; fold BN into weight/bias before calling.

    Each input pack contains whole spatial tiles, so tile reduction is limited
    to the tiles actually present in a pack, rather than total tensor tiles.
    """
    weight = np.asarray(weight)
    co, ci, fh, fw = weight.shape
    if ci != source['co'] or fh != fw or fh % 2 != 1:
        raise ValueError('expected odd square kernel matching input channels')
    nt, hi, ki = source['nt'], source['ho'], source['ko']
    target = shape(co, hi // stride, ki * stride, nt)
    tile_size = hi * hi * ki * ki
    if nt % tile_size:
        raise ValueError('spatial tile must divide the ciphertext')
    slot = np.arange(source['no'] * nt)
    tile, local = slot // tile_size, slot % tile_size
    yy = local // (ki*ki*hi)
    ky = local // (ki*hi) % ki
    xx = local // ki % hi
    kx = local % ki
    channel = tile * ki*ki + ky*ki + kx
    channel = channel.reshape(source['no'], nt)
    yy, xx = yy.reshape(source['no'], nt), xx.reshape(source['no'], nt)
    dest_indices = indices(target)
    outputs = [None] * target['no']
    rotations = {}
    for kh in range(fh):
        for kw in range(fw):
            if not np.any(weight[:, :, kh, kw]):
                continue
            dy, dx = kh - fh//2, kw - fw//2
            delta = dy*ki*ki*hi + dx*ki
            rotations[kh, kw] = [roll(v, -delta, clear) if delta else v for v in values]
    for oc in range(co):
        def products():
            for (kh, kw), rotated in rotations.items():
                dy, dx = kh-fh//2, kw-fw//2
                valid_slots = ((channel < ci) & (yy+dy >= 0) & (yy+dy < hi)
                               & (xx+dx >= 0) & (xx+dx < hi))
                for ip, value in enumerate(rotated):
                    w = weight[oc, np.minimum(channel[ip], ci-1), kh, kw] * valid_slots[ip]
                    if np.any(w):
                        yield value * plain(w, clear)
        reduced = add(products())
        for count, step in ((ki, 1), (ki, ki*hi),
                            (min(source['to'], nt//tile_size), tile_size)):
            if count & (count-1):
                raise ValueError('reduction length must be a power of two')
            for j in range(count.bit_length()-1):
                reduced = reduced + roll(reduced, -(1 << j)*step, clear)
        dest = dest_indices[oc, :valid, :valid].reshape(-1)
        op = int(dest[0] // nt)
        if np.any(dest // nt != op):
            raise ValueError('channel crosses a ciphertext boundary')
        shift = int(dest[0] % nt)
        mask = np.zeros(nt)
        mask[dest % nt] = 1.0
        term = roll(reduced, shift, clear) * plain(mask, clear)
        outputs[op] = term if outputs[op] is None else outputs[op] + term
    packed_bias = pack(np.broadcast_to(np.asarray(bias)[:, None, None],
                                      (co, target['ho'], target['wo'])), target)
    for op in range(target['no']):
        mask = np.zeros((co, target['ho'], target['wo']))
        mask[:, :valid, :valid] = 1.0
        b = packed_bias[op].numpy() * pack(mask, target)[op].numpy()
        if np.any(b):
            outputs[op] = outputs[op] + plain(b, clear)
    return outputs, target


def gather(values, source_indices, destination_indices, output_packs, slots, clear, gain=1.0):
    """Sparse permutation/reduction grouped by source pack, dest pack and rotation."""
    src, dst = np.asarray(source_indices).reshape(-1), np.asarray(destination_indices).reshape(-1)
    keys = np.stack((src//slots, dst//slots, (dst-src) % slots), axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    result = [None] * output_packs
    for group, (ip, op, shift) in enumerate(unique):
        mask = np.zeros(slots)
        np.add.at(mask, dst[inverse == group] % slots, gain)
        term = roll(values[int(ip)], int(shift), clear) * plain(mask, clear)
        result[int(op)] = term if result[int(op)] is None else result[int(op)] + term
    if any(value is None for value in result):
        raise ValueError('empty destination pack')
    return result


def avg_pool(values, source, valid, clear):
    target = shape(source['co'], source['ho']//2, source['ko']*2, source['nt'])
    src, dst = indices(source), indices(target)[:, :valid, :valid]
    c, y, x = np.indices(dst.shape)
    terms = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = y*2+dy, x*2+dx
            keep = (yy >= 0) & (xx >= 0) & (yy < 2*valid) & (xx < 2*valid)
            terms.append(gather(values, src[c[keep], yy[keep], xx[keep]], dst[keep],
                                target['no'], source['nt'], clear, 1/9))
    return [add(t[i] for t in terms) for i in range(target['no'])], target


def global_pool(values, source, valid, clear):
    k, h = source['ko'], source['ho']
    # Average before the reduction: identical mean without large partial sums.
    reciprocal = plain(np.full(source['nt'], 1/(valid*valid)), clear)
    reduced = [value * reciprocal for value in values]
    for step in (k, k*k*h):
        for j in range(h.bit_length()-1):
            reduced = [v + roll(v, -(1 << j)*step, clear) for v in reduced]
    return gather(reduced, indices(source)[:, 0, 0], np.arange(source['co']),
                  1, source['nt'], clear)


def linear(values, weight, bias, slots, clear):
    """Rectangular diagonal matmul; baby/giant rotations keep keys and work small."""
    weight, bias = np.asarray(weight), np.asarray(bias)
    rows, cols = weight.shape
    baby_size = 32
    babies = [roll(values[0], -b, clear) for b in range(baby_size)]
    groups = {}
    for delta in range(-(rows-1), cols):
        giant, baby = divmod(delta, baby_size)
        diagonal = np.zeros(slots)
        row = np.arange(max(0, -delta), min(rows, cols-delta))
        diagonal[row] = weight[row, row+delta]
        if not np.any(diagonal):
            continue
        term = babies[baby] * plain(np.roll(diagonal, giant*baby_size), clear)
        groups[giant] = term if giant not in groups else groups[giant] + term
    out = add(roll(v, -g*baby_size, clear) for g, v in groups.items())
    padded = np.zeros(slots)
    padded[:rows] = bias
    return [out + plain(padded, clear)]
