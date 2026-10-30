"""Token/attention arithmetic expressed through the existing backend interface."""

import numpy as np

from . import approx, multi
from .layout import (FlatLayout, MultiFlatLayout, TokenLayout, ScoreLayout,
                     KernelError, layout_blocks, multi_flat_layout)
from .linear import linear_flat
from .poly import act_cheb, act_minimax
from .slots import memoize, scope, vector


def values(value, layout):
    return list(value) if isinstance(layout, (TokenLayout, ScoreLayout, MultiFlatLayout)) else [value]


def plain(b, size, start=0, coefficient=1.0):
    def build():
        result = np.zeros(b.slots)
        result[start:start+size] = coefficient
        return result
    result = vector(b, ('transformer.plain', b.slots, size, start,
                        float(coefficient).hex()), build)
    return b.const(result)


def accumulate(items):
    result = None
    for item in items:
        result = item if result is None else result+item
    if result is None:
        raise KernelError('empty encrypted reduction')
    return result


def broadcast(b, scalar, width):
    result = scalar
    offset = 1
    while offset < width:
        result = result+result.rotate(-offset)
        offset *= 2
    return result*plain(b, width)


def scalar_sum(b, value, start, size):
    result = value*plain(b, size, start)
    if start:
        result = result.rotate(start)
    offset = 1
    while offset < size:
        result = result+result.rotate(offset)
        offset *= 2
    return result*plain(b, 1)


@scope
def gather(b, xs, source, index, sign=None, *, output=None):
    """Static last-axis gather, including cross-block edges and signed shuffles."""
    output = output or multi_flat_layout(b.slots, len(index))
    input_blocks = layout_blocks(source)
    result = []
    sign = np.ones(len(index), dtype=int) if sign is None else sign
    def build_groups():
        blocks = []
        for lo, hi, block in layout_blocks(output):
            groups = {}
            for j in range(lo, hi):
                k = int(index[j])
                source_block = next(i for i, (a, z, _) in enumerate(input_blocks) if a <= k < z)
                offset = (k-input_blocks[source_block][0])-(j-lo)
                mask = groups.setdefault((source_block, offset), np.zeros(b.slots))
                mask[j-lo] = sign[j]
            ordered = tuple(sorted(groups.items()))
            for _, mask in ordered:
                mask.flags.writeable = False
            blocks.append(ordered)
        return tuple(blocks)
    grouped = memoize(b, ('transformer.gather', b.slots, source, output,
                          tuple(int(v) for v in index),
                          tuple(float(v).hex() for v in sign)), build_groups)
    for groups in grouped:
        result.append(accumulate(xs[i].rotate(offset)*b.const(mask)
                                 for (i, offset), mask in groups))
    return result


@scope
def tall_linear(b, x, lin, weight, bias=None):
    weight = np.asarray(weight, dtype=np.float64)
    O, I = weight.shape
    Ip = 1 << (I-1).bit_length()
    if not isinstance(lin, FlatLayout) or lin.size != I or 2*Ip > b.slots:
        raise KernelError('tall linear requires a single input block')
    repeated, step = x, Ip
    while step < b.slots:
        repeated = repeated+repeated.rotate(-step)
        step *= 2
    rotated = [repeated.rotate(d) for d in range(Ip)]
    output = multi_flat_layout(b.slots, O)
    result = []
    for start, stop, block in layout_blocks(output):
        rows = np.arange(stop-start)
        total = None
        for d in range(Ip):
            def build_mask():
                columns = (rows+d) % Ip
                mask = np.zeros(b.slots)
                valid = columns < I
                mask[rows[valid]] = weight[start+rows[valid], columns[valid]]
                return mask
            mask = vector(b, ('transformer.tall', id(weight), b.slots, lin,
                              start, stop, d), build_mask, owners=(weight,))
            term = rotated[d]*b.const(mask)
            total = term if total is None else total+term
        if bias is not None:
            def build_bias():
                offset = np.zeros(b.slots)
                offset[:stop-start] = bias[start:stop]
                return offset
            offset = vector(b, ('transformer.tall_bias', id(bias), b.slots,
                                start, stop), build_bias, owners=(bias,))
            total = total+b.const(offset)
        result.append(total)
    return (result if isinstance(output, MultiFlatLayout) else result[0]), output


@scope
def linear(b, xs, layout, weight, bias=None, *, algo='bsgs', n1=None, tall_min_ratio=4):
    weight = np.asarray(weight, dtype=np.float64)
    O, I = weight.shape
    Ip = 1 << (I-1).bit_length()
    source = layout.token if isinstance(layout, TokenLayout) else layout
    count = len(layout_blocks(source))
    groups = layout.L if isinstance(layout, TokenLayout) else 1
    result, output = [], None
    for t in range(groups):
        chunk = xs[t*count:(t+1)*count]
        value = chunk if isinstance(source, MultiFlatLayout) else chunk[0]
        if isinstance(source, FlatLayout) and 2*Ip <= b.slots and O/Ip >= tall_min_ratio:
            value, output = tall_linear(b, value, source, weight, bias)
        else:
            run = multi.linear_blocks if isinstance(source, MultiFlatLayout) or O > I else linear_flat
            if isinstance(source, FlatLayout) and O > I:
                value = [value]
                source = MultiFlatLayout(source.n, source.size)
            value, output = run(b, value, source, weight, bias, algo=algo, n1=n1)
        result.extend(values(value, output))
    if isinstance(layout, TokenLayout):
        return result, TokenLayout(b.slots, layout.L, O)
    return (result if isinstance(output, MultiFlatLayout) else result[0]), output


def approximation(a, prefix, fn):
    result=dict(fn=fn, **{key: a.get(prefix+'_'+key) for key in
                ('method', 'domain', 'deg', 'degrees', 'eps', 'deg0', 'iters')})
    if a.get(prefix+'_fit') is not None:result['fit']=a[prefix+'_fit']
    if prefix+'_coefficients' in a:result['coefficients']=a[prefix+'_coefficients']
    return result


def configuration(a):
    if a['method'] in approx.NEW_METHODS:
        return approx.configuration(a)
    if a['method'] == 'minimax_comp':
        return 'minimax_comp', tuple(a['degrees']), *a['domain']
    return 'cheb', a['deg'], *a['domain']


def apply(b, value, a, *, exact=False):
    if exact:
        from ..backends.mock import MockExpr
        return MockExpr(b, approx.function(a['fn'], value.value))
    if a['method'] in approx.NEW_METHODS:
        return approx.activation(b, value, a)
    if a['method'] == 'cheb':
        return act_cheb(b, value, a['fn'], a['deg'], a['domain'])
    if a['method'] == 'minimax_comp':
        return act_minimax(b, value, tuple(a['degrees']), a['domain'])
    if a['method'] == 'exact' and a['fn'] == 'square':
        return value*value
    raise KernelError('unsupported approximation')


@scope
def activation(b, xs, layout, a, *, exact=False):
    from ..backends.mock import MockExpr
    result = []
    for x, (_, _, block) in zip(xs, layout.blocks):
        if exact:
            out = np.zeros_like(x.value)
            out[..., :block.size] = approx.function(a['fn'], x.value[..., :block.size])
            result.append(MockExpr(b, out))
        else:
            result.append(apply(b, x, a)*plain(b, block.size))
    return result


@scope
def normalize(b, xs, layout, weight, bias, eps, a, *, rms=False, exact=False, observe=None):
    if not isinstance(layout, TokenLayout):
        raise KernelError('normalization requires TokenLayout')
    blocks = layout_blocks(layout.token)
    result = []
    for t in range(layout.L):
        chunk = xs[t*len(blocks):(t+1)*len(blocks)]
        mean = accumulate(scalar_sum(b, x, 0, block.size) for x, (_, _, block) in zip(chunk, blocks))*(1/layout.D)
        centered = chunk if rms else [x-broadcast(b, mean, block.size)
                                      for x, (_, _, block) in zip(chunk, blocks)]
        variance = accumulate(scalar_sum(b, x*x, 0, block.size)
                              for x, (_, _, block) in zip(centered, blocks))*(1/layout.D)
        # Broadcast the positive variance before evaluating rsqrt: every slot
        # then has a value in the same declared positive interval.
        variance = broadcast(b, variance, b.slots)+eps
        if observe is not None:
            observe(variance, FlatLayout(b.slots, 1), a)
        inverse = apply(b, variance, a, exact=exact)
        for x, (lo, hi, block) in zip(centered, blocks):
            value = x*inverse
            if weight is not None:
                def build_weight():
                    mask = np.zeros(b.slots); mask[:hi-lo] = weight[lo:hi]
                    return mask
                mask = vector(b, ('transformer.norm_weight', id(weight), b.slots, lo, hi),
                              build_weight, owners=(weight,))
                value = value*b.const(mask)
            else:
                value = value*plain(b, block.size)
            if bias is not None:
                def build_bias():
                    offset = np.zeros(b.slots); offset[:hi-lo] = bias[lo:hi]
                    return offset
                offset = vector(b, ('transformer.norm_bias', id(bias), b.slots, lo, hi),
                                build_bias, owners=(bias,))
                value = value+b.const(offset)
            result.append(value)
    return result, layout


@scope
def matmul(b, left, ll, right, rl, *, transpose_b, heads, kv_heads, scale=None, keep='all'):
    if not isinstance(rl, TokenLayout) or heads < kv_heads or heads % kv_heads:
        raise KernelError('invalid attention head mapping')
    if transpose_b:
        if not isinstance(ll, TokenLayout) or ll.D % heads or rl.D % kv_heads or ll.D//heads != rl.D//kv_heads:
            raise KernelError('Q/K head widths differ')
        dh = ll.D//heads
        out = ScoreLayout(b.slots, ll.L, heads, rl.L)
        qcount, kcount = len(layout_blocks(ll.token)), len(layout_blocks(rl.token))
        result = []
        for i in range(ll.L):
            terms = []
            for h in range(heads):
                q = gather(b, left[i*qcount:(i+1)*qcount], ll.token,
                           range(h*dh, (h+1)*dh), output=FlatLayout(b.slots, dh))[0]
                kh = h//(heads//kv_heads)
                for j in range(rl.L):
                    if keep == 'causal' and j > i:
                        continue
                    k = gather(b, right[j*kcount:(j+1)*kcount], rl.token,
                               range(kh*dh, (kh+1)*dh), output=FlatLayout(b.slots, dh))[0]
                    dot = scalar_sum(b, q*k, 0, dh)
                    if scale is not None:
                        dot = dot*scale
                    terms.append(dot.rotate(-(h*rl.L+j)))
            result.append(accumulate(terms))
        return result, out
    if not isinstance(ll, ScoreLayout) or ll.H != heads or ll.Lk != rl.L or rl.D % kv_heads:
        raise KernelError('P/V dimensions differ')
    dh = rl.D//kv_heads
    out = TokenLayout(b.slots, ll.L, heads*dh)
    vcount = len(layout_blocks(rl.token))
    output_blocks = layout_blocks(out.token)
    result = []
    for i, p in enumerate(left):
        terms = [[] for _ in output_blocks]
        for h in range(heads):
            kh = h//(heads//kv_heads)
            for j in range(rl.L):
                if keep == 'causal' and j > i:
                    continue
                probability = scalar_sum(b, p, h*rl.L+j, 1)
                part = gather(b, right[j*vcount:(j+1)*vcount], rl.token,
                              range(kh*dh, (kh+1)*dh), output=FlatLayout(b.slots, dh))[0]
                product = broadcast(b, probability, dh)*part
                for c, (lo, hi, _) in enumerate(output_blocks):
                    start, stop = max(lo, h*dh), min(hi, (h+1)*dh)
                    if start < stop:
                        clipped = product*plain(b, stop-start, start-h*dh)
                        terms[c].append(clipped.rotate(lo-h*dh))
        result.extend(accumulate(group) for group in terms)
    return result, out


def _square_normalize(b, probabilities, widths, steps, step_inv, exact, observe, final_inv=None):
    for index in range(steps):
        squares = [p*p for p in probabilities]
        total = accumulate(scalar_sum(b, q, 0, width) for q, width in zip(squares, widths))
        total = broadcast(b, total, b.slots)
        last = final_inv is not None and index == steps-1
        attrs = final_inv if last else step_inv
        if observe is not None:
            observe('final_inv' if last else f'step_inv[{index}]', total, FlatLayout(b.slots, 1), attrs, None)
        inverse = apply(b, total, attrs, exact=exact)
        inverse_result = getattr(observe, 'inverse_result', None)
        if inverse_result is not None:
            inverse_result('final_inv' if last else f'step_inv[{index}]', total, inverse)
        probabilities = [q*inverse for q in squares]
    return probabilities


@scope
def softmax(b, xs, layout, exp, inv, *, keep='all', mask=None, mask_layout=None,
            shift=0.0, inv_scale=1.0, steps=0, step_inv=None, final_inv=None, exact=False, observe=None):
    if isinstance(layout, TokenLayout) and len(layout_blocks(layout.token)) > 1:
        blocks = layout_blocks(layout.token)
        result = []
        for i in range(layout.L):
            exponentials = []
            active = min(layout.D, i+1) if keep == 'causal' else layout.D
            for c, (lo, hi, block) in enumerate(blocks):
                width = max(0, min(hi, active)-lo)
                x = xs[i*len(blocks)+c]
                if shift != 0:
                    x = x-shift
                if steps:
                    x = x*(2.0**-steps)
                key = None
                if mask is not None:
                    key = gather(b, mask, mask_layout, range(lo, hi), output=block)[0]
                if observe is not None and width:
                    observe('exp', x, FlatLayout(b.slots, width), exp, key)
                e = apply(b, x, exp, exact=exact)*plain(b, width)
                exponentials.append(e if key is None else e*key)
            total = accumulate(scalar_sum(b, e, 0, block.size)
                               for e, (_, _, block) in zip(exponentials, blocks))
            total = broadcast(b, total, b.slots)
            if inv_scale != 1:
                total = total*(1/inv_scale)
            if observe is not None:
                observe('inv', total, FlatLayout(b.slots, 1), inv, None)
            inverse = apply(b, total, inv, exact=exact)
            inverse_result = getattr(observe, 'inverse_result', None)
            if inverse_result is not None:
                inverse_result('inv', total, inverse)
            if inv_scale != 1:
                inverse = inverse*(1/inv_scale)
            probabilities = [e*inverse for e in exponentials]
            if steps:
                probabilities = _square_normalize(b, probabilities,
                    [block.size for _, _, block in blocks], steps, step_inv, exact, observe, final_inv)
            result.extend(probabilities)
        return result, layout
    if isinstance(layout, ScoreLayout):
        heads, width, length = layout.H, layout.Lk, layout.L
    elif isinstance(layout, TokenLayout) and len(layout_blocks(layout.token)) == 1:
        heads, width, length = 1, layout.D, layout.L
    else:
        raise KernelError('softmax input requires ScoreLayout or a single-block TokenLayout')
    result = []
    key = (gather(b, mask, mask_layout, range(width), output=FlatLayout(b.slots, width))[0]
           if mask is not None and mask_layout is not None else mask)
    for i in range(length):
        terms = []
        for h in range(heads):
            active = min(width, i+1) if keep == 'causal' else width
            if steps:
                # Fold the public 2^-k into the existing head mask multiply.
                segment = (xs[i].rotate(h*width)-shift)*plain(b, active, coefficient=2.0**-steps)
            else:
                segment = xs[i].rotate(h*width)*plain(b, active)
                if shift != 0:
                    segment = segment-shift
            if observe is not None:
                observe('exp', segment, FlatLayout(b.slots, active), exp, key)
            e = apply(b, segment, exp, exact=exact)*plain(b, active)
            if key is not None:
                e = e*key
            total = scalar_sum(b, e, 0, active)
            total = broadcast(b, total, b.slots)
            if inv_scale != 1:
                total = total*(1/inv_scale)
            if observe is not None:
                observe('inv', total, FlatLayout(b.slots, 1), inv, None)
            inverse = apply(b, total, inv, exact=exact)
            inverse_result = getattr(observe, 'inverse_result', None)
            if inverse_result is not None:
                inverse_result('inv', total, inverse)
            if inv_scale != 1:
                inverse = inverse*(1/inv_scale)
            probability = e*inverse
            if steps:
                probability = _square_normalize(b, [probability], [active],
                    steps, step_inv, exact, observe, final_inv)[0]
            terms.append(probability.rotate(-h*width))
        result.append(accumulate(terms))
    return result, layout


@scope
def shuffle(b, xs, layout, index, sign):
    if not isinstance(layout, TokenLayout) or len(index) != layout.D:
        raise KernelError('shuffle requires a token-width static index map')
    count = len(layout_blocks(layout.token))
    return [part for t in range(layout.L) for part in gather(b,
            xs[t*count:(t+1)*count], layout.token, index, sign)], layout


@scope
def select(b, xs, layout, mask, mask_layout):
    if not isinstance(layout, TokenLayout) or not isinstance(mask_layout, (FlatLayout, MultiFlatLayout)) or mask_layout.size != layout.L:
        raise KernelError('select requires token input and a matching one-hot mask')
    count = len(layout_blocks(layout.token))
    result = []
    mask_blocks = layout_blocks(mask_layout)
    for c, (_, _, block) in enumerate(layout_blocks(layout.token)):
        terms = []
        for t in range(layout.L):
            index, (lo, hi, _) = next((j, entry) for j, entry in enumerate(mask_blocks) if entry[0] <= t < entry[1])
            coefficient = scalar_sum(b, mask[index], t-lo, 1)
            terms.append(xs[t*count+c]*broadcast(b, coefficient, block.size))
        result.append(accumulate(terms))
    return result, TokenLayout(layout.n, 1, layout.D)
