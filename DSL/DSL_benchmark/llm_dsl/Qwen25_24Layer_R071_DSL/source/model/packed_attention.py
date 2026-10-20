"""Round-eight candidate: scale Q before dot accumulation and bootstrap.

Pack [key, padded_heads, coordinate]. This aligns query replication and output
gathering with whole-token blocks; K/V groups expand using power-of-two shifts.
"""
import math
import numpy as np
from qwen24.kernels import qwen_layers as ql
from qwen24.kernels import qwen_nonlinear as nl
from qwen24.kernels.expr import Plain


def _add(terms):
    iterator=iter(terms);result=next(iterator)
    for term in iterator:result=result+term
    return result


def reciprocal_certificate(profile, length):
    """Public exact-arithmetic bound, not a bound on CKKS noise.

    Zero-mean z in [-b,b] implies 1 <= mean(exp(z)) <= cosh(b).
    Account for the Taylor remainder before selecting the midpoint.
    """
    b=profile.centered_bound
    if profile.squarings!=0 or b>0.5 or profile.exp_degree<8:
        return None
    remainder=math.exp(b)*b**(profile.exp_degree+1)/math.factorial(profile.exp_degree+1)
    lo=length*(1-remainder);hi=length*(math.cosh(b)+remainder)
    center=(lo+hi)/2
    radius=(hi-lo)/(hi+lo)
    assert lo>0 and radius<0.061
    return dict(lower=lo,upper=hi,center=center,residual_bound=radius,
                exact_relative_error_bound=radius**16,taylor_error_bound=remainder)


def corrected_reciprocal(x, center, constant):
    residual=x*constant(-1/center)+constant(1)
    result=(residual+constant(1))*constant(1/center)
    for _ in range(2):
        residual=residual*residual
        result=result*(residual+constant(1))
    # Recompute from the original denominator to correct accumulated error
    # in the independently propagated residual/product.
    return result*(constant(2)+x*result*constant(-1))


PAIR_DEGREE=13


def pair_odd(a, constant):
    """Approximate E8(a)/(E8(a)+E8(-a)) on [-.5,.5].

    Degree13 rational probability bound is 2.274e-8; lower degrees require their own audit.
    This does not bound CKKS noise. The caller establishes the public domain.
    """
    if PAIR_DEGREE not in (7,9,11,13):raise ValueError('Unsupported public pair polynomial degree')
    c=(1.,-1/3,2/15,-17/315,529/24192,-397/44800,7447/2073600)
    t=a*a;t2=t*t
    lower=constant(c[0])+t*constant(c[1])
    middle=constant(c[2])+t*constant(c[3])
    result=lower+t2*middle
    if PAIR_DEGREE>=9:
        t4=t2*t2;upper=constant(c[4])
        if PAIR_DEGREE>=11:upper=upper+t*constant(c[5])
        if PAIR_DEGREE>=13:upper=upper+t2*constant(c[6])
        result=result+t4*upper
    return a*result



def pair_probability(a, constant):
    return pair_odd(a,constant)*constant(.5)+constant(.5)


def _repeat(clean, width, count):
    """Input has exactly width meaningful prefix slots and a zero tail.

    Result contains the next power of two copies, so callers mask excess copies
    when count is not a power of two. No additional multiplier here.
    """
    result=clean;copies=1
    while copies<count:
        result=result+result.rotate(-copies*width)
        copies*=2
    return result


def _expand_kv(vector, query_heads, kv_heads, head_dim):
    group=query_heads//kv_heads
    # Guard the transient extra copy from circular wrap-around before masking.
    if (1<<(group-1).bit_length())*head_dim>vector.slots:
        raise ValueError('GQA expansion does not fit transient slot capacity')
    terms=[]
    for h in range(kv_heads):
        clean=ql.extract(vector,h*head_dim,head_dim).expr
        repeated=_repeat(clean,head_dim,group)
        if group & (group-1):
            repeated=repeated*Plain(np.ones(group*head_dim))
        offset=h*group*head_dim
        terms.append(repeated.rotate(-offset) if offset else repeated)
    return _add(terms)


def gqa_attention(queries, keys, values, *, query_positions, key_positions,
                  exp=None, reciprocal=None, query_heads=14, kv_heads=2,
                  head_dim=64, public_shift=0.0, normalization=None):
    queries,keys,values=tuple(queries),tuple(keys),tuple(values)
    for value,name in ((query_heads,'query_heads'),(kv_heads,'kv_heads'),(head_dim,'head_dim')):
        ql._integer(value,name)
    if not queries or not keys or len(keys)!=len(values) or query_heads%kv_heads:
        raise ValueError('Invalid Q/K/V counts or GQA ratio')
    if head_dim & (head_dim-1):raise ValueError('Power-of-two head dimension required')
    slots=queries[0].slots;hidden=query_heads*head_dim
    if any(x.slots!=slots or x.size!=hidden for x in queries):raise ValueError('Q shape mismatch')
    if any(x.slots!=slots or x.size!=kv_heads*head_dim for x in keys+values):raise ValueError('KV shape mismatch')
    if not isinstance(normalization,nl.NormalizeSquareSoftmax) or exp is not None or reciprocal is not None or public_shift:
        raise ValueError('An unambiguous normalize-square profile is required')
    qp=ql._positions(query_positions,len(queries),'query_positions')
    kp=ql._positions(key_positions,len(keys),'key_positions')
    token_width=1<<(hidden-1).bit_length()
    outputs=[]
    for query,position in zip(queries,qp):
        visible=[i for i,p in enumerate(kp) if p<=position];length=len(visible)
        if not 0<length<=normalization.max_length:raise ValueError('Visible length exceeds profile')
        padded=1<<(length-1).bit_length()
        certificate=reciprocal_certificate(normalization,length)
        if token_width*padded>slots:raise ValueError('Layout exceeds physical slots')
        if length==1:
            outputs.append(ql.PackedVector(_expand_kv(values[visible[0]],query_heads,kv_heads,head_dim),hidden,slots))
            continue
        if length==2 and normalization.squarings==0 and normalization.exp_degree==8 and normalization.centered_bound<=.5:
            minus=Plain(np.full(slots,-1.))
            def compact(expr):return ql.PackedVector(expr,kv_heads*head_dim,slots)
            # Form one key difference in compact KV-head coordinates, then
            # expand it once. This avoids two key blocks and query replication.
            kd=compact(keys[visible[0]].expr+keys[visible[1]].expr*minus)
            # Fuse public dot scaling into the input prefix mask. Otherwise
            # automatic Boot can see an unscaled sum of magnitude eight even
            # when the final centered score is within [-.5,.5].
            scaled_q=query.expr*Plain(np.full(hidden,.5/math.sqrt(head_dim)))
            a=scaled_q*_expand_kv(kd,query_heads,kv_heads,head_dim)
            step=1
            while step<head_dim:
                a=a+a.rotate(step);step*=2
            mask=np.zeros(slots);mask[:hidden:head_dim]=1.
            a=_repeat(a*Plain(mask),1,head_dim)
            odd=pair_odd(a,lambda x:Plain(np.full(slots,x)))
            # Addition/subtraction commute with the linear GQA expansion.
            # Expand sums/differences directly; do not retain expanded V0/V1.
            vs=compact(values[visible[0]].expr+values[visible[1]].expr)
            vd=compact(values[visible[0]].expr+values[visible[1]].expr*minus)
            out=_expand_kv(vs,query_heads,kv_heads,head_dim)+odd*_expand_kv(vd,query_heads,kv_heads,head_dim)
            outputs.append(ql.PackedVector(out*Plain(np.full(hidden,.5)),hidden,slots))
            continue
        active=np.zeros(slots);starts=np.zeros(slots)
        for t in range(length):
            active[t*token_width:t*token_width+hidden]=1
            starts[t*token_width:t*token_width+hidden:head_dim]=1/math.sqrt(head_dim)
        first=Plain(np.ones(hidden))
        def pack_tokens(vectors):
            terms=[]
            for t,i in enumerate(visible):
                term=_expand_kv(vectors[i],query_heads,kv_heads,head_dim)
                terms.append(term.rotate(-t*token_width) if t else term)
            return _add(terms)
        packed_q=_repeat(ql.extract(query,0,hidden).expr,token_width,padded)
        dots=packed_q*pack_tokens(keys)
        step=1
        while step<head_dim:
            dots=dots+dots.rotate(step);step*=2
        dots=dots*Plain(starts)
        dots=_repeat(dots,1,head_dim)  # Each nonzero start expands within its own head.

        def first_sum(clean):
            # Each token block preserves head coordinates; only the first block
            # is retained, excluding partial/cyclic sums at all other offsets.
            summed=clean;step=token_width
            while step<token_width*padded:
                summed=summed+summed.rotate(step);step*=2
            return summed*first

        def total_clean(clean):
            return _repeat(first_sum(clean),token_width,padded)

        def total(expr):
            # Exponential's padded values are generally nonzero, so this mask
            # remains mandatory for every normalization denominator.
            summed=total_clean(expr*Plain(active))
            if length!=padded:
                summed=summed*Plain(active)
            # Every physical slot enters bootstrap, including unused slots.
            # A zero reciprocal denominator makes Goldschmidt grow to 16 in
            # the two-key profile. Set unused denominators to one (within both
            # the initial and squaring-round upper bounds), then mask the
            # resulting probability as before. Valid arithmetic is unchanged.
            return summed+Plain((length if certificate else 1)*(1-active))

        constant=lambda value:Plain(np.full(slots,value))
        centered=(dots+total_clean(dots)*constant(-1/length))*Plain(active)
        if certificate:
            numerator=nl.small_exp(centered,normalization.exp_degree,constant)
            denominator=total(numerator)
            inverse=corrected_reciprocal(denominator,certificate['center'],constant)
            probabilities=numerator*(inverse*Plain(active))
        else:
            probabilities=nl.normalize_square(centered,normalization,length,constant,total,
                lambda expr:expr*Plain(active),scalar_constant=constant)
        # Both operands have zero invalid slots: avoid a redundant mask and
        # avoid broadcasting a sum that will only be returned in the first block.
        out=first_sum(probabilities*pack_tokens(values))
        outputs.append(ql.PackedVector(out,hidden,slots))
    return tuple(outputs)


def pack_kv(keys,values):
    keys,values=tuple(keys),tuple(values)
    if not keys or len(keys)!=len(values):raise ValueError('Matching nonempty K/V required')
    if any((x.size,x.slots)!=(keys[0].size,keys[0].slots) for x in keys+values):raise ValueError('KV shape mismatch')
    return ql.concatenate(tuple(x for pair in zip(keys,values) for x in pair))


def unpack_kv(cache, *, tokens, kv_size):
    ql._integer(tokens,'tokens');ql._integer(kv_size,'kv_size')
    if cache.size!=tokens*kv_size*2:raise ValueError('Cache metadata mismatch')
    return (tuple(ql.extract(cache,2*i*kv_size,kv_size) for i in range(tokens)),
            tuple(ql.extract(cache,(2*i+1)*kv_size,kv_size) for i in range(tokens)))
