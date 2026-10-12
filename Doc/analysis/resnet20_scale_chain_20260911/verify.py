"""Independent metadata replay for the 22-Q candidate; NOT CKKS simulation."""
import json
import math
from search import Planner, TREES, actual_chain, prime_test


def verify(plan, primes, floor=40):
    logs = [math.log2(q) for q in primes]
    prefix = [0.0]
    for b in logs:
        prefix.append(prefix[-1]+b)
    assert len(primes) == len(set(primes))
    assert all(q % 131072 == 1 and prime_test(q) for q in primes)
    min_cipher = 1000.0
    min_plain = 1000.0
    max_scale = 0.0
    def track(s):
        nonlocal min_cipher, max_scale
        min_cipher = min(min_cipher, s)
        max_scale = max(max_scale, s)
        assert s >= floor-1e-7
    def equal(a, b):
        assert abs(a-b) < 1e-7, (a,b)
    def capacity(k,s):
        assert prefix[-1]-prefix[k]-s >= 32
    def rescale(work,end,pre,post):
        assert work <= end
        equal(pre-(prefix[end]-prefix[work]), post)
        track(pre)
        track(post)
        capacity(work,pre)
        capacity(end,post)
    previous_end, previous_scale = 0,40
    for stage in plan['stages']:
        start, sin = stage['start'], stage['input_scale']
        assert start == previous_end
        equal(sin,previous_scale)
        basis = {1:(start,sin)}
        for e in stage['basis']:
            d = e['degree']
            if d & (d-1) == 0:
                a = b = d//2
                diff = 0
            else:
                a = 1 << (d.bit_length()-1)
                b = d-a
                diff = 2*a-d
            ka,sa = basis[a]
            kb,sb = basis[b]
            assert ka <= e['work'] and kb <= e['work']
            equal(sa+sb,e['pre'])
            if diff:
                kd,sd = basis[diff]
                assert kd <= e['work']
                alignment_scale = e['pre']-sd
                min_plain = min(min_plain,alignment_scale)
                assert alignment_scale >= floor-1e-7
            rescale(e['work'],e['output'],e['pre'],e['scale'])
            basis[d] = e['output'],e['scale']
        def visit(node,tree,degree):
            nonlocal min_plain
            work,end,pre,post = node['work'],node['output'],node['pre'],node['scale']
            rescale(work,end,pre,post)
            if tree is None:
                assert node['type'] == 'leaf' and node['degree'] == degree
                for d,enc in zip(range(1,degree+1,2),node['encoding_scales'],strict=True):
                    k,s = basis[d]
                    assert k <= work
                    equal(s+enc,pre)
                    min_plain = min(min_plain,enc)
                    assert enc >= floor-1e-7
            else:
                split,left,right = tree
                assert node['split'] == split
                k,s = basis[split]
                assert k <= work
                quotient,remainder = node['quotient'],node['remainder']
                assert quotient['output'] == work == remainder['output']
                equal(quotient['scale']+s,pre)
                equal(remainder['scale'],pre)
                visit(quotient,right,degree-split)
                visit(remainder,left,split-1)
        visit(stage['tree'],TREES[stage['degree']][0],stage['degree'])
        assert stage['tree']['output'] == stage['end']
        equal(stage['tree']['scale'],stage['output_scale'])
        previous_end,previous_scale = stage['end'],stage['output_scale']
    assert previous_end <= plan['tail_work']
    assert plan['tail_work']+plan['tail_drop'] == plan['total']
    rescale(plan['tail_work'],plan['total'],40+previous_scale,40)
    equal(prefix[plan['total']],plan['dropped_bits'])
    return {'checks':'PASS', 'total_q_dropped':plan['total'],
            'minimum_cipher_logscale':min_cipher,
            'minimum_plain_logscale':min_plain,
            'maximum_intermediate_logscale':max_scale,
            'output_logscale':40,'output_q_count':len(primes)-plan['total'],
            'warning':'METADATA ONLY: no CKKS encoding, noise or encrypted correctness checked'}


def candidate(mixed=False):
    widths = [30]*38
    scales = (40.1,44)
    if mixed:
        widths[20] = 25
        scales = (40.2,44)
    primes = actual_chain(widths)
    plan = Planner([math.log2(q) for q in primes]).relu_fixed(stage_scales=scales,trace=True)
    return widths,primes,plan


if __name__ == '__main__':
    for mixed in (False,True):
        widths,primes,plan = candidate(mixed)
        print(json.dumps({'mixed':mixed,'verification':verify(plan,primes),
                          'widths_top_first':widths,'primes_top_first':primes,
                          'plan':plan}))
