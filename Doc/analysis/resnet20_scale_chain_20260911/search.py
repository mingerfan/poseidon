"""Public scale/prime metadata search. No HE library, keys, ciphertexts or GPU.

Original odd-baby decomposition is ported from the application's gpu_relu.cpp.
Search is deliberately limited, not a proof of optimality or noise correctness.
"""
import argparse
import functools
import json
import math
import random
import time


def tree_for(degree):
    depth = math.ceil(math.log2(degree))
    best_total = 10000
    best_tree = None
    for l in range(1, depth + 1):
        if 2**l - 1 > degree:
            break
        for m in range(1, depth + 1):
            if 2**(m-1) >= degree:
                break
            costs = [[0]*(depth+1) for _ in range(degree+1)]
            trees = [[None]*(depth+1) for _ in range(degree+1)]
            for odd in range(3, degree+1, 2):
                costs[odd][1] = 10000
            for level in range(2, depth+1):
                for odd in range(1, degree+1, 2):
                    if odd <= 2**l-1 and odd <= 2**(level-1):
                        continue
                    best = 10000
                    candidate_tree = None
                    for s in range(1, min(m, level)):
                        split = 2**s
                        if split >= odd:
                            break
                        cost = costs[odd-split][level-1]+costs[split-1][level]+1
                        if cost < best:
                            best = cost
                            candidate_tree = (split, trees[split-1][level],
                                              trees[odd-split][level-1])
                    costs[odd][level] = best
                    trees[odd][level] = candidate_tree
            total = costs[degree][depth]+2**(l-1)+m-2
            if total < best_total:
                best_total = total
                best_tree = trees[degree][depth], m, l
    return best_tree


TREES = {d: tree_for(d) for d in (15, 27)}


def prime_test(n):
    # Deterministic for unsigned 32-bit inputs.
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for a in (2, 3, 5, 7, 11):
        x = pow(a, d, n)
        if x in (1, n-1):
            continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True


@functools.lru_cache(None)
def ntt_primes(bits):
    stride = 2*65536
    return tuple(n for n in range(((1 << bits)-2)//stride*stride+1,
                                  1 << (bits-1), -stride) if prime_test(n))


def actual_chain(widths):
    used = {}
    chain = []
    for b in widths:
        i = used.get(b, 0)
        candidates = ntt_primes(b)
        if i >= len(candidates):
            raise ValueError(f'not enough distinct {b}-bit NTT primes')
        chain.append(candidates[i])
        used[b] = i+1
    return chain


class Planner:
    def __init__(self, bits, floor=40, coeff_floor=40):
        self.bits = tuple(bits) # top first: consumed index grows downward
        self.prefix = [0.0]
        for b in bits:
            self.prefix.append(self.prefix[-1]+b)
        self.floor = floor
        self.coeff_floor = coeff_floor

    def basis(self, degree, start, input_s, basis_floor):
        _, m, l = TREES[degree]
        basis = {1: (start, input_s)}
        events = []
        def make(d, a, b, diff):
            ka, sa = basis[a]
            kb, sb = basis[b]
            kw, pre = max(ka, kb), sa+sb
            if diff:
                kd, sd = basis[diff]
                kw = max(kw, kd)
                if pre-sd < self.coeff_floor-1e-8:
                    raise ValueError('basis difference scale too small')
            k, s = kw, pre
            while k < len(self.bits) and s-self.bits[k] >= basis_floor-1e-8:
                s -= self.bits[k]
                k += 1
            if s < self.floor-1e-8:
                raise ValueError('basis result below scale floor')
            basis[d] = (k, s)
            events.append(dict(degree=d, work=kw, output=k, pre=pre, scale=s))
        for stage in range(1, math.ceil(math.log2(degree+1))+1):
            if stage <= m-1:
                make(2**stage, 2**(stage-1), 2**(stage-1), 0)
            if stage <= l:
                for d in range(2**(stage-1)+1, 2**stage, 2):
                    make(d, 2**(stage-1), d-2**(stage-1), 2**stage-d)
        return basis, events

    def component(self, degree, start, input_s, output_s, basis_floor,
                  max_drop=15, trace=False):
        try:
            basis, basis_events = self.basis(degree, start, input_s, basis_floor)
        except ValueError:
            return None
        def solve(tree, deg, ko, so):
            if so < self.floor-1e-7 or ko > len(self.bits):
                return None
            # Conservative public capacity headroom, not a CKKS noise bound.
            if sum(self.bits[ko:])+180-so < 32:
                return None
            if tree is None:
                indexes = list(range(1, deg+1, 2))
                kw = max(basis[d][0] for d in indexes)
                if kw > ko:
                    return None
                pre = so+self.prefix[ko]-self.prefix[kw]
                enc = [pre-basis[d][1] for d in indexes]
                if min(enc) < self.coeff_floor-1e-7:
                    return None
                return dict(type='leaf', degree=deg, work=kw, output=ko,
                            scale=so, pre=pre, encoding_scales=enc)
            split, left, right = tree
            kb, sb = basis[split]
            for kw in range(kb, ko+1):
                pre = so+self.prefix[ko]-self.prefix[kw]
                sr = pre-sb
                if sr < self.floor-1e-7:
                    continue
                r = solve(right, deg-split, kw, sr)
                if r is None:
                    continue
                rem = solve(left, split-1, kw, pre)
                if rem is not None:
                    return dict(type='combine', split=split, work=kw,
                                output=ko, scale=so, pre=pre,
                                quotient=r, remainder=rem)
            return None
        for end in range(start, min(start+max_drop, len(self.bits))+1):
            result = solve(TREES[degree][0], degree, end, output_s)
            if result is not None:
                result = dict(degree=degree, start=start, end=end,
                              input_scale=input_s, output_scale=output_s,
                              basis_floor=basis_floor,
                              basis=basis_events if trace else None,
                              tree=result if trace else None)
                return result
        return None

    @functools.lru_cache(maxsize=8192)
    def best_component(self, degree, start, input_s, output_s, step=1,
                       trace=False):
        best = None
        for i in range(round(20/step)+1):
            bf = self.floor+i*step
            c = self.component(degree, start, input_s, output_s, bf, trace=trace)
            if c is not None and (best is None or c['end'] < best['end']):
                best = c
        return best

    def relu_fixed(self, input_s=40, stage_scales=(40, 40), trace=False):
        start = 0
        s = input_s
        stages = []
        for out in stage_scales:
            c = self.best_component(15, start, s, out, trace=trace)
            if c is None:
                return None
            stages.append(c)
            start, s = c['end'], out
        best = None
        # Third polynomial output is chosen jointly with final multiplication.
        for end in range(start+3, min(start+14, len(self.bits))):
            for drops in range(1, 5):
                if end+drops > len(self.bits):
                    continue
                out_s = 40-input_s+self.prefix[end+drops]-self.prefix[end]
                if out_s < self.floor-1e-7:
                    continue
                c = self.best_component(27, start, s, out_s, trace=trace)
                if c is None or c['end'] > end:
                    continue
                candidate = dict(total=end+drops, stages=stages+[c],
                                 tail_work=end, tail_drop=drops,
                                 stage_scales=[*stage_scales, out_s],
                                 dropped_bits=self.prefix[end+drops])
                if best is None or candidate['total'] < best['total']:
                    best = candidate
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['baseline', 'scales', 'fine', 'boundary', 'mixed', 'joint'], default='baseline')
    ap.add_argument('--bits', type=int, default=30)
    ap.add_argument('--floor', type=float, default=40)
    ap.add_argument('--seed-scales', default='40,40')
    ap.add_argument('--actual', action='store_true')
    ap.add_argument('--trials', type=int, default=100)
    args = ap.parse_args()
    def compact(c):
        if c is None:
            return None
        return {k: c[k] for k in ('total', 'stage_scales', 'dropped_bits', 'widths')} | {
            'drops': [s['end']-s['start'] for s in c['stages']]+[c['tail_drop']]}
    print('trees', json.dumps(TREES), flush=True)
    print('prime_supply', {b: len(ntt_primes(b)) for b in range(18, 33)}, flush=True)
    widths = [args.bits]*38
    @functools.lru_cache(maxsize=16)
    def get_planner(width_tuple):
        primes = actual_chain(width_tuple) if args.actual else None
        bits = [math.log2(q) for q in primes] if primes else width_tuple
        return Planner(bits, args.floor, args.floor), primes
    def evaluate(widths, scales=(40, 40), trace=False):
        planner, primes = get_planner(tuple(widths))
        result = planner.relu_fixed(stage_scales=scales, trace=trace)
        if result:
            result['widths'] = widths[:result['total']]
            result['primes'] = primes[:result['total']] if primes else None
        return result
    if args.mode == 'baseline':
        print(json.dumps(evaluate(widths, trace=True)), flush=True)
        return
    initial_scales = tuple(float(x) for x in args.seed_scales.split(','))
    best = evaluate(widths, initial_scales)
    print('initial', json.dumps(compact(best)), flush=True)
    if args.mode in ('scales', 'fine', 'boundary'):
        grid = list(range(40, 61, 2)) if args.mode == 'scales' else [40+i/10 for i in range(31)]
        if args.mode == 'boundary':
            grid = sorted({x+f for x in range(40, 57) for f in (0, .1, .5)})
        grid2 = list(range(40, 61)) if args.mode == 'boundary' else grid
        for s1 in grid:
            for s2 in grid2:
                c = evaluate(widths, (s1, s2))
                if c and (c['total'], c['dropped_bits']) < (best['total'], best['dropped_bits']):
                    best = c
                    print('improved', json.dumps(compact(best)), flush=True)
        print('final', json.dumps(compact(best)), flush=True)
    else:
        rng = random.Random(20260911)
        current = best
        cur_widths = widths[:]
        cur_scales = list(initial_scales)
        for i in range(args.trials):
            trial_widths = cur_widths[:]
            trial_scales = cur_scales[:]
            for _ in range(rng.choice([1, 1, 2, 3])):
                at = rng.randrange(0, min(29, len(widths)))
                trial_widths[at] = rng.randrange(23, args.bits+1)
            if args.mode == 'joint':
                at_scale = rng.randrange(2)
                trial_scales[at_scale] = rng.choice(
                    [40+i*.1 for i in range(30)] if at_scale == 0 else list(range(40, 56)))
            try:
                c = evaluate(trial_widths, trial_scales)
            except ValueError:
                continue
            if c and (c['total'], c['dropped_bits']) < (best['total'], best['dropped_bits']):
                best = c
                print('improved', i, json.dumps(compact(best)), flush=True)
            if c and c['total'] <= current['total']:
                current, cur_widths, cur_scales = c, trial_widths, trial_scales
            if (i+1) % 100 == 0:
                print('progress', i+1, json.dumps(compact(best)), flush=True)
        print('final', json.dumps(compact(best)), flush=True)


if __name__ == '__main__':
    main()
