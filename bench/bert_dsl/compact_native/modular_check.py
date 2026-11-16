#!/usr/bin/env python3
"""Small CPU proof-by-test of the CUDA kernel's unsigned Barrett/RNS MAC.

Uses the same floor(2^64/q), multiply-high, and two conditional reductions as
Poseidon's GPU helpers. Signed CKKS coefficients are canonical residues modulo q;
no signed reinterpretation or unsigned truncation of a negative value is valid.
This does not execute CUDA and is not a GPU correctness/performance claim.
"""
import json
import random


def reduce_u64(value, q):
    assert 0 <= value < (1 << 64)
    ratio = (1 << 64) // q
    quotient = (value * ratio) >> 64
    r = value - quotient * q
    if r >= q:
        r -= q
    if r >= q:
        r -= q
    assert 0 <= r < q
    return r


def main():
    primes = [204865537, 223215617, 245760001, 268042241,
              536608769, 1073479681, 2147352577, 4294967291]
    rng = random.Random(20261115)
    products = 0
    for q in primes:
        edges = [0, 1, q // 2, q - 2, q - 1]
        pairs = [(a, b) for a in edges for b in edges]
        pairs += [(rng.randrange(q), rng.randrange(q)) for _ in range(2048)]
        for a, b in pairs:
            assert reduce_u64(a * b, q) == (a * b) % q
            products += 1
        # Signed coefficients (e.g. negative encoded weights) are represented
        # by their positive RNS residues; modular multiplication preserves it.
        for _ in range(16):
            actual = expected = 0
            for _ in range(192):  # largest 12-input-tile,16-baby giant group
                signed_a = rng.randrange(-q + 1, q)
                signed_b = rng.randrange(-q + 1, q)
                product = reduce_u64((signed_a % q) * (signed_b % q), q)
                total = actual + product
                actual = total - q if total >= q else total
                expected = (expected + signed_a * signed_b) % q
                assert actual == expected
                products += 1
    print(json.dumps(dict(cpu_only=True, unsigned_barrett="ok",
                          signed_rns_mac="ok", primes=len(primes), products=products)))


if __name__ == "__main__":
    main()
