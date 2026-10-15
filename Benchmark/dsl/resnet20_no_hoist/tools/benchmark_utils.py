"""CPU-only decoded SIMD semantics for HEVM; no CKKS noise approximation."""
import numpy as np
from prepare_assets import digest

def operation(op,left,right=None):
    if op==1:
        signed=right if right<32768 else right-65536
        return np.roll(left,-signed)
    if op==2:return -left
    if op in (3,4,10):return left
    if op in (6,7):return left+right
    if op in (8,9):return left*right
    raise ValueError('unsupported opcode '+str(op))
