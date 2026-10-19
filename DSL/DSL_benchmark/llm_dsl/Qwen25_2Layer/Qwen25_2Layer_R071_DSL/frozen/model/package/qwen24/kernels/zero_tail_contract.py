"""Conservative DSL-only provenance for a structurally zero padded tail.

This is a symbolic property, not a claim of zero CKKS approximation error.
External PackedVector inputs remain unmarked and retain the ordinary mask.
Only internal code with a structural zero-tail proof may call _proven_prefix.
"""
from dataclasses import dataclass
from qwen24.kernels import qwen_layers as ql

@dataclass(frozen=True)
class _ZeroTailVector(ql.PackedVector):
    """Internal certificate; never construct for unchecked external ciphertexts."""

def known_zero_tail(vector):
    return isinstance(vector,_ZeroTailVector)

def _proven_prefix(vector):
    return _ZeroTailVector(vector.expr,vector.size,vector.slots)

def install():
    if getattr(ql,'_zero_tail_contract_installed',False):return
    original_rms=ql.rms_norm
    original_mul=ql.mul

    def rms_norm(*args,**kwargs):
        # Original RMSNorm ends with multiplication by prefix-only weights.
        return _proven_prefix(original_rms(*args,**kwargs))

    def mul(x,y):
        result=original_mul(x,y)
        # Coordinatewise product is zero wherever either operand is zero.
        return _proven_prefix(result) if known_zero_tail(x) or known_zero_tail(y) else result

    ql.rms_norm=rms_norm
    ql.mul=mul
    ql._zero_tail_contract_installed=True
