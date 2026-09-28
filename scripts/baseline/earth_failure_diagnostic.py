"""Bounded, advisory interpretation of fixed Dacapo compiler diagnostics.

This reports Earth consumed-level arithmetic, NOT SEAL modulus capacity.
It never accepts a candidate, changes its program, or proves infeasibility.
"""
import hashlib
import json
import re
from compiler_configuration import PROFILE_SHA256

_OPERATION=re.compile(
    r'"earth[.]mul"\([^\n]*?\) : \(tensor<(\d{1,9})x!earth[.](ci|pl)<(\d{1,9}) \* (\d{1,9})>>, '
    r'tensor<(\d{1,9})x!earth[.](ci|pl)<(\d{1,9}) \* (\d{1,9})>>\) ->')
def diagnose(log,profile_bytes):
    if not isinstance(log,str) or len(log)>4000:return None
    if type(profile_bytes) is not bytes or hashlib.sha256(profile_bytes).hexdigest()!=PROFILE_SHA256:
        return None
    if "error: 'earth.mul' op failed to infer returned types" not in log:return None
    matches=list(_OPERATION.finditer(log))
    if len(matches)!=1:return None
    profile=json.loads(profile_bytes)
    m=matches[0].groups()
    operands=[dict(tensor_length=int(m[i]),kind=m[i+1],scale=int(m[i+2]),consumed_level=int(m[i+3]))
              for i in (0,4)]
    left,right=operands
    factor=profile["rescalingFactor"];upper=profile["bootstrapLevelUpperBound"]
    lhs=left["consumed_level"]*factor+left["scale"];rhs=upper*factor
    failed=[]
    if left["consumed_level"]!=right["consumed_level"]:failed.append("operand_level_alignment")
    if left["tensor_length"]!=right["tensor_length"]:failed.append("operand_tensor_shape")
    if lhs>rhs:failed.append("compiler_accumulated_scale_budget")
    if not failed:return None
    return dict(format="fixed-earth-mul-diagnosis-v1",code="earth_mul_constraints_rejected",
                evidence_kind="interpretation_of_untrusted_compiler_log",
                operands=operands,failed_conditions=failed,
                compiler_profile_sha256=PROFILE_SHA256,
                compiler_accumulated_scale=lhs,compiler_budget=rhs,
                compiler_rescaling_factor=factor,compiler_upper_consumed_level=upper,
                level_semantics="Earth consumed levels; not HEVM remaining moduli or SEAL chain_index",
                proves_all_equivalent_programs_impossible=False,
                seal_capacity_checked=False)
