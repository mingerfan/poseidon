"""Explicit bounded tensor mathematical values; no DSL implementation or execution."""
POLICY="descriptors-math-tensors-v4"
FORMAT="poseidon-public-weight-descriptors-v4"
ROLE_FORMAT="poseidon-public-math-roles-v2"
ROLE_SCOPE="public mathematical tensor/scalar values and exact prebound roles; no DSL answer"
from public_math_context import RULES as SCALAR_RULES
RULES=SCALAR_RULES+"""
Public tensor mathematical-role contract v4:
Small tensors used as elementwise operands have checked tensor_operand records.
Element indices use C order within the declared logical shape. Values and exact
immutable scalar Plain bindings describe public mathematics, not a DSL program.
Preserve broadcasting and logical shape. Reconstructing public metadata does not
change ciphertext packing, permitted rotations, security, resource or execution
capability. Every nonzero leaf requires an exact listed prebound scalar Plain;
never assume a tensor is identity or infer values from a content hash.
The existing aggregate256 mathematical elements,49152role bytes and131072public
context bytes remain fixed. Other tensor/layout roles remain opaque as declared.
No private reference, test inputs/answers, weights substitution or DSL recipe.

"""
