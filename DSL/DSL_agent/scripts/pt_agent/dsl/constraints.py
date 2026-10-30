"""Explanations attached to typecheck constraints and rendered in r8-v8."""

from dataclasses import dataclass

from .ops_spec import OPS


@dataclass(frozen=True)
class Constraint:
    code: str
    ops: tuple
    explanation: str


CONSTRAINTS = (
    Constraint('E_MANAGEMENT', ('*',), 'manage="dsl" requires refresh="compiler" and placement="depth-dp".'),
    Constraint('E_APPROX', ('act', 'maxpool'), 'Approximation parameters must match their method: remez uses positive deg; newton uses positive deg0 and nonnegative iters on a positive inv/rsqrt domain; composite uses relu, positive odd degrees, 0<eps<1 and a symmetric domain. Supply only the parameters of the selected method.'),
    Constraint('E_METHOD_UNSUPPORTED', ('act', 'maxpool', 'layernorm', 'rmsnorm', 'softmax'), 'Each approximation function must use one of the methods in the kernel-generated function/method table, before any fitting or numerical execution.'),
    Constraint('E_KIND', ('*',), 'Plain parameters are allowed only as the second add/sub/mul operand. Mask parameters must be packed with layout="flat"; use the packed Mask for softmax mask= or the selection Mask of select. softmax mask= and the second select operand require the Mask type.'),
    Constraint('E_LAYOUT_REPACK', ('pack',), 'pack accepts a raw encrypted input parameter exactly once.'),
    Constraint('E_LAYOUT_UNPACKED', ('*',), 'Pack every raw encrypted input before any computational operation.'),
    Constraint('E_LAYOUT_MISMATCH', ('add', 'sub', 'mul', 'concat', 'matmul'), 'Binary ciphertext operands require the same layout label; attention matmul explicitly converts token/score layouts.'),
    Constraint('E_SHAPE', ('*',), 'All tensor shapes and axes must obey the per-operator shape rules below.'),
    Constraint('E_WEIGHT_MISSING', ('*',), 'Every public W reference must exist in the supplied model weights.'),
    Constraint('E_WEIGHT_SHAPE', ('*',), 'Public weight shapes must match the consuming operation, including all optional biases and folded batch-normalization vectors.'),
    Constraint('E_WEIGHT_UNUSED', ('*',), 'Use every weight consumed by a graph node in its original role, including convolution/linear biases and normalization parameters; keep every model operation.'),
    Constraint('W_UNUSED', ('*',), 'Every intermediate value and input should contribute to the returned network result.'),
    Constraint('W_REFRESH_COMPILER', ('refresh',), 'refresh="compiler" lowers explicit refresh to identity; let the compiler manage ciphertext state.'),
)

SHAPE_RULES = {
    'conv2d': 'Input [1,C,H,W], rank-four weights [C_out,C/groups,Kh,Kw], positive spatial output; groups is 1 or C, and depthwise C_out=C. Bias and BN vectors have C_out entries.',
    'linear': 'Weight [O,I] consumes the last input dimension I; optional bias has O entries.',
    'matmul': 'Attention Q/K are rank-three tokens with equal head widths and divisible heads/kv_heads; P/V are rank-four scores and rank-three values with matching keys. Generic matmul needs equal batch shapes and matching contracted dimensions.',
    'avgpool': 'Input has rank four; the kernel, stride and padding produce positive spatial dimensions.',
    'maxpool': 'Input has rank four; the kernel, stride and padding produce positive spatial dimensions.',
    'mean': 'Axes are distinct and within the tensor rank; a scalar reduction is outside PT-DSL v1.',
    'sum': 'Axes are distinct and within the tensor rank; a scalar reduction is outside PT-DSL v1.',
    'add': 'Ciphertext operands have equal logical shape and layout. A public W operand broadcasts to the input shape; in particular a public tensor of exactly the same logical shape is allowed.',
    'sub': 'Ciphertext operands have equal logical shape; a public W operand broadcasts to the input shape.',
    'mul': 'Ciphertext operands have equal logical shape; a public W operand broadcasts to the input shape.',
    'bias': 'b is a vector matching the selected axis; token bias uses the last dimension. Use add(x,W[...]) for a full logical tensor.',
    'batchnorm': 'Input rank is at least two; weight, bias, running_mean and running_var are channel vectors.',
    'layernorm': 'Normalize the last dimension; gamma and beta, when present, are vectors of that width.',
    'rmsnorm': 'Normalize the last dimension; gamma is a vector of that width.',
    'softmax': 'axis=-1; a key Mask has shape [1,Lk]. shift is finite, inv_scale is positive, and every used approximation role has explicit method/domain and matching parameters.',
    'shuffle': 'Input [1,L,D]; index/sign have D entries, every index is in [0,D), and each sign is +/-1.',
    'select': 'Input [1,L,D] and selection Mask[1,L] produce [1,1,D].',
    'flatten': 'start_dim is an axis within the input rank.',
    'reshape': 'Preserve the element count; at most one inferred -1 dimension, all other dimensions positive.',
    'permute': 'dims is a permutation of every input axis, with valid axis indices.',
    'concat': 'Equal ranks and equal dimensions outside the concatenation axis.',
    'slice': 'The axis is valid and the positive-step slice produces a nonempty dimension.',
    'split': 'The axis is valid and positive split sizes sum to that input dimension.',
    'pad': 'At most one nonnegative padding pair per input dimension, ordered from the last dimension backwards.',
}


def for_op(name):
    result = [c.explanation for c in CONSTRAINTS if '*' in c.ops or name in c.ops]
    if name in SHAPE_RULES:
        result.append(SHAPE_RULES[name])
    # Attribute schemas are the same objects used by the parser/typechecker.
    for key, field in OPS[name].attrs.items():
        detail = 'required' if field.required else 'default '+repr(field.default)
        if field.choices is not None:
            detail += '; choices '+repr(field.choices)
        result.append(f'{key}: {field.type}, {detail}.')
    return result


def explanation(code, name):
    texts = [c.explanation for c in CONSTRAINTS if c.code == code and ('*' in c.ops or name in c.ops)]
    if code in ('E_SHAPE', 'E_WEIGHT_SHAPE') and name in SHAPE_RULES:
        texts.append(SHAPE_RULES[name])
    return ' '.join(texts)
