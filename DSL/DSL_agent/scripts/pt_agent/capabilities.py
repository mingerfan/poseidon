"""The trusted lowering's public operator and pattern registry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    op: str
    signature: str
    semantics: str
    constraints: tuple[str, ...]
    cost: str
    in_prompt: bool = True


@dataclass(frozen=True)
class Pattern:
    name: str
    graph_ops: tuple[str, ...]
    description: str
    dsl_form: str


CAPABILITIES = (
    Capability(
        'pack',
        'pack(x, layout="mux", gap=1) / pack(x, layout="flat")',
        'Encrypts a network input into one ciphertext. Use layout="mux" for an image input of shape [1,C,H,W] and layout="flat" for a vector input of shape [1,F].',
        ('Every input is packed exactly once, before any other use.',),
        'No homomorphic operations.',
    ),
    Capability(
        'conv2d',
        'conv2d(x, w=W[...], b=W[...] (optional), bn="<prefix>" (optional), stride=S, pad=P, algo="mpcb")',
        '2-D convolution on a mux-packed tensor; with bn="<prefix>" the following batch normalization (<prefix>.weight/.bias/.running_mean/.running_var) is folded into it.',
        ('Odd square kernel; pad equals (kernel-1)//2; stride 1 or 2; dilation 1; groups 1. Stand-alone batch normalization is not supported, so a batch normalization must be folded into the convolution that feeds it.',),
        '2 multiplicative levels; rotations grow with kernel size and channel count.',
    ),
    Capability(
        'act',
        'act(x, fn=F, method="cheb", deg=D, domain=(lo, hi)) | act(x, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-B, B))',
        'Element-wise activation F (relu, silu, gelu, sigmoid, tanh or exp). method="cheb" uses a Chebyshev polynomial of degree D fitted on [lo, hi]. method="minimax_comp" (relu only) computes x * (0.5 + s(x / B)), where s is a composition of three minimax polynomials (degrees 15, 15, 27) approximating sign(u) / 2; it is much more accurate than cheb on wide domains.',
        ('For cheb, 1 <= D <= 255. For minimax_comp, fn must be relu, degrees must be (15, 15, 27) and the domain must be symmetric (lo = -hi). Every input value must lie inside the domain; outside it the polynomial grows without bound and the whole network diverges. Method, degree and domain may differ for every act.',),
        'cheb: ceil(log2(D+1)) + 2 multiplicative levels; higher D is more accurate and more expensive; a wider domain is safer and less accurate at the same D. minimax_comp: 17 multiplicative levels (deeper than cheb with D = 255, which needs 10, so it usually costs more bootstraps) but fewer multiplications.',
    ),
    Capability(
        'add',
        'add(a, b)',
        'Element-wise sum.',
        ('Same shape and same layout.',),
        'No multiplicative level.',
    ),
    Capability(
        'slice',
        'slice(x, axis=A, start=S, end=E, step=K)',
        'Strided slice along one axis.',
        ('Only allowed inside the downsample_pad pattern.',),
        'Part of the pattern.',
    ),
    Capability(
        'pad',
        'pad(x, pads=(...))',
        'Zero padding. pads follows torch.nn.functional.pad: pairs are listed from the last dimension backwards, so (0, 0, 0, 0, p, p) keeps W and H and pads the channel axis by p zeros on each side (C -> C + 2p).',
        ('Only allowed inside the downsample_pad pattern.',),
        'Part of the pattern.',
    ),
    Capability(
        'mean',
        'mean(x, axis=(2, 3))',
        'Average over the spatial axes of a mux-packed tensor; the result is a flat vector.',
        ('Must be followed by relayout(..., layout="flat").',),
        '1 multiplicative level.',
    ),
    Capability(
        'relayout',
        'relayout(x, layout="flat")',
        'Marks a flat vector produced by mean.',
        ('Only directly after mean.',),
        'No homomorphic operations.',
    ),
    Capability(
        'linear',
        'linear(x, w=W[...], b=W[...] (optional), algo="bsgs" or "diag")',
        'Matrix-vector product on a flat vector.',
        ('Input is flat (a packed vector input, the result of relayout, or the result of another flat operation).',),
        '2 multiplicative levels; "bsgs" uses fewer rotations than "diag".',
    ),
    Capability(
        'refresh',
        'refresh(x)',
        'Explicit bootstrapping hint.',
        ('With refresh="compiler" it is ignored.',),
        '—', in_prompt=False
    ),
)

PATTERNS = (
    Pattern(
        'conv_bn', ('conv2d', 'batchnorm'),
        'A convolution whose only consumer is a batch normalization.',
        'one statement: conv2d(x, w=W["<conv weight>"], bn="<bn prefix>", stride=S, pad=P, algo="mpcb")',
    ),
    Pattern(
        'downsample_pad', ('slice', 'slice', 'pad'),
        'Spatial subsampling by 2 followed by channel zero padding (a residual shortcut).',
        'three consecutive statements: s = slice(x, axis=2, start=0, end=None, step=2); s = slice(s, axis=3, start=0, end=None, step=2); s = pad(s, pads=<pads copied from the node>)',
    ),
    Pattern(
        'mean_head', ('mean', 'linear'),
        'Global spatial average feeding a linear layer.',
        'z = mean(x, axis=(2, 3)); z = relayout(z, layout="flat"); y = linear(z, ...)',
    ),
)


def render_capabilities() -> str:
    lines = ["Operators supported by the lowering (generated from its capability registry; anything else is rejected):"]
    for capability in CAPABILITIES:
        if capability.in_prompt:
            lines.extend((f"- {capability.signature}",
                          f"  {capability.semantics} Constraints: {' '.join(capability.constraints)} Cost: {capability.cost}"))
    lines.append('Lowering patterns. Groups of input nodes carrying "pattern": "<name>" must be written in this form:')
    lines.extend(f"- {pattern.name}: {pattern.description} Write: {pattern.dsl_form}" for pattern in PATTERNS)
    return "\n".join(lines)
