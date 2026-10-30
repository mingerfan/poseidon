"""The trusted lowering's public operator and pattern registry."""

from dataclasses import dataclass, replace


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

# r8-v4 keeps the exact original registry objects and text.
V4_CAPABILITIES = CAPABILITIES
CAPABILITIES = tuple(
    replace(entry,
            signature=entry.signature.replace('stride=S', 'groups=G, stride=S'),
            constraints=('Odd square kernel; pad equals (kernel-1)//2; stride 1 or 2; dilation 1; groups is 1 or C_in. Depthwise requires groups=C_in and C_out=C_in. Batch normalization may be folded or written separately.',))
    if entry.op == 'conv2d' else
    replace(entry, semantics='Packs an image or flat vector into one or more ciphertext channel/length blocks.')
    if entry.op == 'pack' else
    replace(entry, constraints=('Input is flat or the output of flatten on a mux tensor. Flatten before a linear layer reorders its public weight columns to the actual mux slots, with replica and unused columns zero.',))
    if entry.op == 'linear' else entry
    for entry in V4_CAPABILITIES
) + (
    Capability('avgpool', 'avgpool(x, k=K, stride=S, pad=P)',
               'Channel-wise average pooling, with a fixed divisor K squared (count_include_pad=True).',
               ('Square kernel; symmetric zero padding; channel blocks are independent.',),
               'Depthwise spatial products and output projection.'),
    Capability('maxpool', 'maxpool(x, k=K, stride=S, pad=P, method="cheb", deg=D, domain=(lo,hi)) | maxpool(x, k=K, stride=S, pad=P, method="minimax_comp", degrees=(15,15,27), domain=(-B,B))',
               'Approximate max pooling: merge horizontal neighbors then vertical neighbors using max(a,b)=b+relu(a-b). Padding is excluded by a valid-neighbor mask. Its approximation parameters are tunable independently of act.',
               ('Every valid internal difference must lie inside domain. Chebyshev and minimax parameters follow act with fn="relu". Kernel 2 uses 2 ReLU passes and kernel 3 uses 4; each mask multiply adds one level.',),
               '2*(K-1) polynomial ReLU passes, masks and output projection.'),
    Capability('concat', 'concat(a,b,...,axis=1)',
               'Concatenates the existing ciphertext block lists in channel order, with no movement.',
               ('Mux inputs with the same spatial shape and gap; unequal valid block channel counts are allowed.',),
               'No homomorphic operations.'),
    Capability('flatten', 'flatten(x,start_dim=1)',
               'Flattens CHW logically before linear. The spatial mux slots are retained and the linear weights are reordered.',
               ('Mux input; the next linear layer accepts C*H*W input columns. H=W=1 uses the existing flat relayout.',),
               'No movement for H*W>1; weight reorder is public.'),
    Capability('batchnorm', 'batchnorm(x,bn="<prefix>",eps=1e-5)',
               'Inference batch normalization with public weight, bias, running_mean and running_var.',
               ('First fold into a preceding conv or linear with this BN as its only consumer. Otherwise fold into the sole following linear or a following convolution with padding=0. Preserve activation order. Remaining BN uses a channel-wise affine transform.',),
               'Folded: no extra level; standalone: 1 level.'),
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


V6_CAPABILITIES = tuple(
    replace(entry,
        signature=entry.signature+' | act(x,fn=F,method="remez",deg=D,domain=(lo,hi)) | act(x,fn="relu",method="composite",degrees=(d1,d2,...),eps=E,domain=(-B,B)) | act(x,fn=F,method="newton",deg0=D,iters=I,domain=(lo,hi))',
        semantics=entry.semantics+' Remez fits the declared function directly in a Chebyshev basis; at equal degree its maximum error is no larger than cheb, up to the documented float64 floor. Composite fits odd Chebyshev sign stages. Newton uses a Remez initializer and the fixed inv or rsqrt iteration.',
        constraints=entry.constraints+('Remez supports relu, silu, gelu, sigmoid, tanh, exp, inv, rsqrt, hardswish and square, with positive deg. Composite is for relu only: all degrees positive and odd, 0<eps<1, and a symmetric domain. Newton is for inv/rsqrt only: deg0>0, iters>=0, 0<lo<hi. Nonconvergent explicit fits report an error; choose another setting or method.',),
        cost=entry.cost+' Remez has the same evaluation depth as equal-degree cheb. Newton adds 2 levels per inv iteration or 3 per rsqrt iteration; x_half is computed in parallel with the initializer. Composite depth and operations are predicted from its emitted stages.')
    if entry.op=='act' else
    replace(entry,signature=entry.signature+' | maxpool(x,k=K,stride=S,pad=P,method="remez",deg=D,domain=(lo,hi)) | maxpool(x,k=K,stride=S,pad=P,method="composite",degrees=(d1,d2,...),eps=E,domain=(-B,B))',
            constraints=entry.constraints+('Remez and composite use the same relu fitting parameters as act. Newton is unavailable for maxpool.',))
    if entry.op=='maxpool' else entry
    for entry in CAPABILITIES)


V5_CAPABILITIES = CAPABILITIES
V7_CAPABILITIES = tuple(
    replace(entry,signature=entry.signature+' / pack(x,layout="token") / pack(x,layout="score")',
            semantics=entry.semantics+' TokenLayout uses one ciphertext per token, splitting D>slots/2 into consecutive blocks; ScoreLayout uses one ciphertext per query with head/key intervals.')
    if entry.op=='pack' else
    replace(entry,semantics=entry.semantics+' Token linear acts on every token. A single input block uses the tall route at O/nextpow2(I)>=4, sharing input rotations across output blocks of slots/2.')
    if entry.op=='linear' else entry for entry in V6_CAPABILITIES
) + (
    Capability('sub','sub(x,y) / sub(x,W[...])','Elementwise subtraction.',('Matching token/score layouts or a public broadcast constant.',),'No level.'),
    Capability('mul','mul(x,y) / mul(x,W[...])','Elementwise multiplication; position-dependent public constants are allowed.',('Matching layouts; public constants broadcast to the full logical tensor.',),'One multiplication level.'),
    Capability('scale','scale(x,c=C)','Public scalar multiplication.',('Finite scalar.',),'One plaintext multiplication.'),
    Capability('bias','bias(x,b=W[...])','Public last-axis affine offset.',('Weight broadcasts to the input tensor.',),'No multiplication.'),
    Capability('layernorm','layernorm(x,w=W[...],b=W[...],eps=E,rsqrt_method=M,rsqrt_domain=(lo,hi),rsqrt_deg0=D,rsqrt_iters=I)',
        'Last-token-axis mean, centered variance plus eps, rsqrt, then gamma/beta.',
        ('Method/domain are explicit. U5 cheb/remez use rsqrt_deg; newton uses rsqrt_deg0/rsqrt_iters. Positive variance domain. Optional rsqrt_fit="absolute"|"relative" selects remez fitting or the newton initializer; default absolute.',),
        'Reductions, squares and an independent rsqrt approximation point.'),
    Capability('rmsnorm','rmsnorm(x,w=W[...],eps=E,rsqrt_method=M,rsqrt_domain=(lo,hi),rsqrt_deg0=D,rsqrt_iters=I)',
        'x*rsqrt(mean(x*x)+eps)*gamma; float64 arithmetic.',('Explicit method and positive domain; the same rsqrt knobs and rsqrt_fit="absolute"|"relative" as layernorm.',),'Independent rsqrt approximation point.'),
    Capability('softmax','softmax(x,axis=-1,keep="all"|"causal",mask=am|None,shift=C,inv_scale=S,steps=K,exp_method=M,exp_domain=(lo,hi),exp_deg=D,inv_method=M,inv_domain=(lo,hi),inv_deg0=D,inv_iters=I,step_inv_method=M,step_inv_domain=(lo,hi),step_inv_deg=D,final_inv_method=M,final_inv_domain=(lo,hi),final_inv_deg=D,inv_budget=R)',
        'Initially normalize(exp((scores-shift)*2**(-steps))*mask), using inv(sum/inv_scale)/inv_scale. Then repeat steps times: q=p*p; p=q*step_inv(sum(q)). This is the same softmax function. Mask[1,Lk] contains encrypted 0/1. No encrypted max subtraction.',
        ('Only keep_static[i,j]*m[j]. Public shift defaults to 0; positive inv_scale defaults to 1; steps is an integer in 0..8. steps>0 requires explicit shared step_inv_method/domain and U5 step_inv_deg or step_inv_deg0/iters. Optional final_inv_* replaces only the last squaring inverse, observed as ::final_inv. Omitting it preserves the shared inverse at every step. Optional inv_budget is 0.5, 0.25 or 0.1 and requires steps>0 and final_inv; it has no arithmetic effect. Optional exp_fit, inv_fit, step_inv_fit and final_inv_fit are "absolute"|"relative" for remez or a newton initializer; default absolute. A row scalar c in [1-rho,1+rho] from an intermediate inverse cancels in the next exact normalization; only the final inverse error remains. From exp calibration (lo,hi), use shift=hi, (a,b)=((1.5*lo-hi)/2**K,(1.5*hi-hi)/2**K), exp_domain=(a,b), inv_scale=n*exp(b), inv_domain=(exp(a)/inv_scale,1), where n is Lk. With a final inverse, step_inv_domain=final_inv_domain=((1-rho)**2/(1.5*n),1.5*(1+rho)**2). Initial and intermediate inverse relative error targets are rho, final inverse target 1e-3; K=0 initial inverse target 1e-3. Exp target is 1e-3/2**K. Use the existing degree ladder <=63 and Newton iterations <=5. Compare K=0..8 by predicted latency. Domains use calibration only; heldout judges candidates without changing domains.',),
        'Exp, initial inv and each step_inv[i] are approximation points; repeated steps share one fitted inverse. Token/head reductions, masks and each squaring add depth.'),
    Capability('matmul','matmul(a,b,transpose_b=T,heads=H,kv_heads=K,scale=C|None)',
        'True computes Q*K.T into ScoreLayout; false computes P*V into TokenLayout. kvmap(h)=floor(h/(H/K)).',
        ('Head split/merge, K transpose and GQA expand are folded by the frontend. H is divisible by K. Fold a public scale into Q projection when available.',),
        'Quadratic in sequence length; dot products and broadcasting require rotations.'),
    Capability('shuffle','shuffle(x,index=(...),sign=(...))','Static last-axis signed gather, including rotate_half.',
        ('Index length D; signs +/-1; no token movement.',),'One rotated/masked term per distinct shift.'),
    Capability('select','select(h,sel)','Sum sel[t]*h[t] across tokens before a single LM head.',
        ('sel: Mask[1,L] is the client encrypted one-hot last real position.',),'Encrypted mask broadcasts, products and sum.'),
)


def render_capabilities(version="r8-v4") -> str:
    if version in ('r8-v9','r8-v10'):
        version = 'r8-v8'
    if version not in ("r8-v4", "r8-v5", "r8-v6", 'r8-v7', 'r8-v8'):
        raise ValueError("unknown prompt version")
    capabilities = {'r8-v4':V4_CAPABILITIES,'r8-v5':V5_CAPABILITIES,'r8-v6':V6_CAPABILITIES,
                    'r8-v7':V7_CAPABILITIES,'r8-v8':V7_CAPABILITIES}[version]
    if version == 'r8-v8':
        capabilities = tuple(
            replace(entry, semantics='Element-wise sum. The second operand can be a ciphertext with the same layout or a public W[...] tensor with the same logical shape as the first operand. Use bias only for a last-dimension vector.')
            if entry.op == 'add' else
            replace(entry, signature='bias(x,b=W[...],axis=-1)',
                    semantics='Add a public vector along the last logical dimension; specify axis=-1.',
                    constraints=('The weight is a vector with exactly the last-dimension width. A full logical tensor uses add(x,W[...]).',))
            if entry.op == 'bias' else
            replace(entry, constraints=('Method and domain are explicit. Use rsqrt_deg for a supported polynomial method or rsqrt_deg0/rsqrt_iters for newton; the variance domain is positive. rsqrt_fit selects absolute or relative fitting.',))
            if entry.op in ('layernorm', 'rmsnorm') else entry for entry in capabilities)
    lines = ["Operators supported by the lowering (generated from its capability registry; anything else is rejected):"]
    for capability in capabilities:
        if capability.in_prompt:
            lines.extend((f"- {capability.signature}",
                          f"  {capability.semantics} Constraints: {' '.join(capability.constraints)} Cost: {capability.cost}"))
            if version == 'r8-v8':
                from .dsl.constraints import for_op
                lines.extend('  Check: '+text for text in for_op(capability.op))
    if version == 'r8-v8':
        from .dsl.interface import INPUT_KINDS, SIGNATURE_RULE
        from .dsl.methods import method_table
        from .dsl.constraints import for_op
        from .dsl.ops_spec import OPS
        lines.extend(('Input signature: '+SIGNATURE_RULE,
                      'Mask inputs can only use pack(..., layout="flat").',
                      'Input kinds (generated from the interface registry):'))
        for kind, spec in INPUT_KINDS.items():
            lines.append(f"- {kind}: {spec['dsl_type']}; pack layouts: {', '.join(spec['pack_layouts'])}")
        lines.extend(('Approximation function/method support (generated from the kernels):', method_table()))
        represented = {entry.op for entry in capabilities if entry.in_prompt}
        for name in sorted(OPS.keys()-represented):
            lines.append('Static checks for '+name+' (use only supported lowering patterns):')
            lines.extend('  Check: '+text for text in for_op(name))
    lines.append('Lowering patterns. Groups of input nodes carrying "pattern": "<name>" must be written in this form:')
    lines.extend(f"- {pattern.name}: {pattern.description} Write: {pattern.dsl_form}" for pattern in PATTERNS)
    return "\n".join(lines)
