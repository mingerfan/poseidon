# AV gap packing + masked reciprocal (A2): CPU candidate plan

**Not a GPU result.** Independent read-only derivation accepted the ideal indexing,
full encoder equivalence and static scale schedule. C++ module CPU implementation
is complete: CTest7/7, independent full-slot partial oracle, actual scalar callsite
counts, N1024 encrypted pair, N65536 all319mask encode/decode (max4.55e-16).
This does NOT establish production GPU correctness or full-model HE precision.
Parent Graph/native integration is not yet complete. The opt-in
`_Arithmetic.reciprocal(..., zero_head_gaps=True)` seed helper now passes four
CPU tests: live polynomial, public-gap proof, the constant-doubling fix, and real
prime-depth/count parity versus the folded baseline (L18→L8,no extra Boot). Frozen v6 and
ongoing full12-v6 continue using the original AV and unmodified c34 binary.

## New operator is NOT old full-slot AV

D=64,T=128,tile256. Form the original value diagonal
`D[t,h,j]=V[t+j,h,j]`, with ideal V gap j>=D zero. After real diagonal CPsum/R2:

- `Dpack=D+rotate(192,D)`; three even-baby rotations512/1024/1536.
- Fuse head-live public mask M into **rotated** cyclic-score L/R masks, including
  shift0. Never discard the original score's upper64 key lanes.
- For offsets1..64 the right live mask is identically zero; exactly64 right
  CP/rotations disappear. Offsets65..127 still use both branches.
- `Cpack=C_even+rotate(-64,C_odd)` after each real score CP/R4.
- One true CC/relin/R2 per offset pair;64 products still cover all128 keys.
- Original group/giant sum, then `out=P+rotate(64,P)`.

Let A/B split the complete sum by parity `(key_offset-feature_j)%2`. The four
64-blocks of P are[A0,B0,A1,B1]. Output live lanes equal complete AV, but gap is
`B[t,h,j]+A[next(t,h),j]`, next(0)=(t,1),next(1)=(t+1 modT,0).
This is a **defined full-slot partial-AV function**, not gap0. Its independent
oracle must compare all32768 slots against this function, plus separately compare
live16384 lanes against ordinary AV. Dirty/unknown V gaps are not admitted.

## Why the original consumer is unsafe

`merge_heads` has NO zero mask: it adds one context to its neighbor rotated−64.
For score=1,Vlive=1/gap0, raw new AV is128 in every lane; the old unmasked inverse
1/128 leaves context all1 and merge produces2, not1. A partial node must not
inherit `compact_av`'s zero-gap proof, nor be silently substituted at arbitrary
consumers/finals/dense/rotations.

## A2 fixes the existing context, without lower intermediate scale

Keep denominator d from **all128 scores**. Only the inverse for this AV/context
uses the already-folded first Newton step:

```
r1 = d * plain(-M/T²) + plain(2M/T)
r2 = r1 * (2 - d*r1)
r3 = r2 * (2 - d*r2)
context = partial_AV * r3
```

M²=M proves r_k_masked=M*r_k_original inductively, so live polynomial is unchanged,
gaps0, all three Newton updates remain. The correction `2-d*r` has gap2; the OUTER
r multiplication preserves zeros. Context is full-slot identical in ideal real
arithmetic to old full AV×old inverse. Existing merge/dense then receive identical
features. Keep all Softmax/GELU/LN/residual/dense and nativeBoot.

Current v6 folded seed already costs one CP→one trueR2→oneaddCP. Replace only its
two public coefficient vectors; do not add another mask CP. Do not rely on the old
full-scalar recognizer accepting a vector: explicitly build the folded seed.
Later minus-one multiplications still need scalar-unit optimization/Negate.

Static selective boundaries remain dL18→r1L16→r2L12→r3L8; existing inverse
alignment toL4 and context CC/relin/R2 toL2 remain. No additional depth/Boot is
implied; actual new plan counts and GPU effects still require validation.
Intermediate AV scales stay canonical~55bits, unlike alternative B~37bits.

## CPU-executed scalar callsite work per pair, NOT GPU latency

| Work | oldAV | A2 AV macro |
|---|---:|---:|
| CP |383|319|
| CC/relin |128/128|64/64|
| R2/R4 |129/128|65/128|
| Rescale calls / physical drops |257/770|193/642|
| Direct rotations / adds |298/381|296/319|
| Same-source hoist batches |19|19|

Extra direct steps−64,+64,+192 make55-step superset; real keys/fullP and mixed
prepare view refresh required. MaskBundle identity must include algorithm, since
same physical scales do not imply same masks. Per-ID shared bytes cannot be summed.

## Admission and evidence

- Separate node/callee/manifest algorithm with the defined full-slot function.
  Default oldAV and dirty-gap behavior unchanged; new mode explicit and restricted. Current module source archive is
  `compact-av-build/packed-input2/compact-av-packed-input2-source.tar.gz`, SHA256
  `42c5539ff25178852d20cb353946580534b10ddca5871f126578ce06f40e6cd9`.
- Prove V ideal gaps0 from public IR, NOT actual raw ciphertext values. Partial
  node proof=false; masked r1/r2/r3 and final context proof=true.
- Every reachable old-AV substitution path must reach its own proven masked
  context multiplication before lane-mixing/dense/final output. An export-time
  consumer walk can permit only identity Boot hints until a real multiplication
  by a separately proven gap0 operand; require every fanout path to be cleared,
  reject direct finals/unmasked shared branches/rotation or dense before clearing.
  This is a gap-leak guard, not a substitute for the independent full-model oracle.
  Standalone C++ diagnostic has a different explicit contract.
- Give the partial operator a distinct callee and nonbaseline manifestv2. Merely
  adding a field to oldv1 is insufficient: old consumers may ignore it. Parse all
  mode/key requirements before optional async allocator/CUDA initialization.
- Test offset0/64/65/127, j0/63, final head/token wrap, signed distinct heads,
  upper64 nonzero score keys, dirty V, all masks/fields/mode/key mismatches.
- Constant doubling counterexample must fail without masked seed and agree after
  it; randomized tiny complete model must agree after both rewrites, with unchanged
  endpoints/depth/Boot, not merely matching rewrite counts.
- Real encoding/rotation/rescale noise gaps are NOT zero. Masked seed coefficients
  have amplitude2^-14 with PTscale~55.43 (about41.43-bit amplitude budget), so A2
  does not prove unchanged CKKS error. Context multiplies inverse gap error by
  the partial AV's gap value. Full-model accuracy remains unpassed.

Alternative B's final real mask can preserve all-slot AV without masked inverse,
but requires common scale~37bits (best balanced mask budget~36.9bits), delayed R2
and extra CP. It is not the current implementation target; fewer calls do not
establish a faster or more accurate path.
