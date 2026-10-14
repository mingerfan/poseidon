# MHA ordered-graph composition references

This independent developer tool defines six small static native MHA compositions.
It leaves the old math planner, source draft, production Agent and backend untouched.
`cases.py` is the case definition; `reference_math.py` evaluates original public
parameters using explicit scalar loops and sums. The old independent reference,
native CPU float64 Torch and graph NumPy/Torch provide separate cross-checks.

| Context | Scope |
|---|---|
| self_post_linear_residual | E4/H2 self-MHA → public Linear → input residual; named weights |
| linear_into_separate_cross_none | public Linear 3→4 → cross-MHA, K3/V2; named projected output and static None |
| boolean_padding_residual | MHA Boolean True means blocked; attention plus key-padding masks → residual |
| additive_heads_post_linear | N2/H2 cross-MHA, per-head finite additive plus padding masks → public Linear 4→2 |
| unbatched_cross_none | rank2 cross-MHA and need_weights=False, None retained without tensor slot |
| sequence_first_bias_zero | sequence-first, bias_k/v plus add_zero_attn → Linear/residual and weights |

All six use the original explicit exp Taylor degree6 and reciprocal Goldschmidt
iterations8 profiles. Three **additional configuration variants**, separately named
`__declared_shallow_v1`, use predeclared degree1/iterations1 for later feasibility
checks. They are distinct approximation targets and do not replace old failures or
establish exact Softmax application accuracy.

Standard-library controls:

```sh
cd /home/lhohy/Code/Poseidon/scripts/baseline/benchmarks/tools/staged_mha_ordered_graph_v1/reference_tests
python3 -B -m unittest -v test_controls
```

The parent task alone launches `audit.py --output <fresh directory>` in the already
coordinated Python3.10.14/NumPy1.25.2/Torch2.0.1 CPU environment. The tool never
enters Nix, captures source through the production worker, compiles, invokes a
provider or executes ciphertexts. The body stops at150 seconds. It writes graph
and aggregate evidence; no input or output arrays are printed.

## Current limits and pending execution

The fixed native MHA planner admits rank2/3, static batch/query/key lengths,
E divisible by H, same or separate Q/K/V dimensions, finite float64 public state,
Boolean blocked masks and finite additive masks. At least one key must remain
allowed for every query/head. The v16 source draft rejects negative-infinity
state, training, hooks, subclasses, private control flow and arbitrary calls.
The original source draft is MHA-only; this test's graph composition is not source
intake activation. Production registry contains the matrix, exp, reciprocal,
reduction, broadcast and shape primitives; native MHA is not itself registered.

The original profile includes eight sequential reciprocal squarings, followed by
attention/value multiplication. Graph shape validity and independent plaintext
agreement do not establish current Q13 level/scale capacity or retained memory.
Real tracing/compilation, layout/resource gate, Poseidon numerical execution,
program-package reload and real Agent generation remain separate unrun stages.
`None` requires an explicitly versioned source/output contract; it is not an
extra tensor or ciphertext. Finite additive masks are genuine offsets; Boolean
blocked entries are explicitly excluded from numerator/denominator, rather than
replacing them with a finite large negative sentinel.
