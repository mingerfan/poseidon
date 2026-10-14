# Mixed packed / replicated-cell rule draft

This independent benchmark tool is a deterministic trusted rule baseline. It is
not activated production code and does not rewrite a Provider response. The
original model, request, public registry, weights, layout, compiler parameters,
source/reference and physical output selectors remain unchanged.

`mixed_packed_scalar_lowering.lower(request, source_kind="deterministic_rule_baseline")`
returns `(hecate_source, record)`. Other source authorities are rejected.

Inputs start as canonical C-order `Chunks`. `PackedStorage` and immutable
`BoundCell` references describe logical coordinates. Reshape, flatten, transpose,
permute, rotate, slice, split, concat and stack only rearrange these references.
Canonical Linear inputs delegate to the existing `Chunks.linear` with the
reconstructed original public packing and its exact diagonal/other registered
bindings. Noncanonical Linear inputs are lazily materialized and packed once.

Same-shape canonical pointwise arithmetic uses original chunk coordinates and
public elementwise bindings. Cross-shape cipher arithmetic and reductions consume
replicated cells. Cells stay cells through following polynomial/scalar operations.
The cell cache is scoped to the emitter and keyed by immutable SSA producer name,
slot and period. A bound full-period one-hot slot mask is followed by the admitted
period-doubling rotations/additions; missing slot masks use the existing bound
mask0 and rotated source. No invalid tail slot becomes a logical cell. Outputs
reuse canonical chunks or pack cells into the original exact offset/elements
coverage once. None is handled by the original source/output contract and has no
cipher port; this draft does not reinterpret output trees.

Exact64 scalar fallback retains the previously staged public-origin authority:
original public tensor plus full C-order coordinate, or the existing explicit
mean/pool/BatchNorm-derived provenance. Numeric matches in pv vectors do not grant
scalar authority. Packed Linear does not enumerate scalar weights; the authority
ledger is built only when a missing literal is actually requested. Such a fallback
may still traverse a large original public tensor, so preparation cost is not
claimed bounded by ciphertext work. Implementation records include the unchanged
replicated-scalar helper dependency as well as the actual imported shared files.

The staged operator set is explicit. Convolution, pooling and BatchNorm reject;
they are not covered by this mixed implementation. Public-only arithmetic rejects
(the graph-v2 validator already requires an encrypted first operand). Arbitrary
mapped views never masquerade as canonical chunks. The unchanged 1024 primitive
emission gate and an early 64 KiB statement gate apply; the ordinary candidate
validator still checks total source, AST, native helper expansion and work. There
is no depth, modulus/capacity, peak allocation or execution-time proof.

The new test file defines 28 pure controls, including the three original P4 MHA
requests, generic broadcast/partial reduction/split/multioutput/late Linear, poisoned
tails after uniform public addition, cell and pack caching, exact mask direction,
source/binding integrity and original resource/rejection boundaries. It uses the
existing NumPy dependency and an independent bounded AST periodic-vector
interpreter; it never executes candidate Python or invokes SDK/compiler/FHE/API.

Control entry point for the root's existing pinned environment:

```sh
python3 -B -m unittest discover \
  -s scripts/baseline/benchmarks/tools/mixed_packed_scalar_rule_v1/tests \
  -p 'test_*.py'
```

Author preparation completed only AST parsing and in-memory `compile()` of the
new Python files. The 28 controls and numerical checks have not yet been executed
by the author. Root-owned `backend_acceptance.py` performs a separately frozen,
bounded control/real-backend sequence; any later result must retain its own source,
artifact, memory-gate, numerical, export/reload and cache evidence. Reduced source
operations do not establish that the existing 4 GiB / native 150 s limit fits.

The first root control window ran 25 controls: 16 passed and 9 errored while
recording the internal packing's `excluded_scalars` set as JSON. It stopped before
compilation or FHE. Original tool/test/README bytes are preserved and SHA-verified
in `source-before-control-v1`; the original control log/report is unchanged. The
local repair checks the complete trusted packing field schema, sorts only the
explicit `set[str]` field and binds the versioned serialization of every remaining
field unchanged. Three new controls cover deterministic/nonmutating serialization,
strict type/schema rejection and full descriptor/vector/schema hash sensitivity.
The repaired 28-control version awaits root's separate frozen v2 window.
