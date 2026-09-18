# Object arrays and Hecate Empty: construction-stage evidence

Current status: **versioned request v18 / AST v17**, enabled with `--object-arrays`
in candidate and batch runners. Construction metadata uses schema 12. Existing
request v17 / AST v16 and earlier modes retain their rules and metadata.
Manual real CPU golden validation is now complete for the listed paths; online
Agent generation of this grammar is not yet measured. See
[actual CPU results and precision caveat](object-array-cpu.md).

## Location and purpose

`scripts/baseline/object_arrays.py` supplies checked object storage to
`function_construction.py`. Its elements are symbolic cipher/plain IDs,
construction Empty sentinels, None, or bounded public real scalars. The array is
NOT a ciphertext's CKKS slot array. Mutating a cell selects/rebinds a symbolic
expression; it does not modify encrypted slots in place.

The upstream evidence is `third_party/dacapo/python/hecate/hecate/expr.py`
(`Empty`, `resolveType`, Expr dispatch) and
`third_party/dacapo/python/poly/poly/MPCB.py` (`fint`, `roll`, `SumSlots`).
Tests extract the actual upstream SumSlots AST unchanged from shapeClosure and
lift it with its fint/roll dependencies. No algorithm substitution is used.

## Implemented bounded paths

- Explicit `np.full/empty(..., dtype=object)` and
  `np.array/asarray(..., dtype=object)`; controlled `object`/`np.object_` markers.
- Integer/tuple/slice reads and writes; scalar-cell augmented assignment with
  target/indices evaluated once before RHS effects.
- Shape/size/ndim, len, row iteration, reshape, flatten, copy, transpose/T,
  concatenate. Native pinned NumPy preserves view/copy/overlap behavior.
- Empty()/hc.Empty() addition/subtraction for the verified operand families.
- One-dimensional object-array returns containing the declared ciphertexts.
  Multidimensional returns are rejected, never silently flattened.

Important semantic distinctions:

| Expression | Verified upstream behavior |
|---|---|
| `Empty() + x`, `Empty() - x` for Expr x | Both return x unchanged |
| `x + Empty()`, `x - Empty()` for Expr x | Expr conversion fails; Empty is not numeric zero |
| Scalar/list plus or minus Empty, in either order | resolveType constructs Plain |
| `np.empty(shape, dtype=object)` | Cells initially contain None, not zeros or valid ciphertexts |
| Slice / asarray of object array | Shared storage |
| copy / flatten | New storage, shallow symbolic references |
| reshape | NumPy may use a view or a copy, depending on layout |

Unsupported families are rejected explicitly: vectorized object-array
arithmetic, nested mutable objects in cells, boolean/advanced/secret indexing,
arbitrary NumPy methods, reflection, and opaque Python objects. Arrays currently
retain bounded rank (4) and element count (128); current ciphertext ABI/packing
restrictions are unchanged. These limits are not a claim of full Hecate support.

## Initial construction-only verification on 2026-09-12

1. New focused checks: **15/15 passed, no skips**, in the existing Nix environment
   with NumPy version explicitly asserted to be **1.25.2**.
2. Upstream SumSlots tested at `(m,p) = (1,1), (2,1), (3,1), (4,1), (2,-1), (2,2)`.
   For the period-four input, independent reference output is
   `y[i] = sum(x[(i+j*p) % 4] for j in range(m))`. This covers positive/negative
   stride, power-of-two and non-power-of-two reduction paths.
3. Nine overlap-assignment configurations are compared with native NumPy;
   tests also cover aliasing, non-contiguous reshape, copied storage, lazy row
   iteration, None cells, Empty operand order, augmentation side effects,
   unsafe inputs, and unchanged legacy rejection.
4. Combined new/old semantic tests and existing encrypted-artifact audits:
   **225/225 passed, no skips**. This re-audits historical FHE evidence; it does
   not create new object-array FHE execution evidence.
5. System-Python full offline regression: **645 total, 500 passed, 145
   conditional skips, no failures**. New NumPy tests are skipped in that system
   environment but actually ran in item 1. Expected argparse negative-test and
   mocked provider retry messages are not live API failures/calls.

Reproduce the focused authoritative checks from the WSL source root:

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 160s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_object_array_checks.py
```

The runner reuses existing Nix dependencies without installation and writes a
small report under `/home/lhy/poseidon-work/results/object-array-checks-*/`.
Verified report for this implementation:
`/home/lhy/poseidon-work/results/object-array-checks-nww_2cjh/report.json`.
It records tested source hashes, source stability, test counts, and explicitly
`api_calls=0`, `new_fhe_executions=0`, `agent_contract_enabled=false`.

## Subsequent gates and remaining work

1. Completed: versioned request/AST contract, sandbox dependency binding,
   grammar inventory, CLI/batch selection and evidence replay; 17 focused
   fixed-environment checks pass. Current construction-only report is
   `/home/lhy/poseidon-work/results/object-array-checks-ndzndigf/report.json`.
   The earlier report above remains historical and is not overwritten.
2. Completed for listed paths: 4 positive/2 negative real SEAL CPU goldens,
   plus 2 independent-key repeats of the near-threshold four-slot reduction.
   Combined semantic/artifact audit: 229 passed, no skips; system-Python
   offline suite: 649 total, 500 passed, 149 conditional skips. Details and
   unchanged numerical/security requirements are in the linked CPU record.
3. Still open: actual Agent generation on newly enabled semantics, separately
   from manual goldens. No paid API call occurred in either of these stages.
4. Continue complete semantics work: remaining object/numeric array operations,
   general packing, full high-level helper paths, other public-language rules,
   IR calls, true bootstrap/upscale and backend verification. Full project DSL
   coverage and Poseidon GPU end-to-end correctness remain unproven.
