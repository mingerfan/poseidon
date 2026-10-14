# Unified ordered native MHA capture overlay

This isolated draft registers native eval torch.nn.MultiheadAttention as a node
inside the existing ordered graph. It is not the earlier MHA-only worker route.
Linear before/after attention, residual arithmetic and other tensor ports continue
through the same capture/emission/alias normalization pipeline.

Owned candidates are pytorch_static_trace.py, pytorch_model_capture.py,
pytorch_mha_source.py and pytorch_mha_math.py. The build_capture_overlay.py tool
writes only these four files under overlay and records their hashes in
capture-overlay.json; it verifies the two original production hashes first.
Other overlay files are owned by the parent and independent review agents.
Nothing here applies a patch to production.

The trace and capture opt in with
mha_source_policy="poseidon-mha-ordered-source-v1"; their default None preserves
the previous routes. The parent must restrict this flag to the explicit v16
contract. Signature tracing computes tensor metadata only. The three Q/K/V
operands bind actual FX producers, without model name, factory or case dispatch.
Native MHA returns a real static tuple of tensor proxies and optional literal
None. Selecting a tensor port emits an alias identity so existing arithmetic
source checks can track its producer. None is allowed only by the explicit v16
output-tree policy; it creates no cipher output.

The mathematical helper reuses the frozen mha_source_plan.py formulas.
Only implementation identity lookup changes from the benchmark-relative path to
the actual loaded trusted graph/decomposition/resource module files. The adapter
must lock those trusted imports and hash the same files; no missing file or hash
is ignored. Its packed Q/K/V weight and bias slices have exact
original-state source/indices records in the existing public_constant_views.
The source checker independently rebuilds every mathematical node and constant,
binds all original parameters, and compares masks to the original loaded state
with their original Boolean/floating dtype. Native MHA Boolean True blocks;
the allow mask is its logical inverse. A fully blocked row is rejected. Masks
must be finite original public tensor attributes. Dynamic masks, implicit causal
masking, training behavior and arbitrary subclasses/hooks are not registered.

pytorch_mha_source.record accepts graph, emissions, entries and keyword
normalized_origins, source_origins, public_constant_views, resource_profile.
check_record additionally accepts the all-or-none loaded binding triple
state_constant_sources, declared_records, expected_weight_binding.
The host contract must supply that triple. Reference-only math checks can omit
it after the host has validated the source closure. Both functions require the
external source-origin and public projection-view records. Native MHA origin
markers identify all expanded segments, so omitting an entry is rejected.

check_record returns a dict keyed by RAW emission exp/reciprocal node IDs with
values {source_node: original FX name, module_paths: original ancestor paths}.
It also independently verifies that every such raw nonlinear component has
exactly one corresponding normalized graph node and an unchanged profile.
The existing approximation checker consumes the raw stream. Selection row
node_id remains fx_ plus the original MHA FX name, including repeated calls of
the same module. Graph attrs never determine module source location.

Standard-library controls cover signature rejection, policy defaults, production
and candidate identity, unchanged mathematical functions, packed/separate
projection inventories, complete MHA origin inventory and normalized nonlinear
provenance. Actual pinned Torch capture, complete v16 source-file/worker isolation,
independent 16-probe numerical checks, backend compilation, encrypted execution
and fresh Agent generation are separate gates. No such result is claimed by
these controls. The parent coordinates the real existing Nix Python window.

Production integration still requires the parent's explicit ordered contract,
intake/facade, adapter IMPLEMENTATION whitelist, source-output None version,
approximation source-location check and isolated worker source-hash binding.
The independent reference fixtures include self multihead, separate dimensions,
Boolean plus padding, sequence-first None, bias/extra-token and unbatched
contexts. These representatives do not establish all MHA coverage.


## Observed validation and remaining gates

The owned standard-library suite passed 26/26 controls after the shared-mask
fix (capture-controls-v3.log). The final coordinated pinned Python window passed
40/40 real controls, including the independent dataflow/contract/tamper controls.
The isolated capture-v2 audit passed 9/9 sources with 16 finite reference groups
per source: 144 groups and 1,856 compared values. It used the final helper
b683290bea8d52cc28b45ad9877d875bee0d87e390602de10bf3956fe357652b.
The earlier capture-v1 and failed controls-v1 remain historical evidence bound
to their original tool identities; they are not re-labelled as this version.

The six original high-accuracy configurations account for 96 groups and 1,184
values. Their largest observed polynomial-versus-native MHA error was
2.4733121106734757e-10. Three separate declared_shallow_v1 configurations account
for 48 groups and 672 values; their corresponding largest original-function
approximation errors were 0.2845255132107518 (self/post-Linear/residual),
0.006419956608951352 (separate-dimension cross/None), and 0.7659458046621783
(Boolean/padding/residual). The original six models, references and approximation
profiles remain unchanged. The shallow configurations are separate explicit
profiles, and do not certify the original model's accuracy. Across the nine
captures the largest decomposition discrepancy was 6.661338147750939e-16.

The deterministic_rule_baseline backend attempt passed 0/3. All three completed
real source tracing and stopped at compilation. The original compile logs report
earth.mul returned-type inference failure. The trusted Earth-stage diagnostic
calculates consumed budgets 11*60+75=735, 12*60+60=780, and 11*60+75=735 against
the unchanged 720 gate. This is an Earth compiler-stage budget, not a RuntimePlan
modulus index or a measured CKKS error. No successful compilation, encrypted
execution, decryption/numerical acceptance, package reload or new Agent generation
is claimed by this attempt. API calls remained zero.

The active source remains production 685; MHA v16 remains an isolated development
overlay. The next implementation gate is a generic legal DSL/layout expression
that fits the existing budget while preserving the declared model, parameters,
approximation configuration and reference. Backend/Dacapo code and compiler
parameters are not changed here. A further acceptance scope must be recorded
separately before re-execution. Source integration, numerical approximation
accuracy, compilation, encrypted execution, package reuse, fresh Agent generation
and production activation are distinct milestones.

Evidence is under the ARM results root:
- agent-mha-ordered-controls-v3/report.json
- agent-mha-ordered-capture-v2/report.json
- agent-mha-ordered-backend-v1/report.json

These reports are read-only historical records. Their identities below were
read from actual bytes when this README was updated; the implementation and
source closure are retained in each report.

```
{
  "reports": [
    {
      "binding": null,
      "report": "agent-mha-ordered-controls-v3/report.json",
      "sha256": "e28701cdf49ceada90a25d3c615897ff87a210b21ab936860f32691fe4c03d77",
      "status": "passed"
    },
    {
      "binding": "31424552889607a465a1d63ae7ecb02f9427818b5179e19d6f9676d0bc8c0c30",
      "report": "agent-mha-ordered-capture-v2/report.json",
      "sha256": "2910f4b63c307a759347d309da48eddaa901a6b98e15a98b241efb678a9a9dad",
      "status": "passed"
    },
    {
      "binding": "e36e5ad033cf22afceb32c4e625b79dceea1e3f238574cc581a83f45e706a8f6",
      "report": "agent-mha-ordered-backend-v1/report.json",
      "sha256": "f69fc3bdd24c514cf73ea2b2d26ef9f5e966a7b86ebdae1c48edb537e1c5847b",
      "status": "complete"
    }
  ],
  "standard_library_log_sha256": "3530e853dd09d0b1bd0cb2430e2e75a588e4e1dc38104259e99145fdd851a0c1"
}
```
