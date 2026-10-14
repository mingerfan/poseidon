# MHA v16 source integration overlay prototype

This directory is inactive. It only contains a deterministic review patch and
candidate copies. Nothing is applied to the active 684 production files or five
r1306 execution tools. No API, Nix entry, SDK capture, compiler or FHE is invoked.

The target architecture remains registered per-node MHA expansion inside the
unified ordered graph, with ordinary Linear, residual, views and output-tree
composition. This draft's explicit v16 native-MHA-only worker is a temporary
capture-boundary prototype. It is not the final architecture or a completed
production integration. It must not be activated as a whole-model shortcut.
Dispatch uses an explicit source version and exact native MHA class, never a
case/model/factory name.

The draft changes four candidate entrance files:
pytorch_model_intake.py, pytorch_ordered_contract.py,
pytorch_ordered_intake.py and pytorch_source_contract.py.
It adds five generic adjacent helper candidates:
pytorch_mha_plan.py, pytorch_mha_math.py, pytorch_mha_reference_math.py,
pytorch_mha_outputs.py and pytorch_mha_capture.py.
Normal helper loading has no benchmark-directory or revision-number dependency.

The frozen MHA planner and independent reference are reused. Mathematical
function ASTs are unchanged except the adjacent implementation-path identity;
the independent reference is byte-identical. The static FX grammar and original
projection-slice, tensor-shape, public-state and Boolean-mask checks are reused
from the v16 source draft. The new plan uses existing approximation selection
on each original native MHA node/path, retaining manifest defaults/module/node
selection and rejecting unmatched or conflicting selectors.

Three cipher roles are bound together with all fixed native call options before
graph emission. The complete original state, packed/separate Q/K/V and output
projection origins, source FX text/code, public mask state, approximation
choices, resource profile and None output tree are part of the request/plan
binding. Metadata checking rebuilds the graph and source plan, verifies original
state against declared/loaded NPY package identities, and uses the original
source module and independent exact-MHA reference as finite numerical authority.
MHA Boolean True means blocked; SDPA Boolean True means allowed. Entirely masked
rows and negative-infinity NPY state are explicitly rejected. Literal None has
no tensor/cipher slot and is retained only under v16. Default old output checking
still rejects None and versioned v16 trees; v1-v15 entrance paths are preserved.

The existing bwrap worker command, read-only SDK/source/weight mounts, unshared
network, process limits, 180-second boundary and fresh output path remain in the
candidate ordered intake. These have only been inspected statically for this
overlay. No new isolated capture or numerical/DSL/backend evidence exists.

Root must review and integrate the per-call helper into the unified ordered
pipeline before any source activation:

- pytorch_static_trace.py: native MHA declared-signature metadata for three
  operands and static two-result tuple; None second port only when need_weights
  is statically false. Private control/shape queries remain rejected.
- pytorch_model_capture.py: registered MHA branch before generic single-input
  binding; expand its checked mathematical plan into the emission stream with
  collision-safe names and per-source approximation selectors.
- pytorch_alias_capture_bridge.py and provenance checking: retain original
  packed parameter constants and exact slice/index views, prove Boolean-mask
  inversion/combination from the original state, and independently rebuild
  the complete emitted MHA segment. Keep ordered mutation/alias checks.
- pytorch_ordered_capture.py, pytorch_ordered_contract.py and the original-source
  reference: carry complete MHA call/normalization identities and literal None
  without weakening v1-v15 checking or replacing original numerical authority.
- Validate actual isolated self-multihead, separate K/V dimensions,
  Boolean+padding and None sources. Then validate MHA+Linear/residual/static
  output-container combinations through the normal application path.

Only after that integration should root freeze a new bounded acceptance scope.
Compilation scale, physical resource/layout feasibility, encrypted execution,
package reload/new-input/cache behavior and real Agent generation remain
unvalidated. The earlier four-context MHA source-math audit remains separate
historical evidence; this overlay does not reuse it as new capture acceptance.

build_overlay.py reads fixed current/frozen inputs and writes only fresh sibling
candidate, baseline-snapshot, integration.patch and overlay.json. It never
writes a production target. test_overlay.py executes only standard-library
controls, AST-selected pure functions and original-state byte comparisons.

The completed stdlib run passed 25/25 controls in 0.103 seconds. It includes
exact deterministic reconstruction, in-memory compilation, original target
hash preservation, mathematical AST/reference reuse, original float/Boolean
state-byte rejection, old output/dispatch compatibility and external plan
binding checks. These controls do not exercise the new worker or prove an
isolated capture. overlay.json records each before/after target SHA;
controls.json records this static result. Production target bytes and the four
frozen math/planning inputs still match their baseline identities.
