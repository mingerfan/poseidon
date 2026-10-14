# Static MHA v16 trusted-source draft

This directory is a developer tool. It does not register a production PyTorch
source version or change a model, parameter, reference, compiler or backend.

The proposed source format is `poseidon-trusted-pytorch-v16`.
It reuses the hash-frozen r1063 native-MHA algebra and independent reference
under `../static_mha_source_draft_v1` and the current v15
operator graph/decomposition stack. A caller must explicitly provide a trusted
local eval module, static float64 input shapes and declared exp/reciprocal
approximations. No model-name or case-name dispatch is used in the source plan.

`capture(module, inputs, outputs, approximation=..., trusted=True)`
captures FX without sample shape propagation, then accepts only exact native
eval `torch.nn.MultiheadAttention` calls, original public state
mask attributes, fixed tuple selection and static output containers. It records
the original state bytes, complete call/options, source FX text/code and packed
or separate Q/K/V and output projection origins. Packed weight/bias row slices
are checked against their original source arrays. Boolean MHA masks retain
`True=blocked`, opposite to SDPA boolean masks. All-masked rows
are rejected. CPU float64 finite public state and Boolean state are admitted;
negative-infinity public state, training, hooks, subclasses, arbitrary operations
and private-data control flow are rejected explicitly.

`verify(plan, expected_source_request_sha256=...)` requires an
external source-request anchor and rebuilds the complete graph, decomposition,
provenance and implementation identity. Self-hashing a changed plan cannot
replace the caller's source identity.

The new output contract keeps a literal static `None` when
`need_weights=False`. It does not consume a named tensor or cipher
slot. It is accepted only under v16; the local compatibility check continues to
reject None under v15. This does not modify the existing production output
contract/reference. Tensor output order and nested tuple/list/dict structure
are preserved and independently checked against native and FX source outputs.

There are four small source contexts: multihead self attention, separate K/V
dimensions, Boolean attention plus padding masks, and cross attention with
sequence-first layout and `need_weights=False`. `audit.py`
uses 16 probes per context and stores original state files, FX, source plan,
native/FX outputs, independent exact-MHA and declared-approximation references,
NumPy/Torch lowered outputs, hashes and unit logs in a fresh local evidence
directory. Arrays stay in that directory; stdout contains only aggregates and
hashes. The audit has a 180-second plaintext window and 32-MiB artifact limit.

Standard-library controls need no Nix or SDK:

```sh
cd /home/lhohy/Code/Poseidon/scripts/baseline/benchmarks/tools/static_mha_source_v16_draft_v1
python3 -B -m unittest -v test_controls
```


NumPy/Torch controls and the audit must use the existing coordinated pure Nix
Python, never a separately provisioned environment. The parent task owns that
window. `audit.py` refuses an unpinned interpreter and an existing
evidence directory.

Production integration still needs an explicit v16 source/package/reference
version in `pytorch_model_intake.py` and
`pytorch_ordered_contract.py`, registered static MHA tracing/call
binding in `pytorch_static_shapes.py` and
`pytorch_ordered_capture.py`, original projection slice provenance
through the existing public-weight view contract, the inverse Boolean-MHA mask
checker, and v16-only output restoration/reference validation for static None.
The existing original-source reference must remain the numerical authority.

This draft does not prove full FX composition, all MHA configurations, functional
MHA spellings, alternate precision/device/training behavior, native fast-path
coverage, approximation accuracy outside the explicit profiles, backend scale
or resource feasibility, compilation, encrypted execution, Agent generation,
paid acceptance or production activation. Those require separate bounded
evidence after source intake integration.

The coordinated offline audit completed on 2026-10-08 with 24/24 controls,
four source contexts, 64 reference groups and 768 output values. By context the
counts are 16/256 (self multihead), 16/224 (separate dimensions), 16/160
(Boolean plus padding) and 16/128 (sequence-first cross attention with None).
The largest native-source versus independent-original error was
5.551115123125783e-17; the largest NumPy/Torch lowered versus independent
declared-approximation error was 6.661338147750939e-16. The observed
approximation-versus-original error was at most 8.43769498715119e-15.
These are bounded plaintext observations and do not certify a general
approximation domain.

Evidence is in
`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-source-v16-draft-audit`.
Report SHA256:
`c75bcd9572b9de3447f6ab4e7636d715d2bb0f6fb27c79717094cabbe3d03707`.
Report binding:
`8f9bfa6236aaa37176d17ec62041d29a72c144c7dd9e430fa23c095d4d5bac59`.
Unit log SHA256:
`abdadb9e17eae865a111241ec05351cd7d4d6ae7435fb22ab890a882987d7482`.

An independent standard-library audit rechecked all 67 artifact files
(1,112,369 bytes), report and plan/request bindings, all 17 current/frozen
project source identities and two Torch implementation source identities,
every original NPY state payload against source-request dtype/shape/bytes,
and all local reference counts, errors and file hashes. The measured audit
body took 0.488 seconds; environment startup/import time is excluded.

This run used trusted local FX tracing directly. It did not exercise the
production source worker or its bwrap boundary, isolated untrusted source
capture, DSL generation, SDK/backend tracing, compilation, encryption,
encrypted execution, decryption, Agent/API generation, package execution/cache
reuse or production activation. It does not establish support for all MHA
models. The frozen six Python draft files remain unchanged; this README is
documentation outside their recorded implementation identity.
