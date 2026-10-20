# ResNet18 validated precision correction

Full encrypted model execution and the independent audit PASS for image0.
All 1000 logits satisfy the original absolute0.1 + relative0.006 gate; imaginary
maximum remains below0.01 and prediction65 equals label65. This is single-image
validation, not dataset accuracy.

The final profile combines ratio128/internal scale60. Component maximum error
is1.2492839269e-5 and RMS9.6669820591e-6. Full model maximum error is
0.008928450882893735 and RMS0.002623180777834832 (19.98-fold maximum improvement
from the failed ratio32/scale52 model). All4705 Native boots completed.

The model, weights, d13, average pooling, boundary20, N65536, Q50/P25, H192,
NoHoistBsgs and degree58/generation59 are unchanged. Existing Dacapo plans all
model levels/rescaling/bootstrap placement from the validated capabilities.
The GPU entrance adds only the bootstrap_ratio config read. Runtime and crypto
libraries are reused; no intermediate decrypt/re-encrypt or fixed FHE seed.

- `profiles-combined/`: the final compiler/operator/native configuration.
- `runtime-support/`: GPU test entrance and actual component probe C++ source.
- `evidence/final-audit.json`: independent final acceptance.
- `evidence/gpu-full.json`, `clear-full.json`: complete1000-logit results.
- `evidence/arithmetic-audit.json`: unchanged logical output graph.
- `evidence/*provenance*.json`: exact historical commands and SHA records.

Python files preserve the original server workflow in
`/home/hanshubin/no-hoist-dsl-work/resnet18-precision-combined-v1`, with sibling
`resnet18-model-v2` and `resnet20-framework-v1`. Component generation/comparison
also needs retained historical component artifacts. They are server experiment
scripts, not a portable one-command installer. Paths require adaptation on a
new machine and provenance must be regenerated. Failed ratio-only/scale-only
experiments remain on the server rather than being published as accepted profiles.

`evidence/verify_release.py` checks shipped source hashes and independently
recomputes the final numerical gates from the clear/GPU output reports.
