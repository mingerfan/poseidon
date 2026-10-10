# Qwen compiler component experiments on 188Server

These scripts, profiles, logs, and reports record the initial block compilation,
the slow-pass investigation, and iterations of fast bootstrap placement. They
include failed and superseded experiments as historical evidence. The final
component results are in `fast-boot-summary.json` and
`optimized-fixed-rmsnorm-comparison.json`.

The complete 24-layer compilation and current single-/four-GPU memory results
are in the sibling `qwen24-dacapo-188-20261010/` directory. Full-model source is
under `llm_dsl/Qwen25_24Layer_DSL/`.

Generated IR, RuntimePlan JSON, plaintext bundles, binary payloads, and Python
caches are excluded from Git. Large server artifacts remain in
`/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/`. Scripts use fixed experiment
paths and can overwrite reports; review them before rerunning.
