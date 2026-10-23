# Compiler profiles shipped with pt_agent

Byte-identical copies of the files the toolchain passes to `hecate-opt`, so that pt_agent does not depend on
the legacy SDK selector to locate them.

| File | Source | Used for |
|---|---|---|
| `poseidon-compiler-profile-v1.json` | `DSL/DSL_agent/scripts/baseline/` | `--ckks-config` for compilation without the boot profile |
| `poseidon-operator-spec-v1.json` | `DSL/DSL_agent/scripts/baseline/` | operator-spec digest written into the RuntimePlan metadata |
| `qwen48/compiler-config.json` | `DSL/DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL/config/` | bootstrap48 CKKS configuration and the latency table of the cost model |
| `qwen48/operator-spec.json` | same | operator-spec digest for the bootstrap48 profile |

The operator-spec digest is metadata required by the RuntimePlan format; it is not an integrity check.
