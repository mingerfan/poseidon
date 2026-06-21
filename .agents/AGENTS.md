# Multi-GPU Static Schedule Development Notes

Always read `.agents/mgpu_static_schedule_plan.md` before changing the multi-GPU
runtime, compiler, interpreter, communication, or Dacapo integration.

Mandatory constraints:

- Preserve existing single-GPU GPU use cases and public behavior.
- Keep multi-GPU functionality behind an optional build path.
- V1 only supports ciphertext-level parallelism with full objects on one GPU.
  Do not pass `fields_.size() > 1` ciphertexts into the existing `GpuEvaluator`.
- Do not implement multi-shard execution in V1.
- Do not scatter cross-GPU copies into evaluator or kernel handlers. All
  cross-device movement must go through the multi-GPU communication layer.
- The interpreter executes a static schedule. It must not make dynamic placement
  decisions at runtime.
- Static placement options own upload, compute, and download device choices.
  The interpreter and GPU runtime must consume those assigned devices without
  overriding them.
- When inputs, compute, or downloads live on different devices, rely on the
  copy insertion pass and the mgpu communication layer. Do not add ad hoc
  copies in GPU evaluators or operator handlers.
- Dacapo is expected at `src/poseidon/mgpu/third_party/dacapo` from
  `git@github.com:corelab-src/dacapo.git`.
- Use Nix for Dacapo dependency isolation when needed. Do not modify the system
  package environment for Dacapo dependencies.
  The local shell template is `src/poseidon/mgpu/nix/dacapo-shell.nix`.
- Do not delete files outside this repository. Do not delete unrelated untracked
  cache directories inside the repository.
- Keep memory use conservative. Avoid broad parallel builds unless needed.
- After each small completed feature, run the relevant tests, commit the scoped
  changes, and push the development branch.

Temporary implementation/testing note:

- The current multi-GPU path is intentionally incomplete. Full execution is not
  expected to run yet because lazy rescale support is still needed for the
  low-bit-width GPU path. Until this note is removed, do not require full
  end-to-end ResNet20 or full GPU execution tests at the end of each small
  change; run simple, focused tests for the touched CPU-side/tooling behavior
  instead.
