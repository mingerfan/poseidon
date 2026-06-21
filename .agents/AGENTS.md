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
- Dacapo is expected at `src/poseidon/mgpu/third_party/dacapo` from
  `git@github.com:corelab-src/dacapo.git`.
- Use Nix for Dacapo dependency isolation when needed. Do not modify the system
  package environment for Dacapo dependencies.
- Do not delete files outside this repository. Do not delete unrelated untracked
  cache directories inside the repository.
- Keep memory use conservative. Avoid broad parallel builds unless needed.
- After each small completed feature, run the relevant tests, commit the scoped
  changes, and push the development branch.
