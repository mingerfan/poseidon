# Poseidon / DSL Agent development

## Checkout and branch routing

The integration checkout is /home/lhohy/Code/Poseidon in ARM Ubuntu, on lhy-agent-dsl.
The retained Mac copy is historical and must not be developed or synchronized over Ubuntu.
Codex SSH project tasks execute in Ubuntu directly; verify pwd, uname, Git branch and running jobs first.

poseidon is upstream main. DSL is the future shared team branch and must not be created by this task.
Confirmed feature branch names: lhy-generalization and lhy-resnet20. They require separate Ubuntu working directories and the same committed integration baseline before concurrent development. Do not switch branches in an active shared checkout.

Agent source and tests: DSL/DSL_agent/scripts.
Agent documentation and handoff: DSL/DSL_agent/docs.
Other members' DSL packages: DSL/DSL_benchmark; preserve their original bytes.
Old scripts and docs/baseline paths are compatibility entry points.

Read DSL/DSL_agent/docs/coordination/TASK_STATE.md and PARALLEL_DEVELOPMENT.md before continuing.
A owns generic component/operator/layout work; B owns whole-model and backend certification workloads. Public interfaces and shared files follow the ownership matrix.

## Required boundaries

Preserve all existing changes and historical responses, source hashes, reports and program packages.
Do not modify Poseidon, CKKS Runtime or Dacapo libraries to solve Agent issues without a concrete separately approved proposal. Upstream integration is recorded separately from local backend implementation.
No SEAL fallback, plaintext simulation, intermediate decrypt/reencrypt bootstrap, reference changes or tolerance relaxation.
CKKS: 1e-5 + 1e-4*abs(reference). Independent mathematical difference: 1e-12 + 1e-12*abs(reference).
Never alter networking, proxy, DNS, routes, NAT, bridging or VM network configuration.
Credentials remain at the project root .env; only the approved Provider loader reads them. Do not print, transfer or commit credentials.

Plan tests already have user approval. Freeze each batch and enforce its finite budget; do not re-ask for unchanged test authorization.
Global evidence 48 GiB; reserve disk 4 GiB; per case/batch 4 GiB; compile total 2, link 1; API and encrypted execution initially serial across both branches. Preserve sandbox, network isolation, hashes and native limits.
Installations, downloads, new data/weights, backend/security parameter changes and resource expansion still require separate approval.
Local commits, pushes and PRs each require explicit approval. Do not follow older automatic commit/push guidance.
No blind stash pop, reset --hard, checkout --, clean or loss of unrelated files. Snapshot and retained stash are recovery assets.

Goals are thread-local. Start a new feature Goal only after its branch/work directory and common baseline are ready. Do not mark blocked or unverified milestones complete.
