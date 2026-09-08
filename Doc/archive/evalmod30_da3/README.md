# EvalMod degree-30 + DA3 archive

This directory preserves the artifacts from the removed Git worktree
`/home/liufuyao/Work/poseidon_gpu_evalmod30_da3` on 2026-08-22.

The experiment used `K=16`, requested EvalMod polynomial degree 30, and three
double-angle steps. It improved runtime, but it was not approved for production
because SlotToCoeff and full-bootstrap correctness failed and EvalMod consumed
one additional Q prime.

Preserved files:

- `uncommitted.patch`: all tracked, uncommitted source changes from the worktree.
- `experiment_README.md`: the original experiment README.
- `full_20260813_172603.log`: the final full-bootstrap result log.

Original SHA-256 values:

- `uncommitted.patch`: `6a759e0f9ad5871badeed29a78b72e65926f47fe8b502b10f3f159dd48a9309e`
- `experiment_README.md`: `154c46415f31561646cacfff9eb0a74f60dd65c3bb2be503749fb10aab793501`
- `full_20260813_172603.log`: `8815475b3c29b40bfbdd458d3444bf51d0bf179435edd862d2127f4b7d01b7d7`

The experiment branch base, commit `c8480c4`, is already an ancestor of the
current development branch. The archived patch should therefore be treated as
historical evidence and must not be applied blindly to the newer planner.
