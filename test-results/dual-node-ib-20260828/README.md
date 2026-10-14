# Dual-node IB profile (2026-08-28)

`communication-profile.json` models two MPI ranks on two nodes: `rank 0 -> node 0`
and `rank 1 -> node 1`. The `dual-node-ib` rule applies to device traffic in both
directions and uses the supplied two-rank NCCL all-reduce measurements as an
effective inter-node rate curve. Profile units are microseconds and bytes/us.

The compiler requires rate points to be monotonically nondecreasing. The measured
1 GiB point (45.50 GB/s) is therefore represented by the running maximum
46.20 GB/s; all other points are unchanged. This is a modeling envelope, not a
claim that the raw benchmark value was 46.20 GB/s.

No NVLink benchmark was supplied. `intra_rank` uses a provisional 100 GB/s ceiling
and 2 us startup so placement does not incorrectly prefer IB over the expected
intra-node NVLink path. Replace this link with a measured V100 NVLink curve before
using the plan for performance predictions.
