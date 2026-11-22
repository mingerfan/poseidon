# Controlled diagnostics and narrow profiler capture

## Same-material plan comparison

`multi_plan/` now has an import-only diagnostic runner with one actual key setup,
one input encryption and one complete nativeBoot profile installation per process.
All plans start from the same original encrypted input. Separate Host cache maps
avoid ValueId collisions. This is not chained inference or old/new binary pairing.

The semantic-config compatibility P2 was fixed. A later CPU-only generation supports
v6/v7/v8 mixed attention via per-plan QK/AV adapters, while Linear/CPU material/profile
remain shared.86test methods and162guarded mock/real-fixture processes pass with
zero device calls. Independent bounded source/provenance review found no confirmed
remainingP0/P1/P2, including one-material setup and per-plan activation/drain order;
Later a separate targetGPU nine-forward diagnostic completed:all10AA decoded-f64
pairs bitwise equal(max/RMS0),BB/CCequal, actual207Boot/243Linear/54QK/54AV.
This limited successful lifecycle/replay is not production HEaccuracy or formal
performance acceptance. See[multi-plan GPU results](multi_plan_validation/GPU_RESULTS.md).
OriginalCPU-generation evidence and
fresh-build readonly binary are under
`test-results/bert-a800-20261115/multi-plan-build/attention-modes/`; recommended
binary SHA256 isb8796631a9a534aad894045a556452224ecc5a9f986d4b2109b758b987a4eb14.
Its checked A/A/B/A/C/A schedule has local absolute paths that require remapping to
an independently frozen target root. Mixed contracts are restricted to one block.
Per-plan GPU keys/masks are real additional allocations despite shared CPU key
material; total retained memory is reported, not a fictitious one-global bundle.
Earlier entry binary drift is explicitly preserved in that generation's history;
never label the current old-root ELF with an earlier37-test validation SHA.

Fixing only a main-context PRNG seed is insufficient: nativeBoot's default Q2/P2
subcontext creates another factory. Reuse the installed complete profile and real
keys/ciphertexts; do not repeatedly restart a fixed-seed factory as a production
sampling policy. Cross-process historical error6.41 versus229.08 has no established
binary-only or RNG-only causal explanation.

## Single-run CUDA profiler range

The active single-plan runner accepts `--cuda-profiler-iteration N`, zero-based
INCLUDING warmups. Default is−1 (disabled). For example warmup1/iterations1 withN=1
marks only the second complete Runtime run, leaving profile/key/weight setup outside.
This is a CPU-compiled integration, **not an actual Nsight trace yet**.

`cudaProfilerStart` occurs after prepared-engine reset, immediately before that
Runtime run; `cudaProfilerStop` follows run completion/API drain. The range includes
Runtime initialization, execution and finalization, not merely kernels. Online/wall
report boundaries remain distinct; profiler start/stop overhead is outside wall.
CUPTI overhead inside the interval still makes this an instrumented diagnostic.
The range closes on exceptions; malformed indices/probe misuse/loop-count overflow
fail before fixture load or device calls. Profiler indices now use full-consumption
`from_chars`, rejecting `0junk`/`0.5`/`-1junk`. A reviewed36-case native no-device
suite includes these, original5 profiler gates, async bad-manifest and QK capability
cases. Pure CPU validation precedes async allocator installation. Evidence is in
`qk-packed-inputs-integration/review-fixes/`; this partial-linked CPU binary is NOT
a coherent GPU deployment build. A fresh all-current-source build remains required.

A future worker should first inspect the installed `nsys --help`/CUPTI permissions,
then select its CUDA-profiler-API capture-range options. Do not blindly capture
hundreds of seconds of setup or treat unsupported profiler output as a sample.
No graph/async allocator/default key policy is silently changed.

All reports with root/per-run `instrumented_diagnostic` or explicit
`formal_steady_sample=false` are rejected by `summarize_aligned.runtime_summary`.
A raw diagnostic can retain timings, but cannot enter the ordinary formal latency
table. CPU submission, synchronized callback wall, CUDA-event elapsed and kernel
sums must be labeled separately; none alone proves the remaining bottleneck.
