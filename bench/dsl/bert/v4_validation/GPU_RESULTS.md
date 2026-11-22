# Frozen v4 GPU results (T128, one complete BERT encoder block)

| Configuration | Online median | candidate / Cerium ×100% |
|---|---:|---:|
| algebra + shared rotation rescale (A) | 9.107562999s | 150.314623% |
| + prepared cyclic QK (B) | **8.222786937s** | **135.711948%** |
| Cerium native reference | 6.059s | 100% |

A/B used the same new runner, same exact input/16 parameter hashes, 1warmup+3 complete measured runs,
within one window. B latency was9.714740% lower than A. Each actual run had23 native Boots/27 prepared
Linears; B also had6 real QK pairs. All exposed internal QK counts match the audited schedule.
Final98,304 slots finite, but maximum reference errors A6.1560/B6.1739: **not an accuracy pass**.
Different HE/approximations/Graph policies versus Cerium remain explicit.

A samples(s): `[9.100574836,9.110845954,9.107562999]`, Runtime wall median10.797662298s.
B samples(s): `[8.221144835,8.222786937,8.226259263]`, Runtime wall median9.775053715s.
B main-process wall548.709s was longer than A504.987s because key/setup costs increased;
these online improvements are not equal end-to-end gains.

Standalone QK pair median0.296522763s/max error1.57e-10; standalone AV pair median0.146162465s/
all-slot max error6.24e-9. AV72 prepared IDs shared one1,037,303,808B mask bundle; only one ID was
executed4 times. Neither pair time can substitute for full-model/12-layer latency.

Original evidence under `test-results/bert-a800-20261115/v4-a800/`:
- `20261115T132848Z/`: first QK probe + fresh v3 control9.872900606s; guard stopped before A on0MiB/6%.
  This is a DIFFERENT window and its large control error229.08 is preserved, not attributed to binary alone.
- `20261115T140945Z/`: A/B/AV all completed once; GPU released14:30:12UTC, independently confirmed
  empty14:31:13UTC. `FINAL-REPORT.md`/`machine-summary.json`/raw reports contain exact measurements.
- Full final archive SHA256 `c663c471885e8beec82ced7b045ff90823bf57a4a0b22c97cdad834dcfffa706`.

Small report-only backup: `v4-reports-small.tar.gz` beside this document, with its own SHA manifest.
It intentionally excludes native binaries, giant plans/weights/bundles and full decoded arrays; the full
archive and remote immutable assets remain preserved separately.

Standing user authorization now permits ongoing bounded experiments and nonzero utilization when
resources suffice. Later admission policy changes are NOT retroactively applied to these strict-clean
runs. Active v5 polynomial/v6 AV integration and later12-layer measurements are separate snapshots.
