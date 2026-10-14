# Supplemental manual packed correctness goldens

The three original frozen directed tasks remain rule-preflight blocked at the
unchanged 256-operation limit. These separate manual candidates implement the
same immutable graphs and construction exercises; they are neither rule-emitter
successes nor Agent output. Mask constants follow the public 17-of-32 layout.
The polynomial case implements the logical rotate(-3) using two masks and the
actual provisioned positive rotations before reducing, rather than removing it.
Use each JSON/source pair with run_candidate.py --golden-file,
--unified-profile public-v1 and --unified-exercise from manifest.json.
Record results separately from the default benchmark runner's blocked tasks.
