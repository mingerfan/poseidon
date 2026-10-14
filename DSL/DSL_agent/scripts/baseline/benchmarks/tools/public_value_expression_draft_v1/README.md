# Public value expression draft

Agent-owned mathematical groundwork for static transformations of public buffers.
Operations: reshape, permute, static slice, and explicit bool-to-float32/float64 cast.
Source values, dtype, shape and each transformation are independently bound.
Plain NumPy normalization is checked against real CPU Torch on three contexts per operation,
with four numerical probes per context. This tool is separate from production source capture.

Final: 42 tests passed, 0 failed/errors/skipped; 12 contexts, 48 reference groups.
Initial scalar-reciprocal failure and its original source/log/metadata are retained.
The final controls-v2.json records actual executed counts and source hashes.
No DSL generation, tracing, compiler, encrypted execution or Provider calls occurred.

Next: map FX public-buffer expressions, retain original/effective state provenance,
then join production only at a safe source boundary and qualify tracing, Poseidon CPU and Agent generation.
Static field/operation/type checks remain; encrypted dependencies and arbitrary computation are rejected.
