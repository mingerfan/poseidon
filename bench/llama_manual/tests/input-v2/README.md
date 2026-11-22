# Manual input V2: bounded source fix and CPU-only evidence

This directory documents the new worktree's input contract. It supersedes only
the input/scope statements in `../../MANUAL_LLAMA.md`; no arithmetic, plans,
implementation flags, thresholds, reference semantics, or GPU admission changed.
It is **not** deployment authorization or model/accuracy qualification.

## Closed CLI contracts

Argument positions remain unchanged (12 arguments including the executable):

```text
manual --run                 PLAN FIXTURE WEIGHTS_FILE INDEX EMBEDDINGS REFERENCE 1      38 MANIFEST REPORT
manual --test-inputs-cpu     PLAN FIXTURE WEIGHTS_FILE INDEX EMBEDDINGS REFERENCE 1      38 MANIFEST REPORT
manual --run-v2              PLAN FIXTURE WEIGHTS_DIR  INDEX EMBEDDINGS REFERENCE {1|32} 38 MANIFEST REPORT
manual --test-inputs-cpu-v2  PLAN FIXTURE WEIGHTS_DIR  INDEX EMBEDDINGS REFERENCE {1|32} 38 MANIFEST REPORT
```

`PLAN` is still one of the two existing explicit plan names (both end in `-v1`);
input V2 does **not** introduce a new FHE plan. `Plan()` remains deleted.

- **V1:** magic `MANUAL_LLAMA_INPUTS_V1`; the same six roles and declaration
  grammar as before. `weights` is the pinned regular file `layer.weights.bin`.
  `SafetensorWeights` now receives its canonical **parent directory**, not the
  file itself. The pinned 818-byte index contains only layer 0; requesting 32
  layers is rejected, including in CPU-input mode.
- **V2:** magic `MANUAL_LLAMA_INPUTS_V2`; `weights` is the canonical full-HF shard
  **directory**, declared with `-` rather than a directory hash. The full checked
  index covers all 32 layers. Initially only layers **1 or 32** are allowed, using
  the original corresponding frozen references; there are no new intermediate
  pins or relaxed reference/threshold options.
- Both CPU-input modes return before report creation, Context, Runtime, keys,
  or GPU use. After declaration validation, the executable reads only the public
  fixture literal before that return. The std-only tests do not instantiate the
  executable or parse real fixture/model/reference contents.
- CLI/manifest version disagreement fails on the first manifest line, before
  payload path checks/reads, report creation, or Context.

All manifest, input and shard paths must be absolute, whitespace-free, canonical,
non-symlink paths, with non-symlink parents. Files must be regular; the V2 weights
root must be a directory. The report parent must be canonical and the report
must not exist, including a dangling symlink. The unchanged executable report
writer retains exclusive/no-follow creation, mode 0600, write loop, and fsync.

## Manifest V2 grammar

Exactly ten unique roles; no extra/unknown/missing roles or trailing tokens.
The four shard **role names are their exact fixed filenames**. Their paths must
be exactly `WEIGHTS_DIR/<same fixed filename>`, not arbitrary aliases or paths
that normalize to the same file. Ordering of rows is immaterial.

```text
MANUAL_LLAMA_INPUTS_V2
fixture FIXTURE d9a79047125a6506c6b2d4c335737ccdb52a216f398a8150b133d47a44f72f56
weights WEIGHTS_DIR -
index INDEX fd89f8e33fac43f92e45d26477628bfa263de5b8e8159874ed22c71c8fd087dd
embeddings EMBEDDINGS 5e1a1e39b2021a66119bdb7c5b2dd8a65d4e6fd701c2ce16f52000f27f00839f
reference REFERENCE REFERENCE_PIN
report REPORT -
model-00001-of-00004.safetensors WEIGHTS_DIR/model-00001-of-00004.safetensors f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b
model-00002-of-00004.safetensors WEIGHTS_DIR/model-00002-of-00004.safetensors c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0
model-00003-of-00004.safetensors WEIGHTS_DIR/model-00003-of-00004.safetensors d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383
model-00004-of-00004.safetensors WEIGHTS_DIR/model-00004-of-00004.safetensors e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d
```

Placeholders above are not an admitted manifest. The V2 checked index is the
30,789-byte `LLAMA31_HF_SAFETENSORS_V1` format used by `SafetensorWeights`, not the
upstream JSON index. The four shard pins were read **only from public metadata**:
`llama-cerium-align/gpu_api/bench/llama31_cerium_source15/inputs/WEIGHT_DOWNLOAD_LOCK.json`,
model `LLM-Research/Meta-Llama-3.1-8B`, revision
`24ce1d3ac24d46a14b16dcb666c4fdd9147b87ea`. No weight files were inspected.

Shared reference declarations:

| Requested layers | Exact original reference pin |
| --- | --- |
| 1 | `7cbc25944cfeddc720415e9ceb6b61001f92cdf20fd9e93a389dcaabdf11cc25` |
| 32 (V2 only) | `1d4a5ec9b4686b7d12a83b29e207dd84a7f9463a80e9a9628267eeacf25c3cc8` |

V1 retains its weights-file pin
`3a4780271abf7780094e272be10abe3dcc089f3cbde6301d0cc33329a70e0e7d`
and checked-index pin
`36b6f66f819aa0e34683f9adce037b599d2c4572ae97a3ea670b7600a4d49b32`.

**C++ checks declared pins/roles/paths, not cryptographic payload hashes.**
Trusted external Main CPU preflight must hash the actual admitted files against
these exact pins before any deployment. In particular it must bind V1's indexed
`layer.weights.bin` to the declared file, and V2's four fixed root-relative shard
files to their fixed public pins and full index. Synthetic test declarations are
not evidence of that preflight, payload identity, key identity, or accuracy.
No dynamic arbitrary hashes, generated references, or new benchmark data are
provided here. Main retains deployment and payload-preflight ownership.

## Aggregate reporting

The executable now reports `input_version`, `model_weight_scope`, and
`model_weight_layers_available` in its final aggregate JSON. Values are:

| Version | Weight scope | Layers available |
| --- | --- | --- |
| `MANUAL_LLAMA_INPUTS_V1` | `layer0_only_file_and_index` | 1 |
| `MANUAL_LLAMA_INPUTS_V2` | `full_32_layer_HF_four_shard_directory_and_index` | 32 |

Available weight scope is distinct from the existing actual `blocks` executed.
CPU-input and begin status lines also name the version/scope. Existing
`historical_pass_inherited=false`, production/security qualification false,
reference kind, active38 complex-error gate, all128 finite requirement, and
other aggregate qualifications remain unchanged. No historical accuracy result
is inherited by this source generation.

## Evidence generations and reproduction

- `g01-before/`: immutable pre-edit source snapshots and the original suite's
  fresh result: **11 checks / 328 negative cases**, compile/test exit 0. The
  baseline runner differs from its snapshot only in relocating its hardcoded
  `/tmp` synthetic directory into this permitted evidence directory. This is
  baseline schema evidence, not proof that the loader mismatch was fixed.
- `g02-after/`: independent post-fix source snapshots, compile/test logs and exit
  statuses: **77 checks / 739 negative cases**, compile/test exit 0, empty compiler
  and test stderr. The actual result is in `g02-after/test.stdout`; not inherited
  from `g01-before` or old-worktree evidence. `RESULT.json` summarizes this bounded
  outcome; `CHANGED_FILES.txt` lists every changed/added path, including artifacts.

Both runs are std-only C++17 on **CPU 0**, with `OMP_NUM_THREADS=1` and
`CUDA_VISIBLE_DEVICES` empty. The after runner also sets `TMPDIR` to its exclusive
evidence directory. The expanded suite covers both plans and all four CLI modes;
V1 layer32/all unsupported scopes; V2 layers1/32; directory/file confusion;
wrong pin/path, missing/duplicate/unknown role for every role including each of
the four shards; malformed/wrong/version-mismatched magic before payload checks;
leaf/parent symlinks on manifests, every payload and report paths; shard leaf
symlinks/missing files/directories; and existing files/directories/dangling
symlinks at the report path. Positive checks ensure declaration validation does
not create reports and that loader roots are directories of the right scope.

To create a **new**, exclusive evidence generation from the worktree root:

```sh
bash bench/llama_manual/tests/input-v2/run-cpu-generation.sh g03-independent 0
```

The runner refuses to reuse a generation directory. It compiles/runs once and
stops at the first failure, preserving source snapshots/logs/exit statuses; it
does not automatically retry. Do not overwrite or repurpose an old generation.
The executable target/main was not compiled or linked by these std-only tests;
`SafetensorWeights` model loading and FHE execution were not performed. There is
no target, GPU, model, numerical accuracy, or performance claim.
