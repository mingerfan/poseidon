# Model-to-Hecate Agent: first runnable CLI

New active goal: [complete project DSL semantic support](full-dsl-support.md).
The restricted CPU milestone below is a regression baseline, not completion of
that larger goal. Bootstrap/runtime, frontend grammar and helper gaps remain.

Arithmetic expansion: `run_candidate.py --extended-arithmetic` selects versioned
plain-left +/−/*, ciphertext rebinding and +=/−=/*= support. It works with the
existing prepare/manual/replay/live modes and preserves old request contracts.
Five manual programs now pass real isolated Dacapo/SEAL execution; two wrong
programs are rejected numerically. These are **not new live Agent results**.
See [full DSL progress and reproduction commands](full-dsl-support.md).

Public construction: `run_candidate.py --public-construction` additionally
accepts bounded public for/range, list/tuple construction, indexing, public
conditions, unpacking and shared-list writes. The interpreter saves and validates
a straight-line expansion before real Hecate tracing. Three manual positive
programs and two numerical counterexamples passed their expected gates; no new
live Agent claim is made. See [semantics and CPU evidence](public-construction-cpu.md).

Function composition: `run_candidate.py --function-composition` additionally
accepts undecorated top-level construction helpers, lexical local scopes,
higher-order calls, public bounded recursion and early returns. This is AST
construction, not arbitrary Python execution or multiple decorated IR calls.
Three manual positive programs and three targeted numerical counterexamples
have verified CPU evidence after one documented manual-golden correction.
See [scope, failure diagnosis and reproduction](function-composition-cpu.md).

GPU update: [real Poseidon CKKS Add/drop-modulus primitive validation](poseidon-gpu-modswitch-primitive.md)
now passes, including a tc128 security rejection counterexample. This is not yet
the Agent/HEVM GPU execution path; the existing SEAL CPU results remain separate.

New: [bounded Conv1D/2D and AvgPool1D/2D](spatial-operators-cpu.md) now have
independent references, manual DSL and real isolated CPU evidence. Larger
packing, live generation for these new families and Poseidon GPU remain open.

The Conv subset also accepts [explicit groups and dilation](grouped-dilated-conv-cpu.md),
including bounded depthwise/channel-multiplier fixtures. These remain user-defined
data graphs, not new catalog selections or new primitive/family count claims.

New: [schema-3 user-defined multi-input graphs](schema3-multi-input-agent-path.md)
now pass the isolated candidate pipeline with manual goldens (2..4 encrypted
inputs). This does not yet establish live LLM generation for the new contract.

## Current scope and evidence (2026-09-11)

Latest: the [approved 12-case advanced Agent batch](advanced-agent-12-results.md)
passed 12/12 on first generation using DeepSeek Flash and 10 API workers.
Wider MLP, grouped/dilated Conv and explicit power now have live generation plus
real CKKS evidence. The re-audited union is 126 successful cases: 78 user graphs
and 48 historical catalog descriptors. Older coverage statements below describe
their original cohorts, not the current aggregate.

The active task now targets Agent correctness on the existing Dacapo/SEAL CPU
pipeline. Poseidon GPU integration is out of scope; prior GPU records remain
historical evidence, not a prerequisite for this task. See the
[current grammar coverage and CPU scope](agent-cpu-grammar-coverage.md).

The saved heterogeneous Agent lineage has 96/96 numerically passing cases, but
only 20/32 audited structural DSL feature partitions occur in output dependencies.
In particular, live three/four-input generation and negative rotation syntax are
not established by that cohort. Manual golden tests are a separate evidence set.
The subsequent [six custom DeepSeek Flash cases](custom-agent-batches.md) now
pass real CPU execution on the first generation, including three/four inputs and
negative rotations. Five further [broadcast models](broadcast-alias-cpu.md) now
pass with versioned hash-bound semantic guidance (one TLS-failed call was rerun).
Seven [encrypted-zero user-graph cases](encrypted-zero-cpu.md) additionally pass
on their first live generation, including all-zero weight rows and mixed scalar masks.
The union of audited feature partitions is now 29/32 across 114 cases, not full DSL coverage; these
new-model cohorts remain separate from the old 96 and from manual golden evidence.
Do not rerun a historical paid batch merely to inspect coverage.

The [manual semantic evidence audit](manual-semantic-audit.md) now recomputes
original-input references and actual decrypted arrays for 61 saved manual
programs (42 correct, 19 detected counterexamples), and reparses HEVM/CST.
This is not a new Agent cohort or a new FHE execution batch.

The separate [user-model capability audit](model-capability-coverage.md) identifies
66 schema-2/3 user graphs and 48 legacy catalog descriptors within those 114 runs.
Do not call all 114 runs free-form user-graph evidence. Model parameter coverage
(hidden widths, groups, dilation) is separate from Hecate AST spelling coverage.

The new [self-contained 96-model suite](self-contained-model-suite.md) makes all
benchmark inputs explicit schema-2/3 graphs with public arrays. Both the rule
and Agent batch accept its `--case-manifest`; historical catalog reports are not
reclassified as user-graph Agent evidence.

For an actual **DeepSeek** call, use `--live` (historical alias `--deepseek`).
Current defaults are provider `deepseek`, model `deepseek-flash` (V4.1-Flash),
reasoning `high`, CLI output cap 384000 tokens and API timeout 1200 seconds.
Batch default is `--jobs 10`; memory-intensive native stages share two slots,
independently from API waits. Use `--loopback-proxy-port 6478` for the currently
verified command-specific proxy route. Keys still come only from the single .env.
Other provider adapters and explicit historical model choices remain available,
but old names below describe historical experiments, not the new default model.
Current parser API timeout default is 1200 seconds; historical 900-second settings
below describe older experiments. Keep generation configuration explicit.

## What this does

The entry reads a model description, asks DeepSeek to generate Hecate source,
checks and compiles it, executes the ciphertext program with the existing SEAL
CPU runtime, decrypts and compares against the independent PyTorch reference.
On candidate failure it returns diagnostics to DeepSeek for up to three repairs.
There is no rule-translator fallback when the model service fails.

The rule translator still prepares the fixed public constant registry and
packing layout. Its answer is kept locally for comparison, never sent to the
model in live mode. This is constrained synthesis, not independent layout search.

## Run

One-time setup: edit `D:\Code Space\Poseidon\.env` (WSL:
`/mnt/d/Code Space/Poseidon/.env`) and fill in the selected provider key.
All three provider keys belong in this same file; unused providers may be empty.
The existing DeepSeek key was preserved by renaming the old file.
`.env.example` is the empty template. Never send a real key or `.env` in chat.
The real file is ignored by Git; the example contains no key.

Every later run, including in a new terminal, needs only:

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-subneg-linear.json \
  --live --loopback-proxy-port 6478
```

An existing nonempty `DEEPSEEK_API_KEY` environment variable takes precedence
over the file; unset a stale variable to use the saved file. Only the fixed
project-root file is read, never parent-directory dotenv files. The parser accepts
one literal assignment per registered provider with optional quotes, blank lines
and comments. Unused provider keys may be empty. Old provider-specific files are
not read. Duplicate and unknown assignments are rejected. The parser does
not execute shell commands, interpolate variables or load arbitrary settings.
Offline modes never read the file. Missing/invalid configuration fails before Nix
or any API call and never echoes the file content.

This is a plaintext local credential file, not an encrypted secret store. Its
protection on the Windows-mounted source drive depends on Windows file ACLs;
Git ignore is not access control. Do not include it in shared folders, backups
sent to others or source archives. The generated-program sandbox does not mount
the project root or this file. No API key is copied to experiment evidence.

`--deepseek` performs real service calls. Only that flag retains the resolved key
across the existing pure Nix shell. The inner process removes it from its
environment before compiler/runtime child processes; the HTTPS worker receives
it via stdin. No global Codex, Git, network or system configuration is changed.

CLI defaults (2026-09-11): provider `deepseek`, model `deepseek-flash`, thinking/high,
384000 maximum generated tokens per request, 1200-second API deadline, three repairs.
Batch candidate/API concurrency defaults to 10; native execution stays limited to 2. Options
include `--model deepseek-v4-pro`, `--max-repairs 0`, `--max-tokens 4096`,
`--api-timeout 120`. `--live` is the preferred alias for `--deepseek`.
There is no additional interactive per-case approval prompt. These bounds limit
calls, not an exact currency charge; actual usage is recorded when available.

The explicit current configuration, without changing numerical criteria, is:

```bash
python3 scripts/baseline/run_candidate.py \
  --case scripts/baseline/cases/custom-subneg-linear.json --live --provider deepseek \
  --model deepseek-flash --reasoning-effort high --max-tokens 384000 --api-timeout 1200 \
  --loopback-proxy-port 6478
```

Local upper bounds are 384000 generated tokens and 1200 seconds per API request.
The hard process deadline includes the larger budget for up to three repairs.
The bounded HTTP response envelope allows 32 MiB, including reasoning; the final
candidate JSON/AST limits are unchanged. Reasoning text is not kept in evidence.
These settings can increase actual billed usage and do not guarantee correctness.
The API supports low effort; see the [official thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/).

To retest only failures, use a completed report as the selection source. Changing
generation settings requires the explicit `--allow-config-change` flag and is
recorded as a new experiment, never an overwrite of the old report:

```bash
python3 scripts/baseline/run_agent_batch.py --deepseek --jobs 2 \
  --failed-from /home/lhy/poseidon-work/results/agent-batch-dznntui3/report.json \
  --allow-config-change --reasoning-effort low --max-tokens 65536 --api-timeout 900
```

This historical example selects the 13 remaining failures in that report; add
`--model deepseek-v4-pro` to retain the historical retest model. It makes fresh
paid API calls; do not rerun it merely to read existing results. Reports may form
a hash-checked lineage of failure subsets. Original prompts, references, weights,
inputs, runtime and tolerances are audited separately from declared generation
configuration changes. Default operation still does not retry failed service calls.

New evidence is written under `/home/lhy/poseidon-work/results/agent-deepseek-*`:
`report.json`, generated `attempt-*/candidate.py`, compiler artifacts, logs,
decrypted outputs and numerical comparison. Never share `private-keys` or the
entire results directory. API failure stops with provider diagnostics rather
than inventing a program or counting it as successful execution.

## What a model description means today

For new work, use a self-contained **schema-2/3 data graph**: provide public
weights, input names/shapes and operator connections, rather than choosing a
predefined model number. See [the custom graph guide](custom-graphs-and-semantic-coverage.md),
`scripts/baseline/cases/custom-subneg-linear.json` and the
[96-case self-contained suite](self-contained-model-suite.md).
The single-case entry requires `--case`; the batch entry accepts the same graph
objects in `--case-manifest`. The model id is only a label.

The historical schema-1 format remains available solely for compatibility and
reproduction of the old catalog experiments, for example:

```json
{"schema":1,"id":"linear-1","family":"linear","configuration":1}
```

Historical catalog families are `affine`, `polynomial`, `linear`, `mlp2`, `mlp3`, `fanout`,
`residual`, `flatten_linear`, each with configurations 0..5: 48 small cases.
`id` must be `<family>-<configuration>`. These models and their weights are built
locally from the catalog. They are tests for the Agent, not training data or a
newly trained LLM. Reusing weights/inputs lets different generated programs be
compared fairly against the same reference.

The new graph format is not limited to those catalog entries. Nodes specify
connections, operations and public arrays directly; they do not select a family.
The encrypted ABI still restricts each input to four logical elements and final
outputs to 1..4. Linear hidden layers now permit 1..8 scalar-neuron ciphertexts;
see [wide Linear/MLP verification](wide-linear-cpu.md). General `model.py`, ONNX, arbitrary shapes and private checkpoints
remain unsupported. ReLU/SiLU are not silently approximated. Current live user-graph
evidence is listed above; manual/rule validation of a newly exported fixture does
not by itself establish an additional live Agent success.

DeepSeek needs the public computational structure/constants and the DSL rules to
write an equivalent program. It does not receive the hidden test inputs,
reference answers or encryption keys. Encryption, execution and comparison happen
locally. This is an API-based code-generation Agent, not encrypted cloud inference.

## Verified integration, 2026-09-06

The real entry was exercised with explicit offline HTTP response fixtures:

```bash
python3 scripts/baseline/agent_entry_smoke.py
```

Evidence: `/home/lhy/poseidon-work/results/agent-deepseek-6vd8ecry/report.json`.
The sequence was invalid JSON -> wrong Linear reduction -> correct saved baseline
program. The latter two candidates each ran real tracing, compilation and
encrypted execution; the wrong result was rejected and the correct result passed.
The run records `provider=deepseek_offline`, `agent_calls=0` and
`llm_generation_validated=false`. These fixtures do not establish that DeepSeek
can generate or repair those programs.

The test also checked pure-shell key passthrough with a fake sentinel, removal
before native child processes, and 146 existing trusted CA certificates. The
first run exposed an empty default CA store in pinned Nix Python; the HTTPS
worker now falls back to the existing WSL system CA bundle with certificate and
hostname verification still enabled. No package or certificate was installed.

Final host regression after CLI integration: 137 discovered, 128 passed and 9
Torch-only tests skipped outside the pinned environment. This is separate from
the actual encrypted integration run above and from prior full-catalog results.

The integration checks above describe the earlier offline-fixture milestone.
Subsequent live API experiments are recorded in
[the 48-case report](deepseek-48-case-results-2026-09-06.md) and
[the independent failure retest](deepseek-failure-retest-2026-09-06.md).
The [expanded-budget retest](deepseek-expanded-budget-retest-2026-09-06.md)
subsequently passed all 13 remaining failures (11 first-pass, two after one real
static-check feedback repair each), on the same SEAL CPU execution backend.
Those live results supersede the earlier uncertainty about account connectivity
and real candidate generation. Poseidon single-GPU execution remains unverified.
No bootstrap, GPU simulation or decrypt-and-reencrypt substitute was introduced.

## Provider profiles and current architecture

`--provider deepseek` is the only supported provider. Its fixed official endpoint
uses `DEEPSEEK_API_KEY` from the backend project-root `.env`.
See [provider configuration](provider-subscriptions.md) and
[startup commands](../../scripts/README.md). Subscription adapters were removed;
historical subscription experiment reports remain evidence, not supported launch instructions.

See [the detailed current architecture](project-and-agent-architecture-2026-09-07.md)
for the input boundary, Hecate contract, FHE concepts, compiler/runtime layers,
Agent changes and remaining Poseidon GPU gate.

The fresh [Flash/high full 48-case report](deepseek-flash-48-case-results-2026-09-07.md)
records 45/48 first-pass and 48/48 final success, with 52 live requests and a passed
evidence audit. This does not establish live subscription-account compatibility
or Poseidon GPU execution.

## Opt-in lexical closure construction

`run_candidate.py --closures` selects the new request-v9 / AST-v8 grammar, including
nested helpers, late-bound captures and nonlocal writes. The flag works with
prepare, manual golden, replay and live modes; it does not change the default
or automatically authorize any paid call. See the
[closure semantic rules and real CPU goldens](closure-construction-cpu.md).
Manual encrypted validation is complete for those fixtures; online Agent
generation of the newly added closure grammar has not yet been measured.

`--call-binding` opts into request-v10 / AST-v9, adding definition-time defaults,
positional-only/keyword-only parameters, keywords and bounded *args/**kwargs
forwarding. It includes closures but does not alter the golden encrypted-input
ABI or enable arbitrary Python execution. See
[parameter-binding rules and CPU evidence](call-binding-cpu.md). This syntax
extension has manual encrypted validation, not a measured online Agent cohort.

`--public-iteration` selects request-v11 / AST-v10, including the preceding
call-binding grammar plus scoped list/dict comprehensions and lazy public
enumerate/zip/reversed/iter/next/list/tuple. It never iterates ciphertext slots
or permits encrypted filter decisions. See
[public iteration semantics and actual CPU evidence](public-iteration-cpu.md).
Online generation of this new grammar remains a separate validation gate.

`--function-literals` selects request-v12 / AST-v11, adding lexical lambda
values, calls through returned/container-selected functions, and stable sorted
with public key functions. Sorting ciphertext/opaque plaintext keys is rejected;
no runtime function-call opcode or secret-dependent ordering is introduced.
See [lambda/sorting semantics and actual CPU results](function-literals-cpu.md).

`--public-sequences` selects request-v13 / AST-v12, including previous function
semantics and public list/tuple/string slicing, concatenation, repetition and
list slice writes. This does not allow ciphertext-slot indexing or array slicing
inside named constants. See [public sequence semantics and actual CPU evidence](public-sequences-cpu.md).
Manual goldens do not establish online Agent generation success for the new grammar.

`--public-numbers` selects request-v14 / AST-v13 for checked public real/array
arithmetic and separately hashed derived constants. The batch runner accepts
the same flag; changing DSL contract requires a fresh batch. No implicit
multidimensional flatten, cipher division/power or array writes are introduced.
See [numeric semantics, actual CPU evidence and paid-launch status](public-numbers-cpu.md).

`--public-control` selects request-v15 / AST-v14, adding bounded public while,
for/while-else, break/continue/pass, short-circuit truth and immutable public
mapping keys. It includes numeric construction but does not permit encrypted
conditions or ciphertext comparisons. Both single-case and batch runners accept
the flag; use a fresh batch when changing grammar. See
[public control rules and actual CPU evidence](public-control-cpu.md).
Manual goldens passed; online Agent generation under this contract is unmeasured.

`--public-strings` selects request-v16 / AST-v15 for public coefficient/tree
parsing: strip/split families, partition, replace and join with pinned method
signatures and bounded results. It includes public control and numeric
construction; no string-to-code execution or secret string conversion.
Both candidate and batch runners accept it. See
[string semantics and actual CPU results](public-strings-cpu.md).
Manual encrypted validation is not a measured online Agent generation rate.

`--public-polynomial` selects request-v17 / AST-v16 for checked public
Chebyshev coefficient data/arithmetic, dtype markers and floor/ceil/log2.
It enables measured uses of the actual upstream GenPoly construction closure.
GenPoly's leaf evaluation is odd-only: even/constant terms need a valid explicit
rewrite, not an assumption of generic polynomial support. Use the existing
pinned NumPy/Nix environment, not system Python. See
[GenPoly semantics, actual CPU evidence and remaining boundaries](public-polynomial-cpu.md).

`--object-arrays` selects request-v18 / AST-v17 for bounded symbolic object
arrays, checked view/copy/index writes and actual Hecate Empty semantics. Both
single-case and batch runners accept it; a contract change needs a fresh batch.
These arrays store ciphertext expressions, not slots inside one ciphertext.
Do not interpret Empty as zero or container reshape as ciphertext repacking.
Actual upstream SumSlots and object-array Linear manual CPU goldens passed;
online Agent generation success is not yet measured for this grammar. See
[object-array CPU results and precision caveat](object-array-cpu.md).
