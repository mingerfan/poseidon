# RuntimePlan V4 Waterline

V4 adds `prepare_plaintext` to the existing typed plan. Its record has exactly
`ordinal`, `kind`, `payload` and `output`. Payload uses the existing inline/bundle
representation. The output descriptor is a single-component Device NTT plaintext.
The backend is selected once at complete consumer-group admission. Ordinary
Encode, Transfer, Release, reuse and explicit Fence retain their existing meaning.
Prepare belongs to execution. It requires one rank and the per-device executor.

The required plan-level `waterline` object has:

- `device_capacity_bytes`, `device_reserve_bytes`, `device_fractions`: one entry
  per local GPU. Fractions must be strictly between zero and one. Reserve covers
  keys, tables, allocator overhead and safety space outside dynamic object charges.
- `host_budget_bytes`, `pinned_budget_bytes`, `raw_budget_bytes`: shared budgets.
- `backend`: `cpu`, `gpu` or `auto`; `lookahead_groups`: positive consumer limit.

Pinned staging also counts toward Host usage. CPU-encoded plaintext and its
separate pinned upload source remain alive together, so Host admission reserves
both; the pinned budget separately constrains the staging source.

Dependencies come from existing ValueIds, producer records and operand lists.
A complete group reserves new preparation peaks, output storage, communication
staging and backend compute workspace together. Admission follows a topological
prefix. Release returns credit after the last submitted use actually completes.
Consumers use the result's real Ready dependency. Admitted foreground operations
are queued in batches. A worker continuously submits its queue without waiting
for a dispatcher acknowledgement after each operation; dependencies are resolved
by the existing ValueId waits. Completion tickets keep output storage,
input-use counts and temporary credit until the actual Ready events finish.
The lookahead and byte budgets bound admitted work, including submitted work.
A released output cannot return storage credit while its producer is unfinished,
even if it has no consumers. Host submissions notify the dispatcher immediately;
timed waits for GPU events are used only when no further submission can advance.
The single CPU worker runs ready Host work or admitted CPU preparations; it never owns a wait that
prevents a queued producer from running. GPU preparation has one active complete
Encode per device, retaining exclusive use of its staging buffer through Ready.
Its worker publishes the asynchronous result, waits on that result's Ready,
then continues the next admitted preparation in its queue. This wait neither
blocks the dispatcher nor synchronizes unrelated foreground work. Its independent
lane can overlap foreground submissions and GPU execution. Temporary credits are
returned only after the dispatcher reaps the completed results.
Auto prefers a supported GPU preparation. Temporary budget pressure waits for
admitted groups to complete, instead of choosing CPU because a worker is busy.
If the GPU path is unsupported, or a complete GPU group still exceeds the
available budgets after admitted groups have completed, Auto considers CPU.
Preflight checks the same GPU-first complete-group policy. Explicit CPU/GPU
choices remain fixed, and no preparation is retried or switched after submission.
Resident plans bypass Waterline admission metadata and the dispatcher entirely:
each nonempty device/CPU queue runs continuously, with the existing communication
issuer policy and ValueId dependencies. Explicit Fence and final drain still wait
for actual completion.

If a local backend aborts by throwing, runtime drains already submitted work,
then clears its values and parallel state before propagating the original error.
It does not retry or switch backend after failure. Distributed backends abort
the job before cleanup so draining cannot wait for a failed peer.

CKKSPL01 retains its wire container version. Instruction tag 5 is PreparePlaintext;
its record body matches tag 0 Encode. Waterline configuration lives in container
metadata. Older plan versions reject tag 5. JSON and binary readers, writer,
verifier and printer support the same instruction. Invalid or unsupported plans
fail explicitly. An API without the preparation/resource-query contract rejects
Waterline before execution.

The compiler checks a sequential single-group baseline using physical types and
the requested workspace upper bound. The executor repeats it using backend
resource queries before online submission. A zero compiler workspace option is
an object-only precheck; it does not certify total GPU memory. The reserve must
cover persistent backend caches as well as keys/tables. GPU memory pools retain
free blocks; logical credit is distinct from driver-visible reserved memory.
An optional backend `waterline_allocator_capacity()` query, read after resource
validation, further limits each device's foreground plus plaintext bytes in both
the baseline and rolling admission without adding an IR field; an empty vector
adds no extra limit.
An optional backend `waterline_fixed_resources()` query reserves reusable
Host/pinned preparation buffers for the whole execution, including the baseline
check and explicit Fence boundaries. These buffers remain charged while idle;
their bytes must not also appear in each preparation's dynamic peak. Drain
waits for their readers before releasing the physical buffers.
Resident packed bundle bytes count toward the shared Host budget.

The current Poseidon GPU implementation encodes real slots with FP64 cuFFT,
rounding, uint32 RNS expansion and CUDA NTT. It pads short vectors with zeros,
uses the descriptor's level/scale and target device, keeps pinned/raw/workspace
alive through Ready, and rejects non-finite or out-of-range coefficients.
GPU FFT rounding need not match CPU coefficients bit for bit. Native Boot's
Waterline workspace query is currently unsupported; ordinary explicit native
Boot remains available. CPU decrypt/re-encrypt Boot retains its explicit profile
semantics and secret-key requirement.

Typed Rescale keeps its existing `target_level` and `target_scale_log2` fields.
The physical-level compiler requires at least 30 more nominal dropped modulus
bits than the logical scale reduction, checks the OperatorSpec Rescale drop
limit, and requires the target scale to fit strictly below the remaining nominal
modulus bits. For a 120-bit logical reduction, five 30-bit physical levels supply
that alignment space; four do not. These checks certify available space, not
actual-prime precision or a noise margin. The backend selects native rescalings
using actual primes and an integer coefficient multiplier. It applies that
multiplier to all input ciphertext coefficients and the input scale **before**
native rescaling, so rounding occurs at approximately the target scale. Applying
the multiplier after division can amplify rounding from a much lower scale.
The Poseidon backend requires a target scale of at least 2^40, a finite positive
intermediate scale, and integer-alignment relative error at most 1e-9; unsupported
drops or insufficient alignment space fail explicitly. Any remaining physical
level gap uses ModSwitch. This preserves the plan's lazy rescale boundaries and
target level/scale. Merely overwriting scale metadata after native rescaling
changes the represented value and is invalid.
