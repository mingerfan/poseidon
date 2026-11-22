# NEW node2 core build — PASS

Compilation/provenance evidence only. Exactly one synchronous SSH operation and one frozen builder invocation occurred; there was no retry, repair, restaging, CPU application, GPU application, Boot, or private-payload execution. No numerical/model accuracy, latency, performance, or GPU-acceptance claim is made.

## Execution

- UTC: 2026-11-16 12:39:36–12:44:00.
- Node: `crdnotebook-2121267849739771906-hpuict-12953`.
- Root: `/root/private_data/llama-manual-framework-node2-20261116-v1`.
- Fresh admission: exact `GPU-499236d3-b46c-5d25-4a22-530d47ed5112`, 0 MiB, 29°C; the frozen builder also required no compute processes.
- Existing authentication was reused without reauthentication or master creation/closure. No credentials or locator contents were printed; SSH stderr was empty.
- Remote verification checked canonical paths, exact local/remote sizes and SHA256 for the authorization and four artifacts before starting the builder. No prior build outputs existed.

```text
taskset -c 6 /opt/conda/bin/python3 -I -B -u /root/private_data/llama-manual-framework-node2-20261116-v1/tools/build_node2_core.py
```

| Command | Actual return code | Duration (seconds) |
|---|---:|---:|
| SSH | 0 | 264.764 |
| Frozen builder | 0 | 264.204 |
| configure-core | 0 | 16.144 |
| compile-core | 0 | 244.708 |

The builder enforced CPU6, OMP/BLAS/MKL1, hidden CUDA devices, one build job, and its 900-second work deadline. The outer SSH timeout was 960 seconds and did not fire. All 7103 source hashes were checked by the frozen builder before and after compilation.

## Verified package pins

| Role | SHA256 |
|---|---|
| Actual build-authorization.json | `f88be3dffec92c6a26a1c9ca5896863da9c741844238e5f0f9e9ab775f2573eb` |
| tools/build_node2_core.py | `36b0300f24bc7bd00c2cb656f22d7e4c7f69e6d1d0dda5a37ab5c5ecb4eccc46` |
| tools/owned_cpu_group.py | `6e6f69c5d93daeef20c17b0f7dabde856d0cf9755708a9e140e4c0a35a786668` |
| source.tar.gz | `864924b6175881c1e689030d4949f44a6d051c66345306c2ba2c2924964255e5` |
| source-sha256.json | `d3b0c1a488c4220769a1b02f6c50a353d889b2bdf7cf1ac0ed88577186aa820c` |

Authoritative local package: `../llama-cerium-align/experiments/20261116-manual-llama/node2-v1/core-package-v1/`. GPU execution authorization remained false.

## Actual build scope and outputs

CMake **3.31.6**, NVIDIA NVCC **11.8.89** (`/usr/local/cuda/bin/nvcc`), GNU host compiler **11.4.0**, Release/C++17, explicit **compute_80/sm_80** generation. CUDA source compilation was authorized; CUDA/GPU application execution was not performed. Target selection was exclusively `--target poseidon_gpu --parallel 1`, which also built its CPU-library dependency `poseidon_shared`.

The compilation database contained 85 entries, including six CUDA entries and no manual application entry. The selected target's build log contains 82 object-compilation lines: 76 CXX and six CUDA. Database entry count is not an executed-compilation count. Logs show the GPU evaluator, uploader, double-hoist implementation and CUDA kernels compiling. The four ported source/header hashes are recorded in `validation.json`. Manual application, examples, runtime integration, tests, dependency builds and benches remained disabled; dependency resolution was local-only/disconnected. No fast-math flag was enabled.

Outputs remain remotely in the authorized root; no library/object was copied locally:

| Remote path relative to root | Bytes | SHA256 |
|---|---:|---|
| build/libposeidon_gpu.a | 11493838 | `ba79a5a583ab06dea637d1cefe67427195dd5a8398449ce61c1544e7f37f7adb` |
| build/libposeidon_shared.so | 2545616 | `bd86927386dc3bfe8a3a50104b6b200b97cf1a137cd201aeafbdc0711d0cb688` |

The frozen builder's hashes agree with independent post-exit read-only hashing in the same SSH operation.

## Warnings and cleanup

Warnings were not suppressed, filtered out of the saved logs, or treated as errors:

- Configure: three CMake developer-warning blocks — GNUInstallDirs before language/architecture setup, plus deprecated FetchContent_Populate calls for CCCL and nvtx3.
- Compile: 22 `-Wterminate` warnings at `polynomial_evaluation.h:70/76` (throw in noexcept), plus ten NVCC `#815-D` warnings at `util/exception.h:23` (meaningless return-type qualifier).

Both attested owned CPU sessions reported no remaining live members and no forced kills. After the builder exited, read-only `/proc/locks` inspection found no held locks matching the eight existing leases or the owned host-build lock. All eight existing lease files retained their inode/owner/mode identities. The builder process was absent; the unheld host-build lock file was preserved, not removed or reacquired. No unrelated process cleanup claim is made.

## Local reporting-audit discrepancy

After the successful remote build, one local reporting-only assertion exited **1**: it incorrectly expected 79 CXX object lines by equating the 85-entry compilation database with selected-target execution. Actual logged counts were 76 CXX plus six CUDA. The exact reason for the three-entry difference was not investigated. Execution stopped without retry or remote follow-up; only aggregate-report labels were corrected. All SSH/builder/configure/compile return codes above remain 0. The failed audit is preserved in `post-build-audit.json`; its planned report manifest was never created.

## Evidence files

- `invocation.json`: scope, exact builder argv, sanitized SSH template, package pins, deadlines and timestamps.
- `ssh-status.json`, `ssh.stdout.log`, `ssh.stderr.log`: real SSH result and captured streams. The stdout transcript includes a base64 transport envelope containing only these public logs/result; no binary build outputs.
- `core-build-result.json`: byte-identical remote result, command argv/status/durations, session identities, cleanup and output hashes.
- `configure-core.{stdout,stderr}.log`, `compile-core.{stdout,stderr}.log`: byte-identical remote logs, with transport hashes verified locally.
- `remote-evidence.json`: remote pin checks, output metadata/hashes, selected CMake/cache/compiler metadata, compilation-database aggregate and post-exit resource snapshot.
- `aggregate-result.json`: combined SSH/build evidence.
- `validation.json`: local assertions over collected build evidence, warning counts and ported-file provenance; distinguishes database entries from observed object builds.
- `post-build-audit.json`: failed reporting-only count assumption, actual observed counts and explicit no-retry disposition.

Only this new exclusive local report directory was written locally. This task did not edit implementation, scripts, authentication, dependencies, system settings, old roots or existing lease files. The local report contains only public logs/provenance and aggregate hashes, not objects, libraries, keys or tensors.
