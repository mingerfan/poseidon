# GPU key-load investigation (2026-11-15)

后续首次生成优化已经实现并实测，见 [`../../docs/gpu-qp-first-start-fix-20261115.md`](../../docs/gpu-qp-first-start-fix-20261115.md)。下面的调查报告/证据保留原始基线，不代表当前代码仍只有计时插桩。

中文报告：[`../../docs/gpu-keyload-analysis-20261115.md`](../../docs/gpu-keyload-analysis-20261115.md)。

原始证据：`../../test-results/gpu-keyload-20261115/`（仓库默认忽略测试产物，文件保留本地）。不含私钥、evaluation-key payload、明文数据文件或SSH凭据；`public-resnet20-primes.txt` 只有公开模数。

## Entry points

- Poseidon standalone CMake probe: `../../bench/gpu_keyload/README.md`.
- Cerium public-API probe: `../../bench/cerium_keyload/README.md`.
- `native_profile.cpp`: real Q50/P25 manual bootstrap **preparation**, no Boot execution. Uses public historical primes, generates a fresh synthetic secret in memory.
- `build_native.sh ROOT`: links native probe against `ROOT/build` and bundled headers.
- `build_existing_probe.py ROOT`: reuses the already built library's exact Ninja compiler flags, records its actual CMake options; does not reconfigure/rebuild production libraries.
- `run_matrix.py`: fresh process, exact argv/env/quota/GPU-state capture, exclusive local probe lock, no overwrite of existing cases. It is not a cross-user/global GPU lock.
- `run_poseidon.sh ROOT` / `run_cerium.sh ROOT`: the measured sequential matrix. Fixed case names deliberately refuse rerunning into the same results directory.
- `audit_inventory.py CERIUM_PROGRAM_INPUTS`: static inventory and parameter-root formulas, no GPU.
- `summarize.py RESULTS`: reproduce JSON summary from existing timing/check reports, no GPU.

Server experiment root was `/root/private_data/poseidon-keyload-20261115`. Dependencies were installed/built under its own `tools/`, not system-wide: CMake 4.4.5, Ninja 1.13.2, GMP 6.3.0. Poseidon Release/SM80 uses CUDA 11.8 and **POSEIDON_USE_OPENMP=OFF**; root CMake nevertheless has global `-fopenmp`. Cerium links the pre-existing public runtime with CUDA runtime 12.2, using stable CUDA host API headers from the installed 11.8 toolkit. Exact compile/link commands and library hashes are in the evidence directory.

Build baseline Poseidon GPU library (assuming these tools already exist):

```bash
D=/root/private_data/poseidon-keyload-20261115
"$D/tools/cmake/data/bin/cmake" -S "$D/src" -B "$D/build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$D/tools/bin/ninja" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DGMP_ROOT="$D/tools/gmp" -DCMAKE_BUILD_TYPE=Release \
  -DPOSEIDON_BUILD_DEPS=ON -DPOSEIDON_USE_SPDLOG=ON \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$D/src/third_party/spdlog" \
  -DPOSEIDON_USE_MSGSL=OFF -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF \
  -DPOSEIDON_USE_OPENMP=OFF -DPOSEIDON_BUILD_EXAMPLES=OFF \
  -DPOSEIDON_BUILD_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
"$D/tools/cmake/data/bin/cmake" --build "$D/build" -j6
python3 "$D/src/experiments/gpu_keyload/build_existing_probe.py" "$D"
bash "$D/src/experiments/gpu_keyload/build_native.sh" "$D"
```

The initial investigation added `POSEIDON_GPU_KEYLOAD_TRACE=1` timers only. The subsequent first-start fix adds verified exact small-CRT reconstruction, canonical Q retention and bounded CPU parallel preparation; see the follow-up report above. Native sample one had Boot-phase/key-upload tracing; sample two added QP subphase tracing. Ordinary probe binary was linked before the latter tracing-only change. Final binary hashes do not retroactively attest earlier overwritten executables.

All performance claims and limitations are in the Chinese report. In particular, the main 195-second cost is CPU exact Q→QP plaintext preparation, not PCIe key transfer. Sparse Q-level parameter construction is a validated existing-API proof of concept, **not** a complete production Boot equivalence test. Disk-cache/direct-QP-encoder solutions remain proposed work; the new first-start speedup instead uses an all-Q-verified exact reconstruction shortcut and CPU parallelism.
