#!/usr/bin/env bash
set -euo pipefail
base=/home/xuming/poseidon-luhang-merge-cpu-20261014
parent=19d022428ab07b68c56ba20e32d60f317d836b7c
trap 'echo "$?" > "$base/logs/parent-probe.status"' EXIT
mkdir -p "$base/parent-src" "$base/parent-deps"
git -C "$base" archive "$parent" CMakeLists.txt cmake src | tar -x -C "$base/parent-src"
echo "$parent" > "$base/logs/parent-commit.txt"
for dep in zlib zstd msgsl; do
    mkdir -p "$base/parent-deps/$dep-src"
    git -C "/home/xuming/poseidon/build-runtime-gpu-rmm2412/thirdparty/$dep-src" archive HEAD |
        tar -x -C "$base/parent-deps/$dep-src"
done
cmake -S "$base/parent-src" -B "$base/build-parent-cpu" \
    -DCMAKE_BUILD_TYPE=Release -DPOSEIDON_BUILD_EXAMPLES=OFF \
    -DPOSEIDON_BUILD_GPU=OFF -DPOSEIDON_BUILD_CKKS_RUNTIME_API=OFF \
    -DPOSEIDON_USE_HARDWARE=OFF \
    -DFETCHCONTENT_SOURCE_DIR_ZLIB="$base/parent-deps/zlib-src" \
    -DFETCHCONTENT_SOURCE_DIR_ZSTD="$base/parent-deps/zstd-src" \
    -DFETCHCONTENT_SOURCE_DIR_MSGSL="$base/parent-deps/msgsl-src" \
    > "$base/logs/parent-configure.log" 2>&1
cmake --build "$base/build-parent-cpu" --target poseidon_shared --parallel 2 \
    > "$base/logs/parent-build.log" 2>&1
c++ -std=c++20 -O3 -DNDEBUG -fopenmp -pthread \
    -I"$base/parent-src/src" -I"$base/build-parent-cpu" \
    -I"$base/parent-deps/msgsl-src/include" "$base/bgv_keyswitch_probe.cpp" \
    -L"$base/build-parent-cpu" -lposeidon_shared -lgmpxx -lgmp \
    -Wl,-rpath,"$base/build-parent-cpu" -o "$base/build-parent-cpu/bgv_keyswitch_probe" \
    > "$base/logs/parent-bgv-probe-build.log" 2>&1
set +e
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' timeout 120s \
    "$base/build-parent-cpu/bgv_keyswitch_probe" > "$base/logs/parent-bgv-probe.log" 2>&1
code=$?
set -e
echo "$code" > "$base/logs/parent-bgv-probe.status"
cat "$base/logs/parent-bgv-probe.log"
