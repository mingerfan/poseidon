#!/usr/bin/env bash
# Compatibility build for the exact previously validated standalone GPU tree.
# The normal source build is build.sh; this mode intentionally requires its
# original kernel-object cache, rather than silently mixing library versions.
set -euo pipefail
package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${POSEIDON_VALIDATED_ROOT:?Path to the validated Poseidon GPU source/build tree}
root=$(cd -- "$root" && pwd)
native="$package/runtime/native_snapshot"
app="$native/benchmark/resnet20_gpu"
obj="$root/src/poseidon/tests/bootstrapping/build/CMakeFiles/test_gpu_bootstrap_modraise.dir$root/src/poseidon/gpu"
output=${1:-"$package/build/bin/resnet20_dsl"}
mkdir -p "$(dirname -- "$output")"
cd "$root"
c++ -O1 -std=c++20 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL \
  -DPOSEIDON_GPU_RESNET20_SOURCE_DIR=\"${POSEIDON_DSL_ASSET_SUPPORT:-$package/../../../bench/resnet20_s2c_first/support}\" \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR \
  -I"$root/src" -I"$root/third_party/rmm/include" \
  -I"$root/third_party/cccl/libcudacxx/include" -I"$root/third_party/cccl/cub" -I"$root/third_party/cccl/thrust" \
  -I"$root/third_party/spdlog/include" -I"$root/third_party/fmt/include" -I"$root/third_party/nvtx/c/include" \
  -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include -I"$app" \
  -I"$native/Doc/analysis/resnet20_joint_chain_20260911" \
  "$package/runtime/main_strict_native_dsl.cpp" \
  "$native/bench/resnet20_s2c_first/plain_resnet20_reference.cpp" \
  "$native/bench/resnet20_s2c_first/gpu_activity_timing.cpp" "$app/resnet20_weights.cpp" \
  "$obj"/*.cpp.o "$obj"/*.cu.o "$obj"/kernels/*.cu.o \
  -L"$root/src/poseidon/tests/bootstrapping/build/poseidon_build" -lposeidon_shared \
  -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64 -lcudart -lcupti -ldl -lpthread \
  -Wl,-rpath,"$root/src/poseidon/tests/bootstrapping/build/poseidon_build" \
  -Wl,-rpath,/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/extras/CUPTI/lib64 -o "$output"
printf '%s\n' "$output"
