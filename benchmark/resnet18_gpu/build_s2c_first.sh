#!/usr/bin/env bash
set -euo pipefail

app_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$app_dir/../.." && pwd)
shared_dir="$repo_dir/benchmark/resnet50_gpu"
obj_root="$repo_dir/src/poseidon/tests/bootstrapping/build/CMakeFiles/test_gpu_bootstrap_modraise.dir"
obj_dir=$(find "$obj_root" -type d -path '*/src/poseidon/gpu' -print -quit 2>/dev/null || true)
output=${1:-/tmp/poseidon_gpu_resnet18}

if [[ -z "$obj_dir" ]]; then
  echo "missing GPU build objects below: $obj_root" >&2
  exit 2
fi

cd "$repo_dir"
c++ -O1 -std=c++20 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL \
  -DPOSEIDON_GPU_RESNET18_SOURCE_DIR=\"$app_dir\" \
  -DPOSEIDON_GPU_RESNET50_SOURCE_DIR=\"$shared_dir\" \
  -DPOSEIDON_RESNET20_RELU_FIXTURE=\"$repo_dir/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt\" \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
  -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR \
  -Isrc -Ithird_party/rmm/include \
  -Ithird_party/cccl/libcudacxx/include -Ithird_party/cccl/cub \
  -Ithird_party/cccl/thrust -Ithird_party/spdlog/include \
  -Ithird_party/fmt/include -Ithird_party/nvtx/c/include \
  -I/usr/local/cuda/include \
  -I"$app_dir" -I"$shared_dir" \
  "$app_dir/main.cpp" \
  "$app_dir/gpu_config.cpp" \
  "$app_dir/gpu_resnet18_inference.cpp" \
  "$app_dir/resnet18_topology.cpp" \
  "$app_dir/resnet18_weights.cpp" \
  "$shared_dir/gpu_ckks_runtime.cpp" \
  "$shared_dir/gpu_multiplexed_tensor.cpp" \
  "$shared_dir/gpu_relu_s2c_first.cpp" \
  "$shared_dir/resnet50_config.cpp" \
  "$obj_dir"/*.cpp.o "$obj_dir"/*.cu.o "$obj_dir"/kernels/*.cu.o \
  -Lsrc/poseidon/tests/bootstrapping/build/poseidon_build -lposeidon_shared \
  -L/usr/local/cuda/lib64 -lcudart -ldl -lpthread \
  -Wl,-rpath,"$repo_dir/src/poseidon/tests/bootstrapping/build/poseidon_build" \
  -Wl,-rpath,/usr/local/cuda/lib64 -o "$output"

printf '%s\n' "$output"
