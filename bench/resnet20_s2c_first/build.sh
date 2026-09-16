#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/liufuyao/Work/poseidon_gpu
app_dir=/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu
source_dir="$repo_dir/Doc/analysis/resnet20_joint_chain_20260911"
obj_dir="$repo_dir/src/poseidon/tests/bootstrapping/build/CMakeFiles/test_gpu_bootstrap_modraise.dir/home/liufuyao/Work/poseidon_gpu/src/poseidon/gpu"
output=${1:-/tmp/poseidon_resnet20_s2c_first}

cd "$repo_dir"
c++ -O1 -std=c++20 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL \
  -DPOSEIDON_GPU_RESNET20_SOURCE_DIR=\"$app_dir\" \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DTHRUST_DISABLE_ABI_NAMESPACE \
  -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -Isrc -Ithird_party/rmm/include \
  -Ithird_party/cccl/libcudacxx/include -Ithird_party/cccl/cub -Ithird_party/cccl/thrust \
  -Ithird_party/spdlog/include -Ithird_party/fmt/include -Ithird_party/nvtx/c/include \
  -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include \
  -I"$app_dir" -I"$source_dir" \
  bench/resnet20_s2c_first/main.cpp \
  bench/resnet20_s2c_first/plain_resnet20_reference.cpp \
  bench/resnet20_s2c_first/gpu_activity_timing.cpp \
  "$app_dir/resnet20_weights.cpp" \
  "$obj_dir"/*.cpp.o "$obj_dir"/*.cu.o "$obj_dir"/kernels/*.cu.o \
  -Lsrc/poseidon/tests/bootstrapping/build/poseidon_build -lposeidon_shared \
  -L/usr/local/cuda/lib64 -L/usr/local/cuda/extras/CUPTI/lib64 \
  -lcudart -lcupti -ldl -lpthread \
  -Wl,-rpath,"$repo_dir/src/poseidon/tests/bootstrapping/build/poseidon_build" \
  -Wl,-rpath,/usr/local/cuda/lib64 \
  -Wl,-rpath,/usr/local/cuda/extras/CUPTI/lib64 -o "$output"

printf '%s\n' "$output"
