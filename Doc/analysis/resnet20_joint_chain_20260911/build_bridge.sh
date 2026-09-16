#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/liufuyao/Work/poseidon_gpu
cd "$repo_dir"
bridge_out=${1:?provide a temporary output binary path}
obj_dir=src/poseidon/tests/bootstrapping/build/CMakeFiles/test_gpu_bootstrap_modraise.dir/home/liufuyao/Work/poseidon_gpu/src/poseidon/gpu
c++ -O1 -std=c++20 -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DTHRUST_DISABLE_ABI_NAMESPACE \
  -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -Isrc -Ithird_party/rmm/include \
  -Ithird_party/cccl/libcudacxx/include -Ithird_party/cccl/cub -Ithird_party/cccl/thrust \
  -Ithird_party/spdlog/include -Ithird_party/fmt/include -Ithird_party/nvtx/c/include \
  -I/usr/local/cuda/include Doc/analysis/resnet20_joint_chain_20260911/bootstrap_bridge.cpp \
  "$obj_dir"/*.cpp.o "$obj_dir"/*.cu.o "$obj_dir"/kernels/*.cu.o \
  -Lsrc/poseidon/tests/bootstrapping/build/poseidon_build -lposeidon_shared \
  -L/usr/local/cuda/lib64 -lcudart -ldl -lpthread \
  -Wl,-rpath,"$repo_dir/src/poseidon/tests/bootstrapping/build/poseidon_build" \
  -Wl,-rpath,/usr/local/cuda/lib64 -o "$bridge_out"
