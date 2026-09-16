#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/liufuyao/Work/poseidon_gpu
cd "$repo_dir"
precision_out=${1:?provide a temporary output binary path}
precision_source=${2:-relu_precision.cpp}
case "$precision_source" in
  relu_precision.cpp|bootstrap_precision.cpp|bootstrap_baseline_precision.cpp|bootstrap_relu_precision.cpp|bootstrap_relu_conv_precision.cpp|bootstrap_block_precision.cpp|bootstrap_network_precision.cpp) ;;
  *) echo "unsupported precision source" >&2; exit 2 ;;
esac
app_dir=/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu
obj_dir=src/poseidon/tests/bootstrapping/build/CMakeFiles/test_gpu_bootstrap_modraise.dir/home/liufuyao/Work/poseidon_gpu/src/poseidon/gpu
precision_extra_sources=()
if [[ "$precision_source" == bootstrap_relu_conv_precision.cpp || "$precision_source" == bootstrap_block_precision.cpp || "$precision_source" == bootstrap_network_precision.cpp ]]; then
  precision_extra_sources+=(Doc/analysis/resnet20_joint_chain_20260911/original_conv_reference.cpp)
  precision_extra_sources+=("$app_dir/resnet20_weights.cpp")
fi
c++ -O1 -std=c++20 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -DFMT_HEADER_ONLY=1 -DSPDLOG_FMT_EXTERNAL \
  -DPOSEIDON_GPU_RESNET20_SOURCE_DIR=\""$app_dir"\" \
  -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DTHRUST_DISABLE_ABI_NAMESPACE \
  -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -Isrc -Ithird_party/rmm/include \
  -Ithird_party/cccl/libcudacxx/include -Ithird_party/cccl/cub -Ithird_party/cccl/thrust \
  -Ithird_party/spdlog/include -Ithird_party/fmt/include -Ithird_party/nvtx/c/include \
  -I/usr/local/cuda/include -I"$app_dir" \
  "Doc/analysis/resnet20_joint_chain_20260911/$precision_source" \
  Doc/analysis/resnet20_joint_chain_20260911/original_relu_reference.cpp \
  "${precision_extra_sources[@]}" \
  "$obj_dir"/*.cpp.o "$obj_dir"/*.cu.o "$obj_dir"/kernels/*.cu.o \
  -Lsrc/poseidon/tests/bootstrapping/build/poseidon_build -lposeidon_shared \
  -L/usr/local/cuda/lib64 -lcudart -ldl -lpthread \
  -Wl,-rpath,"$repo_dir/src/poseidon/tests/bootstrapping/build/poseidon_build" \
  -Wl,-rpath,/usr/local/cuda/lib64 -o "$precision_out"
