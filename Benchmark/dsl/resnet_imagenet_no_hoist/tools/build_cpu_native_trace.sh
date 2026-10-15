#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
native=${1:?Pass existing native repository root}
output=${2:?Pass a new isolated output directory}
python3 "$scripts/prepare_cpu_native_trace.py" "$native" "$output"
entrypoint=${3:-"$scripts/main_cpu_native_trace.cpp"}
cp "$entrypoint" "$output/main.cpp"
c++ -std=c++20 -O1 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$output" -I"$output/resnet18" -I"$output/resnet50" \
  -DPOSEIDON_GPU_RESNET18_SOURCE_DIR=\"$native/benchmark/resnet18_gpu\" \
  -DPOSEIDON_GPU_RESNET50_SOURCE_DIR=\"$native/benchmark/resnet50_gpu\" \
  -DPOSEIDON_RESNET20_RELU_FIXTURE=\"$native/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt\" \
  "$output/main.cpp" "$output/resnet18/gpu_config.cpp" \
  "$output/resnet18/resnet18_topology.cpp" "$output/resnet18/resnet18_weights.cpp" \
  "$output/resnet50/resnet50_config.cpp" "$output/resnet50/resnet50_topology.cpp" \
  "$output/resnet50/resnet50_weights.cpp" "$output/resnet50/gpu_multiplexed_tensor.cpp" \
  "$output/resnet50/gpu_relu_s2c_first.cpp" -lz -o "$output/native_trace"
echo "$output/native_trace"
