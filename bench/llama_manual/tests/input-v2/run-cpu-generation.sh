#!/usr/bin/env bash
# New exclusive std-only evidence generation. Never retry or reuse on failure.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(cd -- "$here/../../../.." && pwd -P)
if [ "$#" -ne 2 ] || [[ ! "$1" =~ ^g[0-9]+-[a-zA-Z0-9-]+$ ]] || [[ ! "$2" =~ ^[0-9]+$ ]]; then
    printf 'usage: bash %s gNN-name allowed_cpu\n' "$0" >&2
    exit 64
fi
out="$here/$1"
cpu="$2"
mkdir -- "$out" # Exclusive generation: an existing directory is a hard failure.
mkdir -- "$out/source"
cd -- "$repo"
cp bench/llama_manual/source_manual_io.hpp bench/llama_manual/manual_main.cpp bench/llama_manual/tests/manual_plan_test.cpp "$out/source/"
sha256sum bench/llama_manual/source_manual_io.hpp bench/llama_manual/manual_main.cpp bench/llama_manual/tests/manual_plan_test.cpp bench/llama_manual/source_manual_plan.hpp bench/llama_manual/source_manual_program.hpp src/poseidon/gpu/gpu_double_hoist.h src/poseidon/gpu/gpu_evaluator.cpp src/poseidon/gpu/gpu_evaluator.h src/poseidon/gpu/gpu_uploader.cpp > "$out/source-sha256.txt"
printf 'CPU=%s\nOMP_NUM_THREADS=1\nCUDA_VISIBLE_DEVICES=\nTMPDIR=%s\n' "$cpu" "$out" > "$out/environment.txt"
taskset -c "$cpu" env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR="$out" g++ --version > "$out/compiler.txt"
printf 'taskset -c %q env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=%q g++ -std=c++17 -O2 -Wall -Wextra -pedantic -I %q -I bench/llama_manual %q -o %q\n' "$cpu" "$out" "$out/source" "$out/source/manual_plan_test.cpp" "$out/manual_plan_test" > "$out/commands.txt"
set +e
taskset -c "$cpu" env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR="$out" g++ -std=c++17 -O2 -Wall -Wextra -pedantic -I "$out/source" -I bench/llama_manual "$out/source/manual_plan_test.cpp" -o "$out/manual_plan_test" > "$out/compile.stdout" 2> "$out/compile.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$out/compile.exit"
if [ "$rc" -ne 0 ]; then
    printf 'STOP: compile failed (%s); retained %s; no retry.\n' "$rc" "$out" >&2
    exit "$rc"
fi
printf 'taskset -c %q env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR=%q %q\n' "$cpu" "$out" "$out/manual_plan_test" >> "$out/commands.txt"
set +e
taskset -c "$cpu" env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TMPDIR="$out" "$out/manual_plan_test" > "$out/test.stdout" 2> "$out/test.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$out/test.exit"
sha256sum "$out/manual_plan_test" > "$out/binary-sha256.txt"
if [ "$rc" -ne 0 ]; then
    printf 'STOP: test failed (%s); retained %s; no retry.\n' "$rc" "$out" >&2
    exit "$rc"
fi
printf 'std-only CPU generation passed: %s (CPU %s). No target/GPU/model claim.\n' "$out" "$cpu"
