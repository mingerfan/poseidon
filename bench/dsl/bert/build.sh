#!/usr/bin/env bash
# Configure/build only. Never run CTest, a runner, or a device probe.
set -euo pipefail
source_dir=${BERT_SOURCE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
build_dir=${BERT_BUILD_DIR:-$root/build-bert}
mode=build
usage() {
    printf 'Usage: bash %s [configure|build] [-B BUILD_DIR] [-- CMAKE_ARGS...]\n' "$0"
    printf '\nRespects CC, CXX, CUDACXX, CMAKE_GENERATOR, CMAKE_BUILD_PARALLEL_LEVEL.\n'
    printf 'Example: bash %s build -B build-bert -- -DCMAKE_CUDA_ARCHITECTURES=80\n' "$0"
    printf 'CPU only: add -DBERT_BUILD_GPU=OFF. No tests or GPU tasks run automatically.\n'
}
if [[ ${1:-} == configure || ${1:-} == build ]]; then mode=$1; shift; fi
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -B|--build-dir)
            [[ $# -ge 2 && -n $2 ]] || { printf 'Missing build directory\n' >&2; exit 2; }
            build_dir=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
cmake_command=${CMAKE:-cmake}
"$cmake_command" -S "$source_dir" -B "$build_dir" "$@"
if [[ $mode == build ]]; then
    build_args=(--build "$build_dir")
    if [[ -n ${BERT_BUILD_JOBS:-} ]]; then build_args+=(--parallel "$BERT_BUILD_JOBS"); fi
    "$cmake_command" "${build_args[@]}"
fi
