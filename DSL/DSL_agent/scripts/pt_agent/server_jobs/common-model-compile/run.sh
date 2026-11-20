#!/usr/bin/env bash
set -euo pipefail

task_factory=${1:?Pass an original r13_common_models factory}
task_system=${2:?Pass dacapo-optimized or dacapo-depth-dp}
task_output=${3:?Pass a fresh output directory under the configured results root}
case "$task_system" in
  dacapo-optimized|dacapo-depth-dp) ;;
  *) echo 'Agent compilation uses the CLI with an independent frozen-result audit.' >&2; exit 2 ;;
esac
task_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd -- "$task_scripts"
exec python3 -B -m pt_agent.experiments.r29_local_compile_matrix \
  --model-factory "$task_factory" --systems "$task_system" --out "$task_output" \
  --memory-gib 12 --case-gib 32 --trace-seconds 1800 --compile-seconds 7200
