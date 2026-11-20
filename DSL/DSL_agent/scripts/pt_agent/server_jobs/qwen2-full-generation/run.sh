#!/usr/bin/env bash
set -euo pipefail
QWEN_JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$QWEN_JOB_DIR/../../.."
exec python3 -B -m pt_agent.experiments.qwen_server_job "$@"
