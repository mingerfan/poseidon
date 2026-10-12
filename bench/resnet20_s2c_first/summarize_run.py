#!/usr/bin/env python3
"""Validate a standalone ResNet20 run log and summarize bootstrap timing.

The reported timing is the sum of the synchronized S2C-first GPU stage timers
already emitted by the executable.  It intentionally is not presented as
end-to-end online inference or as a CUPTI activity-union measurement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


TOTAL_RE = re.compile(
    r"^BOOTSTRAP_GPU_TIMING total_stages_ms=(?P<value>[-+0-9.eE]+)\b"
)
STAGE_RE = re.compile(
    r"^BOOTSTRAP_GPU_TIMING stage=(?P<stage>\S+) "
    r"elapsed_ms=(?P<value>[-+0-9.eE]+)\b"
)
SCRIPT_TIME_RE = re.compile(
    r"^Script (?P<event>started|done) on (?P<time>\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})"
)
EXPECTED_STAGES = (
    "S2C",
    "prepare",
    "ModRaise",
    "C2S",
    "EvalMod_real",
    "EvalMod_imag",
    "recombine_project_real",
)


def _fields(line: str) -> dict[str, str]:
    return {
        key: value
        for item in line.split()[1:]
        if "=" in item
        for key, value in [item.split("=", 1)]
    }


def summarize(path: Path, require_complete: bool = True) -> dict[str, object]:
    raw = path.read_bytes()
    lines = raw.decode(errors="replace").replace("\r", "").splitlines()
    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    script_times: dict[str, dt.datetime] = {}

    for line in lines:
        if match := TOTAL_RE.match(line):
            totals.append(float(match.group("value")))
        elif match := STAGE_RE.match(line):
            stages.setdefault(match.group("stage"), []).append(
                float(match.group("value"))
            )
        if match := SCRIPT_TIME_RE.match(line):
            script_times[match.group("event")] = dt.datetime.fromisoformat(
                match.group("time")
            )

    blocks = [line for line in lines if line.startswith("NETWORK_BLOCK_RESULT ")]
    bootstrap_phases = sum(
        line.startswith("PHASE bootstrap_existing_ciphertext ") for line in lines
    )
    cache_misses = sum(
        line == "BOOTSTRAP_OFFLINE_CACHE matrices=miss" for line in lines
    )
    cache_hits = sum(
        line == "BOOTSTRAP_OFFLINE_CACHE matrices=hit" for line in lines
    )
    final_lines = [
        line
        for line in lines
        if line.startswith("NETWORK_RESULT ") and "staged=false" in line
    ]
    final = _fields(final_lines[-1]) if final_lines else {}
    errors = [line for line in lines if line.startswith(("ERROR ", "STOP "))]

    try:
        predictions_match = (
            int(final["plain_prediction"]) == int(final["gpu_prediction"])
        )
        errors_within_contract = (
            float(final["max_logit_error"]) <= 0.1
            and float(final["max_boundary_error"]) <= 1e-3
        )
    except (KeyError, TypeError, ValueError):
        predictions_match = errors_within_contract = False
    stage_counts_match = set(stages) == set(EXPECTED_STAGES) and all(
        len(stages[name]) == 18 for name in EXPECTED_STAGES
    )
    stage_total_matches = bool(totals) and math.isclose(
        sum(totals),
        sum(sum(values) for values in stages.values()),
        rel_tol=0,
        abs_tol=0.1,
    )
    complete = (
        len(blocks) == 9
        and bootstrap_phases == 18
        and len(totals) == 18
        and cache_misses == 1
        and cache_hits == 17
        and stage_counts_match
        and stage_total_matches
        and len(final_lines) == 1
        and final.get("integration") == "PASS"
        and final.get("completed_blocks") == "9"
        and final.get("bootstraps") == "18"
        and final.get("convolutions") == "18"
        and final.get("residual_adds") == "9"
        and final.get("input_encryptions") == "27"
        and final.get("intermediate_reencryptions") == "0"
        and final.get("full_network_tested") == "true"
        and final.get("first_failure") == "none"
        and predictions_match
        and errors_within_contract
        and not errors
    )
    if require_complete and not complete:
        raise ValueError(
            "incomplete/failed standalone run: "
            f"blocks={len(blocks)} bootstrap_phases={bootstrap_phases} "
            f"bootstrap_timings={len(totals)} "
            f"final_results={len(final_lines)} errors={len(errors)}"
        )
    if any(not math.isfinite(value) or value < 0 for value in totals):
        raise ValueError("invalid bootstrap timing value")

    wall_seconds = None
    if "started" in script_times and "done" in script_times:
        wall_seconds = (
            script_times["done"] - script_times["started"]
        ).total_seconds()

    bootstrap: dict[str, object] = {
        "count": len(totals),
        "sum_ms": sum(totals),
        "mean_ms": statistics.fmean(totals) if totals else None,
        "median_ms": statistics.median(totals) if totals else None,
        "min_ms": min(totals) if totals else None,
        "max_ms": max(totals) if totals else None,
        "stage_sum_ms": {name: sum(values) for name, values in stages.items()},
        "stage_mean_ms": {
            name: statistics.fmean(values) for name, values in stages.items()
        },
    }
    correctness = {
        key: final.get(key)
        for key in (
            "integration",
            "true_label",
            "plain_prediction",
            "gpu_prediction",
            "max_logit_error",
            "max_boundary_error",
            "first_failure",
        )
    }
    return {
        "schema": "poseidon_resnet20_s2c_first_run_v1",
        "log": str(path.resolve()),
        "log_sha256": hashlib.sha256(raw).hexdigest(),
        "complete": complete,
        "blocks": len(blocks),
        "bootstrap_phases": bootstrap_phases,
        "offline_cache": {"misses": cache_misses, "hits": cache_hits},
        "correctness": correctness,
        "bootstrap_gpu_stages": bootstrap,
        "validation_wall_seconds": wall_seconds,
        "timing_scope": {
            "kind": "synchronized_host_wall_time_around_gpu_bootstrap_stages",
            "includes": list(EXPECTED_STAGES),
            "excludes": [
                "offline_matrix_and_key_preparation",
                "reference_checks",
                "convolution",
                "relu",
                "stem",
                "head",
            ],
            "cupti_activity_union": False,
            "end_to_end_online_inference": False,
        },
        "security_approved": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="emit a partial summary instead of rejecting an incomplete log",
    )
    args = parser.parse_args()
    try:
        report = summarize(args.log, require_complete=not args.allow_incomplete)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
