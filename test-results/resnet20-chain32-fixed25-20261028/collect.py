"""Save small, reviewable evidence; leave model binaries and bundles on 188Server."""

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import chain32_probe as p

review = p.OUT / "review"
review.mkdir(exist_ok=True)
all_rows = []
metadata = {}
for candidate in sorted((p.OUT / "candidates").iterdir()):
    metadata[candidate.name] = json.loads((candidate / "candidate.json").read_text())
    for f in sorted(candidate.glob("*/result.json")):
        row = json.loads(f.read_text())
        if f.parent.name.startswith("full-"):
            assert "returncode" in row, str(f)
        row["remote_result"] = str(f)
        all_rows.append(row)

full = [r for r in all_rows if r.get("phase") == "full"]
assert len(full) >= 44, len(full)
assert all(r["returncode"] == 0 and r["validation_code"] == 0 for r in full)
assert all(r["max_rescale_formula_error"] < 1e-8 for r in full)
full_names = sorted({r["name"] for r in full})
for r in full:
    # A level difference across Boot resets is not a consumed-Q depth.
    for value in r.get("stage_outputs", []):
        value.pop("consumed_q", None)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def packed(path, value):
    with gzip.open(path, "wt") as f:
        json.dump(value, f, ensure_ascii=False)


diagnostics = [r for r in all_rows if r.get("diagnostic")]
packed(review / "diagnostic-results.json.gz", diagnostics)
packed(review / "candidate-manifest.json.gz", metadata)
write_json(review / "full-results.json", full)
gpu = [r for r in all_rows if "max_abs_error" in r]
write_json(review / "gpu-results.json", gpu)
for name in full_names:
    source = p.OUT / "candidates" / name
    destination = review / "candidates" / name
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("candidate.json", "fixture.json", "operator-spec.json"):
        shutil.copyfile(source / filename, destination / filename)
    for phase in ("relu", "pair", "stages", "prefix1"):
        f = source / phase / "result.json"
        if f.exists():
            shutil.copyfile(f, destination / (phase + "-result.json"))
    f = source / "full-optimized-q31" / "boot-boundaries.json"
    if f.exists():
        shutil.copyfile(f, destination / "boot-boundaries.json")

checks = review / "checks"
checks.mkdir(exist_ok=True)
for f in (p.OUT / "checks").glob("*.log"):
    shutil.copyfile(f, checks / f.name)
shutil.copyfile(p.EXPERIMENT / "chain32-native-budget.log", checks / "budget.log")

successful = [r for r in diagnostics if r["returncode"] == 0 and r.get("boots") == 0]
pairs = [r for r in successful if r["phase"] == "pair"]
relus = [r for r in successful if r["phase"] in ("relu", "stages")]
summary = dict(
    date="2026-10-28",
    branch="experiment/native-scale-planning",
    compiler_sha256=(p.EXPERIMENT / "binary.sha256").read_text().strip(),
    model_sha256=hashlib.sha256(
        Path("/home/xuming/poseidon-mixed-q-20261026/data/ResNet20.mlirbc").read_bytes()
    ).hexdigest(),
    constraints=dict(
        working_q=25,
        boot_input_q=6,
        boot_output_q=31,
        boot_input_scale=40,
        boot_output_scale=40,
        poly_degree=p.SPEC["context"]["poly_degree"],
        max_prime_bits=32,
        waterline=40,
        rescale_policy="threshold; no target options/goals/schedules",
    ),
    prepared_candidates=len(metadata),
    distinct_working_chains=len({tuple(m["moduli"]) for m in metadata.values()}),
    diagnostic_evaluations=len(diagnostics),
    diagnostic_outcomes=dict(
        Counter(
            "failed"
            if r["returncode"]
            else "has_boot"
            if r.get("boots")
            else "unbooted"
            for r in diagnostics
        )
    ),
    diagnostic_phases=dict(Counter(r["phase"] for r in diagnostics)),
    full_candidates=len(full_names),
    full_compilations=len(full),
    full_boot_counts={
        f"{strategy}-q{initial}": dict(
            Counter(
                r["boots"]
                for r in full
                if r["strategy"] == strategy and r["initial_q"] == initial
            )
        )
        for strategy in ("greedy", "optimized")
        for initial in (31, 34)
    },
    min_optimized=min(r["boots"] for r in full if r["strategy"] == "optimized"),
    min_greedy_q31=min(
        r["boots"] for r in full if r["strategy"] == "greedy" and r["initial_q"] == 31
    ),
    min_greedy_q34=min(
        r["boots"] for r in full if r["strategy"] == "greedy" and r["initial_q"] == 34
    ),
    min_unbooted_relu_q=min(r["consumed_q"] for r in relus),
    min_unbooted_pair_q=min(r["consumed_q"] for r in pairs),
    min_unbooted_pair_q_with_alignment=min(r["q_including_alignment"] for r in pairs),
    regression=dict(
        checks=607,
        graph_cases=540,
        valid_graph_plans=489,
        infeasible_graph_budgets=45,
        documented_single_prefix_rejections=6,
        native_planning=25,
        manual_boot=9,
        native_artifacts=18,
        budget_and_removed_options=15,
    ),
    remote_artifacts=str(p.OUT),
    limitations=[
        "Finite chain search; not a proof of impossibility.",
        "Optimized retains one prefix per cut and minimizes estimated latency; not a global Boot optimum.",
        "Diagnostics lower the supported minimum level and remove Boot scale contract only to measure unbooted depth; only production full plans count toward the objective.",
        "GPU checks cover one no-Boot ReLU only, not full encrypted inference or real Boot with changed parameters.",
        "peak_child_rss_kib uses cumulative child-process high-water mark; do not compare it as isolated per-case RSS.",
    ],
)
write_json(review / "summary.json", summary)
with (review / "full-results.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "candidate",
            "initial_q",
            "strategy",
            "boot",
            "rescale",
            "seconds",
            "working_actual_log2",
            "cpp_validation",
            "max_rescale_error",
        ]
    )
    for r in full:
        writer.writerow(
            [
                r["name"],
                r["initial_q"],
                r["strategy"],
                r["boots"],
                r["counts"]["rescale"],
                r["seconds"],
                metadata[r["name"]]["working_log2"],
                r["validation_code"],
                r["max_rescale_formula_error"],
            ]
        )

with tarfile.open(p.OUT / "review.tar.gz", "w:gz") as archive:
    archive.add(review, arcname="evidence")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("archive bytes", (p.OUT / "review.tar.gz").stat().st_size)
