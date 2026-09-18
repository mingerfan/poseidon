"""Offline audit/export of a completed live batch; no API or credential access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from run_agent_batch import (case_metrics, summarize, save, load_failed_source,
                             generation_settings, configuration_delta, row_family,
                             validate_construction_continuation)


def validate_cohort_labels(report, baseline):
    """Audit independent benchmark metadata as well as model graph identity."""
    current, prior = report.get("benchmark"), baseline.get("benchmark")
    if bool(current) != bool(prior):
        raise ValueError("Agent/baseline benchmark metadata mismatch")
    if current:
        for key in ("version", "descriptor_set_sha256", "families", "new_family_definitions"):
            if current.get(key) != prior.get(key):
                raise ValueError("Agent/baseline benchmark identity mismatch")
    old = {r["descriptor"]["id"]: r for r in baseline["cases"]}
    for row in report["cases"]:
        match = old.get(row["descriptor"]["id"])
        if match is None or row_family(row) != row_family(match):
            raise ValueError("Agent/baseline model-family mismatch")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz_equal(left, right):
    with zipfile.ZipFile(left) as a, zipfile.ZipFile(right) as b:
        return set(a.namelist()) == set(b.namelist()) and all(a.read(n) == b.read(n) for n in a.namelist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    root, baseline = args.batch.resolve(), args.baseline.resolve()
    report = json.loads((root / "report.json").read_text())
    prior = json.loads((baseline / "report.json").read_text())
    rows = report["cases"]
    if not rows or report["status"] not in ("passed", "completed_with_failures"):
        raise ValueError("Expected a completed batch")
    expected = prior["selected_descriptors"]
    selection = report.get("selection")
    original_cases = {}
    if selection:
        origin = Path(selection["source_report"])
        if not origin.resolve().is_relative_to(root.parent) or digest(origin) != selection["source_sha256"]:
            raise ValueError("Original full-batch report changed")
        original, expected = load_failed_source(expected, origin, root.parent)
        from types import SimpleNamespace
        validate_construction_continuation(original,SimpleNamespace(public_numbers=report.get('public_numbers',False),
                                                                   public_control=report.get('public_control',False),
                                                                   public_strings=report.get('public_strings',False),
                                                                   public_polynomial=report.get('public_polynomial',False),
                                                                   object_arrays=report.get('object_arrays',False)))
        delta = configuration_delta(generation_settings(original), generation_settings(report))
        if delta != selection.get("configuration_changes", {}):
            raise ValueError("Unrecorded generation configuration change")
        original_cases = {r["descriptor"]["id"]:r for r in original["cases"]}
    if [r["descriptor"] for r in rows] != expected:
        raise ValueError("Agent/baseline catalog mismatch")
    validate_cohort_labels(report, prior)
    checks, errors = [], []
    diagnostics = []
    old_by_id = {r["descriptor"]["id"]:r for r in prior["cases"]}
    common_hashes = None
    for row in rows:
        case = row["descriptor"]["id"]
        old = old_by_id[case]
        if "evidence" not in row:
            errors.append(dict(case=case, error="missing_case_evidence"))
            continue
        folder = Path(row["evidence"])
        if folder.parent != root.parent or not folder.name.startswith("agent-deepseek-"):
            raise ValueError("Evidence path outside batch results root")
        raw = json.loads((folder / "report.json").read_text())
        if any(key in report for key in ('public_numbers','public_control','public_strings','public_polynomial','object_arrays')):
            request=json.loads((folder/'request.json').read_text())
            expected_objects = bool(report.get('object_arrays',False))
            expected_polynomial = bool(report.get('public_polynomial',False)) and not expected_objects
            expected_strings = bool(report.get('public_strings',False)) and not (expected_polynomial or expected_objects)
            expected_control = bool(report.get('public_control',False)) and not (expected_strings or expected_polynomial or expected_objects)
            expected_numeric = bool(report.get('public_numbers',False)) and not (expected_control or expected_strings or expected_polynomial or expected_objects)
            if ((request.get('task') == 'hecate-function-synthesis-v18') != expected_objects or
                (request.get('task') == 'hecate-function-synthesis-v17') != expected_polynomial or
                (request.get('task') == 'hecate-function-synthesis-v16') != expected_strings or
                (request.get('task') == 'hecate-function-synthesis-v15') != expected_control or
                (request.get('task') == 'hecate-function-synthesis-v14') != expected_numeric):
                errors.append(dict(case=case,error='batch_DSL_contract_mismatch'))
        if selection:
            previous_folder = Path(original_cases[case]["evidence"])
            if previous_folder.parent != root.parent:
                raise ValueError("Original case evidence outside results root")
            previous = json.loads((previous_folder / "report.json").read_text())
            if (raw.get("request_id") != previous.get("request_id") or
                    raw.get("frozen_hashes", {}).get("request.json") != previous.get("frozen_hashes", {}).get("request.json")):
                errors.append(dict(case=case, error="retest_prompt_changed"))
            for setting, value in generation_settings(original).items():
                if previous.get("provider_metrics", {}).get(setting, "deepseek" if setting == "service_provider" else None) != value:
                    errors.append(dict(case=case, error="source_configuration_mismatch", field=setting))
        for setting, value in generation_settings(report).items():
            if raw.get("provider_metrics", {}).get(setting, "deepseek" if setting == "service_provider" else None) != value:
                errors.append(dict(case=case, error="batch_configuration_mismatch", field=setting))
        for call in raw.get("provider_metrics", {}).get("calls", []):
            diagnostics.append(dict(case=case, **{k:call[k] for k in (
                "index", "status", "error", "finish_reason", "usage_status", "usage",
                "content_characters", "reasoning_content_characters", "refusal_present", "tool_calls_present") if k in call}))
        if case_metrics(raw) != row["metrics"]:
            errors.append(dict(case=case, error="metrics_mismatch"))
        if json.loads((folder / "model.json").read_text()) != row["descriptor"]:
            errors.append(dict(case=case, error="descriptor_mismatch"))
        for name, expected in raw.get("frozen_hashes", {}).items():
            path = folder / name
            if not path.resolve().is_relative_to(folder) or digest(path) != expected:
                errors.append(dict(case=case, error="frozen_input_mismatch", file=name))
        for attempt in raw.get("attempts", []):
            output = folder / f"attempt-{attempt['index']:02d}" / "output"
            for name, expected in attempt.get("artifact_hashes", {}).items():
                path = output / name
                if not path.resolve().is_relative_to(output) or digest(path) != expected:
                    errors.append(dict(case=case, error="artifact_mismatch", file=name))
        for name in ("arrays.npz", "weights.npz"):
            if not npz_equal(folder / name, baseline / old["folder"] / name):
                errors.append(dict(case=case, error="reference_baseline_mismatch", file=name))
        if (raw.get("compiler_profile_sha256") != prior["profile_sha256"] or
                raw.get("runtime_sha256") != prior["runtime_sha256"] or
                raw.get("tolerance") != prior["tolerance"]):
            errors.append(dict(case=case, error="backend_or_tolerance_mismatch"))
        if common_hashes is None:
            common_hashes = raw.get("source_hashes")
        elif raw.get("source_hashes") != common_hashes:
            errors.append(dict(case=case, error="source_changed_mid_batch"))
        checks.append(case)
    summary = summarize(rows)
    if summary != report["summary"]:
        errors.append(dict(error="aggregate_summary_mismatch"))
    successful = [r for r in rows if r["metrics"]["passed"]]
    count = sum(r["metrics"]["comparison"]["compared_values"] for r in successful)
    numeric = dict(compared_values=count,
        weighted_mae=sum(r["metrics"]["comparison"]["mae"]*r["metrics"]["comparison"]["compared_values"]
                         for r in successful)/count if count else None,
        max_absolute_error=max((r["metrics"]["comparison"]["max_absolute_error"] for r in successful), default=None),
        max_nonzero_relative_error=max((r["metrics"]["comparison"]["max_nonzero_reference_relative_error"]
                                       for r in successful), default=None))
    audit = dict(status="passed" if not errors else "failed", checked_cases=len(checks), errors=errors,
                 baseline=str(baseline), summary=summary, successful_numerics=numeric,
                 provider_diagnostics=diagnostics)
    save(root / "audit.json", audit)
    fields = ("case", "family", "first_passed", "passed", "api_calls", "repairs_attempted",
              "failure_layer", "failure_category", "provider_error", "mae", "max_absolute_error", "evidence")
    with (root / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            m = row["metrics"]
            writer.writerow(dict(case=row["descriptor"]["id"], family=row_family(row),
                **{k:m[k] for k in fields[2:9]}, mae=(m["comparison"] or {}).get("mae"),
                max_absolute_error=(m["comparison"] or {}).get("max_absolute_error"), evidence=row.get("evidence")))
    lines = [f"# {len(rows)}-case live DeepSeek evaluation", "", f"Evidence: `{root}`", "",
        f"Batch status: **{report['status']}**. Evidence audit: **{audit['status']}**.", "",
        "Backend: existing SEAL HEVM CPU; Poseidon GPU is not validated.", "",
        f"## Success metrics (all {len(rows)} planned cases remain in denominators)", "",
        "| Stage | First candidate |", "|---|---:|"]
    for name, value in summary["first"].items():
        lines.append(f"| {name} | {value['numerator']}/{value['denominator']} |")
    lines += ["", f"Final success after at most three repairs: {summary['final_success']['numerator']}/{len(rows)}.",
              f"API requests: {summary['api_calls']}; known tokens: {summary['usage']['total_tokens']}; "
              f"requests without usage: {summary['calls_without_usage']}.", "",
              "## Model families", "", "| Family | First success | Final success | Holdout |", "|---|---:|---:|---|"]
    for family, value in summary["families"].items():
        lines.append(f"| {family} | {value['first_success']}/{value['planned']} | {value['final_success']}/{value['planned']} | {value['holdout']} |")
    lines += ["", "## All cases", "", "| Case | First | Final | Requests | Failure | Max abs error |", "|---|---|---|---:|---|---:|"]
    for row in rows:
        m = row["metrics"]
        value = (m["comparison"] or {}).get("max_absolute_error")
        err = f"{value:.6g}" if value is not None else "N/A"
        lines.append(f"| {row['descriptor']['id']} | {m['first_passed']} | {m['passed']} | {m['api_calls']} | "
                     f"{m['provider_error'] or m['failure_layer'] or '-'} | {err} |")
    lines += ["", "## Interpretation limits", "",
        f"Fresh API generation per case; no deterministic-answer fallback. Model: {report['model']}; "
        f"effort: {report.get('reasoning_effort', 'high')}; timeout: {report['api_timeout']}s; "
        f"max tokens: {report['max_tokens']}. Prompt and numerical tolerances unchanged. "
        "No transport retries. Provider failures do not establish DSL incorrectness.",
        "", f"The selected deterministic baseline contains {len(prior['selected_descriptors'])} planned cases "
        f"with status {prior['status']}; the audit compares actual saved weights/inputs/reference arrays, "
        "compiler profile, runtime binary and tolerances. This is a historical correctness comparison, not a simultaneous performance benchmark.",
        "", "Holdout indicates exclusion from examples, not proven absence from model pretraining. Current input is a fixed data-only "
        "cohort with supplied constants/layout (including explicit user graphs when selected), not arbitrary PyTorch programs. "
        "Finite tests are not formal equivalence proofs.", ""]
    (root / "summary.md").write_text("\n".join(lines))
    print(json.dumps(audit, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
