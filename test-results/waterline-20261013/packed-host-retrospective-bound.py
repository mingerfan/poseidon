#!/usr/bin/env python3
"""Check a completed old run's retrospective managed-Host accounting bound.

This checks an inequality for the old admission trajectory, not execution of
the corrected binary or independent numerical validation. No GPU is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


UINT64_MAX = 2**64 - 1
SHARED = re.compile(
    rb"waterline shared host_peak=([0-9]+) pinned_peak=([0-9]+) raw_peak=([0-9]+)"
)
HEADER = re.compile(
    rb"QWEN inputs=[0-9]+ outputs=[0-9]+ devices=[0-9]+ slots=[0-9]+ "
    rb"backend=(gpu|cpu|auto|explicit) host_boot_instructions=[0-9]+"
)
COMPLETE = re.compile(rb"FORWARD complete runtime_ms=[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def positive_bytes(value):
    if not re.fullmatch(r"[0-9]+", value) or not 0 < int(value) <= UINT64_MAX:
        raise argparse.ArgumentTypeError("expected a positive uint64 byte count")
    return int(value)


def fingerprint(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def inspect_file(path, visit):
    """Hash the exact parsed bytes and reject inputs changed during the read."""
    digest = hashlib.sha256()
    size = 0
    before = path.stat()
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            size += len(raw)
            visit(number, raw.rstrip(b"\r\n"))
    if fingerprint(before) != fingerprint(path.stat()) or size != before.st_size:
        raise ValueError("input changed during read: " + str(path))
    return dict(path=str(path.resolve()), bytes=size, sha256="sha256:" + digest.hexdigest())


def check(args):
    status_lines = []
    status_binding = inspect_file(args.exit_status, lambda number, line: status_lines.append(line))
    if status_lines != [b"0"]:
        raise ValueError("--exit-status must contain exactly one line: 0 (matching completed old run)")

    summaries, headers, begins, completes = [], [], [], []

    def visit(number, line):
        if b"waterline shared" in line:
            match = SHARED.fullmatch(line)
            if match is None:
                raise ValueError("malformed shared ledger summary at line " + str(number))
            values = tuple(int(value) for value in match.groups())
            if any(value > UINT64_MAX for value in values):
                raise ValueError("shared ledger value exceeds uint64 at line " + str(number))
            summaries.append((number, values))
        if line.startswith(b"QWEN "):
            match = HEADER.fullmatch(line)
            if match is None:
                raise ValueError("malformed QWEN header at line " + str(number))
            headers.append((number, match.group(1).decode("ascii")))
        if line == b"FORWARD begin":
            begins.append(number)
        if line.startswith(b"FORWARD complete"):
            if COMPLETE.fullmatch(line) is None:
                raise ValueError("malformed FORWARD completion at line " + str(number))
            completes.append(number)

    log_binding = inspect_file(args.model_log, visit)
    if not all(len(values) == 1 for values in (summaries, headers, begins, completes)):
        raise ValueError("require exactly one QWEN header, FORWARD begin/complete and shared ledger summary")
    if headers[0][1] != "gpu":
        raise ValueError("this baseline proof requires fixed backend=gpu, not Auto/CPU/explicit")
    if not headers[0][0] < begins[0] < summaries[0][0] < completes[0]:
        raise ValueError("shared summary must follow FORWARD begin and precede FORWARD complete")

    host, pinned, raw = summaries[0][1]
    extra = (raw + 1) // 2
    corrected = host + extra
    established = corrected <= args.host_budget_bytes
    checker_binding = inspect_file(Path(__file__), lambda number, line: None)
    return dict(
        schema_version=1,
        claim="retrospective_managed_host_accounting_bound_for_old_admission_trajectory",
        bound_established=established,
        new_binary_full_execution_verified=False,
        independent_numerical_validation_verified=False,
        old_run_process_exit_code=0,
        backend="gpu",
        shared_summary_line_number=summaries[0][0],
        old_host_peak_bytes=host,
        old_pinned_peak_bytes=pinned,
        old_raw_peak_bytes=raw,
        additional_host_upper_bound_bytes=extra,
        corrected_host_upper_bound_bytes=corrected,
        host_budget_bytes=args.host_budget_bytes,
        host_budget_source="explicit --host-budget-bytes; caller must bind to this run's effective configuration",
        headroom_bytes=args.host_budget_bytes - corrected,
        baseline_scope="single-rank global-prefix fixed-GPU executor; source assumptions in companion document",
        inputs=dict(model_log=log_binding, exit_status=status_binding, checker=checker_binding),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-log", type=Path, required=True, help="completed runner's merged stdout/stderr")
    parser.add_argument("--exit-status", type=Path, required=True, help="matching completed process's exit-status file")
    parser.add_argument("--host-budget-bytes", type=positive_bytes, required=True,
                        help="effective managed Host budget of that run, in decimal bytes; no default")
    args = parser.parse_args()
    try:
        result = check(args)
    except (OSError, ValueError) as error:
        parser.exit(2, "error: " + str(error) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["bound_established"] else 1


if __name__ == "__main__":
    sys.exit(main())
