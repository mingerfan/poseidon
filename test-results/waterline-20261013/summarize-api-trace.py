#!/usr/bin/env python3
"""Stream a closed Poseidon runtime CSV into host API timing statistics.

Usage: python3 summarize-api-trace.py TRACE.csv > SUMMARY.json
The caller must confirm the run finished and closed its trace. File stability
during this read does not establish run completion.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


# Runtime::open_trace/trace_event in third_party/ckks-runtime/runtime/runtime.hpp.
HEADER = ["rank", "event", "id", "ordinal", "consumer_ordinal", "op", "duration_ns"]
UNTIMED_EVENTS = {"admit", "release"}  # Current writer emits zero placeholders.


class Stats:
    def __init__(self):
        self.count = self.measured = self.total = 0
        self.minimum = self.maximum = None

    def add(self, duration):
        self.count += 1
        if duration is not None:
            self.measured += 1
            self.total += duration
            self.minimum = duration if self.minimum is None else min(self.minimum, duration)
            self.maximum = duration if self.maximum is None else max(self.maximum, duration)

    def report(self):
        return {"count": self.count, "measured_count": self.measured,
                "untimed_count": self.count - self.measured,
                "sum_ns": self.total if self.measured else None,
                "mean_ns": self.total / self.measured if self.measured else None,
                "min_ns": self.minimum, "max_ns": self.maximum}


def file_state(stat):
    return {"device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def hashing_lines(stream, digest, size):
    for raw in stream:
        digest.update(raw)
        size[0] += len(raw)
        yield raw.decode("utf-8")


def unsigned(text, field, line):
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError(f"CSV line {line}: {field} must be an unsigned integer")
    return int(text)


def summarize(path):
    path = Path(path).expanduser().resolve(strict=True)
    digest, size = hashlib.sha256(), [0]
    by_event, by_op, by_pair = (defaultdict(Stats) for _ in range(3))
    total, ranks = Stats(), set()
    with path.open("rb") as stream:
        before = file_state(os.fstat(stream.fileno()))
        if file_state(path.stat()) != before:
            raise RuntimeError("input path changed while opening the trace")
        reader = csv.reader(hashing_lines(stream, digest, size), strict=True)
        if next(reader, None) != HEADER:
            raise ValueError("CSV header must be exactly: " + ",".join(HEADER))
        for row in reader:
            line = reader.line_num
            if len(row) != len(HEADER):
                raise ValueError(f"CSV line {line}: expected {len(HEADER)} fields, got {len(row)}")
            rank, event, value_id, ordinal, consumer, op, duration_text = row
            ranks.add(unsigned(rank, "rank", line))
            unsigned(value_id, "id", line)
            unsigned(ordinal, "ordinal", line)
            if consumer:
                unsigned(consumer, "consumer_ordinal", line)
            if not event or not op:
                raise ValueError(f"CSV line {line}: event and op must be nonempty")
            duration = unsigned(duration_text, "duration_ns", line) if duration_text else None
            if event in UNTIMED_EVENTS:
                duration = None
            for stats in (total, by_event[event], by_op[op], by_pair[event, op]):
                stats.add(duration)
        after = file_state(os.fstat(stream.fileno()))
        path_after = file_state(path.stat())
        if before != after or before != path_after or size[0] != before["bytes"]:
            raise RuntimeError("trace changed during reading; discard this summary and use a closed trace")
    return {
        "schema_version": 1,
        "input": {"path": str(path), "sha256": digest.hexdigest(), "bytes": size[0],
                  "unchanged_during_read": True, "before": before, "after": after},
        "format_source": "third_party/ckks-runtime/runtime/runtime.hpp: Runtime::open_trace/trace_event",
        "header": HEADER,
        "notes": [
            "Host API durations in nanoseconds; these are not CUDA kernel durations.",
            "Calls across workers and nested calls can overlap. Sums are not wall-clock allocations or fractions.",
            "Rows and durations do not measure GPU completion, run completion, progress, or remaining time.",
            "The caller must supply a closed trace from a finished run; unchanged during this read does not prove completion.",
            "admit/release are untimed bookkeeping events, including their current zero placeholders. Empty durations are also untimed.",
            "count includes all rows; measured_count excludes untimed rows. Timing statistics use measured_count only; no samples gives null.",
            "event, op, and event/op tables regroup the same rows and must not be added together."
        ],
        "ranks": sorted(ranks), "totals": total.report(),
        "by_event": [{"event": key, **stats.report()} for key, stats in sorted(by_event.items())],
        "by_op": [{"op": key, **stats.report()} for key, stats in sorted(by_op.items())],
        "by_event_op": [{"event": key[0], "op": key[1], **stats.report()}
                        for key, stats in sorted(by_pair.items())]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path, help="explicit path to a closed runtime CSV trace")
    args = parser.parse_args()
    try:
        result = summarize(args.trace)
    except (OSError, UnicodeError, ValueError, csv.Error, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
