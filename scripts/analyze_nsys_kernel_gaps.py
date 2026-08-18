#!/usr/bin/env python3
"""Compare actual GPU kernel execution gaps in Nsight Systems reports.

The analyzer uses CUPTI GPU activity timestamps, not CUDA launch API duration.
It correlates each GPU kernel back to the CPU thread that submitted it, then
orders kernels independently on each (CPU thread, GPU device, CUDA stream)
lane.  Reports may be ``.nsys-rep`` files or pre-exported SQLite databases.

Generated files:

* ``REPORT.md``: human-readable method, per-lane summaries, and comparison.
* ``thread_summary.csv``: one row per CPU-thread/GPU/stream lane.
* ``comparison.csv``: one row per report.
* ``<label>_gpu_kernel_intervals.csv``: every actual GPU kernel and its gap.
"""

# Usage:
#   python3 scripts/analyze_nsys_kernel_gaps.py \
#     1GPU=/path/probe_1gpu.nsys-rep \
#     4GPU=/path/probe_4gpu_4w.nsys-rep \
#     8GPU=/path/8gpu-probe-strong.nsys-rep \
#     --output-dir nsys-kernel-analysis
#
# Re-running into the same output directory:
#   python3 scripts/analyze_nsys_kernel_gaps.py ... \
#     --output-dir nsys-kernel-analysis --force
#
# Optional controls:
#   --max-gap-ms 10              Exclude longer inter-kernel gaps from gap stats.
#   --anchor 1GPU=first-nvtx     Override one report's automatic anchor.
#   --anchor 4GPU=14442126652    Use an exact anchor timestamp in nanoseconds.

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TID_MASK = 0xFFFFFF
PROCESS_MASK = -0x1000000
NS_PER_US = 1_000
NS_PER_MS = 1_000_000
SYNC_NAME_PATTERN = "%Synchronize%"


class AnalysisError(RuntimeError):
    """Raised when a report cannot be analyzed without misleading output."""


@dataclass(frozen=True)
class InputSpec:
    label: str
    safe_label: str
    path: Path
    anchor: str = "auto"


@dataclass(frozen=True)
class Kernel:
    thread_tid: int
    device_id: int
    stream_id: int
    correlation_id: int
    name: str
    gpu_start_ns: int
    gpu_end_ns: int
    cpu_api: str
    cpu_start_ns: int
    cpu_end_ns: int


@dataclass(frozen=True)
class Interval:
    kernel: Kernel
    kernel_index: int
    previous_node: str
    gap_ns: int
    gap_included: bool


@dataclass(frozen=True)
class LaneSummary:
    thread_tid: int
    device_id: int
    stream_id: int
    kernel_count: int
    anchor_to_first_ns: int
    inter_gap_count: int
    excluded_inter_gap_count: int
    inter_gap_total_ns: int
    inter_gap_min_ns: int
    inter_gap_max_ns: int
    gaps_gt_10ms: int
    busy_ns: int
    wall_ns: int
    gpu_span_ns: int
    last_kernel_end_ns: int

    @property
    def inter_gap_avg_ns(self) -> float:
        if not self.inter_gap_count:
            return 0.0
        return self.inter_gap_total_ns / self.inter_gap_count

    @property
    def busy_pct(self) -> float:
        return 100.0 * self.busy_ns / self.wall_ns if self.wall_ns else 0.0


@dataclass(frozen=True)
class SyncPoint:
    api: str
    thread_tid: int
    start_ns: int
    end_ns: int
    return_after_last_kernel_ns: int


@dataclass
class ProfileAnalysis:
    spec: InputSpec
    anchor_ns: int
    anchor_kind: str
    intervals: list[Interval]
    lanes: list[LaneSummary]
    kernel_mix: Counter[str]
    final_sync: SyncPoint | None

    @property
    def kernel_count(self) -> int:
        return len(self.intervals)

    @property
    def thread_count(self) -> int:
        return len({item.kernel.thread_tid for item in self.intervals})

    @property
    def gpu_count(self) -> int:
        return len({item.kernel.device_id for item in self.intervals})

    @property
    def lane_count(self) -> int:
        return len(self.lanes)

    @property
    def aggregate_busy_ns(self) -> int:
        return sum(item.busy_ns for item in self.lanes)

    @property
    def overall_wall_ns(self) -> int:
        return max(item.last_kernel_end_ns for item in self.lanes) - self.anchor_ns

    @property
    def inter_gap_count(self) -> int:
        return sum(item.inter_gap_count for item in self.lanes)

    @property
    def excluded_inter_gap_count(self) -> int:
        return sum(item.excluded_inter_gap_count for item in self.lanes)

    @property
    def inter_gap_total_ns(self) -> int:
        return sum(item.inter_gap_total_ns for item in self.lanes)

    @property
    def inter_gap_avg_ns(self) -> float:
        if not self.inter_gap_count:
            return 0.0
        return self.inter_gap_total_ns / self.inter_gap_count

    @property
    def inter_gap_max_ns(self) -> int:
        return max((item.inter_gap_max_ns for item in self.lanes), default=0)

    @property
    def gaps_gt_10ms(self) -> int:
        return sum(item.gaps_gt_10ms for item in self.lanes)

    @property
    def average_lane_capacity_pct(self) -> float:
        capacity_ns = self.lane_count * self.overall_wall_ns
        return 100.0 * self.aggregate_busy_ns / capacity_ns if capacity_ns else 0.0

    @property
    def kernels_per_second(self) -> float:
        if not self.overall_wall_ns:
            return 0.0
        return self.kernel_count * 1_000_000_000.0 / self.overall_wall_ns


def tid(global_tid: int) -> int:
    return int(global_tid) & TID_MASK


def ns_to_us(value: int | float) -> float:
    return value / NS_PER_US


def ns_to_ms(value: int | float) -> float:
    return value / NS_PER_MS


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def safe_filename(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not safe:
        raise AnalysisError(f"label has no filename-safe characters: {label!r}")
    return safe


def default_label(path: Path) -> str:
    name = path.name
    if name.endswith(".nsys-rep"):
        return name.removesuffix(".nsys-rep")
    return path.stem


def parse_inputs(raw_inputs: Sequence[str], anchors: Sequence[str]) -> list[InputSpec]:
    anchor_map: dict[str, str] = {}
    for raw in anchors:
        if "=" not in raw:
            raise AnalysisError(f"--anchor must be LABEL=MODE_OR_NS, got {raw!r}")
        label, value = raw.split("=", 1)
        if not label or not value:
            raise AnalysisError(f"invalid --anchor value: {raw!r}")
        anchor_map[label] = value

    parsed: list[tuple[str, Path]] = []
    for raw in raw_inputs:
        if "=" in raw:
            label, path_text = raw.split("=", 1)
            if not label or not path_text:
                raise AnalysisError(f"invalid input specification: {raw!r}")
            path = Path(path_text).expanduser()
        else:
            path = Path(raw).expanduser()
            label = default_label(path)
        parsed.append((label, path))

    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise AnalysisError("report labels must be unique")
    unknown_anchors = sorted(set(anchor_map) - set(labels))
    if unknown_anchors:
        raise AnalysisError(
            "--anchor references unknown labels: " + ", ".join(unknown_anchors)
        )

    specs = [
        InputSpec(label, safe_filename(label), path, anchor_map.get(label, "auto"))
        for label, path in parsed
    ]
    safe_labels = [item.safe_label for item in specs]
    if len(safe_labels) != len(set(safe_labels)):
        raise AnalysisError("report labels collide after filename sanitization")
    for item in specs:
        if not item.path.is_file():
            raise AnalysisError(f"report does not exist: {item.path}")
        if not (item.path.name.endswith(".nsys-rep") or item.path.suffix == ".sqlite"):
            raise AnalysisError(f"input must be .nsys-rep or .sqlite: {item.path}")
    return specs


def export_report(report: Path, output: Path, nsys: str) -> None:
    command = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--quiet",
        "true",
        "--output",
        str(output),
        str(report),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise AnalysisError(f"nsys export failed for {report}: {detail}")


def require_tables(connection: sqlite3.Connection, tables: Iterable[str]) -> None:
    available = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = sorted(set(tables) - available)
    if missing:
        raise AnalysisError("SQLite export is missing tables: " + ", ".join(missing))


def has_tables(connection: sqlite3.Connection, tables: Iterable[str]) -> bool:
    available = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    return set(tables).issubset(available)


def load_kernels(connection: sqlite3.Connection) -> list[Kernel]:
    require_tables(
        connection,
        [
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "StringIds",
        ],
    )
    rows = connection.execute(
        """
        SELECT
            r.globalTid AS global_tid,
            k.deviceId AS device_id,
            k.streamId AS stream_id,
            k.correlationId AS correlation_id,
            k.start AS gpu_start_ns,
            k.end AS gpu_end_ns,
            kernel_name.value AS kernel_name,
            api_name.value AS cpu_api,
            r.start AS cpu_start_ns,
            r.end AS cpu_end_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
          ON r.correlationId = k.correlationId
         AND (
              k.globalPid IS NULL
              OR (r.globalTid & ?) = k.globalPid
         )
        JOIN StringIds AS kernel_name ON kernel_name.id = k.shortName
        JOIN StringIds AS api_name ON api_name.id = r.nameId
        ORDER BY r.globalTid, k.deviceId, k.streamId,
                 k.start, k.end, k.correlationId
        """,
        (PROCESS_MASK,),
    ).fetchall()
    kernel_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
    )
    if len(rows) != kernel_count:
        raise AnalysisError(
            "GPU kernel correlation is not one-to-one: "
            f"{kernel_count} kernel rows but {len(rows)} correlated rows"
        )
    if not rows:
        raise AnalysisError("report contains no GPU kernel activity")
    return [
        Kernel(
            thread_tid=tid(row["global_tid"]),
            device_id=int(row["device_id"]),
            stream_id=int(row["stream_id"]),
            correlation_id=int(row["correlation_id"]),
            name=str(row["kernel_name"]),
            gpu_start_ns=int(row["gpu_start_ns"]),
            gpu_end_ns=int(row["gpu_end_ns"]),
            cpu_api=str(row["cpu_api"]),
            cpu_start_ns=int(row["cpu_start_ns"]),
            cpu_end_ns=int(row["cpu_end_ns"]),
        )
        for row in rows
    ]


def pthread_join_anchor(
    connection: sqlite3.Connection, first_kernel_ns: int
) -> int | None:
    if not has_tables(connection, ["OSRT_API", "StringIds"]):
        return None
    row = connection.execute(
        """
        SELECT o.start
        FROM OSRT_API AS o
        JOIN StringIds AS name ON name.id = o.nameId
        WHERE name.value = 'pthread_join'
          AND o.start <= ?
          AND o.end >= ?
        ORDER BY o.start DESC
        LIMIT 1
        """,
        (first_kernel_ns, first_kernel_ns),
    ).fetchone()
    return int(row[0]) if row else None


def first_nvtx_anchor(
    connection: sqlite3.Connection, first_kernel_ns: int
) -> int | None:
    if not has_tables(connection, ["NVTX_EVENTS"]):
        return None
    row = connection.execute(
        """
        SELECT MIN(start)
        FROM NVTX_EVENTS
        WHERE start <= ? AND end IS NOT NULL
        """,
        (first_kernel_ns,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def resolve_anchor(
    connection: sqlite3.Connection, first_kernel_ns: int, selection: str
) -> tuple[int, str]:
    normalized = selection.strip().lower()
    if normalized.isdigit():
        anchor = int(normalized)
        if anchor > first_kernel_ns:
            raise AnalysisError(
                f"explicit anchor {anchor} is after first kernel {first_kernel_ns}"
            )
        return anchor, "explicit"
    if normalized not in {"auto", "pthread-join", "first-nvtx", "first-kernel"}:
        raise AnalysisError(
            f"unknown anchor mode {selection!r}; expected auto, pthread-join, "
            "first-nvtx, first-kernel, or a nanosecond timestamp"
        )
    if normalized in {"auto", "pthread-join"}:
        anchor = pthread_join_anchor(connection, first_kernel_ns)
        if anchor is not None:
            return anchor, "pthread_join.start"
        if normalized == "pthread-join":
            raise AnalysisError("no pthread_join spans the first GPU kernel")
    if normalized in {"auto", "first-nvtx"}:
        anchor = first_nvtx_anchor(connection, first_kernel_ns)
        if anchor is not None:
            return anchor, "first_compute_nvtx.start"
        if normalized == "first-nvtx":
            raise AnalysisError("no NVTX range starts before the first GPU kernel")
    return first_kernel_ns, "first_gpu_kernel.start"


def find_final_sync(
    connection: sqlite3.Connection, last_kernel_end_ns: int
) -> SyncPoint | None:
    row = connection.execute(
        """
        SELECT r.start, r.end, r.globalTid, name.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        JOIN StringIds AS name ON name.id = r.nameId
        WHERE name.value LIKE ?
          AND r.end >= ?
        ORDER BY r.end
        LIMIT 1
        """,
        (SYNC_NAME_PATTERN, last_kernel_end_ns),
    ).fetchone()
    if not row:
        return None
    return SyncPoint(
        api=str(row[3]),
        thread_tid=tid(row[2]),
        start_ns=int(row[0]),
        end_ns=int(row[1]),
        return_after_last_kernel_ns=int(row[1]) - last_kernel_end_ns,
    )


def build_intervals(
    kernels: Sequence[Kernel], anchor_ns: int, max_gap_ns: int | None
) -> tuple[list[Interval], list[LaneSummary]]:
    grouped: dict[tuple[int, int, int], list[Kernel]] = defaultdict(list)
    for kernel in kernels:
        if kernel.gpu_start_ns >= anchor_ns:
            grouped[(kernel.thread_tid, kernel.device_id, kernel.stream_id)].append(
                kernel
            )
    if not grouped:
        raise AnalysisError("no GPU kernels occur at or after the selected anchor")

    intervals: list[Interval] = []
    lanes: list[LaneSummary] = []
    for key, lane_kernels in sorted(grouped.items()):
        lane_kernels.sort(
            key=lambda item: (
                item.gpu_start_ns,
                item.gpu_end_ns,
                item.correlation_id,
            )
        )
        previous_end_ns = anchor_ns
        previous_node = "anchor"
        lane_intervals: list[Interval] = []
        for index, kernel in enumerate(lane_kernels, start=1):
            gap_ns = kernel.gpu_start_ns - previous_end_ns
            if gap_ns < 0:
                raise AnalysisError(
                    "overlapping kernels found in one thread/device/stream lane: "
                    f"TID {key[0]}, GPU {key[1]}, stream {key[2]}"
                )
            included = index == 1 or max_gap_ns is None or gap_ns <= max_gap_ns
            lane_intervals.append(
                Interval(kernel, index, previous_node, gap_ns, included)
            )
            previous_end_ns = kernel.gpu_end_ns
            previous_node = kernel.name
        intervals.extend(lane_intervals)

        inter = lane_intervals[1:]
        included_gaps = [item.gap_ns for item in inter if item.gap_included]
        busy_ns = sum(
            item.kernel.gpu_end_ns - item.kernel.gpu_start_ns for item in lane_intervals
        )
        first = lane_intervals[0].kernel
        last = lane_intervals[-1].kernel
        lanes.append(
            LaneSummary(
                thread_tid=key[0],
                device_id=key[1],
                stream_id=key[2],
                kernel_count=len(lane_intervals),
                anchor_to_first_ns=first.gpu_start_ns - anchor_ns,
                inter_gap_count=len(included_gaps),
                excluded_inter_gap_count=len(inter) - len(included_gaps),
                inter_gap_total_ns=sum(included_gaps),
                inter_gap_min_ns=min(included_gaps, default=0),
                inter_gap_max_ns=max(included_gaps, default=0),
                gaps_gt_10ms=sum(item.gap_ns > 10 * NS_PER_MS for item in inter),
                busy_ns=busy_ns,
                wall_ns=last.gpu_end_ns - anchor_ns,
                gpu_span_ns=last.gpu_end_ns - first.gpu_start_ns,
                last_kernel_end_ns=last.gpu_end_ns,
            )
        )
    intervals.sort(
        key=lambda item: (
            item.kernel.thread_tid,
            item.kernel.device_id,
            item.kernel.stream_id,
            item.kernel_index,
        )
    )
    return intervals, lanes


def analyze_database(
    spec: InputSpec, database: Path, max_gap_ns: int | None
) -> ProfileAnalysis:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        kernels = load_kernels(connection)
        first_kernel_ns = min(item.gpu_start_ns for item in kernels)
        anchor_ns, anchor_kind = resolve_anchor(
            connection, first_kernel_ns, spec.anchor
        )
        intervals, lanes = build_intervals(kernels, anchor_ns, max_gap_ns)
        last_kernel_end_ns = max(item.last_kernel_end_ns for item in lanes)
        final_sync = find_final_sync(connection, last_kernel_end_ns)
        return ProfileAnalysis(
            spec=spec,
            anchor_ns=anchor_ns,
            anchor_kind=anchor_kind,
            intervals=intervals,
            lanes=lanes,
            kernel_mix=Counter(item.kernel.name for item in intervals),
            final_sync=final_sync,
        )
    finally:
        connection.close()


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def mix_relation(profile: ProfileAnalysis, baseline: ProfileAnalysis) -> str:
    if profile.kernel_mix == baseline.kernel_mix:
        return "same"
    names = set(profile.kernel_mix) | set(baseline.kernel_mix)
    if all(
        profile.kernel_mix[name] * baseline.kernel_count
        == baseline.kernel_mix[name] * profile.kernel_count
        for name in names
    ):
        return "scaled"
    return "different"


def comparison_rows(
    profiles: Sequence[ProfileAnalysis], max_gap_ns: int | None
) -> list[dict[str, str]]:
    baseline = profiles[0]
    rows = []
    for profile in profiles:
        relation = mix_relation(profile, baseline)
        normalized_wall_ns = (
            profile.overall_wall_ns * baseline.kernel_count / profile.kernel_count
            if relation in {"same", "scaled"}
            else None
        )
        rows.append(
            {
                "profile": profile.spec.label,
                "gpu_count": str(profile.gpu_count),
                "thread_count": str(profile.thread_count),
                "lane_count": str(profile.lane_count),
                "kernel_count": str(profile.kernel_count),
                "kernel_mix_vs_baseline": relation,
                "workload_vs_baseline": format_float(
                    profile.kernel_count / baseline.kernel_count, 6
                ),
                "wall_ms": format_float(ns_to_ms(profile.overall_wall_ns)),
                "normalized_wall_ms": (
                    format_float(ns_to_ms(normalized_wall_ns))
                    if normalized_wall_ns is not None
                    else ""
                ),
                "aggregate_gpu_busy_ms": format_float(
                    ns_to_ms(profile.aggregate_busy_ns)
                ),
                "busy_us_per_kernel": format_float(
                    ns_to_us(profile.aggregate_busy_ns) / profile.kernel_count
                ),
                "inter_gap_count": str(profile.inter_gap_count),
                "excluded_inter_gap_count": str(profile.excluded_inter_gap_count),
                "inter_gap_total_us": format_float(
                    ns_to_us(profile.inter_gap_total_ns)
                ),
                "inter_gap_avg_us": format_float(ns_to_us(profile.inter_gap_avg_ns)),
                "inter_gap_max_us": format_float(ns_to_us(profile.inter_gap_max_ns)),
                "gaps_gt_10ms": str(profile.gaps_gt_10ms),
                "average_lane_capacity_pct": format_float(
                    profile.average_lane_capacity_pct
                ),
                "kernels_per_second": format_float(profile.kernels_per_second),
                "throughput_vs_baseline": format_float(
                    profile.kernels_per_second / baseline.kernels_per_second
                ),
                "max_gap_ms_filter": (
                    format_float(ns_to_ms(max_gap_ns)) if max_gap_ns is not None else ""
                ),
            }
        )
    return rows


def render_report(profiles: Sequence[ProfileAnalysis], max_gap_ns: int | None) -> str:
    sections = [
        "# Nsight Systems GPU Kernel Gap Analysis",
        "",
        "Actual CUPTI GPU execution timestamps are used. CUDA launch API duration "
        "is included only for correlation back to the submitting CPU thread.",
        "",
        "For each CPU-thread/GPU/stream lane:",
        "",
        "```text",
        "wall = anchor_to_first + inter_kernel_gaps + GPU_kernel_busy",
        "```",
        "",
    ]
    if max_gap_ns is None:
        sections.append("All non-negative inter-kernel gaps are included.")
    else:
        sections.append(
            "Inter-kernel gaps greater than "
            f"{format_float(ns_to_ms(max_gap_ns))} ms are excluded from gap "
            "totals and averages. Wall and GPU busy time are never filtered."
        )
    sections.append("")

    for profile in profiles:
        sections.extend(
            [
                f"## {profile.spec.label}",
                "",
                f"Source: `{profile.spec.path}`",
                "",
                f"Anchor: `{profile.anchor_ns}` ns ({profile.anchor_kind})",
                "",
            ]
        )
        lane_rows = []
        for lane in profile.lanes:
            lane_rows.append(
                [
                    str(lane.thread_tid),
                    str(lane.device_id),
                    str(lane.stream_id),
                    str(lane.kernel_count),
                    format_float(ns_to_us(lane.anchor_to_first_ns)),
                    format_float(ns_to_us(lane.inter_gap_total_ns)),
                    format_float(ns_to_us(lane.inter_gap_avg_ns)),
                    format_float(ns_to_us(lane.busy_ns)),
                    format_float(ns_to_us(lane.wall_ns)),
                    str(lane.excluded_inter_gap_count),
                ]
            )
        sections.extend(
            [
                markdown_table(
                    [
                        "TID",
                        "GPU",
                        "Stream",
                        "Kernels",
                        "Anchor->first us",
                        "Gap total us",
                        "Gap avg us",
                        "GPU busy us",
                        "Wall us",
                        "Excluded gaps",
                    ],
                    lane_rows,
                ),
                "",
            ]
        )
        if profile.final_sync:
            sync = profile.final_sync
            sections.extend(
                [
                    f"First synchronization returning after the last kernel: "
                    f"`{sync.api}` on TID {sync.thread_tid}; return is "
                    f"{format_float(ns_to_us(sync.return_after_last_kernel_ns))} us "
                    "after the last kernel end.",
                    "",
                ]
            )
        else:
            sections.extend(
                [
                    "No CUDA synchronization returning after the last kernel was found.",
                    "",
                ]
            )

    compare = comparison_rows(profiles, max_gap_ns)
    compare_table_rows = [
        [
            row["profile"],
            row["gpu_count"],
            row["kernel_count"],
            row["kernel_mix_vs_baseline"],
            row["wall_ms"],
            row["normalized_wall_ms"],
            row["aggregate_gpu_busy_ms"],
            row["inter_gap_avg_us"],
            row["average_lane_capacity_pct"],
            row["throughput_vs_baseline"],
        ]
        for row in compare
    ]
    sections.extend(
        [
            "## Comparison",
            "",
            markdown_table(
                [
                    "Profile",
                    "GPUs",
                    "Kernels",
                    "Mix vs baseline",
                    "Wall ms",
                    "Normalized wall ms",
                    "GPU busy ms",
                    "Gap avg us",
                    "Lane capacity %",
                    "Throughput vs baseline",
                ],
                compare_table_rows,
            ),
            "",
            "The first input report is the comparison baseline. Normalized wall "
            "time is emitted only when the kernel-name distribution is identical "
            "or exactly scaled.",
            "",
        ]
    )
    return "\n".join(sections)


def ensure_outputs_available(paths: Sequence[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = "\n".join(f"  {path}" for path in existing)
        raise AnalysisError(
            "output files already exist; pass --force to replace them:\n" + rendered
        )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_interval_csv(profile: ProfileAnalysis, path: Path) -> None:
    fieldnames = [
        "thread_tid",
        "device_id",
        "stream_id",
        "kernel_index",
        "previous_node",
        "kernel_name",
        "gpu_start_from_anchor_us",
        "gpu_end_from_anchor_us",
        "gpu_duration_us",
        "gap_from_previous_us",
        "gap_included",
        "correlation_id",
        "cpu_api",
        "cpu_launch_start_from_anchor_us",
        "cpu_launch_end_from_anchor_us",
    ]
    rows = []
    for item in profile.intervals:
        kernel = item.kernel
        previous_node = (
            profile.anchor_kind if item.kernel_index == 1 else item.previous_node
        )
        rows.append(
            {
                "thread_tid": kernel.thread_tid,
                "device_id": kernel.device_id,
                "stream_id": kernel.stream_id,
                "kernel_index": item.kernel_index,
                "previous_node": previous_node,
                "kernel_name": kernel.name,
                "gpu_start_from_anchor_us": format_float(
                    ns_to_us(kernel.gpu_start_ns - profile.anchor_ns)
                ),
                "gpu_end_from_anchor_us": format_float(
                    ns_to_us(kernel.gpu_end_ns - profile.anchor_ns)
                ),
                "gpu_duration_us": format_float(
                    ns_to_us(kernel.gpu_end_ns - kernel.gpu_start_ns)
                ),
                "gap_from_previous_us": format_float(ns_to_us(item.gap_ns)),
                "gap_included": "true" if item.gap_included else "false",
                "correlation_id": kernel.correlation_id,
                "cpu_api": kernel.cpu_api,
                "cpu_launch_start_from_anchor_us": format_float(
                    ns_to_us(kernel.cpu_start_ns - profile.anchor_ns)
                ),
                "cpu_launch_end_from_anchor_us": format_float(
                    ns_to_us(kernel.cpu_end_ns - profile.anchor_ns)
                ),
            }
        )
    write_csv(path, fieldnames, rows)


def write_lane_summary(profiles: Sequence[ProfileAnalysis], path: Path) -> None:
    fieldnames = [
        "profile",
        "source",
        "anchor_ns",
        "anchor_kind",
        "thread_tid",
        "device_id",
        "stream_id",
        "kernel_count",
        "anchor_to_first_us",
        "inter_gap_count",
        "excluded_inter_gap_count",
        "inter_gap_total_us",
        "inter_gap_avg_us",
        "inter_gap_min_us",
        "inter_gap_max_us",
        "gaps_gt_10ms",
        "gpu_busy_us",
        "compute_wall_us",
        "gpu_span_us",
        "busy_pct",
        "last_kernel_end_ns",
    ]
    rows = []
    for profile in profiles:
        for lane in profile.lanes:
            rows.append(
                {
                    "profile": profile.spec.label,
                    "source": str(profile.spec.path),
                    "anchor_ns": profile.anchor_ns,
                    "anchor_kind": profile.anchor_kind,
                    "thread_tid": lane.thread_tid,
                    "device_id": lane.device_id,
                    "stream_id": lane.stream_id,
                    "kernel_count": lane.kernel_count,
                    "anchor_to_first_us": format_float(
                        ns_to_us(lane.anchor_to_first_ns)
                    ),
                    "inter_gap_count": lane.inter_gap_count,
                    "excluded_inter_gap_count": lane.excluded_inter_gap_count,
                    "inter_gap_total_us": format_float(
                        ns_to_us(lane.inter_gap_total_ns)
                    ),
                    "inter_gap_avg_us": format_float(ns_to_us(lane.inter_gap_avg_ns)),
                    "inter_gap_min_us": format_float(ns_to_us(lane.inter_gap_min_ns)),
                    "inter_gap_max_us": format_float(ns_to_us(lane.inter_gap_max_ns)),
                    "gaps_gt_10ms": lane.gaps_gt_10ms,
                    "gpu_busy_us": format_float(ns_to_us(lane.busy_ns)),
                    "compute_wall_us": format_float(ns_to_us(lane.wall_ns)),
                    "gpu_span_us": format_float(ns_to_us(lane.gpu_span_ns)),
                    "busy_pct": format_float(lane.busy_pct),
                    "last_kernel_end_ns": lane.last_kernel_end_ns,
                }
            )
    write_csv(path, fieldnames, rows)


def write_outputs(
    profiles: Sequence[ProfileAnalysis],
    output_dir: Path,
    max_gap_ns: int | None,
    force: bool,
) -> str:
    report_path = output_dir / "REPORT.md"
    lane_path = output_dir / "thread_summary.csv"
    comparison_path = output_dir / "comparison.csv"
    interval_paths = [
        output_dir / f"{profile.spec.safe_label}_gpu_kernel_intervals.csv"
        for profile in profiles
    ]
    ensure_outputs_available(
        [report_path, lane_path, comparison_path, *interval_paths], force
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = render_report(profiles, max_gap_ns)
    report_path.write_text(report, encoding="utf-8")
    write_lane_summary(profiles, lane_path)
    compare = comparison_rows(profiles, max_gap_ns)
    write_csv(comparison_path, list(compare[0]), compare)
    for profile, path in zip(profiles, interval_paths, strict=True):
        write_interval_csv(profile, path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze actual GPU kernel gaps in one or more Nsight Systems reports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/analyze_nsys_kernel_gaps.py \\
    1GPU=/tmp/probe_1gpu.nsys-rep \\
    4GPU=/tmp/probe_4gpu_4w.nsys-rep \\
    8GPU=/tmp/8gpu-probe-strong.nsys-rep \\
    -o nsys-kernel-analysis

  python3 scripts/analyze_nsys_kernel_gaps.py report.sqlite \\
    --anchor report=first-kernel --max-gap-ms 10 -o analysis
""",
    )
    parser.add_argument(
        "reports",
        nargs="+",
        metavar="[LABEL=]REPORT",
        help="input .nsys-rep or exported .sqlite file; first input is baseline",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("nsys-kernel-analysis"),
        help="output directory (default: ./nsys-kernel-analysis)",
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        metavar="LABEL=MODE_OR_NS",
        help=(
            "anchor override; mode is auto, pthread-join, first-nvtx, "
            "first-kernel, or an exact nanosecond timestamp"
        ),
    )
    parser.add_argument(
        "--max-gap-ms",
        type=float,
        help="exclude larger inter-kernel gaps from gap totals and averages",
    )
    parser.add_argument(
        "--nsys",
        default="nsys",
        help="Nsight Systems CLI executable (default: nsys)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace generated files that already exist in the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.max_gap_ms is not None and args.max_gap_ms < 0:
            raise AnalysisError("--max-gap-ms must be non-negative")
        max_gap_ns = (
            round(args.max_gap_ms * NS_PER_MS) if args.max_gap_ms is not None else None
        )
        specs = parse_inputs(args.reports, args.anchor)
        needs_export = any(item.path.name.endswith(".nsys-rep") for item in specs)
        nsys = shutil.which(args.nsys) if needs_export else args.nsys
        if needs_export and not nsys:
            raise AnalysisError(
                f"Nsight Systems CLI not found: {args.nsys}; pass --nsys PATH"
            )

        profiles: list[ProfileAnalysis] = []
        with tempfile.TemporaryDirectory(prefix="nsys-kernel-gaps-") as temp_text:
            temp_dir = Path(temp_text)
            for index, spec in enumerate(specs):
                if spec.path.name.endswith(".nsys-rep"):
                    database = temp_dir / f"{index}_{spec.safe_label}.sqlite"
                    export_report(spec.path, database, str(nsys))
                else:
                    database = spec.path
                profiles.append(analyze_database(spec, database, max_gap_ns))

        report = write_outputs(
            profiles,
            args.output_dir.expanduser(),
            max_gap_ns,
            args.force,
        )
        print(report)
        print(f"Outputs: {args.output_dir.expanduser().resolve()}")
        return 0
    except (AnalysisError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
