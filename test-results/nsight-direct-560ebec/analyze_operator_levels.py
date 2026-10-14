#!/usr/bin/env python3
"""Recover per-operator, per-level GPU timings from the direct Nsight traces.

The traces predate level-labelled NVTX ranges.  The runtime nevertheless records a
CUDA completion event after every compute operation.  This script uses those events
as operation boundaries, assigns kernels/internal D2D copies/memsets to each boundary,
and joins the result with RuntimePlan value metadata to recover input/output levels.

The 4-GPU dependency scheduler may execute operations out of plan ordinal order.  Its
samples are therefore classified using CUDA activity signatures learned from the
sequential 1-GPU trace.  The script rejects ambiguous or missing signatures.

All reported durations are GPU timestamps:

* span_us: last activity end - first activity start for an operation.  This includes
  intra-operation gaps but excludes waits before the first activity.
* busy_us: sum of kernel, internal D2D-copy, and memset durations.  Peer transfers are
  deliberately excluded.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


Group = tuple[str, int, int]
Signature = tuple[tuple[tuple[object, ...], int], ...]


@dataclass
class Sample:
    group: Group
    device: int
    event_id: int
    device_sequence: int
    span_us: float
    busy_us: float
    kernel_count: int
    kernel_launch_api_us: float
    kernel_launch_api_samples_us: tuple[float, ...]
    kernel_submit_window_us: float
    kernel_busy_us: float
    signature: Signature
    ordinal: int | None = None


@dataclass
class GroupStats:
    group: Group
    samples: list[Sample]
    inliers: list[Sample]
    outliers: list[Sample]
    mean_us: float
    clean_mean_us: float
    median_us: float
    p95_us: float
    maximum_us: float
    clean_busy_mean_us: float


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sample")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def tukey_fences(samples: list[Sample]) -> tuple[float, float]:
    if len(samples) < 4:
        return -math.inf, math.inf
    values = [sample.span_us for sample in samples]
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def summarize(samples: list[Sample]) -> dict[Group, GroupStats]:
    grouped: dict[Group, list[Sample]] = collections.defaultdict(list)
    for sample in samples:
        grouped[sample.group].append(sample)

    result: dict[Group, GroupStats] = {}
    for group, members in grouped.items():
        low, high = tukey_fences(members)
        outliers = [sample for sample in members if not low <= sample.span_us <= high]
        outlier_ids = {id(sample) for sample in outliers}
        inliers = [sample for sample in members if id(sample) not in outlier_ids]
        spans = [sample.span_us for sample in members]
        result[group] = GroupStats(
            group=group,
            samples=members,
            inliers=inliers,
            outliers=outliers,
            mean_us=statistics.mean(spans),
            clean_mean_us=statistics.mean(sample.span_us for sample in inliers),
            median_us=statistics.median(spans),
            p95_us=percentile(spans, 0.95),
            maximum_us=max(spans),
            clean_busy_mean_us=statistics.mean(sample.busy_us for sample in inliers),
        )
    return result


def all_actions(plan: dict) -> list[dict]:
    return plan["initialization"] + plan["execution"] + plan["finalization"]


def compute_group(op: dict, values: dict[str, dict]) -> Group:
    return op["op"], values[op["inputs"][0]]["level"], values[op["output"]]["level"]


def activity_signature(activities: list[dict]) -> Signature:
    counts: collections.Counter[tuple[object, ...]] = collections.Counter()
    for activity in activities:
        if activity["kind"] == "kernel":
            key = (
                "kernel",
                activity["name"],
                activity["gridX"],
                activity["gridY"],
                activity["gridZ"],
                activity["blockX"],
                activity["blockY"],
                activity["blockZ"],
                activity["dynamicSharedMemory"],
            )
        elif activity["kind"] == "memcpy":
            key = "memcpy", activity["copyKind"], activity["bytes"]
        else:
            key = "memset", activity["bytes"], activity["value"]
        counts[key] += 1
    return tuple(sorted(counts.items(), key=repr))


def extract_samples(
    database_path: Path,
    plan: dict,
    signature_groups: dict[Signature, Group] | None = None,
) -> tuple[list[Sample], int]:
    values = {value["id"]: value for value in plan["values"]}
    compute_ops = [action for action in all_actions(plan) if action["kind"] == "compute"]

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    kernel_stream_rows = connection.execute(
        """
        SELECT contextId, streamId, deviceId, COUNT(*) AS launches
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY contextId, streamId, deviceId
        """
    ).fetchall()
    compute_streams = {
        (row["contextId"], row["streamId"]): row["deviceId"]
        for row in kernel_stream_rows
    }
    if not compute_streams:
        raise RuntimeError(f"no CUDA kernel streams in {database_path}")

    activities: list[dict] = []
    activity_queries = (
        (
            "kernel",
            """
            SELECT a.contextId, a.streamId, a.start, a.end, r.start AS launch_start,
                   r.end AS launch_end, s.value AS name, a.gridX, a.gridY, a.gridZ,
                   a.blockX, a.blockY, a.blockZ, a.dynamicSharedMemory
            FROM CUPTI_ACTIVITY_KIND_KERNEL a
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = a.correlationId
            JOIN StringIds s ON s.id = a.shortName
            """,
        ),
        (
            "memcpy",
            """
            SELECT a.contextId, a.streamId, a.start, a.end, r.start AS launch_start,
                   r.end AS launch_end,
                   a.copyKind, a.bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY a
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = a.correlationId
            WHERE a.copyKind = 8
            """,
        ),
        (
            "memset",
            """
            SELECT a.contextId, a.streamId, a.start, a.end, r.start AS launch_start,
                   r.end AS launch_end,
                   a.bytes, a.value
            FROM CUPTI_ACTIVITY_KIND_MEMSET a
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = a.correlationId
            """,
        ),
    )
    for kind, query in activity_queries:
        for row in connection.execute(query):
            if (row["contextId"], row["streamId"]) in compute_streams:
                activities.append({**dict(row), "kind": kind})
    if not activities:
        raise RuntimeError(f"no compute activities in {database_path}")

    # Initialization events share the compute stream in the sequential trace.  The
    # first compute activity enqueue separates them from online compute completions.
    first_compute_enqueue = min(activity["launch_start"] for activity in activities)
    samples: list[Sample] = []

    for stream_key, device in compute_streams.items():
        context_id, stream_id = stream_key
        completion_rows = connection.execute(
            """
            SELECT e.eventId, r.start AS record_start
            FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT e
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = e.correlationId
            WHERE e.contextId = ? AND e.streamId = ?
              AND e.eventId IN (
                  SELECT eventId
                  FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT
                  GROUP BY eventId
                  HAVING COUNT(*) = 1
              )
              AND r.start > ?
            ORDER BY r.start
            """,
            (context_id, stream_id, first_compute_enqueue),
        ).fetchall()
        record_starts = [row["record_start"] for row in completion_rows]
        stream_activities = sorted(
            (
                activity
                for activity in activities
                if (activity["contextId"], activity["streamId"]) == stream_key
            ),
            key=lambda activity: activity["launch_start"],
        )
        buckets: list[list[dict]] = [[] for _ in completion_rows]
        for activity in stream_activities:
            bucket = bisect.bisect_left(record_starts, activity["launch_start"])
            if bucket >= len(buckets):
                raise RuntimeError("GPU activity occurs after the last completion event")
            buckets[bucket].append(activity)

        device_ops = sorted(
            (op for op in compute_ops if op["place"]["index"] == device),
            key=lambda op: op["ordinal"],
        )
        if len(device_ops) != len(buckets):
            raise RuntimeError(
                f"device {device}: {len(device_ops)} plan ops but "
                f"{len(buckets)} completion boundaries"
            )

        for sequence, (event, bucket) in enumerate(zip(completion_rows, buckets)):
            if not bucket:
                raise RuntimeError(f"device {device} completion {sequence} has no activity")
            signature = activity_signature(bucket)
            ordinal: int | None = None
            if signature_groups is None:
                # The single-GPU runtime is sequential, so plan order is execution order.
                op = device_ops[sequence]
                group = compute_group(op, values)
                ordinal = op["ordinal"]
            else:
                if signature not in signature_groups:
                    raise RuntimeError(
                        f"device {device} completion {sequence} has an unknown signature"
                    )
                group = signature_groups[signature]

            kernels = [activity for activity in bucket if activity["kind"] == "kernel"]
            launch_api_samples = tuple(
                (activity["launch_end"] - activity["launch_start"]) / 1_000.0
                for activity in kernels
            )

            samples.append(
                Sample(
                    group=group,
                    device=device,
                    event_id=event["eventId"],
                    device_sequence=sequence,
                    span_us=(
                        max(activity["end"] for activity in bucket)
                        - min(activity["start"] for activity in bucket)
                    )
                    / 1_000.0,
                    busy_us=sum(
                        activity["end"] - activity["start"] for activity in bucket
                    )
                    / 1_000.0,
                    kernel_count=len(kernels),
                    kernel_launch_api_us=sum(launch_api_samples),
                    kernel_launch_api_samples_us=launch_api_samples,
                    kernel_submit_window_us=(
                        (
                            max(activity["launch_end"] for activity in kernels)
                            - min(activity["launch_start"] for activity in kernels)
                        )
                        / 1_000.0
                        if kernels
                        else 0.0
                    ),
                    kernel_busy_us=sum(
                        activity["end"] - activity["start"] for activity in kernels
                    )
                    / 1_000.0,
                    signature=signature,
                    ordinal=ordinal,
                )
            )

    connection.close()
    if len(samples) != len(compute_ops):
        raise RuntimeError(
            f"recovered {len(samples)} samples for {len(compute_ops)} compute ops"
        )
    return samples, len(activities)


def learn_signature_groups(samples: list[Sample]) -> dict[Signature, Group]:
    groups: dict[Signature, set[Group]] = collections.defaultdict(set)
    for sample in samples:
        groups[sample.signature].add(sample.group)
    ambiguous = {signature: names for signature, names in groups.items() if len(names) != 1}
    if ambiguous:
        raise RuntimeError(f"ambiguous 1-GPU activity signatures: {ambiguous}")
    return {signature: next(iter(names)) for signature, names in groups.items()}


def expected_group_counts(plan: dict) -> collections.Counter[Group]:
    values = {value["id"]: value for value in plan["values"]}
    return collections.Counter(
        compute_group(action, values)
        for action in all_actions(plan)
        if action["kind"] == "compute"
    )


def observed_group_counts(samples: list[Sample]) -> collections.Counter[Group]:
    return collections.Counter(sample.group for sample in samples)


def group_costs(stats: dict[Group, GroupStats], busy: bool = False) -> dict[Group, float]:
    return {
        group: row.clean_busy_mean_us if busy else row.clean_mean_us
        for group, row in stats.items()
    }


def estimate_schedule(plan: dict, costs: dict[Group, float], devices: int) -> dict:
    values = {value["id"]: value for value in plan["values"]}
    actions = all_actions(plan)
    aliases: dict[str, str] = {}
    for action in actions:
        if action["kind"] == "transfer":
            aliases.update(zip(action["outputs"], action["inputs"]))

    def semantic_value(value_id: str) -> str:
        while value_id in aliases:
            value_id = aliases[value_id]
        return value_id

    ops = [action for action in actions if action["kind"] == "compute"]
    producer = {semantic_value(op["output"]): index for index, op in enumerate(ops)}
    predecessors: list[set[int]] = [set() for _ in ops]
    successors: list[set[int]] = [set() for _ in ops]
    durations: list[float] = []
    for index, op in enumerate(ops):
        durations.append(costs[compute_group(op, values)])
        for input_id in op["inputs"]:
            dependency = producer.get(semantic_value(input_id))
            if dependency is not None and dependency != index:
                if dependency >= index:
                    raise RuntimeError("plan compute operations are not topologically ordered")
                predecessors[index].add(dependency)
                successors[dependency].add(index)

    work = sum(durations)
    critical_finish: list[float] = []
    for index, duration in enumerate(durations):
        critical_finish.append(
            duration
            + max((critical_finish[pred] for pred in predecessors[index]), default=0.0)
        )
    critical_path = max(critical_finish)

    loads = [0.0] * devices
    for index, op in enumerate(ops):
        loads[op["place"]["index"]] += durations[index]

    # A reproducible compact schedule that preserves plan order on each fixed device.
    fixed_predecessors = [set(items) for items in predecessors]
    last_on_device: list[int | None] = [None] * devices
    for index, op in enumerate(ops):
        device = op["place"]["index"]
        if last_on_device[device] is not None:
            fixed_predecessors[index].add(last_on_device[device])
        last_on_device[device] = index
    fixed_finish: list[float] = []
    for index, duration in enumerate(durations):
        fixed_finish.append(
            duration
            + max(
                (fixed_finish[pred] for pred in fixed_predecessors[index]),
                default=0.0,
            )
        )

    # HEFT-style list schedule: free placement, critical-tail priority, zero transfer cost.
    upward_rank = [0.0] * len(ops)
    for index in reversed(range(len(ops))):
        upward_rank[index] = durations[index] + max(
            (upward_rank[succ] for succ in successors[index]), default=0.0
        )
    order = sorted(range(len(ops)), key=lambda index: (-upward_rank[index], index))
    device_available = [0.0] * devices
    greedy_finish = [0.0] * len(ops)
    for index in order:
        dependency_ready = max(
            (greedy_finish[pred] for pred in predecessors[index]), default=0.0
        )
        device = min(
            range(devices),
            key=lambda candidate: max(device_available[candidate], dependency_ready),
        )
        start = max(device_available[device], dependency_ready)
        greedy_finish[index] = start + durations[index]
        device_available[device] = greedy_finish[index]

    return {
        "work_us": work,
        "critical_path_us": critical_path,
        "unrestricted_lower_bound_us": max(work / devices, critical_path),
        "fixed_device_loads_us": loads,
        "fixed_load_bound_us": max(loads),
        "fixed_plan_order_us": max(fixed_finish),
        "free_placement_greedy_us": max(greedy_finish),
    }


def print_stats(title: str, stats: dict[Group, GroupStats]) -> None:
    print(f"\n## {title}")
    print("| op | level | n | mean us | clean mean us | median us | p95 us | max us | outliers |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in sorted(stats):
        row = stats[group]
        op, input_level, output_level = group
        print(
            f"| {op} | {input_level}->{output_level} | {len(row.samples)} "
            f"| {row.mean_us:.3f} | {row.clean_mean_us:.3f} "
            f"| {row.median_us:.3f} | {row.p95_us:.3f} "
            f"| {row.maximum_us:.3f} | {len(row.outliers)} |"
        )
    print("\nTukey outliers (1.5 IQR; groups with n < 4 are not tested):")
    for group in sorted(stats):
        row = stats[group]
        if not row.outliers:
            continue
        labels = []
        for sample in sorted(row.outliers, key=lambda item: item.span_us, reverse=True):
            identity = (
                f"ordinal={sample.ordinal}"
                if sample.ordinal is not None
                else f"device_sequence={sample.device_sequence}"
            )
            labels.append(
                f"gpu={sample.device} event={sample.event_id} {identity} "
                f"span_us={sample.span_us:.3f}"
            )
        print(f"- {group[0]} L{group[1]}->{group[2]}: " + "; ".join(labels))


def metric_inliers(samples: list[Sample], attribute: str) -> list[Sample]:
    if len(samples) < 4:
        return samples
    values = [getattr(sample, attribute) for sample in samples]
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [sample for sample in samples if low <= getattr(sample, attribute) <= high]


def print_launch_comparison(one_samples: list[Sample], four_samples: list[Sample]) -> None:
    print("\n## Kernel launch comparison")
    for label, samples in (("1 GPU", one_samples), ("4 GPU", four_samples)):
        launches = [
            duration
            for sample in samples
            for duration in sample.kernel_launch_api_samples_us
        ]
        print(
            f"- {label}: {len(launches)} launches, "
            f"API total={sum(launches) / 1000.0:.3f} ms, "
            f"mean={statistics.mean(launches):.3f} us, "
            f"median={statistics.median(launches):.3f} us, "
            f"p95={percentile(launches, 0.95):.3f} us, "
            f"p99={percentile(launches, 0.99):.3f} us, "
            f"max={max(launches):.3f} us"
        )

    one_groups: dict[Group, list[Sample]] = collections.defaultdict(list)
    four_groups: dict[Group, list[Sample]] = collections.defaultdict(list)
    for sample in one_samples:
        one_groups[sample.group].append(sample)
    for sample in four_samples:
        four_groups[sample.group].append(sample)

    print(
        "\n| op | level | kernels/op | 1G launch mean us | 4G launch mean us "
        "| ratio | 1G clean us | 4G clean us | ratio | 1G submit clean us "
        "| 4G submit clean us | ratio |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for group in sorted(one_groups):
        one = one_groups[group]
        four = four_groups[group]
        one_launch_inliers = metric_inliers(one, "kernel_launch_api_us")
        four_launch_inliers = metric_inliers(four, "kernel_launch_api_us")
        one_submit_inliers = metric_inliers(one, "kernel_submit_window_us")
        four_submit_inliers = metric_inliers(four, "kernel_submit_window_us")
        one_mean = statistics.mean(sample.kernel_launch_api_us for sample in one)
        four_mean = statistics.mean(sample.kernel_launch_api_us for sample in four)
        one_clean = statistics.mean(
            sample.kernel_launch_api_us for sample in one_launch_inliers
        )
        four_clean = statistics.mean(
            sample.kernel_launch_api_us for sample in four_launch_inliers
        )
        one_submit = statistics.mean(
            sample.kernel_submit_window_us for sample in one_submit_inliers
        )
        four_submit = statistics.mean(
            sample.kernel_submit_window_us for sample in four_submit_inliers
        )

        def ratio(numerator: float, denominator: float) -> str:
            return f"{numerator / denominator:.2f}" if denominator else "n/a"

        print(
            f"| {group[0]} | {group[1]}->{group[2]} "
            f"| {statistics.mean(sample.kernel_count for sample in one):.1f} "
            f"| {one_mean:.3f} | {four_mean:.3f} | {ratio(four_mean, one_mean)} "
            f"| {one_clean:.3f} | {four_clean:.3f} | {ratio(four_clean, one_clean)} "
            f"| {one_submit:.3f} | {four_submit:.3f} "
            f"| {ratio(four_submit, one_submit)} |"
        )


def print_estimate(label: str, estimate: dict) -> None:
    print(f"\n### {label}")
    for key, value in estimate.items():
        if isinstance(value, list):
            rendered = ", ".join(f"{item / 1000.0:.3f}" for item in value)
            print(f"- {key}: [{rendered}] ms")
        else:
            print(f"- {key}: {value / 1000.0:.3f} ms")


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    direct = repository / "test-results/nsight-direct-560ebec"
    artifacts = repository / "test-results/e2e-new-profile-e63ec18/artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-db", type=Path, default=direct / "direct-1gpu.sqlite")
    parser.add_argument(
        "--one-plan",
        type=Path,
        default=artifacts / "1gpu/mlp.optimized._hecate_MLP.runtime-plan.json",
    )
    parser.add_argument("--four-db", type=Path, default=direct / "direct-4gpu.sqlite")
    parser.add_argument(
        "--four-plan",
        type=Path,
        default=artifacts / "4gpu/mlp.optimized._hecate_MLP.runtime-plan.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    one_plan = json.loads(args.one_plan.read_text())
    four_plan = json.loads(args.four_plan.read_text())

    one_samples, one_activity_count = extract_samples(args.one_db, one_plan)
    signature_groups = learn_signature_groups(one_samples)
    four_samples, four_activity_count = extract_samples(
        args.four_db, four_plan, signature_groups
    )
    if expected_group_counts(one_plan) != observed_group_counts(one_samples):
        raise RuntimeError("1-GPU recovered group counts do not match its plan")
    if expected_group_counts(four_plan) != observed_group_counts(four_samples):
        raise RuntimeError("4-GPU recovered group counts do not match its plan")

    one_stats = summarize(one_samples)
    four_stats = summarize(four_samples)
    print(
        "Validation: "
        f"1-GPU {len(one_samples)} ops/{one_activity_count} activities; "
        f"4-GPU {len(four_samples)} ops/{four_activity_count} activities; "
        f"{len(signature_groups)} unique unambiguous op-level signatures."
    )
    print_stats("1 GPU", one_stats)
    print_stats("4 GPU", four_stats)
    print_launch_comparison(one_samples, four_samples)

    print("\n## No-communication compact-schedule estimates")
    print_estimate(
        "1 GPU, outlier-filtered measured operator spans",
        estimate_schedule(one_plan, group_costs(one_stats), 1),
    )
    print_estimate(
        "1 GPU, GPU-activity busy-time floor",
        estimate_schedule(one_plan, group_costs(one_stats, busy=True), 1),
    )
    print_estimate(
        "4 GPU, outlier-filtered measured 4-GPU operator spans",
        estimate_schedule(four_plan, group_costs(four_stats), 4),
    )
    print_estimate(
        "4 GPU, isolated 1-GPU operator spans replicated across four GPUs",
        estimate_schedule(four_plan, group_costs(one_stats), 4),
    )
    print_estimate(
        "4 GPU, GPU-activity busy-time floor",
        estimate_schedule(four_plan, group_costs(four_stats, busy=True), 4),
    )


if __name__ == "__main__":
    main()
