#!/usr/bin/env python3
"""Analyze the online runtime portion of the direct 1-GPU/4-GPU Nsight traces.

The online windows are anchored at the end of the initialization cleanup seen in
Nsight and use the exact online_execution_seconds reported by the executable.
They correspond to Runtime::run() from online_execution_start through
synchronize_final_outputs().
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/poseidon-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "test-results" / "nsight-direct-560ebec"
PLAN_DIR = ROOT / "test-results" / "e2e-new-profile-e63ec18" / "artifacts"
OUT_DIR = TRACE_DIR / "online-analysis"


@dataclass(frozen=True)
class ProfileSpec:
    label: str
    db_path: Path
    report_path: Path
    plan_path: Path
    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


PROFILES = {
    "1GPU": ProfileSpec(
        "1GPU",
        TRACE_DIR / "direct-1gpu.sqlite",
        TRACE_DIR / "direct-1gpu.json",
        PLAN_DIR / "1gpu" / "mlp.optimized._hecate_MLP.runtime-plan.json",
        23_553_067_959,
        23_627_879_758,
    ),
    "4GPU": ProfileSpec(
        "4GPU",
        TRACE_DIR / "direct-4gpu.sqlite",
        TRACE_DIR / "direct-4gpu.json",
        PLAN_DIR / "4gpu" / "mlp.optimized._hecate_MLP.runtime-plan.json",
        24_579_446_692,
        24_644_462_381,
    ),
}


def configure_fonts() -> None:
    font_path = Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#f7f9fc"
    plt.rcParams["axes.facecolor"] = "white"


def tid(global_tid: int) -> int:
    return int(global_tid) & 0xFFFFFF


def clipped(start: int, end: int, lo: int, hi: int) -> tuple[int, int] | None:
    start = max(start, lo)
    end = min(end, hi)
    return (start, end) if end > start else None


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((int(s), int(e)) for s, e in intervals if e > s)
    if not ordered:
        return []
    out = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = out[-1]
        if start <= prev_end:
            out[-1] = (prev_start, max(prev_end, end))
        else:
            out.append((start, end))
    return out


def interval_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_duration(
    first: Sequence[tuple[int, int]], second: Sequence[tuple[int, int]]
) -> int:
    i = j = 0
    total = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if end > start:
            total += end - start
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return total


def subtract_intervals(
    base: Sequence[tuple[int, int]], remove: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    j = 0
    for start, end in base:
        cursor = start
        while j < len(remove) and remove[j][1] <= cursor:
            j += 1
        k = j
        while k < len(remove) and remove[k][0] < end:
            rem_start, rem_end = remove[k]
            if rem_start > cursor:
                result.append((cursor, min(rem_start, end)))
            cursor = max(cursor, rem_end)
            if cursor >= end:
                break
            k += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def overlaps_any(start: int, end: int, merged: Sequence[tuple[int, int]]) -> bool:
    if not merged:
        return False
    starts = [item[0] for item in merged]
    index = bisect_right(starts, end - 1) - 1
    return index >= 0 and merged[index][1] > start


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


class ProfileData:
    def __init__(self, spec: ProfileSpec):
        self.spec = spec
        self.conn = sqlite3.connect(spec.db_path)
        self.conn.row_factory = sqlite3.Row
        self.report = json.loads(spec.report_path.read_text())
        self.plan = json.loads(spec.plan_path.read_text())
        self.values = {str(item["id"]): item for item in self.plan["values"]}
        self.kernels = self._load_kernels()
        self.p2p = self._load_p2p()
        self.compute_intervals = self._compute_intervals()
        self.p2p_intervals = self._p2p_intervals()
        for kernel in self.kernels:
            kernel["p2p_overlap"] = overlaps_any(
                kernel["start"], kernel["end"],
                self.p2p_intervals.get(kernel["device"], []),
            )
        self.thread_roles = self._thread_roles()
        self.cpu_intervals = self._categorized_cpu_intervals()
        self.cpu_breakdown = self._cpu_breakdown()
        self.key_ranges = self._map_keyswitch_ranges()
        self._assign_keyswitch_levels()

    def close(self) -> None:
        self.conn.close()

    def _load_kernels(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT k.start, k.end, k.deviceId, k.correlationId,
                   k.gridX, k.gridY, k.gridZ,
                   k.blockX, k.blockY, k.blockZ,
                   k.dynamicSharedMemory, k.staticSharedMemory,
                   short.value AS short_name,
                   r.start AS api_start, r.end AS api_end,
                   r.globalTid AS api_global_tid
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds short ON short.id = k.shortName
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
              ON r.correlationId = k.correlationId
            JOIN StringIds api_name ON api_name.id = r.nameId
            WHERE k.start < ? AND k.end > ?
              AND api_name.value LIKE 'cudaLaunchKernel%'
            ORDER BY k.start
            """,
            (self.spec.end_ns, self.spec.start_ns),
        ).fetchall()
        kernels = []
        for row in rows:
            gpu_interval = clipped(row["start"], row["end"], self.spec.start_ns, self.spec.end_ns)
            if gpu_interval is None:
                continue
            kernels.append(
                {
                    "start": gpu_interval[0],
                    "end": gpu_interval[1],
                    "device": int(row["deviceId"]),
                    "correlation": int(row["correlationId"]),
                    "name": row["short_name"],
                    "grid": (int(row["gridX"]), int(row["gridY"]), int(row["gridZ"])),
                    "block": (int(row["blockX"]), int(row["blockY"]), int(row["blockZ"])),
                    "dynamic_shared": int(row["dynamicSharedMemory"]),
                    "static_shared": int(row["staticSharedMemory"]),
                    "api_start": int(row["api_start"]),
                    "api_end": int(row["api_end"]),
                    "api_tid": tid(row["api_global_tid"]),
                    "gpu_us": (gpu_interval[1] - gpu_interval[0]) / 1e3,
                    "launch_us": (row["api_end"] - row["api_start"]) / 1e3,
                    "level": None,
                    "logical_op": None,
                }
            )
        return kernels

    def _load_p2p(self) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT start, end, bytes, srcDeviceId, dstDeviceId, correlationId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind = 10 AND start < ? AND end > ?
            ORDER BY start
            """,
            (self.spec.end_ns, self.spec.start_ns),
        ).fetchall()
        result = []
        for row in rows:
            interval = clipped(row["start"], row["end"], self.spec.start_ns, self.spec.end_ns)
            if interval is None:
                continue
            result.append(
                {
                    "start": interval[0],
                    "end": interval[1],
                    "bytes": int(row["bytes"]),
                    "src": int(row["srcDeviceId"]),
                    "dst": int(row["dstDeviceId"]),
                    "correlation": int(row["correlationId"]),
                }
            )
        return result

    def _compute_intervals(self) -> dict[int, list[tuple[int, int]]]:
        by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for kernel in self.kernels:
            by_device[kernel["device"]].append((kernel["start"], kernel["end"]))
        return {device: merge_intervals(items) for device, items in by_device.items()}

    def _p2p_intervals(self) -> dict[int, list[tuple[int, int]]]:
        by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for copy in self.p2p:
            interval = (copy["start"], copy["end"])
            by_device[copy["src"]].append(interval)
            by_device[copy["dst"]].append(interval)
        return {device: merge_intervals(items) for device, items in by_device.items()}

    def _thread_roles(self) -> dict[int, str]:
        launch_counts = Counter(kernel["api_tid"] for kernel in self.kernels)
        peer_api_counts = Counter()
        rows = self.conn.execute(
            """
            SELECT r.globalTid, COUNT(*) AS count
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id = r.nameId
            WHERE r.start < ? AND r.end > ?
              AND s.value LIKE 'cudaMemcpyPeer%'
            GROUP BY r.globalTid
            """,
            (self.spec.end_ns, self.spec.start_ns),
        ).fetchall()
        for row in rows:
            peer_api_counts[tid(row["globalTid"])] = int(row["count"])

        if self.spec.label == "1GPU":
            main_tid = launch_counts.most_common(1)[0][0]
            return {main_tid: "1GPU 主线程"}

        roles: dict[int, str] = {}
        compute_device: dict[int, int] = {}
        counts: dict[int, Counter] = defaultdict(Counter)
        for kernel in self.kernels:
            counts[kernel["api_tid"]][kernel["device"]] += 1
        for thread_id, device_counts in counts.items():
            compute_device[thread_id] = device_counts.most_common(1)[0][0]
            roles[thread_id] = f"GPU{compute_device[thread_id]} 计算线程"
        for thread_id in peer_api_counts:
            previous = thread_id - 1
            device = compute_device.get(previous)
            roles[thread_id] = f"GPU{device} 通信线程" if device is not None else f"通信线程 TID {thread_id}"

        main_row = self.conn.execute(
            """
            SELECT o.globalTid, COUNT(*) AS calls
            FROM OSRT_API o JOIN StringIds s ON s.id=o.nameId
            WHERE o.start < ? AND o.end > ? AND s.value='pthread_create'
            GROUP BY o.globalTid ORDER BY calls DESC LIMIT 1
            """,
            (self.spec.end_ns, self.spec.start_ns),
        ).fetchone()
        if main_row is not None:
            roles[tid(main_row["globalTid"])] = "4GPU 主线程"
        return roles

    def _categorized_cpu_intervals(
        self,
    ) -> dict[int, dict[str, list[tuple[int, int]]]]:
        """Split each selected CPU thread's online window into exclusive classes.

        Priority is launch > cond_wait > other_visible > blank.  In particular,
        an OSRT mutex nested inside cudaLaunchKernel remains launch time instead
        of being counted twice as another visible event.
        """
        runtime_rows = self.conn.execute(
            """
            SELECT r.start, r.end, r.globalTid, s.value AS name
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
            WHERE r.start < ? AND r.end > ?
            """,
            (self.spec.end_ns, self.spec.start_ns),
        ).fetchall()
        osrt_rows = self.conn.execute(
            """
            SELECT o.start, o.end, o.globalTid, s.value AS name
            FROM OSRT_API o JOIN StringIds s ON s.id=o.nameId
            WHERE o.start < ? AND o.end > ?
            """,
            (self.spec.end_ns, self.spec.start_ns),
        ).fetchall()
        visible: dict[int, list[tuple[int, int]]] = defaultdict(list)
        launch: dict[int, list[tuple[int, int]]] = defaultdict(list)
        cond: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in runtime_rows:
            thread_id = tid(row["globalTid"])
            if thread_id not in self.thread_roles:
                continue
            interval = clipped(row["start"], row["end"], self.spec.start_ns, self.spec.end_ns)
            if interval is None:
                continue
            visible[thread_id].append(interval)
            if row["name"].startswith("cudaLaunchKernel"):
                launch[thread_id].append(interval)
        for row in osrt_rows:
            thread_id = tid(row["globalTid"])
            if thread_id not in self.thread_roles:
                continue
            interval = clipped(row["start"], row["end"], self.spec.start_ns, self.spec.end_ns)
            if interval is None:
                continue
            visible[thread_id].append(interval)
            if row["name"] in {"pthread_cond_wait", "pthread_cond_timedwait"}:
                cond[thread_id].append(interval)

        categorized = {}
        full_window = [(self.spec.start_ns, self.spec.end_ns)]
        for thread_id in self.thread_roles:
            launch_u = merge_intervals(launch[thread_id])
            cond_u = subtract_intervals(merge_intervals(cond[thread_id]), launch_u)
            higher = merge_intervals([*launch_u, *cond_u])
            visible_u = merge_intervals(visible[thread_id])
            other_u = subtract_intervals(visible_u, higher)
            all_visible_u = merge_intervals([*launch_u, *cond_u, *other_u])
            categorized[thread_id] = {
                "launch": launch_u,
                "cond_wait": cond_u,
                "other_visible": other_u,
                "blank": subtract_intervals(full_window, all_visible_u),
            }
        return categorized

    def _cpu_breakdown(self) -> list[dict]:
        rows = []
        role_order = {"1GPU 主线程": 0, "4GPU 主线程": 0}
        for device in range(4):
            role_order[f"GPU{device} 计算线程"] = 1 + device * 2
            role_order[f"GPU{device} 通信线程"] = 2 + device * 2
        for thread_id, role in self.thread_roles.items():
            categories = self.cpu_intervals[thread_id]
            launch_ns = interval_duration(categories["launch"])
            cond_ns = interval_duration(categories["cond_wait"])
            other_ns = interval_duration(categories["other_visible"])
            blank_ns = interval_duration(categories["blank"])
            rows.append(
                {
                    "profile": self.spec.label,
                    "tid": thread_id,
                    "role": role,
                    "launch_ms": launch_ns / 1e6,
                    "cond_wait_ms": cond_ns / 1e6,
                    "other_visible_ms": other_ns / 1e6,
                    "blank_ms": blank_ns / 1e6,
                    "launch_pct": 100 * launch_ns / self.spec.duration_ns,
                    "cond_wait_pct": 100 * cond_ns / self.spec.duration_ns,
                    "other_visible_pct": 100 * other_ns / self.spec.duration_ns,
                    "blank_pct": 100 * blank_ns / self.spec.duration_ns,
                    "sort": role_order.get(role, 99),
                }
            )
        rows.sort(key=lambda item: item["sort"])
        return rows

    def _logical_key_actions(self) -> dict[int, list[dict]]:
        result: dict[int, list[dict]] = defaultdict(list)
        for action in self.plan["execution"]:
            if action.get("kind") != "compute" or action.get("op") not in {"rotate", "relinearize"}:
                continue
            device = int(action["place"]["index"])
            input_desc = self.values[str(action["inputs"][0])]
            result[device].append(
                {
                    "ordinal": int(action["ordinal"]),
                    "op": action["op"],
                    "level": int(input_desc["level"]),
                }
            )
        return result

    def _map_keyswitch_ranges(self) -> dict[int, list[dict]]:
        raw: dict[int, list[dict]] = defaultdict(list)
        rows = self.conn.execute(
            """
            SELECT start, end, globalTid FROM NVTX_EVENTS
            WHERE text='keyswitch.hybrid' AND start < ? AND end > ?
            ORDER BY globalTid, start
            """,
            (self.spec.end_ns, self.spec.start_ns),
        ).fetchall()
        for row in rows:
            raw[tid(row["globalTid"])].append(
                {"start": int(row["start"]), "end": int(row["end"])}
            )

        actions = self._logical_key_actions()
        device_to_tid = {}
        for thread_id, role in self.thread_roles.items():
            if role.startswith("GPU") and role.endswith("计算线程"):
                device_to_tid[int(role.removeprefix("GPU").split()[0])] = thread_id
            elif role == "1GPU 主线程":
                device_to_tid[0] = thread_id
        mapped: dict[int, list[dict]] = defaultdict(list)
        for device, device_actions in actions.items():
            thread_id = device_to_tid[device]
            ranges = raw[thread_id]
            if len(ranges) != len(device_actions):
                raise RuntimeError(
                    f"{self.spec.label} GPU{device}: {len(ranges)} keyswitch ranges != "
                    f"{len(device_actions)} rotate/relinearize actions"
                )
            for range_item, action in zip(ranges, device_actions):
                mapped[thread_id].append({**range_item, **action, "device": device})
        return mapped

    def _assign_keyswitch_levels(self) -> None:
        ranges_by_tid = self.key_ranges
        starts_by_tid = {
            thread_id: [item["start"] for item in ranges]
            for thread_id, ranges in ranges_by_tid.items()
        }
        for kernel in self.kernels:
            ranges = ranges_by_tid.get(kernel["api_tid"], [])
            if not ranges:
                continue
            index = bisect_right(starts_by_tid[kernel["api_tid"]], kernel["api_start"]) - 1
            if index < 0:
                continue
            item = ranges[index]
            if kernel["api_start"] >= item["start"] and kernel["api_end"] <= item["end"]:
                kernel["level"] = item["level"]
                kernel["logical_op"] = item["op"]

    def op_level_counts(self) -> Counter:
        counts = Counter()
        for action in self.plan["execution"]:
            if action.get("kind") != "compute":
                continue
            input_level = self.values[str(action["inputs"][0])].get("level") if action.get("inputs") else None
            output_level = self.values[str(action["output"])].get("level") if action.get("output") is not None else None
            device = int(action["place"].get("index", -1))
            counts[(action["op"], input_level, output_level, device)] += 1
        return counts


def signature(kernel: dict, include_level: bool = False) -> tuple:
    item = (
        kernel["name"], kernel["grid"], kernel["block"],
        kernel["dynamic_shared"], kernel["static_shared"],
    )
    return (*item, kernel["level"]) if include_level else item


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_p2p_coverage(data: ProfileData) -> list[dict]:
    result = []
    for device in sorted(data.compute_intervals):
        compute = data.compute_intervals[device]
        for role in ("Rx", "Tx"):
            copies = [
                item for item in data.p2p
                if (role == "Rx" and item["dst"] == device)
                or (role == "Tx" and item["src"] == device)
            ]
            comm = merge_intervals((item["start"], item["end"]) for item in copies)
            wall_ns = interval_duration(comm)
            overlap_ns = intersection_duration(comm, compute)
            result.append(
                {
                    "device": device,
                    "role": role,
                    "copies": len(copies),
                    "bytes": sum(item["bytes"] for item in copies),
                    "service_sum_ms": sum(item["end"] - item["start"] for item in copies) / 1e6,
                    "wall_union_ms": wall_ns / 1e6,
                    "overlap_ms": overlap_ns / 1e6,
                    "uncovered_ms": (wall_ns - overlap_ns) / 1e6,
                    "coverage_pct": 100 * overlap_ns / wall_ns if wall_ns else float("nan"),
                }
            )
    return result


def build_p2p_matrix(data: ProfileData) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in data.p2p:
        grouped[(item["src"], item["dst"])].append(item)
    rows = []
    for (src, dst), items in sorted(grouped.items()):
        total_ns = sum(item["end"] - item["start"] for item in items)
        total_bytes = sum(item["bytes"] for item in items)
        rows.append(
            {
                "src": src,
                "dst": dst,
                "copies": len(items),
                "bytes": total_bytes,
                "service_ms": total_ns / 1e6,
                "effective_GBps": total_bytes / total_ns if total_ns else float("nan"),
            }
        )
    return rows


def build_op_comparison(one: ProfileData, four: ProfileData) -> list[dict]:
    one_counts = one.op_level_counts()
    four_counts = four.op_level_counts()
    keys = sorted(
        {(op, in_level, out_level) for op, in_level, out_level, _ in one_counts}
        | {(op, in_level, out_level) for op, in_level, out_level, _ in four_counts}
    )
    rows = []
    for op, input_level, output_level in keys:
        one_total = sum(count for (o, i, out, _), count in one_counts.items() if (o, i, out) == (op, input_level, output_level))
        by_device = [
            four_counts.get((op, input_level, output_level, device), 0)
            for device in range(4)
        ]
        rows.append(
            {
                "op": op,
                "input_level": input_level,
                "output_level": output_level,
                "1gpu_total": one_total,
                "4gpu_total": sum(by_device),
                "gpu0": by_device[0],
                "gpu1": by_device[1],
                "gpu2": by_device[2],
                "gpu3": by_device[3],
                "consistent": one_total == sum(by_device),
            }
        )
    return rows


def build_signature_counts(one: ProfileData, four: ProfileData) -> list[dict]:
    count_one = Counter(signature(item) for item in one.kernels)
    count_four = Counter(signature(item) for item in four.kernels)
    rows = []
    for key in sorted(set(count_one) | set(count_four), key=lambda item: (item[0], item[1])):
        name, grid, block, dynamic_shared, static_shared = key
        rows.append(
            {
                "kernel": name,
                "grid": "x".join(map(str, grid)),
                "block": "x".join(map(str, block)),
                "dynamic_shared": dynamic_shared,
                "static_shared": static_shared,
                "1gpu_count": count_one[key],
                "4gpu_count": count_four[key],
                "consistent": count_one[key] == count_four[key],
            }
        )
    return rows


def build_level_comparison(one: ProfileData, four: ProfileData) -> list[dict]:
    groups_one: dict[tuple[str, int], list[dict]] = defaultdict(list)
    groups_four: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in one.kernels:
        if item["level"] is not None:
            groups_one[(item["name"], item["level"])].append(item)
    for item in four.kernels:
        if item["level"] is not None:
            groups_four[(item["name"], item["level"])].append(item)
    rows = []
    for key in sorted(set(groups_one) | set(groups_four)):
        first = groups_one.get(key, [])
        second = groups_four.get(key, [])
        no_p2p = [item for item in second if not item["p2p_overlap"]]
        with_p2p = [item for item in second if item["p2p_overlap"]]
        gpu1 = [item["gpu_us"] for item in first]
        launch1 = [item["launch_us"] for item in first]
        gpu4 = [item["gpu_us"] for item in second]
        launch4 = [item["launch_us"] for item in second]
        gpu4_no = [item["gpu_us"] for item in no_p2p]
        gpu4_yes = [item["gpu_us"] for item in with_p2p]
        base_gpu_p50 = percentile(gpu1, 50)
        base_gpu_mean = float(np.mean(gpu1)) if gpu1 else float("nan")
        gpu4_mean = float(np.mean(gpu4)) if gpu4 else float("nan")
        base_launch50 = percentile(launch1, 50)
        base_launch95 = percentile(launch1, 95)
        rows.append(
            {
                "kernel": key[0],
                "level": key[1],
                "1gpu_count": len(first),
                "4gpu_count": len(second),
                "4gpu_no_p2p_count": len(no_p2p),
                "4gpu_p2p_count": len(with_p2p),
                "1gpu_gpu_mean_us": base_gpu_mean,
                "4gpu_gpu_mean_us": gpu4_mean,
                "gpu_all_mean_ratio": gpu4_mean / base_gpu_mean if base_gpu_mean else float("nan"),
                "1gpu_gpu_p50_us": base_gpu_p50,
                "4gpu_gpu_p50_us": percentile(gpu4, 50),
                "4gpu_no_p2p_gpu_p50_us": percentile(gpu4_no, 50),
                "4gpu_p2p_gpu_p50_us": percentile(gpu4_yes, 50),
                "gpu_no_p2p_ratio": percentile(gpu4_no, 50) / base_gpu_p50 if base_gpu_p50 and gpu4_no else float("nan"),
                "gpu_p2p_ratio": percentile(gpu4_yes, 50) / base_gpu_p50 if base_gpu_p50 and gpu4_yes else float("nan"),
                "1gpu_launch_p50_us": base_launch50,
                "4gpu_launch_p50_us": percentile(launch4, 50),
                "launch_p50_ratio": percentile(launch4, 50) / base_launch50 if base_launch50 else float("nan"),
                "1gpu_launch_p95_us": base_launch95,
                "4gpu_launch_p95_us": percentile(launch4, 95),
                "launch_p95_ratio": percentile(launch4, 95) / base_launch95 if base_launch95 else float("nan"),
                "1gpu_gpu_total_ms": sum(gpu1) / 1e3,
            }
        )
    rows.sort(key=lambda item: item["1gpu_gpu_total_ms"], reverse=True)
    return rows


def draw_timeline(four: ProfileData, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    green = "#27a77b"
    tx_color = "#f59e0b"
    rx_color = "#ef6a55"
    yticks = []
    ylabels = []
    for device in range(4):
        base = (3 - device) * 2.0
        compute_y = base + 0.85
        copy_y = base + 0.20
        kernels = [item for item in four.kernels if item["device"] == device]
        ax.broken_barh(
            [((item["start"] - four.spec.start_ns) / 1e6, (item["end"] - item["start"]) / 1e6) for item in kernels],
            (compute_y, 0.42), facecolors=green, linewidth=0,
        )
        tx = [item for item in four.p2p if item["src"] == device]
        rx = [item for item in four.p2p if item["dst"] == device]
        ax.broken_barh(
            [((item["start"] - four.spec.start_ns) / 1e6, (item["end"] - item["start"]) / 1e6) for item in tx],
            (copy_y, 0.24), facecolors=tx_color, linewidth=0,
        )
        ax.broken_barh(
            [((item["start"] - four.spec.start_ns) / 1e6, (item["end"] - item["start"]) / 1e6) for item in rx],
            (copy_y + 0.27, 0.24), facecolors=rx_color, linewidth=0,
        )
        yticks.extend([compute_y + 0.21, copy_y + 0.25])
        ylabels.extend([
            f"GPU{device} 计算（kernel）",
            f"GPU{device} P2P通信（Tx/Rx）",
        ])
        ax.axhline(base + 1.55, color="#e5eaf1", linewidth=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlim(0, four.spec.duration_ns / 1e6)
    ax.set_ylim(-0.1, 8.0)
    ax.set_xlabel("相对 runtime online 起点的时间（ms）")
    fig.suptitle(
        "4GPU runtime online：GPU device 时间轴（真实数据）",
        x=0.095, y=0.98, ha="left", fontsize=19, fontweight="bold",
    )
    fig.text(
        0.095, 0.925,
        f"online = {four.spec.duration_ns / 1e6:.3f}ms；绿色=kernel执行，黄色=从本GPU发送(Tx)，红色=本GPU接收(Rx)；横向重叠表示GPU物理并发。",
        color="#64748b",
    )
    ax.grid(axis="x", color="#e4e9f0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_coverage(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda item: (item["device"], 0 if item["role"] == "Rx" else 1), reverse=True)
    role_label = {"Rx": "接收 Rx（目标端）", "Tx": "发送 Tx（源端）"}
    labels = [f"GPU{item['device']} {role_label[item['role']]}" for item in ordered]
    overlap = np.array([item["overlap_ms"] for item in ordered])
    uncovered = np.array([item["uncovered_ms"] for item in ordered])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.barh(y, overlap, color="#25a18e", label="被本GPU kernel执行掩盖（物理重叠）")
    ax.barh(y, uncovered, left=overlap, color="#f3a54a", label="暴露：未与本GPU kernel重叠")
    for index, item in enumerate(ordered):
        total = item["wall_union_ms"]
        ax.text(total + 0.03, index, f"{item['coverage_pct']:.1f}%  ({item['overlap_ms']:.3f}/{total:.3f}ms)", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("P2P activity 的 wall-time 区间并集（ms，不重复累计并发 copy）")
    ax.set_title("4GPU runtime online：P2P通信被本GPU计算掩盖的时间（真实数据）", loc="left", fontsize=19, fontweight="bold", pad=28)
    ax.text(
        0, 1.01,
        "Rx 以目标GPU为分母，Tx 以源GPU为分母；标注为 覆盖率（重叠时间 / 该方向P2P wall time）。",
        transform=ax.transAxes, color="#64748b", fontsize=10,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#e4e9f0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlim(0, max(overlap + uncovered) * 1.55)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_cpu(one: ProfileData, four: ProfileData, path: Path) -> None:
    rows = one.cpu_breakdown + four.cpu_breakdown
    labels = [item["role"] for item in rows]
    values = np.array(
        [[item["launch_pct"], item["cond_wait_pct"], item["other_visible_pct"], item["blank_pct"]] for item in rows]
    )
    colors = ["#4f7cff", "#8758ef", "#8a97aa", "#e8edf3"]
    names = ["cudaLaunchKernel API", "pthread_cond_wait", "其他 CUDA/OSRT 可见活动", "无 CUDA/OSRT 可见事件"]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
    left = np.zeros(len(rows))
    for column in range(values.shape[1]):
        bars = ax.barh(y, values[:, column], left=left, color=colors[column], label=names[column], edgecolor="white", linewidth=0.5)
        for index, bar in enumerate(bars):
            width = values[index, column]
            if width >= 7:
                text_color = "#1f2937" if column == 3 else "white"
                ax.text(left[index] + width / 2, index, f"{width:.1f}%", ha="center", va="center", fontsize=8.5, color=text_color, fontweight="bold")
        left += values[:, column]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("占该 CPU host thread 的 runtime-online wall window 比例（每行独立为 100%）")
    fig.suptitle(
        "Runtime online：CPU host thread 时间构成（真实数据）",
        x=0.095, y=0.98, ha="left", fontsize=19, fontweight="bold",
    )
    fig.text(
        0.095, 0.925,
        "计算线程负责发射本GPU kernel；通信线程负责P2P任务。Launch含内部driver mutex；主线程“其他”主要是pthread_join；各行不能相加。",
        color="#64748b",
    )
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="x", color="#e4e9f0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.89))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_cpu_timeline(one: ProfileData, four: ProfileData, path: Path) -> None:
    categories = [
        ("launch", "cudaLaunchKernel API", "#4f7cff"),
        ("cond_wait", "pthread_cond_wait", "#8758ef"),
        ("other_visible", "其他 CUDA/OSRT 可见活动", "#8a97aa"),
        ("blank", "无 CUDA/OSRT 可见事件", "#e8edf3"),
    ]
    role_orders = [
        ["1GPU 主线程"],
        ["4GPU 主线程", *[f"GPU{device} {kind}线程" for device in range(4) for kind in ("计算", "通信")]],
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 9.5), dpi=150,
        gridspec_kw={"height_ratios": [1.15, 5.0]},
    )
    legend_handles = []
    for panel_index, (ax, data, desired_roles) in enumerate(
        zip(axes, (one, four), role_orders)
    ):
        role_to_tid = {role: thread_id for thread_id, role in data.thread_roles.items()}
        plotted_roles = [role for role in desired_roles if role in role_to_tid]
        for y, role in enumerate(plotted_roles):
            thread_id = role_to_tid[role]
            for category, label, color in categories:
                spans = [
                    ((start - data.spec.start_ns) / 1e6, (end - start) / 1e6)
                    for start, end in data.cpu_intervals[thread_id][category]
                ]
                collection = ax.broken_barh(
                    spans, (y - 0.36, 0.72),
                    facecolors=color, edgecolors="none",
                    label=label if panel_index == 0 and y == 0 else None,
                )
                if panel_index == 0 and y == 0:
                    legend_handles.append(collection)
        ax.set_yticks(np.arange(len(plotted_roles)))
        ax.set_yticklabels(plotted_roles)
        ax.set_ylim(len(plotted_roles) - 0.5, -0.5)
        ax.set_xlim(0, data.spec.duration_ns / 1e6)
        ax.set_title(
            f"{data.spec.label} CPU host threads  ·  online {data.spec.duration_ns / 1e6:.3f} ms",
            loc="left", fontsize=12, fontweight="bold", pad=8,
        )
        ax.set_xlabel("相对各自 runtime online 起点的时间（ms）")
        ax.grid(axis="x", color="#dce3ec", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "Runtime online CPU 时间轴（真实数据）",
        x=0.09, y=0.985, ha="left", fontsize=19, fontweight="bold",
    )
    fig.text(
        0.09, 0.945,
        "计算线程发射本GPU kernel，通信线程处理P2P任务。优先级：launch > cond_wait > 其他 > blank；blank仅表示无CUDA/OSRT事件，不等价于CPU空闲。",
        color="#64748b", fontsize=10,
    )
    fig.legend(
        legend_handles, [item[1] for item in categories],
        ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.012),
    )
    fig.tight_layout(rect=(0.03, 0.07, 1, 0.91), h_pad=2.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def short_kernel_name(name: str) -> str:
    replacements = {
        "inverse_ntt_fourstep_phase_kernel": "inverse_ntt_fourstep",
        "hybrid_forward_ntt_modup_qp_fused_stage_kernel": "hybrid_fwd_ntt_fused",
        "hybrid_forward_ntt_modup_qp_final_mul_accumulate_fused_decomp_q_kernel": "hybrid_final_mul_accum",
        "hybrid_forward_ntt_q_two_components_fused_stage_kernel": "hybrid_fwd_ntt_q_c2",
        "hybrid_modup_qp_forward_ntt_first_stage_row_tiled8_kernel": "modup_first_tiled8",
        "hybrid_convert_p_to_q_forward_ntt_first_stage_kernel": "convert_p_to_q_first",
        "hybrid_apply_moddown_ntt_add_back_two_components_kernel": "moddown_add_back_c2",
        "hybrid_modup_qp_forward_ntt_first_stage_kernel": "modup_first",
    }
    return replacements.get(name, name.removesuffix("_kernel"))


def draw_consistency(rows: list[dict], path: Path) -> None:
    display = [item for item in rows if item["1gpu_count"] >= 8][:14]
    columns = ["gpu_all_mean_ratio", "gpu_no_p2p_ratio", "gpu_p2p_ratio", "launch_p50_ratio", "launch_p95_ratio"]
    labels = [
        "GPU执行时间 Mean\n全部样本",
        "GPU执行时间 P50\n未与P2P重叠",
        "GPU执行时间 P50\n与P2P重叠",
        "CPU Launch API时间\nP50",
        "CPU Launch API时间\nP95",
    ]
    matrix = np.array([[item[column] for column in columns] for item in display], dtype=float)
    fig_height = max(6, 0.5 * len(display) + 2.3)
    fig, ax = plt.subplots(figsize=(13, fig_height), dpi=150)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#e5e7eb")
    image = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=0.75, vcenter=1.0, vmax=max(3.0, np.nanmax(matrix))), aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(display)))
    ax.set_yticklabels([f"{short_kernel_name(item['kernel'])} @ L{item['level']}  (n={item['1gpu_count']})" for item in display], fontsize=9)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text = "—" if np.isnan(value) else f"{value:.2f}×"
            if not np.isnan(value) and column_index == 1:
                text += f"\nn={display[row_index]['4gpu_no_p2p_count']}"
            elif not np.isnan(value) and column_index == 2:
                text += f"\nn={display[row_index]['4gpu_p2p_count']}"
            ax.text(column_index, row_index, text, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.axvline(2.5, color="#334155", linewidth=3.0)
    ax.set_title(
        "同 level keyswitch kernel：4GPU / 1GPU 时间比（真实数据）",
        loc="left", fontsize=18, fontweight="bold", pad=50,
    )
    ax.text(
        0, 1.055,
        "GPU执行：全部样本用 Mean，P2P子集用 P50（并标样本数）    |    CPU发射：cudaLaunchKernel API 的 P50 / P95",
        transform=ax.transAxes, color="#334155", fontsize=10.5, fontweight="bold",
    )
    ax.text(
        0, 1.015,
        "每格均为同kernel、同level的 4GPU / 1GPU；前三列来自GPU device timeline，后两列来自CPU host thread；覆盖4432/5099个kernel。",
        transform=ax.transAxes, color="#64748b", fontsize=9.5,
    )
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="4GPU / 1GPU")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    one = ProfileData(PROFILES["1GPU"])
    four = ProfileData(PROFILES["4GPU"])
    try:
        coverage = build_p2p_coverage(four)
        matrix = build_p2p_matrix(four)
        op_rows = build_op_comparison(one, four)
        signature_rows = build_signature_counts(one, four)
        level_rows = build_level_comparison(one, four)

        write_csv(OUT_DIR / "p2p_coverage.csv", list(coverage[0].keys()), coverage)
        write_csv(OUT_DIR / "p2p_matrix.csv", list(matrix[0].keys()), matrix)
        write_csv(OUT_DIR / "op_level_counts.csv", list(op_rows[0].keys()), op_rows)
        write_csv(OUT_DIR / "kernel_signature_counts.csv", list(signature_rows[0].keys()), signature_rows)
        write_csv(OUT_DIR / "kernel_level_comparison.csv", list(level_rows[0].keys()), level_rows)
        cpu_rows = one.cpu_breakdown + four.cpu_breakdown
        cpu_fields = [key for key in cpu_rows[0].keys() if key != "sort"]
        write_csv(OUT_DIR / "cpu_thread_breakdown.csv", cpu_fields, cpu_rows)

        launch_summary = {}
        for data in (one, four):
            durations = [item["launch_us"] for item in data.kernels]
            launch_summary[data.spec.label] = {
                "count": len(durations),
                "total_thread_ms": sum(durations) / 1e3,
                "mean_us": float(np.mean(durations)),
                "p50_us": percentile(durations, 50),
                "p95_us": percentile(durations, 95),
                "p99_us": percentile(durations, 99),
                "max_us": max(durations),
            }
        baseline_by_signature: dict[tuple, float] = {}
        one_by_signature: dict[tuple, list[float]] = defaultdict(list)
        for item in one.kernels:
            one_by_signature[signature(item)].append(item["gpu_us"])
        for key, values in one_by_signature.items():
            baseline_by_signature[key] = float(np.mean(values))
        overlap_extra_us = 0.0
        no_overlap_extra_us = 0.0
        overlap_kernel_count = 0
        for item in four.kernels:
            extra_us = item["gpu_us"] - baseline_by_signature[signature(item)]
            if item["p2p_overlap"]:
                overlap_kernel_count += 1
                overlap_extra_us += extra_us
            else:
                no_overlap_extra_us += extra_us
        total_extra_us = overlap_extra_us + no_overlap_extra_us
        summary = {
            "online_windows": {
                label: {
                    "start_s": spec.start_ns / 1e9,
                    "end_s": spec.end_ns / 1e9,
                    "duration_ms": spec.duration_ns / 1e6,
                }
                for label, spec in PROFILES.items()
            },
            "kernel_count": {"1GPU": len(one.kernels), "4GPU": len(four.kernels)},
            "kernel_sum_ms": {
                "1GPU": sum(item["end"] - item["start"] for item in one.kernels) / 1e6,
                "4GPU": sum(item["end"] - item["start"] for item in four.kernels) / 1e6,
            },
            "kernel_extra_decomposition": {
                "method": "Each 4GPU kernel uses the 1GPU mean of the same kernel name/grid/block/shared-memory signature as baseline.",
                "p2p_overlap_kernel_count": overlap_kernel_count,
                "no_p2p_overlap_kernel_count": len(four.kernels) - overlap_kernel_count,
                "p2p_overlap_extra_ms": overlap_extra_us / 1e3,
                "no_p2p_overlap_extra_ms": no_overlap_extra_us / 1e3,
                "total_extra_ms": total_extra_us / 1e3,
                "p2p_overlap_extra_share_pct": 100 * overlap_extra_us / total_extra_us if total_extra_us else float("nan"),
            },
            "kernel_by_device_4gpu": {
                str(device): {
                    "count": sum(item["device"] == device for item in four.kernels),
                    "sum_ms": sum(item["end"] - item["start"] for item in four.kernels if item["device"] == device) / 1e6,
                }
                for device in range(4)
            },
            "mapped_level_kernels": {
                "1GPU": sum(item["level"] is not None for item in one.kernels),
                "4GPU": sum(item["level"] is not None for item in four.kernels),
            },
            "p2p": {
                "count": len(four.p2p),
                "bytes": sum(item["bytes"] for item in four.p2p),
                "service_sum_ms": sum(item["end"] - item["start"] for item in four.p2p) / 1e6,
            },
            "launch": launch_summary,
            "all_op_level_counts_consistent": all(item["consistent"] for item in op_rows),
            "all_kernel_signatures_consistent": all(item["consistent"] for item in signature_rows),
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

        draw_timeline(four, OUT_DIR / "01-online-gpu-timeline.png")
        draw_coverage(coverage, OUT_DIR / "02-online-p2p-coverage.png")
        draw_cpu(one, four, OUT_DIR / "03-online-cpu-breakdown.png")
        draw_consistency(level_rows, OUT_DIR / "04-online-level-consistency.png")
        draw_cpu_timeline(one, four, OUT_DIR / "05-online-cpu-timeline.png")
    finally:
        one.close()
        four.close()


if __name__ == "__main__":
    main()
