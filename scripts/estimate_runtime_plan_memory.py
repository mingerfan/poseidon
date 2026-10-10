#!/usr/bin/env python3
"""Estimate Poseidon value storage, without executing a RuntimePlan.

The serial estimate assumes each operation finishes before the next one starts.
It is not a prediction or lower bound for the per-device asynchronous runtime.
The allocation sum bounds only the modeled Q-only value storage, not total VRAM.
Uses Python's standard library; does not change the input plan.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


PHASES = ("initialization", "execution", "finalization")
SCENARIOS = ("as_written", "last_use", "last_use_and_reuse")
REUSE_OPS = {"host": {"negate", "rotate"}, "device": {"add_cp", "sub_cp", "rotate"}}
EXCLUDED = [
    "evaluation keys and parameter tables",
    "operator scratch, including Rotate/key-switch/Boot workspace and intermediates",
    "communication staging buffers",
    "Encode slot data and bundle caches",
    "allocator alignment, fragmentation, reserved-but-unused memory and object metadata",
    "objects retained by the caller from previous runs",
]


def place_name(desc: dict) -> str:
    p = desc["place"]
    if p["kind"] == "host":
        return f"rank{p['rank']}/host"
    if p["kind"] == "device":
        return f"rank{p['rank']}/gpu{p['index']}"
    raise ValueError(f"unsupported place: {p['kind']}")


def value_bytes(desc: dict, degree: int) -> int:
    level, components = desc["level"], desc["components"]
    if type(level) is not int or level < 0 or type(components) is not int or components <= 0:
        raise ValueError(f"invalid level/components for value {desc['id']}")
    if desc["kind"] not in ("plaintext", "ciphertext"):
        raise ValueError(f"unsupported value kind: {desc['kind']}")
    place_name(desc)
    word_bytes = 8 if desc["place"]["kind"] == "host" else 4
    return degree * (level + 1) * components * word_bytes


def estimate(plan: dict, spec: dict, scenario: str = "as_written") -> dict:
    """Read a verified plan; only check errors needed for meaningful accounting.

    Comparison scenarios ignore explicit Release positions and instead assume
    immediate release after last use. They are hypothetical, not compiler output.
    The proposed V2 Release/reuse fields are understood by this tool even before
    the C++ runtime implements them.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if plan["format_version"] not in (1, 2):
        raise ValueError("only RuntimePlan V1 and proposed V2 are supported")
    degree = spec["context"]["poly_degree"]
    if type(degree) is not int or degree <= 0:
        raise ValueError("poly_degree must be a positive integer")
    reference = plan["target"]["operator_spec"]
    if (reference["id"], reference["version"]) != (spec["spec_id"], spec["version"]):
        raise ValueError("plan and OperatorSpec id/version do not match")
    values = {v["id"]: v for v in plan["values"]}
    if len(values) != len(plan["values"]):
        raise ValueError("duplicate value descriptor")
    for v in values.values():
        if v["context"] != spec["context"]["context_id"]:
            raise ValueError(f"context mismatch for value {v['id']}")
    sizes = {v: value_bytes(desc, degree) for v, desc in values.items()}
    places = {v: place_name(desc) for v, desc in values.items()}
    seq = [(phase, op) for phase in PHASES for op in plan[phase]]
    uses = Counter(v for _, op in seq for v in op.get("inputs", []))
    last_use = {v: n for n, (_, op) in enumerate(seq) for v in op.get("inputs", [])}
    compute_outputs = {op["output"] for _, op in seq if op["kind"] == "compute"}
    external = set(plan["external_inputs"])
    roots = set(plan["final_outputs"])

    # available tracks logical names; resident also includes caller-held inputs.
    available: set = set()
    resident: set = set()
    defined: set = set()
    current: Counter = Counter()
    allocation_sum: Counter = Counter()
    peaks: dict = {}
    phase_peaks: dict = {phase: Counter() for phase in PHASES}
    reuse_count = 0

    def allocate(v):
        if v in defined:
            raise ValueError(f"value {v} is defined twice")
        defined.add(v)
        available.add(v)
        resident.add(v)
        current[places[v]] += sizes[v]
        allocation_sum[places[v]] += sizes[v]

    def drop(v):
        available.remove(v)
        if v not in external:
            resident.remove(v)
            current[places[v]] -= sizes[v]

    def snapshot(phase, ordinal, kind):
        for place, count in current.items():
            phase_peaks[phase][place] = max(phase_peaks[phase][place], count)
            if place in peaks and count <= peaks[place]["serial_peak_bytes"]:
                continue
            held = sorted((v for v in resident if places[v] == place),
                          key=lambda v: (-sizes[v], str(v)))
            peaks[place] = {
                "serial_peak_bytes": count,
                "peak_at": {"phase": phase, "ordinal": ordinal, "kind": kind},
                "plaintext_bytes_at_peak": sum(sizes[v] for v in held
                                                if values[v]["kind"] == "plaintext"),
                "ciphertext_bytes_at_peak": sum(sizes[v] for v in held
                                                 if values[v]["kind"] == "ciphertext"),
                "largest_values_at_peak": [{"value": v, "bytes": sizes[v]} for v in held[:5]],
            }

    def can_reuse(op):
        if op["kind"] != "compute" or not op["inputs"]:
            return False
        x, y = op["inputs"][0], op["output"]
        a, b = values[x], values[y]
        same_storage_shape = all(a[k] == b[k] for k in
                                 ("kind", "place", "context", "level", "scale_log2", "ntt", "components"))
        return (op["op"] in REUSE_OPS[a["place"]["kind"]]
                and x in compute_outputs and x not in external | roots
                and uses[x] == 1 and same_storage_shape)

    for v in external:
        allocate(v)
    snapshot("initialization", None, "external_inputs")
    n = 0
    for phase in PHASES:
        snapshot(phase, None, "phase_start")
        for op in plan[phase]:
            kind = op["kind"]
            if plan["format_version"] == 1 and (kind == "release" or "reuse_input" in op):
                raise ValueError("Release and reuse_input require V2")
            if kind == "release":
                v = op["value"]
                if v in roots:
                    raise ValueError(f"cannot release final output {v}")
                if "wait" in op:
                    raise ValueError("this estimator models the current proposal without a wait field")
                if scenario == "as_written":
                    if v not in available:
                        raise ValueError(f"cannot release unavailable value {v}")
                    drop(v)
            else:
                if kind not in ("encode", "compute", "transfer", "replicate"):
                    raise ValueError(f"unsupported instruction kind: {kind}")
                for v in op.get("inputs", []):
                    if v not in available:
                        raise ValueError(f"instruction {op['ordinal']} uses unavailable value {v}")
                outputs = [op["output"]] if kind in ("encode", "compute") else op["outputs"]
                declared_reuse = "reuse_input" in op
                if declared_reuse and (type(op["reuse_input"]) is not int
                                       or op["reuse_input"] != 0 or not can_reuse(op)):
                    raise ValueError(f"unsupported reuse at instruction {op['ordinal']}")
                reuse = declared_reuse or (scenario == "last_use_and_reuse" and can_reuse(op))
                if reuse:
                    x, y = op["inputs"][0], op["output"]
                    if y in defined:
                        raise ValueError(f"value {y} is defined twice")
                    # Same allocation, only its logical name changes.
                    available.remove(x)
                    resident.remove(x)
                    available.add(y)
                    resident.add(y)
                    defined.add(y)
                    reuse_count += 1
                else:
                    for v in outputs:
                        allocate(v)
                # Input and newly allocated output coexist during computation/copy.
                snapshot(phase, op["ordinal"], kind)
                if scenario != "as_written":
                    for v in set(op.get("inputs", []) + outputs):
                        if v in available and v not in roots and last_use.get(v, -1) <= n:
                            drop(v)
            n += 1
    if not roots <= available:
        raise ValueError("a final output is undefined or has been consumed")
    for place, peak in peaks.items():
        peak["all_value_allocations_bytes"] = allocation_sum[place]
        peak["phase_serial_peaks_bytes"] = {phase: phase_peaks[phase][place] for phase in PHASES}
    return {"scenario": scenario, "hypothetical": scenario != "as_written",
            "reuse_count": reuse_count, "places": dict(sorted(peaks.items()))}


def build_report(plan: dict, spec: dict, compare: bool = False) -> dict:
    # Account for the original plan first so comparison mode cannot hide bad Release positions.
    results = [estimate(plan, spec)]
    if compare:
        results += [estimate(plan, spec, mode) for mode in SCENARIOS[1:]]
    return {
        "model": "poseidon_full_q_value_storage",
        "word_bytes": {"host": 8, "device": 4},
        "assumptions": [
            "serial execution in initialization/execution/finalization and instruction order",
            "each operation completes before the next begins; input and output coexist",
            "external input storage is retained by the caller for this run",
            "each normal value owns a full Q-only allocation; reuse preserves its capacity",
            "comparison modes preserve placement and Transfer order; no prefetch",
        ],
        "excluded": EXCLUDED,
        "total_process_memory_estimate": None,
        "allocation_sum_meaning": "sum of distinct modeled value allocations if none are reclaimed; not total-memory bound",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--operator-spec", type=Path, required=True)
    parser.add_argument("--compare-last-use", action="store_true",
                        help="also estimate hypothetical last-use release, then eligible reuse")
    parser.add_argument("--json-output", type=Path, help="save bytes, assumptions and peak locations")
    args = parser.parse_args()
    try:
        report = build_report(json.loads(args.plan.read_text()),
                              json.loads(args.operator_spec.read_text()), args.compare_last_use)
        report["plan"] = str(args.plan)
        report["operator_spec"] = str(args.operator_spec)
        if args.json_output:
            args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print("仅估算明文/密文数据；假设逐条执行并完成，不代表实际总内存/显存。")
    print("Host 外部输入按调用方保留到本次运行结束计入。")
    labels = {"as_written": "原计划", "last_use": "假设最后使用后释放",
              "last_use_and_reuse": "假设再加符合条件的原位"}
    for result in report["results"]:
        print(f"\n{labels[result['scenario']]}（原位 {result['reuse_count']} 次）")
        print(f"{'位置':<20} {'串行峰值 MiB':>14} {'全部不同数据块 MiB':>18}  峰值位置")
        for place, data in result["places"].items():
            at = data["peak_at"]
            print(f"{place:<22} {data['serial_peak_bytes'] / 2**20:>14.2f} "
                  f"{data['all_value_allocations_bytes'] / 2**20:>18.2f}  "
                  f"{at['phase']}:{at['ordinal'] if at['ordinal'] is not None else at['kind']}")
    print("\n未计入密钥、参数、算子工作区、传输暂存区、编码数据缓存和分配器额外占用。")
    print("全部不同数据块之和仅供保守参考；多卡各自峰值不能相加当作同一时刻总峰值。")


if __name__ == "__main__":
    main()
