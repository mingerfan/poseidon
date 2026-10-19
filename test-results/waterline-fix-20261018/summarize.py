#!/usr/bin/env python3
"""Verify the recorded runs and compare the original and fixed binaries."""
import hashlib
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
GROUPS = {
    "local-qproj": "RTX 4060 / q_proj, window 128",
    "local-qproj-w8": "RTX 4060 / q_proj, GPU, window 8",
    "local-qproj-auto-w8": "RTX 4060 / q_proj, Auto, window 8",
    "local-mlp": "RTX 4060 / MLP",
    "server-mlp1": "V100 / MLP, 1 GPU",
    "server-mlp2": "V100 / MLP, 2 GPUs",
    "local-pressure": "RTX 4060 / 3.125 GiB weights",
    "server-pressure": "V100 / 12.5 GiB weights",
}


def read(name):
    return json.loads((OUT / name / "summary.json").read_text())


rows, verified = [], []
for group, label in GROUPS.items():
    pair = {}
    for tag in ("before", "after"):
        summary = read(f"{group}-{tag}")
        samples = list(summary["samples"])
        if "pressure" in group:
            samples += read(f"{group}-eager-{tag}")["samples"]
        pair[tag] = {}
        for sample in samples:
            assert sample["passed"] and sample["exit_code"] == 0, (group, tag, sample["case"])
            assert sample["report"]["passed"]
            case = sample["case"]
            pair[tag].setdefault(case, []).append(sample)
            folder = OUT / f"{group}{'-eager' if 'pressure' in group and case == 'eager' else ''}-{tag}" / f"r{sample['repeat']}-{case}"
            plan = json.loads(Path(json.loads((folder / "command.json").read_text())[1]).read_text()) if group.startswith("local") else None
            report = sample["report"]
            if "device_measurements" in report:
                assert report["comparison"]["slots_checked"] == 8192
                assert report["pinned_transfer_live_after_plan_bytes"] == 0
                assert report["pinned_transfer_live_after_validation_bytes"] == 0
                for device in report["device_measurements"]:
                    if device["managed_pool"]:
                        assert device["allocated_peak_bytes"] <= device["pool_capacity_bytes"]
                if case != "eager":
                    count = 1025 if "qproj" in group else (4096 if group.startswith("server") else 1024)
                    prepared = report["preparation_measurements"]["cpu_prepare_calls"] + sum(d["gpu_encode_calls"] for d in report["device_measurements"])
                    assert prepared == count, (folder, prepared, count)
                    assert report["runtime_timing"]["bundle_read_calls"] == count, folder
            log = (folder / "run.log").read_text()
            peaks = re.findall(r"waterline GPU(\d+) front_peak=(\d+) plain_peak=(\d+) remaining_front=(\d+) remaining_plain=(\d+)", log)
            for device, front, plain, remaining_front, remaining_plain in peaks:
                assert int(remaining_plain) == 0, folder
                c = plan["waterline"] if plan else report.get("waterline")
                if c:
                    d = int(device)
                    assert int(plain) <= int(c["device_fractions"][d] * c["device_capacity_bytes"][d])
                    assert int(front) <= c["device_capacity_bytes"][d] - c["device_reserve_bytes"][d]
            shared = re.search(r"waterline shared host_peak=(\d+) pinned_peak=(\d+) raw_peak=(\d+)", log)
            c = plan.get("waterline") if plan else report.get("waterline")
            if shared and c:
                assert all(int(used) <= c[budget] for used, budget in zip(shared.groups(), ("host_budget_bytes", "pinned_budget_bytes", "raw_budget_bytes")))
            verified.append(dict(group=group, variant=tag, case=case, repeat=sample["repeat"], passed=True, peaks=peaks))
        assert set(pair[tag]) == set(summary["selected_case_names"]) | ({"eager"} if "pressure" in group else set())
    assert set(pair["before"]) == set(pair["after"])
    for case in pair["before"]:
        row = dict(group=group, label=label, case=case)
        reports = [s["report"] for tag in ("before", "after") for s in pair[tag][case]]
        for key in ("plan_source_sha256", "plan_sha256", "operator_spec_sha256", "fixture_sha256", "model_sha256"):
            if key in reports[0]:
                assert len({r[key] for r in reports}) == 1, (group, case, key)
        for tag in ("before", "after"):
            samples = pair[tag][case]
            row[tag] = dict(repeats=len(samples), **{
                metric: dict(median=statistics.median(s[metric] for s in samples),
                             min=min(s[metric] for s in samples), max=max(s[metric] for s in samples))
                for metric in ("online_ms", "forward_ms", "process_wall_ms")})
        row["online_speedup"] = row["before"]["online_ms"]["median"] / row["after"]["online_ms"]["median"]
        row["online_reduction_percent"] = 100 * (1 - 1 / row["online_speedup"])
        rows.append(row)

cpu = {}
for tag in ("before", "after"):
    samples = read("local-cpu-cost-" + tag)["samples"]
    assert len(samples) == 3 and all(s["passed"] for s in samples)
    cpu[tag] = {metric: [s[metric] for s in samples] for metric in ("process_cpu_seconds", "online_ms", "process_wall_ms")}
    cpu[tag]["median_cpu_seconds"] = statistics.median(cpu[tag]["process_cpu_seconds"])
default_window_cpu = {
    tag: [s["process_cpu_seconds"] for s in read("local-qproj-w8-" + tag)["samples"]]
    for tag in ("before", "after")
}
default_auto_cpu = {
    tag: [s["process_cpu_seconds"] for s in read("local-qproj-auto-w8-" + tag)["samples"]]
    for tag in ("before", "after")
}

result = dict(scope="Fresh-process, unchanged plans, weights, ciphertext work and pool limits. Online excludes eager preparation; forward includes it. GPU 2/3 only on server. Pressure eager controls have one repeat; other comparisons have three.",
              comparisons=rows, cpu_cost=cpu, window_8_gpu_cpu_seconds=default_window_cpu,
              default_auto_cpu_seconds=default_auto_cpu,
              verified_samples=len(verified)+6, verification=verified)
(OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
table = ["All values in seconds; online execution. Eager pressure controls n=1; all other rows n=3.",
         "host/workload | case | before | after | speedup | reduction"]
for row in rows:
    table.append(f"{row['label']} | {row['case']} | {row['before']['online_ms']['median']/1000:.6f} | {row['after']['online_ms']['median']/1000:.6f} | {row['online_speedup']:.3f}x | {row['online_reduction_percent']:.1f}%")
(OUT / "measurements.txt").write_text("\n".join(table) + "\n")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
plot_groups = [(g, label) for g, label in GROUPS.items() if g not in ("local-qproj", "local-qproj-w8")]
for ax, (group, label) in zip(axes.flat, plot_groups):
    selected = [r for r in rows if r["group"] == group and (r["case"] == "eager" or "-gpu-" in r["case"] or group == "local-qproj-auto-w8")]
    if group == "local-qproj-auto-w8":
        # Eager has no lookahead setting; reuse its measured non-Waterline plan.
        selected = [r for r in rows if r["group"] == "local-qproj" and r["case"] == "eager"] + selected
    selected.sort(key=lambda r: r["case"] != "eager")
    x = np.arange(len(selected))
    for offset, tag, color in [(-.19, "before", "#8d99ae"), (.19, "after", "#008f82")]:
        values = [r[tag]["online_ms"]["median"]/1000 for r in selected]
        err = [[r[tag]["online_ms"]["median"]/1000-r[tag]["online_ms"]["min"]/1000 for r in selected],
               [r[tag]["online_ms"]["max"]/1000-r[tag]["online_ms"]["median"]/1000 for r in selected]]
        bars = ax.bar(x + offset, values, .36, label=tag.capitalize(), color=color, yerr=err, capsize=2)
        for bar, value, upper in zip(bars, values, err[1]):
            ax.annotate(f"{value:.2f}", (bar.get_x()+bar.get_width()/2, bar.get_height()+upper),
                        xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.set_title(label, loc="left", weight="bold")
    ax.set_xticks(x, ["Resident" if r["case"] == "eager" else r["case"].split("-")[1] for r in selected], fontsize=8)
    ax.set_ylabel("Online execution (seconds)")
    ax.set_xlabel("Plaintext waterline / resident baseline")
    ax.set_ylim(0, ax.get_ylim()[1]*1.15)
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
axes.flat[0].legend(frameon=False)
fig.suptitle("Bounded GPU submission removes dispatcher stalls", fontsize=16, weight="bold")
fig.savefig(OUT / "comparison.png", dpi=170)
plt.close(fig)

paths = ["third_party/ckks-runtime/runtime/runtime.hpp", "src/poseidon/runtime_api/poseidon_gpu_api.cpp", "src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp"]
provenance = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
(OUT / "source-sha256.json").write_text(json.dumps(provenance, indent=2) + "\n")
print("\n".join(table))
print("CPU cost:", cpu)

checks = {
    "core-tests-final.log": "100% tests passed",
    "core-waterline-repeat.log": "100% tests passed",
    "local-gpu-tests-final.log": "100% tests passed",
    "server-gpu-api-tests-final.log": "100% tests passed",
    "server-gpu-oom-tests-final.log": "100% tests passed",
    "local-pipeline-negative-control.log": "Waterline waited for the first GPU completion before submitting its consumer",
    "local-gpu-detailed.log": "[PASS] Waterline pipelines GPU submissions and retains window and byte credits through completion",
}
for name, expected in checks.items():
    assert expected in (OUT/name).read_text(), name
nsys = json.loads((OUT/"nsys-comparison.json").read_text())
assert nsys["same_kernel_shapes_and_counts"]
assert all(p["passed"] for p in nsys["profiles"].values())
assert json.loads((OUT/"matching-source-sha256.json").read_text())["match"]
(OUT/"verification.json").write_text(json.dumps(dict(unprofiled_measurements=result["verified_samples"],
    all_measurements_passed=True, same_plan_spec_fixture_hashes=True, final_plain_credits_zero=True,
    observed_waterline_and_pool_peaks_within_limits=True, pinned_live_after_run_zero=True,
    no_repeated_plaintext_preparation=True, profiled_measurements=2, same_kernel_shapes_and_counts=True,
    source_hashes_match_across_hosts=True, checks=checks), indent=2)+"\n")

def entry(group, case):
    return next(r for r in rows if r["group"] == group and r["case"] == case)

selected = [
    ("4060 q_proj，脚本默认 Auto/窗口 8，水位 64 MiB", "local-qproj-auto-w8", "wl-64MiB-auto-w8"),
    ("4060 q_proj，GPU/窗口 8，水位 64 MiB", "local-qproj-w8", "wl-64MiB-gpu-w8"),
    ("4060 q_proj，窗口 128，水位 64 MiB", "local-qproj", "wl-64MiB-gpu-w128"),
    ("4060 MLP，水位 64 MiB", "local-mlp", "wl-64MiB-gpu-w128"),
    ("V100 单卡 MLP，水位 64 MiB", "server-mlp1", "wl-64MiB-gpu-w128"),
    ("V100 双卡 MLP，水位 64 MiB", "server-mlp2", "wl-64MiB-gpu-w128"),
    ("4060 大负载，水位 256 MiB", "local-pressure", "wl-256MiB-gpu-w512"),
    ("V100 大负载，水位 1024 MiB", "server-pressure", "wl-1024MiB-gpu-w512"),
]
lines = [
    "Waterline 调度修复及验证，2026-10-18", "",
    "已修复 GPU 提交与执行完成被绑在一起的问题。后续操作可在前一项真正完成前排队；输入、输出、临时空间和准备缓冲区仍按真实完成事件回收。", "",
    "代码变化：",
    "1. 设备工作线程在 API 提交返回后继续接收任务；独立完成队列保留额度，窗口和字节上限仍约束全部已准入工作。",
    "2. GPU 准备可与前台计算重叠，每卡仍只允许一个完整 GPU Prepare，保护共享 pinned/raw 缓冲区。未完成生产者的输出，即使已 Release 且无消费者，也不能提前归还额度。",
    "3. 工作线程提交完成后用条件变量立即唤醒调度器；只在无法继续推进时定时检查 GPU 完成，不使用忙循环 yield。",
    "4. 等待时间统计支持多个未完成事件；同一显式 CUDA 流中，一项尚未完成时不重复查询其后各项，其他流和设备独立查询。PTDS、传输和 NCCL 请求不套用此简化。", "",
    "修复前后速度：除标明 Auto 外均为 GPU Prepare，单位秒，3 次独立进程的在线时间中位数。",
    "负载 | 修复前 | 修复后 | 加速倍数",
]
for label, group, case in selected:
    r = entry(group, case)
    lines.append(f"{label} | {r['before']['online_ms']['median']/1000:.4f} | {r['after']['online_ms']['median']/1000:.4f} | {r['online_speedup']:.2f}x")
lines += ["", "与全加载后执行的比较：",
          "全加载 online 不含提前编码和上传，Waterline online 包含运行中的准备。两者的 online 比值不能当作纯调度开销。完整首轮 forward 还包含初始化；二者都在 comparison.json 和 measurements.txt 中保存。"]
for group in ("local-pressure", "server-pressure"):
    eager = entry(group, "eager")
    dynamic = [r for r in rows if r["group"] == group and "-gpu-" in r["case"]]
    lo, hi = min(r["after"]["online_ms"]["median"] for r in dynamic), max(r["after"]["online_ms"]["median"] for r in dynamic)
    forward = [r["after"]["forward_ms"]["median"] for r in dynamic]
    lines.append(f"{GROUPS[group]}：修复后全加载 online {eager['after']['online_ms']['median']/1000:.4f}s，水位 online {lo/1000:.4f}–{hi/1000:.4f}s；全加载 forward {eager['after']['forward_ms']['median']/1000:.3f}s，水位 forward {min(forward)/1000:.3f}–{max(forward)/1000:.3f}s。全加载对照因 CPU 编码耗时较长，本轮各测 1 次。")
lines += [
    "全加载使用 CPU 提前编码，Waterline 此表使用 GPU 编码，不能把完整首轮差距全部归因于调度。轻计算的大负载修复后仍明显慢于已驻留权重的计算阶段，剩余成本包含读取、编码和上传；本次没有消除这些实际工作。",
    "Qwen 编译脚本实际设置为 Auto/窗口 8，本轮单独验证了该配置。该 q_proj 对照前后每次均有 361 次 CPU Prepare、664 次 GPU Prepare；修复后 CPU 编码耗时仍约 1.47–1.75s，所以 Auto/8 的 2.69s 仍明显慢于 GPU/8 的 1.47s。",
    f"同一 q_proj 的 Auto/128 中位数为 {entry('local-qproj','wl-64MiB-auto-w128')['after']['online_ms']['median']/1000:.3f}s，较大窗口可隐藏更多准备开销。GPU/8 与 Auto/128 可作为后续完整 Qwen 的配置对照，尚未证明哪一个是完整模型的最优设置。图中 q_proj 的驻留基线取自同一不带 Waterline 的 eager 计划，与窗口无关。",
    "全部水位和 CPU/Auto 对照见 measurements.txt。CPU 编码路径收益较小且有波动，例如 V100 单卡 MLP 的 CPU Prepare 中位数没有改善；不声称所有路径都加速。", "",
    "验证证据：",
    f"{result['verified_samples']} 次未加 profiling 的测量全部通过。MLP 与独立参考比较，q_proj/大负载逐一检查 8192 个复数槽。每个配置前后 plan/spec/fixture 哈希一致；没有删减密文算子、改变放置或重新生成权重。",
    "所有动态明文额度在结束时归零；检查账本峰值和池上限，运行结束后 pinned staging 为零。Prepare/读取次数与权重数一致，没有重复加载来伪造对比。MLP 水位为 64/8/1 MiB；4060 大负载为 4096/256/32/8 MiB，V100 为 16384/1024/64/8 MiB。",
    "核心 8 项测试通过，Waterline 并发回归额外重复 3 次。真实 GPU 测试覆盖提前提交、1 组窗口、只能放两份密文的字节边界、Release/原位复用、多卡通信、OOM 排空和显式再次运行。旧调度器的负对照会在“必须在第一项完成前提交第二项”的断言处失败。",
    "服务器功能测试用物理 GPU 2、3；OOM 测试单独用 GPU 2。最初把 OOM 测试与双卡测试放在同一可见设备环境，触发了它要求恰好一张卡的检查；修正测试环境后通过。中间试跑、错误配置和优化前结果保留在日志或 pipeline-first 中，不计入最终测量。", "",
    "独立 Nsight 采样，窗口 128、64 MiB、GPU Prepare，不计入速度中位数：",
]
for tag, label in (("before", "修复前"), ("after", "修复后")):
    p = nsys["profiles"][tag]
    queries = sum(q["calls"] for q in p["cuda_event_query_api"])
    lines.append(f"{label}：{p['kernels']} 个 kernel，kernel 活动并集 {p['gpu_kernel_union_ms']/1000:.4f}s，首末 kernel 跨度 {p['gpu_kernel_span_ms']/1000:.4f}s，无 kernel 活动间隙 {p['no_kernel_between_first_last_ms']/1000:.4f}s，cudaEventQuery {queries} 次。")
lines += [
    "两版 kernel 名称、grid/block 和次数完全相同。无 kernel 间隙可能包含 memcpy 引擎活动，不能称为整个 GPU 完全空闲。", "",
    "CPU 成本：",
    f"GPU 路径窗口 128 的配对进程 CPU 总秒数中位数 {cpu['before']['median_cpu_seconds']:.3f}→{cpu['after']['median_cpu_seconds']:.3f}；GPU/窗口 8 为 {statistics.median(default_window_cpu['before']):.3f}→{statistics.median(default_window_cpu['after']):.3f}；Auto/窗口 8 为 {statistics.median(default_auto_cpu['before']):.3f}→{statistics.median(default_auto_cpu['after']):.3f}。该指标含密钥生成、上下文和校验，不是 forward-only CPU 时间；窗口 8 的样本也有明显波动。较大窗口会增加并发管理成本。",
    "连续提交的第一版额外产生较多事件查询；最终版将单独采样中的查询数从中间版约 67.9 万降至本报告数值。", "",
    "环境和边界：",
    "本地 RTX 4060 Laptop 8 GiB；服务器 V100-SXM2 32 GiB。MLP 使用相同 Q21 配置；大负载 N=16384/Q50/P8，本地 1024 份权重约 3.125 GiB，服务器 4096 份约 12.5 GiB，均可全放入目标 GPU。",
    "MLP 和本地大负载固定物理池 6 GiB，服务器大负载固定 24 GiB，前后一致。nvidia-smi 显示的池预留空间不会因逻辑额度回收而立即变小。系统未清文件缓存，也未锁定频率；每个样本使用新进程，图中误差线表示三次最小/最大值。",
    "q_proj 是合成算子负载，logical scale 120；完整 Qwen 使用其他图规模和 scale 配置。未重跑四卡完整 Qwen：服务器 0 号卡有 ECC 初始化故障，1 号卡有其他任务，本轮只使用 2、3 号卡。不能据此宣称完整 Qwen 的速度问题全部解决。",
    "两个仓库均未提交或推送。本轮修改在主仓库及 third_party/ckks-runtime 子模块工作区中；最终源文件跨主机 SHA-256 相同。", "",
    "复现入口：remaining-local-jobs.json、final-local-jobs.json、final-extra-local-jobs.json、default-auto-jobs.json、remaining-server-jobs.json、final-extra-server-jobs.json。复现前将输出目录换成新目录，脚本有意不覆盖旧结果。每个样本目录另保存精确命令、日志和数值报告；comparison.json 保存全部中位数、范围和加速比。",
]
(OUT/"diagnosis.zh-CN.txt").write_text("\n".join(lines)+"\n")
