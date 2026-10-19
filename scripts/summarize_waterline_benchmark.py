#!/usr/bin/env python3
"""Summarize measured runs, allocator peaks and admission-credit peaks separately."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def summarize(folder, reference=None):
    source = json.loads((folder/"summary.json").read_text())
    groups = {}
    for sample in source["samples"]:
        if not sample["passed"]:
            raise RuntimeError(f"failed measured sample in {folder}: {sample['case']}")
        case = sample["case"]
        report = sample["report"]
        values = {k:sample[k] for k in ("forward_ms", "online_ms", "init_ms", "setup_ms", "process_wall_ms")}
        timing = report["runtime_timing"]
        if "preparation_measurements" in report:
            prepare = report["preparation_measurements"]
            devices = report["device_measurements"]
            for key in ("cpu_encode_ms", "cpu_prepare_calls", "cpu_plain_h2d_bytes", "transport_h2d_bytes", "transport_h2d_ms"):
                values[key] = prepare[key]
            for key in ("gpu_encode_calls", "gpu_encode_ms", "gpu_raw_h2d_bytes", "gpu_raw_h2d_ms", "consumer_wait_ms"):
                values[key] = sum(d[key] for d in devices)
            values["allocator_peak_gib"] = max(d["allocated_peak_bytes"] for d in devices)/2**30
            values["bundle_read_ms"] = timing["bundle_read_ms"]
            values["bundle_read_bytes"] = timing["bundle_read_bytes"]
            values["compute_calls"] = timing["compute_calls"]
            values["encode_submission_ms"] = timing["encode_submission_ms"]
            values["max_abs_error"] = report["comparison"]["max_abs_error"]
            values["checked_complex_slots"] = report["comparison"]["slots_checked"]
        else:
            values["max_abs_error"] = report["mock_comparison"]["max_abs"]
        log = (folder/f"r{sample['repeat']}-{case}"/"run.log").read_text()
        credit = [int(n) for n in re.findall(r" plain_peak=(\d+)", log)]
        if credit:
            values["max_plain_admission_credit_mib"] = max(credit)/2**20
            remaining = [int(n) for n in re.findall(r"remaining_plain=(\d+)", log)]
            if any(remaining):
                raise RuntimeError("plaintext credits did not return to zero")
        groups.setdefault(case, []).append(values)
    results = {}
    for name, rows in groups.items():
        results[name] = dict(samples=len(rows), metrics={key:dict(
            median=statistics.median(r[key] for r in rows), min=min(r[key] for r in rows), max=max(r[key] for r in rows))
            for key in rows[0]})
    for result in results.values():
        for metric in ("online_ms", "forward_ms"):
            eager = results.get("eager") if reference is None else reference["cases"]["eager"]
            result[metric.removesuffix("_ms")+"_ratio_to_eager"] = result["metrics"][metric]["median"]/eager["metrics"][metric]["median"]
    return dict(suite=source["suite"], cuda=source["cuda"], pool_mib=source["pool_mib"],
        input_summary_sha256="sha256:"+hashlib.sha256((folder/"summary.json").read_bytes()).hexdigest(), cases=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    directory = args.directory.resolve()
    labels = {"local-mlp-q21":"RTX 4060 / MLP", "server-mlp1-gpu2":"V100 / MLP / 1 GPU",
        "server-mlp2-gpu23":"V100 / MLP / 2 GPUs", "local-pressure":"RTX 4060 / 3.125 GiB weights",
        "server-pressure-gpu2":"V100 / 12.5 GiB weights", "local-qproj":"RTX 4060 / synthetic Qwen q_proj"}
    results = {}
    for name in labels:
        folder = directory/name
        if (folder/"summary.json").exists():
            results[name] = summarize(folder)
    binary_labels = {"local-pressure-binary":("RTX 4060 / 3.125 GiB / binary manifest","local-pressure"),
        "server-pressure-binary-gpu2":("V100 / 12.5 GiB / binary manifest","server-pressure-gpu2"),
        "local-qproj-binary":("RTX 4060 / synthetic q_proj / binary manifest","local-qproj")}
    for name, (label, reference) in binary_labels.items():
        folder = directory/name
        if (folder/"summary.json").exists():
            results[name] = summarize(folder, results[reference])
            results[name]["eager_reference"] = reference
    (directory/"comparison.json").write_text(json.dumps(results, indent=2)+"\n")
    lines = ["Waterline performance measurements — 2026-10-18", "",
        "forward = runtime setup + plaintext preparation / initialization + online execution.",
        "online = execution after eager initialization, or execution including demand preparation for Waterline.",
        "Fresh process per sample; OS file cache was not flushed; keys/input encryption are outside forward.",
        "One rank; V100 single GPU uses physical 2, dual GPU uses physical 2 and 3.",
        "Physical V100 0 failed CUDA initialization with uncorrectable ECC; physical 1 had another user's job.",
        "MLP: Q21/drop5 uniformly used for current integer Rescale; model/weights unchanged.",
        "Pressure: N16384/Q50/P8, deterministic full-slot weighted sum, no Rotate/Boot/Rescale.",
        "q_proj: existing synthetic cyclic projection path; this is not a complete pretrained Qwen forward.",
        "GPU times and consumer waits are event sums, not disjoint parts of wall time.",
        "Allocator peaks include persistent state; admission credits include pending allocations and workspace bounds.",
        "Binary controls reuse the original eager online baseline: blob bytes and all ciphertext work are identical.",
        "Their complete-forward ratio uses the original JSON eager preparation baseline, not an independently rerun binary eager initialization.", ""]
    for name, result in results.items():
        title = labels[name] if name in labels else binary_labels[name][0]
        lines += [title, "case | n | forward ms | online ms | online/eager | allocated peak GiB | plain credit peak MiB"]
        for case, row in result["cases"].items():
            m = row["metrics"]
            def v(key, places=3):
                return f"{m[key]['median']:.{places}f}" if key in m else "—"
            lines.append(f"{case} | {row['samples']} | {v('forward_ms')} | {v('online_ms')} | {row['online_ratio_to_eager']:.3f} | {v('allocator_peak_gib')} | {v('max_plain_admission_credit_mib')}")
        lines.append("")
    (directory/"measurements.txt").write_text("\n".join(lines)+"\n")
    if not results:
        raise RuntimeError("no measurements found")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)
    for ax, name in zip(axes.flat, labels):
        if name not in results:
            ax.set_visible(False)
            continue
        cases = results[name]["cases"]
        gpu = [c for c in cases if "-gpu-" in c]
        names = ["eager"]+gpu
        x = np.arange(len(names))
        for offset, metric, color, label in ((-0.18,"online_ms","#286eae","Online"),(0.18,"forward_ms","#e69a32","Complete forward")):
            values = [cases[c]["metrics"][metric]["median"] for c in names]
            errors = np.array([[cases[c]["metrics"][metric]["median"]-cases[c]["metrics"][metric]["min"] for c in names],
                [cases[c]["metrics"][metric]["max"]-cases[c]["metrics"][metric]["median"] for c in names]])
            ax.bar(x+offset, values, width=0.34, color=color, label=label, yerr=errors, capsize=3)
        ax.set_xticks(x, ["Preloaded" if c=="eager" else c.split("-")[1] for c in names], rotation=25)
        ax.set_ylabel("Wall time (ms)")
        if "pressure" in name or "qproj" in name:
            ax.set_yscale("log")
            ax.set_ylabel("Wall time (ms, log scale)")
        ax.set_title(labels[name]+(" / JSON" if "pressure" in name or "qproj" in name else ""), fontweight="bold")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        if name==next(iter(results)):
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("GPU plaintext preparation vs preloaded weights — median, min/max; 3 processes per configuration", fontsize=14, fontweight="bold")
    fig.savefig(directory/"comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for ax, (name, (title, reference)) in zip(axes, binary_labels.items()):
        if name not in results:
            ax.set_visible(False)
            continue
        cases = results[name]["cases"]
        eager = results[reference]["cases"]["eager"]
        gpu = [c for c in cases if "-gpu-" in c]
        rows = [eager]+[cases[c] for c in gpu]
        values = [r["metrics"]["online_ms"]["median"]/1000 for r in rows]
        errors = np.array([[(r["metrics"]["online_ms"]["median"]-r["metrics"]["online_ms"]["min"])/1000 for r in rows],
            [(r["metrics"]["online_ms"]["max"]-r["metrics"]["online_ms"]["median"])/1000 for r in rows]])
        ax.bar(range(len(rows)), values, color=["#626e7b"]+["#286eae"]*len(gpu), yerr=errors, capsize=3)
        ax.set_xticks(range(len(rows)),["Preloaded"]+[c.split("-")[1] for c in gpu],rotation=25)
        for i, value in enumerate(values):
            label_y = rows[i]["metrics"]["online_ms"]["max"]/1000+max(values)*.025
            ax.text(i, label_y, f"{value:.2f}s", ha="center",fontsize=9)
        ax.set_ylim(0, max(r["metrics"]["online_ms"]["max"]/1000 for r in rows)*1.18)
        ax.set_ylabel("Online wall time (s)")
        ax.set_title(title, fontweight="bold",fontsize=11)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Binary manifest loader: demand preparation vs preloaded GPU weights (median and min/max, n=3)",fontsize=13,fontweight="bold")
    fig.savefig(directory/"binary-online.png",dpi=180)
    plt.close(fig)
    poll_path, profile_path = directory/"local-poll-binary/summary.json", directory/"nsys-poll-binary.json"
    if poll_path.exists() and profile_path.exists():
        poll, profile = json.loads(poll_path.read_text()), json.loads(profile_path.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
        names = ("100us", "10us", "yield")
        values = [poll["statistics"][n]["runtime_ms"]["median"]/1000 for n in names]
        errors = np.array([[(poll["statistics"][n]["runtime_ms"]["median"]-poll["statistics"][n]["runtime_ms"]["min"])/1000 for n in names],
            [(poll["statistics"][n]["runtime_ms"]["max"]-poll["statistics"][n]["runtime_ms"]["median"])/1000 for n in names]])
        axes[0].bar(names, values, color=["#626e7b","#5d94c1","#286eae"],yerr=errors,capsize=4)
        for i,value in enumerate(values): axes[0].text(i,value*1.03,f"{value:.3f}s",ha="center")
        axes[0].set_ylim(0,max(values)*1.2)
        axes[0].set_ylabel("Complete forward wall time (s)")
        axes[0].set_title("Only the dispatcher's no-progress wait changes")
        names = ("100us", "yield")
        busy = [profile["profiles"][n]["gpu_kernel_union_ms"]/1000 for n in names]
        gap = [(profile["profiles"][n]["gpu_kernel_span_ms"]-profile["profiles"][n]["gpu_kernel_union_ms"])/1000 for n in names]
        axes[1].bar(names,busy,color="#286eae",label="CUDA kernel active")
        axes[1].bar(names,gap,bottom=busy,color="#d6dde4",label="No CUDA kernel active")
        axes[1].set_ylabel("First-to-last kernel window (s)")
        axes[1].set_title("92,375 identical kernel launches in both profiles")
        axes[1].legend(frameon=False,fontsize=9)
        for ax in axes:
            ax.grid(axis="y",alpha=.18)
            ax.set_axisbelow(True)
        fig.suptitle("Dispatcher wait diagnosis — same encrypted q_proj, binary weights, GPU preparation, window=8",fontweight="bold",fontsize=12)
        fig.savefig(directory/"dispatcher-wait.png",dpi=180)
        plt.close(fig)
    print(directory/"measurements.txt")


if __name__=="__main__":
    main()
