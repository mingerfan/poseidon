#!/usr/bin/env python3
"""Plot per-process memory samples; allocator peaks remain separate evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--devices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output prefix")
    parser.add_argument("--label", default="Qwen3-0.6B process memory")
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.memory.read_text().splitlines() if line]
    if not samples or len({s["pid"] for s in samples}) != 1:
        raise ValueError("one nonempty process history is required")
    elapsed = [s["elapsed_seconds"] for s in samples]
    if any(b < a for a, b in zip(elapsed, elapsed[1:])):
        raise ValueError("sample elapsed time must be monotonic")
    with args.devices.open() as source:
        devices = [{k.strip(): v.strip() for k, v in row.items()}
                   for row in csv.DictReader(source)]
    indices = {row["uuid"]: int(row["index"]) for row in devices}
    uuids = sorted({uuid for sample in samples for uuid in sample["gpu_process_bytes"]},
                   key=lambda uuid: indices[uuid])
    png, summary_path = args.output.with_suffix(".png"), args.output.with_suffix(".json")
    if png.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite existing evidence")
    png.parent.mkdir(parents=True, exist_ok=True)
    gib = 2**30
    minutes = [(value - elapsed[0]) / 60 for value in elapsed]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    gpu_peaks = []
    for uuid in uuids:
        values = [s["gpu_process_bytes"].get(uuid, 0) for s in samples]
        axes[0].plot(minutes, [v/gib for v in values], label=f"Physical GPU {indices[uuid]}",
                     linestyle=("-", "--", "-.", ":")[indices[uuid] % 4])
        gpu_peaks.append(dict(physical_gpu=indices[uuid], uuid=uuid, sampled_peak_bytes=max(values)))
    rss = [s["host"].get("VmRSS_bytes", float("nan")) for s in samples]
    hwm = [s["host"].get("VmHWM_bytes", float("nan")) for s in samples]
    valid_rss = [v for v in rss if math.isfinite(v)]
    valid_hwm = [v for v in hwm if math.isfinite(v)]
    if not valid_rss or not valid_hwm:
        raise ValueError("history lacks Host RSS/HWM measurements")
    axes[1].plot(minutes, [v/gib for v in rss], label="RSS")
    axes[1].plot(minutes, [v/gib for v in hwm], label="Observed VmHWM", linestyle="--")
    axes[0].set_ylabel("Driver memory (GiB)")
    axes[1].set_ylabel("Host memory (GiB)")
    axes[1].set_xlabel("Minutes since first sample")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", ncols=2)
    fig.suptitle(args.label)
    fig.supxlabel("Interval samples of this PID only; allocator live/reserved peaks are reported separately.",
                  fontsize=8)
    fig.savefig(png, dpi=180)
    plt.close(fig)
    summary = dict(pid=samples[0]["pid"], samples=len(samples),
        first_timestamp=samples[0]["timestamp"], last_timestamp=samples[-1]["timestamp"],
        sample_span_seconds=elapsed[-1]-elapsed[0],
        peak_sampled_rss_bytes=max(valid_rss), peak_observed_hwm_bytes=max(valid_hwm), gpu=gpu_peaks,
        missing_rss_samples=len(rss)-len(valid_rss), missing_hwm_samples=len(hwm)-len(valid_hwm),
        inputs={str(path.resolve()): "sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (args.memory, args.devices)}, command=[sys.executable, *sys.argv],
        scope="Interval samples, not continuous allocator peaks; no other PID's GPU memory is included.",
        host_scope="RSS includes context, keys, graph and caller data outside the managed Host budget.")
    summary_path.write_text(json.dumps(summary, indent=2)+"\n")
    print(summary_path)


if __name__ == "__main__":
    main()
