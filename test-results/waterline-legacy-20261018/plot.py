#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
data = json.loads((out / "comparison.json").read_text())
labels = {"legacy-eager": "Pre-Waterline eager", "unfixed-eager": "Unfixed eager",
          "current-eager": "Fixed eager", "current-waterline": "Fixed GPU Waterline",
          "unfixed-waterline": "Unfixed GPU Waterline",
          "unfixed-auto-w8": "Unfixed Auto / window 8",
          "current-auto-w8": "Fixed Auto / window 8"}
colors = {"legacy-eager": "#64748b", "unfixed-eager": "#c46b61", "current-eager": "#008f82",
          "current-waterline": "#4b83c3", "current-auto-w8": "#8a6cb0",
          "unfixed-waterline": "#df9487", "unfixed-auto-w8": "#b5a2cc"}
order = ["legacy-eager", "unfixed-eager", "unfixed-waterline", "current-eager", "current-waterline",
         "unfixed-auto-w8", "current-auto-w8"]
titles = {"local-mlp": "RTX 4060 / MLP\n64 MiB, window 128",
          "local-qproj": "RTX 4060 / synthetic q_proj\nGPU: 64 MiB, window 128; Auto: window 8",
          "local-pressure": "RTX 4060 / 3.125 GiB weights\n256 MiB, window 512",
          "server-mlp1": "V100 / MLP, 1 GPU\n64 MiB, window 128",
          "server-mlp2": "V100 / MLP, 2 GPUs\n64 MiB, window 128",
          "server-pressure": "V100 / 12.5 GiB weights\n1024 MiB, window 512"}
titles = {g: title for g, title in titles.items() if any(r["group"] == g for r in data["comparisons"])}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
height = 5.8 if len(titles) == 3 else 10.2
fig, axes = plt.subplots((len(titles) + 2) // 3, 3, figsize=(15, height), layout="constrained")
for axis, (group, title) in zip(axes.flat, titles.items()):
    rows = [r for r in data["comparisons"] if r["group"] == group]
    rows.sort(key=lambda r: order.index(r["name"]))
    for i, row in enumerate(rows):
        t = row["online_ms"]
        axis.barh(i, t["median"] / 1000, color=colors[row["name"]], height=.65,
                  xerr=[[max(0, t["median"] - t["min"]) / 1000],
                        [max(0, t["max"] - t["median"]) / 1000]], capsize=3)
        axis.text(t["max"] / 1000, i, f"  {t['median']/1000:.3f}", va="center", fontsize=9)
    axis.set_yticks(range(len(rows)), [labels[r["name"]] for r in rows])
    axis.invert_yaxis()
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Online execution (seconds)")
    axis.set_xlim(0, max(r["online_ms"]["max"] for r in rows) / 1000 * 1.19)
    axis.grid(axis="x", alpha=.15)
    axis.set_axisbelow(True)
fig.text(.5, .985, "Historical executor vs fixed executor — identical current GPU backend",
         ha="center", va="top", fontsize=13)
fig.get_layout_engine().set(rect=(0, .10, 1, .82))
fig.text(.5, .025, "Fresh processes; RTX 4060 MLP: n=7, others: n=3. Bars: median; whiskers: min/max.\n"
         "Eager excludes upfront preparation; Waterline includes runtime preparation. q_proj GPU window = 128.",
         ha="center", fontsize=9)
fig.savefig(out / "comparison.png", dpi=170)
