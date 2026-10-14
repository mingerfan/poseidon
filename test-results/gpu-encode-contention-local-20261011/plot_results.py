"""Regenerate the two result figures with matplotlib; no GPU required."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

base = Path(__file__).resolve().parent
saturation = json.loads((base / "repeated/summary.json").read_text())
demand = json.loads((base / "demand/summary.json").read_text())
ops = ["add", "multiply_plain", "multiply", "rescale", "relinearize", "rotate"]
labels = ["add", "mul_plain", "multiply", "rescale", "relin", "rotate"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
for ax, (q, b) in zip(axes.flat, [(8, 1), (8, 8), (32, 1), (32, 8)]):
    rows = {r["operator"]: r for r in saturation["rows"] if r["q_limbs"] == q and r["batch"] == b}
    for offset, name, color, label in [
        (-.2, "cuda_encode", "#3984b8", "CUDA Encode"),
        (.2, "tilelang_tensor_encode", "#dc8740", "TileLang Tensor Encode"),
    ]:
        vals = [rows[op]["backgrounds"][name]["operator_slowdown"] for op in ops]
        ax.bar([i + offset for i in range(6)], vals, .36, color=color, label=label)
        for i, v in enumerate(vals):
            ax.text(i + offset, v * 1.055, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1, color="#555", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(.8, 12)
    ax.set_yticks([1, 2, 5, 10])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}x"))
    ax.set_xticks(range(6), labels, rotation=25, ha="right")
    ax.set_title(f"Q={q}, Encode batch={b}")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
axes[0, 0].legend(loc="upper left", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("Operator median / paired alone baseline")
fig.suptitle("Continuous background Encode: operator interference on RTX 4060\n"
             "N=65536, Q/P primes 30-bit, P=4; 3 process runs, raw H2D included", fontsize=12)
fig.savefig(base / "continuous-slowdown.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, q in zip(axes, [8, 32]):
    for r in demand["rows"]:
        if r["q_limbs"] != q:
            continue
        b = r["backgrounds"]["tilelang_tensor_encode_demand"]
        x, y = b["encoded_plaintexts_per_operator"], b["operator_mean_slowdown"]
        relin = r["operator"] == "relinearize"
        batch = r["batch"]
        ax.scatter(x, y, s=65, c="#3984b8" if relin else "#dc8740", marker="o" if batch == 1 else "s")
        offset, align = (8, -18), "left"
        if relin and batch == 8:
            offset = (8, 10)
        elif relin and q == 8:
            offset, align = (-8, -15), "right"
        elif not relin and batch == 8 and q == 32:
            offset, align = (-8, 14), "right"
        ax.annotate(f"{r['operator']} B{batch}", (x, y), xytext=offset,
                    textcoords="offset points", fontsize=8, ha=align)
    ax.axvline(1, color="#666", linestyle="--", linewidth=1)
    ax.axhline(1, color="#666", linestyle="--", linewidth=1)
    ax.set_xlim(.4, 1.5)
    ax.set_ylim(.98, 1.35)
    ax.grid(alpha=.15)
    ax.set_title(f"Q={q}, P=4")
    ax.set_xlabel("Encoded plaintexts / operator call")
axes[0].set_ylabel("Operator mean latency / alone baseline")
fig.suptitle("Demand-limited Tensor Encode: supply and compute overhead\n"
             "RTX 4060, N=65536; target 1 plaintext/operator, 3 process runs", fontsize=11)
fig.savefig(base / "demand-tradeoff.png", dpi=150)
