"""Plot paired full-GPU / CPU-FFT-cache measurements; no GPU required."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

base = Path(__file__).resolve().parent
continuous = json.loads((base / "continuous/summary.json").read_text())
demand = json.loads((base / "demand/summary.json").read_text())
cases = [(8, 1), (8, 8), (32, 1), (32, 8)]
ops = ["add", "multiply_plain", "multiply", "rescale", "relinearize", "rotate"]
labels = ["add", "mul_plain", "multiply", "rescale", "relin", "rotate"]
colors = ["#d8893f", "#3487b7"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
for ax, (q, batch) in zip(axes.flat, cases):
    rows = {r["operator"]: r for r in continuous["rows"]
            if r["q_limbs"] == q and r["batch"] == batch}
    for offset, name, color, label in zip(
            [-.2, .2], ["tilelang_tensor_encode", "cpu_fft_tilelang_tensor_encode"],
            colors, ["Full GPU Encode", "CPU FFT cache + GPU tail"]):
        values = [rows[op]["backgrounds"][name]["operator_mean_slowdown"] for op in ops]
        ax.bar([i + offset for i in range(6)], values, .36, color=color, label=label)
        for i, value in enumerate(values):
            ax.text(i + offset, value * 1.06, f"{value:.2f}", fontsize=7, ha="center")
    ax.axhline(1, color="#555", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(.8, max(12, max(values) * 1.3))
    ax.set_yticks([1, 2, 5, 10])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}x"))
    ax.set_xticks(range(6), labels, rotation=25, ha="right")
    ax.set_title(f"Q={q}, Encode batch={batch}")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
axes[0, 0].legend(fontsize=8, loc="upper left")
for ax in axes[:, 0]:
    ax.set_ylabel("Operator mean / paired alone baseline")
fig.suptitle("Continuous Tensor NTT Encode: does precomputing CPU FFT reduce interference?\n"
             "RTX 4060, N=65536, P=4; 3 process runs; actual input H2D included", fontsize=12)
fig.savefig(base / "continuous-comparison.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for col, q in enumerate([8, 32]):
    cases_q = [(1, "relinearize"), (8, "relinearize"), (1, "rotate"), (8, "rotate")]
    rows = {(r["batch"], r["operator"]): r for r in demand["rows"] if r["q_limbs"] == q}
    for offset, name, color, label in zip(
            [-.2, .2], ["tilelang_tensor_encode_demand", "cpu_fft_tilelang_tensor_encode_demand"],
            colors, ["Full GPU Encode", "CPU FFT cache + GPU tail"]):
        measured = [rows[case]["backgrounds"][name] for case in cases_q]
        for ax, field in zip(axes[:, col], ["operator_mean_slowdown", "encoded_plaintexts_per_operator"]):
            values = [r[field] for r in measured]
            ax.bar([i + offset for i in range(4)], values, .36, color=color, label=label)
            for i, value in enumerate(values):
                ax.text(i + offset, value + .015, f"{value:.2f}", fontsize=8, ha="center")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    for ax in axes[:, col]:
        ax.axhline(1, color="#555", linestyle="--", linewidth=1)
        ax.set_xticks(range(4), ["relin B1", "relin B8", "rotate B1", "rotate B8"])
    maximum = max(r["backgrounds"][name]["operator_mean_slowdown"]
                  for r in rows.values() for name in r["backgrounds"])
    axes[0, col].set_ylim(.95, max(1.4, maximum + .12))
    axes[0, col].set_title(f"Q={q}, P=4")
    maximum_supply = max(r["backgrounds"][name]["encoded_plaintexts_per_operator"]
                         for r in rows.values() for name in r["backgrounds"])
    axes[1, col].set_ylim(0, max(1.5, maximum_supply + .2))
axes[0, 0].legend(fontsize=8, loc="upper left")
axes[0, 0].set_ylabel("Operator mean / paired alone baseline")
axes[1, 0].set_ylabel("Observed encoded plaintexts / operator")
fig.suptitle("Demand-limited Tensor NTT Encode: compute overhead and actual supply\n"
             "Target 1 plaintext/operator; full GPU FFT vs cached CPU FFT; 3 process runs", fontsize=12)
fig.savefig(base / "demand-comparison.png", dpi=150)
