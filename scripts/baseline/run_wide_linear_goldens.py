"""Manual 8-neuron golden and omitted-neuron counterexample, no Agent calls."""
from hecate_python_env import ROOT
from run_schema3_goldens import main as run_suite

if __name__ == "__main__":
    raise SystemExit(run_suite(
        plans=[("custom-wide-mlp-8", "wide8", False),
               ("custom-wide-mlp-8", "wide8-wrong-neuron", True)],
        golden_dir=ROOT / "scripts/baseline/golden_cases/wide_linear",
        prefix="wide-linear-goldens-", title="Wide Linear", input_counts=(1,)))
