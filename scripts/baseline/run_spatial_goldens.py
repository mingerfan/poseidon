"""Nine manual spatial goldens and two wrong-window counterexamples; no LLM calls."""
from hecate_python_env import ROOT
from run_schema3_goldens import main as run_suite

CASES = ("conv1d-valid", "conv1d-stride-pad", "conv2d-valid", "conv2d-channels",
         "avg1d-pad-count", "avg1d-pad-exclude", "avg2d-valid", "avg2d-wide", "conv-square-pool")
PLANS = [(name, name, False) for name in CASES] + [
    (name, name+"-wrong-window", True) for name in ("conv1d-valid", "avg1d-pad-exclude")]

if __name__ == "__main__":
    raise SystemExit(run_suite(plans=PLANS, golden_dir=ROOT / "scripts/baseline/golden_cases/spatial",
                              prefix="spatial-golden-batch-", title="Spatial", input_counts=(1,)))
