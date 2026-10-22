"""Six grouped/dilated manual goldens and two counterexamples, no LLM calls."""
from hecate_python_env import ROOT
from run_schema3_goldens import main as run_suite

CASES = ("conv1d-dilation2", "conv1d-dilation3", "conv2d-dilation2",
         "conv1d-groups2", "conv2d-depthwise-multiplier", "conv2d-grouped-dilated-pad")
PLANS = [(name, name, False) for name in CASES] + [
    ("conv1d-dilation2", "conv1d-dilation2-wrong-tap", True),
    ("conv1d-groups2", "conv1d-groups2-wrong-group", True)]

if __name__ == "__main__":
    raise SystemExit(run_suite(plans=PLANS, golden_dir=ROOT / "scripts/baseline/golden_cases/grouped_spatial",
                              prefix="grouped-spatial-goldens-", title="Grouped/dilated Conv", input_counts=(1,)))
