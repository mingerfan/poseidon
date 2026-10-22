"""Offline extension checks in the pinned environment, no provider calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import unittest
def main():
    p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    from hecate_python_env import enter_nix,VENV,WORK,ROOT
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--output",str(a.output)]),seconds=600)
    if a.output.exists():raise ValueError("Preserve previous result")
    a.output.mkdir(parents=True)
    names=["test_model_decomposition","test_operator_extension","test_pytorch_capture","test_tensor_expansion","test_extrema_expansion"]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    source={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (ROOT/"scripts/baseline").glob("*.py")}
    report=dict(tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),source_hashes=source,paid_calls=0,agent_generated=False)
    (a.output/"unit.json").write_text(json.dumps(report,indent=2))
    if not result.wasSuccessful():return 1
    from trusted_pytorch_import import run
    report["capture"]=run(ROOT/"scripts/baseline/cases/trusted-pytorch-v1/manifest.json",
                          a.output/"trusted-capture",trusted=True)
    # These comparisons use a separate trusted reconstruction of the fixture.
    import numpy as np
    from benchmark_graph import samples
    from benchmark_math import evaluate as math_ref
    from benchmark_torch import evaluate as torch_ref
    g=json.loads((a.output/"trusted-capture/model.json").read_text())
    largest=0.
    for values in samples(g,16):
        h=values["x"]@np.array([[.1,-.2],[.2,.1],[.3,.2],[.4,.1]],dtype=np.float32).astype(np.float64)
        h+=np.array([.01,-.02],dtype=np.float32).astype(np.float64)
        expected=(h*h)@np.array([[.4,-.1],[.2,.3]],dtype=np.float32).astype(np.float64)+np.array([.02,.03],dtype=np.float32).astype(np.float64)
        for evaluate in (math_ref,torch_ref):
            actual=evaluate(g,values)["prediction"]
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            largest=max(largest,float(np.max(np.abs(actual-expected))))
    report["capture_reference"]=dict(probes=16,max_absolute_error=largest)
    (a.output/"acceptance.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="source_hashes"}))
    return 0
if __name__=="__main__":raise SystemExit(main())
