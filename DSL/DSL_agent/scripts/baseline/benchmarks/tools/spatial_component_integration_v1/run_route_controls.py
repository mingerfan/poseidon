"""Pinned component checks only; no compiler, encryption or model API."""
import sys,json,hashlib,os,shlex,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix
if "--inside"not in sys.argv:
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=180,keep_env=(),environment="poseidon-cpu"))
RESULT=HERE/"route-controls-r1399";RESULT.mkdir()
import unittest
from benchmark_graph import digest
from operator_spatial_graph import implementation_sources
modules=["test_operator_spatial_graph","test_operator_arithmetic_graph","test_operator_graph_v2","test_model_decomposition"]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in modules)
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(status="passed"if result.wasSuccessful()and not result.skipped else "failed",
    passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
    failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    modules=modules,implementation_sources=implementation_sources(),api_calls=0,compilations=0,encrypted_calls=0)
report["binding"]=digest(report)
with(RESULT/"controls.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps(report))
raise SystemExit(0 if report["status"]=="passed"else 1)
