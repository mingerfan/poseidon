"""Current-code compatibility checks for trusted rule scheduling, no provider calls."""
import json
import shlex
import sys
import unittest
from pathlib import Path
ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT/"scripts/baseline"))

def main():
    from hecate_python_env import VENV, WORK, enter_nix
    output=Path(sys.argv[1])
    if "--inside" not in sys.argv[2:]:
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),str(output),"--inside"])
        return enter_nix(command,seconds=600,environment="poseidon-cpu")
    from benchmark_graph import require, canonical
    from check_poseidon_helpers import snapshot
    from poseidon_backend_identity import identity
    require(not output.exists() and output.parent.resolve()==WORK/"results","Fresh current-code result")
    output.mkdir()
    frozen=snapshot()
    backend=identity()
    names=["test_unified_graph_scheduling","test_packed_prefix_lowering","test_unified_graph",
           "test_validation_adapter","test_application_component","test_agent_component",
           "test_candidate_bundle","test_compiler_configuration","test_unified_poseidon_guidance"]
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    require(snapshot()==frozen and identity()==backend,"Source/backend changed during tests")
    (output/"report.json").write_bytes(canonical(dict(
        tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        source_hashes=frozen,backend_identity=backend,names=names,paid_calls=0,
        actual_compile=False,encrypted_execution=False)))
    return 0 if result.wasSuccessful() else 1

if __name__=="__main__":raise SystemExit(main())
