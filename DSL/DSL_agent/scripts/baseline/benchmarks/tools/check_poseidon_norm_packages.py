"""Export and replay two numerical Poseidon fixtures without generation or recompilation."""
import contextlib
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path
ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT/"scripts/baseline"))

def main():
    from hecate_python_env import VENV, WORK, enter_nix
    output = Path(sys.argv[1])
    if "--inside" not in sys.argv[2:]:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()), str(output), "--inside"])
        return enter_nix(command, seconds=650, environment="poseidon-cpu")
    from benchmark_graph import require, canonical, digest
    from semantic_benchmark_execution import runtime_sources, verify_numerical_report
    from candidate_bundle import export_bundle, check_bundle, replay_bundle
    require(output.resolve().parent == WORK/"results" and not output.exists(), "Fresh platform evidence")
    output.mkdir()
    source = runtime_sources()
    inputs = [
        ("layer_norm_2", WORK/"results/poseidon-normalization-rest-r318/report.json"),
        ("batched_matmul_1", WORK/"results/poseidon-norm-matmul-r314/report.json"),
    ]
    plan = dict(format="poseidon-normalization-package-plan-v1", sources=source,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                max_seconds=600, max_bytes=1024**3, cases=[{"id":i, "report":str(p),
                "report_sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for i,p in inputs],
                agent_calls=0, compiler_calls=0, paid_calls=0)
    (output/"plan.json").write_bytes(canonical(plan))
    rows = []
    start = time.monotonic()
    def save():
        (output/"report.json").write_bytes(canonical(dict(
            format="poseidon-normalization-package-acceptance-v1",
            binding=digest(plan), rows=rows, planned=2, passed=sum(r["status"] == "passed" for r in rows),
            failed=sum(r["status"] != "passed" for r in rows), not_run=2-len(rows),
            seconds=time.monotonic()-start, agent_calls=0, compiler_calls=0, paid_calls=0)))
    save()
    for identifier, prior in inputs:
        require(runtime_sources() == source and time.monotonic()-start < 600, "Source/time boundary")
        prior_result = next(r for r in json.loads(prior.read_text())["rows"] if r["id"] == identifier)
        require(prior_result["status"] == "passed", "Numerical source evidence")
        evidence = Path(prior_result["evidence"])
        source_report = json.loads((evidence/"report.json").read_text())
        verify_numerical_report(source_report, "poseidon-cpu-eva-w45-v1")
        require(source_report["agent_calls"] == 0 and not source_report["llm_generation_validated"], "Manual source provenance")
        folder = output/identifier
        folder.mkdir()
        package = folder/"program"
        manifest = export_bundle(evidence, package)
        checked = check_bundle(package)
        require(checked["format"] == "poseidon-dsl-bundle-v4" and checked["validation"]["level"] == "numerical",
                "Numerical RuntimePlan package")
        before = set((WORK/"results").glob("poseidon-package-replay-*"))
        with (folder/"replay.log").open("x") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            require(replay_bundle(package) == 0, "Direct RuntimePlan replay")
        added = set((WORK/"results").glob("poseidon-package-replay-*"))-before
        require(len(added) == 1, "One reload evidence directory")
        reload_root = next(iter(added))
        reload = json.loads((reload_root/"report.json").read_text())
        require(reload["status"] == "passed" and reload["agent_calls"] == 0 and reload["compiler_calls"] == 0,
                "Reload numerical result with no generation or compile")
        require(reload.get("encrypted_execution") is True, "Actual encrypted reload")
        rows.append(dict(id=identifier, status="passed", program=str(package),
            package_binding=manifest["binding_sha256"], evidence=str(evidence),
            reload_evidence=str(reload_root), reload_report_sha256=hashlib.sha256((reload_root/"report.json").read_bytes()).hexdigest(),
            comparison={k:reload["comparison"][k] for k in ["max_absolute_error", "compared_values"]},
            actual_compile=False, encrypted_execution=True, agent_calls=0, compiler_calls=0))
        save()
        print("PACKAGE_RESULT", identifier, "passed", flush=True)
        require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= 1024**3, "Artifact boundary")
    require(runtime_sources() == source, "Final source boundary")
    save()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
