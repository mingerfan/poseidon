"""Developer audit of explicit PyTorch input spellings, using isolated capture.

This records capture/reference acceptance and refusal. It does not generate DSL,
compile, execute FHE or certify general support for an operator.
"""
import argparse, hashlib, json, os, shlex, shutil, sys, time
from pathlib import Path
ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT / "scripts/baseline"))
CASES = ROOT / "scripts/baseline/cases/pytorch_form_audit_v1"
FORMAT = "poseidon-pytorch-input-form-audit-v1"

def inventory():
    from benchmark_runner import strict_file
    from benchmark_graph import require
    from trusted_pytorch_import import prepare
    index = strict_file(CASES / "index.json", 131072)
    require(set(index) == {"format", "cases"} and index["format"] == FORMAT, "Audit index")
    require(type(index["cases"]) is list and 1 <= len(index["cases"]) <= 48, "Audit case count")
    records = []
    for case in index["cases"]:
        require(set(case) == {"id", "manifest", "expected", "reason"}, "Audit case fields")
        require(case["expected"] in ("reference_passed", "rejected"), "Audit expectation")
        name = case["manifest"]
        require(type(name) is str and Path(name).name == name and name.endswith(".json"), "Audit manifest path")
        spec, _ = prepare(CASES / name)
        records.append(dict(**case, manifest_sha256=hashlib.sha256((CASES / name).read_bytes()).hexdigest(),
                            source_hashes=spec["files"]))
    require(len({c["id"] for c in records}) == len(records), "Duplicate audit cases")
    return records

def run(output, selected=None):
    from benchmark_graph import canonical, digest, require
    from hecate_python_env import WORK, VENV
    from semantic_benchmark_execution import runtime_sources
    from trusted_pytorch_import import run as capture
    require(os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV, "Pinned audit worker")
    require(not output.exists() and output.parent.resolve() == (WORK / "results").resolve(), "Fresh audit output")
    frozen_cases = inventory()
    cases = frozen_cases
    if selected:
        require(len(selected)==len(set(selected)) and set(selected)<={c["id"] for c in frozen_cases},
                "Explicit audit selection")
        cases = [c for c in frozen_cases if c["id"] in selected]
    sources = runtime_sources()
    tool_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(mode=0o700)
    plan = dict(format=FORMAT, cases=cases, source_hashes=sources, tool_sha256=tool_hash,
                max_seconds=900, artifact_bytes=32*1024**2, paid_calls=0, encrypted_execution=False)
    plan["binding"] = digest(plan)
    (output / "plan.json").write_bytes(canonical(plan))
    rows = []
    started = time.monotonic()
    def boundary():
        require(runtime_sources() == sources and inventory() == frozen_cases
                and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == tool_hash,
                "Audit source drift")
        require(time.monotonic() - started < plan["max_seconds"], "Audit wall budget")
        require(shutil.disk_usage(WORK).free >= 4*1024**3, "Audit disk reserve")
        require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < plan["artifact_bytes"],
                "Audit artifact budget")
    def save(status):
        report = dict(format=FORMAT, binding=plan["binding"], status=status, planned=len(cases),
            rows=rows, reference_passed=sum(r["status"]=="reference_passed" for r in rows),
            rejected=sum(r["status"]=="rejected" for r in rows),
            unexpected=sum(r["status"]!=r["expected"] for r in rows), seconds=time.monotonic()-started,
            paid_calls=0, compiler_calls=0, encrypted_execution=False)
        (output / "report.json").write_bytes(canonical(report))
    save("running")
    try:
        for case in cases:
            boundary()
            folder = output / case["id"]
            row = dict(id=case["id"], expected=case["expected"], reason=case["reason"])
            try:
                result = capture(CASES/case["manifest"], folder,
                                 resource_profile="general-tensors-v1", trusted=True)
                row.update(status="reference_passed", capture=result)
                ref = json.loads((folder/"reference-report.json").read_text())
                row["reference"] = {k:ref[k] for k in ("probes", "max_decomposition_error",
                    "max_original_module_error", "max_original_function_approximation_error",
                    "source_precision_execution_compared")}
            except ValueError as error:
                # Only a child-side capture refusal is a probe result. Admission,
                # integrity and environment failures stop the batch.
                require(str(error)=="Trusted capture failed; inspect capture.log", "Audit admission/environment error")
                log = (folder/"capture.log").read_text()
                row.update(status="rejected", diagnostic=str(error),
                           child_error=next((line[:512] for line in reversed(log.splitlines())
                               if "Error:" in line and not line.startswith(" ")), "Child capture error"))
            rows.append(row)
            boundary()
            save("running")
            print(json.dumps(dict(id=row["id"], status=row["status"],
                                  expected=row["expected"])), flush=True)
        save("complete")
        return 0 if all(r["status"]==r["expected"] for r in rows) else 1
    except BaseException:
        save("stopped")
        raise

def main():
    from hecate_python_env import VENV, enter_nix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Run only named, original audit cases")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.inside:
        command = [str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()),
                   "--inside", "--output", str(args.output.resolve())]
        for name in args.case or []: command += ["--case", name]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 '
                         'OMP_NUM_THREADS=1 '+shlex.join(command), seconds=930, environment="poseidon-cpu")
    return run(args.output.resolve(), args.case)

if __name__=="__main__": raise SystemExit(main())
