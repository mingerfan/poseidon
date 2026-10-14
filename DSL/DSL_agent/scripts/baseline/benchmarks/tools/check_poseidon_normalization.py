"""Bounded Poseidon normalization expressions, separate from real Agent evaluation."""
import argparse
import hashlib
import json
import shlex
import shutil
import sys
import time
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT/"scripts/baseline"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inside", action="store_true")
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-id", action="append", default=[])
    args = p.parse_args()
    from hecate_python_env import VENV, WORK, enter_nix
    if not args.inside:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()), "--inside",
             "--preparation", str(args.preparation), "--output", str(args.output),
             *[v for identifier in args.case_id for v in ("--case-id", identifier)]])
        return enter_nix(command, seconds=650, environment="poseidon-cpu")
    from benchmark_graph import canonical, digest, require
    from semantic_benchmark_execution import runtime_sources
    from compiler_configuration import configuration
    from unified_graph_contract import prepare, validate_candidate
    from component_backend import qualify
    from rmsnorm_expression_probe import plaintext_check
    require(args.output.resolve().is_relative_to(WORK/"results") and not args.output.exists(), "Fresh evidence")
    source = json.loads(args.preparation.read_text())
    rows = source["rows"]
    require(all(r["status"] == "ready" for r in rows), "Prepared rows")
    if args.case_id:
        require(len(set(args.case_id)) == len(args.case_id) and set(args.case_id) <= {r["id"] for r in rows},
                "Selected case identities")
        rows = [r for r in rows if r["id"] in args.case_id]
    require(0 < len(rows) <= 48, "Shard size")
    frozen = runtime_sources()
    require(frozen == source["sources"], "Preparation source boundary")
    cfg = configuration("poseidon-cpu-eva-w45-v1")
    args.output.mkdir()
    plan = dict(format="poseidon-normalization-expression-plan-v1",
                sources=frozen, preparation_sha256=hashlib.sha256(args.preparation.read_bytes()).hexdigest(),
                cases=rows, backend="poseidon_cpu", compiler_configuration=cfg["name"],
                max_seconds=600, max_bytes=1024**3, global_bytes=32*1024**3,
                free_bytes=4*1024**3, max_native_tasks=1, paid_calls=0,
                manual_candidate=True, agent_generated=False,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/"plan.json").write_bytes(canonical(plan))
    start = time.monotonic()
    results = [dict(id=r["id"], status="not_run") for r in rows]
    def size(paths):
        return sum(f.stat().st_size for path in paths for f in path.rglob("*") if f.is_file())
    def save():
        report = dict(format="poseidon-normalization-expression-acceptance-v1", binding=digest(plan),
                      planned=len(rows), rows=results, seconds=time.monotonic()-start,
                      passed=sum(r["status"] == "passed" for r in results),
                      failed=sum(r["status"] == "failed" for r in results),
                      not_run=sum(r["status"] == "not_run" for r in results),
                      paid_calls=0, agent_generated=False, manual_candidate=True)
        (args.output/"report.json").write_bytes(canonical(report))
    save()
    for index, row in enumerate(rows):
        require(runtime_sources() == frozen, "Source changed")
        require(time.monotonic()-start < 600 and shutil.disk_usage(WORK).free >= 4*1024**3, "Time/disk boundary")
        model = json.loads(Path(row["model"]).read_text())
        source_code = Path(row["source_path"]).read_text()
        require(digest(model) == row["model_sha256"] and
                hashlib.sha256(source_code.encode()).hexdigest() == row["candidate_source_sha256"],
                "Frozen candidate/model")
        request = prepare(model, cfg["ckks_config_sha256"], cfg)
        require(request["request_id"] == row["request_id"], "Request identity")
        candidate = dict(schema=1, request_id=request["request_id"], hecate_source=source_code)
        validate_candidate(candidate, request)
        plaintext_check(source_code, request)
        result = qualify(request, candidate)
        results[index] = dict(id=row["id"], candidate_source_sha256=row["candidate_source_sha256"],
                              source_sha256=row["source_sha256"], model_sha256=row["model_sha256"],
                              **result)
        save()
        print("CASE_RESULT", row["id"], result["status"], result.get("failure", {}).get("layer"), flush=True)
        paths = [args.output]+[Path(r["evidence"]) for r in results if r.get("evidence")]
        require(size(paths) <= 1024**3 and size([WORK/"results"]) <= 32*1024**3, "Artifact boundary")
        require(result.get("failure", {}).get("layer") not in ("environment", "integrity", "sandbox"),
                "Shared environment/integrity fault")
    require(runtime_sources() == frozen, "Final source boundary")
    save()
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
