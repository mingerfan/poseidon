"""Build/check a bounded paid-evaluation proposal. Never reads credentials or calls an API."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from benchmark_graph import digest, validate
from benchmark_runner import dump, load, strict_file

SUITE = BASE / "benchmarks/semantic-v2-ledger-r38"
CONSTRUCTIONS = ("construct_000_0", "construct_057_0", "construct_134_0")
HELPERS = ("upstream_silu_nested_native", "upstream_bn_multi_residual")
CONFIG = "seal-cpu-eva-w40-v1"
PAID = dict(provider="deepseek", model="deepseek-flash", reasoning_effort="high",
            stream=True, api_timeout_seconds=1200, max_tokens=384000,
            max_repairs=3, provider_retries=3)
LIMITS = dict(concurrency=1, compile_jobs=2, link_jobs=1,
              max_wall_seconds=3600, max_retained_mib=1024,
              maximum_generations=48, maximum_http_attempts=192,
              monetary_cap=None, timeout_retries_may_duplicate_billing=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_binding(plan):
    if plan.get("binding") != digest({k:v for k,v in plan.items() if k != "binding"}):
        raise ValueError("Proposal binding changed")

def small(model):
    shape = validate(model)
    return (sum(math.prod(x["shape"]) for x in model["inputs"]) <= 32
            and sum(math.prod(x) for x in shape["output_shapes"].values()) <= 16
            and len(model["nodes"]) <= 16)

def build_plan():
    from semantic_benchmark_execution import runtime_sources, preflight, representative
    from compiler_configuration import PROFILE_SHA256, configuration
    from unified_graph_contract import prepare, validate_request
    from hecate_python_env import VENV
    from platform_config import identity, require_python_packages
    from deepseek_provider import Config, MODEL_OUTPUT_LIMITS
    import numpy as np
    import torch
    require_python_packages(torch, np)
    Config(model=PAID["model"], service_provider=PAID["provider"],
           reasoning_effort=PAID["reasoning_effort"], max_calls=4,
           max_tokens=PAID["max_tokens"], timeout_seconds=PAID["api_timeout_seconds"],
           stream=True, provider_retries=3)
    if MODEL_OUTPUT_LIMITS[PAID["model"]] < PAID["max_tokens"]:
        raise ValueError("Provider output limit changed")
    sources = runtime_sources()
    accepted_path = ROOT / "docs/baseline/stage2-offline-acceptance-r94.json"
    accepted = strict_file(accepted_path, 8*1024**2)
    check_binding(accepted)
    if not accepted["offline_engineering_acceptance_complete"] or digest(sources) != accepted["current_runtime_sha256"]:
        raise ValueError("Current runtime is not the accepted offline version")
    # Historical corpus source metadata stays immutable. load still checks every
    # shard, model signature, topology and split; current sources bind separately.
    rows, index = load(SUITE, check_sources=False)
    ledger = strict_file(SUITE / "coverage.json", 4*1024**2)
    models = {r["model"]["id"]:r for r in rows}
    eligible = [r for r in rows if r["split"] == "development" and small(r["model"])]
    selected = representative(preflight(eligible, need_rule=False), 6)
    if len(selected) != 6:
        raise ValueError("Six development free cases required; never substitute based on FHE success")
    specs = [dict(id="free_"+x["id"], track="free", model=models[x["id"]]["model"],
                  origin=dict(model_id=x["id"], split="development",
                              topology=models[x["id"]]["topology"])) for x in selected]
    tasks = {x["id"]:x for x in ledger["directed_tasks"]}
    for task_id in CONSTRUCTIONS:
        t = tasks[task_id]
        m = models[t["model_id"]]
        specs.append(dict(id=task_id, track="directed_construction", model=m["model"],
                          exercise=t["exercise"], construction_profile=t["profile"],
                          origin=dict(task_sha256=t["task_sha256"], requirement=t["requirement"],
                                      split=m["split"], topology=m["topology"])))
    helper_tasks = {x["id"]:x for x in ledger["helper_directed_tasks"]}
    for task_id in HELPERS:
        t = helper_tasks[task_id]
        specs.append(dict(id=task_id, track="directed_helper", model=t["model"],
                          helper_profile=t["profile"], required_helpers=t["required_helpers"],
                          compiler_configuration=t["compiler_configuration"],
                          origin=dict(task_sha256=t["task_sha256"], topology=t["topology"])))
    from fixed_polynomial_cases import cases
    fixed = next(x for x in cases() if x["id"] == "fixed_Poly_Default_2")
    # Extract mathematical model and requirement only, never the fixture DSL answer.
    specs.append(dict(id=fixed["id"], track="directed_helper", model=fixed["model"],
                      helper_profile=fixed["profile"], required_helpers=[fixed["helper"]],
                      origin=dict(fixture_id=fixed["id"])))
    for spec in specs:
        profile = spec.get("construction_profile")
        public = profile if profile == "hecate-unified-public-v1" else None
        config = spec.setdefault("compiler_configuration", CONFIG)
        request = prepare(spec["model"], PROFILE_SHA256, configuration(config),
                          spec.get("exercise"), construction_profile=public,
                          helper_profile=spec.get("helper_profile"),
                          helper_exercise=spec.get("required_helpers"))
        validate_request(request)
        args = ["--compiler-configuration", config]
        if public:
            args += ["--unified-profile", "public-v1"]
        if spec.get("exercise"):
            args += ["--unified-exercise", spec["exercise"]]
        if spec.get("helper_profile"):
            args += ["--unified-helpers", spec["helper_profile"]]
            for name in spec["required_helpers"]:
                args += ["--unified-helper-exercise", name]
        args += ["--live", "--provider", PAID["provider"], "--model", PAID["model"],
                 "--reasoning-effort", PAID["reasoning_effort"], "--stream",
                 "--api-timeout", str(PAID["api_timeout_seconds"]),
                 "--max-tokens", str(PAID["max_tokens"]), "--max-repairs", "3",
                 "--provider-retries", "3"]
        spec.update(request=request, request_id=request["request_id"],
                    model_sha256=digest(spec["model"]), candidate_arguments=args,
                    ops=sorted({n["op"] for n in spec["model"]["nodes"]}),
                    status="not_run", agent_generated=False)
    files = {str(p.relative_to(ROOT)):sha(p) for p in
             (Path(__file__), Path(__file__).with_name("fixed_polynomial_cases.py"),
              Path(__file__).with_name("run_stage2_agent_pilot.py"),
              Path(__file__).with_name("stage2_agent_provenance.py"),
              Path(__file__).with_name("audit_stage2_live_candidate.py"),
              Path(__file__).with_name("test_stage2_agent_provenance.py"),
              Path(__file__).with_name("test_stage2_agent_pilot.py"),
              accepted_path, SUITE/"index.json", SUITE/"coverage.json")}
    plan = dict(format="poseidon-stage2-paid-pilot-proposal-v1",
                source_hashes=sources, proposal_files=files, platform=identity(),
                corpus_model_set_sha256=index["model_set_sha256"], cases=specs,
                paid_configuration=PAID, limits=LIMITS, paid_calls=0,
                approved=False, executable=str(VENV/"bin/python"),
                entrypoint=str(BASE/"run_candidate.py"),
                selection="Six small development models: greedy semantic gain then node cost; no FHE outcome filtering. Six explicit directed tasks.",
                outbound_content=["public model and weights", "frozen layout and DSL rules",
                                  "response format", "specified construction requirements",
                                  "candidate code and restricted repair diagnostics"],
                never_outbound=["test inputs", "reference output arrays", "golden DSL answer",
                                ".env", "FHE secret keys", "repository contents"],
                limitations=["Pilot only; not full Stage2 Agent acceptance",
                             "Directed model split is disclosed, not treated as hidden holdout",
                             "No currency hard cap; no implicit further batch approval",
                             "No plan command performs execution or reads credentials"])
    if runtime_sources() != sources:
        raise ValueError("Source changed while planning")
    plan["binding"] = digest(plan)
    return plan

def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="New proposal JSON, never overwrite")
    mode.add_argument("--check", type=Path, help="Recompute all bindings without executing")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    from hecate_python_env import VENV, enter_nix
    if not a.inside:
        command = [str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()),
                   *sys.argv[1:], "--inside"]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
                         shlex.join(command), seconds=300)
    if os.environ.get("IN_NIX_SHELL") != "pure" or Path(sys.prefix) != VENV:
        p.error("Pinned pure Nix Python required")
    if a.output and a.output.exists():
        p.error("Preserve proposal; use --check or a new path")
    plan = build_plan()
    if a.check:
        old = strict_file(a.check, 8*1024**2)
        check_binding(old)
        if old != plan:
            raise ValueError("Proposal/source/configuration changed; new review required")
    else:
        dump(a.output, plan)
    print(json.dumps(dict(binding=plan["binding"], cases=[
        dict(id=c["id"],track=c["track"],ops=c["ops"],request_id=c["request_id"])
        for c in plan["cases"]], limits=plan["limits"], paid_calls=0,
        approval_status="not_requested_by_tool", checked=bool(a.check)), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
