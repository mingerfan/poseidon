"""Offline acceptance of the explicit worker adapter using existing program paths."""
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT / "scripts/baseline"))

def run(output):
    import numpy as np
    from application_execution.context import WorkerContext, canonical, sha
    from application_execution.backend import ContextBackend, application
    from application_component import Application
    from benchmark_graph import require
    from benchmark_math import evaluate as mathematical
    from benchmark_torch import evaluate as tensor_reference
    from candidate_bundle import check_bundle, check_execution_dependencies, replay_bundle
    from numerical_results import compare
    from unified_graph_lowering import candidate_source
    from hecate_python_env import WORK
    from benchmarks.tools.run_application_matrix import freeze, items_from_plan

    frozen = json.loads((ROOT / "docs/baseline/agent-pytorch-state-live-proposal-r635.json").read_text())
    require(freeze(items_from_plan(frozen)) == frozen, "Pending paid proposal binding changed")
    old = json.loads((ROOT / "docs/baseline/agent-generalization-live-proposal-r620.json").read_text())
    linear = next(c for c in old["cases"] if c["id"] == "weights_linear")
    selected = [linear] + [next(c for c in frozen["cases"] if c["id"] == name)
        for name in ("pytorch_state_mlp", "pytorch_state_matrix_fusion")]
    require(not output.exists() and output.parent.resolve() == (WORK / "results").resolve(), "Fresh evidence directory")
    output.mkdir(mode=0o700)
    context = WorkerContext.capture(output / "tasks")
    backend = ContextBackend(context)
    plan = dict(format="poseidon-context-offline-plan-v1", cases=[
        dict(id=c["id"], imported_directory=c["imported_directory"], options=c["options"],
             model_sha256=sha((Path(c["imported_directory"]) / "model.json").read_bytes()))
        for c in selected], worker_binding=context.binding,
        source_authority=context.record()["source_authority"], backend=context.record()["backend_identity"],
        pending_paid_binding=frozen["binding"], source="rule baseline", paid_calls=0,
        tool_sha256=sha(Path(__file__).read_bytes()), max_seconds=3600,
        artifact_bytes=4*1024**3, global_bytes=32*1024**3, free_bytes=4*1024**3,
        new_samples=3, repetitions=2, random_seed=20261006, atol=1e-5, rtol=1e-4)
    plan["binding"] = sha(canonical(plan))
    (output / "plan.json").write_bytes(canonical(plan))
    (output / "source-snapshot").mkdir(mode=0o700)
    snapshot = {}
    for p in sorted((ROOT / "scripts/baseline/application_execution").rglob("*.py")):
        raw = p.read_bytes()
        relative = str(p.relative_to(ROOT / "scripts/baseline/application_execution"))
        target = output / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        snapshot[relative] = sha(raw)
    (output / "source-snapshot/manifest.json").write_bytes(canonical(snapshot))
    rows, retained, started = [], {output}, time.monotonic()

    def boundary():
        context.check_current()
        require(time.monotonic() - started < plan["max_seconds"], "Offline wall budget")
        require(shutil.disk_usage(WORK).free >= plan["free_bytes"], "Free disk budget")
        total = int(subprocess.check_output(["du", "-sb", str(WORK / "results")],
                                             text=True, timeout=30).split()[0])
        require(total <= plan["global_bytes"], "Global artifact budget")
        size = sum(p.stat().st_size for folder in retained for p in folder.rglob("*") if p.is_file())
        require(size <= plan["artifact_bytes"], "Offline artifact budget")

    def save(status):
        report = dict(format="poseidon-context-offline-report-v1", binding=plan["binding"], status=status,
            planned=len(selected), passed=sum(r["status"] == "passed" for r in rows),
            failed=sum(r["status"] == "failed" for r in rows), not_run=len(selected)-len(rows),
            rows=rows, seconds=time.monotonic()-started, source="rule baseline",
            agent_generated=False, paid_calls=0, pending_paid_binding=frozen["binding"])
        (output / "report.json").write_bytes(canonical(report))

    def collect(before):
        retained.update(set((WORK / "results").iterdir()) - before)

    class Offline:
        kind = "scripted_replay"
        agent_calls = 0
        def generate(self, request, history):
            return json.dumps(dict(schema=1, request_id=request["request_id"],
                                   hecate_source=candidate_source(request)[0]))

    app = application(context)
    save("running")
    try:
        for index, case in enumerate(selected):
            boundary()
            before = set((WORK / "results").iterdir())
            task = app.submit_import(case["imported_directory"], options=case["options"],
                limits=dict(generations=1, http_attempts=1, wall_seconds=1800,
                            api_timeout_seconds=1200, max_tokens=384000))
            row = dict(id=case["id"], task_id=task, status="running", new_input_runs=[])
            rows.append(row); save("running")
            result = app.run(task, Offline())
            collect(before); row["result"] = result; save("running")
            if result["status"] != "succeeded":
                row["status"] = "failed"; save("running")
                print(json.dumps(dict(id=case["id"], status="failed", failure=result.get("failure"))), flush=True)
                if result["status"] in ("service_error", "timed_out", "cancelled"):
                    raise RuntimeError("Context acceptance infrastructure failure")
                continue
            task_path = output / "tasks" / task
            evidence = json.loads((task_path / "validation-0.json").read_text())
            require(evidence["run_context"]["worker_binding"] == context.binding,
                    "Qualification worker receipt")
            package = task_path / "program"
            manifest = check_bundle(package)
            require(not manifest["agent_generated"] and result["http_attempts"] == 0,
                    "Offline source classification")
            before = set((WORK / "results").iterdir())
            require(replay_bundle(package) == 0, "Original four-input package replay")
            collect(before)
            replay_paths = [p for p in set((WORK / "results").iterdir()) - before
                            if p.name.startswith("poseidon-package-replay-")]
            require(len(replay_paths) == 1, "Unique package replay evidence")
            replay = json.loads((replay_paths[0] / "report.json").read_text())
            row.update(package=str(package), package_binding=manifest["binding"], reload=replay,
                       qualification_context=evidence["run_context"])

            # A fresh host instance reloads the same checked context and task store.
            fresh_context = WorkerContext.capture(output / "tasks")
            require(fresh_context == context, "Worker context reload")
            restored = application(fresh_context)
            require(restored.result(task)["status"] == "succeeded", "Application task reload")
            checked, request = check_execution_dependencies(package)
            require(checked == manifest, "Checked package request identity")
            model = request["model"]
            samples, expected = [], []
            for probe in range(plan["new_samples"]):
                rng = np.random.default_rng(plan["random_seed"] + index*100 + probe)
                sample = {s["name"]: rng.uniform(-.25, .25, s["shape"]).tolist() for s in model["inputs"]}
                inputs = {name: np.asarray(value, dtype=np.float64) for name, value in sample.items()}
                values = mathematical(model, inputs); torch_values = tensor_reference(model, inputs)
                for name in values:
                    np.testing.assert_allclose(values[name], torch_values[name], atol=1e-12, rtol=1e-12)
                samples.append(sample)
                expected.append(np.concatenate([np.asarray(values[s["name"]]).reshape(-1)
                                                for s in model["outputs"]]))
            (output / (case["id"] + "-inputs.json")).write_bytes(canonical(samples))
            row["new_input_sha256"] = sha(canonical(samples))
            for repetition in range(plan["repetitions"]):
                boundary(); before = set((WORK / "results").iterdir())
                execution = restored.execute(task, samples)
                collect(before)
                require(execution["status"] == "executed" and execution["encrypted_execution"],
                        "Real context package execution")
                require(execution["agent_calls"] == execution["compiler_calls"] == 0 and
                        execution["executor_secret_key"] is False, "Execution boundaries")
                require(execution["run_context"]["worker_binding"] == context.binding,
                        "Execution worker receipt")
                actual = np.asarray([np.concatenate([np.asarray(values[s["name"]]).reshape(-1)
                                       for s in model["outputs"]]) for values in execution["outputs"]])
                compared = compare(actual, np.asarray(expected), plan["atol"], plan["rtol"])
                require(compared["passed"], "New input numerical comparison")
                row["new_input_runs"].append(dict(repetition=repetition, evidence=execution["evidence"],
                    comparison=compared, run_context=execution["run_context"],
                    output_structure=execution.get("output_structure"),
                    compiler_calls=0, agent_calls=0))
                save("running")

            # Cache reuse must neither generate nor compile.
            reused_task = restored.submit_import(case["imported_directory"], options=case["options"])
            class NoGeneration:
                kind = "scripted_replay"; agent_calls = 0
                def generate(self, *_):
                    raise AssertionError("Cache reuse must not generate")
            reused = restored.run(reused_task, NoGeneration())
            require(reused["status"] == "succeeded" and reused["package"]["reused"] and
                    reused["generations"] == reused["http_attempts"] == 0, "Bounded package cache reuse")
            row.update(status="passed", cache_reuse=reused)
            save("running")
            print(json.dumps(dict(id=case["id"], status="passed",
                max_absolute_error=result["acceptance"]["maximum_absolute_error"],
                worker_binding=context.binding)), flush=True)
        boundary()
        require(freeze(items_from_plan(frozen)) == frozen, "Pending paid binding drift")
        save("complete")
    except BaseException:
        if rows and rows[-1]["status"] == "running":
            rows[-1]["status"] = "failed"
        save("stopped")
        raise
    return 0 if all(row["status"] == "passed" for row in rows) else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))
