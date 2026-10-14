"""Actual source-task preparation using unchanged production capture and Poseidon identity.

This staged API audit stops at queued/cancelled; no Provider, tracing, compilation
or encrypted execution is performed. Cancellation uses an actual capture worker.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

from benchmark_graph import canonical, digest, require
from application_backend import PoseidonBackend
from candidate_bundle import load_model_provenance, check_provenance
from hecate_python_env import VENV, WORK, ROOT
from benchmarks.tools.model_source_component_prototype import SourceApplication, FORMAT, own_identity
from strict_json_files import dump


def frozen(proposal):
    for section in ("source_hashes", "case_hashes", "tool_hashes"):
        for name, expected in proposal[section].items():
            p = Path(name)
            if not p.is_absolute():
                p = ROOT / name
            require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == expected,
                    "Pending backend freeze changed: "+name)


def processes(task):
    rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdecimal():
            continue
        try:
            raw = (p/"cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if task.encode() in raw and any(x in raw for x in (b"prepare_model.py",b"bwrap",b"pytorch_model_intake.py")):
            rows.append(dict(pid=int(p.name),kind="sandbox" if b"bwrap" in raw else "intake_supervisor"))
    return rows


def run(output):
    require(os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV,
            "Pinned source acceptance environment")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),
            "Fresh evidence directory")
    proposal = json.loads((ROOT/"docs/baseline/agent-operator-shape-offline-proposal-r991.json").read_text())
    frozen(proposal)
    output.mkdir(mode=0o700)
    fixtures = output/"fixtures"
    fixtures.mkdir()
    rows = []
    catalog = ROOT/"scripts/baseline/cases/trusted-pytorch-operator-shape-v10"
    options = {}
    for name in ("adaptive_pool_0_functional_query","instance_norm_2_functional_query"):
        graph = json.loads((catalog/name/"operator-model.json").read_text())
        rows.append(dict(id="graph_"+name,kind="graph",source=dict(format=FORMAT,kind="graph",model=graph),
                         options=options,expected=None))
        rows.append(dict(id="pytorch_"+name,kind="trusted_pytorch",
                         source=dict(format=FORMAT,kind="trusted_pytorch",
                                     manifest=str(catalog/name/"manifest.json"),trust_local_code=True),
                         options=options,expected=None))
        rows[-1]["expected"] = WORK/"results/agent-operator-shape-public-v10-r988/captures"/name
    for width in (2,4):
        case = fixtures/("restricted_"+str(width))
        case.mkdir()
        sources = {
            "model.py":"from helper import transform\n\ndef model(x):\n    return transform(x)\n",
            "helper.py":"import torch\n\ndef transform(value):\n    return torch.square(value)\n",
        }
        spec = dict(format="poseidon-python-model-v1",id="source_square_"+str(width),
                    files={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()},
                    entry="model.model",inputs=[dict(name="x",shape=[width])],constants={},outputs=["result"])
        dump(case/"manifest.json",spec)
        for name,text in sources.items():
            (case/name).write_text(text)
        rows.append(dict(id="restricted_"+str(width),kind="restricted_python",
                         source=dict(format=FORMAT,kind="restricted_python",manifest=str(case/"manifest.json")),
                         options={},expected=None))
    app = SourceApplication(output/"tasks",PoseidonBackend())
    results = []
    retained = own_identity()
    total_groups = 0
    for row in rows:
        task = app.submit_source(row["source"],options=row["options"],validation_level="numerical",
                                 resource_profile="legacy-v1" if row["kind"]=="restricted_python" else "general-tensors-v1")
        initial = app.status(task)
        require(initial["status"]=="queued" and initial["phase"]=="intake_queued","Source admission")
        result = app.prepare_source(task)
        entry = dict(id=row["id"],kind=row["kind"],task_id=task,
                     status=result["status"],phase=result["phase"],failure=result["failure"],
                     encrypted_execution=False,agent_generated=False,paid_calls=0)
        if result["status"] == "queued":
            meta = app._checked_task_metadata(task)
            require(meta["validation_level"]=="numerical" and
                    result["generations"]==result["http_attempts"]==0 and
                    result["package"] is None and result["acceptance"] is None,
                    "No generation or backend acceptance")
            provenance = meta.get("model_provenance")
            require(provenance is not None,"Independent import provenance")
            check_provenance(provenance,meta["request"]["model"])
            report = provenance["reference_report"]
            require(report["reference_status"]=="passed" and report["probes"]==16 and
                    report["encrypted_execution"] is False and report["paid_calls"]==0,
                    "Independent source reference scope")
            total_groups += 16
            if row["expected"] is not None:
                expected=json.loads((row["expected"]/"model.json").read_text())
                require(canonical(expected)==canonical(meta["request"]["model"]),
                        "Same mathematical model as validated public source")
            entry.update(model_sha256=digest(meta["request"]["model"]),
                         request_id=meta["request"]["request_id"],reference_groups=16,
                         original_module_checked=report.get("original_module_checked",False),
                         active_seconds=result["intake"]["active_seconds"],
                         remaining_wall_seconds=meta["budget"]["wall_seconds"],
                         source_bound=meta["model_source"])
            require(app.cancel(task)["status"]=="cancelled","Queued conversion cancel")
        results.append(entry)
        dump(output/"progress.json",dict(rows=results))
        require(own_identity()==retained,"Staged controller changed")
    # Snapshot isolation: edits to the caller's files after submit cannot change
    # the captured source. These are owned fixture copies, not production cases.
    source_copy=fixtures/"external_source"
    shutil.copytree(catalog/"adaptive_pool_0_functional_query",source_copy)
    task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",
                               manifest=str(source_copy/"manifest.json"),trust_local_code=True),
                           options=options,resource_profile="general-tensors-v1")
    snapshot=(app._dir(task)/"source/models.py").read_bytes()
    (source_copy/"models.py").write_bytes(snapshot+b"\n# caller changed after submission\n")
    unchanged=app.prepare_source(task)
    require(unchanged["status"]=="queued","Caller source snapshot capture")
    preserved=app._checked_task_metadata(task)["model_provenance"]["trusted_pytorch"]["files"]["models.py"]
    require(preserved.encode()==snapshot,"Captured source snapshot changed")
    app.cancel(task)
    # Real cancellation once an actual snapshot-bound capture process is running.
    task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",
                               manifest=str(catalog/"instance_norm_2_functional_query/manifest.json"),
                               trust_local_code=True),options=options,resource_profile="general-tensors-v1")
    worker=multiprocessing.get_context("fork").Process(target=app.prepare_source,args=(task,))
    observed=[]
    worker.start()
    try:
        end=time.monotonic()+8
        while worker.is_alive() and time.monotonic()<end:
            observed=processes(task)
            if any(row["kind"]=="sandbox" for row in observed):
                break
            time.sleep(.01)
        require(observed and worker.is_alive(),"Actual capture process was not observed")
        import resource
        supervisor = next(row["pid"] for row in observed if row["kind"]=="intake_supervisor")
        resource_limits = {name: list(resource.prlimit(supervisor, identifier)) for name,identifier in (
            ("core",resource.RLIMIT_CORE),("address_space",resource.RLIMIT_AS),
            ("cpu",resource.RLIMIT_CPU),("file_size",resource.RLIMIT_FSIZE),
            ("open_files",resource.RLIMIT_NOFILE))}
        require(resource_limits==dict(core=[0,0],address_space=[4*1024**3]*2,
            cpu=[150,155],file_size=[16*1024**2]*2,open_files=[128,128]),
            "Actual intake supervisor resource limits")
        cancelled=app.cancel(task)
        worker.join(4)
        require(not worker.is_alive(),"Cancelled source worker did not terminate")
        cancelled=app.status(task)
        require(cancelled["status"]=="cancelled","Actual source cancellation status")
        end=time.monotonic()+3
        remaining=processes(task)
        while remaining and time.monotonic()<end:
            time.sleep(.05);remaining=processes(task)
        require(not remaining,"Orphaned source capture process")
        cancel=dict(status="cancelled",observed_processes=observed,resource_limits=resource_limits,
                    orphaned_processes=remaining,encrypted_execution=False,paid_calls=0)
    finally:
        if worker.is_alive():
            worker.terminate();worker.join(3)
    frozen(proposal)
    report=dict(format="poseidon-model-source-prototype-acceptance-v1",implementation=retained,
                rows=results,source_kinds=3,source_variants=len(rows),
                passed=sum(r["status"]=="queued" for r in results),
                failed=sum(r["status"]!="queued" for r in results),
                reference_groups=total_groups,snapshot_isolation="passed",
                actual_capture_cancellation=cancel,
                backend_identity=app.backend.identity_manifest(),
                production_integrated=False,pending_proposal_binding=proposal["binding"],
                pending_freeze_unchanged=True,compiler_calls=0,
                encrypted_execution=False,agent_generated=False,paid_calls=0)
    report["binding"]=digest(report)
    dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","source_kinds","source_variants",
            "passed","failed","reference_groups","production_integrated","compiler_calls","paid_calls")}))
    return 0 if not report["failed"] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    return run(p.parse_args().output)


if __name__=="__main__":
    raise SystemExit(main())
