"""Production application terminal-state checks for public source-spelling probes."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from benchmark_graph import canonical,digest,require
from hecate_python_env import ROOT,VENV,WORK
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_model_source import FORMAT,own_identity
from pytorch_model_intake import source_identity
from strict_json_files import dump

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def file_stats(directory):
    return {str(p.relative_to(directory)):(p.stat().st_size,p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}
def run(output,cli_report):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure environment")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh application evidence")
    reference=json.loads(cli_report.read_text());require(reference["cases"]==6,"Frozen source probe count")
    fixture=ROOT/"scripts/baseline/benchmarks/tools/fixtures/pytorch-source-spellings-v1"
    index=json.loads((fixture/"index.json").read_text())
    require(index["binding"]==reference["fixture_binding"],"Public probe identity")
    for row in index["rows"]:
        for rel,d in row["files"].items():require(sha(ROOT/rel)==d,"Public source changed")
    frozen=source_identity();tool_sha=sha(Path(__file__))
    output.mkdir(mode=0o700);context=WorkerContext.capture(output/"tasks")
    context.persist();app=application(context)
    controller=own_identity();backend=app.backend.identity_manifest();started=time.monotonic();rows=[]
    for item in reference["rows"]:
        require(time.monotonic()-started<180 and source_identity()==frozen,"Application freeze/time budget")
        source=dict(format=FORMAT,kind="trusted_pytorch",manifest=str(fixture/item["id"]/"manifest.json"),trust_local_code=True)
        task=app.submit_source(source,validation_level="numerical",resource_profile="general-tensors-v1",options={})
        admission=app.status(task)
        require(admission["phase"]=="intake_queued" and admission["status"]=="queued","Actual source admission")
        state=app.prepare_source(task)
        require(state["generations"]==state["http_attempts"]==0 and state["acceptance"] is None and state["package"] is None,
                "Source checks must not generate or claim encrypted acceptance")
        row=dict(id=item["id"],task_id=task,status=state["status"],phase=state["phase"],failure=state["failure"],
            accepted=item["status"]=="input_accepted",reference_groups=0,paid_calls=0,compiler_calls=0,encrypted_execution=False)
        if item["status"]=="input_gap":
            require(state["status"]=="validation_failed" and state["phase"]=="complete" and state["failure"]["layer"]=="input"
                and state["failure"]["code"]=="model_source_intake_rejected" and state["failure"]["retry_policy"]=="never_automatic",
                "Unsupported source must return actual input terminal, not environment failure or generation")
            require(state["failure"]["context"]=={} and state["failure"]["location"] is None,
                    "Record current redacted source-feedback scope")
            folder=app._dir(task)
            require(folder.is_dir(),"Actual task artifact directory")
            before=file_stats(folder)
            repeat=app.prepare_source(task)
            require(canonical(repeat)==canonical(state) and file_stats(folder)==before,"Terminal repeated prepare must not rerun or modify artifacts")
            row.update(expected_input_terminal=True,terminal_repeat_preserved=True,feedback_precise_source_code=False)
        else:
            require(state["status"]=="queued" and state["phase"]=="queued","Known valid source preparation")
            meta=app._checked_task_metadata(task);model=meta["request"]["model"]
            require(digest(model)==item["model_sha256"],"Application and public CLI graph identity")
            row["reference_groups"]=meta["model_provenance"]["reference_report"]["probes"]
            require(row["reference_groups"]==16,"Actual reference scope")
            require(app.cancel(task)["status"]=="cancelled","Stop before generation")
            row.update(expected_input_terminal=None,terminal_repeat_preserved=None,stopped_before_generation=True)
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status","phase")}),flush=True)
    require(source_identity()==frozen and own_identity()==controller and app.backend.identity_manifest()==backend and sha(Path(__file__))==tool_sha,"Application identity changed")
    report=dict(format="poseidon-application-source-spelling-terminals-v1",rows=rows,source_identity=frozen,
        controller_identity=controller,backend_identity=backend,context_binding=context.binding,cli_binding=reference["binding"],
        fixture_binding=index["binding"],tool_sha256=tool_sha,cases=6,input_accepted=1,expected_input_terminals=5,
        terminal_repeat_preserved=5,skipped=0,reference_groups=16,reference_groups_overlap_cli=True,
        source_feedback_current_code="model_source_intake_rejected",source_feedback_location=None,
        precise_source_feedback_pending=True,compiler_calls=0,encrypted_execution=False,paid_calls=0,http_attempts=0,
        new_hecate_tracing=0,agent_generation=0,seconds=time.monotonic()-started,
        scope="production application source preparation and repeated terminal-state handling; no generation/backend")
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","cases","expected_input_terminals","terminal_repeat_preserved","reference_groups","seconds")}))
    return 0
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--cli-report",required=True,type=Path);a=p.parse_args();raise SystemExit(run(a.output,a.cli_report))
