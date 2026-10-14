"""Context-bound application intake for V11; stops before generation and backend execution."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from benchmark_graph import canonical,digest,require
from hecate_python_env import ROOT,VENV,WORK
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_model_source import FORMAT,own_identity
from model_source_provenance import verify
from pytorch_model_intake import source_identity
from candidate_bundle import check_provenance
from strict_json_files import dump

CASES=("residual_functional","dropout_spatial","attention_fused","projected_attention_none_0",
       "projected_attention_boolean_0","projected_attention_causal_1","projected_attention_additive_2","public_alpha_none_0")

def run(output,captures):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure environment")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh application evidence")
    frozen=source_identity();output.mkdir(mode=0o700);context=WorkerContext.capture(output/"tasks")
    context.persist();app=application(context);controller=own_identity();identities=app.backend.identity_manifest()
    rows=[];start=time.monotonic()
    for name in CASES:
        require(time.monotonic()-start<180 and source_identity()==frozen,"Application intake freeze/budget")
        source=dict(format=FORMAT,kind="trusted_pytorch",manifest=str(ROOT/"scripts/baseline/cases/trusted-pytorch-common-v11"/name/"manifest.json"),trust_local_code=True)
        task=app.submit_source(source,validation_level="numerical",resource_profile="general-tensors-v1",options={})
        initial=app.status(task);require(initial["phase"]=="intake_queued" and initial["status"]=="queued","Snapshot admission")
        state=app.prepare_source(task)
        row=dict(id=name,task_id=task,status=state["status"],phase=state["phase"],failure=state["failure"],
                 compiler_calls=0,encrypted_execution=False,agent_generated=False,paid_calls=0)
        if state["status"]=="queued":
            meta=app._checked_task_metadata(task);provenance=meta["model_provenance"]
            check_provenance(provenance,meta["request"]["model"])
            origin=verify(meta["model_source"],meta["request"],provenance,controller_sha256=controller["controller_sha256"])
            expected=json.loads((captures/name/"model.json").read_text())
            require(canonical(expected)==canonical(meta["request"]["model"]),"Public CLI and application graph differ")
            require(state["generations"]==state["http_attempts"]==0 and state["acceptance"] is None and state["package"] is None,
                    "Intake must not claim generation/validation")
            require("model_source" not in meta["request"],"Source origin leaked into generation request")
            row.update(accepted=True,model_sha256=digest(expected),request_id=meta["request"]["request_id"],
                       origin_binding=origin["binding_sha256"] if "binding_sha256" in origin else digest(origin),
                       reference_groups=provenance["reference_report"]["probes"],
                       remaining_wall_seconds=meta["budget"]["wall_seconds"])
            require(app.cancel(task)["status"]=="cancelled","Cancel queued conversion")
        else:row.update(accepted=False,reference_groups=0)
        rows.append(row);dump(output/"progress.json",dict(rows=rows));print(json.dumps(row),flush=True)
    require(source_identity()==frozen and own_identity()==controller and app.backend.identity_manifest()==identities,
            "Application source/backend identity drift")
    report=dict(format="poseidon-application-common-pytorch-v11-audit-v1",rows=rows,
                passed=sum(r["accepted"] for r in rows),failed=sum(not r["accepted"] for r in rows),skipped=0,
                reference_groups=sum(r["reference_groups"] for r in rows),source_identity=frozen,
                controller_identity=controller,backend_identity=identities,context_binding=context.binding,
                seconds=time.monotonic()-start,compiler_calls=0,encrypted_execution=False,paid_calls=0,
                scope="production context-bound application source preparation; stopped before generation and backend validation")
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","passed","failed","reference_groups","seconds")}))
    return 0 if not report["failed"] else 1
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--captures",required=True,type=Path);a=p.parse_args();raise SystemExit(run(a.output,a.captures))
