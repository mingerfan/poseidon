"""Audit ordinary PyTorch idioms through the public isolated intake only."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from benchmark_graph import canonical,digest,require
from hecate_python_env import ROOT,VENV,WORK
from strict_json_files import dump

PROPOSAL="docs/baseline/agent-application-source-offline-proposal-r1025.json"
CORPUS="scripts/baseline/cases/trusted-pytorch-common-idioms-v1/index.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bound(value):
    payload={k:v for k,v in value.items() if k!="binding"}
    require(value.get("binding")==digest(payload),"Audit input binding")
    return value

def frozen(proposal):
    counts={}
    for section in ("source_hashes","extra_source_hashes","case_hashes","tool_hashes"):
        rows=proposal[section]
        for name,expected in rows.items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            require(path.is_file() and sha(path)==expected,"Frozen production input changed: "+name)
        counts[section]=len(rows)
    return counts

def corpus_frozen(index):
    for row in index["rows"]:
        for name,expected in row["files"].items():
            p=ROOT/name
            require(p.is_relative_to(ROOT) and not p.is_symlink() and sha(p)==expected,
                    "Common source fixture changed")
    return sum(len(row["files"]) for row in index["rows"])

def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,
            "Pinned pure environment required")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),
            "Fresh audit evidence directory required")
    import numpy as np
    import torch
    from platform_config import require_python_packages
    from candidate_bundle import load_model_provenance
    from pytorch_model_intake import source_identity
    require_python_packages(torch,np)
    proposal=bound(json.loads((ROOT/PROPOSAL).read_text()))
    index=bound(json.loads((ROOT/CORPUS).read_text()))
    counts=frozen(proposal);fixture_count=corpus_frozen(index)
    identity=source_identity()
    output.mkdir(mode=0o700)
    (output/"captures").mkdir()
    rows=[];started=time.monotonic()
    runtime=dict(python=sys.version,executable=str(Path(sys.executable).resolve()),
                 python_sha256=sha(Path(sys.executable).resolve()),numpy=np.__version__,
                 numpy_module_sha256=sha(Path(np.__file__)),torch=torch.__version__,
                 torch_module_sha256=sha(Path(torch.__file__)),cuda=torch.version.cuda,hip=torch.version.hip)
    tool_sha=sha(Path(__file__))
    dump(output/"inputs.json",dict(proposal_binding=proposal["binding"],corpus=index,
                                 capture_identity=identity,runtime=runtime,tool_sha256=tool_sha))
    for item in index["rows"]:
        require(time.monotonic()-started<240,"Plaintext audit time boundary")
        require(source_identity()==identity,"Capture implementation changed")
        target=output/"captures"/item["id"];log=output/(item["id"]+".cli.log")
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
                 "--trusted-pytorch-manifest",item["manifest"],"--trust-local-code",
                 "--resource-profile","general-tensors-v1","--write","--inside","--output",str(target)]
        tick=time.monotonic()
        with log.open("x") as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code=process.wait(timeout=min(30,max(1,240-(tick-started))))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
                dump(output/"timeout.json",dict(id=item["id"],status="environment_or_budget_failure"))
                raise RuntimeError("Isolated source audit timed out; no dependent work started")
        row=dict(id=item["id"],family=item["family"],spelling=item["spelling"],
                 semantic_peer_group=item["semantic_peer_group"],exit_code=code,
                 seconds=time.monotonic()-tick,compiler_calls=0,encrypted_execution=False,
                 agent_generated=False,paid_calls=0,cli_log_sha256=sha(log))
        if code==0:
            model=json.loads((target/"model.json").read_text())
            provenance=load_model_provenance(target,model,max_bytes=4*1024**2)
            report=provenance["reference_report"]
            require(report["probes"]==16 and report["reference_status"]=="passed" and
                    report["original_module_checked"] is True and
                    report["normalized_module_checked"] is True,"Independent source reference")
            row.update(status="passed",reference_groups=16,model_sha256=digest(model),
                       original_sha256=digest(provenance["original"]),
                       reference_sha256=sha(target/"reference-report.json"),
                       max_decomposition_error=report["max_decomposition_error"],
                       max_original_module_error=report["max_original_module_error"],
                       max_normalization_error=report["max_arithmetic_normalization_error"],
                       max_approximation_error=report["max_original_function_approximation_error"],
                       artifacts={str(p.relative_to(output)):sha(p) for p in target.rglob("*") if p.is_file()})
        else:
            capture_log=target/"capture.log"
            text=capture_log.read_text() if capture_log.is_file() else log.read_text()
            tail=text[-5500:]
            row.update(status="failed",failure_layer="model_intake",
                       diagnostic=tail,capture_log_sha256=sha(capture_log) if capture_log.exists() else None,
                       reference_groups=0)
        rows.append(row)
        dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status","seconds")}),flush=True)
    require(source_identity()==identity and sha(Path(__file__))==tool_sha,"Audit implementation changed")
    frozen(proposal);corpus_frozen(index)
    report=dict(format="poseidon-common-pytorch-intake-audit-v1",corpus_binding=index["binding"],
                proposal_binding=proposal["binding"],source_identity=identity,runtime=runtime,
                audit_tool_sha256=tool_sha,source_contexts=len(rows),semantic_peer_groups=6,
                mathematical_models_not_added_to_benchmark=True,
                passed=sum(r["status"]=="passed" for r in rows),
                failed=sum(r["status"]=="failed" for r in rows),skipped=0,
                reference_groups=sum(r["reference_groups"] for r in rows),rows=rows,
                frozen_counts=counts,fixture_files=fixture_count,production_freeze_preserved=True,
                seconds=time.monotonic()-started,compiler_calls=0,encrypted_execution=False,
                paid_calls=0,backend_changes=0,parameters_changed=False,
                scope="public isolated source intake and finite plaintext reference; no DSL generation or backend acceptance")
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","passed","failed","reference_groups","seconds")}))
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    raise SystemExit(run(parser.parse_args().output))
