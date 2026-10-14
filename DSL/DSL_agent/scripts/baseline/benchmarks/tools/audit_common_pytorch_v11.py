"""Public isolated V11 intake and independent plaintext checks; no DSL/backend run."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from benchmark_graph import digest,require
from hecate_python_env import ROOT,VENV,WORK
from strict_json_files import dump

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(index_path,output,seconds):
    import numpy as np,torch
    from platform_config import require_python_packages
    from pytorch_model_intake import source_identity,COMMON_FORMAT
    from candidate_bundle import load_model_provenance
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure environment")
    require_python_packages(torch,np)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh audit directory")
    index=json.loads(index_path.read_text())
    require(index["binding"]==digest({k:v for k,v in index.items() if k!="binding"}),"V11 corpus binding")
    def verify():
        for row in index["rows"]:
            for name,expected in row["files"].items():
                p=ROOT/name
                require(p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink() and sha(p)==expected,"V11 fixture identity")
    verify();frozen=source_identity();tool_sha=sha(Path(__file__))
    output.mkdir(mode=0o700);(output/"captures").mkdir()
    runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,
        cuda=torch.version.cuda,hip=torch.version.hip)
    dump(output/"inputs.json",dict(corpus_binding=index["binding"],capture_identity=frozen,runtime=runtime,tool_sha256=tool_sha))
    started=time.monotonic();rows=[]
    for item in index["rows"]:
        require(source_identity()==frozen,"Capture implementation changed")
        remaining=seconds-(time.monotonic()-started);require(remaining>0,"Plaintext batch time boundary")
        target=output/"captures"/item["id"];log=output/(item["id"]+".cli.log")
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
            "--trusted-pytorch-manifest",item["manifest"],"--trust-local-code",
            "--resource-profile","general-tensors-v1","--write","--inside","--output",str(target)]
        tick=time.monotonic()
        with log.open("x") as stream:
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(30,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                dump(output/"timeout.json",dict(id=item["id"],status="timed_out",no_dependent_work_started=True))
                raise RuntimeError("Isolated source timeout")
        row=dict(id=item["id"],family=item["family"],context=item["context"],exit_code=code,
                 seconds=time.monotonic()-tick,reference_groups=0,compiler_calls=0,
                 encrypted_execution=False,agent_generated=False,paid_calls=0,cli_log_sha256=sha(log))
        if code==0:
            model=json.loads((target/"model.json").read_text())
            provenance=load_model_provenance(target,model,max_bytes=4*1024**2)
            reference=provenance["reference_report"]
            require(reference["probes"]==16 and reference["reference_status"]=="passed" and
                    reference["common_semantics_preserved"] is True and reference["normalized_module_checked"] is True,
                    "Actual V11 independent finite references")
            require(provenance["trusted_pytorch"]["manifest"]["format"]==COMMON_FORMAT,"Actual V11 package version")
            row.update(status="passed",reference_groups=16,model_sha256=digest(model),
                original_sha256=digest(provenance["original"]),input_signature=model["inputs"],
                output_names=model["outputs"],original_nodes=len(provenance["original"]["nodes"]),
                decomposed_nodes=len(model["nodes"]),
                common_rules=[e["rule"] for e in reference["capture_metadata"]["common_capture"]["entries"]],
                max_decomposition_error=reference["max_decomposition_error"],
                max_original_module_error=reference["max_original_module_error"],
                max_normalization_error=reference["max_arithmetic_normalization_error"],
                max_approximation_error=reference["max_original_function_approximation_error"],
                approximation_accuracy_certified=False,
                artifacts={str(p.relative_to(output)):sha(p) for p in target.rglob("*") if p.is_file()})
        else:
            capture_log=target/"capture.log"
            text=capture_log.read_text() if capture_log.exists() else log.read_text()
            row.update(status="failed",failure_layer="model_intake",diagnostic=text[-4500:],
                capture_log_sha256=sha(capture_log) if capture_log.exists() else None)
        rows.append(row);dump(output/"progress.json",dict(rows=rows))
        print(json.dumps({k:row[k] for k in ("id","status","seconds")}),flush=True)
    verify();require(source_identity()==frozen and sha(Path(__file__))==tool_sha,"Audit code changed")
    report=dict(format="poseidon-common-pytorch-production-v11-audit-v1",corpus_binding=index["binding"],
        source_identity=frozen,runtime=runtime,audit_tool_sha256=tool_sha,source_contexts=len(rows),
        passed=sum(r["status"]=="passed" for r in rows),failed=sum(r["status"]=="failed" for r in rows),
        skipped=0,reference_groups=sum(r["reference_groups"] for r in rows),rows=rows,
        seconds=time.monotonic()-started,compiler_calls=0,encrypted_execution=False,paid_calls=0,
        backend_changes=0,parameters_changed=False,
        scope="public isolated model-source intake and plaintext reference; no Hecate/DSL or encrypted acceptance")
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","passed","failed","reference_groups","seconds")}))
    return 0 if not report["failed"] else 1
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--index",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--seconds",type=int,default=360)
    a=p.parse_args();require(1<=a.seconds<=600,"Bounded plaintext audit time");raise SystemExit(run(a.index,a.output,a.seconds))
