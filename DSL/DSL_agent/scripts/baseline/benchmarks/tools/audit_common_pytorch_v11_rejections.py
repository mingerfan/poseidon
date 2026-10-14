"""Real V11 isolated-source rejection checks; not positive models or FHE tests."""
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,digest,require
from hecate_python_env import ROOT,VENV,WORK
from pytorch_model_intake import COMMON_FORMAT,source_identity
from strict_json_files import dump

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def freeze_cases():
    old=ROOT/"scripts/baseline/benchmarks/tools/fixtures/common-pytorch-rejections-v1"
    output=ROOT/"scripts/baseline/benchmarks/tools/fixtures/common-pytorch-rejections-v11"
    if output.exists():
        index=json.loads((output/"index.json").read_text())
        require(index["binding"]==digest({k:v for k,v in index.items() if k!="binding"}) and
                all(sha(p)==h for p,h in index["files"].items()),"Frozen negative fixture identity")
        return index
    output.mkdir()
    entries=[]
    index=json.loads((old/"index.json").read_text())
    for row in index["rows"]:
        if row["id"]=="causal_sdpa":continue
        folder=output/row["id"];folder.mkdir()
        spec=json.loads(Path(row["manifest"]).read_text());spec["format"]=COMMON_FORMAT
        for name,h in spec["files"].items():
            p=Path(row["manifest"]).parent/name;require(sha(p)==h,"Original negative source identity")
            (folder/name).write_bytes(p.read_bytes())
        dump(folder/"manifest.json",spec)
        reasons=[row["expected_reason"]]
        if row["id"]=="dynamic_alpha":reasons=["alpha' must be Number","Finite public scalar"]
        entries.append(dict(id=row["id"],manifest=str(folder/"manifest.json"),expected=reasons))
    for name in ("encrypted_mask","fully_masked_row","integer_mask","mask_broadcast","nonfinite_additive"):
        folder=output/name;folder.mkdir()
        init=""
        inputs=[dict(name=n,shape=[1,1,2,2]) for n in ("q","k","v")];weights={}
        if name=="encrypted_mask":
            inputs.append(dict(name="mask",shape=[2,2]));args="q,k,v,mask"
        else:
            masks={"fully_masked_row":np.asarray([[False,False],[True,False]],dtype=bool),
                   "integer_mask":np.ones((2,2),dtype="<i8"),
                   "mask_broadcast":np.ones((3,2),dtype="<f8"),
                   "nonfinite_additive":np.asarray([[0.,-np.inf],[0.,0.]],dtype="<f8")}
            value=masks[name];p=folder/"mask.npy";np.save(p,value,allow_pickle=False)
            weights["mask"]=dict(file="mask.npy",bytes=p.stat().st_size,shape=list(value.shape),dtype=value.dtype.str,sha256=sha(p))
            dtype="torch.bool" if value.dtype==bool else "torch.int64" if value.dtype.kind=="i" else "torch.float64"
            init="    def __init__(self):\n        super().__init__()\n        self.register_buffer('mask',torch.empty("+repr(list(value.shape))+",dtype="+dtype+"))\n"
            args="q,k,v,self.mask"
        signature="q,k,v,mask" if name=="encrypted_mask" else "q,k,v"
        source="import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n"+init+            "    def forward(self,"+signature+"):\n        return F.scaled_dot_product_attention("+args+",dropout_p=0.)\n"+            "def make_model():\n    return Model().double().eval()\n"
        (folder/"models.py").write_text(source)
        spec=dict(format=COMMON_FORMAT,entry="models.make_model",factory=dict(args=[],kwargs={}),
            files={"models.py":sha(folder/"models.py")},inputs=inputs,outputs=["result"],weights=weights,
            dtype_policy="strict-float64-v1",approximations=dict(format="poseidon-pytorch-approximations-v1",
                defaults=dict(exp=dict(method="taylor",domain=[-8.,8.],degree=2),
                    reciprocal=dict(method="goldschmidt",domain=[.5,100.],iterations=1)),modules=[],nodes=[]))
        dump(folder/"manifest.json",spec)
        reasons={"encrypted_mask":["public state"],"fully_masked_row":["fully masked"],
                 "integer_mask":["mask dtype"],"mask_broadcast":["Shape mismatch","must match","mask broadcast"],
                 "nonfinite_additive":["State floating values","finite","Finite"]}[name]
        entries.append(dict(id=name,manifest=str(folder/"manifest.json"),expected=reasons))
    index=dict(format="poseidon-common-pytorch-v11-negatives",rows=entries,positive_benchmark_count=0,
        causal_sdpa="covered by positive v11 cases; historical v10 rejection preserved",
        files={str(p):sha(p) for p in output.rglob("*") if p.is_file()})
    index["binding"]=digest(index);dump(output/"index.json",index);return index
def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned pure rejection environment")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh negative evidence")
    index=freeze_cases();identity=source_identity();output.mkdir(mode=0o700);rows=[];started=time.monotonic()
    for row in index["rows"]:
        require(time.monotonic()-started<90 and source_identity()==identity,"Rejection freeze/time boundary")
        capture=output/row["id"];log=output/(row["id"]+".cli.log")
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
            "--trusted-pytorch-manifest",row["manifest"],"--trust-local-code","--resource-profile",
            "general-tensors-v1","--write","--inside","--output",str(capture)]
        with log.open("x") as stream:
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise RuntimeError("Rejection timed out")
        path=capture/"capture.log";text=(path if path.exists() else log).read_text()
        terminal=text.splitlines()[-1] if text.splitlines() else ""
        ok=code!=0 and any(reason in terminal for reason in ({"mask_broadcast":["Shape mismatch"],
            "nonfinite_additive":["State floating values"]}.get(row["id"],row["expected"])))
        result=dict(id=row["id"],status="rejected_as_expected" if ok else "failed",exit_code=code,
                    diagnostic=terminal,log_sha256=sha(path if path.exists() else log),
                    compiler_calls=0,encrypted_execution=False,paid_calls=0)
        rows.append(result);dump(output/"progress.json",dict(rows=rows));print(json.dumps(result),flush=True)
    require(source_identity()==identity and all(sha(p)==h for p,h in index["files"].items()),"Rejection identity drift")
    report=dict(format="poseidon-common-pytorch-v11-rejections-v1",rows=rows,
                source_identity=identity,fixture_binding=index["binding"],
                passed=sum(r["status"]=="rejected_as_expected" for r in rows),
                failed=sum(r["status"]=="failed" for r in rows),skipped=0,positive_model_count=0,
                seconds=time.monotonic()-started,compiler_calls=0,encrypted_execution=False,paid_calls=0)
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","passed","failed","seconds")}))
    return 0 if not report["failed"] else 1
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path)
    raise SystemExit(run(p.parse_args().output))
