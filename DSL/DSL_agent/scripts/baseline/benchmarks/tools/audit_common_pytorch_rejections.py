"""Actual sandbox rejection checks for staged common PyTorch normalization."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from benchmark_graph import canonical,digest,require
from hecate_python_env import ROOT,VENV,WORK
from strict_json_files import dump
from pytorch_model_intake import prepare
from benchmarks.tools.audit_common_pytorch_idioms import bound,frozen
from benchmarks.tools.audit_common_pytorch_prototype import prototype_sources
from benchmarks.tools.audit_operator_shape_capture_prototype import sandbox

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def cases(folder):
    require(not folder.exists(),"Preserve rejection fixtures")
    folder.mkdir()
    single=[dict(name="x",shape=[1,2,2,2])]
    attention=[dict(name=n,shape=[1,1,2,2]) for n in ("q","k","v")]
    values=[
        ("dynamic_alpha","return torch.add(x,x,alpha=x)",single,"","Finite public scalar"),
        ("bool_alpha","return torch.add(x,x,alpha=True)",single,"","Finite public scalar"),
        ("out_mutation","return torch.add(x,x,out=x)",single,"","out mutation"),
        ("training_dropout","return F.dropout(x,p=0.,training=True)",single,"","Only inference Dropout"),
        ("inplace_dropout","return F.dropout(x,training=False,inplace=True)",single,"","In-place Dropout"),
        ("spatial_rank","return F.dropout1d(x,training=False)",single,"","Spatial Dropout input rank"),
        ("causal_sdpa","return F.scaled_dot_product_attention(q,k,v,is_causal=True)",attention,"","explicit dense attention only"),
        ("random_sdpa","return F.scaled_dot_product_attention(q,k,v,dropout_p=.25)",attention,"","SDPA dropout must be zero"),
        ("unknown_call","return torch.flip(x,(0,))",single,"","Unsupported PyTorch call"),
        ("encrypted_branch","if x.sum()>0:\n            return x\n        return -x",single,"","symbolically traced variables cannot be used as inputs to control flow"),
    ]
    rows=[]
    for name,body,inputs,init,reason in values:
        case=folder/name;case.mkdir()
        args=",".join(x["name"] for x in inputs)
        source=("import torch\nfrom torch.nn import functional as F\nclass Model(torch.nn.Module):\n"
                "    def forward(self,"+args+"):\n        "+body+"\n"
                "def make_model():\n    return Model().double().eval()\n")
        (case/"models.py").write_text(source)
        spec=dict(format="poseidon-trusted-pytorch-v10",entry="models.make_model",
                  factory=dict(args=[],kwargs={}),files={"models.py":sha(case/"models.py")},
                  inputs=inputs,outputs=["result"],weights={},dtype_policy="strict-float64-v1",
                  approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]))
        dump(case/"manifest.json",spec)
        prepare(case/"manifest.json")
        rows.append(dict(id=name,manifest=str(case/"manifest.json"),expected_reason=reason,
                         files={str(p.relative_to(ROOT)):sha(p) for p in case.iterdir()}))
    index=dict(format="poseidon-common-pytorch-rejection-fixtures-v1",rows=rows,positive_benchmark_count=0)
    index["binding"]=digest(index);dump(folder/"index.json",index);return index

def run(output):
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned existing shell")
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK/"results"),"Fresh negative evidence")
    prototype=ROOT/"scripts/baseline/benchmarks/tools/fixtures/common-pytorch-capture-prototype-v3"
    value,current,expected=prototype_sources(prototype)
    proposal=bound(json.loads((ROOT/"docs/baseline/agent-application-source-offline-proposal-r1025.json").read_text()))
    frozen(proposal)
    index=cases(ROOT/"scripts/baseline/benchmarks/tools/fixtures/common-pytorch-rejections-v1")
    output.mkdir();rows=[];tick=time.monotonic()
    for item in index["rows"]:
        require(time.monotonic()-tick<90,"Actual rejection time boundary")
        spec,paths=prepare(Path(item["manifest"]));target=output/item["id"]
        code=sandbox(spec,paths,{},target,expected,prototype,20)
        log=(target/"capture.log").read_text()
        terminal=log.splitlines()[-1] if log.splitlines() else ""
        match=code!=0 and item["expected_reason"] in terminal
        rows.append(dict(id=item["id"],status="rejected_as_expected" if match else "failed",
                         exit_code=code,diagnostic=terminal,log_sha256=sha(target/"capture.log"),
                         encrypted_execution=False,compiler_calls=0,paid_calls=0))
        dump(output/"progress.json",dict(rows=rows))
    require(prototype_sources(prototype)[2]==expected,"Staged rejection source identity")
    frozen(proposal)
    report=dict(format="poseidon-common-pytorch-sandbox-rejections-v1",prototype_binding=value["binding"],
                proposal_binding=proposal["binding"],fixture_binding=index["binding"],rows=rows,
                planned=len(rows),passed=sum(r["status"]=="rejected_as_expected" for r in rows),
                failed=sum(r["status"]=="failed" for r in rows),skipped=0,
                production_integrated=False,production_freeze_preserved=True,seconds=time.monotonic()-tick,
                tool_sha256=sha(Path(__file__)),compiler_calls=0,encrypted_execution=False,paid_calls=0)
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","planned","passed","failed","skipped")}))
    return 0 if report["failed"]==0 else 1

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    raise SystemExit(run(p.parse_args().output))
