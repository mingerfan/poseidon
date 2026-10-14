"""Audit public PyTorch source-spelling gaps without modifying active intake code.

This is a development-only capture audit, never an application request dependency.
It does not generate DSL, compile, execute ciphertexts, or call a provider.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def write_new(path,value):
    with path.open("x") as stream:stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n")

ROWS=[
dict(id="multihead_attention",shape=[1,2,4],group="transformer",target="nn.MultiheadAttention",profile="attention",hypothesis="registered module adapter absent"),
dict(id="functional_max_pool2d",shape=[1,1,4,4],group="cnn",target="F.max_pool2d",profile="abs",hypothesis="functional target mapping absent"),
dict(id="gelu_tanh",shape=[1,4],group="activation",target='nn.GELU(approximate="tanh")',profile="gelu",hypothesis="original-function variant rejected"),
dict(id="relu_inplace",shape=[1,4],group="activation_alias",target="nn.ReLU(inplace=True)",profile="relu",hypothesis="alias/mutation contract rejects"),
dict(id="contiguous_view",shape=[1,2,3],group="shape",target="Tensor.contiguous().view",profile=None,hypothesis="storage-identity mapping absent"),
dict(id="public_division",shape=[1,4],group="arithmetic",target="torch.div(x,2.0)",profile=None,hypothesis="public-scalar division mapping absent")
]

def run(root,fixture_dir,output,seconds):
    import numpy as np,torch
    from hecate_python_env import ROOT,VENV,WORK
    from platform_config import require_python_packages
    from pytorch_model_intake import source_identity,COMMON_FORMAT
    from candidate_bundle import load_model_provenance
    if ROOT!=root or os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:
        raise RuntimeError("Pinned pure Python environment required")
    require_python_packages(torch,np)
    if not fixture_dir.is_relative_to(ROOT/"scripts/baseline/benchmarks/tools/fixtures") or output.exists() or not output.parent.resolve().is_relative_to(WORK/"results"):
        raise RuntimeError("Fresh project fixture and result paths required")
    source=fixture_dir/"models.py"
    if not source.is_file() or source.is_symlink():raise RuntimeError("Frozen public source required")
    sys.path.insert(0,str(fixture_dir))
    from models import make_model
    frozen=source_identity();tool_digest=sha(__file__);rows=[];manifest_rows=[]
    for i,item in enumerate(ROWS):
        case_dir=fixture_dir/item["id"]
        case_dir.mkdir(mode=0o700)
        target=case_dir/"models.py"
        with target.open("x") as f:f.write(source.read_text())
        torch.manual_seed(901+i)
        model=make_model(item["id"]);state={}
        rng=np.random.default_rng(9041+i)
        for j,(name,tensor) in enumerate(model.state_dict().items()):
            value=rng.uniform(-.125,.125,size=tuple(tensor.shape)).astype(np.float64)
            file=case_dir/("state_%04d.npy"%j)
            with file.open("xb") as f:np.save(f,value,allow_pickle=False)
            state[name]=dict(file=file.name,shape=list(value.shape),dtype=value.dtype.str,bytes=file.stat().st_size,sha256=sha(file))
        approximation={"format":"poseidon-pytorch-approximations-v1","defaults":{},"modules":[],"nodes":[]}
        if item["profile"]=="attention":
            approximation["defaults"]={"exp":dict(method="taylor",degree=2,domain=[-8.,8.]),"reciprocal":dict(method="goldschmidt",iterations=1,domain=[.5,100.])}
        elif item["profile"] is not None:
            approximation["defaults"][item["profile"]]=dict(method="polynomial",basis="power",domain=[-2.,2.],coefficients=[0.,.5,.25] if item["profile"]!="abs" else [0.,0.,1.])
        manifest=dict(format=COMMON_FORMAT,entry="models.make_model",files={"models.py":sha(target)},
            factory=dict(args=[],kwargs=dict(case=item["id"])),dtype_policy="strict-float64-v1",
            inputs=[dict(name="x",shape=item["shape"])],outputs=["result"],weights=state,approximations=approximation)
        write_new(case_dir/"manifest.json",manifest)
        manifest_rows.append(dict(**item,manifest=str(case_dir/"manifest.json"),
            files={str(p.relative_to(ROOT)):sha(p) for p in case_dir.iterdir() if p.is_file()}))
    index=dict(format="poseidon-source-spelling-gap-fixtures-v1",source_version=COMMON_FORMAT,rows=manifest_rows,
        frozen_benchmark_modified=False,new_mathematical_model_count_claimed=False,compiler_calls=0,paid_calls=0)
    index["binding"]=digest(index);write_new(fixture_dir/"index.json",index)
    output.mkdir(mode=0o700);(output/"captures").mkdir()
    runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,hip=torch.version.hip)
    write_new(output/"inputs.json",dict(fixture_binding=index["binding"],source_identity=frozen,tool_sha256=tool_digest,runtime=runtime))
    started=time.monotonic()
    for item in manifest_rows:
        if source_identity()!=frozen or sha(__file__)!=tool_digest:raise RuntimeError("Capture source changed")
        remaining=seconds-(time.monotonic()-started)
        if remaining<=0:raise RuntimeError("Finite plaintext audit budget exhausted")
        directory=output/"captures"/item["id"];log=output/(item["id"]+".cli.log")
        command=[str(VENV/"bin/python"),"-B",str(ROOT/"scripts/prepare_model.py"),
            "--trusted-pytorch-manifest",item["manifest"],"--trust-local-code",
            "--resource-profile","general-tensors-v1","--write","--inside","--output",str(directory)]
        tick=time.monotonic()
        with log.open("x") as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(30,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                write_new(output/"timeout.json",dict(id=item["id"],status="timeout",compiler_calls=0,paid_calls=0))
                raise RuntimeError("Source capture timed out; retained evidence")
        row=dict(id=item["id"],target=item["target"],group=item["group"],hypothesis=item["hypothesis"],
            exit_code=code,seconds=time.monotonic()-tick,compiler_calls=0,encrypted_execution=False,paid_calls=0,
            cli_log_sha256=sha(log),reference_groups=0)
        if code==0:
            model=json.loads((directory/"model.json").read_text())
            provenance=load_model_provenance(directory,model,max_bytes=4*1024**2)
            reference=provenance["reference_report"]
            if reference["reference_status"]!="passed" or reference["probes"]!=16 or reference["common_semantics_preserved"] is not True:
                raise RuntimeError("Successful intake lacks actual finite reference evidence")
            row.update(status="input_accepted",model_sha256=digest(model),reference_groups=16,
                max_decomposition_error=reference["max_decomposition_error"],
                max_original_module_error=reference["max_original_module_error"],
                max_normalization_error=reference["max_arithmetic_normalization_error"],
                original_function_accuracy_certified=False)
        else:
            capture=directory/"capture.log"
            text=capture.read_text() if capture.exists() else log.read_text()
            errors=[line.strip() for line in text.splitlines() if line.startswith(("ValueError:","TypeError:","RuntimeError:","NotImplementedError:","torch.fx.proxy.TraceError:"))]
            row.update(status="input_gap",failure_layer="model_intake",diagnostic=errors[-1] if errors else text[-2000:],
                evidence_log_sha256=sha(capture) if capture.exists() else sha(log))
        rows.append(row)
        (output/"progress.json").write_text(json.dumps(dict(rows=rows),ensure_ascii=False,sort_keys=True,indent=2)+"\n")
        print(json.dumps({k:row[k] for k in ("id","status","seconds")},ensure_ascii=False),flush=True)
    for row in manifest_rows:
        for rel,expected in row["files"].items():
            if sha(ROOT/rel)!=expected:raise RuntimeError("Frozen fixture changed")
    if source_identity()!=frozen:raise RuntimeError("Production intake changed")
    report=dict(format="poseidon-source-spelling-gap-audit-v1",runtime=runtime,source_identity=frozen,
        fixture_binding=index["binding"],audit_tool_sha256=tool_digest,cases=len(rows),
        input_accepted=sum(r["status"]=="input_accepted" for r in rows),
        input_gaps=sum(r["status"]=="input_gap" for r in rows),skipped=0,
        reference_groups=sum(r["reference_groups"] for r in rows),rows=rows,seconds=time.monotonic()-started,
        compiler_calls=0,new_hecate_tracing=0,encrypted_execution=False,paid_calls=0,http_attempts=0,
        backend_changes=0,parameters_changed=False,
        scope="actual isolated public source intake; unsupported source spelling does not establish DSL/backend inability",
        frozen_benchmark_modified=False,new_mathematical_model_count_claimed=False)
    report["binding"]=digest(report);write_new(output/"report.json",report)
    print(json.dumps({k:report[k] for k in ("binding","cases","input_accepted","input_gaps","reference_groups","seconds")},ensure_ascii=False))
    return 0
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture-dir",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seconds",type=int,default=180)
    a=p.parse_args()
    if not 1<=a.seconds<=300:raise ValueError("Finite plaintext-only audit limit")
    from hecate_python_env import ROOT
    raise SystemExit(run(ROOT,a.fixture_dir.resolve(),a.output.resolve(),a.seconds))
