"""Read-only dev audit of public-first arithmetic in isolated trusted PyTorch capture."""
import ast,hashlib,io,json,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from benchmarks.tools.audit_operator_graph_versions import preservation,runtime_identity

MODEL_SOURCE = '''"""Owned deterministic arithmetic models for isolated intake testing."""
import torch
from torch import nn

class Arithmetic(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind == "tensor":
            self.register_buffer("public", torch.zeros((2,2),dtype=torch.float64))
        else:
            self.public = 0.375

class AddLeft(Arithmetic):
    def forward(self, x):
        return self.public + x
class AddRight(Arithmetic):
    def forward(self, x):
        return x + self.public
class SubtractLeft(Arithmetic):
    def forward(self, x):
        return self.public - x
class SubtractRight(Arithmetic):
    def forward(self, x):
        return x - self.public
class MultiplyLeft(Arithmetic):
    def forward(self, x):
        return self.public * x
class MultiplyRight(Arithmetic):
    def forward(self, x):
        return x * self.public

def make_model(op, kind, side):
    classes = {
        ("add","left"): AddLeft, ("add","right"): AddRight,
        ("subtract","left"): SubtractLeft, ("subtract","right"): SubtractRight,
        ("multiply","left"): MultiplyLeft, ("multiply","right"): MultiplyRight,
    }
    return classes[(op,side)](kind).eval().double()
'''
FIXTURES=Path(__file__).parent/"fixtures/public-operand-role-v1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def generate(directory):
    """Source fixtures are separate from the frozen benchmark and from captured evidence."""
    import numpy as np
    directory=Path(directory);require(not directory.exists(),"Fresh role fixtures")
    ast.parse(MODEL_SOURCE)
    directory.mkdir(parents=True)
    cases=[]
    for op in ("add","subtract","multiply"):
        for kind in ("scalar","tensor"):
            for side in ("left","right"):
                name="role_"+op+"_"+kind+"_"+side
                folder=directory/name;folder.mkdir()
                (folder/"models.py").write_text(MODEL_SOURCE)
                weights={}
                if kind=="tensor":
                    public=np.asarray([[.125,-.25],[.375,-.5]],dtype=np.float64)
                    stream=io.BytesIO();np.save(stream,public,allow_pickle=False)
                    raw=stream.getvalue();(folder/"state_0000.npy").write_bytes(raw)
                    weights["public"]=dict(file="state_0000.npy",dtype=public.dtype.str,
                        shape=list(public.shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                spec=dict(format="poseidon-trusted-pytorch-v7",entry="models.make_model",
                    files={"models.py":sha(folder/"models.py")},
                    factory=dict(args=[op,kind,side],kwargs={}),inputs=[dict(name="x",shape=[2,2])],
                    outputs=["result"],weights=weights,dtype_policy="strict-float64-v1",
                    approximations=dict(format="poseidon-pytorch-approximations-v1",
                                        defaults={},modules=[],nodes=[]))
                (folder/"manifest.json").write_bytes(canonical(spec))
                accepted=side=="right" or (kind=="scalar" and op in ("add","multiply"))
                cases.append(dict(id=name,op=op,public_kind=kind,public_side=side,
                                  manifest=name+"/manifest.json",expected_capture="accepted" if accepted else "rejected"))
    catalog=dict(format="poseidon-public-operand-role-source-fixtures-v1",planned=12,cases=cases,
                 frozen_benchmark_changed=False)
    catalog["binding"]=digest(catalog);(directory/"catalog.json").write_bytes(canonical(catalog))
    return catalog

def run(output,proposal,seconds=300):
    from trusted_pytorch_import import run as capture,source_identity
    from semantic_benchmark_execution import runtime_sources
    output=Path(output);require(not output.exists(),"Fresh capture audit")
    require(180<=seconds<=600,"Capture audit budget")
    frozen=preservation(proposal);runtime=runtime_identity();sources=runtime_sources()
    capture_source=source_identity()
    inventory={str(p.relative_to(FIXTURES)):sha(p) for p in FIXTURES.rglob("*") if p.is_file()}
    catalog=json.loads((FIXTURES/"catalog.json").read_text())
    require(digest({k:v for k,v in catalog.items() if k!="binding"})==catalog["binding"] and
            catalog["planned"]==12 and len(catalog["cases"])==12,"Role fixture catalog")
    own=sha(__file__);start=time.monotonic();rows=[]
    output.mkdir(mode=0o700);(output/"captures").mkdir()
    def write():
        report=dict(format="poseidon-public-operand-role-capture-audit-v1",
            planned=12,completed=len(rows),
            isolated_capture_accepted=sum(r["capture_admission"]=="accepted" for r in rows),
            isolated_capture_rejected=sum(r["capture_admission"]=="rejected" for r in rows),
            unexpected_outcomes=sum(not r["expected_outcome"] for r in rows),
            not_run=12-len(rows),rows=rows,
            module_reference_groups=sum(r.get("module_reference_groups",0) for r in rows),
            runtime=runtime,frozen_proposal=frozen,capture_source_identity=capture_source,
            fixture_hashes=inventory,tool_sha256=own,seconds=time.monotonic()-start,
            production_adapter_changed=False,hecate_tracing=False,compiled=False,
            encrypted_execution=False,agent_generated=False,paid_calls=0,
            scope="Isolated input capture and finite module/reference checks; not FHE/Agent coverage")
        report["binding"]=digest(report);(output/"report.json").write_bytes(canonical(report))
        return report
    for case in catalog["cases"]:
        if time.monotonic()-start>seconds-180:break
        require(preservation(proposal)==frozen and source_identity()==capture_source,"Frozen capture changed")
        destination=output/"captures"/case["id"]
        row=dict(id=case["id"],operator=case["op"],public_kind=case["public_kind"],
                 public_side=case["public_side"],expected_capture=case["expected_capture"])
        try:
            result=capture(FIXTURES/case["manifest"],destination,
                           resource_profile="general-tensors-v1",trusted=True)
        except ValueError as error:
            log=(destination/"capture.log").read_text() if (destination/"capture.log").exists() else ""
            # Only the documented operand-role rejection is counted as this intake gap.
            known="First operand must be encrypted" in log
            row.update(capture_admission="rejected",failure_layer="agent_input_decomposition" if known else "capture",
                       reason="First operand must be encrypted" if known else str(error),
                       log_sha256=sha(destination/"capture.log") if log else None,
                       expected_outcome=case["expected_capture"]=="rejected" and known)
        else:
            ref=json.loads((destination/"reference-report.json").read_text())
            require(ref["probes"]==16 and ref["reference_status"]=="passed" and
                    ref["original_module_checked"] is True,"Actual module reference evidence")
            row.update(capture_admission="accepted",expected_outcome=case["expected_capture"]=="accepted",
                       capture_result=result,module_reference_groups=16,
                       max_decomposition_error=ref["max_decomposition_error"],
                       max_original_module_error=ref["max_original_module_error"],
                       capture_report_sha256=sha(destination/"reference-report.json"))
        rows.append(row);write()
        print(json.dumps({k:v for k,v in row.items() if k not in ("capture_result",)},ensure_ascii=False),flush=True)
    require(runtime_sources()==sources and preservation(proposal)==frozen and
            source_identity()==capture_source and runtime_identity()==runtime and own==sha(__file__),
            "Frozen identity changed")
    require(inventory=={str(p.relative_to(FIXTURES)):sha(p) for p in FIXTURES.rglob("*") if p.is_file()},
            "Source fixtures changed")
    report=write()
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ("rows","runtime","capture_source_identity","fixture_hashes")},ensure_ascii=False))
    return 0 if report["unexpected_outcomes"]==0 and report["not_run"]==0 else 1

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--frozen-proposal",type=Path)
    p.add_argument("--seconds",type=int,default=300)
    args=p.parse_args()
    if args.generate:
        print(json.dumps(generate(FIXTURES)))
    else:
        p.error("--output and --frozen-proposal required") if args.output is None or args.frozen_proposal is None else None
        raise SystemExit(run(args.output,args.frozen_proposal,args.seconds))
