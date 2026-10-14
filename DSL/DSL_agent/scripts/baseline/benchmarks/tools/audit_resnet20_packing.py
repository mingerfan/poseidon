"""Audit public ResNet convolution packing costs without changing admission gates."""
import argparse, hashlib, json, math, resource, sys, time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from benchmark_graph import validate
from unified_capacity_spatial import vectors,bsgs_vectors
from unified_capacity_constants import packing
from unified_request_storage import compact_request

def audit(model):
    checked=validate(model);rows=[]
    conv=next(n for n in model["nodes"] if n["op"]=="conv2d")
    for period in (256,512,1024,2048,4096):
        row=dict(period=period,configured_runtime=period<=256,
            input_chunks=sum(math.ceil(math.prod(s["shape"])/period) for s in model["inputs"]),
            intermediate_chunks=math.ceil(math.prod(checked["shapes"][conv["outputs"][0]])/period))
        try:
            routes=vectors(conv,checked["shapes"],model["constants"],period)
            transformed=bsgs_vectors(routes,period)
            groups={};babies={}
            for (_,src,dst,step),v in routes.items():
                baby=step%min(16,period);giant=step-baby
                groups.setdefault((src,dst,giant),0);groups[src,dst,giant]+=1
                babies.setdefault(src,set()).add(baby)
            rotate_cost=0
            for steps in babies.values():
                prev=0
                for step in sorted(steps):rotate_cost+=(step-prev).bit_count();prev=step
            # Includes products, sums, binary giant rotations and output merges.
            conv_ops=rotate_cost+sum(2*n-1+g.bit_count() for (_,_,g),n in groups.items())
            targets={}
            for src,dst,g in groups:targets[dst]=targets.get(dst,0)+1
            conv_ops+=sum(max(0,n-1) for n in targets.values())
            row.update(routes=len(routes),unique_conv_vectors=len({tuple(v) for v in transformed.values()}),
                conv_cipher_ops_bound=conv_ops)
            packed=packing(model,{"input_slot_period":period})
            row.update(packed_vectors=len(packed["vectors"]),packed_bindings=len(packed["bindings"]),
                excluded_scalars=sorted(packed["excluded_scalars"]))
            wire=compact_request(dict(layout={"input_slot_period":period},public_constants=packed["vectors"],
                constant_origins={"capacity_tensor_packing":packed["descriptor"]}))
            row["public_packing_wire_bytes"]=len(json.dumps(wire,sort_keys=True,separators=(",",":")).encode())
            row["status"]="structural_cost_only"
        except ValueError as exc:row.update(status="structural_budget_rejection",reason=str(exc))
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(90,95))
    raw=a.model.read_bytes();model=json.loads(raw);start=time.monotonic()
    report=dict(format="poseidon-resnet-packing-cost-v1",model_sha256=hashlib.sha256(raw).hexdigest(),
        model_id=model["id"],rows=audit(model),compiled=False,encrypted_execution=False,new_agent_generation=False,
        paid_calls=0,parameters_changed=False,admission_gates_changed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start)
    a.output.parent.mkdir(parents=True,exist_ok=False)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
