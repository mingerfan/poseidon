"""Developer fixtures for unchanged failed models; not an application generator."""
import argparse,json,math,sys,hashlib
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from rmsnorm_expression_probe import plaintext_check
from workspace_paths import WORK

def rotation(name,step,period,lines):
    value=name
    for bit in range(period.bit_length()-1):
        if (step%period)&(1<<bit):
            nxt="r"+str(len(lines));lines.append("    "+nxt+" = "+value+".rotate("+str(1<<bit)+")");value=nxt
    return value

def scalar(request,value):
    return next(k for k,v in request["public_constants"].items() if type(v) in (int,float) and v==value)

def make_source(model,request):
    nodes=model["nodes"];period=request["layout"]["input_slot_period"]
    if [n["op"] for n in nodes]==["square","rotate"]:
        n=math.prod(model["inputs"][0]["shape"])
        require(len(model["inputs"])==1 and len(model["outputs"])==1 and nodes[1]["attrs"]["step"]==-2 and period-n>=2,
                "Two-lane wrap fixture domain")
        lines=['@hc.func("c,c")','def golden(x, zero_ct):','    u = x * x']
        tail=rotation("u",-2,period,lines);wrap=rotation("u",n-2,period,lines)
        lines+=["    a = "+wrap+" * mask0"]
        last=rotation("u",n-1,period,lines)
        lines.append("    b = "+last+" * mask0")
        second=rotation("b",-1,period,lines)
        lines.append("    return ["+tail+" + a + "+second+"]")
        return "\n".join(lines)+"\n","packed-two-lane-wrap"
    if [n["op"] for n in nodes]==["subtract","mean"]:
        n=math.prod(model["inputs"][0]["shape"])
        require(len(model["inputs"])==1 and len(model["outputs"])==1 and model["inputs"][0]["shape"]==[n] and
                nodes[1]["attrs"]==dict(axes=[0],keepdims=False),"Full-axis mean fixture domain")
        shift=float(model["constants"][nodes[0]["inputs"][1]])
        lines=['@hc.func("c,c")','def golden(x, zero_ct):'];value="x"
        for bit in range(period.bit_length()-1):
            nxt="s"+str(bit);lines.append("    "+nxt+" = "+value+" + "+value+".rotate("+str(1<<bit)+")");value=nxt
        lines.append("    return ["+value+" * "+scalar(request,1./n)+" - "+scalar(request,shift)+"]")
        return "\n".join(lines)+"\n","mean-public-bias-factoring"
    require(any(n["op"]=="polynomial" and n["attrs"]["basis"]=="chebyshev" for n in nodes),
            "Unrecognized explicit developer fixture")
    return lower(request,packed_prefix=True,balanced_chebyshev=True),"exact-balanced-chebyshev"

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--fixtures",type=Path,required=True)
    p.add_argument("--case",action="append")
    a=p.parse_args()
    require(not a.output.exists() and not a.fixtures.exists(),"Fresh preparation directories")
    a.output.mkdir();a.fixtures.mkdir(parents=True)
    rows=[]
    for i in range(1,6):
        shard=WORK/"results"/("poseidon-math-shard%02d-r263"%i)
        for r in json.loads((shard/"report.json").read_text())["rows"]:
            if r["status"]=="passed" or r.get("failure_layer")=="poseidon_runtime" or (a.case and r["model_id"] not in a.case):continue
            model=json.loads((shard/r["model_id"]/"model.json").read_text())
            require(digest(model)==r["model_sha256"],"Frozen original model identity")
            cfg=configuration("poseidon-cpu-eva-w45-v1");request=prepare(model,cfg["ckks_config_sha256"],cfg)
            record=dict(model_id=model["id"],model_sha256=digest(model),request_id=request["request_id"],
                        original_status=r["status"],original_evidence=r.get("evidence"))
            try:
                source,strategy=make_source(model,request)
                validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
                checked=plaintext_check(source,request)
                path=a.fixtures/(model["id"]+".py");path.write_text(source)
                record.update(status="ready",strategy=strategy,plaintext=checked,source_sha256=hashlib.sha256(source.encode()).hexdigest(),source_path=str(path))
            except (ValueError,AssertionError,TypeError,StopIteration) as error:
                record.update(status="blocked",diagnostic=str(error))
            rows.append(record)
    report=dict(format="poseidon-math-expression-preparation-v1",rows=rows,
                planned=len(rows),ready=sum(r["status"]=="ready" for r in rows),
                reference_changed=False,model_changed=False,paid_calls=0)
    (a.output/"report.json").write_bytes(canonical(report))
    print(json.dumps({k:v for k,v in report.items() if k!="rows"}),flush=True)
if __name__=="__main__":main()
