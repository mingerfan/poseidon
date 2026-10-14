"""Packed native developer fixtures for unchanged frozen model failures."""
import argparse,ast,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from workspace_paths import WORK
from benchmark_graph import canonical,digest,require,validate
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import Emitter,Scalar
from chebyshev_lowering import balanced
from decorated_functions import _analyze
from unified_native_coverage import run_probe
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from check_poseidon_helpers import snapshot

class NativePacked:
    """Lazy logical views; gather only before actual arithmetic."""
    def __init__(self,request):
        self.request=request;self.period=request["layout"]["input_slot_period"]
        self.lines=[];self.helpers=[];self.gathers={};self.scales={}
    def emit(self,text):
        require(len(self.lines)<1024,"Developer fixture work boundary")
        name="n"+str(len(self.lines));self.lines.append("    "+name+" = "+text);return name
    def literal(self,value):
        return next(k for k,v in self.request["public_constants"].items() if type(v) in (int,float) and v==float(value))
    def rotate(self,name,step):
        step%=self.period
        for bit in range(self.period.bit_length()-1):
            if step&(1<<bit):name=self.emit(name+".rotate("+str(1<<bit)+")")
        return name
    def materialize(self,view):
        key=tuple(view.flat)
        if key in self.gathers:return self.gathers[key]
        require(view.size<=self.period,"Materialized prefix exceeds physical period")
        if all(cell is not None and cell[0]==key[0][0] and cell[1]==i for i,cell in enumerate(key)):
            return key[0][0]
        total=None
        for dst,cell in enumerate(key):
            if cell is None:continue
            mask="mask"+str(dst);require(mask in self.request["public_constants"],"Missing immutable destination selector")
            term=self.emit(self.rotate(cell[0],cell[1]-dst)+" * "+mask)
            total=term if total is None else self.emit(total+" + "+term)
        result=total or "zero_ct";self.gathers[key]=result;return result
    def view(self,expr,shape):
        array=np.empty(tuple(shape),dtype=object)
        for i in range(array.size):array.flat[i]=(expr,i)
        return array
    def uniform(self,view,op,number):
        number=float(number)
        if op=="multiply" and number==0:
            return np.full(view.shape,None,dtype=object)
        result=np.empty(view.shape,dtype=object)
        for i,cell in enumerate(view.flat):
            if cell is None and (number==0 or op=="multiply"):result.flat[i]=None;continue
            expr,index=("zero_ct",0) if cell is None else cell
            key=(expr,op,number)
            if key not in self.scales:
                token={"add":"+","subtract":"-","multiply":"*"}[op]
                self.scales[key]=self.emit(expr+" "+token+" "+self.literal(number))
            result.flat[i]=(self.scales[key],index)
        return result
    def negate(self,view):
        result=np.empty(view.shape,dtype=object);done={}
        for i,cell in enumerate(view.flat):
            if cell is None:result.flat[i]=None;continue
            expr,index=cell
            if expr not in done:done[expr]=self.emit("-"+expr)
            result.flat[i]=(done[expr],index)
        return result
    def polynomial(self,view,coeff,basis):
        require(basis=="chebyshev","This fixture only lowers exact Chebyshev basis")
        helper="series"+str(len(self.helpers));em=Emitter(self.request)
        em.zero=Scalar(em,"aux",True)
        value=balanced(Scalar(em,"a"),coeff,em.zero)
        source='@hc.func("c,c")\ndef '+helper+'(a, aux):\n'+"\n".join(em.lines)+"\n    return "+value.name+"\n"
        self.helpers.append(source)
        result=self.emit(helper+"("+self.materialize(view)+", zero_ct)")
        return self.view(result,view.shape)

def source_for(request):
    model=request["model"];check=validate(model);em=NativePacked(request)
    values={k:np.asarray(v,dtype=np.float64) for k,v in model["constants"].items()}
    for spec,binding in zip(model["inputs"],request["layout"]["inputs"]):
        values[spec["name"]]=em.view(binding["dsl_name"],spec["shape"])
    for node in model["nodes"]:
        op=node["op"];attrs=node["attrs"];xs=[values[k] for k in node["inputs"]];x=xs[0]
        if op in ("add","subtract","multiply"):
            if xs[1].dtype!=object:
                require(xs[1].size==1,"Public broadcast fixture domain")
                y=em.uniform(x,op,xs[1].flat[0])
            else:
                require(x.shape==xs[1].shape,"Packed pair shape equality")
                token={"add":"+","subtract":"-","multiply":"*"}[op]
                y=em.view(em.emit(em.materialize(x)+" "+token+" "+em.materialize(xs[1])),x.shape)
        elif op=="negate":y=em.negate(x)
        elif op=="slice":
            indices=[slice(None)]*x.ndim;indices[attrs["axis"]]=slice(attrs["start"],attrs["stop"],attrs["step"])
            y=x[tuple(indices)]
        elif op=="concat":y=np.concatenate(xs,axis=attrs["axis"])
        elif op=="reshape":y=x.reshape(attrs["shape"])
        elif op=="flatten":y=x.reshape(-1)
        elif op=="polynomial":y=em.polynomial(x,xs[1],attrs["basis"])
        else:raise ValueError("Developer native packed fixture does not implement "+op)
        require(list(y.shape)==check["shapes"][node["outputs"][0]],"Logical view shape")
        values[node["outputs"][0]]=y
    results=[em.materialize(values[o["value"]]) for o in model["outputs"]]
    names=[b["dsl_name"] for b in request["layout"]["inputs"]]+["zero_ct"]
    golden='@hc.func("'+",".join("c" for _ in names)+'")\ndef golden('+", ".join(names)+'):\n'
    golden+="\n".join(em.lines)+"\n    return ["+", ".join(results)+"]\n"
    return golden+"\n"+"\n".join(em.helpers)

def semantic_probe(source,request):
    names=tuple(s["dsl_name"] for s in request["layout"]["inputs"])+("zero_ct",)
    nodes,plan=_analyze(source,request["public_constants"],request["layout"]["output_ciphertexts"],
                        names,array_mutation=True,slot_period=request["layout"]["input_slot_period"])
    largest=0.
    for probe in range(16):
        selected,events=run_probe(nodes,plan,request,probe,[0],{})
        inputs={}
        for j,b in enumerate(request["layout"]["inputs"]):
            count=math.prod(b["shape"])
            inputs[b["name"]]=np.asarray([(((i*i+3*i+7*probe+5*j+1)%23)-11)/32 for i in range(count)]).reshape(b["shape"])
        for evaluate in (mathematical,torch_reference):
            expected=evaluate(request["model"],inputs)
            flat=np.concatenate([expected[o["name"]].reshape(-1) for o in request["model"]["outputs"]])
            np.testing.assert_allclose(selected,flat,atol=1e-12,rtol=1e-12)
            largest=max(largest,float(np.max(np.abs(np.asarray(selected)-flat))))
    return dict(probes=16,references=2,max_absolute_error=largest,
                kind="checked AST native vectors; not FHE or Agent evidence")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--fixtures",type=Path,required=True)
    a=p.parse_args();require(not a.output.exists() and not a.fixtures.exists(),"Fresh output and source fixtures")
    a.output.mkdir();a.fixtures.mkdir(parents=True)
    frozen=snapshot();aggregate=json.loads((WORK/"results/poseidon-math-aggregate-r267.json").read_text())
    suite=ROOT/"scripts/baseline/benchmarks/semantic-v2-ledger-r38"
    from benchmark_runner import load
    corpus,index=load(suite,check_sources=False)
    lookup={r["model_sha256"]:r["model"] for r in corpus}
    for row in json.loads((suite/"semantic-component-math_contexts.json").read_text())["supplemental_models"]:
        lookup[row["model_sha256"]]=row["model"]
    rows=[]
    for item in aggregate["rows"]:
        if item["accepted"]:continue
        model=lookup[item["model_sha256"]];require(digest(model)==item["model_sha256"],"Frozen model identity")
        cfg=configuration("poseidon-cpu-eva-w45-v1");request=prepare(model,cfg["ckks_config_sha256"],cfg)
        row=dict(model_id=model["id"],model_sha256=digest(model),request_id=request["request_id"])
        try:
            source=source_for(request)
            result=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
            checked=semantic_probe(source,request)
            path=a.fixtures/(model["id"]+".py");path.write_text(source)
            row.update(status="ready",source_path=str(path),source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       semantic_probe=checked,ast_nodes=sum(1 for _ in ast.walk(ast.parse(source))),
                       expanded_work=result["functions"]["golden"]["expanded_cost"],functions=len(result["functions"]))
        except (ValueError,TypeError,AssertionError,IndexError,StopIteration) as error:
            row.update(status="blocked",diagnostic=str(error)[:3000])
        rows.append(row)
    require(snapshot()==frozen,"Shared source changed")
    report=dict(format="poseidon-mathematical-native-fixture-v1",rows=rows,planned=len(rows),
                ready=sum(r["status"]=="ready" for r in rows),sources=frozen,paid_calls=0,
                model_or_reference_changed=False,limits_changed=False)
    (a.output/"report.json").write_bytes(canonical(report))
    print(json.dumps(dict(planned=len(rows),ready=report["ready"],paid_calls=0)),flush=True)
if __name__=="__main__":main()
