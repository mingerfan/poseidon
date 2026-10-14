"""Developer-only lazy lane schedule for unchanged frozen arithmetic graphs."""
import ast,hashlib,json,sys,math
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
import numpy as np
from benchmark_graph import canonical,digest,require,validate
from compiler_configuration import configuration
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import Emitter,Scalar
from chebyshev_lowering import balanced
from prepare_poseidon_math_native import NativePacked,semantic_probe
from check_poseidon_helpers import snapshot
from workspace_paths import WORK

class LanePacked(NativePacked):
    def __init__(self,request):
        super().__init__(request);self.pairs={}
    def uniform(self,view,op,number):
        if op=="multiply" and float(number) in (1.,-1.,2.,-2.):
            if float(number)==1.:return view.copy()
            if float(number)==-1.:return self.negate(view)
            doubled=self.pair(view,view,"add")
            return self.negate(doubled) if float(number)==-2. else doubled
        return super().uniform(view,op,number)
    def pair(self,x,y,op):
        require(x.shape==y.shape,"Logical pair shape")
        result=np.empty(x.shape,dtype=object)
        for i,(a,b) in enumerate(zip(x.flat,y.flat)):
            if a is None and b is None:result.flat[i]=None;continue
            index=a[1] if a is not None else b[1]
            left=a[0] if a is not None else "zero_ct"
            right=b[0] if b is not None else "zero_ct"
            step=(b[1]-index)%self.period if b is not None else 0
            key=(left,right,step,op)
            if key not in self.pairs:
                self.pairs[key]=self.emit(left+" "+{"add":"+","subtract":"-","multiply":"*"}[op]+" "+self.rotate(right,step))
            result.flat[i]=(self.pairs[key],index)
        return result
    def polynomial(self,view,coeff,basis):
        require(basis=="chebyshev","Exact registered Chebyshev basis")
        helper="series"+str(len(self.helpers));em=Emitter(self.request);em.zero=Scalar(em,"aux",True)
        value=balanced(Scalar(em,"a"),coeff,em.zero)
        self.helpers.append('@hc.func("c,c")\ndef '+helper+'(a, aux):\n'+"\n".join(em.lines)+"\n    return "+value.name+"\n")
        result=np.empty(view.shape,dtype=object);mapped={}
        for i,cell in enumerate(view.flat):
            expression,index=cell if cell is not None else ("zero_ct",0)
            if expression not in mapped:mapped[expression]=self.emit(helper+"("+expression+", zero_ct)")
            result.flat[i]=(mapped[expression],index)
        return result

def source_for(request):
    model=request["model"];checked=validate(model);em=LanePacked(request)
    values={k:np.asarray(v,dtype=np.float64) for k,v in model["constants"].items()}
    for spec,binding in zip(model["inputs"],request["layout"]["inputs"]):
        values[spec["name"]]=em.view(binding["dsl_name"],spec["shape"])
    for node in model["nodes"]:
        op=node["op"];attrs=node["attrs"];xs=[values[k] for k in node["inputs"]];x=xs[0]
        if op in ("add","subtract","multiply"):
            if xs[1].dtype!=object:
                require(xs[1].size==1,"Public scalar fixture domain")
                y=em.uniform(x,op,xs[1].flat[0])
            else:y=em.pair(x,xs[1],op)
        elif op=="negate":y=em.negate(x)
        elif op=="slice":
            indexes=[slice(None)]*x.ndim;indexes[attrs["axis"]]=slice(attrs["start"],attrs["stop"],attrs["step"])
            y=x[tuple(indexes)].copy()
        elif op=="concat":y=np.concatenate(xs,axis=attrs["axis"])
        elif op=="polynomial":y=em.polynomial(x,xs[1],attrs["basis"])
        else:raise ValueError("Developer lane fixture operator "+op)
        require(list(y.shape)==checked["shapes"][node["outputs"][0]],"Logical output shape")
        values[node["outputs"][0]]=y
    result=[em.materialize(values[o["value"]]) for o in model["outputs"]]
    names=[b["dsl_name"] for b in request["layout"]["inputs"]]+["zero_ct"]
    return '@hc.func("'+",".join("c" for _ in names)+'")\ndef golden('+", ".join(names)+'):\n'+"\n".join(em.lines)+"\n    return ["+", ".join(result)+"]\n\n"+"\n".join(em.helpers)

def main():
    out=WORK/"results/poseidon-lane-doubles-prepare-r291"
    fixtures=ROOT/"scripts/baseline/golden_cases/poseidon_lane_doubles_v1"
    require(not out.exists() and not fixtures.exists(),"Fresh developer fixture paths")
    out.mkdir();fixtures.mkdir();frozen=snapshot()
    historical=WORK/"results/poseidon-math-native-r269"
    plan=json.loads((historical/"plan.json").read_text());cfg=configuration("poseidon-cpu-eva-w45-v1")
    rows=[]
    for item in plan["models"]:
        name=item["model_id"];model=json.loads((historical/name/"model.json").read_text())
        row=dict(model_id=name,model_sha256=digest(model),status="blocked")
        try:
            require(row["model_sha256"]==item["model_sha256"],"Immutable original model identity")
            request=prepare(model,cfg["ckks_config_sha256"],cfg);source=source_for(request)
            checked=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
            probe=semantic_probe(source,request)
            path=fixtures/(name+".py");path.write_text(source)
            row.update(status="ready",candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),request_id=request["request_id"],
                       source_path=str(path),semantic_probe=probe,
                       ast_nodes=sum(1 for _ in ast.walk(ast.parse(source))),
                       work=checked["functions"]["golden"]["expanded_cost"])
        except (ValueError,TypeError,AssertionError,IndexError,StopIteration) as error:
            row.update(diagnostic=str(error)[:2000])
        rows.append(row)
    require(snapshot()==frozen,"Application source unchanged")
    report=dict(format="poseidon-developer-lane-schedule-v1",planned=len(rows),ready=sum(r["status"]=="ready" for r in rows),
                rows=rows,sources=frozen,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                paid_calls=0,agent_successes=0,actual_compile=False,encrypted_execution=False,models_changed=False,
                parameters_changed=False,reference_changed=False)
    (out/"report.json").write_bytes(canonical(report))
    print(json.dumps(dict(planned=len(rows),ready=report["ready"],blocked=12-report["ready"],paid_calls=0)),flush=True)
    for row in rows:print(row["model_id"],row["status"],row.get("work"),row.get("diagnostic","")[:120],flush=True)
if __name__=="__main__":main()
