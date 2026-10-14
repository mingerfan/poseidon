"""Exact weighted Chebyshev recurrence; isolated developer feasibility probes."""
import ast,copy,hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
import numpy as np
import prepare_poseidon_math_native as original
from unified_graph_lowering import Emitter,Scalar
from unified_graph_contract import prepare,validate_candidate
from compiler_configuration import configuration
from benchmark_graph import require,digest,canonical
from decorated_functions import _analyze
from check_poseidon_helpers import snapshot
from workspace_paths import WORK

import prepare_poseidon_lane_doubles as lazy
from prepare_poseidon_lane_doubles import LanePacked

class WeightedPacked(LanePacked):
    threshold_mode="high"
    def polynomial(self,view,coeff,basis):
        require(basis=="chebyshev","Exact Chebyshev recurrence")
        helper="series"+str(len(self.helpers));em=Emitter(self.request);em.zero=Scalar(em,"aux",True);x=Scalar(em,"a")
        terms={0:em.zero+1.,1:x};weights={}
        def term(n):
            if n not in terms:
                k=n//2;v=term(k)*term(k+1) if n%2 else term(k)*term(k)
                terms[n]=v+v-x if n%2 else v+v-1.
            return terms[n]
        def weighted(n,c):
            key=(n,float(c))
            if key not in weights:
                if n==0:y=em.zero+float(c)
                elif n==1:y=x*float(c)
                else:
                    k=n//2
                    product=weighted(k,c)*term(k+1 if n%2 else k)
                    y=product+product-(weighted(1,c) if n%2 else weighted(0,c))
                weights[key]=y
            return weights[key]
        highest=(len(coeff)-1).bit_length()-1
        threshold=2 if len(coeff)>16 else len(coeff)
        result=em.zero
        for n,c in enumerate(coeff):
            if float(c)==0:continue
            result=result+(weighted(n,c) if n>=threshold else term(n)*float(c))
        self.helpers.append('@hc.func("c,c")\ndef '+helper+'(a, aux):\n'+"\n".join(em.lines)+"\n    return "+result.name+"\n")
        if self.threshold_mode=="native":
            return self.view(self.emit(helper+"("+self.materialize(view)+", zero_ct)"),view.shape)
        result=np.empty(view.shape,dtype=object);mapped={}
        for i,cell in enumerate(view.flat):
            expression,index=cell if cell is not None else ("zero_ct",0)
            if expression not in mapped:mapped[expression]=self.emit(helper+"("+expression+", zero_ct)")
            result.flat[i]=(mapped[expression],index)
        return result

def inline_pipeline(source):
    tree=ast.parse(source);functions={n.name:n for n in tree.body}
    fn=functions["golden"];body=[];counter=0
    for stmt in fn.body:
        if not isinstance(stmt,ast.Assign) or not isinstance(stmt.value,ast.Call) or not isinstance(stmt.value.func,ast.Name) or stmt.value.func.id not in functions:
            body.append(stmt);continue
        callee=functions[stmt.value.func.id]
        require(len(callee.args.args)==len(stmt.value.args),"Static helper arity")
        params={a.arg:v for a,v in zip(callee.args.args,stmt.value.args)}
        locals={x.targets[0].id for x in callee.body if isinstance(x,ast.Assign)}
        prefix="inline"+str(counter)+"_";counter+=1
        class Names(ast.NodeTransformer):
            def visit_Name(self,node):
                if isinstance(node.ctx,ast.Load) and node.id in params:return copy.deepcopy(params[node.id])
                if node.id in locals:return ast.copy_location(ast.Name(id=prefix+node.id,ctx=node.ctx),node)
                return node
        for inner in callee.body:
            inner=Names().visit(copy.deepcopy(inner))
            if isinstance(inner,ast.Return):
                body.append(ast.Assign(targets=copy.deepcopy(stmt.targets),value=inner.value))
            else:body.append(inner)
    fn.body=body
    def depth(n):return 1+max((depth(x) for x in ast.iter_child_nodes(n)),default=0)
    counts={}
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load):counts[node.id]=counts.get(node.id,0)+1
    definitions={};compacted=[]
    class Inline(ast.NodeTransformer):
        def visit_Name(self,node):
            if isinstance(node.ctx,ast.Load) and node.id in definitions:return copy.deepcopy(definitions[node.id])
            return node
    for statement in fn.body:
        statement=Inline().visit(copy.deepcopy(statement))
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id
            if counts.get(name)==1 and isinstance(statement.value,(ast.BinOp,ast.UnaryOp)) and depth(statement.value)<=56:
                definitions[name]=statement.value
                continue
        compacted.append(statement)
    fn.body=compacted
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])))+"\n"

def main():
    out=WORK/"results/poseidon-weighted-lazy-prepare-r309"
    fixtures=ROOT/"scripts/baseline/golden_cases/poseidon_weighted_lazy_v1"
    require(not out.exists() and not fixtures.exists(),"Fresh source/diagnostic paths")
    out.mkdir();fixtures.mkdir();frozen=snapshot()
    prior=WORK/"results/poseidon-math-native-r269";plan=json.loads((prior/"plan.json").read_text());rows=[]
    def save():
        report=dict(format="poseidon-weighted-chebyshev-feasibility-v1",planned=24,rows=rows,sources=frozen,
                    ready=sum(x["status"]=="ready" for x in rows),not_run=24-len(rows),
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    coefficients_truncated=False,models_changed=False,reference_changed=False,parameters_changed=False,
                    actual_compile=False,encrypted_execution=False,agent_calls=0,paid_calls=0)
        temporary=out/"report.json.partial";temporary.write_bytes(canonical(report));temporary.replace(out/"report.json")
    save()
    for mode in ("native","lane"):
        WeightedPacked.threshold_mode=mode
        original.NativePacked=WeightedPacked
        folder=fixtures/mode;folder.mkdir()
        for item in plan["models"]:
            model=json.loads((prior/item["model_id"]/"model.json").read_text());cfg=configuration("poseidon-cpu-eva-w45-v1")
            request=prepare(model,cfg["ckks_config_sha256"],cfg);lazy.LanePacked=WeightedPacked
            source=inline_pipeline(original.source_for(request) if mode=="native" else lazy.source_for(request))
            path=folder/(model["id"]+".py");path.write_text(source)
            row=dict(model_id=model["id"],mode=mode,model_sha256=digest(model),source_path=str(path),
                     source_sha256=hashlib.sha256(source.encode()).hexdigest(),ast_nodes=sum(1 for _ in ast.walk(ast.parse(source))),
                     status="blocked",compile_attempted=False,encrypted_execution=False)
            try:
                check=validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
                probe=original.semantic_probe(source,request)
                row.update(status="ready",expanded_cost=check["functions"]["golden"]["expanded_cost"],semantic_probe=probe,
                           request_id=request["request_id"])
            except (ValueError,TypeError,AssertionError,IndexError,StopIteration) as error:
                row.update(diagnostic=str(error)[:500])
            rows.append(row);save()
            print(mode,model["id"],row["status"],row["ast_nodes"],row.get("expanded_cost"),row.get("diagnostic",""),flush=True)
    require(snapshot()==frozen,"Frozen application source")
    report=dict(format="poseidon-weighted-chebyshev-feasibility-v1",planned=len(rows),rows=rows,sources=frozen,
                ready=sum(x["status"]=="ready" for x in rows),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                coefficients_truncated=False,models_changed=False,reference_changed=False,parameters_changed=False,
                actual_compile=False,encrypted_execution=False,agent_calls=0,paid_calls=0)
    (out/"report.json").write_bytes(canonical(report))
if __name__=="__main__":main()
