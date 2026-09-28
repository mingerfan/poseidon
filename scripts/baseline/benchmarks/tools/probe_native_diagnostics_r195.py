"""Offline diagnostic repairs of retained answers; NEVER count as Agent generation."""
import ast,copy,json,sys,shlex,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
def repair(source,public):
 tree=ast.parse(source);helpers={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!="golden"}
 entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="golden");clones=[];changes=[]
 class Calls(ast.NodeTransformer):
  def visit_Call(self,node):
   node=self.generic_visit(node)
   if not isinstance(node.func,ast.Name) or node.func.id not in helpers:return node
   fn=helpers[node.func.id];kinds=fn.decorator_list[0].args[0].value.split(",")
   assert len(node.args)==len(fn.args.args)==len(kinds) and not node.keywords
   remove={p.arg:a for p,k,a in zip(fn.args.args,kinds,node.args) if k=="p" and isinstance(a,ast.Name) and a.id in public}
   clone=copy.deepcopy(fn);clone.name="repaired_scope_"+str(len(clones))
   for n in ast.walk(clone):
    assert not (isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id in remove)
   class Bind(ast.NodeTransformer):
    def visit_Name(self,n):return copy.deepcopy(remove[n.id]) if n.id in remove else n
   clone.body=[Bind().visit(n) for n in clone.body]
   keep=[i for i,p in enumerate(fn.args.args) if p.arg not in remove]
   clone.args.args=[clone.args.args[i] for i in keep]
   clone.decorator_list[0].args[0].value=",".join(kinds[i] for i in keep)
   assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in helpers for n in ast.walk(clone))
   clones.append(clone);changes.append(dict(original_helper=fn.name,clone=clone.name,public_bindings={k:v.id for k,v in remove.items()}))
   node.func=ast.Name(id=clone.name,ctx=ast.Load());node.args=[node.args[i] for i in keep];return node
 entry=Calls().visit(entry)
 tree.body=clones+[entry]
 class Negative(ast.NodeTransformer):
  def visit_BinOp(self,n):
   n=self.generic_visit(n)
   if isinstance(n.op,ast.Mult) and isinstance(n.left,ast.UnaryOp) and isinstance(n.left.op,ast.USub) and isinstance(n.left.operand,ast.Name) and n.left.operand.id in public:
    changes.append(dict(negative_plain=n.left.operand.id,action="Negate entire product"))
    return ast.UnaryOp(op=ast.USub(),operand=ast.BinOp(left=n.left.operand,op=ast.Mult(),right=n.right))
   return n
 tree=Negative().visit(tree)
 return ast.unparse(ast.fix_missing_locations(tree))+"\n",changes
def main():
 from hecate_python_env import VENV,enter_nix,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=1200)
 from semantic_benchmark_execution import runtime_sources
 from component_backend import qualify
 from benchmark_graph import digest
 report=json.loads((ROOT/"docs/baseline/stage2-agent-repair-result-r194.json").read_text())
 out=WORK/"results/native-diagnostic-probes-r195";out.mkdir()
 sources=runtime_sources();rows=[]
 guards=json.loads((WORK/"results/seal-gate-r181/before.json").read_text())["compiler"]
 for row in report["rows"]:
  if row["completion_failure_layer"]!="static_check":continue
  assert runtime_sources()==sources and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in guards.items())
  folder=Path(row["evidence"]);r=json.loads((folder/"report.json").read_text());index=r["attempts"][-1]["index"]
  original=json.loads((folder/("attempt-%02d"%index)/"response.txt").read_text())
  q=json.loads((folder/"request.json").read_text())
  repaired,changes=repair(original["hecate_source"],q["public_constants"])
  assert changes
  candidate=dict(original,hecate_source=repaired)
  result=qualify(q,candidate)
  item=dict(id=row["id"],original_evidence=str(folder),original_source_sha256=hashlib.sha256(original["hecate_source"].encode()).hexdigest(),
   repaired_source_sha256=hashlib.sha256(repaired.encode()).hexdigest(),changes=changes,result=result,new_agent_generation=False)
  rows.append(item)
  value=dict(format="poseidon-native-diagnostic-probes-r195",source_hashes=sources,rows=rows,paid_calls=0,
   compiler_changed=False,manual_only=True)
  value["binding"]=digest(value);(out/"report.json").write_text(json.dumps(value,indent=2))
  print(json.dumps(dict(id=row["id"],result=result,changes=len(changes))),flush=True)
 assert len(rows)==7
 return 0
if __name__=="__main__":raise SystemExit(main())
