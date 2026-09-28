"""Read-only audit of package exports omitted by the original four-file inventory.

Does not import runner.py, load keys, instantiate PyTorch models or broaden the
candidate contract. Missing __all__ names are unimplemented declarations.
"""
import argparse,ast,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump,strict_file
RUNTIME={"reinit_lw","setLibnHW","HEVM","HEVM.__init__","HEVM.load","HEVM.run",
 "HEVM.setInput","HEVM.setDebug","HEVM.setToGPU","HEVM.getOutput","HEVM.printer"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def definitions(tree):
 result={}
 def visit(body,prefix=""):
  for node in body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
    result[prefix+node.name]=node
    visit(node.body,prefix+node.name+".")
 visit(tree.body)
 return result
def top_bindings(tree):
 """Static syntactic bindings, not an assertion that arbitrary imports execute."""
 names=set()
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):names.add(node.name)
  elif isinstance(node,ast.Import):names.update(x.asname or x.name.split(".")[0] for x in node.names)
  elif isinstance(node,ast.ImportFrom):
   if any(x.name=="*" for x in node.names):raise ValueError("Unexpected nested wildcard import")
   names.update(x.asname or x.name for x in node.names)
  elif isinstance(node,(ast.Assign,ast.AnnAssign)):
   for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
    if isinstance(target,ast.Name):names.add(target.id)
 return names
def build():
 hecate=ROOT/"third_party/dacapo/python/hecate/hecate";poly=ROOT/"third_party/dacapo/python/poly/poly"
 source_paths=sorted([*hecate.rglob("*.py"),*poly.rglob("*.py")])
 sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
 parsed={p:ast.parse(p.read_text()) for p in source_paths}
 ledger=strict_file(BASE/"benchmarks/semantic-v2-ledger-r38/coverage.json",4*1024**2)
 for name,h in ledger["upstream_source_hashes"].items():
  if sha(ROOT/name)!=h:raise ValueError("Pinned DSL source drift")
  sources[name]=h
 init=parsed[hecate/"__init__.py"];declared=None
 imports=[ast.unparse(n) for n in init.body if isinstance(n,ast.ImportFrom)]
 if imports!=["from .expr import *","from .runner import *"]:raise ValueError("Unreviewed hecate import graph")
 for n in init.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="__all__" for t in n.targets):
   declared=ast.literal_eval(n.value)
 if not isinstance(declared,list) or any(type(x) is not str for x in declared):raise ValueError("Dynamic exports need review")
 expr=parsed[hecate/"expr.py"];runner=parsed[hecate/"runner.py"]
 available=top_bindings(expr)|top_bindings(runner)
 unary=next(n.value for n in expr.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="toUnary" for t in n.targets))
 if ast.literal_eval(unary)!={"bootstrap":0}:raise ValueError("Unreviewed generated unary operation")
 available.add("bootstrap")
 runtime=definitions(runner)
 if set(runtime)!=RUNTIME:raise ValueError("Unreviewed runtime interface")
 rows=[]
 for name,node in runtime.items():
  rows.append(dict(symbol="runner."+name,source=str((hecate/"runner.py").relative_to(ROOT)),
   line=node.lineno,end_line=node.end_lineno,ast_sha256=digest(ast.dump(node,include_attributes=False)),
   disposition="trusted_runtime_administration",candidate_callable=False,
   reason="Library/context/key/file/encrypt/decrypt/run/debug administration owned by trusted harness; not a generated DSL expression",
   execution_claim=False))
 exports=[dict(symbol=name,state="statically_bound" if name in available else "declared_without_definition",
               source="third_party/dacapo/python/hecate/hecate/__init__.py",
               execution_claim=False) for name in declared]
 # Models are a namespace containing reference PyTorch applications, not imported
 # wildcard helper definitions. Do not count model architectures as DSL operators.
 models=poly/"models"
 if (models/"__init__.py").exists():raise ValueError("Model namespace export graph changed")
 expected_imports=["from .MPCB import *","from .Poly import *","from .Func import *","from .models import *"]
 if [ast.unparse(n) for n in parsed[poly/"__init__.py"].body if isinstance(n,ast.ImportFrom)]!=expected_imports:
  raise ValueError("Unreviewed poly import graph")
 templates=[]
 for p in sorted(models.glob("*.py")):
  for node in parsed[p].body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
    templates.append(dict(symbol=p.stem+"."+node.name,source=str(p.relative_to(ROOT)),
     line=node.lineno,end_line=node.end_lineno,ast_sha256=digest(ast.dump(node,include_attributes=False)),
     disposition="reference_model_definition",candidate_callable=False,execution_claim=False))
 # Verify the 15 real HE_* helper entry definitions were already inventoried.
 helper_names={n.name for n in parsed[poly/"Func.py"].body if isinstance(n,ast.FunctionDef)}
 inventoried={r["symbol"] for r in ledger["upstream_api"] if r["source"].endswith("/Func.py") and "." not in r["symbol"]}
 if helper_names!=inventoried:raise ValueError("Missing public HE helper definition")
 return dict(format="poseidon-public-export-audit-v1",source_hashes=sources,
  ledger_sha256=sha(BASE/"benchmarks/semantic-v2-ledger-r38/coverage.json"),
  original_api_inventory=107,original_inventory_scope="Named definitions in expr.py, Func.py, MPCB.py, Poly.py",
  reviewed_package_python_files=len(source_paths),runtime_interfaces=rows,reference_models=templates,
  hecate_declared_exports=exports,he_helpers_already_inventoried=sorted(helper_names),
  unknown_executable_helper_definitions=[],
  classification_only=True,actual_compile=False,actual_ciphertext_execution=False,paid_calls=0,
  whole_python_support=False,candidate_permissions_unchanged=True,
  limitation="Static source/export audit; exported imported modules do not grant candidates arbitrary Python/NumPy/Torch access")
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error("Preserve prior audit")
 r=build();r["runner_sha256"]=sha(Path(__file__));r["binding"]=digest(r)
 a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,r)
 print(json.dumps(dict(package_files=r["reviewed_package_python_files"],runtime_interfaces=len(r["runtime_interfaces"]),
  reference_definitions=len(r["reference_models"]),
  declared_but_missing=[e["symbol"] for e in r["hecate_declared_exports"] if e["state"]=="declared_without_definition"],
  actual_helper_definitions=len(r["he_helpers_already_inventoried"]),paid_calls=0)))
if __name__=="__main__":main()
