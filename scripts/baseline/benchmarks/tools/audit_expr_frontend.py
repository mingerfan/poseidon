"""Pinned Expr frontend behavior checks, separately scored from FHE execution."""
import argparse,hashlib,json,os,shlex,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import strict_file,dump
WORKER=r'''
import copy,json,struct
from pathlib import Path
from candidate_trace import load_frontend
hc=load_frontend()
mode=json.loads(Path("/payload.json").read_text())["context"]
if mode not in (0,1,2):raise ValueError("Trusted context")
records=[]
def reject(name,fn):
 try:fn()
 except Exception as error:
  records.append(dict(check=name,status="rejected",exception=type(error).__name__))
 else:raise AssertionError("Missing rejection: "+name)
@hc.func("c")
def golden(x):
 value=x if mode==0 else -x if mode==1 else x.rotate(1)
 for method in ("__add__","__radd__","__iadd__","__sub__","__rsub__","__isub__"):
  actual=getattr(hc.Empty(),method)(value)
  if actual is not value:raise AssertionError("Empty must return the same Expr")
  records.append(dict(check="Empty."+method,status="passed",same_expr=True))
 if hc.resolveType(value) is not value:raise AssertionError("Expr conversion identity")
 records.append(dict(check="resolveType.Expr",status="passed"))
 import numpy as np,torch
 values=[3,.25,[.5,-.25],np.array([.75,-.5]),torch.tensor([.125,-.625],dtype=torch.float64)]
 converted=[]
 for item in values:
  plain=hc.resolveType(item)
  if not isinstance(plain,hc.Plain):raise AssertionError("Public conversion type")
  converted.append(plain)
  records.append(dict(check="resolveType."+type(item).__name__,status="passed"))
 # Native constructor must own a copy before the caller changes its input array.
 data=np.array([.125,.25,.375,.5]);owned=hc.Plain(data);data[:]=9.
 records.append(dict(check="Plain.copy",status="pending_serialized_constant_check"))
 for name,receiver in (("Expr",value),("Plain",owned),("Func",golden)):
  reject("copy."+name,lambda receiver=receiver:copy.copy(receiver))
  reject("deepcopy."+name,lambda receiver=receiver:copy.deepcopy(receiver))
 for name,item in (("tuple",(1.,)),("mapping",{}),("object",object())):
  reject("resolveType."+name,lambda item=item:hc.resolveType(item))
 # Directly invoke the installed hook as well; bound copy/deepcopy may reject
 # earlier on the pinned hook's intentionally reported argument mismatch.
 reject("copy.installed_hook",lambda:hc.Expr.__copy__())
 return [value+plain for plain in converted]+[value+owned]
hc.save("/out","/out")
raw=Path("/out/_hecate_golden.cst").read_bytes()
count,=struct.unpack_from("<q",raw);offset=8;constants=[]
for i in range(count):
 n,=struct.unpack_from("<q",raw,offset);offset+=8
 if not 0<n<=16384:raise AssertionError("CST bounds")
 constants.append(list(struct.unpack_from("<"+"d"*n,raw,offset)));offset+=8*n
if offset!=len(raw):raise AssertionError("CST trailing bytes")
expected=[[3.],[.25],[.5,-.25],[.75,-.5],[.125,-.625],[.125,.25,.375,.5]]
if constants!=expected:raise AssertionError("Independent serialized constant reference")
for r in records:
 if r["check"]=="Plain.copy":r["status"]="passed"
Path("/out/frontend-checks.json").write_text(json.dumps(dict(context=mode,records=records,
 serialized_constants=constants,expected_constants=expected,actual_frontend=True,
 actual_compile=False,actual_ciphertext_execution=False)))
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from workspace_paths import RESULTS
 from hecate_python_env import enter_nix,VENV
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform output")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=180)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 from semantic_benchmark_execution import runtime_sources
 import candidate_sandbox as sandbox
 sources=runtime_sources();a.output.mkdir(parents=True)
 plan=dict(runtime_sources=sources,runner_sha256=sha(Path(__file__)),worker_sha256=digest(WORKER),
           contexts=["input_expr","negated_expr","rotated_expr"],seconds_per_context=45,paid_calls=0)
 plan["binding"]=digest(plan);dump(a.output/"plan.json",plan);rows=[]
 for context in range(3):
  folder=a.output/str(context);folder.mkdir();out=folder/"output";out.mkdir()
  dump(folder/"payload.json",dict(context=context))
  code=sandbox.run(folder/"payload.json",out,[str(VENV/"bin/python"),"-B","-c",WORKER],folder/"trace.log",seconds=45)
  row=dict(context=context,exit_code=code,status="trace_failed")
  if code==0:row.update(status="passed",checks=strict_file(out/"frontend-checks.json",131072))
  row["files"]={str(f.relative_to(folder)):sha(f) for f in folder.rglob("*") if f.is_file()}
  rows.append(row);dump(a.output/"progress.json",rows)
 if runtime_sources()!=sources:raise ValueError("Frontend source drift")
 report=dict(format="poseidon-expr-frontend-audit-v1",plan_binding=plan["binding"],rows=rows,
   status="passed" if all(r["status"]=="passed" for r in rows) else "failed",
   actual_frontend=True,actual_compile=False,actual_ciphertext_execution=False,paid_calls=0,
   scope="Actual upstream frontend identity, conversion, copy rejection and constant ownership; not FHE or arbitrary parameters")
 report["binding"]=digest(report);dump(a.output/"report.json",report)
 print(json.dumps(dict(status=report["status"],contexts=len(rows),failed=[r["context"] for r in rows if r["status"]!="passed"],paid_calls=0)))
 return int(report["status"]!="passed")
if __name__=="__main__":raise SystemExit(main())
