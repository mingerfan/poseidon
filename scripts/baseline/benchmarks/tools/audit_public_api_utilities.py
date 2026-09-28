"""Actual pinned public utility behavior; no ciphertext/Agent coverage claims."""
import argparse,copy,hashlib,json,math,os,shlex,sys
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump
def inferred(s):
 r=dict(s);r.update(ho=s["hi"]//s["s"],wo=s["wi"]//s["s"],ko=s["s"]*s["ki"])
 r["ti"]=(s["ci"]+s["ki"]**2-1)//s["ki"]**2
 r["to"]=(s["co"]+r["ko"]**2-1)//r["ko"]**2
 for side in ("i","o"):
  volume=r["k"+side]**2*r["h"+side]*r["w"+side]*r["t"+side]
  r["n"+side]=(volume+s["nt"]-1)//s["nt"]
  repeats=1
  while repeats*2*volume<=s["nt"]:repeats*=2
  r["p"+side]=repeats
 r["q"]=(s["co"]+r["pi"]-1)//r["pi"]
 return r
def base_cases():
 return [dict(nt=nt,bb=bb,fh=1,fw=1,s=1,hi=h,wi=w,ki=k,ci=c,co=c)
         for nt,bb,h,w,k,c in [(16,1.,2,2,1,2),(32,2.,2,2,2,3),(8,.5,2,2,2,3)]]
def expected_pack(array,s,side):
 import numpy as np
 k,h,w,t,n,p,c=(s[x+side] for x in ("k","h","w","t","n","p","c"))
 flat=[0.]*(n*s["nt"]//p)
 for ch in range(c):
  for y in range(h):
   for x in range(w):
    index=((((ch//(k*k))*h+y)*k+(ch//k)%k)*w+x)*k+ch%k
    flat[index]=float(array[0,ch,y,x])/s["bb"]
 if side=="i":
  width=s["nt"]//p
  return np.asarray([flat[j*width:(j+1)*width]*p for j in range(n)])
 return np.asarray(flat*p).reshape(n,s["nt"])
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from hecate_python_env import enter_nix,VENV
 from workspace_paths import RESULTS,ROOT
 if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform output")
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=180)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
 import numpy as np,torch
 from upstream_candidate_helpers import verify_sources
 from poly_dependencies import verify
 sources=verify_sources();dependency=verify()
 from upstream_adapters.test_batch_norm import BatchNormAdapterTests
 BatchNormAdapterTests.setUpClass()
 m=BatchNormAdapterTests.mpcb;hc=sys.modules["hecate"];records=[]
 def record(symbol,context,actual,expected,kind="plaintext_value"):
  if isinstance(actual,np.ndarray):np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12);actual=actual.tolist();expected=np.asarray(expected).tolist()
  elif actual!=expected:raise ValueError(symbol+" reference mismatch")
  records.append(dict(symbol=symbol,context=context,kind=kind,status="passed",actual=actual,expected=expected))
 def reject(symbol,context,call,exceptions):
  try:call()
  except exceptions as error:records.append(dict(symbol=symbol,context=context,kind="upstream_rejection",status="passed",exception=type(error).__name__));return
  raise ValueError("Expected rejection: "+symbol)
 for i,value in enumerate([-1.25,0.,1.25]):
  record("MPCB.cint",str(i),m.cint(value),math.ceil(value))
  record("MPCB.fint",str(i),m.fint(value),math.floor(value))
 for symbol,fn in [("MPCB.cint",m.cint),("MPCB.fint",m.fint)]:
  reject(symbol,"nonfinite",lambda:fn(float("nan")),(ValueError,))
 for i,(value,shape,flat) in enumerate([(2.,[],[2.]),([1,2],[2],[1,2]),([[1,2],[3,4]],[2,2],[1,2,3,4])]):
  record("expr.recType",str(i),hc.recType(value),shape)
  record("expr.flatten",str(i),list(hc.flatten([value] if not isinstance(value,list) else value)),flat)
 reject("expr.recType","ragged",lambda:hc.recType([[1],[2,3]]),(Exception,))
 reject("expr.recType","empty",lambda:hc.recType([]),(IndexError,))
 reject("expr.flatten","noniterable",lambda:list(hc.flatten(1)),(TypeError,))
 for i,base in enumerate(base_cases()):
  geometry=inferred(base);actual=m.InferShapes(dict(base))
  record("MPCB.InferShapes",str(i),actual,geometry,"public_geometry")
  for name in ("CascadePool","CascadeDS","CascadeConv","CascadeMax","CascadeConcat"):
   prev=copy.deepcopy(geometry);expected=dict(geometry)
   expected.update(hi=geometry["ho"],wi=geometry["wo"],ki=geometry["ko"],ci=geometry["co"])
   if name=="CascadePool":expected.update(fh=1,fw=1,s=1,co=geometry["co"]);got=m.CascadePool(geometry)
   elif name=="CascadeDS":expected.update(fh=1,fw=1,s=2,co=geometry["co"]*2);got=m.CascadeDS(geometry)
   elif name=="CascadeConv":
    conv=SimpleNamespace(kernel_size=(3,1),stride=(1,1),in_channels=geometry["co"],out_channels=geometry["co"]+1)
    expected.update(fh=1,fw=3,s=1,co=geometry["co"]+1);got=m.CascadeConv(geometry,conv)
   elif name=="CascadeMax":
    kernel=2 if i==0 else (3,1);stride=1 if i==0 else (2,2)
    maxop=SimpleNamespace(kernel_size=kernel,stride=stride)
    expected.update(fh=2 if i==0 else 1,fw=2 if i==0 else 3,s=1 if i==0 else 2,co=geometry["co"]);got=m.CascadeMax(geometry,maxop)
   else:
    # Concat requires channel count divisible by the current packing square.
    geom=dict(geometry);geom["co"]=geom["ko"]**2*max(1,geom["to"])
    expected=dict(geom);expected.update(fh=1,fw=1,s=1,ci=geom["co"],co=2*geom["co"],hi=geom["ho"],wi=geom["wo"],ki=geom["ko"])
    got=m.CascadeConcat(geom,dict(geom))
   record("MPCB."+name,str(i),got,inferred(expected),"public_geometry")
   if geometry!=prev:raise ValueError("Unexpected cascade input mutation")
  closure=m.shapeClosure(**geometry)
  for side,key in [("i","MPP"),("o","OP")]:
   shape=(1,geometry["c"+side],geometry["h"+side],geometry["w"+side])
   data=(torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)-3.)/8
   actual=closure[key](data).numpy();want=expected_pack(data,geometry,side)
   record("MPCB.shapeClosure."+("MultParPack" if side=="i" else "OutPack"),str(i),actual,want,"public_tensor_pack")
  changed=dict(geometry);changed["nt"]*=2
  reject("MPCB.CascadeConcat","incompatible_geometry_"+str(i),lambda:m.CascadeConcat(geometry,changed),(ValueError,))
 # These utilities compute only public shapes or arrays. Importing the actual
 # frontend does not make their results ciphertext-execution evidence.
 a.output.mkdir(parents=True);dump(a.output/"records.json",records)
 symbols=sorted({r["symbol"] for r in records})
 report=dict(format="poseidon-public-api-utility-audit-v1",status="passed",symbols=symbols,
  positive_contexts={n:len([r for r in records if r["symbol"]==n and r["kind"]!="upstream_rejection"]) for n in symbols},
  rejection_controls=len([r for r in records if r["kind"]=="upstream_rejection"]),
  sources=sources,dependency=dependency,records_sha256=hashlib.sha256((a.output/"records.json").read_bytes()).hexdigest(),
  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  actual_upstream_python=True,actual_frontend_tracing=False,actual_compile=False,
  actual_ciphertext_execution=False,direct_candidate_access_added=False,paid_calls=0,
  scope="Public preprocessing/legacy literal behavior only; no numerical DSL partition or FHE success credit")
 report["binding"]=digest(report);dump(a.output/"report.json",report)
 print(json.dumps({k:v for k,v in report.items() if k not in ("sources","dependency")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
