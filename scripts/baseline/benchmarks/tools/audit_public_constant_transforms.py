"""Independent public-coordinate references for pinned MPCB constant transforms.

Calls the actual closures; no replacement helper, frontend trace or FHE claim.
"""
import argparse,hashlib,json,os,shlex,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from benchmark_runner import dump
from audit_public_api_utilities import inferred
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def closures(functions):
    import inspect
    found={};visited=set()
    def visit(fn):
        if not inspect.isfunction(fn) or id(fn) in visited:return
        visited.add(id(fn));found[fn.__name__]=fn
        for cell in fn.__closure__ or ():
            value=cell.cell_contents
            if inspect.isfunction(value):visit(value)
    for fn in functions.values():visit(fn)
    return found
def coordinates(index,s,side):
    k,h,w=(s[x+side] for x in ("k","h","w"))
    low=index%k;index//=k;x=index%w;index//=w
    high=index%k;index//=k;y=index%h;group=index//h
    return group*k*k+high*k+low,y,x
def run_case(m,s):
    import numpy as np,torch
    funcs=closures(m.shapeClosure(**s));result=[]
    def check(name,args,expected):
        actual=funcs[name](*args)
        if isinstance(actual,tuple):
            actual=np.stack([v.numpy() for v in actual])
        else:actual=actual.numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        result.append(dict(symbol="MPCB.shapeClosure."+name,status="passed",
            shape=list(actual.shape),actual=actual.tolist(),reference=expected.tolist()))
    nt,ni,no,pi,po,ki,ti,co,ci,fh,fw,q=(s[k] for k in ("nt","ni","no","pi","po","ki","ti","co","ci","fh","fw","q"))
    gain=torch.arange(1,max(ci,co)+1,dtype=torch.float64)/8
    for side,name in [("i","ParInBNConst"),("o","ParBNConst")]:
        n,p,c=s["n"+side],s["p"+side],s["c"+side]
        expected=np.zeros((n,nt))
        for block in range(n):
            for slot in range(nt):
                channel,y,x=coordinates(block*(nt//p)+slot%(nt//p),s,side)
                if channel<c:expected[block,slot]=float(gain[channel])
        check(name,[gain[:c]],expected)
    expected=np.zeros((no,co,nt))
    for block in range(no):
        for slot in range(nt):
            channel,y,x=coordinates(block*nt+slot,s,"o")
            if channel<co:expected[block,channel,slot]=1.
    check("Selecting",[],expected)
    weight=torch.arange(co*ci*fh*fw,dtype=torch.float64).reshape(co,ci,fh,fw)/64-.25
    expected=np.zeros((ni,q,fh,fw,nt))
    for block,group,dy,dx,slot in np.ndindex(expected.shape):
        outch=group*pi+slot//(nt//pi)
        inch,y,x=coordinates(block*(nt//pi)+slot%(nt//pi),s,"i")
        valid=(0<=y+dy-(fh-1)//2<s["hi"] and 0<=x+dx-(fw-1)//2<s["wi"])
        if outch<co and inch<ci and valid:expected[block,group,dy,dx,slot]=float(weight[outch,inch,dy,dx])
    check("ParMultWgt",[weight],expected)
    weight=torch.arange(ci*fh*fw,dtype=torch.float64).reshape(ci,1,fh,fw)/32-.25
    expected=np.zeros((ni,fh,fw,nt))
    for block,dy,dx,slot in np.ndindex(expected.shape):
        inch,y,x=coordinates(block*(nt//pi)+slot%(nt//pi),s,"i")
        valid=(0<=y+dy-(fh-1)//2<s["hi"] and 0<=x+dx-(fw-1)//2<s["wi"])
        if inch<ci and valid:expected[block,dy,dx,slot]=float(weight[inch,0,dy,dx])
    check("DwMultWgt",[weight],expected)
    for name,outside,inside in [("ParMPDA",-.5,0.),("ParMPDM",0.,1.)]:
        expected=np.zeros((fh,fw,nt))
        for dy,dx,slot in np.ndindex(expected.shape):
            _,y,x=coordinates(slot,s,"i")
            valid=(0<=y+dy-(fh-1)//2<s["hi"] and 0<=x+dx-(fw-1)//2<s["wi"])
            expected[dy,dx,slot]=inside if valid else outside
        check(name,[],expected)
    expected=np.zeros((ni,ki,ti,nt))
    for block,outer,group,slot in np.ndindex(expected.shape):
        channel,y,x=coordinates(block*nt+slot,s,"i")
        expected[block,outer,group,slot]=float(channel//(ki*ki)==group and channel//ki%ki==outer
            and y%s["s"]==0 and x%s["s"]==0 and y<s["hi"]//s["s"]*s["s"] and x<s["wi"]//s["s"]*s["s"])
    check("DownSelecting",[],expected)
    expected=np.zeros((ni,ki*ti,nt))
    for block,row,slot in np.ndindex(expected.shape):
        index=block*nt+slot
        if index//ki==row:expected[block,row,slot]=1./(s["hi"]*s["wi"])
    check("PoolSelecting",[],expected)
    expected=np.zeros((2,nt));width=nt//po;extent=min(co*s["ho"]*s["wo"],nt);tail=ci*s["hi"]*s["wi"]%extent
    for slot in range(nt):
        local=slot%width
        if local<extent:expected[0 if local<tail else 1,slot]=1.
    check("ConcatSelecting",[],expected)
    # Public channel statistics and weight permutation have independent references.
    from types import SimpleNamespace
    bn=SimpleNamespace(weight=torch.arange(co,dtype=torch.float64)/8+.5,
        running_var=torch.arange(co,dtype=torch.float64)/4+.25,
        running_mean=torch.arange(co,dtype=torch.float64)/8-.5,
        bias=torch.arange(co,dtype=torch.float64)/16-.25,eps=.125)
    g,h=m.abstractBN(bn)
    import math
    expected_g=[float(bn.weight[i])/math.sqrt(float(bn.running_var[i])+bn.eps) for i in range(co)]
    expected_h=[float(bn.bias[i])-expected_g[i]*float(bn.running_mean[i]) for i in range(co)]
    actual=np.stack([g.numpy(),h.numpy()]);expected=np.asarray([expected_g,expected_h])
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    result.append(dict(symbol="MPCB.abstractBN",status="passed",actual=actual.tolist(),reference=expected.tolist()))
    count=s["to"]*s["ko"]**2*s["ho"]*s["wo"]
    weight=torch.arange(2*count,dtype=torch.float64).reshape(2,count)/16-.5
    expected=np.zeros((2,count))
    for row,slot in np.ndindex(expected.shape):
        channel,y,x=coordinates(slot,s,"o")
        original=(channel*s["ho"]+y)*s["wo"]+x
        expected[row,slot]=float(weight[row,original])
    actual=m.Reshape(weight,s).numpy();np.testing.assert_array_equal(actual,expected)
    result.append(dict(symbol="MPCB.Reshape",status="passed",actual=actual.tolist(),reference=expected.tolist()))
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stride",type=int,choices=(1,2),default=1)
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
    from hecate_python_env import enter_nix,VENV
    from workspace_paths import RESULTS
    if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):p.error("New platform result directory")
    if not a.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=180)
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:p.error("Pinned pure environment")
    from upstream_candidate_helpers import verify_sources
    from poly_dependencies import verify
    from upstream_adapters.test_batch_norm import BatchNormAdapterTests
    sources=verify_sources();dependency=verify();BatchNormAdapterTests.setUpClass()
    cases=[inferred(dict(nt=nt,bb=bb,fh=kernel,fw=kernel,s=stride,hi=2,wi=2,ki=k,ci=c,co=2*c))
           for nt,bb,k,c,kernel,stride in [(32,1.,1,2,3,a.stride),(64,2.,2,3,1,a.stride),(16,.5,2,5,3,a.stride)]]
    a.output.mkdir(parents=True);dump(a.output/"plan.json",dict(cases=cases,sources=sources,
        runner_sha256=sha(Path(__file__)),actual_ciphertext_execution=False,paid_calls=0))
    records=[];failures=[]
    for i,s in enumerate(cases):
        try:records.extend(dict(r,context=i,geometry=s) for r in run_case(BatchNormAdapterTests.mpcb,s))
        except Exception as error:failures.append(dict(context=i,type=type(error).__name__,reason=str(error)))
    dump(a.output/"records.json",records)
    from collections import Counter
    report=dict(format="poseidon-public-constant-audit-v1",status="failed" if failures else "passed",
        positive_contexts=dict(Counter(r["symbol"] for r in records)),failures=failures,
        plan_sha256=sha(a.output/"plan.json"),records_sha256=sha(a.output/"records.json"),
        sources=sources,dependency=dependency,runner_sha256=sha(Path(__file__)),
        actual_upstream_python=True,actual_frontend_tracing=False,actual_compile=False,
        actual_ciphertext_execution=False,direct_candidate_access_added=False,paid_calls=0,
        scope="Bounded public constant preprocessing; never helper numerical FHE coverage")
    report["binding"]=digest(report);dump(a.output/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("sources","dependency")}))
    return int(bool(failures))
if __name__=="__main__":raise SystemExit(main())
