"""Bounded plaintext controls and independent composition audit; root owns launch."""
import argparse,hashlib,json,math,sys,time,unittest
from pathlib import Path
from cases import CASES,APPROXIMATION,build,samples,execution_candidates
from planner_composition import BASE,build_graph,FROZEN,LEGACY
from reference_math import composition
OWNED=("cases.py","reference_math.py","planner_composition.py","test_controls.py","test_math.py","audit.py")
def write(path,value):
 with path.open("x") as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args();output=Path(args.output)
 if output.exists():raise ValueError("fresh audit output required")
 import numpy as np,torch
 if (list(sys.version_info[:3]),np.__version__,torch.__version__,torch.version.cuda)!=([3,10,14],"1.25.2","2.0.1",None):raise ValueError("fixed ARM pinned Python")
 start=time.monotonic();output.mkdir(parents=True)
 import test_controls,test_math
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_controls,test_math)])
 with (output/"unit.log").open("x") as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 if not result.wasSuccessful():raise ValueError("reference controls failed; log preserved")
 from benchmark_math import evaluate
 from benchmark_torch import evaluate as torch_graph
 rows=[]
 for case in tuple(CASES)+execution_candidates():
  profile=case.get("approximation",APPROXIMATION);module,signature,outputs=build(case);graph,lowered,binding,mini=build_graph(case,module,signature,profile)
  count=0;values=0;exact_max=approx_max=lowered_max=0.
  for batch in samples(signature):
   exact=composition(case,module,batch,profile,approximate=False);approx=composition(case,module,batch,profile,approximate=True)
   native,_=test_math.MathControls().native(module,batch,signature,outputs)
   computed=evaluate(lowered,batch,resource_profile="general-tensors-v1");tensor=torch_graph(lowered,batch,resource_profile="general-tensors-v1")
   for name in outputs:
    np.testing.assert_allclose(native[name],exact[name],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(computed[name],approx[name],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(tensor[name],approx[name],atol=1e-12,rtol=1e-12)
    exact_max=max(exact_max,float(np.max(np.abs(native[name]-exact[name]))));approx_max=max(approx_max,float(np.max(np.abs(approx[name]-exact[name]))));lowered_max=max(lowered_max,float(np.max(np.abs(computed[name]-approx[name]))),float(np.max(np.abs(tensor[name]-approx[name]))));values+=exact[name].size
   count+=1
   if time.monotonic()-start>150:raise ValueError("plaintext audit 150 second body boundary")
  rows.append(dict(id=case["id"],configuration_kind="separate_declared_shallow_feasibility" if "approximation" in case else "fixed_original_math_profile",inputs=signature,outputs=outputs,approximation=profile,groups=count,compared_values=int(values),original_nodes=len(graph["nodes"]),decomposed_nodes=len(lowered["nodes"]),max_native_vs_independent_exact=exact_max,max_graph_vs_independent_declared=lowered_max,max_declared_vs_exact=approx_max,original_accuracy_certified=False,backend_status="not_run",source_capture_status="not_run",agent_status="not_run"))
  write(output/(case["id"]+".graph.json"),graph)
 report=dict(format="poseidon-ordered-mha-composition-reference-v1",status="passed",tests=result.testsRun,passed=result.testsRun-len(result.skipped),failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),cases=rows,elapsed_seconds=time.monotonic()-start,api_calls=0,nix_entries=0,compiler_calls=0,encrypted_executions=0,production_activation=False,implementation={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in OWNED},fixed_math={n:hashlib.sha256((LEGACY/n).read_bytes()).hexdigest() for n in FROZEN},runtime=dict(python=list(sys.version_info[:3]),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda))
 report["binding"]=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest();write(output/"report.json",report)
 print(json.dumps({k:report[k] for k in ("status","tests","passed","failed","errors","skipped","binding","elapsed_seconds")},sort_keys=True))
if __name__=="__main__":main()
