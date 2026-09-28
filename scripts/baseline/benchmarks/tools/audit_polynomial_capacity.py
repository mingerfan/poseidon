"""Prove the observed fixed-profile capacity rejection without claiming impossibility."""
import argparse,hashlib,json,re,sys,subprocess
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
from benchmark_graph import digest,signature
from benchmark_runner import strict_file,dump

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(value,reason):
 if not value:raise ValueError(reason)
def bound(p):
 d=strict_file(p,8*1024**2)
 require(digest({k:v for k,v in d.items() if k!="binding"})==d["binding"],"Report identity")
 return d
def check_operands(left,right,factor,upper):
 # Mirror the reviewed guard only for DIAGNOSTICS, never to bypass the compiler.
 shape,scale,level=left
 rshape,rscale,rlevel=right
 return dict(equal_level=level==rlevel,equal_shape=shape==rshape,
   required=level*factor+scale,capacity=upper*factor,
   within_capacity=level*factor+scale<=upper*factor)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error("Preserve previous audit")
 from workspace_paths import RESULTS
 profile=ROOT/"third_party/dacapo/profiled_SEAL_CPU.json";config=strict_file(profile,1024**2)
 cpp=ROOT/"third_party/dacapo/lib/Dialect/Earth/IR/EarthDialect.cpp";text=cpp.read_text()
 start=text.index("::mlir::LogicalResult hecate::earth::MulOp::inferReturnTypes")
 end=text.index("mlir::RankedTensorType hecate::earth::getTensorType",start)
 guard=text[start:end]
 pinned="4616402710f39df3e5f5bd7930a6c036025aaac3"
 historical=subprocess.run(["git","-C",str(ROOT/"third_party/dacapo"),"show",pinned+":lib/Dialect/Earth/IR/EarthDialect.cpp"],capture_output=True,check=True,timeout=10).stdout
 require(hashlib.sha256(historical).hexdigest()==sha(cpp),"Compiler guard differs from fixed gitlink")
 expected=("lScale.getLevel() == rScale.getLevel()","lTensor.getShape()[0] == rTensor.getShape()[0]",
  "(EarthDialect::bootstrapLevelUpperBound)*EarthDialect::rescalingFactor >=",
  "lScale.getLevel() * EarthDialect::rescalingFactor + lScale.getScale()")
 require(all(x in guard for x in expected),"Reviewed compiler guard changed")
 factor=config["rescalingFactor"];upper=config["bootstrapLevelUpperBound"]
 require((factor,upper)==(60,13),"Fixed profile drift")
 regex=r'tensor<(\d+)x!earth.ci<(\d+) \* (\d+)>>'
 records=[];parents={};negative=0
 for batch in ("stage2-polynomial-r68","stage2-polynomial-scaled-w40-r77"):
  folder=RESULTS/batch;plan=bound(folder/"plan.json");report=bound(folder/"report.json")
  require(plan["binding"]==report["plan_binding"],"Execution plan drift")
  require(sha(folder/"runner.py")==plan["runner_sha256"],"Frozen runner drift")
  for name,hsh in plan["binaries"].items():require(sha(Path(name))==hsh,"Executed binary identity drift")
  parents[batch]=dict(plan_sha256=sha(folder/"plan.json"),report_sha256=sha(folder/"report.json"),
    binding=report["binding"],path=str(folder))
  models={x["id"]:x for x in plan["models"]}
  for row in report["rows"]:
   if row["regime"] not in ("sign","relua","genRelu6"):continue
   spec=models[row["id"]];evidence=Path(row["folder"])
   require(row["status"]=="failed" and row["failure_layer"]=="compiler","Missing observed compiler rejection")
   result=strict_file(evidence/"report.json",8*1024**2);request=strict_file(evidence/"request.json",1024**2)
   require(not result["compiled"] and not result["encrypted_execution"],"Misclassified execution")
   require(request["compiler_configuration"]["ckks_config_sha256"]==sha(profile),"Profile identity")
   for name,hsh in row["files"].items():
    rel=Path(name)
    require(not rel.is_absolute() and ".." not in rel.parts and sha(evidence/rel)==hsh,"Retained evidence changed")
   log=(evidence/"compile.log").read_text()
   require("'earth.mul' op failed to infer returned types" in log,"Wrong compiler failure")
   matches=re.findall(regex,log)
   require(len(matches)==3,"Ambiguous failing operand types")
   left,right=[tuple(map(int,x)) for x in matches[:2]]
   check=check_operands(left,right,factor,upper)
   require(check["equal_level"] and check["equal_shape"] and not check["within_capacity"],"Capacity not isolated cause")
   # A passing-boundary synthetic operand must not be labelled over-capacity.
   control=check_operands((left[0],0,upper),(right[0],0,upper),factor,upper)
   require(control["within_capacity"],"Boundary negative control");negative+=1
   calls=result["calls"]["events"]
   require(any(e["symbol"]=="Poly.sign" for e in calls),"Actual nested sign trace missing")
   if row["regime"] in ("relua","genRelu6"):
    require(any(e["symbol"]=="Poly.relua" for e in calls),"Actual relua trace missing")
   clean=dict(spec["model"]);clean.pop("id")
   records.append(dict(id=row["id"],regime=row["regime"],context=spec["context"],
      topology=signature(spec["model"],True),model_without_id_sha256=digest(clean),
      compiler_configuration=request["compiler_configuration"]["name"],folder=str(evidence),
      source_batch=batch,operand_types=dict(left=left,right=right),guard=check,
      report_sha256=sha(evidence/"report.json"),log_sha256=sha(evidence/"compile.log"),
      calls_sha256=sha(evidence/"output/dependency-calls.json")))
 mappings=[]
 for regime in ("sign","relua","genRelu6"):
  subset=[r for r in records if r["regime"]==regime]
  require(len(subset)==6,"Three contexts per fixed configuration")
  for context in range(3):
   pair=[r for r in subset if r["context"]==context]
   require(len(pair)==2 and len({r["model_without_id_sha256"] for r in pair})==1,"Cross-profile model drift")
  require(len({r["topology"] for r in subset})==3,"Distinct context requirement")
  mappings.append(dict(symbol="Poly."+regime,state="configured_compiler_capacity_blocked",
    tested_configurations=["seal-cpu-eva-w40-v1","seal-cpu-eva-w45-v1"],
    contexts=3,attempts=6,reason="Actual unchanged helper traces; MulOp guard fails solely at fixed-profile capacity",
    source_contains_bootstrap=False,encrypted_passes=0,
    alternative_schedule_impossibility_proven=False,records=subset))
 result=dict(format="poseidon-fixed-capacity-audit-v1",mappings=mappings,parents=parents,
  sources={str(p.relative_to(ROOT)):sha(p) for p in (profile,cpp,ROOT/"third_party/dacapo/python/poly/poly/Poly.py",ROOT/"third_party/dacapo/python/poly/poly/MPCB.py")},
  guard_source=guard,guard_sha256=digest(guard),diagnostic_attempts=len(records),
  boundary_negative_controls=negative,profile_unchanged=True,security_unchanged=True,
  no_bootstrap_substitution=True,new_encrypted_executions=0,paid_calls=0,runner_sha256=sha(Path(__file__)))
 result["binding"]=digest(result);dump(a.output,result)
 print(json.dumps(dict(mapped_blockers=len(mappings),diagnostic_attempts=len(records),encrypted_passes=0,new_fhe=0,binding=result["binding"])))
 return 0
if __name__=="__main__":raise SystemExit(main())
