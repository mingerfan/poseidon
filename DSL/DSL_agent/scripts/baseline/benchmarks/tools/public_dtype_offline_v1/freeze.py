"""Add tool and immutable task bindings without altering the prepared draft."""
import os,sys,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parents[2]))
from scope import *
def main():
 require(not PROPOSAL.exists()and not AUTH.exists()and not(OUT/"launch.json").exists(),"dtype_fresh_ready_proposal")
 old=read(DRAFT);require(old["binding"]==digest({k:v for k,v in old.items()if k!="binding"}),"dtype_prepared_draft")
 plan={k:v for k,v in old.items()if k!="binding"}
 plan.update(format="poseidon-public-dtype-offline-ready-v1",prepared_draft_binding=old["binding"],
  tools={str((HERE/n).relative_to(ROOT)):sha(HERE/n)for n in("prepare.py","scope.py","run.py","freeze.py")},
  task_metadata={r["case"]:sha(Path(r["task_store"])/r["task_id"]/"task.json")for r in old["cases"]})
 groups=values=0
 for item in old["cases"]:
  request=read(item["request"]);n=8 if item["case"]in REUSE else 4
  groups+=n;values+=n*sum(math.prod(o["shape"])for o in request["layout"]["outputs"])
 plan.update(planned_input_groups=groups,planned_output_values=values);plan["binding"]=digest(plan)
 verify(plan);write(PROPOSAL,plan);write(OUT/"ready-proposal.json",plan)
 print(json.dumps(dict(status="frozen_unapproved",binding=plan["binding"],cases=len(plan["cases"]),
  unique_requests=plan["unique_requests"],groups=groups,values=values,provider_calls=0,compiler_calls=0,encrypted_calls=0)))
if __name__=="__main__":main()
