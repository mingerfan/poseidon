"""Read-only output delivery check of three current, already executed packages."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from application_backend import PoseidonBackend
from application_component import Application
from task_output_adapter import TaskOutputAdapter
from model_output_contract import restore
from hecate_python_env import VENV
work=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
source=work/"agent-common-pytorch-backend-r1082";out=work/"agent-current-output-package-read-only-r1090"
require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned read-only context")
report=json.loads((work/"agent-common-pytorch-reuse-continuation-r1091/report.json").read_text())
require(report["status"]=="complete","Current full offline evidence")
plan=json.loads((work/"agent-common-pytorch-reuse-continuation-r1091/plan.json").read_text())
selected=[r for name in plan["cases"] for r in report["rows"] if r["id"]==name["id"] and r["status"]=="passed" and r.get("reuse_status")=="passed"][:3]
require(len(selected)==3 and not out.exists(),"Three current reused packages and fresh evidence")
out.mkdir(mode=0o700);started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory=lambda p:{str(f.relative_to(p)):sha(f) for f in p.rglob("*") if f.is_file()}
class ReadOnlyBackend(PoseidonBackend):
 def qualify(self,*args,**kwargs):raise AssertionError("No compilation in read-only delivery audit")
 def export(self,*args,**kwargs):raise AssertionError("No package mutation in read-only delivery audit")
 def execute(self,*args,**kwargs):raise AssertionError("No ciphertext execution in read-only delivery audit")
rows=[]
for old in selected:
 require(time.monotonic()-started<45,"Read-only delivery bound")
 task=source/"tasks"/old["task_id"];before=inventory(task);metadata=json.loads((task/"task.json").read_text())
 app=Application(task.parent,ReadOnlyBackend());result=app.result(task.name)
 require(result["status"]=="succeeded","Current actual application result")
 names=[o["name"] for o in metadata["request"]["model"]["outputs"]]
 tree=dict(kind="tuple",items=[*[dict(kind="tensor",name=name) for name in names],dict(kind="none")])
 adapter=TaskOutputAdapter(app);frame=adapter.bind(task.name,tree)
 require(frame["status"]=="succeeded" and "receipt" in frame,"Actual checked output receipt")
 receipt=json.loads(json.dumps(frame["receipt"]));reload=adapter.result(task.name,receipt,expected_binding=receipt["binding"])
 require(reload["status"]=="succeeded","JSON receipt reload")
 raw=json.loads((Path(old["new_input_runs"][0]["evidence"])/"report.json").read_text())
 require(raw["encrypted_execution"] and not raw["executor_secret_key"] and raw["agent_calls"]==raw["compiler_calls"]==0,"Retained actual ciphertext execution")
 contract=receipt["delivery"]["contract"]
 for sample in raw["outputs"]:
  restored=restore(contract,sample,expected_binding=receipt["delivery"]["contract_binding"])
  require(type(restored) is tuple and restored[-1] is None,"Explicit host output structure")
  for value,name in zip(restored[:-1],names):require(np.array_equal(np.asarray(value),np.asarray(sample[name])),"Exact decoded value preservation")
 require(inventory(task)==before,"Original package and task bytes unchanged")
 (out/(old["id"]+".receipt.json")).write_bytes(canonical(receipt))
 rows.append(dict(id=old["id"],task_id=task.name,package=old["package"],status="passed",
   receipt_binding=receipt["binding"],current_application_status=result["status"],receipt_reload_status=reload["status"],
   decoded_samples_restored=len(raw["outputs"]),raw_execution_evidence=old["new_input_runs"][0]["evidence"],
   package_hashes=before,new_compiler_calls=0,new_encrypted_executions=0,paid_calls=0,
   origin="current_rule_program; actual ciphertext outputs already audited",output_projection="explicit host tuple with static None"))
result=dict(format="poseidon-current-output-read-only-acceptance-v1",status="passed",passed=3,failed=0,not_run=0,
 rows=rows,original_offline_binding=report["binding"],seconds=time.monotonic()-started,
 new_compiler_calls=0,new_encrypted_executions=0,new_decryptions=0,new_agent_generations=0,paid_calls=0)
result["binding"]=digest(result);(out/"report.json").write_bytes(canonical(result));print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)
