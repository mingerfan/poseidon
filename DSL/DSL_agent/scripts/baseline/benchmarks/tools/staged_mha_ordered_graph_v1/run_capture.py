"""Finite isolated ordered-source capture; no compiler, Provider or backend calls."""
import hashlib,json,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE/"overlay"))
from benchmark_graph import canonical,digest,require
from pytorch_ordered_contract import source_identity
from pytorch_ordered_intake import run
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
 fixtures=HERE/"reference_tests";index=json.loads((fixtures/"source_fixtures/index.json").read_text())
 require(index["binding"]==digest({k:v for k,v in index.items() if k!="binding"}),"Fixture index binding")
 before=source_identity();start=time.monotonic()
 report=dict(format="poseidon-mha-ordered-source-capture-audit-v1",cases=[],implementation=before,
   fixture_binding=index["binding"],api_calls=0,compiler_calls=0,encrypted_executions=0,production_applied=False)
 def save():
  report.update(elapsed_seconds=time.monotonic()-start,
   passed=sum(c["status"]=="passed" for c in report["cases"]),
   failed=sum(c["status"]=="failed" for c in report["cases"]),skipped=0,
   not_run=len(index["cases"])-len(report["cases"]))
  report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
  (root/"report.json").write_bytes(canonical(report))
 for case in index["cases"]:
  require(time.monotonic()-start<480,"Source capture batch time budget")
  require(source_identity()==before,"Capture implementation changed")
  folder=fixtures/case["manifest"];item=dict(id=case["id"],configuration_kind=case["configuration_kind"])
  try:
   require(sha(folder)==case["manifest_sha256"],"Fixture manifest changed")
   item["capture"]=run(folder,root/case["id"],resource_profile="general-tensors-v1",trusted=True)
   r=json.loads((root/case["id"]/"reference-report.json").read_text())
   f=r["finite_reference"]
   item.update(status="passed",groups=f["groups"],compared_values=f["compared_values"],
    max_ideal_error=f["max_original_torch_vs_independent_ideal"],
    max_decomposition_error=f["max_decomposition_error"],max_approximation_error=f["max_approximation_error"],
    source_hashes=r["source_hashes"],reference_binding=f["binding"],
    artifacts={str(p.relative_to(root/case["id"])):sha(p) for p in (root/case["id"]).rglob("*") if p.is_file()})
  except Exception as error:
   item.update(status="failed",error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
  report["cases"].append(item);save()
 require(source_identity()==before,"Capture implementation changed at close")
 report["source_frozen"]=True;report["status"]="passed" if report["failed"]==0 and report["not_run"]==0 else "failed";save()
 print(json.dumps({k:v for k,v in report.items() if k not in ("cases","implementation")}))
 for c in report["cases"]:
  if c["status"]=="failed":print(c["id"],c["error"])
 return 0 if report["status"]=="passed" else 1
if __name__=="__main__":raise SystemExit(main())
