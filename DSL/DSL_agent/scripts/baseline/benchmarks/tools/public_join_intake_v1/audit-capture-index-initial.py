"""Bind successful isolated captures to the common task-preparation index."""
import sys,shlex,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
if "--inside"not in sys.argv:
 live.no_active()
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=180,environment="poseidon-cpu"))
from candidate_bundle import load_model_provenance
from benchmark_graph import digest
from qualification_scope import read,write,bound,sha
plan=read(HERE/"capture-plan.json");bound(plan)
report=read(Path(plan["campaign"])/"report.json");bound(report)
assert report["passed"]==21 and report["failed"]==report["skipped"]==report["not_run"]==0
rows=[];groups=values=0;facts=[]
for case in read(HERE/"cases.json"):
 name=case["name"];out=Path(plan["campaign"])/name
 row=next(r for r in report["cases"]if r["case"]==name)
 assert row["status"]=="passed"and row["capture"]["reference_verified"]is True
 model=read(out/"model.json");provenance=load_model_provenance(out,model,max_bytes=4*1024**2)
 ref=provenance["reference_report"]["finite_reference"]
 assert ref["passed"]is True
 groups+=row["groups"];values+=row["values"]
 rows.append(dict(case,model_import=str(out)))
 facts.append(dict(case=name,model_sha256=digest(model),provenance_digest=digest(provenance),
  reference_report_sha256=sha(out/"reference-report.json"),groups=row["groups"],values=row["values"]))
write(HERE/"capture-index.json",rows)
audit=dict(format="poseidon-public-join-capture-provenance-audit-v1",status="passed",plan_binding=plan["binding"],
 report_binding=report["binding"],sdk=plan["sdk"],cases=facts,input_groups=groups,output_values=values,
 uploaded_code_executed_by_audit=False,compiler_calls=0,encrypted_calls=0,provider_calls=0)
audit["binding"]=digest(audit);write(HERE/"capture-audit.json",audit)
print(json.dumps({k:audit[k]for k in("status","input_groups","output_values","binding")}))
