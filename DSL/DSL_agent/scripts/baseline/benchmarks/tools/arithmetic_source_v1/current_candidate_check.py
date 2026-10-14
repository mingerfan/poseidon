"""Read-only comparison of frozen rule candidates with current Agent lowering."""
import hashlib,json,pathlib,sys,os,shlex
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def require(ok,code):
 if not ok:raise ValueError(code)
def inside():
 from unified_graph_lowering import candidate_source
 from application_identity import current_identity_manifest
 BASE=pathlib.Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
 old=BASE/"agent-exact-arithmetic-v21-source-acceptance-v1"
 current=BASE/"agent-exact-arithmetic-v21-mask-support-v1"
 identity=current_identity_manifest();require(identity==read(current/"identity.json"),"current_backend_identity_changed")
 rows=[]
 for phase in [old,current]:
  report=read(phase/"report.json")
  for r in report["rows"]:
   if r["status"]!="passed":continue
   request=read(phase/"tasks"/r["task_id"]/"task.json")["request"]
   generated=candidate_source(request)[0]
   actual=hashlib.sha256(generated.encode()).hexdigest()
   require(sha(phase/r["case"]/"rule_candidate.py")==r["candidate_sha256"],"historical_candidate_changed")
   rows.append(dict(case=r["case"],phase=phase.name,old_candidate_sha256=r["candidate_sha256"],
                    current_candidate_sha256=actual,source_unchanged=actual==r["candidate_sha256"],
                    backend_identity_unchanged=identity==read(phase/"identity.json")))
 result=dict(format="poseidon-exact-arithmetic-current-candidate-check-v1",status="checked",rows=rows,
             source_sha256=sha(__file__),schedule_sha256=sha(ROOT/"scripts/baseline/capacity_algebraic_schedule.py"),
             compiler_calls=0,encrypted_calls=0,provider_calls=0)
 with(HERE/"current-candidate-check.json").open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
 print(json.dumps(dict(checked=len(rows),unchanged=sum(r["source_unchanged"]for r in rows),
                      changed=[r["case"]for r in rows if not r["source_unchanged"]])))
 return 0
if __name__=="__main__":
 if sys.argv[1:]==["--inside"]:raise SystemExit(inside())
 os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(pathlib.Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=90,keep_env=(),environment="poseidon-cpu"))
