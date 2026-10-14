"""Audit original and affected-only real-identity admission evidence."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline";HERE=Path(__file__).resolve().parent
RESULTS=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
sys.path[:0]=[str(HERE.parent/"staged_common_fused_integration_v1"),str(BASE)]
import check
from application_identity import check_manifest
from benchmark_graph import digest,canonical
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def methods(raw):
    tree=ast.parse(raw);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ApplicationAdmissionTests")
    return {n.name:digest(ast.dump(n,include_attributes=False)) for n in cls.body if isinstance(n,ast.FunctionDef)}
changed="test_missing_weight_and_wrong_source_hash_are_public_input_failures"
old=RESULTS/"agent-source-component-admission-r1289"
new=RESULTS/"agent-source-component-admission-r1290"
original,current=read(old/"report.json"),read(new/"report.json")
require(original["tests"]==8 and original["failed"]==0 and original["errors"]==3 and original["skipped"]==0
        and original["status"]=="failed","Retained original failure")
require(current["tests"]==1 and current["suite"]=="integrity" and current["status"]=="passed" and
        current["failed"]==current["errors"]==current["skipped"]==0,"Affected-only success")
previous=methods((old/"source-snapshot/inside.py").read_text())
latest=methods((HERE/"inside.py").read_text())
retest=methods((new/"source-snapshot/inside.py").read_text())
require(latest==retest and len(latest)==8 and
        {name for name in latest if latest[name]!=previous[name]}=={changed},"Only rejected-input fixture method changed")
require((old/"source-snapshot/inside.py").read_text().split("class ApplicationAdmissionTests")[0]==
        (HERE/"inside.py").read_text().split("class ApplicationAdmissionTests")[0],"Shared admission harness unchanged")
rawlog=(old/"tests.log").read_text()
require(rawlog.count("ERROR: "+changed)==3 and "FAIL:" not in rawlog,"Original errors confined to corrected fixture")
for name in set(latest)-{changed}:require(name+" (__main__.ApplicationAdmissionTests) ... ok" in rawlog,"Original method passed")
verified={}
identities=[]
for folder,report in ((old,original),(new,current)):
    receipt=read(folder/"sandbox.json")
    require(receipt["exit_code"]==(1 if folder==old else 0) and receipt["combined_sources"]==706
            and receipt["clearenv"] is True and receipt["unshare_all"] is True
            and receipt["implementation_overlay_readonly"] is True,"Actual sandbox scope")
    require(report["source_models"]==9 and report["production_applied"] is False and
            report["source_factory_calls"]==report["compiler_calls"]==report["encrypted_executions"]==
            report["provider_calls"]==report["new_dependencies"]==0 and report["subprocess_attempts"]==[],"Admission-only scope")
    require(report["runtime"]==dict(python=[3,10,14],numpy="1.25.2",torch="2.0.1",cuda=None),"Pinned SDK")
    for path,want in receipt["retained_hashes"].items():
        require(sha(path)==want,"Captured source/request evidence changed");verified[path]=want
    for path,want in receipt["source_hashes"].items():
        relative=Path(path).relative_to(HERE)
        require(sha(folder/"source-snapshot"/relative)==want,"Captured test harness changed")
        verified[str(folder/"source-snapshot"/relative)]=want
    identity=check_manifest(read(folder/"application-identities.json"))
    require(identity["manifest_sha256"]==report["identity_sha256"] and
            digest(identity["backend"])==report["backend_identity"],"Real identity binding")
    require(identity["backend"]==read(folder/"plan.json")["backend_identity"],"Original backend unchanged")
    identities.append(identity)
require(canonical(identities[0])==canonical(identities[1]),"Same combined identity for original and affected runs")
records=original["records"]
require(len(records)==18 and len({r["case"] for r in records})==9,"Nine graph/source admission pairs")
require(sum(r["phase"]=="graph_admission" and r["status"]=="queued" for r in records)==9 and
        sum(r["phase"]=="source_snapshot" and r["status"]=="cancelled" for r in records)==9,"Actual admission/cancellation records")
boundary=check.check()
require(not boundary["active_processes"] and boundary["production_applied"] is False,"Idle unmodified production")
value=dict(format="poseidon-source-real-identity-admission-acceptance-v1",
 stage="staged_application_admission_verified",source_models=9,current_method_successes=8,
 evidence_mode="seven unchanged methods from retained failed run plus one corrected affected-only method",
 current_method_hashes=latest,original_failed_report=str(old/"report.json"),affected_report=str(new/"report.json"),
 retained_original=dict(status="failed",tests=8,errors=3,failed=0,skipped=0,reason="weightless ciphertext fixtures"),
 negative_contexts=dict(missing_weight=6,undeclared_weight=3,changed_source_hash=9),
 graph_admissions=9,source_snapshots=9,cancelled_source_tasks=9,
 identity_sha256=identities[0]["manifest_sha256"],combined_sources=706,combined_plan_binding=boundary["plan_binding"],
 verified_hashes=len(verified),source_factory_calls=0,compiler_calls=0,encrypted_executions=0,provider_calls=0,
 new_dependencies=0,production_applied=False,production_checks=704,protected_backend_checks=648,protected_reference_checks=6,
 verification_level="task admission and binding only",
 pending=["combined compiler and Poseidon numerical qualification","compiled V14 package reload and new input execution","new real Agent quality tests"],
 runtime=original["runtime"])
value["binding"]=digest(value)
if sys.argv[1:]==["--write"]:
    path=HERE/"acceptance.json";require(not path.exists(),"Fresh acceptance record");path.write_bytes(canonical(value))
elif sys.argv[1:]:raise SystemExit("Usage: audit.py [--write]")
print(json.dumps({k:value[k] for k in ("stage","source_models","current_method_successes","verified_hashes",
    "identity_sha256","binding","compiler_calls","encrypted_executions","provider_calls","production_applied")}))
