"""Developer-only activation at a verified post-batch boundary."""
import sys,json,subprocess,time,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
SCOPE=BASE/"benchmarks/tools/generic_capacity_backend_v10"
POST=BASE/"benchmarks/tools/generic_capacity_post_audit_v1"
sys.path[:0]=[str(BASE),str(SCOPE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import qualification_scope as q
import run as live

def proposal():
    p=q.read(HERE/"activation-scope.json");q.bound(p)
    for path,want in p["frozen_files"].items():
        q.require(q.sha(path)==want,"activation_scope_changed:"+path)
    draft=q.read(HERE/"draft.json");q.bound(draft)
    for name,want in draft["base_sources"].items():
        q.require(q.sha(BASE/name)==want,"changed_target:"+name)
    checks=q.read(HERE/"controls-v3.json");q.bound(checks)
    q.require(checks["status"]=="passed"and checks["tests"]==25 and checks["patch_apply_check"]==0,
        "prototype controls required")
    return p,draft

def pinned(script,log,seconds=180):
    from hecate_python_env import VENV,enter_nix
    q.require(not log.exists(),"preserve_activation_log")
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(script)])
    with log.open("x")as out:
        # enter_nix owns its bounded subprocesses; shell quoting covers the fixed log path.
        redirect=command+" > "+shlex.quote(str(log))+" 2>&1"
    return enter_nix(redirect,seconds=seconds,environment="poseidon-cpu")

p,draft=proposal()
if "--activate" not in sys.argv:
    print(json.dumps(dict(status="prepared",scope_binding=p["binding"],
        targets=list(draft["base_sources"]),mutation_performed=False,
        requirement="no active batch; both queued readonly audits passed; pinned prototype controls passed")))
    raise SystemExit(0)
q.require(set(sys.argv[1:])=={"--activate"},"activation mode")
live.no_active()
post=q.read(POST/"after-batch-report.json");q.bound(post)
q.require(post["status"]=="passed"and len(post["commands"])==2 and
    all(c["returncode"]==0 for c in post["commands"]),"readonly audit must finish")
for name in("after-batch-owner.json",):
    owner=q.read(POST/name);v=live.process_info(owner["pid"])
    q.require(v is None or v["start"]!=owner["start"]or v["state"]in("Z","X"),
        "readonly worker still live")
terminal_plan=q.verify("live");q.require(terminal_plan["binding"]==p["live_plan_binding"],"parent plan")
root=Path(terminal_plan["campaign"])
q.require((root/"terminal.json").exists(),"missing original terminal")
q.require(q.calls(q.states(root),52,208)["calls_resolved"],"unsettled original HTTP")
started=time.monotonic();checks=[]
pre=pinned(HERE/"related_tests.py",HERE/"pinned-proposal-controls.log")
checks.append(dict(phase="pinned_proposal_static",returncode=pre))
q.require(pre==0,"pinned proposal regression failed; production preserved")
live.no_active();proposal();q.verify("live")
branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
q.require(branch=="lhy-agent-dsl","development branch")
patch=HERE/"candidate-static-diagnostics.patch"
subprocess.run(["git","apply","--check",str(patch)],cwd=ROOT,check=True)
subprocess.run(["git","apply",str(patch)],cwd=ROOT,check=True)
for name,want in draft["proposed_sources"].items():
    q.require(q.sha(BASE/name)==want,"activated target differs")
record=dict(format="poseidon-native-static-diagnostic-activation-v1",
    scope_binding=p["binding"],base_sources=draft["base_sources"],
    activated_sources=draft["proposed_sources"],patch_sha256=q.sha(patch),
    old_sdk=p["old_sdk"],readonly_audit_binding=post["binding"],source_policy_changed=False,
    new_provider_calls=0,new_compiler_calls=0,new_encrypted_calls=0,
    runtime_compiler_or_library_changed=False)
record["binding"]=q.digest(record);q.write(HERE/"activation.json",record)
direct=subprocess.run([sys.executable,"-B",str(HERE/"test_diagnostics.py"),"-v"],
    capture_output=True,text=True,timeout=45)
(HERE/"activated-focused-controls.log").write_text(direct.stdout+direct.stderr)
checks.append(dict(phase="production_focused_static",tests=25,returncode=direct.returncode))
postcode=pinned(HERE/"related_tests_baseline.py",HERE/"pinned-activated-controls.log")
checks.append(dict(phase="production_related_static",returncode=postcode))
from application_backend import PoseidonBackend
identity=PoseidonBackend().identity_manifest()
result=dict(format="poseidon-native-static-diagnostic-activation-controls-v1",
    status="passed"if all(c["returncode"]==0 for c in checks)else"failed",
    activation_binding=record["binding"],new_sdk=identity["manifest_sha256"],
    identity=identity,checks=checks,wall_seconds=time.monotonic()-started,
    provider_calls=0,compiler_calls=0,encrypted_calls=0,
    actual_encrypted_regression="separate frozen replay scope still required")
result["binding"]=q.digest(result);q.write(HERE/"activation-controls.json",result)
print(json.dumps({k:v for k,v in result.items()if k!="identity"}))
raise SystemExit(0 if result["status"]=="passed"else 1)
