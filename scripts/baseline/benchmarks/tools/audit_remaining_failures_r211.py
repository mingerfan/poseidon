"""Read-only parent verification and a fresh layered closure report. No API."""
import sys,json,hashlib,collections,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from earth_failure_diagnostic import diagnose
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
DEST=ROOT/"docs/baseline/stage2-remaining-closure-r211.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def main():
    if "--verify" in sys.argv:
        d=json.loads(DEST.read_text());binding=d.pop("binding");assert digest(d)==binding
        for group in ("parents","compiler_guard","source_hashes"):
            assert all(sha(p)==v for p,v in d[group].items()),group
        print(json.dumps(dict(verified=True,remaining=d["remaining_agent_failures"],binding=binding)));return
    if DEST.exists():raise ValueError("Refuse to overwrite closure")
    parents={}
    def read(p):
        parents[str(p)]=sha(p);return json.loads(Path(p).read_text())
    old=read(ROOT/"docs/baseline/stage2-expression-closure-r205.json")
    originals={r["id"]:r for r in old["rows"] if r["classification"]!="passed"}
    a=read(R/"stage2-expression-repairs-r206/report.json")
    b=read(R/"stage2-expression-repairs-r207/report.json")
    c=read(R/"stage2-expression-repairs-r210/report.json")
    # Keep the rejected experiments as explicit parents, never replace their evidence.
    read(R/"stage2-expression-repairs-r208/report.json")
    read(R/"stage2-expression-repairs-r209/report.json")
    unit=read(R/"stage2-expression-repairs-r210/unit.json")
    compatibility=read(R/"stage2-expression-repairs-r210/compatibility.json")
    regression=read(R/"expression-repairs-r208-linear-mlp-silu/report.json")
    before=read(R/"stage2-expression-repairs-r206/before.json")
    assert (unit["failures"],unit["errors"],unit["skipped"])==(0,0,0)
    witnesses={}
    for report in (a,b,c):
        for row in report["rows"]:
            if row.get("plaintext",{}).get("passed") and row.get("result",{}).get("failure",{}).get("layer")=="compiler":
                witnesses[row["id"]]=row
    assert set(witnesses)==set(originals) and len(witnesses)==26
    rows=[];profile=(ROOT/"third_party/dacapo/profiled_SEAL_CPU.json").read_bytes()
    for task,row in sorted(witnesses.items()):
        job=read(Path(row["job"]));assert job["request"]["model"]==originals[task]["public_model"]
        assert hashlib.sha256(job["candidate"]["hecate_source"].encode()).hexdigest()==row["source_sha256"]
        evidence=Path(row["result"]["evidence"])
        attempt=read(evidence/"attempt-00/report.json")
        log=evidence/"attempt-00/compile.log";parents[str(log)]=sha(log)
        cause=diagnose(log.read_bytes()[-4000:].decode("utf-8",errors="replace"),profile)
        assert cause and cause["failed_conditions"]==["compiler_accumulated_scale_budget"],task
        assert row["result"]["observed_stages"]["traced"] and not attempt["compiled"]
        assert not attempt["executed"]
        rows.append(dict(id=task,historical_agent_terminal=originals[task]["classification"],
                         manual_static_passed=True,manual_plaintext=row["plaintext"],
                         manual_expression=row["strategy"],manual_evidence=str(evidence),
                         manual_job=row["job"],compiler_diagnosis=cause,
                         agent_retested=False,new_agent_pass=False))
    guard=before["compiler"]
    assert all(sha(p)==v for p,v in guard.items())
    current={str(p):sha(p) for p in BASE.rglob("*.py")}
    prior={str(ROOT/Path(p)):v for p,v in before["source_hashes"].items()}
    changes=dict(modified=[p for p in prior if p in current and prior[p]!=current[p]],
                 new=[p for p in current if p not in prior],deleted=[p for p in prior if p not in current])
    assert not changes["deleted"]
    for path in (ROOT/"third_party/dacapo/python/poly/poly/Func.py",
                 ROOT/"third_party/dacapo/lib/Dialect/Earth/IR/EarthDialect.cpp"):
        parents[str(path)]=sha(path)
    report=dict(format="poseidon-stage2-remaining-closure-r211",remaining_agent_failures=26,
        historical_agent_terminal_counts=dict(collections.Counter(r["historical_agent_terminal"] for r in rows)),
        manual_models_with_static_and_16_group_plaintext_witness=26,
        manual_compiler_budget_rejections=26,new_agent_passes=0,paid_calls=0,
        unit_tests={k:unit[k] for k in ("run","failures","errors","skipped")},
        old_requests_exact=compatibility["old_requests_exact"],
        real_fhe_regressions=regression["rows"],regression_binding_note="r208 runtime binding; final r210 change affects compiler-failure feedback only",
        upstream_bootstrap_locations=["poly/Func.py:26-41","poly/Func.py:74-81"],
        backend_note="Fixed upstream helpers use bootstrap. No real SEAL HEVM bootstrap is available. This is not proof that every equivalent unbootstrapped expression is impossible.",
        own_probe_defects_fixed=["wide padded concat materialized modulo P before negate",
                                "new structured feedback field rejected by the existing provider whitelist; structure is now local only"],
        own_launcher_failure="r210 first probe launch used wrong work root; Nix offline preflight rejected it before compilation. stable.log retained; corrected run uses stable-retry.log.",
        compiler_modified=False,models_modified=False,reference_modified=False,security_parameters_modified=False,
        atol=1e-5,rtol=1e-4,x86_executed=False,automatic_paid_retry=False,
        local_changes=changes,parents=parents,compiler_guard=guard,source_hashes=current,rows=rows)
    report["binding"]=digest(report);DEST.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("remaining_agent_failures","manual_models_with_static_and_16_group_plaintext_witness","unit_tests","old_requests_exact","binding")}))
if __name__=="__main__":main()
