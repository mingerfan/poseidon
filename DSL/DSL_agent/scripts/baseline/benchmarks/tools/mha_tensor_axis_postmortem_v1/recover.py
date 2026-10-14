"""Audit and recover a confirmed lost MHA worker without replaying an HTTP request."""
import hashlib,json,os,shlex,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(BASE/"benchmarks/tools/mha_tensor_axis_live_v1"))
import scope,run
def write_new(path,value):
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb")as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
def main():
    if not sys.argv[1:]:
        scope.verify();run.no_active()
        from hecate_python_env import VENV,enter_nix
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
        return enter_nix(command,seconds=45,keep_env=(),environment="poseidon-cpu")
    assert sys.argv[1:]==["--inside"]
    from hecate_python_env import VENV
    assert os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV
    from application_execution.context import WorkerContext
    from application_execution.backend import application
    from application_transport_receipt import snapshot
    proposal=scope.verify();run.no_active()
    plan=scope.read(scope.RUN/"plan.json")
    assert not run.same_process(plan["owner"]),"batch_still_running"
    boot=next(int(line.split()[1])for line in Path("/proc/stat").read_text().splitlines()if line.startswith("btime "))
    assert boot>plan["created"],"observed_reboot_required"
    raw_running=scope.read(scope.RUN/"report.json")
    assert raw_running["status"]=="running","raw_observation_expected"
    raw_report_sha=scope.sha(scope.RUN/"report.json")
    rows=[];received_usage=[]
    for name in scope.CASES:
        folder=scope.RUN/name;marker=scope.read(folder/"task-id.json");task=marker["task_id"];tasks=folder/"tasks"
        state_path=tasks/task/"state.json";before=scope.read(state_path)
        row={"case":name,"task":task,"state_before_sha256":scope.sha(state_path),
             "generations":before["generations"],"http_attempts":before["http_attempts"]}
        checks=[e for e in before["ledger"]if e.get("event")=="provider_checkpoint"]
        outcomes=checks[-1].get("outcomes",[])if checks else[]
        row["call_outcomes"]=[{"index":o["index"],"status":o["status"]}for o in outcomes]
        row["unknown_reserved_http_indices"]=[i for i in range(before["http_attempts"])
           if i>=len(outcomes)or outcomes[i].get("status")not in("failed","response_received")]
        validation=[]
        for path in sorted(state_path.parent.glob("validation-*.json")):
            data=scope.read(path);diag=data.get("failure")or{}
            validation.append({"round":int(path.stem.split("-")[-1]),"status":data["status"],
              "layer":diag.get("layer"),"reason_code":diag.get("reason_code"),"location":diag.get("location"),
              "context":diag.get("context",{}),"record_sha256":scope.sha(path)})
        row["validation"]=validation
        row["candidate_hashes"]=[]
        for response_path in sorted(state_path.parent.glob("response-*.json")):
            response=scope.read(response_path);candidate=json.loads(response["raw"])
            row["candidate_hashes"].append({"round":int(response_path.stem.split("-")[-1]),
              "response_sha256":scope.sha(response_path),
              "source_sha256":hashlib.sha256(candidate["hecate_source"].encode()).hexdigest()})
        metrics=scope.read(state_path.parent/"provider-metrics.json")
        for call in metrics.get("calls",[]):
            usage=call.get("usage")
            if type(usage)is dict:
                received_usage.append({k:usage[k]for k in("prompt_tokens","completion_tokens","total_tokens")if k in usage})
        sidecar=scope.read(state_path.parent/"provider-transport.json")
        assert sidecar==snapshot(sidecar),"transport_receipt_binding"
        row["transport_receipt_sha256"]=scope.sha(state_path.parent/"provider-transport.json")
        row["temp_states"]=[{"path":p.name,"sha256":scope.sha(p)}for p in state_path.parent.glob("state.json.*.tmp")]
        if before["status"]=="running":
            backup=state_path.parent/"state.before-recovery.json";assert not backup.exists()
            with os.fdopen(os.open(backup,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb")as f:
                f.write(state_path.read_bytes());f.flush();os.fsync(f.fileno())
            assert scope.sha(backup)==row["state_before_sha256"],"original_state_bytes_preserved"
            app=application(WorkerContext.capture(tasks));result=app.recover(task)
            after=scope.read(state_path)
            assert after["status"]=="service_error"and after["phase"]=="complete"
            assert after["failure"]=={"code":"interrupted_outcome_unknown","layer":"worker"}
            assert after["ledger"]==before["ledger"]and after["http_attempts"]==before["http_attempts"]
            assert after["generations"]==before["generations"]
            row.update(status="interrupted",terminal_task_status=after["status"],
              failure={"code":"interrupted_outcome_unknown","layer":"worker"},
              recovered=True,state_after_sha256=scope.sha(state_path))
        else:
            assert before["status"]=="budget_exhausted"
            row.update(status="failed",terminal_task_status=before["status"],failure=before["failure"],recovered=False)
        rows.append(row)
    totals=scope.totals(run.states(scope.RUN))
    assert totals=={"generations":5,"http_attempts":8,"calls_resolved":False}
    assert sum(len(r["unknown_reserved_http_indices"])for r in rows)==1
    assert len(rows[0]["candidate_hashes"])==4 and not rows[1]["candidate_hashes"]
    assert all(v["layer"]=="static_check"for r in rows for v in r["validation"])
    assert scope.sha(scope.RUN/"report.json")==raw_report_sha,"historical_running_report_changed"
    report={"format":"poseidon-mha-tensor-axis-live-postmortem-v1","status":"stopped_interrupted",
       "proposal_binding":proposal["binding"],"plan_binding":plan["binding"],
       "source_stage_identity_binding":scope.digest(proposal["source_stage_identity"]),
       "frozen_sources_verified_before_changes":True,
       "original_running_report_sha256":raw_report_sha,"original_running_report_preserved":True,
       "actual_owner_absent":True,"guest_reboot_after_batch_start":True,
       "planned":2,"passed":0,"failed":1,"interrupted":1,"not_started":0,
       **totals,"unknown_http_attempts":1,"received_usage":received_usage,
       "compiler_calls":0,"encrypted_calls":0,"decryptions":0,"numerical_checks":0,
       "packages_delivered":0,"new_paid_calls_after_recovery":0,"rows":rows,
       "billing_amount_known":False,"automatic_replay_allowed":False,
       "authorization_closed":True,"resources_and_parameters_changed":False}
    report["binding"]=scope.digest(report)
    write_new(ROOT/"docs/baseline/agent-mha-tensor-axis-live-results-r1335.json",report)
    print(json.dumps({k:report[k]for k in("status","binding","planned","passed","failed","interrupted","generations",
       "http_attempts","calls_resolved","unknown_http_attempts","compiler_calls","encrypted_calls","new_paid_calls_after_recovery")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
