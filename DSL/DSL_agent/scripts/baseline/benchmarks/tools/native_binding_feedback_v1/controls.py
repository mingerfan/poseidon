"""Affected static/feedback controls and replay of the unchanged failed candidate."""
import hashlib,json,os,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if not sys.argv[1:]:
        from hecate_python_env import VENV,enter_nix
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
          [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
        return enter_nix(command,seconds=90,keep_env=(),environment="poseidon-cpu")
    assert sys.argv[1:]==["--inside"]
    from application_identity import current_identity_manifest
    from candidate_contract import validate_candidate
    from candidate_check_diagnostic import CandidateCheckError
    from component_contract import failure
    from application_diagnostics import attach,repeat_key
    from unified_graph_contract import validate_request
    from benchmark_graph import digest
    before=current_identity_manifest()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
      "test_native_reserved_binding_feedback","test_native_diagnostics",
      "test_application_diagnostics","test_native_public_loops.NativeLoopTests"])
    start=time.monotonic()
    with (HERE/"controls.log").open("x")as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    assert result.wasSuccessful()and not result.skipped,"affected_controls"
    proposal=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json").read_text())
    report=json.loads((ROOT/"docs/baseline/agent-mha-tensor-axis-live-results-r1335.json").read_text())
    requests=[json.loads(Path(c["request_file"]).read_text())for c in proposal["cases"]]
    for c,r in zip(proposal["cases"],requests):
        assert sha(c["request_file"])==c["request_file_sha256"]
        validate_request(r)
    oldroot=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-tensor-axis-live-v1")
    tasks=oldroot/proposal["cases"][0]["id"]/"tasks"
    response=next(tasks.glob("*/response-1.json"));raw=json.loads(response.read_text());candidate=json.loads(raw["raw"])
    assert sha(response)==report["rows"][0]["candidate_hashes"][1]["response_sha256"]
    try:validate_candidate(candidate,requests[0])
    except CandidateCheckError as error:
        assert error.reason=="native_reserved_binding"and error.location==dict(kind="source",line=55,column=4)
        new=attach({"status":"failed","failure":failure("static_check",error)},candidate=candidate,request=requests[0])
        assert new["failure"]["reason_code"]==error.reason and repeat_key(new["failure"])is None
        replay={"source_sha256":hashlib.sha256(candidate["hecate_source"].encode()).hexdigest(),
          "source_modified":False,"reason_code":error.reason,"location":error.location,
          "controlled_public_message":new["public_feedback"]["diagnostic"].split("\nBinding and shape guidance:")[0]}
    else:raise AssertionError("original_invalid_candidate_accepted")
    protected=json.loads((ROOT/"docs/baseline/agent-v12-live-rebound-proposal-r1304.json").read_text())["protected_sources"]
    assert all(sha(ROOT/rel)==want for rel,want in protected.items())
    assert current_identity_manifest()==before,"control_sources_changed"
    data={"format":"poseidon-native-binding-feedback-controls-v1","status":"passed","tests_run":result.testsRun,
      "passed":result.testsRun,"failed":0,"errors":0,"skipped":0,"seconds":time.monotonic()-start,
      "historical_failed_source_replay":replay,"requests_preserved":2,"protected_files_preserved":len(protected),
      "identity":before,"source_stage_binding":digest(before),
      "generation_calls":0,"http_attempts":0,"compiler_calls":0,"encrypted_calls":0,"decryptions":0}
    data["binding"]=digest(data)
    with (HERE/"controls.json").open("x")as f:json.dump(data,f,sort_keys=True,indent=2);f.write(chr(10))
    print(json.dumps({k:v for k,v in data.items()if k!="identity"}))
    return 0
if __name__=="__main__":raise SystemExit(main())
