"""Recompile one unchanged historical Agent MHA source after diagnostic-only changes."""
import hashlib, importlib.util, json, os, shlex, shutil, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
ENGINE=ROOT/"scripts/baseline/benchmarks/tools/tensor_axis_budget_v1/backend_acceptance.py"
spec=importlib.util.spec_from_file_location("historical_axis_acceptance",ENGINE)
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
engine.OUT=engine.RESULTS/"agent-native-binding-feedback-acceptance-v1"
SCOPE=engine.SCOPE
STAGES=("generation","compilation","execution","validation")
def outer():
    require=engine.require
    require(not engine.OUT.exists(),"fresh_output_directory")
    controls=engine.read(HERE/"controls.json")
    require(controls["status"]=="passed" and controls["passed"]==39
        and controls["failed"]==controls["errors"]==controls["skipped"]==0,"controls_required")
    ancestors=set();pid=os.getpid()
    while pid>0 and pid not in ancestors:
        ancestors.add(pid)
        try:pid=int((Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()[1])
        except (OSError,ValueError,IndexError):break
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if int(p.name) in ancestors:continue
            args=[x.decode(errors="replace") for x in (p/"cmdline").read_bytes().split(b"\0") if x]
        except (OSError,ValueError):continue
        require(not any(x.endswith(("/candidate_worker.py","/candidate_trace.py",
            "/mha_tensor_axis_live_v1/run.py","/native_binding_feedback_v1/backend_acceptance.py"))
            for x in args),"no_conflicting_offline_worker")
    require(engine.measure(engine.RESULTS)<=SCOPE["global_bytes"]-SCOPE["new_bytes"]
        and shutil.disk_usage(engine.WORK).free>=SCOPE["reserve_bytes"]+SCOPE["new_bytes"],"offline_space_boundary")
    sources={}
    for stage in STAGES:
        for name,want in controls["identity"][stage]["sources"].items():
            path=ROOT/"scripts/baseline"/name
            require(engine.sha(path)==want,"controls_source_changed")
            sources[str(path.relative_to(ROOT))]=want
    for path in [ENGINE,Path(__file__).resolve(),HERE/"controls.json",
        ROOT/"scripts/baseline/test_native_reserved_binding_feedback.py",
        ROOT/"scripts/agent_assessment.py",
        ROOT/"scripts/baseline/application_assessment_entry/cli.py",
        ROOT/"scripts/baseline/application_assessment_entry/supervision.py"]:
        sources[str(path.relative_to(ROOT))]=engine.sha(path)
    history=[engine.ORIGINAL/n for n in ("request.json","candidate.py","manifest.json")]
    history += [ROOT/"docs/baseline/agent-mha-tensor-axis-live-results-r1335.json",
        ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json",
        ROOT/"docs/baseline/agent-mha-feedback-live-results-r1331.json"]
    plan=dict(format="poseidon-native-binding-feedback-offline-plan-v1",
        scope=SCOPE,cases=["historical_masked_mha"],sources=sources,
        controls_binding=controls["binding"],identity_manifest_sha256=controls["identity"]["manifest_sha256"],
        historical_files={str(p):engine.sha(p) for p in history},
        protected_sources_checked=engine.protected(),
        existing_results=[p.name for p in engine.RESULTS.iterdir()],
        created_wall=time.time(),deadline_wall=time.time()+SCOPE["wall_seconds"],
        reason="Verify the new controlled immutable-binding diagnostic using a real recompile and package reuse of unchanged historical Agent source.",
        authorization="Existing Agent-only implementation request; no API, installation, downloads, parameters or backend changes.")
    plan["binding"]=hashlib.sha256(engine.canonical(plan)).hexdigest()
    engine.OUT.mkdir(mode=0o700)
    engine.write(engine.OUT/"plan.json",plan)
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(engine.VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    return engine.enter_nix(cmd,seconds=max(1,int(plan["deadline_wall"]-time.time())),
        keep_env=(),environment="poseidon-cpu")
def inside():
    result=engine.inside()
    report=engine.read(engine.OUT/"report.json")
    plan=engine.read(engine.OUT/"plan.json")
    engine.require(report["identity_sha256"]==plan["identity_manifest_sha256"],"controlled_identity_mismatch")
    report["format"]="poseidon-native-binding-feedback-offline-report-v1"
    report["controls_binding"]=plan["controls_binding"]
    report["source_provenance"]="historical_real_agent_source_unmodified"
    report["current_agent_generation"]=False
    report["binding"]=hashlib.sha256(engine.canonical(report)).hexdigest()
    engine.write(engine.OUT/"report.json",report)
    return result
if __name__=="__main__":
    raise SystemExit(inside() if sys.argv[1:]==["--inside"] else outer())
