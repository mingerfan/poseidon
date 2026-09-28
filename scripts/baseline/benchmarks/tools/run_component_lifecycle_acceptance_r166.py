"""Two isolated qualification workers and retained-answer provenance tests; no API."""
import hashlib,json,os,shlex,subprocess,sys,time,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    from hecate_python_env import enter_nix,VENV
    if "--inside" not in sys.argv:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=600)
    from workspace_paths import RESULTS
    from benchmark_graph import digest
    from benchmark_runner import dump
    from candidate_bundle import implementation
    output=RESULTS/"component-integration-r161/lifecycle-r166";output.mkdir()
    frozen=implementation()
    sys.path.insert(0,str(Path(__file__).parent))
    import test_stage2_agent_provenance as tests
    from agent_response_provenance import verify_answers
    tests.verify_answers=verify_answers
    class CurrentProvenance(tests.AgentProvenance):
        @classmethod
        def setUpClass(cls):
            job=json.loads((RESULTS/"component-integration-r161/acceptance-r163/native/job.json").read_text())
            cls.spec=dict(request_id=job["request"]["request_id"],request=job["request"],id="current-native")
            cls.source=job["candidate"]["hecate_source"]
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CurrentProvenance)
    with (output/"provenance-tests.log").open("x") as log:
        tested=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    assert tested.wasSuccessful() and not tested.skipped
    rows=[];running=[];logs=[];peak_rss=0;minimum_available=10**12
    start=time.monotonic()
    try:
        for name in ("native","multioutput"):
            job=RESULTS/"component-integration-r161/acceptance-r163"/name/"job.json"
            log=(output/(name+".log")).open("x");logs.append(log)
            proc=subprocess.Popen([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/agent_component.py"),
                                   "--job",str(job)],cwd=output,stdout=subprocess.PIPE,stderr=log,text=True)
            running.append((name,proc))
        while any(p.poll() is None for _,p in running):
            assert time.monotonic()-start<300,"Bounded concurrency batch timeout"
            data=[]
            for procdir in Path("/proc").iterdir():
                if not procdir.name.isdigit():continue
                try:
                    fields=dict(line.split(":",1) for line in (procdir/"status").read_text().splitlines())
                    data.append((int(fields["Pid"]),int(fields["PPid"]),int(fields.get("VmRSS","0 kB").split()[0])))
                except (FileNotFoundError,ProcessLookupError,PermissionError):
                    continue
            owned={p.pid for _,p in running}
            for _ in range(16):
                new=owned|{pid for pid,parent,rss in data if parent in owned}
                if new==owned:break
                owned=new
            peak_rss=max(peak_rss,sum(rss for pid,_,rss in data if pid in owned))
            avail=int(next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")).split()[1])
            minimum_available=min(minimum_available,avail)
            assert avail>=1024**2,"Guest memory reserve exhausted"
            time.sleep(.1)
        for name,proc in running:
            stdout,_=proc.communicate(timeout=10);result=json.loads(stdout)
            dump(output/(name+".json"),result)
            assert proc.returncode==0 and result["numerically_validated"] and result["encrypted_execution"]
            evidence=Path(result["evidence"])
            assert not (evidence/"private-keys").exists()
            rows.append(dict(case=name,result=result,report_sha256=sha(evidence/"report.json")))
        assert len({r["result"]["evidence"] for r in rows})==2
        assert len({r["result"]["request_id"] for r in rows})==2
        assert frozen==implementation()
    finally:
        for _,proc in running:
            if proc.poll() is None:proc.terminate()
        for _,proc in running:
            if proc.poll() is None:proc.wait(timeout=30)
        for log in logs:log.close()
    report=dict(format="poseidon-component-lifecycle-r166",rows=rows,passed=2,failed=0,
        max_workers=2,compiler_build_jobs=0,peak_process_tree_rss_kib=peak_rss,
        minimum_guest_available_kib=minimum_available,seconds=time.monotonic()-start,
        provenance_tests=dict(run=tested.testsRun,failed=len(tested.failures),errors=len(tested.errors),skipped=len(tested.skipped)),
        implementation=frozen,new_paid_calls=0,new_agent_generation=False)
    report["binding"]=digest(report);dump(output/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","implementation")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
