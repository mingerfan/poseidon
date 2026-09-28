"""Final targeted qualification-stage regression; no provider or SDK rebuild."""
import argparse,hashlib,json,shlex,subprocess,sys,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
    from hecate_python_env import enter_nix,VENV
    p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
    if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=600)
    from workspace_paths import RESULTS
    from semantic_benchmark_execution import runtime_sources
    from benchmark_runner import dump
    from benchmark_graph import digest
    from candidate_bundle import export_bundle
    root=RESULTS/"validation-adapter-r167";out=root/"api-final";out.mkdir()
    frozen=runtime_sources()
    names="test_validation_adapter test_agent_component test_candidate_pipeline test_candidate_bundle test_typed_witness_repairs test_unified_public_coverage test_unified_public_arithmetic test_unified_public_unary test_unified_public_storage test_unified_public_views test_unified_native_coverage test_upstream_helper_coverage test_portability test_result_retention test_compiler_configuration".split()
    with (out/"tests.log").open("x") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    tests=dict(run=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
               failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
               skip_reasons=[dict(test=str(t),reason=s) for t,s in result.skipped])
    dump(out/"tests.json",tests)
    assert result.wasSuccessful()
    rows=[]
    for name in ("construct_083_2","construct_116_0","unified-native-copy-2"):
        folder=out/name;folder.mkdir()
        job=root/"acceptance-final"/name/"job.json"
        with (folder/"worker.log").open("x") as log:
            proc=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/agent_component.py"),"--job",str(job)],stdout=subprocess.PIPE,stderr=log,text=True,timeout=300)
        response=json.loads(proc.stdout);dump(folder/"result.json",response)
        if name=="construct_083_2":
            assert not response["encrypted_execution"] and response["observed_stages"]["compiled"]
            assert response["failure"]["layer"]=="artifact_gate"
        elif name=="construct_116_0":
            assert response["encrypted_execution"] and not response["numerically_validated"]
            assert response["status"]!="passed" and response["observed_stages"]["numerically_compared"]
        else:
            assert response["status"]=="passed" and response["numerically_validated"]
            manifest=export_bundle(Path(response["evidence"]),out/"composition-bundle")
            with (out/"bundle-replay.log").open("x") as log:
                replay=subprocess.run([str(VENV/"bin/python"),"-B",str(ROOT/"scripts/dsl_bundle.py"),"replay",
                    "--bundle",str(out/"composition-bundle"),"--execute"],stdout=log,stderr=subprocess.STDOUT,timeout=300)
            assert replay.returncode==0
        rows.append(dict(name=name,result=response))
        dump(out/"progress.json",dict(rows=rows))
    assert frozen==runtime_sources()
    guard=json.loads((root/"before.json").read_text())["compiler"]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in guard.items())
    old=json.loads((root/"acceptance-final/report.json").read_text())["source_hashes"]
    changed=[p for p in set(old)|set(frozen) if old.get(p)!=frozen.get(p)]
    assert set(changed)=={"scripts/baseline/component_backend.py","scripts/baseline/test_validation_adapter.py"}
    report=dict(format="poseidon-validation-adapter-api-r171",tests=tests,rows=rows,
        expected_failures_retained=2,composition_bundle_replay="passed",composition_bundle_binding=manifest["binding"],
        compiler_files_unchanged=len(guard),source_hashes=frozen,source_delta_since_matrix=changed,
        new_paid_calls=0,new_agent_generation=False)
    report["binding"]=digest(report);dump(out/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","source_hashes")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
