"""Output-tree controls and historical-data diagnosis; no new FHE/API calls."""
import os,sys,shlex,json,time,unittest,traceback,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";sys.path.insert(0,str(BASE))
from hecate_python_env import enter_nix,VENV
def main():
    if "--inside"not in sys.argv:
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
        return enter_nix(command,seconds=120,keep_env=(),environment="poseidon-cpu")
    import numpy as np
    from benchmark_graph import require,digest
    from candidate_bundle import check_execution_dependencies
    from program_execution import package_execution_metadata,named_outputs
    from pytorch_source_contract import restore_outputs
    from application_identity import current_identity_manifest
    require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned output controls")
    start=time.time()
    suite=unittest.TestSuite()
    for name in ["test_program_output_tree_contract","test_program_execution","test_source_precision_reference"]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    summary=dict(format="poseidon-public-expression-output-controls-v1",tests=tests.testsRun,
       passed=tests.testsRun-len(tests.failures)-len(tests.errors)-len(tests.skipped),
       failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped),
       provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_calls=0,
       reused_historical_fhe_data=True,seconds=time.time()-start)
    if tests.wasSuccessful():
        case=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-expression-v23-cpu-v1/named")
        q=json.loads((case/"qualification.json").read_text())
        package=case/"tasks"/q["task_id"]/"program"
        manifest=json.loads((package/"manifest.json").read_text())
        request=json.loads((package/"request.json").read_text())
        metadata=package_execution_metadata(package,manifest,request["model"])
        require(metadata["output_structure_policy"]=="poseidon-mha-ordered-source-v1","Inherited bound output policy")
        failed=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-program-execution-f6nw93ta")
        flat=np.load(failed/"output/decrypted.npy",allow_pickle=False)
        outputs=named_outputs(request,flat,3)
        restored=[restore_outputs(v,metadata["output_structure"],allow_none=True)for v in outputs]
        require(len(restored)==3 and all(list(v)==["primary","saved"]and v["saved"][1]is None for v in restored),
                "Real historical None-containing tree restored")
        try:check_execution_dependencies(package)
        except ValueError as error:summary["old_execution_identity_rejected"]=True;summary["old_identity_rejection"]=str(error)
        else:raise ValueError("Changed execution implementation cannot reuse the old execution binding")
        summary.update(historical_groups_restored=3,historical_values_restored=int(flat.size),
          historical_decrypted_sha256=hashlib.sha256((failed/"output/decrypted.npy").read_bytes()).hexdigest(),
          current_sdk_identity=current_identity_manifest()["manifest_sha256"])
    summary["binding"]=digest(summary)
    with(HERE/"controls.json").open("x")as f:json.dump(summary,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(summary))
    return 0 if tests.wasSuccessful()else 1
if __name__=="__main__":raise SystemExit(main())
