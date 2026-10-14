"""Verify actual fixed helper reports; mathematical baseline passes do not count."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from benchmark_graph import digest,require
from benchmark_runner import DEFAULT,ROOT,load,strict_file,dump
from semantic_benchmark_execution import runtime_sources

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_manifest(folder):
    manifest=strict_file(folder/"evidence-manifest.json",4*1024**2)
    require(manifest.get("schema")==1 and manifest.get("private_keys_retained") is False,"Helper manifest identity")
    require(not (folder/"private-keys").exists(),"Intermediate keys were not cleaned")
    files=manifest.get("files")
    require(type(files) is dict and "report.json" in files and len(files)<=256,"Helper file manifest")
    for name,expected in files.items():
        relative=Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and
                "private-keys" not in relative.parts,"Unsafe helper evidence path")
        path=folder/relative
        require(path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink() and path.is_file(),
                "Missing or indirect helper evidence")
        require(path.stat().st_size<=64*1024**2 and sha(path)==expected,"Changed helper evidence: "+name)
    return manifest

def audit(folders):
    from benchmark_suite import BOOTSTRAP
    from hecate_python_env import WORK
    from python_compiler_smoke import BUILD
    from seal_cpu_golden import KEY_BUILD,compare
    from platform_config import identity
    rows,index=load(DEFAULT)
    from upstream_adapters.scenarios import supplemental_rows
    helper_rows=[r for r in rows if r["metadata"].get("helper")]
    by_hash={r["model_sha256"]:r for r in helper_rows+supplemental_rows()}
    states={h:dict(model_id=r["model"]["id"],helper=r["metadata"]["helper"],model_sha256=h,scope=r["metadata"].get("scope","frozen"),
                  status="blocked_bootstrap" if r["metadata"]["helper"] in BOOTSTRAP else "not_run")
            for h,r in by_hash.items()}
    seen=set();values=0;maximum=0.;cases=[]
    binaries={str(p):sha(p) for p in (BUILD/"bin/hecate-opt",BUILD/"lib/libSEAL_HEVM.so",
              KEY_BUILD/"seal_packed_keys",KEY_BUILD/"libseal_packed_metadata.so",*list((BUILD/"lib").glob("*.so")))}
    current=runtime_sources()
    for folder in folders:
        require(folder.resolve().is_relative_to((WORK/"results").resolve()),"Helper report outside results")
        manifest=verify_manifest(folder);report=strict_file(folder/"report.json",8*1024**2)
        model_hash=report["model_sha256"];require(model_hash in states and model_hash not in seen,"Unknown or duplicate helper model")
        seen.add(model_hash);state=states[model_hash]
        require(report.get("scenario")==("frozen" if state["scope"]=="frozen" else "asymmetric-v1"),"Helper scenario identity")
        require(state["helper"]==report["helper"],"Helper/model identity")
        require(report["platform"]==identity() and report["agent_calls"]==0,"Manual helper platform/Agent identity")
        sources=report["source_hashes"]
        require(all(sources.get(k)==v for k,v in current.items()),"Historical helper runtime; use its frozen source")
        for name,expected in sources.items():
            path=ROOT/name
            require(not Path(name).is_absolute() and ".." not in Path(name).parts and
                    path.resolve().is_relative_to(ROOT.resolve()) and sha(path)==expected,"Changed helper source")
        require(report["binary_hashes"]==binaries,"Helper binary identity")
        state.update(status=report["status"],evidence=str(folder),failure_layer=report.get("failure_layer"))
        if report["status"]=="passed":
            needed=("model.json","request.json","arrays.npz","out/decrypted.npy","out/execution.json",
                    "out/upstream-calls.json","out/upstream_helper_probe.mlir","out/lowered.earth.mlir",
                    "out/lowered.ckks.mlir","out/lowered._hecate_golden.hevm","out/_hecate_golden.cst")
            require(all(n in manifest["files"] for n in needed),"Missing real helper pipeline evidence")
            model=strict_file(folder/"model.json",131072)
            require(digest(model)==model_hash,"Changed helper logical model")
            request=strict_file(folder/"request.json",131072)
            from unified_graph_contract import validate_request
            validate_request(request)
            require(digest(request["model"])==model_hash,"Helper request/model mismatch")
            require(all(report.get(k) is True for k in ("compiled","encrypted_execution","actual_helper")),"Missing real FHE helper pass")
            calls=strict_file(folder/"out/upstream-calls.json",1024**2)
            require(calls==report["helper_calls"] and calls["actual_upstream_python"] and calls["real_frontend"] and
                    calls["agent_generated"] is False and len(calls["calls"])==1 and
                    calls["calls"][0]["helper"]==state["helper"] and calls["calls"][0]["returned_to_golden"],
                    "Missing actual upstream helper call")
            if state["helper"]=="HE_BN":
                from upstream_adapters.constants import canonicalize
                require(all(n in manifest["files"] for n in ("out/upstream-original.cst","out/constant-layout.json")),
                        "Missing original helper constant evidence")
                canonical,record=canonicalize((folder/"out/upstream-original.cst").read_bytes(),
                                              request["layout"]["input_slot_period"])
                require(canonical==(folder/"out/_hecate_golden.cst").read_bytes() and
                        record==report["constant_layout"]==strict_file(folder/"out/constant-layout.json",1024**2),
                        "Helper constant layout was not an exact periodic serialization")
            execution=strict_file(folder/"out/execution.json",8*1024**2)
            require(execution==report["execution"] and execution["encrypted_execution"],"Helper execution mismatch")
            import numpy as np
            from benchmark_graph import samples
            from benchmark_math import evaluate as math_reference
            from benchmark_torch import evaluate as torch_reference
            import math
            expected_inputs=[];expected_reference=[]
            period=request["layout"]["input_slot_period"]
            for probe in samples(model,4):
                first=math_reference(model,probe);second=torch_reference(model,probe)
                for name in first:np.testing.assert_allclose(first[name],second[name],atol=1e-12,rtol=1e-12)
                expected_reference.append(np.concatenate([first[o["name"]].reshape(-1) for o in model["outputs"]]))
                expected_inputs.append(np.stack([np.pad(probe[i["name"]].reshape(-1),(0,period-math.prod(i["shape"]))) for i in model["inputs"]]))
            packed=np.stack(expected_inputs)
            if len(model["inputs"])==1:packed=packed[:,0,:]
            with np.load(folder/"arrays.npz",allow_pickle=False) as data:
                require(np.array_equal(data["inputs"],packed) and np.array_equal(data["reference"],np.stack(expected_reference)),
                        "Helper stored inputs/reference differ from independently reconstructed data")
                comparison=compare(np.load(folder/"out/decrypted.npy",allow_pickle=False),data["reference"],1e-5,1e-4)
            from seal_artifact_gate import inspect_artifacts
            from compiler_configuration import verify_artifact_configuration
            from candidate_contract import request_input_names,request_rotations
            from cipher_abi import artifact_options
            from seal_cpu_golden import PROFILE
            gate=inspect_artifacts((folder/"out/lowered._hecate_golden.hevm").read_bytes(),
                 (folder/"out/_hecate_golden.cst").read_bytes(),rotation_steps=request_rotations(request),
                 expected_inputs=len(request_input_names(request)),**artifact_options(request["layout"]))
            verify_artifact_configuration(request,gate,sha(PROFILE))
            require(gate==report["artifact_gate"],"Changed helper artifact gate evidence")
            require(comparison==report["comparison"] and comparison["passed"],"Helper numerical evidence mismatch")
            values+=comparison["compared_values"];maximum=max(maximum,comparison["max_absolute_error"])
        cases.append(dict(path=str(folder),report_sha256=sha(folder/"report.json"),
                          manifest_sha256=sha(folder/"evidence-manifest.json")))
    return dict(schema=1,model_set_sha256=index["model_set_sha256"],helper_models=120,
        counts=dict(Counter(s["status"] for s in states.values() if s["scope"]=="frozen")),
        supplemental_counts=dict(Counter(s["status"] for s in states.values() if s["scope"]!="frozen")),models=list(states.values()),evidence=cases,
        compared_values=values,max_absolute_error=maximum,agent_calls=0,
        interpretation="Manual actual helper execution, not Agent generation; other helpers remain in the denominator.")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inside",action="store_true")
    args=parser.parse_args()
    from hecate_python_env import VENV,enter_nix,WORK
    if not args.inside:
        import shlex,sys
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=900)
    require(args.output.resolve().is_relative_to((WORK/"results").resolve()) and not args.output.exists(),"New helper audit path required")
    result=audit(args.run);dump(args.output,result)
    print(json.dumps({k:result[k] for k in ("counts","supplemental_counts","compared_values","max_absolute_error","agent_calls")},indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
