"""Manual upstream HE_SiLU / HE_BN probes through the real isolated CPU pipeline.

This is not an Agent program. The fixed helper is called unchanged; no bootstrap
or placeholder helper is substituted. Other helpers remain separately blocked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time

def trace():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(45,50))
    from candidate_trace import load_frontend
    hc=load_frontend()
    sys.modules["hecate"]=hc  # The real fixed expr frontend; avoid runner/key imports.
    sys.path[:0]=["/poly-deps","/upstream-poly"]
    import einops
    if einops.__version__!="0.6.1":raise ValueError("Helper dependency version")
    import poly.Func as helpers
    payload=json.loads(Path("/payload.json").read_text())
    helper=payload["manual_helper"]
    if helper not in ("HE_SiLU","HE_BN"):raise ValueError("Unregistered manual upstream helper")
    calls=[]
    @hc.func("c,c")
    def golden(x,zero_ct):
        if helper=="HE_BN":
            from upstream_adapters.batch_norm import apply
            import poly.MPCB as mpcb
            result,record=apply(payload["request"]["model"],x,helpers,mpcb)
        else:
            result=helpers.HE_SiLU(x)
            record=dict(helper="HE_SiLU",source="poly/Func.py",returned_to_golden=True)
        calls.append(record)
        return [result]
    hc.save("/out","/out")
    Path("/out/upstream-calls.json").write_text(json.dumps(dict(calls=calls,
        actual_upstream_python=True,real_frontend=True,agent_generated=False)))
    return 0

def main():
    if sys.argv[1:]==["--trace"]:return trace()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--helper",choices=("HE_SiLU","HE_BN"),default="HE_SiLU")
    parser.add_argument("--scenario",choices=("frozen","asymmetric-v1"),default="frozen")
    parser.add_argument("--variant",type=int,choices=range(8),default=0)
    parser.add_argument("--compiler-configuration",choices=("seal-cpu-eva-w40-v1","seal-cpu-eva-w45-v1"),default="seal-cpu-eva-w45-v1")
    args=parser.parse_args()
    from hecate_python_env import ROOT,WORK,VENV,enter_nix,digest
    if not args.inside:
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--helper",args.helper,"--scenario",args.scenario,"--variant",str(args.variant),"--compiler-configuration",args.compiler_configuration]),seconds=900)
    if not os.environ.get("IN_NIX_SHELL") or Path(sys.prefix)!=VENV:raise ValueError("Locked Python required")
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from poly_dependencies import verify
    from benchmark_runner import load,DEFAULT,dump,ideal_reference
    from unified_graph_prepare import prepare_case
    from unified_graph_contract import prepare
    from compiler_configuration import configuration,verify_artifact_configuration
    from candidate_contract import request_input_names,request_rotations
    from cipher_abi import artifact_options
    from python_compiler_smoke import logged,BUILD
    from seal_cpu_golden import KEY_BUILD,PROFILE,compare
    from seal_artifact_gate import inspect_artifacts
    from native_execution_slots import native_slot
    from result_retention import cleanup_run
    import candidate_sandbox as sandbox
    dependency=verify()
    rows,_=load(DEFAULT)
    if args.scenario=="asymmetric-v1":
        if args.helper!="HE_BN":raise ValueError("Asymmetric probe is registered only for HE_BN")
        from upstream_adapters.scenarios import supplemental_rows
        row=supplemental_rows()[args.variant]
    else:row=[r for r in rows if r["metadata"].get("helper")==args.helper][args.variant]
    os.umask(0o077)
    result=Path(tempfile.mkdtemp(prefix="upstream-bn-" if args.helper=="HE_BN" else "upstream-silu-",dir=WORK/"results"))
    print("Helper evidence: "+str(result),flush=True)
    output=result/"out";output.mkdir()
    sources={str(p.relative_to(ROOT)):digest(p) for p in
             (ROOT/"third_party/dacapo/python/poly").rglob("*") if p.is_file() and p.suffix in (".py",".txt")}
    from semantic_benchmark_execution import runtime_sources
    sources.update(runtime_sources())
    binaries={str(p):digest(p) for p in (BUILD/"bin/hecate-opt",BUILD/"lib/libSEAL_HEVM.so",
                                         KEY_BUILD/"seal_packed_keys",KEY_BUILD/"libseal_packed_metadata.so",*list((BUILD/"lib").glob("*.so")))}
    report=dict(status="running",helper=args.helper,scenario=args.scenario,model_sha256=row["model_sha256"],agent_calls=0,
                source_hashes=sources,binary_hashes=binaries,dependency=dependency,compiled=False,
                encrypted_execution=False,actual_helper=False)
    from platform_config import identity
    report["platform"]=identity()
    stage="preparation";begin=time.monotonic()
    def run(payload,argv,log,seconds=60,keys=None,helpers=False):
        with native_slot(WORK/"cache/agent-native-slots"):
            return sandbox.run(payload,output,argv,result/log,seconds,keys,helpers=helpers)
    try:
        prepare_case(row["model"],result)
        from benchmark_graph import samples
        from benchmark_math import evaluate
        if row["metadata"].get("ideal"):
            report["approximation_max_abs_error"]=max(float(np.max(np.abs(
                next(iter(evaluate(row["model"],x).values()))-ideal_reference(row,x)))) for x in samples(row["model"],4))
        else:report["approximation_max_abs_error"]=None
        report["approximation_error_is_separate_from_execution_error"]=True
        request=prepare(row["model"],digest(PROFILE),configuration(args.compiler_configuration))
        report["compiler_configuration"]=request["compiler_configuration"]
        payload=result/"payload.json";dump(payload,dict(request=request,manual_helper=args.helper))
        dump(result/"model.json",row["model"])
        dump(result/"request.json",request)
        stage="sandbox"
        sentinel=result/"private-sentinel";sentinel.write_text("probe only")
        probe=result/"probe.json"
        dump(probe,dict(sentinel=str(sentinel),workspace=str(ROOT),
                        net_ns=os.readlink("/proc/self/ns/net"),pid_ns=os.readlink("/proc/self/ns/pid")))
        if run(probe,[str(VENV/"bin/python"),"/app/candidate_worker.py","probe"],"probe.log",helpers=True)!=0:
            raise ValueError("Sandbox capability probe failed")
        stage="real_helper_trace"
        if run(payload,[str(VENV/"bin/python"),"/app/upstream_helper_probe.py","--trace"],"trace.log",helpers=True)!=0:
            raise ValueError("Actual upstream helper tracing failed")
        report["helper_calls"]=json.loads((output/"upstream-calls.json").read_text())
        calls=report["helper_calls"]
        if not (calls["actual_upstream_python"] and calls["real_frontend"] and
                len(calls["calls"])==1 and calls["calls"][0]["helper"]==args.helper and
                calls["calls"][0]["returned_to_golden"]):
            raise ValueError("Actual helper call identity")
        report["actual_helper"]=True
        stage="compiler"
        command=["/hecate-opt","/out/upstream_helper_probe.mlir","--eva","--ckks-config=/profile.json",
                 "--waterline="+str(request["compiler_configuration"]["waterline"]),"--enable-debug-printer","--mlir-disable-threading","--verify-each","-o","/out/lowered.mlir"]
        if run(payload,command,"compile.log")!=0:raise ValueError("Actual upstream helper compiler failed")
        report["compiled"]=True
        if args.helper=="HE_BN":
            stage="constant_layout"
            from upstream_adapters.constants import canonicalize
            cst=output/"_hecate_golden.cst"
            original=cst.read_bytes()
            canonical,record=canonicalize(original,request["layout"]["input_slot_period"])
            (output/"upstream-original.cst").write_bytes(original)
            dump(output/"constant-layout.json",record)
            cst.write_bytes(canonical)
            report["constant_layout"]=record
        stage="artifact_gate"
        gate=inspect_artifacts((output/"lowered._hecate_golden.hevm").read_bytes(),
            (output/"_hecate_golden.cst").read_bytes(),rotation_steps=request_rotations(request),
            expected_inputs=len(request_input_names(request)),**artifact_options(request["layout"]))
        verify_artifact_configuration(request,gate,digest(PROFILE));report["artifact_gate"]=gate
        stage="key_setup"
        keys=result/"private-keys";keys.mkdir(mode=0o700)
        with native_slot(WORK/"cache/agent-native-slots"):
            if logged([str(KEY_BUILD/"seal_packed_keys"),str(keys),str(request["layout"]["input_slot_period"])],
                      result/"parameters.json",120)!=0:raise ValueError("Key generation failed")
        with np.load(result/"arrays.npz",allow_pickle=False) as data:
            np.savez(output/"arrays.npz",inputs=data["inputs"])
        artifact_hashes={p.name:digest(p) for p in output.iterdir() if p.is_file()}
        stage="seal_runtime"
        if run(payload,[str(VENV/"bin/python"),"/app/candidate_worker.py","execute"],"execute.log",150,keys)!=0:
            raise ValueError("Actual upstream helper SEAL execution failed")
        report["execution"]=json.loads((output/"execution.json").read_text())
        report["encrypted_execution"]=True
        if any(digest(output/name)!=h for name,h in artifact_hashes.items()):raise ValueError("Artifact changed")
        stage="numerical_comparison"
        with np.load(result/"arrays.npz",allow_pickle=False) as data:
            report["comparison"]=compare(np.load(output/"decrypted.npy",allow_pickle=False),data["reference"],1e-5,1e-4)
        if not report["comparison"]["passed"]:raise ValueError("Frozen numerical threshold")
        report["status"]="passed"
    except Exception as e:
        report.update(status="failed",failure_layer=stage,diagnostic=str(e))
    finally:
        if any(digest(ROOT/p)!=h for p,h in sources.items()) or any(digest(Path(p))!=h for p,h in binaries.items()):
            report.update(status="failed",failure_layer="integrity",diagnostic="Source or binary changed")
        if verify()!=dependency:report.update(status="failed",failure_layer="integrity",diagnostic="Helper dependency changed")
        report["seconds"]=time.monotonic()-begin
        dump(result/"report.json",report)
        if (result/"private-keys").exists():cleanup_run(result,WORK/"results")
        # Post-cleanup integrity evidence; the report itself is not rewritten.
        retained={str(p.relative_to(result)):digest(p) for p in result.rglob("*")
                  if p.is_file() and p.name!="evidence-manifest.json"}
        dump(result/"evidence-manifest.json",dict(schema=1,files=retained,private_keys_retained=False))
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_hashes","binary_hashes","dependency","execution")},indent=2))
    return int(report["status"]!="passed")

if __name__=="__main__":raise SystemExit(main())
