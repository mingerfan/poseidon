#!/usr/bin/env python3
"""Prepare a bound layout and report pre-compilation resource facts."""
import argparse,json,os,shlex,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[3] if Path(__file__).resolve().parents[1].name == "DSL_agent" and Path(__file__).resolve().parents[2].name == "DSL" else Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True,help="Normalized model JSON")
    parser.add_argument("--options",type=Path,help="Application options JSON")
    parser.add_argument("--layout-plan",type=Path,help="Existing bound layout decision")
    parser.add_argument("--memory-mib",type=int,help="Requested memory budget, bounded by the existing sandbox")
    parser.add_argument("--samples",type=int,default=4,choices=range(1,17))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=parser.parse_args()
    if a.layout_plan and a.options:parser.error("--options and --layout-plan are mutually exclusive")
    from hecate_python_env import VENV,enter_nix
    if not a.inside:
        argv=[str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside","--case",str(a.case.resolve()),
              "--output",str(a.output.resolve()),"--samples",str(a.samples)]
        if a.options:argv+=["--options",str(a.options.resolve())]
        if a.layout_plan:argv+=["--layout-plan",str(a.layout_plan.resolve())]
        if a.memory_mib is not None:argv+=["--memory-mib",str(a.memory_mib)]
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(argv)
        return enter_nix(command,seconds=600,environment="poseidon-cpu")
    from application_planning.layout import plan_model,check_plan
    from application_planning.resources import assess_plan
    from benchmark_runner import strict_file
    from benchmark_graph import canonical,require
    from public_weight_context import TRUSTED_BYTES
    require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned resource worker")
    require(not a.output.exists() and a.output.resolve().parent.is_dir(),"Fresh resource output")
    model=strict_file(a.case,TRUSTED_BYTES)
    planned=strict_file(a.layout_plan,8*1024**2) if a.layout_plan else plan_model(model,
        options=strict_file(a.options,131072) if a.options else None)
    if planned["status"]!="prepared":
        record=dict(format="poseidon-model-resource-preflight-v1",status=planned["status"],
            failure=planned["failure"],layout_plan=planned,resource_analysis=None,
            compiled=False,encrypted_execution=False,paid_calls=0)
    else:
        budget=None if a.memory_mib is None else a.memory_mib*1024**2
        analysis=assess_plan(planned,model,memory_budget_bytes=budget,sample_count=a.samples)
        record=dict(format="poseidon-model-resource-preflight-v1",status=analysis["status"],
            failure=analysis["failure"],layout_plan=planned,resource_analysis=analysis,
            compiled=False,encrypted_execution=False,paid_calls=0)
    with a.output.open("xb") as stream:stream.write(canonical(record))
    print(json.dumps(dict(status=record["status"],output=str(a.output.resolve()),
        execution_readiness="unknown",compiled=False,encrypted_execution=False,paid_calls=0)))
    return 0 if record["status"]=="assessed" else 1
if __name__=="__main__":raise SystemExit(main())
