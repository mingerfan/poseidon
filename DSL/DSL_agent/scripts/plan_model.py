"""Prepare a deterministic Agent input layout; compilation is a separate stage."""
import argparse,json,os,shlex,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[3] if Path(__file__).resolve().parents[1].name == "DSL_agent" and Path(__file__).resolve().parents[2].name == "DSL" else Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT/"scripts/baseline"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True)
    parser.add_argument("--options",type=Path,help="Component options JSON")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--wall-seconds",type=int,default=120)
    parser.add_argument("--max-candidates",type=int,default=8)
    parser.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    from hecate_python_env import VENV,enter_nix
    from benchmark_graph import require
    pinned=os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV
    if not pinned:
        require(not args.inside,"Pinned planning worker required")
        forwarded=["--case",str(args.case.resolve()),"--wall-seconds",str(args.wall_seconds),
                   "--max-candidates",str(args.max_candidates),"--inside"]
        if args.options:forwarded+=["--options",str(args.options.resolve())]
        if args.output:forwarded+=["--output",str(args.output.resolve())]
        command="cd "+shlex.quote(str(ROOT))+" && LD_LIBRARY_PATH=\"$HECATE_PYTHON_LIBRARY_PATH\" "+shlex.join(
            [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*forwarded])
        return enter_nix(command,seconds=210,environment="poseidon-cpu")
    from model_weight_package import regular_bytes
    from public_weight_context import parse_data
    from application_planning.layout import plan_model
    model=parse_data(regular_bytes(args.case,32*1024**2).decode())
    options=None if args.options is None else parse_data(regular_bytes(args.options,8192).decode())
    plan=plan_model(model,options=options,limits=dict(wall_seconds=args.wall_seconds,
                                                    candidate_limit=args.max_candidates))
    raw=json.dumps(plan,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)+"\n"
    if args.output:
        with args.output.open("x") as stream:stream.write(raw)
        print(json.dumps(dict(status=plan["status"],binding=plan["binding"],
            selection=plan["selection"],paid_calls=0,compiled=False,output=str(args.output))))
    else:print(raw)
    return 0 if plan["status"]=="prepared" else 1
if __name__=="__main__":raise SystemExit(main())
