"""Local command entry for bounded source preparation, in the pinned environment."""
import argparse,json,os,shlex,sys
from pathlib import Path
MAX_REQUEST=5*1024**2
FORMAT="poseidon-source-preparation-result-v1"

def parser():
    p=argparse.ArgumentParser(description="Prepare explicit model approximation profiles before DSL generation.")
    p.add_argument("action",choices=("inspect","observe","freeze"))
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--timeout",type=int,default=240)
    p.add_argument("--trust-local-code",action="store_true")
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    if not 1<=args.timeout<=300:parser().error("timeout must be 1..300 seconds")
    if not args.inside:
        from workspace_paths import require_linux_backend
        require_linux_backend()
        from hecate_python_env import VENV,enter_nix
        script=Path(__file__).resolve().parents[1]/"agent_calibration.py"
        forwarded=list(sys.argv[1:]if argv is None else argv)
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"),"-B",str(script),*forwarded,"--inside"])
        return enter_nix(command,seconds=args.timeout,environment="poseidon-cpu")
    from source_preparation_service import SourcePreparation,public_summary
    from strict_json_files import strict_file
    try:
        if not args.request.is_absolute()or args.request!=args.request.resolve()or args.request.is_symlink():
            raise ValueError("request_path")
        request=strict_file(args.request,MAX_REQUEST)
        if type(request)is not dict or request.get("action")!=args.action:raise ValueError("request_action")
        result=SourcePreparation().run(request,args.output,trust_local_code=args.trust_local_code)
        summary=public_summary(result)
    except (ValueError,TypeError,KeyError,OSError):
        summary=dict(format=FORMAT,status="validation_failed",failure=dict(
            code="source_preparation_input_rejected",layer="input",retry_policy="never_automatic"),
            api_calls=0,compiler_calls=0,encrypted_calls=0)
    print(json.dumps(summary,sort_keys=True,allow_nan=False),flush=True)
    return 0 if summary["status"]in("described","observed","frozen")else 2

if __name__=="__main__":raise SystemExit(main())
