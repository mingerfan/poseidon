#!/usr/bin/env python3
"""Compile an in-flight remote GPU API trace on the local LLVM/MLIR host.

The GPU application waits for an atomic `ready` marker while retaining its
prepared keys and inputs. Call this only after starting --unsafe-trace-plan.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", help="SSH alias")
    parser.add_argument("remote_dir", help="empty directory used by --unsafe-trace-plan")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    remote = args.remote_dir.rstrip("/")
    quoted = shlex.quote(remote)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout
    try:
        while True:
            status = subprocess.run(["ssh", args.server, "test", "-f", quoted + "/trace.mlir.ready"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
            if status == 0:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("trace.mlir.ready not produced before timeout")
            time.sleep(3)
        subprocess.run(["rsync", "-a", f"{args.server}:{remote}/trace.mlir", str(output / "trace.mlir")], check=True)
        subprocess.run(["python3", str(ROOT / "bench/resnet20_s2c_first/compile_prepared.py"),
                        str(output / "trace.mlir"), "--format", "json", "--output-dir", str(output),
                        "--hecate-opt", str(args.hecate_opt)], check=True)
        plans = list(output.glob("scheduled.*.runtime-plan.json"))
        if len(plans) != 1:
            raise RuntimeError(f"expected exactly one compiler plan, found {len(plans)}")
        subprocess.run(["rsync", "-a", str(plans[0]), f"{args.server}:{remote}/plan.json"], check=True)
        subprocess.run(["rsync", "-a", str(output / "operator-spec.json"),
                        f"{args.server}:{remote}/operator-spec.json"], check=True)
        subprocess.run(["ssh", args.server, f"touch {quoted}/ready.tmp && mv {quoted}/ready.tmp {quoted}/ready"], check=True)
    except Exception:
        subprocess.run(["ssh", args.server, f"touch {quoted}/error"], check=False)
        raise
    print(f"compiled trace and published RuntimePlan in {args.server}:{remote}")


if __name__ == "__main__":
    main()
