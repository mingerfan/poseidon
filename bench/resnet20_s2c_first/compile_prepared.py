#!/usr/bin/env python3
"""Schedule prepared ResNet20 CKKS API calls on one GPU without replanning."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DACAPO = ROOT / "third_party/ckks-runtime/third_party/dacapo"


def spec_from_mlir(input_path):
    """Import only context metadata; API-call costs stay in the MLIR calls."""
    source = input_path.read_bytes()
    module = re.search(r"\bmodule\s+attributes\s*\{(.*?)\}\s*\{",
                       source.decode("utf-8"), re.DOTALL)
    if not module:
        raise ValueError("expected an MLIR module attributes block")
    arrays = re.findall(r"\bckks\.rns_moduli\s*=\s*array\s*<\s*i64\s*:\s*([^>]*)>",
                        module.group(1))
    if len(arrays) != 1 or not re.fullmatch(r"\s*[0-9]+(?:\s*,\s*[0-9]+)*\s*", arrays[0]):
        raise ValueError("expected one module ckks.rns_moduli = array<i64: ...>")
    moduli = [int(value.strip()) for value in arrays[0].split(",")]
    bits = [value.bit_length() for value in moduli]
    if (len(moduli) != 50 or len(set(moduli)) != 50 or
            any(width not in (30, 31, 32) for width in bits) or
            any(value % (2 * 65536) != 1 for value in moduli)):
        raise ValueError("prepared ResNet20 requires 50 distinct 30/31/32-bit N65536 Q moduli")
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    ordinary = ("add_cc", "add_cp", "sub_cc", "sub_cp", "mul_cc", "mul_cp",
                "negate", "rotate", "rescale", "mod_switch", "relinearize")
    operators = {name: {"supported": False, "latency_us_by_level": None,
                        "noise_by_level": None} for name in ordinary}
    operators["rescale"]["max_levels_per_op"] = 1
    operators["boot"] = {"supported": False}
    return {
        "spec_format_version": 2,
        "spec_id": "poseidon-manual-prepared-v1",
        "version": 1,
        "status": "imported",
        "target_id": "poseidon-manual-prepared-v1",
        "rescale_mode": "native",
        "noise_unit": None,
        "provenance": {
            "kind": "prepared-api-mlir",
            "repository": "poseidon",
            "revision": digest,
            "path": input_path.name,
            "source_sha256": digest,
        },
        "context": {
            "context_id": "manual-resnet20-q50",
            "poly_degree": 65536,
            "rns_moduli_log2": bits,
            "rns_moduli": moduli,
            "max_modulus_log2": max(bits),
            "default_scale_log2": 40,
        },
        "levels": {"lower_bound": 0, "upper_bound": 49},
        "operators": operators,
        "boot_profiles": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="MLIR emitted by the prepared API graph")
    parser.add_argument("--operator-spec", type=Path,
                        help="OperatorSpec V2; default: import Q50 from MLIR into output-dir/operator-spec.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", choices=("1",), default="1",
                        help="Only one GPU is supported (default: 1)")
    parser.add_argument("--format", choices=("json", "binary"), default="json")
    parser.add_argument("--hecate-opt", type=Path,
                        default=DACAPO / "build/manual/bin/hecate-opt")
    parser.add_argument("--compiler-config", type=Path, default=DACAPO / "config.json",
                        help="Driver configuration; no scale or Boot planner is run")
    args = parser.parse_args()

    input_path = args.input.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.operator_spec is None:
        spec_path = output / "operator-spec.json"
        spec_path.write_text(json.dumps(spec_from_mlir(input_path), indent=2) + "\n")
    else:
        spec_path = args.operator_spec.resolve(strict=True)
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    command = [
        str(args.hecate_opt.resolve()), str(input_path),
        f"--ckks-config={args.compiler_config.resolve(strict=True)}",
        "--ckks-to-runtime-plan",
        "--runtime-plan-coalesce-rescales=false",
        "--runtime-plan-levels-per-logical-level=1",
        "--runtime-plan-device-counts=1",
        f"--runtime-plan-operator-spec-path={spec_path}",
        f"--runtime-plan-target-id={spec['target_id']}",
        f"--runtime-plan-operator-spec-id={spec['spec_id']}",
        f"--runtime-plan-operator-spec-version={spec['version']}",
        f"--runtime-plan-operator-spec-sha256=sha256:{hashlib.sha256(raw).hexdigest()}",
        f"--runtime-plan-context-id={spec['context']['context_id']}",
        "--runtime-plan-memory", "--runtime-plan-memory-reuse=false",
        "--runtime-plan-memory-report",
        f"--runtime-plan-format={args.format}",
        "--dump-pass-pipeline", "-o", str(output / "scheduled.mlir"),
    ]
    # ApiCall callees already select the native prepared Boot configuration.
    # Do not run --dacapo/--dacapo-precompile, provide a physical-level planning
    # spec, or add a Boot/scale planning pass to this already-physical graph.
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    log_path = output / "compiler.log"
    with log_path.open("w") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"hecate-opt exited with status {error.returncode}; see {log_path}"
            ) from error
    print(output)


if __name__ == "__main__":
    main()
