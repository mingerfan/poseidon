#!/usr/bin/env python3
"""Compile production BSGS/RMS primitives at scales 40 and 120; run real CKKS.

All generated plans, payloads and decoded slots stay in --output (a new build
directory). This is synthetic small-graph coverage, not complete Qwen validation.
Both scales are positive controls. Integer coefficient correction precedes
native rescale, preserving the compiler's original lazy 40-bit schedule.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SLOTS = 8192


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verify_scale_contract(command, spec):
    settings = [int(v.split("=", 1)[1]) for v in command if v.startswith("--waterline=")]
    if settings != [spec["context"]["default_scale_log2"]]:
        raise ValueError("compiler --waterline must equal operator-spec context.default_scale_log2")


def native_numpy_frontend(output, dacapo):
    """Use the project C++ frontend, removing only its unused torch adapter.

    This host lacks torch. The exact two source fragments are required, so a
    future frontend change fails visibly instead of applying a loose rewrite.
    """
    original = dacapo / "python/hecate/hecate"
    target = output / "frontend_numpy/hecate"
    target.mkdir(parents=True)
    code = (original / "expr.py").read_text()
    fragments = ("import torch\n", "    elif isinstance(other, torch.Tensor) :\n        return Plain( torch.flatten(other).tolist() )\n")
    for fragment in fragments:
        if code.count(fragment) != 1:
            raise RuntimeError("native frontend torch adapter changed; inspect it before rerunning")
        code = code.replace(fragment, "")
    (target / "expr.py").write_text(code)
    (target / "__init__.py").write_bytes((original / "__init__.py").read_bytes())
    os.environ["HECATE"] = str(dacapo)
    os.environ["HECATE_BUILD"] = str(dacapo / "build/nix")
    sys.path.insert(0, str(target.parent))
    sys.path.insert(0, str(ROOT / "scripts/qwen3_dsl"))
    return {"original_expr_sha256": digest(original / "expr.py"),
            "numpy_adapter_sha256": digest(target / "expr.py"),
            "native_library_sha256": digest(dacapo / "build/nix/lib/libHecateFrontend.so")}


def compiler_command(args, profiles, spec_path, spec, source, output, scale):
    boot = spec["boot_profiles"][0]
    command = [str(args.compiler), "--dacapo", "--boot-placement=depth-dp",
        f"--ckks-config={profiles / 'compiler-profile.json'}", f"--waterline={scale}",
        "--mlir-print-op-on-diagnostic=false", "--mlir-elide-elementsattrs-if-larger=16",
        "--runtime-plan-id=4270", f"--runtime-plan-target-id={spec['target_id']}",
        "--runtime-plan-capability-version=1", f"--runtime-plan-operator-spec-id={spec['spec_id']}",
        f"--runtime-plan-operator-spec-version={spec['version']}",
        f"--runtime-plan-operator-spec-sha256={digest(spec_path)}",
        f"--runtime-plan-context-id={spec['context']['context_id']}", "--runtime-plan-ntt=true",
        f"--runtime-plan-boot-profile={boot['profile_id']}",
        f"--runtime-plan-boot-implementation={boot['implementation']}",
        "--runtime-plan-inline-payload-max-bytes=4096",
        f"--runtime-plan-physical-level-operator-spec-path={spec_path}",
        "--runtime-plan-levels-per-logical-level=5", "--runtime-plan-device-count=1",
        "--runtime-plan-device-counts=1", f"--runtime-plan-operator-spec-path={spec_path}",
        f"--runtime-plan-communication-profile-path={profiles / 'communication-profile.json'}",
        "--runtime-plan-memory=true", "--runtime-plan-memory-reuse=true",
        "--runtime-plan-memory-report=true", "--runtime-plan-plaintext-schedule=waterline",
        "--runtime-plan-prepare-backend=auto", f"--runtime-plan-gpu-budget-bytes={7 * 2**30}",
        f"--runtime-plan-reserve-bytes={3 * 2**30}",
        f"--runtime-plan-workspace-bytes-per-op={600 * 2**20}",
        f"--runtime-plan-host-budget-bytes={2 * 2**30}",
        f"--runtime-plan-pinned-budget-bytes={256 * 2**20}",
        f"--runtime-plan-raw-budget-bytes={64 * 2**20}",
        "--runtime-plan-waterline-fraction=0.5", "--runtime-plan-lookahead-groups=8",
        "--runtime-plan-format=json", "--runtime-plan-bundle-format=pack", str(source),
        "-o", str(output / "optimized.mlir")]
    verify_scale_contract(command, spec)
    return command


def run(command, logfile, env):
    with logfile.open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def build_normalizer_probe(output, runner, env):
    """Call the production inline normalizer with the runner's actual Q primes."""
    source = output / "normalizer_probe.cpp"
    source.write_text(r'''#include "poseidon/runtime_api/rescale_normalization.h"
#include <iomanip>
#include <iostream>
int main() {
    std::size_t count, rows;
    std::cin >> count;
    std::vector<poseidon::Modulus> primes;
    for (std::size_t i=0; i<count; ++i) {
        std::uint64_t prime; std::cin >> prime; primes.emplace_back(prime);
    }
    std::cin >> rows;
    for (std::size_t i=0; i<rows; ++i) {
        int in_level, out_level, in_scale, out_scale;
        std::cin >> in_level >> out_level >> in_scale >> out_scale;
        std::vector<poseidon::Modulus> active(primes.begin(), primes.begin()+in_level+1);
        try {
            const auto result = poseidon::runtime_api::make_rescale_normalization(
                active, std::ldexp(1.0,in_scale), in_level-out_level, std::ldexp(1.0,out_scale));
            const double prescale = std::ldexp(1.0,in_scale) * result.coefficient_multiplier;
            double native = prescale;
            double uncorrected = std::ldexp(1.0,in_scale);
            for (std::size_t j=0; j<result.native_drop_count; ++j) {
                const auto prime = active[active.size()-j-1].value();
                native /= prime;
                uncorrected /= prime;
            }
            std::cout << i << " accepted " << result.native_drop_count << ' '
                      << result.coefficient_multiplier << ' ' << std::setprecision(17)
                      << result.relative_rounding_error << ' ' << std::log2(native) << ' '
                      << std::log2(prescale) << ' ' << std::log2(uncorrected) << ' '
                      << std::abs(native/std::ldexp(1.0,out_scale)-1.0) << '\n';
        } catch (const std::exception &error) {
            std::cout << i << " rejected " << error.what() << '\n';
        }
    }
    return std::cin ? 0 : 1;
}
''')
    build = runner.resolve().parents[1]
    binary = output / "normalizer_probe"
    command = ["c++", "-std=c++17", "-O2", f"-I{ROOT / 'src'}", f"-I{build}",
        str(source), f"-L{build}", f"-Wl,-rpath,{build}", "-lposeidon_shared", "-o", str(binary)]
    write_json(output / "normalizer-probe-command.json", command)
    if run(command, output / "normalizer-probe-build.log", env):
        raise RuntimeError("production normalizer probe did not build")
    return binary


def audit_normalizer(binary, report, rescales, output):
    lines = [str(len(report["actual_q_moduli"])), " ".join(map(str, report["actual_q_moduli"])), str(len(rescales))]
    lines += [" ".join(str(row[k]) for k in ("input_level", "output_level", "input_scale_log2", "output_scale_log2"))
              for row in rescales]
    result = subprocess.run([str(binary)], input="\n".join(lines) + "\n", text=True, capture_output=True, check=True)
    (output / "normalizer-probe.log").write_text(result.stdout + result.stderr)
    records = []
    for line in result.stdout.splitlines():
        fields = line.split()
        record = dict(index=int(fields[0]), accepted=fields[1] == "accepted")
        if record["accepted"]:
            record.update(native_drop_count=int(fields[2]), coefficient_multiplier=int(fields[3]),
                relative_rounding_error=float(fields[4]), native_scale_log2=float(fields[5]),
                prescale_log2=float(fields[6]), uncorrected_native_scale_log2=float(fields[7]),
                corrected_scale_relative_error=float(fields[8]),
                target_scale_log2=rescales[record["index"]]["output_scale_log2"])
        else:
            record["error"] = " ".join(fields[2:])
        records.append(record)
    if len(records) != len(rescales):
        raise RuntimeError("normalizer probe did not check every Rescale")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    dacapo = ROOT / "third_party/ckks-runtime/third_party/dacapo"
    parser.add_argument("--compiler", type=Path, default=dacapo / "build/nix/bin/hecate-opt")
    parser.add_argument("--runner", type=Path, default=ROOT / "build-runtime-gpu-api-release/bin/poseidon_runtime_gpu_qwen_e2e")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu", "auto"), default=["cpu", "gpu", "auto"])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    frontend = native_numpy_frontend(args.output, dacapo)
    import hecate as hc
    from qwen3.kernels import ops
    from qwen3.kernels import qwen_layers as ql

    rng = np.random.default_rng(20261014)
    size, width = 8, 12
    weight = rng.normal(0, 0.2, (width, size))
    bias = rng.normal(0, 0.03, width)
    gain = rng.uniform(0.8, 1.2, width)
    logical = rng.uniform(-0.4, 0.4, size)
    physical = np.r_[logical, rng.normal(0, 1000, SLOTS - size)]
    rsqrt = ql.PolynomialApproximation("rsqrt", (1.5, -0.5), (0.001, 2.0))
    coefficients = (0.125, -0.75, 0.5)

    @hc.func("c")
    def composite_scale_regression(expression):
        projected = ql.linear(ql.PackedVector(expression, size, SLOTS), weight, bias)
        squared = ops.square(projected.expr)
        total = ql._total_repeated(ql.PackedVector(squared, width, SLOTS))
        polynomial = ops.polynomial(projected.expr, coefficients, slots=width)
        polynomial = ql.extract(ql.PackedVector(polynomial, width, SLOTS), 0, width).expr
        rms = ql.rms_norm(projected, gain, rsqrt=rsqrt).expr
        # Enough real dependency depth for the ordinary depth-DP planner to
        # insert Boot; an explicit frontend bootstrap can be optimized away.
        deep = rms
        for _ in range(4):
            deep = ops.square(deep) * hc.Plain(np.full(width, 0.25)) + hc.Plain(np.full(width, 0.1))
        return [projected.expr, squared, total, polynomial, rms, deep]

    source = Path(hc.save(str(args.output), str(args.output), bytecode=True))
    # Independent dense oracle: no diagonal, rotation or BSGS helper is used.
    projected = weight @ logical + bias
    squared = projected**2
    denominator = float(np.mean(squared) + 1e-6)
    rms = projected * (1.5 - 0.5 * denominator) * gain
    pad = lambda values: np.pad(values, (0, SLOTS - len(values))).tolist()
    deep = rms.copy()
    for _ in range(4):
        deep = 0.25 * deep**2 + 0.1
    expected = [pad(projected), pad(squared), np.full(SLOTS, np.sum(squared)).tolist(),
                pad(np.polynomial.polynomial.polyval(projected, coefficients)), pad(rms), pad(deep)]
    names = ["dense_bsgs_linear", "square", "all_slots_reduction", "polynomial_mask", "polynomial_rms", "deep_polynomial_with_boot"]
    profiles = ROOT / "scripts/qwen3_dsl/profiles"
    spec_template = json.loads((profiles / "operator-spec.json").read_text())
    if spec_template["context"]["rns_moduli_log2"] != [30] * 50:
        raise RuntimeError("regression requires production Q50 with 30-bit primes")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0")
    normalizer_probe = build_normalizer_probe(args.output, args.runner, environment)
    summary = dict(scope="synthetic small production BSGS/RMS graph; not full Qwen", frontend=frontend,
        source=str(source), source_sha256=digest(source), compiler_sha256=digest(args.compiler),
        runner_sha256=digest(args.runner), slots=SLOTS, input_size=size, output_size=width,
        regression_script_sha256=digest(Path(__file__)),
        normalizer_header_sha256=digest(ROOT / "src/poseidon/runtime_api/rescale_normalization.h"),
        normalization_order="coefficient and metadata multiplied before native rescale",
        logical_to_physical_levels=5, q_count=50, p_count=8,
        oracle="independent numpy dense W@x; exact polynomial RMS, not exact inverse sqrt",
        dirty_tail_max_abs=float(np.max(np.abs(physical[size:]))), cases=[])
    for scale in (40, 120):
        output = args.output / f"scale{scale}"
        output.mkdir()
        spec = json.loads(json.dumps(spec_template))
        spec["context"]["default_scale_log2"] = scale
        spec["context"]["context_id"] = f"scale-regression-n16384-q50-p8-s{scale}"
        spec["spec_id"] = f"scale-regression-s{scale}"
        spec_path = output / "operator-spec.json"
        write_json(spec_path, spec)
        command = compiler_command(args, profiles, spec_path, spec, source, output, scale)
        write_json(output / "compile-command.json", command)
        if run(command, output / "compile.log", environment):
            raise RuntimeError(f"compiler failed: {output / 'compile.log'}")
        plans = list(output.glob("*.runtime-plan.json"))
        if len(plans) != 1:
            raise RuntimeError("exactly one compiled plan required")
        plan_path = plans[0]
        plan = json.loads(plan_path.read_text())
        values = {v["id"]: v for v in plan["values"]}
        operations = plan["initialization"] + plan["execution"] + plan["finalization"]
        rescales = []
        for operation in operations:
            if operation["kind"] == "compute" and operation["op"] == "rescale":
                x, y = values[operation["inputs"][0]], values[operation["output"]]
                rescales.append(dict(input_level=x["level"], output_level=y["level"],
                    input_scale_log2=x["scale_log2"], output_scale_log2=y["scale_log2"]))
        minimum_scale = min(v["scale_log2"] for v in values.values())
        if minimum_scale < scale:
            raise RuntimeError(f"compiler emitted a typed value below configured scale {scale}")
        if scale == 40 and not any(r["input_scale_log2"] == 160 and r["output_scale_log2"] == 40 for r in rescales):
            raise RuntimeError("lazy 40-bit plan did not reproduce 160->40 Rescale")
        fixture = dict(format_version=1, model=dict(name=f"scale{scale}-production-primitives-small", synthetic_weights=True,
            complete_qwen=False), context=dict(poly_degree=16384, q_moduli_bits=[30] * 50,
            p_moduli_bits=[30] * 8, default_scale_log2=scale), tolerances=dict(atol=0.001, rtol=0.0001),
            inputs=[dict(name="dirty_input", value_id=int(plan["external_inputs"][0]), values=physical.tolist())],
            expected_outputs=[dict(name=name, value_id=int(value_id), values=oracle)
                for name, value_id, oracle in zip(names, plan["final_outputs"], expected)])
        if len(plan["final_outputs"]) != len(expected):
            raise RuntimeError("compiler output count differs from independent oracle")
        fixture_path = output / "fixture.json"
        write_json(fixture_path, fixture)
        case = dict(scale=scale, cli_spec_scale_match=True, minimum_typed_scale_log2=minimum_scale,
            scale_histogram=dict(Counter(str(v["scale_log2"]) for v in values.values())),
            value_count=len(values), operation_count=len(operations),
            final_types=[dict(level=values[i]["level"], scale_log2=values[i]["scale_log2"])
                         for i in plan["final_outputs"]],
            boot_boundaries=[dict(ordinal=o["ordinal"],
                inputs=[dict(level=values[i]["level"], scale_log2=values[i]["scale_log2"]) for i in o["inputs"]],
                output=dict(level=values[o["output"]]["level"], scale_log2=values[o["output"]]["scale_log2"]))
                for o in operations if o.get("op") == "boot"],
            op_histogram=dict(Counter(o.get("op", o["kind"]) for o in operations)), rescales=rescales, runs=[])
        if case["op_histogram"].get("boot", 0) < 1:
            raise RuntimeError("positive graph did not exercise the Boot implementation")
        for backend in args.backends:
            report_path = output / f"{backend}-report.json"
            runner_command = [str(args.runner), str(plan_path), str(spec_path), str(fixture_path),
                str(report_path), str(output / f"{backend}-decoded.json"), "--bundle",
                str(plan_path).removesuffix(".runtime-plan.json") + ".bundle", "--backend", backend]
            write_json(output / f"{backend}-command.json", runner_command)
            code = run(runner_command, output / f"{backend}.log", environment)
            report = json.loads(report_path.read_text())
            if code or not report["passed"]:
                raise RuntimeError(f"positive GPU execution failed: {output / (backend + '.log')}")
            case["runs"].append(dict(backend=backend, returncode=code, report=report))
            print(f"scale={scale} backend={backend} code={code} stage={report['stage']}", flush=True)
        case["normalization"] = audit_normalizer(normalizer_probe, report, rescales, output)
        # The integer is rounded to nearest; corrected native scale can be
        # below the exact target by the existing <=1e-9 rounding allowance.
        # The requested output retains the production >=2^40 contract.
        if not all(r["accepted"] and r["target_scale_log2"] >= 40 and r["relative_rounding_error"] <= 1e-9
                   and r["corrected_scale_relative_error"] <= 1e-9
                   for r in case["normalization"]):
            raise RuntimeError("positive Rescale did not preserve the production normalizer precision contract")
        summary["cases"].append(case)
        write_json(args.output / "summary.json", summary)
    mismatch = json.loads(json.dumps(spec_template))
    mismatch["context"]["default_scale_log2"] = 120
    try:
        verify_scale_contract(["--waterline=40"], mismatch)
    except ValueError as error:
        summary["cli_spec_mismatch_negative_control"] = str(error)
    else:
        raise RuntimeError("CLI/spec mismatch was not rejected")
    summary["passed"] = True
    write_json(args.output / "summary.json", summary)
    print(f"PASS {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
