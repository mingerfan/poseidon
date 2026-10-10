#!/usr/bin/env python3
"""Compile small placed IR to ordinary/Release/reuse plans and memory reports."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_ir(counts):
    places = [
        (rank, device) for rank, count in enumerate(counts) for device in range(count)
    ]
    ciphertext = "tensor<1x!ckks.poly<2 * 40 * 6>>"
    plaintext = "tensor<1x!ckks.poly<1 * 40 * 6>>"
    lines = [
        "module {",
        f"  func.func @memory_smoke(%arg0: {ciphertext} {{dist.rank = 0 : i64, dist.device = -1 : i64}}) -> {ciphertext} attributes {{dist.device_counts = array<i64: {', '.join(map(str, counts))}>}} {{",
    ]
    logical_id, transfer_id = 1, 0

    def compute(name, inputs, place, plain=False, offset=None):
        nonlocal logical_id
        output = f"%v{logical_id}"
        attrs = f" <{{offset = array<i64: {offset}>}}>" if offset is not None else ""
        types = [ciphertext] + ([plaintext] if plain else [])
        lines.append(
            f'    {output} = "ckks.{name}"({", ".join(inputs)}){attrs} {{dist.logical_id = {logical_id} : i64, dist.rank = {place[0]} : i64, dist.device = {place[1]} : i64}} : ({", ".join(types if len(inputs) == 1 or plain else [ciphertext] * len(inputs))}) -> {ciphertext}'
        )
        logical_id += 1
        return output

    def transfer(value, source, destination, initialization, plain=False):
        nonlocal transfer_id
        output = f"%t{transfer_id}"
        ty = plaintext if plain else ciphertext
        lines.append(
            f'    {output} = "dist.transfer"({value}) <{{transfer_id = {transfer_id} : i64, source_rank = {source[0]} : i64, source_device = {source[1]} : i64, destination_rank = {destination[0]} : i64, destination_device = {destination[1]} : i64, initialization = {str(initialization).lower()}}}> {{dist.rank = {destination[0]} : i64, dist.device = {destination[1]} : i64}} : ({ty}) -> {ty}'
        )
        transfer_id += 1
        return output

    # One scalar at slot zero; every branch computes x+4w and rotates out/back.
    lines.append(
        f'    %w = "ckks.encode"() <{{payload = dense<1.0> : tensor<1xf64>}}> {{dist.logical_id = {logical_id} : i64, dist.rank = 0 : i64, dist.device = -1 : i64}} : () -> {plaintext}'
    )
    logical_id += 1
    copies = [
        (transfer("%arg0", (0, -1), p, True), transfer("%w", (0, -1), p, True, True))
        for p in places
    ]
    results = []
    for place, (x, w) in zip(places, copies):
        value = compute("addcp", [x, w], place, True)
        for _ in range(3):
            value = compute("addcp", [value, w], place, True)
        value = compute("rotatec", [value], place, offset=1)
        value = compute("rotatec", [value], place, offset=-1)
        results.append(
            value if place == places[0] else transfer(value, place, places[0], False)
        )
    value = results[0]
    for result in results[1:]:
        value = compute("addcc", [value, result], places[0])
    lines += [f"    return {value} : {ciphertext}", "  }", "}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hecate-opt",
        type=Path,
        default=ROOT
        / "third_party/ckks-runtime/third_party/dacapo/build/nix/bin/hecate-opt",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(
        (
            ROOT
            / "third_party/ckks-runtime/docs/operator-spec/v1/profiles/poseidon-ckks-gpu.v1.json"
        ).read_text()
    )
    spec["context"].update(
        context_id="poseidon-runtime-memory-smoke",
        poly_degree=4096,
        rns_moduli_log2=[30] * 7,
        max_modulus_log2=30,
        default_scale_log2=40,
    )
    spec["levels"] = {"lower_bound": 0, "upper_bound": 6}
    spec["boot_profiles"] = []
    spec["spec_id"] = "poseidon-runtime-memory-smoke-v1"
    spec["status"] = "validated"
    for name, support in spec["operators"].items():
        support["supported"] = name in ("add_cp", "add_cc", "rotate", "rescale")
        if "latency_us_by_level" in support:
            support["latency_us_by_level"] = [0] * 7
    spec["operators"]["rescale"]["max_levels_per_op"] = 4
    spec_path = out / "operator-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    for topology, counts in [("1gpu", [1]), ("4gpu", [4]), ("2x2", [2, 2])]:
        ir = out / (topology + ".mlir")
        ir.write_text(make_ir(counts))
        for variant in ("baseline", "release", "reuse"):
            prefix = out / (topology + "." + variant)
            passes = []
            if variant != "baseline":
                passes.append(
                    "plan-runtime-memory"
                    + ("{enable-reuse=false}" if variant == "release" else "")
                )
            passes += [
                f"estimate-runtime-memory{{prefix={prefix} operator-spec={spec_path}}}",
                f"emit-runtime-plan{{prefix={prefix} plan-id=104 target-id={spec['target_id']} operator-spec-id={spec['spec_id']} operator-spec-sha256={digest} context-id={spec['context']['context_id']}}}",
            ]
            subprocess.run(
                [
                    str(args.hecate_opt.resolve()),
                    str(ir),
                    "-p=builtin.module(func.func(" + ",".join(passes) + "))",
                    "-o",
                    str(prefix) + ".mlir",
                ],
                cwd=ROOT / "third_party/ckks-runtime/third_party/dacapo",
                check=True,
            )
        expected = {
            "indices": list(range(8)),
            "values": [5 * sum(counts)] + [0] * 7,
            "absolute_tolerance": 1e-4,
        }
        (out / (topology + ".expected.json")).write_text(
            json.dumps(expected, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
