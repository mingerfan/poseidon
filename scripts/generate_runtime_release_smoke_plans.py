#!/usr/bin/env python3
"""Generate hand-written four-GPU fanout plans, with and without Release."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def with_releases(plan):
    plan = copy.deepcopy(plan)
    plan["format_version"] = 2
    last_use = {value: 0 for value in plan["external_inputs"]}
    index = 0
    for phase in ("initialization", "execution", "finalization"):
        for instruction in plan[phase]:
            for value in instruction.get("inputs", []) + instruction.get("outputs", []):
                last_use[value] = index
            if "output" in instruction:
                last_use[instruction["output"]] = index
            index += 1
    for value in plan["final_outputs"]:
        del last_use[value]
    index = 0
    ordinal = 0
    for phase in ("initialization", "execution", "finalization"):
        instructions = []
        for instruction in plan[phase]:
            instruction["ordinal"] = ordinal
            ordinal += 1
            instructions.append(instruction)
            for value, last in last_use.items():
                if last == index:
                    instructions.append({"ordinal": ordinal, "kind": "release", "value": value})
                    ordinal += 1
            index += 1
        plan[phase] = instructions
    return plan


def make_plan(spec, digest, counts):
    host = {"kind": "host", "rank": 0}
    devices = [{"kind": "device", "rank": rank, "index": index}
               for rank, count in enumerate(counts) for index in range(count)]
    plan = {
        "format_version": 1, "plan_id": "103",
        "target": {"target_id": "poseidon-ckks-gpu", "capability_version": 1,
                   "world_size": len(counts), "device_counts": counts,
                   "operator_spec": {"id": spec["spec_id"], "version": spec["version"],
                                     "source_sha256": digest}},
        "values": [], "external_inputs": [], "initialization": [],
        "execution": [], "finalization": [], "final_outputs": [],
    }
    places = {}
    next_transfer = 100

    def value(place):
        value_id = str(len(plan["values"]))
        places[value_id] = place
        plan["values"].append({"id": value_id, "kind": "ciphertext", "place": place,
                               "context": spec["context"]["context_id"], "level": 6,
                               "scale_log2": spec["context"]["default_scale_log2"],
                               "ntt": True, "components": 2})
        return value_id

    def communicate(phase, source, destinations):
        nonlocal next_transfer
        outputs = [value(place) for place in destinations]
        plan[phase].append({"kind": "transfer" if len(outputs) == 1 else "replicate",
                            "transfer_id": str(next_transfer),
                            "hint": "point_to_point" if len(outputs) == 1 else "broadcast",
                            "inputs": [source], "outputs": outputs,
                            "sources": [places[source]], "destinations": destinations,
                            "output_kinds": ["ciphertext"] * len(outputs)})
        next_transfer += 1
        return outputs

    def compute(op, inputs):
        place = places[inputs[0]]
        output = value(place)
        plan["execution"].append({"kind": "compute", "op": op, "place": place,
                                  "inputs": inputs, "output": output})
        return output

    external = value(host)
    plan["external_inputs"] = [external]
    source = communicate("initialization", external, [devices[0]])[0]
    # One local copy and one grouped remote submission in the 2x2 case.
    copies = [source] + communicate("execution", source, devices[1:])
    partials = [compute("negate", [copies[index % 4]]) for index in range(10)]
    while len(partials) > 1:
        reduced = []
        for index in range(0, len(partials), 2):
            left = partials[index]
            if index + 1 == len(partials):
                reduced.append(left)
                continue
            right = partials[index + 1]
            if places[left] != places[right]:
                right = communicate("execution", right, [places[left]])[0]
            reduced.append(compute("add_cc", [left, right]))
        partials = reduced
    # Multi-rank device workers require Device-only online communication.
    # The numerical test runner downloads this final Device value for decoding.
    plan["final_outputs"] = communicate("finalization", partials[0], [devices[1]])
    ordinal = 0
    for phase in ("initialization", "execution", "finalization"):
        for instruction in plan[phase]:
            instruction["ordinal"] = ordinal
            ordinal += 1
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = ROOT / "third_party/ckks-runtime/docs/operator-spec/v1/profiles/poseidon-ckks-gpu.v1.json"
    spec = json.loads(profile.read_text(encoding="utf-8"))
    spec["spec_id"] = "poseidon-runtime-release-smoke-v1"
    spec["status"] = "validated"
    spec["context"].update(context_id="poseidon-runtime-release-smoke", poly_degree=4096,
                           rns_moduli_log2=[30] * 7, max_modulus_log2=30, default_scale_log2=40)
    spec["levels"] = {"lower_bound": 0, "upper_bound": 6}
    spec["boot_profiles"] = []
    for name, support in spec["operators"].items():
        support["supported"] = name in ("negate", "add_cc", "rescale")
        if "latency_us_by_level" in support:
            support["latency_us_by_level"] = [0] * 7
    # GPU preflight requires this backend capability even without a Rescale op.
    spec["operators"]["rescale"]["max_levels_per_op"] = 4
    spec_path = args.output_dir / "operator-spec.json"
    write_json(spec_path, spec)
    digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
    for name, counts in (("1x4", [4]), ("2x2", [2, 2])):
        plan = make_plan(spec, digest, counts)
        write_json(args.output_dir / f"{name}.baseline.json", plan)
        write_json(args.output_dir / f"{name}.release.json", with_releases(plan))
    print(f"Generated 1x4 and 2x2 baseline/Release plans in {args.output_dir}")


if __name__ == "__main__":
    main()
