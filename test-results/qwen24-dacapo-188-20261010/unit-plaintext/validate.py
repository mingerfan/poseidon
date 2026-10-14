"""Validate full GPU plans, Release/reuse lifetimes, payloads and object peaks."""
import argparse
from array import array
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import heapq
import json
from pathlib import Path
import struct
import time
from decoded_graph import decoded_graph

ROOT = Path("/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
OUT = ROOT / "artifacts/qwen24-unit-plaintext/gpu-plans"
BASE = ROOT / "artifacts/qwen24-native"
OP_CODES = {name: index for index, name in enumerate(("mul_cc", "relinearize", "mul_cp", "rescale", "rotate",
                                                    "add_cc", "add_cp", "boot", "mod_switch", "negate"))}


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()



def reference():
    path = BASE / "qwen24.depth-dp._hecate_qwen25_24layer.runtime-plan.json"
    validation = json.loads((BASE / "compilation-validation.json").read_text())
    assert digest(path) == validation["plan_sha256"]
    plan = json.loads(path.read_text())
    record = {**decoded_graph(plan), "native_value_count": len(plan["values"]),
              "op_counts": validation["op_counts"], "final_outputs": plan["final_outputs"],
              "base_plan_sha256": validation["plan_sha256"]}
    (OUT / "semantic-reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def validate(devices):
    started = time.monotonic()
    directory = OUT / f"gpu{devices}"
    path = directory / "qwen24._hecate_qwen25_24layer.runtime-plan.json"
    ref = json.loads((OUT / "semantic-reference.json").read_text())
    plan = json.loads(path.read_text())
    assert plan["format_version"] == 2
    assert plan["target"]["world_size"] == 1 and plan["target"]["device_counts"] == [devices]
    assert len(plan["external_inputs"]) == 2 and len(plan["final_outputs"]) == 245
    spec_path = ROOT / "profiles/operator-spec.json"
    spec = json.loads(spec_path.read_text())
    assert digest(spec_path) == plan["target"]["operator_spec"]["source_sha256"]
    profiles = {p["profile_id"]: p for p in spec["boot_profiles"]}
    values = plan["values"]
    count = len(values)
    state = bytearray(count)
    producer = bytearray(count)  # 1 external, 2 encode, 3 compute, 4 transfer
    remaining = array("I", [0]) * count
    roots = array("I", range(count))
    born = array("I", [0]) * count
    retired = array("I", [2**32 - 1]) * count
    blocks = array("I", range(count))
    sizes = array("H", [0]) * count  # units of N * 4 bytes
    places = bytearray(count)
    final = bytearray(count)
    for value in plan["final_outputs"]:
        final[int(value)] = 1
    for index, value in enumerate(values):
        assert int(value["id"]) == index
        assert 0 <= value["level"] <= 39 and value["components"] in (1, 2, 3)
        assert value["kind"] == ("plaintext" if value["components"] == 1 else "ciphertext")
        assert value["context"] == spec["context"]["context_id"] and value["ntt"]
        place = value["place"]
        assert place["rank"] == 0
        if place["kind"] == "host":
            word_multiplier = 2
        else:
            assert place["kind"] == "device" and 0 <= place["index"] < devices
            places[index] = place["index"] + 1
            word_multiplier = 1
        sizes[index] = (value["level"] + 1) * value["components"] * word_multiplier
    for phase in ("initialization", "execution", "finalization"):
        for step in plan[phase]:
            for value in step.get("inputs", []):
                remaining[int(value)] += 1
    stats = [{"plaintext": 0, "ciphertext": 0, "peak": -1, "allocated": 0, "peak_event": None,
              "plaintext_peak": 0, "ciphertext_peak": 0}
             for _ in range(devices + 1)]
    unit = spec["context"]["poly_degree"] * 4
    aggregate_gpu_peak = {"bytes": 0, "instruction_ordinal": None, "plaintext_bytes": 0, "ciphertext_bytes": 0}

    def allocate(index):
        assert state[index] == 0
        state[index] = 1
        row = stats[places[index]]
        row[values[index]["kind"]] += sizes[index] * unit
        row["allocated"] += sizes[index] * unit

    def sample(phase, ordinal, op):
        for row in stats:
            row["plaintext_peak"] = max(row["plaintext_peak"], row["plaintext"])
            row["ciphertext_peak"] = max(row["ciphertext_peak"], row["ciphertext"])
            current = row["plaintext"] + row["ciphertext"]
            if current > row["peak"]:
                row["peak"] = current
                row["peak_event"] = {"phase": phase, "instruction_ordinal": ordinal, "operation": op,
                                     "plaintext_bytes": row["plaintext"], "ciphertext_bytes": row["ciphertext"]}
        plaintext = sum(row["plaintext"] for row in stats[1:])
        ciphertext = sum(row["ciphertext"] for row in stats[1:])
        if plaintext + ciphertext > aggregate_gpu_peak["bytes"]:
            aggregate_gpu_peak.update(bytes=plaintext + ciphertext, instruction_ordinal=ordinal,
                                      plaintext_bytes=plaintext, ciphertext_bytes=ciphertext)

    for value in plan["external_inputs"]:
        index = int(value)
        producer[index] = 1
        assert places[index] == 0 and values[index]["kind"] == "ciphertext"
        allocate(index)
    sample("external_inputs", None, "external_inputs")
    counts = Counter()
    instruction_counts = Counter()
    total_steps = 0
    for phase in ("initialization", "execution", "finalization"):
        for step in plan[phase]:
            ordinal = step["ordinal"]
            assert ordinal == total_steps
            total_steps += 1
            kind = step["kind"]
            instruction_counts[kind] += 1
            op = kind
            if kind == "release":
                index = int(step["value"])
                assert state[index] == 1 and remaining[index] == 0 and not final[index]
                state[index] = 2
                if producer[blocks[index]] != 1:
                    stats[places[index]][values[index]["kind"]] -= sizes[index] * unit
                    retired[index] = ordinal + 1
            else:
                inputs = [int(v) for v in step.get("inputs", [])]
                assert all(state[v] == 1 for v in inputs)
                for index in inputs:
                    assert remaining[index] > 0
                    remaining[index] -= 1
                outputs = [int(v) for v in step.get("outputs", [step.get("output")])]
                assert len(outputs) == 1
                index = outputs[0]
                assert state[index] == 0
                born[index] = ordinal + 1
                after = values[index]
                before = [values[v] for v in inputs]
                if kind == "encode":
                    assert phase == "initialization" and places[index] == 0 and after["kind"] == "plaintext"
                    producer[index] = 2
                elif kind == "transfer":
                    assert len(inputs) == 1 and step["sources"] == [before[0]["place"]]
                    assert step["destinations"] == [after["place"]]
                    assert step["output_kinds"] == [after["kind"]]
                    assert before[0]["place"] != after["place"]
                    assert all(before[0][key] == after[key] for key in ("kind", "context", "level", "scale_log2", "ntt", "components"))
                    roots[index] = roots[inputs[0]]
                    producer[index] = 4
                else:
                    assert kind == "compute" and phase == "execution"
                    producer[index] = 3
                    op = step["op"]
                    counts[op] += 1
                    assert all(v["place"] == after["place"] == step["place"] for v in before)
                    if op in ("mul_cc", "mul_cp", "add_cc", "add_cp"):
                        assert before[0]["level"] == before[1]["level"] == after["level"]
                        if op.startswith("mul"):
                            assert after["scale_log2"] == sum(v["scale_log2"] for v in before)
                        else:
                            assert before[0]["scale_log2"] == before[1]["scale_log2"] == after["scale_log2"]
                        assert after["components"] == (sum(v["components"] for v in before) - 1 if op == "mul_cc" else before[0]["components"])
                    elif op in ("rescale", "mod_switch"):
                        assert after["level"] == step["attrs"]["target_level"] < before[0]["level"]
                        assert before[0]["components"] == after["components"]
                        if op == "rescale":
                            assert before[0]["level"] - after["level"] <= 4
                            assert after["scale_log2"] == step["attrs"]["target_scale_log2"]
                        else:
                            assert before[0]["scale_log2"] == after["scale_log2"]
                    elif op == "boot":
                        profile = profiles[step["attrs"]["operator_profile"]]
                        assert places[index] == 0 and profile["input_level_min"] <= before[0]["level"] <= profile["input_level_max"]
                        assert before[0]["components"] == profile["input_components"]
                        assert after["level"] == profile["output_level"] == step["attrs"]["target_level"]
                        assert after["scale_log2"] == profile["output_scale_log2"] == step["attrs"]["target_scale_log2"]
                        assert after["components"] == profile["output_components"]
                    elif op == "relinearize":
                        assert before[0]["components"] == 3 and after["components"] == 2
                        assert before[0]["level"] == after["level"] and before[0]["scale_log2"] == after["scale_log2"]
                    else:
                        assert op in ("rotate", "negate")
                        assert all(before[0][key] == after[key] for key in ("kind", "level", "scale_log2", "components"))
                if "reuse_input" in step:
                    assert kind == "compute" and step["reuse_input"] == 0
                    old = inputs[0]
                    assert remaining[old] == 0 and not final[old] and producer[old] == 3
                    assert op in (("negate", "rotate") if places[index] == 0 else ("add_cp", "rotate"))
                    assert all(values[old][key] == after[key] for key in ("kind", "place", "context", "level", "scale_log2", "ntt", "components"))
                    state[old] = 2
                    state[index] = 1
                    blocks[index] = blocks[old]
                    retired[old] = ordinal + 1
                    instruction_counts["reuse_input"] += 1
                else:
                    allocate(index)
            operation = {"encode": "ckks.encode", "transfer": "dist.transfer", "release": "dist.release"}.get(kind)
            if kind == "compute":
                operation = "ckks." + {"mul_cc": "mulcc", "mul_cp": "mulcp", "add_cc": "addcc", "add_cp": "addcp",
                                         "rotate": "rotatec", "negate": "negatec", "boot": "bootstrapc",
                                         "rescale": "rescalec", "mod_switch": "modswitchc", "relinearize": "relinearize"}[op]
            sample(phase, ordinal, operation)
            if ordinal % 1000000 == 0:
                print("validated", devices, ordinal, flush=True)
    assert counts["boot"] == ref["op_counts"]["boot"]
    assert all(state[int(v)] == 1 for v in plan["final_outputs"])
    assert all(state[i] == 2 or final[i] or producer[i] == 1 for i in range(count))
    logical = decoded_graph(plan)
    assert logical["decoded_graph_sha256"] == ref["decoded_graph_sha256"]
    assert logical["final_metadata"] == ref["final_metadata"]
    semantic = logical["decoded_graph_sha256"]
    memory = json.loads((directory / "qwen24._hecate_qwen25_24layer.memory.json").read_text())
    assert memory["memory_planned"]
    for row in memory["places"]:
        place = row["place"]
        index = 0 if place["kind"] == "host" else place["index"] + 1
        actual = stats[index]
        assert row["peak_bytes"] == actual["peak"]
        assert row["distinct_allocation_bytes"] == actual["allocated"]
        assert row["retained_bytes"] == actual["plaintext"] + actual["ciphertext"]
        for key, value in actual["peak_event"].items():
            assert row["peak"][key] == value, (devices, index, key, row["peak"], actual)
        event = 0 if row["peak"]["instruction_ordinal"] is None else row["peak"]["instruction_ordinal"] + 1
        largest = heapq.nsmallest(5, (i for i in range(count) if places[i] == index and born[i] <= event < retired[i]),
                                 key=lambda i: (-sizes[i], i))
        assert [str(i) for i in largest] == [v["value_id"] for v in row["peak"]["largest_objects"]]
    bundle = directory / "qwen24._hecate_qwen25_24layer.bundle"
    manifest_path = bundle / "manifest.json"
    assert digest(manifest_path) == plan["plaintext_bundle"]["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["bundle_id"] == plan["plaintext_bundle"]["id"]
    blobs = {b["content"]: b for b in manifest["blobs"]}
    assert len(blobs) == len(manifest["blobs"])
    for step in plan["initialization"]:
        if step["kind"] == "encode" and step["payload"]["kind"] == "bundle":
            assert step["payload"]["content"] in blobs

    def check_blob(blob):
        file = bundle / "data" / (blob["content"][7:] + ".bin")
        assert file.stat().st_size == blob["byte_length"] and digest(file) == blob["content"]

    with ThreadPoolExecutor(max_workers=12) as executor:
        for start in range(0, len(manifest["blobs"]), 4096):
            list(executor.map(check_blob, manifest["blobs"][start:start + 4096]))
    record = {"devices": devices, "plan": str(path.relative_to(ROOT)), "format_version": 2,
              "plan_sha256": digest(path), "plan_bytes": path.stat().st_size, "value_count": count,
              "instruction_counts": dict(instruction_counts), "op_counts": dict(counts),
              "input_ciphertexts": 2, "output_ciphertexts": 245,
              "decoded_graph_sha256": semantic, "matches_complete_decoded_host_graph": True,
              "full_unit_encode_count": logical["full_unit_encode_count"],
              "distinct_full_unit_types": logical["distinct_full_unit_types"],
              "hashes_verified": True, "topology_verified": True, "physical_metadata_verified": True,
              "release_reuse_verified": True, "memory_report_independently_verified": True,
              "bundle_blobs": len(blobs), "bundle_bytes": sum(b["byte_length"] for b in blobs.values()),
              "object_memory": [{"place_index": index - 1, "peak_bytes": row["peak"],
                                 "plaintext_peak_bytes": row["plaintext_peak"],
                                 "ciphertext_peak_bytes": row["ciphertext_peak"]}
                                for index, row in enumerate(stats)],
              "aggregate_gpu_peak": aggregate_gpu_peak,
              "runtime_execution_tested": False, "seconds": time.monotonic() - started}
    (directory / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--devices", type=int, choices=(1, 4))
    args = parser.parse_args()
    if args.reference:
        reference()
    else:
        assert args.devices
        validate(args.devices)
