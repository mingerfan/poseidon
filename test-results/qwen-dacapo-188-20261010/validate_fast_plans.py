"""Validate exported metadata and hashes; optionally interpret decoded slots.

The NumPy interpreter treats Boot/rescale/modswitch/relinearize as identities
on decoded values. It checks graph semantics, not encrypted CKKS accuracy.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path, root, execute=False):
    started = time.monotonic()
    plan = json.loads(path.read_text())
    bundle = path.with_name(path.name.replace(".runtime-plan.json", ".bundle"))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert digest(manifest_path) == plan["plaintext_bundle"]["manifest_sha256"]
    assert manifest["bundle_id"] == plan["plaintext_bundle"]["id"]
    blobs = {blob["content"]: blob for blob in manifest["blobs"]}
    assert len(blobs) == len(manifest["blobs"])
    for content, blob in blobs.items():
        file = bundle / "data" / (content[7:] + ".bin")
        assert file.stat().st_size == blob["byte_length"]
        assert digest(file) == content
    spec_path = root / "profiles/operator-spec.json"
    spec = json.loads(spec_path.read_text())
    assert digest(spec_path) == plan["target"]["operator_spec"]["source_sha256"]
    profiles = {p["profile_id"]: p for p in spec["boot_profiles"]}
    values = {value["id"]: value for value in plan["values"]}
    assert len(values) == len(plan["values"])
    defined = set(plan["external_inputs"])
    assert defined <= values.keys()
    for step in plan["initialization"]:
        assert step["kind"] == "encode" and step["output"] not in defined
        assert values[step["output"]]["kind"] == "plaintext"
        if step["payload"]["kind"] == "bundle":
            assert step["payload"]["content"] in blobs
        defined.add(step["output"])
    counts = Counter()
    for step in plan["execution"]:
        assert step["kind"] == "compute"
        assert set(step["inputs"]) <= defined
        assert step["output"] not in defined
        before = [values[value] for value in step["inputs"]]
        after = values[step["output"]]
        op = step["op"]
        counts[op] += 1
        assert 0 <= after["level"] <= 39
        assert all(x["context"] == after["context"] and x["ntt"] == after["ntt"]
                   and x["place"] == after["place"] for x in before)
        if op in ("add_cc", "add_cp", "mul_cc", "mul_cp"):
            assert before[0]["level"] == before[1]["level"] == after["level"]
            if op.startswith("add"):
                assert before[0]["scale_log2"] == before[1]["scale_log2"] == after["scale_log2"]
            else:
                assert after["scale_log2"] == sum(x["scale_log2"] for x in before)
            expected_components = (sum(x["components"] for x in before) - 1
                                   if op == "mul_cc" else before[0]["components"])
            assert after["components"] == expected_components
        elif op in ("rescale", "mod_switch"):
            assert after["level"] == step["attrs"]["target_level"] < before[0]["level"]
            assert after["components"] == before[0]["components"]
            if op == "rescale":
                assert before[0]["level"] - after["level"] <= 4
                assert after["scale_log2"] == step["attrs"]["target_scale_log2"]
            else:
                assert after["scale_log2"] == before[0]["scale_log2"]
        elif op == "boot":
            profile = profiles[step["attrs"]["operator_profile"]]
            assert profile["input_level_min"] <= before[0]["level"] <= profile["input_level_max"]
            assert before[0]["components"] == profile["input_components"]
            assert after["level"] == profile["output_level"] == step["attrs"]["target_level"]
            assert after["scale_log2"] == profile["output_scale_log2"] == step["attrs"]["target_scale_log2"]
            assert after["components"] == profile["output_components"]
        elif op == "relinearize":
            assert before[0]["components"] == 3 and after["components"] == 2
            assert before[0]["level"] == after["level"]
            assert before[0]["scale_log2"] == after["scale_log2"]
        else:
            assert op in ("rotate", "negate")
            assert all(before[0][key] == after[key] for key in
                       ("level", "scale_log2", "components"))
        defined.add(step["output"])
    assert set(plan["final_outputs"]) <= defined
    record = dict(plan=str(path.relative_to(root)), plan_bytes=path.stat().st_size,
                  plan_sha256=digest(path), value_count=len(values),
                  execution_steps=len(plan["execution"]), op_counts=dict(counts),
                  bundle_blobs=len(blobs), bundle_bytes=sum(x["byte_length"] for x in blobs.values()),
                  hashes_verified=True, topology_verified=True, physical_metadata_verified=True,
                  encrypted_execution_tested=False)
    if execute:
        import numpy as np
        slots = 32768
        inputs = np.load(root / "artifacts/fixture/input_slots.npy", allow_pickle=False)
        storage = {value: inputs[index].copy()
                   for index, value in enumerate(plan["external_inputs"])}
        cache = {}
        for step in plan["initialization"]:
            payload = step["payload"]
            if payload["kind"] == "bundle":
                content = payload["content"]
                if content not in cache:
                    file = bundle / "data" / (content[7:] + ".bin")
                    cache[content] = np.frombuffer(file.read_bytes(), dtype="<f8")
                array = cache[content]
            else:
                array = np.asarray(payload["values"], dtype=np.float64)
            assert 0 < len(array) <= slots and np.isfinite(array).all()
            storage[step["output"]] = array
        remaining = Counter(value for step in plan["execution"] for value in step["inputs"])
        remaining.update(plan["final_outputs"])
        for index, step in enumerate(plan["execution"]):
            operands = [storage[value] for value in step["inputs"]]
            op = step["op"]
            if op == "mul_cc":
                result = operands[0] * operands[1]
            elif op == "add_cc":
                result = operands[0] + operands[1]
            elif op == "mul_cp":
                result = np.zeros(slots)
                result[:len(operands[1])] = operands[0][:len(operands[1])] * operands[1]
            elif op == "add_cp":
                result = operands[0].copy()
                result[:len(operands[1])] += operands[1]
            elif op == "rotate":
                result = np.roll(operands[0], -step["attrs"]["steps"])
            elif op == "negate":
                result = -operands[0]
            else:
                assert op in ("boot", "rescale", "mod_switch", "relinearize")
                result = operands[0]
            storage[step["output"]] = result
            for value in step["inputs"]:
                remaining[value] -= 1
                if remaining[value] == 0:
                    del storage[value]
            if index % 50000 == 0:
                print("interpreted", index, "of", len(plan["execution"]), flush=True)
        result = storage[plan["final_outputs"][0]]
        reference = np.load(root / "artifacts/numpy-dsl-reference/prefill.npy", allow_pickle=False)[:1792]
        difference = float(np.max(np.abs(result[:len(reference)] - reference)))
        tail = float(np.max(np.abs(result[len(reference):])))
        np.testing.assert_allclose(result[:len(reference)], reference, rtol=1e-10, atol=1e-10)
        assert tail <= 1e-12
        record.update(decoded_numpy_execution_verified=True,
                      reference_max_absolute_difference=difference, tail_max_absolute=tail,
                      note="Ideal decoded arithmetic only; no CKKS noise or encrypted Boot is simulated")
    record["validation_seconds"] = time.monotonic() - started
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--strategy", choices=("depth-dp", "greedy"), default="depth-dp")
    args = parser.parse_args()
    records = []
    suffix = "depth-dp-fixed" if args.strategy == "depth-dp" else "lazy-fixed"
    report = "fast-plan-validation.json" if args.strategy == "depth-dp" else "greedy-plan-validation.json"
    for path in sorted((args.root / "artifacts").glob(f"*.{suffix}.*.runtime-plan.json")):
        print("validating", path.name, flush=True)
        record = validate(path, args.root, args.execute_full and path.name.startswith("block-prefill."))
        records.append(record)
        (args.root / "artifacts" / report).write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
