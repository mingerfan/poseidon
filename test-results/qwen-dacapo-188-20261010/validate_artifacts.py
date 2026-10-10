"""Check exported plan references and bundle hashes, without executing CKKS."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
summaries = []
for path in sorted((ROOT / "artifacts").glob("*.runtime-plan.json")):
    plan = json.loads(path.read_text())
    bundle = path.with_name(path.name.replace(".runtime-plan.json", ".bundle"))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert plan["plaintext_bundle"]["manifest_sha256"] == (
        "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    for blob in manifest["blobs"]:
        data = (bundle / "data" / (blob["content"].split(":")[1] + ".bin")).read_bytes()
        assert len(data) == blob["byte_length"]
        assert blob["content"] == "sha256:" + hashlib.sha256(data).hexdigest()
    values = {v["id"] for v in plan["values"]}
    assert len(values) == len(plan["values"])
    defined = set(plan["external_inputs"])
    assert defined <= values
    for step in plan["initialization"] + plan["execution"]:
        assert set(step.get("inputs", [])) <= defined
        if "output" in step:
            assert step["output"] not in defined
            assert step["output"] in values
            defined.add(step["output"])
    assert set(plan["final_outputs"]) <= defined
    assert plan["target"]["operator_spec"]["source_sha256"] == (
        "sha256:" + hashlib.sha256((ROOT / "profiles/operator-spec.json").read_bytes()).hexdigest())
    summaries.append(dict(
        plan=str(path.relative_to(ROOT)), value_count=len(values),
        execution_steps=len(plan["execution"]),
        op_counts=dict(Counter(step.get("op", step["kind"]) for step in plan["execution"])),
        bundle_blobs=len(manifest["blobs"]),
        bundle_bytes=sum(blob["byte_length"] for blob in manifest["blobs"]),
        bundle_hashes_verified=True, operator_spec_hash_verified=True,
        topological_references_verified=True,
        encrypted_runtime_execution_tested=False))
(ROOT / "artifact-validation.json").write_text(json.dumps(summaries, indent=2) + "\n")
print(json.dumps(summaries, indent=2))
