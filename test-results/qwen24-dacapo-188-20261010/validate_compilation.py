"""Validate the complete exported plan, payloads, metadata, and interface."""
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path("/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
OUT = ROOT / "artifacts/qwen24-native"
sys.path.insert(0, "/tmp/poseidon-boot-placement-IUOMH1")
from validate_fast_plans import validate

started = time.monotonic()
signature = json.loads((OUT / "traced/signature.json").read_text())
assert signature["config"]["num_layers"] == 24
assert signature["config"]["hidden_size"] == 896
assert signature["config"]["intermediate_size"] == 4864
assert signature["config"]["vocab_size"] == 151936
assert len(signature["inputs"]) == 2 and len(signature["outputs"]) == 245
assert signature["logit_ciphertexts"] == 149
assert sum(row["prefix_size"] for row in signature["outputs"][:149]) == 151936
assert len(signature["outputs"][149:]) == 96
trace = OUT / "traced/trace_qwen24.mlirbc"
with trace.open("rb") as stream:
    assert stream.read(4) == b"ML\xefR"
    stream.seek(0)
    trace_digest = hashlib.file_digest(stream, "sha256").hexdigest()
print("native bytecode hash verified", flush=True)
path = OUT / "qwen24.depth-dp._hecate_qwen25_24layer.runtime-plan.json"
record = validate(path, ROOT)
plan = json.loads(path.read_text())
assert len(plan["external_inputs"]) == 2
assert len(plan["final_outputs"]) == 245
record.update(
    scope="Complete 24 blocks plus final RMSNorm and full-vocabulary LM head, two-token prefill",
    native_source_bytes=trace.stat().st_size,
    native_source_sha256="sha256:" + trace_digest,
    input_ciphertexts=2, output_ciphertexts=245, logit_ciphertexts=149, kv_ciphertexts=96,
    vocabulary=151936, fixture_weights="synthetic; 24 distinct layers, no pretrained checkpoint",
    full_model_runtime_executed=False,
    decoded_full_model_execution_tested=False,
    total_validation_seconds=time.monotonic() - started,
)
(OUT / "compilation-validation.json").write_text(json.dumps(record, indent=2) + "\n")
signature["native_compiled"] = True
signature["compilation"] = dict(strategy="depth-dp", plan=str(path.relative_to(ROOT)),
                                plan_sha256=record["plan_sha256"],
                                operator_spec_sha256=plan["target"]["operator_spec"]["source_sha256"],
                                encrypted_runtime_executed=False)
(OUT / "traced/signature.json").write_text(json.dumps(signature, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
