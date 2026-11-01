"""Summarize independently verified full-model plans, without inferred results."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
rows = []
for strategy in ("greedy", "optimized"):
    for q in (31, 34):
        audit = json.loads((here / f"{strategy}-q{q}-fixed40-audit.jsonl").read_text())
        run = json.loads((here / f"regional-{strategy}-q{q}-s40/result.json").read_text())
        assert run["returncode"] == 0 and audit["verifier_passed"]
        assert audit["inputs"] == [{"q": q, "s": 40.0}]
        assert abs(audit["output_s"] - 40) < 1e-8
        rows.append(dict(strategy=strategy, input_q=q,
                         boots=audit["counts"].get("boot", 0),
                         output_q=audit["output_q"], output_s=audit["output_s"],
                         first_boot=audit["boots"][0], counts=audit["counts"],
                         compile_seconds=run["seconds"],
                         content_set_sha256=audit["content_set_sha256"]))
assert len({row["content_set_sha256"] for row in rows}) == 1
report = dict(results=rows, same_encoded_payload_content_set=True,
              measured_gpu_latency=False, encrypted_accuracy_validated=False,
              million_instruction_evaluation=False,
              boot_contract="Q6/s40 -> Q31/s40", arithmetic_floor_q=5,
              search="DaCapo single-prefix estimated-latency DP using regional scale evaluation; not a global optimum guarantee")
(here / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
for row in rows:
    print(row["strategy"], row["input_q"], row["boots"], row["output_q"],
          row["output_s"], row["compile_seconds"])
