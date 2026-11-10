#!/usr/bin/env python3
"""Check the two recorded single-GPU runs and print their acceptance summary."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fields(line):
    return dict(token.split("=", 1) for token in line.split()[1:])


def main():
    plan = json.loads((ROOT / "compiled-json/scheduled.resnet20_manual.runtime-plan.json").read_text())
    expected = [int(ins["attrs"]["call_id"]) for ins in plan["execution"]
                if ins["kind"] == "compute"]
    result = {"single_gpu": True, "task_granularity": "prepared_api", "runs": {}}
    for name in ("bin", "json"):
        lines = (ROOT / f"network-{name}.log").read_text().splitlines()
        run = fields(next(line for line in lines if line.startswith("PREPARED_RUNTIME_RESULT ")))
        order = [int(v) for v in next(line for line in lines
                 if line.startswith("PREPARED_RUNTIME_CALL_ORDER ")).split()[1:]]
        assert order == expected
        assert run["accuracy"] == "PASS" and run["exact_rns"] == "true"
        assert int(run["tasks"]) == int(run["runtime_compute_calls"]) == 77
        assert int(run["bootstraps"]) == 18 and int(run["encrypted_logits_q"]) == 4
        assert int(run["exact_residues"]) == 524288
        assert float(run["max_logit_error"]) <= 0.1
        assert run["prediction"] == run["reference_prediction"] == "3"
        assert run["intermediate_reencryptions"] == "0"
        assert run["scale_planning"] == run["boot_planning"] == "false"
        if run["cupti"] == "true":
            activity = fields(next(line for line in lines if line.startswith("GPU_ACTIVITY_TIMING ")))
            assert activity["host_transfers"] == activity["host_transfer_bytes"] == "0"
            run["gpu_activity"] = activity
        run["executed_compiler_order"] = True
        run["reordered_from_source"] = order != list(range(77))
        result["runs"][name] = run
    print(json.dumps(result, indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
