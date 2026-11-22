#!/usr/bin/env python3
"""CPU-only check that a prepared RuntimePlan obeys the engine sequence contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(manifest: dict, plan: dict) -> dict:
    matrices = {m["matrix_id"]: m for m in manifest["matrices"]}
    calls = {str(c["call_id"]): c for c in manifest["calls"]}
    assert len(calls) == len(manifest["calls"]), "duplicate prepared call id"
    state = None
    completed = []
    seen = set()
    for instruction in plan["execution"]:
        if instruction.get("op") != "ApiCall":
            continue
        attrs = instruction["attrs"]
        if attrs["callee"] != "bert.compact_linear":
            continue
        cid = str(attrs["call_id"])
        assert cid in calls and cid not in seen, "missing/duplicate RuntimePlan prepared call"
        seen.add(cid)
        call = calls[cid]
        mid = call["matrix_id"]
        matrix = matrices[mid]
        count = matrix["output_width"] // matrix["tile"]
        output = call["output_tile"]
        physical = tuple(call[k] for k in ("input_level", "input_scale_log2", "output_level",
                                           "output_scale_log2", "rescale_count"))
        assert physical[0] - physical[2] == physical[4], "incorrect physical rescale drop"
        assert call["input_width"] == matrix["input_width"]
        assert call["output_width"] == matrix["output_width"]
        assert len(instruction["inputs"]) == matrix["input_width"] // matrix["tile"]
        if state is None:
            assert output == 0, "matrix output sequence does not start with output0"
            state = (mid, 0, instruction["inputs"], physical)
        expected_mid, expected_output, expected_inputs, expected_physical = state
        assert (mid, output) == (expected_mid, expected_output), "interleaved or reordered matrix output calls"
        assert instruction["inputs"] == expected_inputs, "matrix sequence changes input SSA values"
        assert physical == expected_physical, "one prepared matrix has multiple physical specs"
        if output + 1 == count:
            completed.append(dict(matrix_id=mid, outputs=count,
                                  input_level=physical[0], output_level=physical[2]))
            state = None
        else:
            state = (mid, output + 1, expected_inputs, expected_physical)
    assert state is None, "incomplete final matrix output sequence"
    assert seen == set(calls), "manifest calls are not all present in RuntimePlan"
    # A prepared id cannot be reused with a different physical spec, even in
    # a separate sequence, without choosing a different engine id.
    by_matrix = {}
    for call in calls.values():
        spec = tuple(call[k] for k in ("input_level", "input_scale_log2", "output_level",
                                       "output_scale_log2", "rescale_count"))
        assert call["matrix_id"] not in by_matrix or by_matrix[call["matrix_id"]] == spec
        by_matrix[call["matrix_id"]] = spec
    return dict(sequence_contract="ok", prepared_calls=len(calls), sequences=completed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("runtime_plan", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.manifest.read_bytes()), json.loads(args.runtime_plan.read_bytes()))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
