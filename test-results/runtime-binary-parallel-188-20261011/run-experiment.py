#!/usr/bin/env python3
"""Reuse the existing full binary plan; no JSON conversion or payload hashing."""
import argparse
import json
import resource
import subprocess
from pathlib import Path


def limits():
    budget = 32 << 30
    resource.setrlimit(resource.RLIMIT_AS, (budget, budget))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--operator-spec", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / "no-sha-parallel-20261011"
    work.mkdir()
    source = root / "full-qwen-binary-corrected-20261011/qwen24.v3.plan.bin"
    tool = root / "build-o2/runtime_binary_io_experiment"
    spec = args.operator_spec.resolve()
    inputs = {"source": str(source), "source_bytes": source.stat().st_size,
              "operator_spec": str(spec), "threads": [1, 2, 4, 8], "runs": 2,
              "address_space_limit_bytes": 32 << 30, "timeout_seconds": 180,
              "source_hashing": False, "blob_payloads_read": 0,
              "cache_condition": "shared server, no cache eviction"}
    (work / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")

    def invoke(label, *arguments):
        print(f"starting {label}", flush=True)
        with (work / f"{label}.json").open("w") as output, (work / f"{label}.stderr").open("w") as errors:
            subprocess.run(["nice", "-n", "10", str(tool), *map(str, arguments)], check=True,
                           stdout=output, stderr=errors, preexec_fn=limits, timeout=180)
        result = json.loads((work / f"{label}.json").read_text())
        assert result["hash_seconds"] == 0 and result["blob_payloads_read"] == 0
        assert result["values"] == 11064667 and result["instructions"] == 22149239
        print(json.dumps({"label": label, **result}), flush=True)
        return result

    comparison = invoke("field-comparison", "compare-binary-readers", source, 8)
    assert comparison["all_fields_equal"]
    results = []
    stream = invoke("stream-no-sha", "plan-binary", source, spec)
    results.append({"label": "stream-no-sha", **stream})
    for run, order in enumerate(([1, 2, 4, 8], [8, 4, 2, 1]), 1):
        for threads in order:
            label = f"parallel-{threads}-run-{run}"
            result = invoke(label, "plan-binary-parallel", source, spec, threads)
            assert result["capabilities"] == stream["capabilities"] == 4
            assert result["keys"] == stream["keys"] == 42566
            results.append({"label": label, "run": run, **result})
    (work / "loads.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
