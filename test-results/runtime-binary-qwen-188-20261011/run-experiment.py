#!/usr/bin/env python3
"""Convert the existing full Qwen JSON once, compare fields, measure a fresh binary process."""
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
    parser.add_argument("source", type=Path)
    parser.add_argument("--operator-spec", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work_dir.resolve() if args.work_dir else root / "full-qwen-binary-20261011"
    work.mkdir(exist_ok=True)
    binary = root / "build-o2/runtime_binary_io_experiment"
    spec = args.operator_spec.resolve()
    json.loads(spec.read_text())
    packed = work / "qwen24.v3.plan.bin"
    if packed.exists():
        raise FileExistsError(packed)
    metadata = {
        "source": str(args.source.resolve()), "source_bytes": args.source.stat().st_size,
        "binary": str(binary), "output": str(packed), "operator_spec": str(spec),
        "address_space_limit_bytes": 32 << 30, "process_timeout_seconds": 900,
        "cache_condition": "shared server, no page cache eviction; binary read after conversion",
        "json_full_loads": 1, "blob_payloads_read": 0,
    }
    (work / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def invoke(name, *arguments):
        print(f"starting {name}", flush=True)
        with (work / f"{name}.json").open("w") as output, (work / f"{name}.stderr").open("w") as errors:
            subprocess.run(["nice", "-n", "10", str(binary), *map(str, arguments)],
                           stdout=output, stderr=errors, check=True, timeout=900, preexec_fn=limits)
        result = json.loads((work / f"{name}.json").read_text())
        print(json.dumps({"stage": name, "result": result}), flush=True)
        return result

    converted = invoke("conversion-roundtrip", "convert-plan-check", args.source, packed, spec)
    loaded = invoke("binary-load", "plan-binary", packed, spec)
    baseline = converted["json_baseline"]
    if baseline["values"] != loaded["values"] or baseline["instructions"] != loaded["instructions"]:
        raise ValueError("fresh binary load counts differ")
    summary = {
        "values": baseline["values"], "instructions": baseline["instructions"],
        "json_bytes": baseline["bytes"], "binary_bytes": loaded["bytes"],
        "json_load_seconds": baseline["load_seconds"], "binary_load_seconds": loaded["load_seconds"],
        "load_speedup": baseline["load_seconds"] / loaded["load_seconds"],
        "parse_build_speedup": baseline["parse_build_seconds"] / loaded["parse_build_seconds"],
        "json_load_verify_seconds": baseline["load_seconds"] + baseline["verify_seconds"],
        "binary_load_verify_seconds": loaded["load_seconds"] + loaded["verify_seconds"],
        "all_fields_equal": converted["all_fields_equal"], "blob_payloads_read": 0,
    }
    summary["load_verify_speedup"] = summary["json_load_verify_seconds"] / summary["binary_load_verify_seconds"]
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
