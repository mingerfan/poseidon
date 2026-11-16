#!/usr/bin/env python3
"""Compare common parameter roots from separate full/sparse microbench processes."""
import argparse
import json
import sys
from pathlib import Path


def compare(full, sparse):
    if full["status"] != "ok" or sparse["status"] != "ok":
        raise ValueError("both benchmark processes must finish with status=ok")
    for field in ("N", "q_count", "p_count", "q_moduli", "p_moduli"):
        if full["parameters"][field] != sparse["parameters"][field]:
            raise ValueError(f"parameter mismatch: {field}")
    a, b = full["gpu_parameters"], sparse["gpu_parameters"]
    if a["mode"] != "full" or b["mode"] not in ("sparse", "top"):
        raise ValueError("expected full.json then sparse.json (or top.json)")
    if a["fingerprint_mode"] != "full" or b["fingerprint_mode"] != "full":
        raise ValueError("use --check full for BOTH runs; sampled fingerprints are not full-table checks")
    if a["fingerprint_algorithm"] != b["fingerprint_algorithm"]:
        raise ValueError("fingerprint algorithms differ")
    reference = {(level["q_count"], level["p_count"]): level for level in a["levels"]}
    if not b["levels"]:
        raise ValueError("sparse run contains no levels")
    checked_families = checked_bytes = 0
    for level in b["levels"]:
        key = level["q_count"], level["p_count"]
        if key not in reference:
            raise ValueError(f"sparse level absent from full run: {key}")
        expected = reference[key]
        if len(level["shards"]) != len(expected["shards"]):
            raise ValueError(f"shard count differs at level {key}")
        for actual_shard, expected_shard in zip(level["shards"], expected["shards"]):
            if (len(actual_shard["root_families"]) != 7 or
                    actual_shard["root_families"].keys() != expected_shard["root_families"].keys()):
                raise ValueError(f"expected the same seven root families at level {key}")
            for family, actual in actual_shard["root_families"].items():
                wanted = expected_shard["root_families"][family]
                if actual != wanted:
                    raise ValueError(f"root bytes/count/hash differ: level={key}, family={family}")
                checked_families += 1
                checked_bytes += actual["allocated_bytes"]
    return {"status": "ok", "checked_sparse_levels": len(b["levels"]),
            "checked_root_families": checked_families, "checked_root_bytes": checked_bytes,
            "comparison": "all-word 64-bit fingerprints, not collision-free bytewise comparison"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_json", type=Path)
    parser.add_argument("sparse_json", type=Path)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.full_json.read_text()), json.loads(args.sparse_json.read_text()))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"parameter comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
