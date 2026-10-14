#!/usr/bin/env python3
"""Verify a half-NTT view against saved, complete CPU Encode results.

This is a CPU corpus check, not a GPU consumer implementation or timing test.
Binary paths are resolved from the repository root, as in benchmark manifests.
"""
import argparse
import array
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for manifest in args.manifest:
        report = json.loads(manifest.read_text())
        # These benchmark reports use a fixed degree and uint32 GPU residues.
        degree = 65536
        for case in report["cases"]:
            matches = [entry for entry in case["files"]
                       if entry["representation"].endswith("-ntt-gpu-uint32")]
            if len(matches) != 1:
                raise ValueError(f"{manifest}: expected one dumped NTT buffer")
            entry = matches[0]
            raw = Path(entry["file"]).read_bytes()
            if len(raw) != entry["bytes"] or len(raw) != report["batch"] * report["q_limbs"] * degree * 4:
                raise ValueError("unexpected NTT payload size")
            values = array.array("I")
            if values.itemsize != 4:
                raise ValueError("requires a four-byte unsigned int array")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
            mismatches = sum(
                values[start + i] != values[start + degree - 1 - i]
                for start in range(0, len(values), degree)
                for i in range(degree // 2)
            )
            rows.append({
                "manifest": str(manifest),
                "payload_sha256": hashlib.sha256(raw).hexdigest(),
                "pattern": case["pattern"],
                "batch": report["batch"],
                "q_limbs": report["q_limbs"],
                "integer_antisymmetry_exact": case["integer_real_antisymmetry_exact"],
                "pairs_checked": len(values) // 2,
                "mirror_mismatches": mismatches,
                "full_ntt_bytes": len(raw),
                "half_ntt_bytes": len(raw) // 2,
                "full_to_half_ratio": 2,
            })
    result = {
        "scope": "CPU verification of existing real-slot corpus; no GPU timing or runtime integration",
        "degree": 65536,
        "ntt_order": "Poseidon bit-reversed negacyclic NTT",
        "identity": "P[b,limb,i] == P[b,limb,N-1-i]",
        "half_view_index": "i < N/2 ? i : N-1-i",
        "basis": "a[N-i]=-a[i], a[N/2]=0 imply a(X^-1)=a(X) modulo X^N+1; inverse NTT roots have equal evaluations",
        "proposed_consumer": "Store [batch,limb,N/2] uint32 NTT values; multiply_plain reads via half_view_index without materializing a full plaintext",
        "limits": [
            "Enable only after exact symmetry verification; complex slots or asymmetric rounded coefficients may fail.",
            "The index relation depends on the NTT root order; this check is for Poseidon's current order.",
            "A compressed-aware consumer needs new plaintext layout metadata and kernel reads; existing GpuEvaluator requires a full layout.",
            "No standalone unpack, RNS or NTT would be needed; address calculation still uses ordinary instructions, and bandwidth/latency impact is unmeasured.",
            "Eight synthetic batch-one cases from two existing dumped manifests, not pretrained Qwen or arbitrary inputs.",
        ],
        "all_exact": bool(rows) and all(row["mirror_mismatches"] == 0 for row in rows),
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_exact"]:
        raise SystemExit("NTT mirror verification failed; do not enable half layout")
    print(f"{len(rows)} cases, {sum(row['pairs_checked'] for row in rows)} pairs: exact NTT mirror verified")


if __name__ == "__main__":
    main()
