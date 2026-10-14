#!/usr/bin/env python3
"""Recompute summaries from all three process runs and CPU codec reports."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


format_rows, codec_rows = [], []
for q in (8, 32):
    for batch in (1, 8):
        reports = [read(f"q{q}-b{batch}-run{run}.json") for run in (1, 2, 3)]
        for pattern in ("dense", "periodic1024", "prefix896", "scalar"):
            cases = [next(c for c in r["cases"] if c["pattern"] == pattern) for r in reports]
            row = {"q_limbs": q, "batch": batch, "pattern": pattern,
                   "coefficient_bits": [c["coefficient_bits"] for c in cases],
                   "encoded_gpu_bytes_per_plaintext": cases[0]["encoded_gpu_uint32_bytes"] // batch,
                   "half_integer_bytes_per_plaintext": statistics.median(
                       c["integer_realhalf_bitpacked_bytes"] / batch for c in cases),
                   "replication_group": cases[0]["replication_group"], "modes": {}}
            for mode in cases[0]["modes"]:
                modes = [next(m for m in c["modes"] if m["name"] == mode["name"]) for c in cases]
                assert all(m["exact_cpu_ntt_match"] for m in modes)
                value = {}
                for scope in ("device_only", "with_h2d"):
                    if scope not in mode:
                        continue
                    value[scope] = {stage + "_us_per_plaintext": statistics.median(
                        m[scope][stage]["median_ms"] * 1000 / batch for m in modes)
                        for stage in ("total", "h2d", "unpack_or_rns", "ntt", "wall")}
                if "decompressed_output_gb_per_second" in mode:
                    value["decompressed_output_gb_per_second"] = statistics.median(
                        m["decompressed_output_gb_per_second"] for m in modes)
                row["modes"][mode["name"]] = value
            format_rows.append(row)
        reports = [read(f"nvcomp-q{q}-b{batch}-run{run}.json") for run in (1, 2, 3)]
        assert all(r["nvcomp_version"] == "5.3.0.16" for r in reports)
        for first in reports[0]["rows"]:
            rows = [next(row for row in r["rows"] if row["pattern"] == first["pattern"]
                         and row["representation"] == first["representation"]
                         and row["algorithm"] == first["algorithm"]) for r in reports]
            assert all(row["exact_round_trip"] for row in rows)
            item = {"q_limbs": q, "batch": batch, "pattern": first["pattern"],
                    "representation": first["representation"], "algorithm": first["algorithm"],
                    "ratio": statistics.median(r["ratio"] for r in rows),
                    "raw_bytes_per_plaintext": first["raw_bytes"] / batch,
                    "compressed_bytes_per_plaintext": statistics.median(
                        r["compressed_bytes"] / batch for r in rows)}
            for scope in ("device_only", "with_h2d"):
                item[scope] = {"us_per_plaintext": statistics.median(
                    r[scope]["ms_per_plaintext"] * 1000 for r in rows),
                    "decompressed_output_gb_per_second": statistics.median(
                        r[scope]["decompressed_output_gb_per_second"] for r in rows)}
            codec_rows.append(item)

write("format-summary.json", {"runs": 3, "all_exact_cpu_ntt_matches": True,
      "aggregation": "median of per-process medians; event includes host submission; no locked clocks",
      "rows": format_rows})
write("nvcomp-summary.json", {"runs": 3, "all_exact_round_trips": True,
      "aggregation": "median of process medians; 10 async API calls per sample; Python dispatch gaps included; no locked clocks",
      "format": "GPU-generated nvCOMP-native 64KiB chunks; batch8 duplicates one payload",
      "scope": "CUDA decode, not Tensor Core; CPU payload interoperability not tested", "rows": codec_rows})
cpu_rows = []
for q in (8, 32):
    report = read(f"cpu-codecs-q{q}.json")
    for row in report["rows"]:
        for codec in row["codecs"]:
            assert codec["exact_round_trip"]
            cpu_rows.append({"q_limbs": q, "pattern": row["pattern"],
                "representation": row["representation"], "algorithm": codec["algorithm"],
                "level": codec["level"], "raw_bytes": row["raw_bytes"],
                "compressed_bytes": codec["compressed_bytes"], "ratio": codec["ratio"],
                "cpu_decompress_gb_per_second": codec["decompress"]["gb_per_second"]})
write("cpu-codec-summary.json", {"versions": report["versions"],
      "scope": "CPU whole-buffer Zstd1/3/9 and LZ4 raw block; GPU speeds are in nvcomp-summary",
      "rows": cpu_rows})
print("all summaries written; every correctness flag passed")
