"""Validate saved reports and export within-process CPU-FFT comparisons."""
import json
from pathlib import Path
import statistics

base = Path(__file__).resolve().parent
analysis = {"source_commit": json.loads((base / "environment.json").read_text())["source_commit"],
            "measurement_reports": 0, "rows": []}
for directory in ["continuous", "demand"]:
    reports = [json.loads(p.read_text()) for p in sorted((base / directory).glob("q*-run*.json"))]
    assert len(reports) == 12, (directory, len(reports))
    for r in reports:
        assert r["compare_cpu_fft"] and r["input_h2d"]
        assert r["main_stream_id"] != r["encode_stream_id"]
        prepared = r["encode_preparation"]
        assert prepared["cpu_fft_validation"]["cpu_exact_match"]
        assert prepared["cpu_fft_validation"]["cpu_ntt_residue_differences"] == 0
        assert prepared["cpu_fft_validation"]["max_input_error"] <= prepared["cpu_fft_validation"]["tolerance"]
        assert prepared["full_gpu_validation"]["max_input_error"] <= prepared["full_gpu_validation"]["tolerance"]
        assert prepared["invalid_coefficients_rejected"]
        assert prepared["coefficient_cache_bytes"] == r["batch"] * 65536 * 8
        assert prepared["coefficient_cache_bytes"] == prepared["raw_cache_bytes"] * 2
        for standalone in prepared["standalone"].values():
            assert len(standalone["samples_ms"]) == 30
        assert len(r["rows"]) == (2 if directory == "demand" else 6)
        for row in r["rows"]:
            assert row["cpu_exact_match"]
            assert len(row["backgrounds"]) == (2 if directory == "demand" else 4)
            for b in row["backgrounds"].values():
                assert b["operator_and_encode_exact_match"]
                assert len(b["operator"]["samples_ms"]) == 30
                assert b["operator_calls"] == 120
                assert b["input_bytes_per_batch"] == r["batch"] * 65536 * 8 / (1 if b["precomputed_cpu_fft"] else 2)
            if directory == "demand":
                periods = [b["target_batch_period_ms"] for b in row["backgrounds"].values()]
                assert periods[0] == periods[1] and periods[0] > 0
    analysis["measurement_reports"] += len(reports)
    for q in [8, 32]:
        for batch in [1, 8]:
            group = [r for r in reports if r["q_limbs"] == q and r["batch"] == batch]
            assert len(group) == 3
            for operator in sorted({row["operator"] for r in group for row in r["rows"]}):
                selected = [row for r in group for row in r["rows"] if row["operator"] == operator]
                full_names = ["tilelang_tensor_encode_demand"] if directory == "demand" else ["cuda_encode", "tilelang_tensor_encode"]
                for full_name in full_names:
                    ratios = [row["backgrounds"]["cpu_fft_" + full_name]["operator"]["mean_ms"] /
                              row["backgrounds"][full_name]["operator"]["mean_ms"] for row in selected]
                    analysis["rows"].append({
                        "mode": directory, "q_limbs": q, "batch": batch, "operator": operator,
                        "ntt": "tensor" if "tensor" in full_name else "cuda",
                        "cached_over_full_operator_mean_ratio": statistics.median(ratios),
                        "paired_process_ratios": ratios,
                        "full_operator_mean_slowdowns": [row["backgrounds"][full_name]["operator_mean_slowdown"] for row in selected],
                        "cached_operator_mean_slowdowns": [row["backgrounds"]["cpu_fft_" + full_name]["operator_mean_slowdown"] for row in selected],
                        "full_supply_per_operator": [row["backgrounds"][full_name]["encoded_plaintexts_per_operator"] for row in selected],
                        "cached_supply_per_operator": [row["backgrounds"]["cpu_fft_" + full_name]["encoded_plaintexts_per_operator"] for row in selected],
                    })
(base / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
print(f"Validated {analysis['measurement_reports']} reports; exported {len(analysis['rows'])} paired comparisons")
