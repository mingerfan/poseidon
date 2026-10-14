#!/usr/bin/env python3
"""Fail-closed validation for the resident, continuous ResNet20 performance log."""

from __future__ import annotations

import pathlib
import re
import sys


def one(lines: list[str], prefix: str) -> str:
    matches = [line for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {prefix!r} line, got {len(matches)}")
    return matches[0]


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([^\s=]+)=([^\s]+)", line))


def summarize(text: str) -> dict[str, float | int]:
    lines = text.splitlines()
    memory = fields(one(lines, "MEMORY "))
    key_preflight = fields(one(lines, "APPLICATION_KEY_PREFLIGHT "))
    keys_ready = fields(one(lines, "APPLICATION_KEYS_READY "))
    compression = fields(one(lines, "BOOTSTRAP_QP_COMPRESSION "))
    inventory = fields(one(lines, "CONTINUOUS_PREPARED "))
    activity = fields(one(lines, "GPU_ACTIVITY_TIMING "))
    relu_moddrop = fields(one(lines, "RELU_MODDROP "))
    accuracy = fields(one(lines, "POST_TIMING_ACCURACY "))
    result = fields(one(lines, "CONTINUOUS_TIMING_RESULT "))
    basis_fusion = result.get("relu_basis_fusion", "false")
    if basis_fusion not in ("true", "false"):
        raise ValueError("invalid ReLU basis fusion flag")
    basis_fused = basis_fusion == "true"
    if "relu_basis_fusion" in result:
        for prefix, preparation in (("RELU_BASIS_FUSION_PREPARED ", True),
                                    ("RELU_BASIS_FUSION ", False)):
            basis = fields(one(lines, prefix))
            expected_basis = {
                "enabled": basis_fusion,
                "calls": "285" if basis_fused else "0",
                "constants": "190" if basis_fused else "0",
                "products": "95" if basis_fused else "0",
                "exact_checks": "285" if basis_fused and preparation else "0",
                "exact_residues": "824311808" if basis_fused and preparation else "0",
                "online_observer": "false", "coefficients_changed": "false",
                "rescale_changed": "false",
            }
            for name, expected in expected_basis.items():
                if basis.get(name) != expected:
                    raise ValueError(f"{prefix}{name} differs from {expected!r}")
    elif any(line.startswith("RELU_BASIS_FUSION") for line in lines):
        raise ValueError("ReLU basis fusion inventory has no result flag")
    conv_batch = result.get("conv_plain_batch", "false")
    if conv_batch not in ("true", "false"):
        raise ValueError("invalid Conv plain batch flag")
    batched = conv_batch == "true"
    if "conv_plain_batch" in result:
        for prefix, preparation in (("CONV_PLAIN_BATCH_PREPARED ", True),
                                    ("CONV_PLAIN_BATCH ", False)):
            batch = fields(one(lines, prefix))
            expected_batch = {
                "enabled": conv_batch,
                "chains": "160" if batched else "0",
                "terms": "1440" if batched else "0",
                "calls": "480" if batched else "0",
                "exact_checks": "480" if batched and preparation else "0",
                "exact_residues": "566231040" if batched and preparation else "0",
                "pending": "0", "online_observer": "false",
                "coefficients_changed": "false", "rescale_changed": "false",
            }
            for name, expected in expected_batch.items():
                if batch.get(name) != expected:
                    raise ValueError(f"{prefix}{name} differs from {expected!r}")
    elif any(line.startswith("CONV_PLAIN_BATCH") for line in lines):
        raise ValueError("Conv plain batch inventory has no result flag")
    leaf_fusion = result.get("relu_leaf_fusion", "false")
    if leaf_fusion not in ("true", "false"):
        raise ValueError("invalid ReLU leaf fusion flag")
    fused = leaf_fusion == "true"
    # Historical logs predate this optimization. New logs must report its
    # complete inventory, for either side of a same-binary A/B comparison.
    if "relu_leaf_fusion" in result:
        leaf = fields(one(lines, "RELU_LEAF_FUSION "))
        expected_leaf = {
            "enabled": leaf_fusion,
            "calls": "266" if fused else "0",
            "terms": "570" if fused else "0",
            "adds_removed": "304" if fused else "0",
            "exact_checks": "0",
            "online_observer": "false",
            "coefficients_changed": "false",
            "rescale_changed": "false",
        }
        for name, expected in expected_leaf.items():
            if leaf.get(name) != expected:
                raise ValueError(f"ReLU leaf fusion {name} differs from {expected!r}")
    elif any(line.startswith("RELU_LEAF_FUSION ") for line in lines):
        raise ValueError("ReLU leaf fusion inventory has no result flag")
    breakdown_rows = [fields(line) for line in lines
                      if line.startswith("GPU_ACTIVITY_BREAKDOWN ")]
    category_rows = [fields(line) for line in lines
                     if line.startswith("GPU_ACTIVITY_CATEGORY ")]

    required = {
        "mode": "single_resident_online_inference",
        "blocks": "9",
        "bootstraps": "18",
        "dnum": "2",
        "input_q": "32",
        "stem_output_q": "31",
        "application_keyswitch": "fixed_dnum2",
        "hoisted_rotation": "true",
        "host_transfers": "0",
        "input_upload_included": "false",
        "public_material_upload_included": "false",
        "per_stage_sync": "false",
        "observer_decryptions": "0",
        "post_timing_decryptions": "1",
        "accuracy": "PASS",
        "intermediate_reencryptions": "0",
        "swaps": "false",
        "compressed_qp_plaintexts": "true",
        "bootstrap_c2s_baby_tile": "8",
        "relu_zero_copy_moddrop": "true",
        "relu_q_prefix_views": "true",
        "encrypted_logits_q": "4",
    }
    for name, expected in required.items():
        if result.get(name) != expected:
            raise ValueError(f"performance result {name}={result.get(name)!r}, expected {expected!r}")
    if accuracy.get("result") != "PASS":
        raise ValueError("post-timing encrypted-logit accuracy did not pass")
    if accuracy.get("max_logit_error") != result.get("max_logit_error"):
        raise ValueError("post-timing accuracy/result error mismatch")
    if memory.get("pool_cap_GiB") != "30" or memory.get("pool_initial_GiB") != "30":
        raise ValueError("performance run did not use the fixed 30 GiB resident pool")
    expected_key_preflight = {
        "level_rotation_MiB": "1469",
        "level_relin_MiB": "363",
        "level_total_MiB": "1832",
        "active_parameter_ntt_MiB": "862.75",
        "resident": "true",
        "swaps": "false",
    }
    for name, expected in expected_key_preflight.items():
        if key_preflight.get(name) != expected:
            raise ValueError(
                f"application key preflight {name}={key_preflight.get(name)!r}, "
                f"expected {expected!r}")
    expected_keys_ready = {
        "rotation_levels": "6",
        "relin_levels": "13",
        "relin_contexts": "11",
        "parameter_levels": "19",
        "payload_MiB": "1832",
        "dnum": "2",
        "resident": "true",
        "swaps": "false",
    }
    for name, expected in expected_keys_ready.items():
        if keys_ready.get(name) != expected:
            raise ValueError(
                f"resident application keys {name}={keys_ready.get(name)!r}, "
                f"expected {expected!r}")
    if compression.get("enabled") != "true" or compression.get("exact") != "true":
        raise ValueError("bootstrap QP plaintext compression is not exact and enabled")
    if compression.get("stc_diagonals") != "158" or compression.get("cts_diagonals") != "158":
        raise ValueError("bootstrap compressed QP diagonal inventory changed")
    full_qp_mib = float(compression["full_MiB"])
    compact_qp_mib = float(compression["compact_MiB"])
    compression_ratio = float(compression["ratio"])
    if not (0 < compact_qp_mib < full_qp_mib and compression_ratio > 1):
        raise ValueError("bootstrap compressed QP storage did not shrink")
    expected_inventory = {
        "inputs": "27",
        "stem_plaintexts": "28",
        "conv_plaintexts": "1696",
        "shortcut_plaintexts": "48",
        "relu_plaintexts": "46",
        "head_plaintexts": "93",
        "bootstrap_s2c_baby_tile": "8",
        "bootstrap_c2s_baby_tile": "8",
        "bootstrap_c2s_tile_batches": "3",
    }
    for name, expected in expected_inventory.items():
        if inventory.get(name) != expected:
            raise ValueError(f"resident inventory {name}={inventory.get(name)!r}, expected {expected!r}")
    if activity.get("host_transfers") != "0" or activity.get("host_transfer_bytes") != "0":
        raise ValueError("CUPTI activity summary contains a host/device transfer")
    expected_relu_moddrop = {
        "zero_copy": "true",
        "q_prefix_views": "true",
        "materialized_calls": "19",
        "inplace_calls": "494",
        "inplace_discarded_q_limbs": "0",
        "prefix_multiply_calls": "513",
        "prefix_multiply_plain_calls": str((95 if fused else 665) - (95 if basis_fused else 0)),
        "prefix_source_views": "1463",
        "prefix_discarded_q_limbs": "1159",
    }
    for name, expected in expected_relu_moddrop.items():
        if relu_moddrop.get(name) != expected:
            raise ValueError(
                f"ReLU ModDrop {name}={relu_moddrop.get(name)!r}, "
                f"expected {expected!r}")
    if any(line.startswith(("GPU_ACTIVITY_STAGE ", "TIMING_PREPARED_", "BOOTSTRAP_GPU_TIMING "))
           for line in lines):
        raise ValueError("fragmented/stage-local timing leaked into performance log")

    expected_calls = {
        "bootstrap.C2S": 18,
        "bootstrap.EvalMod": 36,
        "bootstrap.ModRaise": 18,
        "bootstrap.S2C": 18,
        "bootstrap.prepare": 18,
        "bootstrap.recombine": 18,
        "conv_bn": 18,
        "head": 1,
        "relu": 19,
        "residual.add": 9,
        "residual.clone": 9,
        "shortcut": 2,
        "stem.conv_bn": 1,
    }
    breakdown: dict[str, float] = {}
    for row in breakdown_rows:
        category = row.get("category", "")
        if category in breakdown:
            raise ValueError(f"duplicate GPU breakdown category {category!r}")
        if category not in expected_calls:
            raise ValueError(f"unexpected GPU breakdown category {category!r}")
        if int(row.get("calls", "-1")) != expected_calls[category]:
            raise ValueError(f"GPU breakdown call count changed for {category}")
        breakdown[category] = float(row["gpu_ms"])
    if breakdown.keys() != expected_calls.keys():
        missing = sorted(expected_calls.keys() - breakdown.keys())
        raise ValueError(f"missing GPU breakdown categories: {missing}")

    wall = float(result["wall_ms"])
    event = float(result["cuda_event_ms"])
    gpu = float(result["gpu_activity_union_ms"])
    enqueue = float(result["host_enqueue_ms"])
    activities = int(result["activities"])
    # The start CUDA event is timestamped in the GPU stream before the host
    # wall clock starts.  CUPTI callback overhead can therefore make the CUDA
    # event interval slightly longer than wall time.  Both intervals must
    # still contain the GPU activity union, and their disagreement is bounded
    # to a small instrumentation allowance.
    clock_tolerance_ms = max(50.0, wall * 0.01)
    if not (0 < gpu <= event and gpu <= wall and
            abs(event - wall) <= clock_tolerance_ms and
            0 < enqueue <= wall and activities > 0):
        raise ValueError("invalid continuous timing relationship")
    if abs(float(activity["gpu_total_ms"]) - gpu) > 1e-6:
        raise ValueError("CUPTI summary/result mismatch")
    if int(activity["activities"]) != activities:
        raise ValueError("CUPTI activity count mismatch")
    breakdown_sum = sum(breakdown.values())
    if abs(breakdown_sum - gpu) > 1e-3:
        raise ValueError("GPU breakdown does not partition the total activity union")
    bootstrap = sum(milliseconds for category, milliseconds in breakdown.items()
                    if category.startswith("bootstrap."))
    categories = {
        "Conv+BN": breakdown["conv_bn"] + breakdown["stem.conv_bn"],
        "ReLU": breakdown["relu"],
        "Bootstrap": bootstrap,
        "Shortcut": breakdown["shortcut"],
        "Pool+FC": breakdown["head"],
        "Residual/copy": (breakdown["residual.add"] +
                          breakdown["residual.clone"]),
    }
    category_sum = sum(categories.values())
    if abs(category_sum - gpu) > 1e-3:
        raise ValueError("six-category report does not partition the total GPU activity")

    if category_rows:
        reported_categories: dict[str, float] = {}
        for row in category_rows:
            category = row.get("category", "")
            if category in reported_categories:
                raise ValueError(f"duplicate report category {category!r}")
            if category not in categories:
                raise ValueError(f"unexpected report category {category!r}")
            reported_categories[category] = float(row["gpu_ms"])
        if reported_categories.keys() != categories.keys():
            missing = sorted(categories.keys() - reported_categories.keys())
            raise ValueError(f"missing report categories: {missing}")
        for category, expected in categories.items():
            if abs(reported_categories[category] - expected) > 1e-6:
                raise ValueError(f"reported category total changed for {category}")

    return {
        "wall_ms": wall,
        "cuda_event_ms": event,
        "gpu_activity_union_ms": gpu,
        "host_enqueue_ms": enqueue,
        "activities": activities,
        "prepared_free_MiB": float(inventory["free_MiB"]),
        "bootstrap_qp_full_MiB": full_qp_mib,
        "bootstrap_qp_compact_MiB": compact_qp_mib,
        "bootstrap_qp_compression_ratio": compression_ratio,
        "bootstrap_c2s_baby_tile": int(inventory["bootstrap_c2s_baby_tile"]),
        "bootstrap_c2s_tile_batches": int(inventory["bootstrap_c2s_tile_batches"]),
        "relu_materialized_moddrops": int(relu_moddrop["materialized_calls"]),
        "relu_zero_copy_moddrops": int(relu_moddrop["inplace_calls"]),
        "relu_q_prefix_source_views": int(relu_moddrop["prefix_source_views"]),
        "relu_q_prefix_discarded_q_limbs": int(
            relu_moddrop["prefix_discarded_q_limbs"]),
        "relu_fused_leaf_calls": 266 if fused else 0,
        "relu_fused_leaf_terms": 570 if fused else 0,
        "relu_fused_basis_calls": 285 if basis_fused else 0,
        "relu_fused_basis_constants": 190 if basis_fused else 0,
        "relu_fused_basis_products": 95 if basis_fused else 0,
        "conv_plain_batch_calls": 480 if batched else 0,
        "conv_plain_batch_terms": 1440 if batched else 0,
        "breakdown_sum_ms": breakdown_sum,
        "breakdown_overlap_ms": breakdown_sum - gpu,
        "conv_bn_gpu_ms": categories["Conv+BN"],
        "relu_gpu_ms": categories["ReLU"],
        "bootstrap_gpu_ms": categories["Bootstrap"],
        "shortcut_gpu_ms": categories["Shortcut"],
        "pool_fc_gpu_ms": categories["Pool+FC"],
        "residual_copy_gpu_ms": categories["Residual/copy"],
        "category_sum_ms": category_sum,
        "max_logit_error": float(accuracy["max_logit_error"]),
        "gpu_prediction": int(accuracy["gpu_prediction"]),
    }


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_performance.py LOG")
    path = pathlib.Path(sys.argv[1])
    summary = summarize(path.read_text())
    print("PERFORMANCE_LOG_VALIDATION result=PASS " + " ".join(
        f"{name}={value}" for name, value in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
