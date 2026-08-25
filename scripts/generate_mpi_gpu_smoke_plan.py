#!/usr/bin/env python3
"""Generate the small 1x4 Dacapo/Poseidon MPI GPU smoke plan."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "third_party" / "ckks-runtime"
DACAPO = RUNTIME / "third_party" / "dacapo"
PROFILE_GENERATOR = (
    RUNTIME / "integrations" / "dacapo" /
    "generate_mlp_gpu_e2e_profiles.py"
)
DEFAULT_HECATE_OPT = DACAPO / "build" / "nix" / "bin" / "hecate-opt"
SOURCE = (
    ROOT / "src" / "poseidon" / "tests" / "runtime_api" / "fixtures" /
    "mpi_gpu_fanout.mlir"
)


def parse_x_list(encoded: str, minimum: int, what: str) -> list[int]:
    try:
        values = [int(item) for item in encoded.split("x")]
    except ValueError as error:
        raise ValueError(f"{what} must be an x-separated integer list") from error
    if not values or any(value < minimum for value in values):
        raise ValueError(f"{what} entries must be at least {minimum}")
    return values


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_communication_profile(rank_to_node: list[int]) -> dict:
    rules = []
    if len(rank_to_node) > 1:
        rules = [
            {
                "id": "rank0-device0-to-rank1-device0",
                "from": {"kind": "device", "rank": 0, "device": 0},
                "to": {"kind": "device", "rank": 1, "device": 0},
                "direction": "both",
                "transport": "nccl",
                "cost": {
                    "startup_latency_us": 4,
                    "max_rate_bytes_per_us": 80000,
                    "saturation_bytes": 524288,
                },
            },
            {
                "id": "node0-to-node1-fallback",
                "from": {
                    "kind": "device",
                    "node": rank_to_node[0],
                    "rank": "*",
                    "device": "*",
                },
                "to": {
                    "kind": "device",
                    "node": rank_to_node[1],
                    "rank": "*",
                    "device": "*",
                },
                "direction": "both",
                "transport": "nccl",
                "cost": {
                    "startup_latency_us": 20,
                    "max_rate_bytes_per_us": 20000,
                    "saturation_bytes": 1048576,
                },
            },
        ]
    return {
        "format_version": 2,
        "coefficient_bytes": 4,
        "topology": {"rank_to_node": rank_to_node},
        "links": {
            "host_device": {
                "startup_latency_us": 8,
                "max_rate_bytes_per_us": 12000,
                "saturation_bytes": 1048576,
            },
            "intra_rank": {
                "startup_latency_us": 2,
                "max_rate_bytes_per_us": 100000,
                "saturation_bytes": 524288,
            },
            "inter_rank": {
                "startup_latency_us": 20,
                "max_rate_bytes_per_us": 20000,
                "saturation_bytes": 1048576,
            },
        },
        "rules": rules,
    }


def source_place(communication: dict, destination_index: int) -> dict:
    sources = communication.get("sources", [])
    if len(sources) == 1:
        return sources[0]
    return sources[destination_index]


def validate_plan(plan: dict, device_counts: list[int]) -> dict:
    target = plan.get("target", {})
    if target.get("world_size") != len(device_counts):
        raise RuntimeError("generated plan has the wrong world size")
    if target.get("device_counts") != device_counts:
        raise RuntimeError("generated plan has the wrong device counts")
    if len(plan.get("external_inputs", [])) != 1 or len(
        plan.get("final_outputs", [])
    ) != 1:
        raise RuntimeError("generated plan must have one input and one output")

    values = {value["id"]: value for value in plan["values"]}
    input_desc = values[plan["external_inputs"][0]]
    if input_desc["place"].get("kind") != "host":
        raise RuntimeError("generated plan external input is not on Host")

    instructions = (
        plan.get("initialization", [])
        + plan.get("execution", [])
        + plan.get("finalization", [])
    )
    computes = [item for item in instructions if item.get("kind") == "compute"]
    negates = [item for item in computes if item.get("op") == "negate"]
    adds = [item for item in computes if item.get("op") == "add_cc"]
    if len(negates) != 10 or len(adds) != 9 or len(computes) != 19:
        raise RuntimeError("generated plan does not preserve the fanout graph")

    expected_places = {
        (rank, device)
        for rank, count in enumerate(device_counts)
        for device in range(count)
    }
    compute_places = {
        (item["place"]["rank"], item["place"]["index"])
        for item in computes
    }
    if compute_places != expected_places:
        raise RuntimeError(
            "placement did not use every configured GPU: "
            f"expected={sorted(expected_places)} actual={sorted(compute_places)}"
        )

    cross_rank = []
    cross_rank_device = []
    cross_rank_host_uploads = []
    local_device_transfers = []
    exact_pair = []
    for item in instructions:
        if item.get("kind") not in ("transfer", "replicate"):
            continue
        destinations = item.get("destinations", [])
        for index, destination in enumerate(destinations):
            source = source_place(item, index)
            if source["rank"] == destination["rank"]:
                if (source["kind"] == "device" and
                        destination["kind"] == "device" and
                        source.get("index") != destination.get("index")):
                    local_device_transfers.append((source, destination))
                continue
            cross_rank.append((source, destination))
            if source["kind"] == "host" and destination["kind"] == "device":
                cross_rank_host_uploads.append((source, destination))
                continue
            if source["kind"] != "device" or destination["kind"] != "device":
                raise RuntimeError(
                    "generated plan contains an unsupported remote transfer"
                )
            cross_rank_device.append((source, destination))
            endpoints = {
                (source["rank"], source["index"]),
                (destination["rank"], destination["index"]),
            }
            if endpoints == {(0, 0), (1, 0)}:
                exact_pair.append((source, destination))
    if len(device_counts) > 1 and not cross_rank_device:
        raise RuntimeError("generated plan contains no cross-rank Device transfer")
    if len(device_counts) > 1 and not exact_pair:
        raise RuntimeError(
            "generated plan never exercises the first exact communication rule"
        )
    if len(device_counts) == 1 and device_counts[0] > 1 and not local_device_transfers:
        raise RuntimeError("generated plan contains no same-rank Device transfer")
    return {
        "compute_places": sorted(compute_places),
        "cross_rank_transfers": len(cross_rank),
        "cross_rank_device_transfers": len(cross_rank_device),
        "cross_rank_host_uploads": len(cross_rank_host_uploads),
        "local_device_transfers": len(local_device_transfers),
        "exact_rule_transfers": len(exact_pair),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hecate-opt", type=Path, default=DEFAULT_HECATE_OPT)
    parser.add_argument("--device-counts", default="1x4")
    parser.add_argument("--rank-to-node", default="0x1")
    parser.add_argument("--poly-degree", type=int, default=4096)
    args = parser.parse_args()

    device_counts = parse_x_list(args.device_counts, 1, "device-counts")
    rank_to_node = parse_x_list(args.rank_to_node, 0, "rank-to-node")
    if len(device_counts) != len(rank_to_node):
        raise ValueError("device-counts and rank-to-node lengths differ")
    if args.poly_degree < 2 or args.poly_degree & (args.poly_degree - 1):
        raise ValueError("poly-degree must be a power of two")
    if not args.hecate_opt.is_file():
        raise FileNotFoundError(f"hecate-opt not found: {args.hecate_opt}")
    if not SOURCE.is_file():
        raise FileNotFoundError(f"MLIR fixture not found: {SOURCE}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(PROFILE_GENERATOR),
            "--output-dir",
            str(output_dir),
            "--poly-degree",
            str(args.poly_degree),
            "--profile-preset",
            "synthetic",
        ],
        cwd=ROOT,
        check=True,
    )

    operator_spec_path = output_dir / "operator-spec.json"
    operator_spec = json.loads(operator_spec_path.read_text(encoding="utf-8"))
    operator_spec["spec_id"] = "poseidon-ckks-gpu-mpi-smoke-v1"
    operator_spec["context"]["context_id"] = "poseidon-ckks-gpu-mpi-smoke"
    operator_spec["provenance"]["path"] = "scripts/generate_mpi_gpu_smoke_plan.py"
    level_count = len(operator_spec["context"]["rns_moduli_log2"])
    for name in ("negate", "add_cc"):
        operator_spec["operators"][name]["latency_us_by_level"] = [
            1000
        ] * level_count
    write_json(operator_spec_path, operator_spec)

    communication_profile_path = output_dir / "communication-profile.json"
    write_json(
        communication_profile_path,
        make_communication_profile(rank_to_node),
    )
    spec_digest = "sha256:" + hashlib.sha256(
        operator_spec_path.read_bytes()
    ).hexdigest()

    prefix = output_dir / "mpi-gpu-smoke"
    placement = (
        "assign-ckks-placement{"
        f"device-counts={args.device_counts} "
        f"operator-spec={operator_spec_path} "
        "intra-rank-communication-cost=10 "
        "inter-rank-communication-cost=20 "
        f"communication-profile={communication_profile_path}"
        "}"
    )
    emit = (
        "emit-runtime-plan{"
        f"prefix={prefix} "
        "plan-id=104 "
        "target-id=poseidon-ckks-gpu "
        "capability-version=1 "
        f"operator-spec-id={operator_spec['spec_id']} "
        f"operator-spec-version={operator_spec['version']} "
        f"operator-spec-sha256={spec_digest} "
        f"context-id={operator_spec['context']['context_id']} "
        "device-count=0 ntt=true"
        "}"
    )
    pipeline = (
        "builtin.module(func.func("
        f"{placement},materialize-ckks-communication,{emit}"
        "))"
    )
    placed_mlir = output_dir / "mpi-gpu-smoke.placed.mlir"
    subprocess.run(
        [
            str(args.hecate_opt.resolve()),
            str(SOURCE),
            f"-p={pipeline}",
            "-o",
            str(placed_mlir),
        ],
        cwd=DACAPO,
        check=True,
    )

    plan_path = output_dir / "mpi-gpu-smoke.mpi_gpu_fanout.runtime-plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"RuntimePlan was not generated: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = validate_plan(plan, device_counts)
    summary_path = output_dir / "generation-summary.json"
    write_json(
        summary_path,
        {
            "format_version": 1,
            "device_counts": device_counts,
            "rank_to_node": rank_to_node,
            "plan": str(plan_path),
            "operator_spec": str(operator_spec_path),
            "communication_profile": str(communication_profile_path),
            **summary,
        },
    )
    print(f"RuntimePlan: {plan_path}")
    print(f"OperatorSpec: {operator_spec_path}")
    print(f"communication profile: {communication_profile_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
