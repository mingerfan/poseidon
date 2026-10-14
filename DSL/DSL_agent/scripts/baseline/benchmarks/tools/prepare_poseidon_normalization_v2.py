"""Manual normalization expression checks; not an application or Agent path."""
import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
ROOT = (Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0, str(ROOT/"scripts/baseline"))
from benchmark_graph import require, digest, canonical


def expression(high, request):
    import numpy as np
    require(len(high["nodes"]) == 1, "Single normalization operator")
    node = high["nodes"][0]
    op = node["op"]
    require(op in ("layer_norm", "group_norm", "instance_norm"), "Normalization operator")
    shape = high["inputs"][0]["shape"]
    period = request["layout"]["input_slot_period"]
    count = math.prod(shape)
    require(request["layout"]["inputs"][0]["shape"] == shape and count <= period, "One packed input")
    require(node["attrs"]["approximation"] == dict(method="newton", domain=[.1, 4.], iterations=2),
            "Frozen two-iteration approximation")
    constants = request["public_constants"]
    def scalar(value):
        names = [k for k, v in constants.items() if type(v) in (float, int) and v == value]
        require(bool(names), "Bound scalar unavailable: "+repr(value))
        return names[0]
    for i in range(count):
        require(constants.get("mask"+str(i)) == [float(j == i) for j in range(period)], "Bound output mask")
    if op == "layer_norm":
        width = math.prod(node["attrs"]["normalized_shape"])
        gain = np.broadcast_to(np.asarray(high["constants"]["gain"]), shape).reshape(-1)
        bias = np.broadcast_to(np.asarray(high["constants"]["bias"]), shape).reshape(-1)
    else:
        groups = shape[1] if op == "instance_norm" else node["attrs"]["groups"]
        width = math.prod(shape[1:]) // groups
        broadcast_shape = [1, shape[1]] + [1]*(len(shape)-2)
        gain = np.broadcast_to(np.asarray(high["constants"]["gain"]).reshape(broadcast_shape), shape).reshape(-1)
        bias = np.broadcast_to(np.asarray(high["constants"]["bias"]).reshape(broadcast_shape), shape).reshape(-1)
    lines = ['@hc.func("c,c")', "def golden(x, zero_ct):"]
    sequence = 0
    def emit(value):
        nonlocal sequence
        name = "v"+str(sequence)
        sequence += 1
        lines.append("    "+name+" = "+value)
        return name
    def add(values):
        result = values[0]
        for value in values[1:]:
            result = emit(result+" + "+value)
        return result
    lanes = []
    for i in range(count):
        rotated = "x"
        for bit in range(period.bit_length()-1):
            step = 1 << bit
            if i & step:
                rotated = emit(rotated+".rotate("+str(step)+")")
        value = emit(rotated+" * mask0")
        shift = 1
        while shift < period:
            rotated = emit(value+".rotate("+str(shift)+")")
            value = emit(value+" + "+rotated)
            shift *= 2
        lanes.append(value)
    bias_parts = [emit("("+emit("zero_ct + "+scalar(float(bias[i])))+") * mask"+str(i))
                  for i in range(count)]
    packed_bias = add(bias_parts)
    results = []
    for start in range(0, count, width):
        indices = list(range(start, start+width))
        mean = emit(add([lanes[i] for i in indices])+" * "+scalar(1/width))
        centered = [emit(lanes[i]+" - "+mean) for i in indices]
        squares = [emit(z+" * "+z) for z in centered]
        variance = emit(add(squares)+" * "+scalar(1/width))
        variance = emit(variance+" + "+scalar(high["nodes"][0]["attrs"]["eps"]))
        factor = emit(variance+" * "+scalar(-.5))
        scaled = [emit(z+" * "+scalar(float(gain[i]))) for z, i in zip(centered, indices)]
        packed_gain = add([emit(z+" * mask"+str(i)) for z, i in zip(scaled, indices)])
        inverse = emit("zero_ct + "+scalar(.5))
        for _ in range(2):
            square = emit(inverse+" * "+inverse)
            correction = emit(square+" * "+factor)
            correction = emit(correction+" + "+scalar(1.5))
            inverse = emit(inverse+" * "+correction)
        results.append(emit(packed_gain+" * "+inverse))
    result = emit(add(results)+" + "+packed_bias)
    lines.append("    return ["+result+"]")
    return "\n".join(lines)+"\n"


def prepare(output):
    from hecate_python_env import WORK
    from compiler_configuration import configuration
    from unified_graph_contract import prepare as prepare_request, validate_candidate
    from rmsnorm_expression_probe import plaintext_check
    from semantic_benchmark_execution import runtime_sources
    case_root = ROOT/"scripts/baseline/cases/operator-extension-v5"
    manifest = json.loads((case_root/"manifest.json").read_text())
    cfg = configuration("poseidon-cpu-eva-w45-v1")
    require(not output.exists(), "Preserve previous preparation")
    output.mkdir()
    fixture_root = ROOT/"scripts/baseline/golden_cases/poseidon_normalization_v2"
    require(not fixture_root.exists(), "Preserve previous fixture")
    fixture_root.mkdir()
    frozen = runtime_sources()
    rows = []
    def save():
        (output/"report.json").write_bytes(canonical(dict(format="poseidon-normalization-expression-preparation-v1",
            planned=9, rows=rows, sources=frozen, paid_calls=0, agent_generated=False,
            actual_compile=False, encrypted_execution=False,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
    save()
    for entry in manifest["cases"]:
        if entry["operator"] == "matmul":
            continue
        require(runtime_sources() == frozen, "Source boundary")
        model_path = case_root/entry["model"]
        low = json.loads(model_path.read_text())
        high = json.loads((case_root/entry["source"]).read_text())
        require(digest(high) == entry["source_sha256"] and digest(low) == entry["model_sha256"], "Model identity")
        row = dict(id=entry["id"], model=str(model_path), model_sha256=digest(low), source_sha256=digest(high))
        try:
            request = prepare_request(low, cfg["ckks_config_sha256"], cfg)
            source = expression(high, request)
            candidate = dict(schema=1, request_id=request["request_id"], hecate_source=source)
            validate_candidate(candidate, request)
            checked = plaintext_check(source, request)
            source_path = fixture_root/(entry["id"]+".py")
            source_path.write_text(source)
            row.update(status="ready", source_path=str(source_path),
                       candidate_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       request_id=request["request_id"], semantic_probe=checked)
        except ValueError as error:
            row.update(status="static_blocked", diagnostic=str(error))
        rows.append(row)
        save()
        print(entry["id"], row["status"], row.get("diagnostic", ""), flush=True)
    return 0 if all(r["status"] == "ready" for r in rows) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from hecate_python_env import VENV, enter_nix
    if not args.inside:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
            [str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()), "--inside", "--output", str(args.output)])
        return enter_nix(command, seconds=300, environment="poseidon-cpu")
    return prepare(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
