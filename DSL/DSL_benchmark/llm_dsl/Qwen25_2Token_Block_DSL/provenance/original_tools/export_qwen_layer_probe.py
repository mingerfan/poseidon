"""Trace actual Qwen layer helpers to a CPU diagnostic DAG, not RuntimePlan.

Synthetic weights test conversion and packing. RoPE uses the actual 896/14/64
Qwen dimensions. Every trace is checked against independent NumPy arithmetic.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/hecate/hecate"
SLOTS = 16384


def padded(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) > SLOTS:
        raise ValueError("expected a vector fitting CKKS slots")
    return np.pad(values, (0, SLOTS - len(values)))


def compact(values):
    nonzero = np.flatnonzero(values)
    return values[:int(nonzero[-1]) + 1 if len(nonzero) else 1].tolist()


class Expr:
    def __init__(self, graph, op, values, **fields):
        self.graph, self.values = graph, padded(values)
        self.index = len(graph)
        tail = float(self.values[-1])
        different = np.flatnonzero(self.values != tail)
        expected = self.values[:int(different[-1])+1 if len(different) else 1].tolist()
        graph.append({"op": op, "expected": expected, "expected_tail": tail, **fields})

    def operation(self, other, op):
        if not isinstance(other, Expr):
            raise TypeError("explicit Plain required")
        values = self.values + other.values if op == "add" else self.values * other.values
        if isinstance(other, Plain):
            if np.all(other.values == other.values[0]):
                return Expr(self.graph, op + "_plain", values, a=self.index,
                            constant=float(other.values[0]))
            return Expr(self.graph, op + "_vector", values, a=self.index,
                        constant=compact(other.values))
        if other.graph is not self.graph:
            raise ValueError("operands belong to different graphs")
        return Expr(self.graph, op, values, a=self.index, b=other.index)

    def __add__(self, other):
        return self.operation(other, "add")

    def __mul__(self, other):
        return self.operation(other, "multiply")

    def rotate(self, steps):
        return Expr(self.graph, "rotate", np.roll(self.values, -steps), a=self.index, steps=int(steps))


class Plain(Expr):
    def __init__(self, values):
        self.values = padded(values)


def load_layers():
    # Isolated package avoids importing the unavailable native Hecate extension.
    package = types.ModuleType("_qwen_ckks_layer_probe")
    package.__path__ = [str(SOURCE)]
    expr = types.ModuleType(package.__name__ + ".expr")
    expr.Expr, expr.Plain = Expr, Plain
    sys.modules[package.__name__] = package
    sys.modules[expr.__name__] = expr
    spec = importlib.util.spec_from_file_location(package.__name__ + ".qwen_layers", SOURCE / "qwen_layers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = load_layers()
    rng = np.random.default_rng(20260917)
    programs = []

    def trace(name, inputs, evaluate, target, *, clean_output=True):
        graph, packed = [], []
        for x in inputs:
            # Deliberately dirty unused input slots to expose missing masks.
            values = np.concatenate((x, rng.normal(scale=.2, size=3)))
            packed.append(layers.PackedVector(Expr(graph, "input", values), len(x), SLOTS))
        result = evaluate(*packed)
        target = np.asarray(target)
        np.testing.assert_allclose(result.expr.values[:result.size], target, rtol=1e-12, atol=1e-12)
        if clean_output:
            np.testing.assert_allclose(result.expr.values[result.size:], 0, atol=1e-12)
        programs.append({"name": name, "packing": "prefix", "slots": SLOTS,
                         "nodes": graph, "output": result.expr.index, "target": target.tolist(),
                         "output_size": result.size, "zero_output_tail": clean_output,
                         "max_abs_error": 1e-5})

    x, y = rng.normal(scale=.2, size=(2, 7))
    trace("residual_7", [x, y], layers.add, x+y, clean_output=False)
    trace("elementwise_multiply_7", [x, y], layers.mul, x*y, clean_output=False)
    for ins, outs in ((7, 3), (3, 7)):
        x = rng.normal(scale=.2, size=ins)
        w, b = rng.normal(scale=.3, size=(outs, ins)), rng.normal(scale=.1, size=outs)
        trace(f"linear_{ins}_to_{outs}", [x], lambda v: layers.linear(v, w, b), w@x+b)
    x = rng.normal(scale=.2, size=896)
    angles = 511 / (1_000_000.0 ** (np.arange(0, 64, 2)/64))
    angles = np.tile(angles, 2)
    heads = x.reshape(14, 64)
    rotated = np.concatenate((-heads[:, 32:], heads[:, :32]), axis=-1)
    target = (heads*np.cos(angles)+rotated*np.sin(angles)).ravel()
    trace("rope_896_position_511", [x], lambda v: layers.rope(v, 511), target)
    x, w = rng.normal(scale=.2, size=7), rng.normal(scale=.3, size=(9, 7))
    for chunk in range(2):
        trace(f"lm_head_7_to_9_chunk_{chunk}", [x],
              lambda v: layers.lm_head(v, w, chunk_size=5)[chunk], (w@x)[chunk*5:(chunk+1)*5])
    hashes = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
              for name in ("qwen_layers.py", "ops.py")}
    payload = {"format": "qwen-ckks-diagnostic-dag-v1", "probe_kind": "actual_layer_functions",
               "candidate_sha256": None, "kernel_sha256": hashes["qwen_layers.py"],
               "source_sha256": hashes, "synthetic_weights": True, "programs": programs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(payload, output, allow_nan=False)
    print(f"Exported and NumPy-checked {len(programs)} actual layer traces")


if __name__ == "__main__":
    main()
