"""R1 kind/layout checks, shape inference and unused-value diagnostics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from .errors import Diag
from .ir import Program
from .ops_spec import OPS
from .shapes import ShapeError, WeightError, infer


@dataclass
class CheckResult:
    diags: list[Diag]
    shapes: dict[str, tuple | None]
    layouts: dict[str, str]

    @property
    def ok(self) -> bool:
        return not any(d.severity == "error" for d in self.diags)


def check(program: Program, weights: Mapping[str, Sequence[int]]) -> CheckResult:
    diags = []
    shapes = {p.name: p.shape for p in program.params}
    layouts = {p.name: "raw" if p.kind == "cipher" else "plain" for p in program.params}
    plain = {p.name for p in program.params if p.kind == "plain"}

    def wshape(name):
        if name not in weights:
            raise WeightError("E_WEIGHT_MISSING", name)
        return tuple(weights[name])

    def emit(code, op, msg):
        diags.append(Diag(code, "warning" if code.startswith("W_") else "error", op.lineno, op.id, msg,
                          op.call_lines, op.macro_stack))

    for op in program.ops:
        for index, vid in enumerate(op.inputs):
            if vid in plain and not (op.kind in ("add", "sub", "mul") and index == 1):
                emit("E_KIND", op, "Plain parameters apply only to the second add/sub/mul input")
        tags = [layouts[vid] for vid in op.inputs]
        if op.kind == "pack":
            if tags[0] != "raw":
                emit("E_LAYOUT_REPACK", op, "pack requires a raw Cipher parameter")
            tag = op.attrs["layout"]
        else:
            if "raw" in tags:
                emit("E_LAYOUT_UNPACKED", op, "Pack all raw tensor inputs before this operation")
            nonplain = [tag for tag in tags if tag != "plain"]
            if OPS[op.kind].layout == "binary" and len(set(nonplain)) > 1:
                emit("E_LAYOUT_MISMATCH", op, "Binary inputs require matching nonplain layout labels")
            tag = op.attrs["layout"] if op.kind == "relayout" else tags[0]
        for vid in op.outputs:
            layouts[vid] = tag
        input_shapes = [shapes[vid] for vid in op.inputs]
        output_shapes = [None] * len(op.outputs)
        if all(shape is not None for shape in input_shapes):
            try:
                output_shapes = infer(op.kind, input_shapes, op.attrs, wshape)
            except ShapeError as exc:
                emit("E_SHAPE", op, f"{exc}; input shapes: {input_shapes}")
            except WeightError as exc:
                emit(exc.code, op, str(exc))
        shapes.update(zip(op.outputs, output_shapes))
    used = {vid for op in program.ops for vid in op.inputs} | set(program.outputs)
    for vid in shapes:
        if vid not in used:
            producer = program.producer(vid)
            param = program.param(vid)
            diags.append(Diag("W_UNUSED", "warning", producer.lineno if producer else param.lineno,
                              producer.id if producer else None, f"Unused value {vid}",
                              producer.call_lines if producer else (), producer.macro_stack if producer else ()))
    if program.header.refresh == "compiler":
        for op in program.ops:
            if op.kind == "refresh":
                emit("W_REFRESH_COMPILER", op, "Compiler refresh mode lowers this refresh to identity")
    return CheckResult(diags, shapes, layouts)


def load_weight_shapes(path) -> dict[str, tuple[int, ...]]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Weight shapes must be a JSON object")
    result = {}
    for name, shape in data.items():
        if not isinstance(shape, list) or not shape or not all(type(d) is int and d >= 1 for d in shape):
            raise ValueError(f"{name}: expected a list of positive dimensions")
        result[name] = tuple(shape)
    return result
