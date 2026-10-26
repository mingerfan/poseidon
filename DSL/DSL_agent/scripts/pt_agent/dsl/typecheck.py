"""R1 kind/layout checks, shape inference and unused-value diagnostics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

from .errors import Diag
from .ir import Program, WeightRef
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


def referenced_weights(program: Program) -> set[str]:
    """Collect explicit references after static macro expansion, including fused BN."""
    def refs(value):
        if isinstance(value, WeightRef):
            return {value.name}
        if isinstance(value, Mapping):
            return set().union(*(refs(v) for v in value.values()))
        if isinstance(value, (tuple, list)):
            return set().union(*(refs(v) for v in value))
        return set()

    used = set()
    for op in program.ops:
        used.update(refs(op.attrs))
        if op.kind in ("conv2d", "batchnorm") and op.attrs.get("bn") is not None:
            used.update(op.attrs["bn"]+"."+suffix for suffix in
                        ("weight", "bias", "running_mean", "running_var"))
    return used


def consumed_weights(graph) -> dict[str, list[tuple]]:
    """Use the graph's weight-valued roles and constant inputs, never name suffixes."""
    from ..intake.core_ops import CORE_OPS

    consumed = {}
    for node in graph.nodes:
        for role, kind in CORE_OPS[node.op]["attrs"].items():
            name = node.attrs.get(role)
            if kind in ("weight", "weight?") and name in graph.weights:
                consumed.setdefault(name, []).append((node, role))
        for index, name in enumerate(node.inputs):
            if name in graph.weights:
                consumed.setdefault(name, []).append((node, f"input[{index}]"))
    return consumed


def weight_usage_diags(program: Program, graph) -> list[Diag]:
    consumed, used = consumed_weights(graph), referenced_weights(program)
    op_refs = {op.id: referenced_weights(Program(program.name, program.header, program.params,
                (op,), program.outputs, program.regions)) for op in program.ops}

    def corresponding(node):
        names = {name for name, consumers in consumed.items()
                 if any(consumer is node for consumer, role in consumers)}
        matches = [op for op in program.ops if op.kind == node.op and names & op_refs[op.id]]
        if len(matches) == 1:
            return matches[0]
        if not matches and node.op == "batchnorm":
            producer = graph.producer(node.inputs[0])
            if producer is not None and producer.op == "conv2d":
                return corresponding(producer)
        return None

    diags = []
    for name, consumers in consumed.items():
        if name in used:
            continue
        details, locations = [], []
        for node, role in consumers:
            role = {"mean": "running_mean", "var": "running_var"}.get(role, role) if node.op == "batchnorm" else role
            op = corresponding(node)
            detail = f"图节点 {node.op} {node.id} ({', '.join(node.outputs)}) 的 {role}"
            if op is not None:
                locations.append(op)
                detail += f"，对应程序第 {op.lineno} 行的 {op.kind}"
            if node.op in ("conv2d", "linear") and role == "bias":
                detail += f"；请加上 b=W[{json.dumps(name)}]"
            details.append(detail)
        op = locations[0] if locations else None
        diags.append(Diag("E_WEIGHT_UNUSED", "error", op.lineno if op else 0,
            op.id if op else None, json.dumps(name)+" 未被使用："+"；".join(details),
            op.call_lines if op else (), op.macro_stack if op else ()))
    return diags


def check(program: Program, weights: Mapping[str, Sequence[int]], *, graph=None) -> CheckResult:
    diags = []
    if program.header.manage == "dsl" and (program.header.refresh != "compiler" or
                                           program.header.placement != "depth-dp"):
        diags.append(Diag("E_MANAGEMENT", "error", program.header.lineno, None,
                          '暂不支持：manage="dsl" requires refresh="compiler", placement="depth-dp"'))
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
        if op.kind == "conv2d" and input_shapes[0] is not None:
            channels = input_shapes[0][1]
            groups = op.attrs["groups"]
            if groups not in (1, channels):
                emit("E_SHAPE", op, "暂不支持 this grouped convolution; groups must be 1 or C_in")
            elif groups == channels and channels > 1:
                shape = weights.get(op.attrs["w"].name)
                if shape is not None and shape[0] != channels:
                    emit("E_SHAPE", op, "暂不支持 depthwise channel multipliers; C_out must equal C_in")
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
    if graph is not None:
        diags.extend(weight_usage_diags(program, graph))
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
