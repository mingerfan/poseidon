"""Immutable PT-DSL records and a name-independent structural representation."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WeightRef:
    name: str


@dataclass(frozen=True)
class ValueRef:
    name: str


@dataclass(frozen=True)
class Header:
    slots: int
    refresh: str = "compiler"
    placement: str = "depth-dp"
    l_max: int | None = None
    boot_out_level: int | None = None
    lineno: int = 0
    manage: str = "compiler"


@dataclass(frozen=True)
class Param:
    name: str
    kind: str
    shape: tuple[int, ...]
    lineno: int


@dataclass(frozen=True)
class Region:
    name: str
    level_budget: int
    lineno: int


@dataclass(frozen=True)
class Op:
    id: int
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict
    lineno: int
    region: str | None
    macro_stack: tuple[str, ...]
    call_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class Program:
    name: str
    header: Header
    params: tuple[Param, ...]
    ops: tuple[Op, ...]
    outputs: tuple[str, ...]
    regions: dict[str, Region]

    def producer(self, vid: str) -> "Op | None":
        return next((op for op in self.ops if vid in op.outputs), None)

    def param(self, name: str) -> "Param | None":
        return next((p for p in self.params if p.name == name), None)


def canonical(program: Program) -> dict:
    def value(v):
        if isinstance(v, WeightRef):
            return {"weight": v.name}
        if isinstance(v, ValueRef):
            return {'value': ids[v.name]}
        if isinstance(v, (tuple, list)):
            return [value(x) for x in v]
        if isinstance(v, dict):
            return {k: value(x) for k, x in v.items()}
        return v

    ids = {p.name: "p:" + p.name for p in program.params}
    ops, regions = [], []
    seen_regions = set()
    output_index = 0
    for op in program.ops:
        inputs = [ids[x] for x in op.inputs]
        outputs = []
        for out in op.outputs:
            ids[out] = f"%{output_index}"
            outputs.append(ids[out])
            output_index += 1
        ops.append([op.kind, inputs, outputs, value(op.attrs), op.region])
        if op.region is not None and op.region not in seen_regions:
            seen_regions.add(op.region)
            regions.append([op.region, program.regions[op.region].level_budget])
    header = asdict(program.header)
    header.pop("lineno")
    if header["manage"] == "compiler":
        header.pop("manage")
    return dict(header=header,
                params=[[p.name, p.kind, list(p.shape)] for p in program.params],
                ops=ops, outputs=[ids[x] for x in program.outputs], regions=regions)
