"""Straight-line Earth values, with PT-DSL ownership retained outside MLIR attributes."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Type:
    cipher: bool
    scale: int
    level: int


@dataclass
class Node:
    id: int
    kind: str
    args: list[int]
    type: Type
    attrs: dict = field(default_factory=dict)
    data: object = None
    op_id: int | None = None
    lineno: int = 0


@dataclass
class Function:
    name: str
    inputs: list[Type]
    nodes: list[Node]
    outputs: list[int]
    result_types: list[Type]
    attrs: dict
    profile: dict
    cuts: list[int] = field(default_factory=list)
    boundaries: list = field(default_factory=list)

    def types(self):
        return {**{-i-1: t for i, t in enumerate(self.inputs)}, **{n.id: n.type for n in self.nodes}}

    def summary(self):
        return dict(mode='dsl', profile=self.profile, cuts=self.cuts,
                    boots=sum(n.kind == 'bootstrap' for n in self.nodes),
                    inputs=[dict(scale=t.scale, level=t.level) for t in self.inputs],
                    values=[dict(id=n.id, kind=n.kind, scale=n.type.scale, level=n.type.level,
                                 op_id=n.op_id, lineno=n.lineno) for n in self.nodes],
                    outputs=self.outputs, attributes=self.attrs)

