"""Straight-line Earth values, with PT-DSL ownership retained outside MLIR attributes."""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Type:
    cipher: bool
    scale: int
    level: int


@lru_cache(maxsize=None)
def shared_type(cipher, scale, level):
    """Reuse immutable Earth types without sharing mutable Node attributes."""
    return Type(cipher, scale, level)


@dataclass(slots=True)
class Node:
    live_count: ClassVar[int] = 0
    id: int
    kind: str
    args: list[int]
    type: Type
    attrs: dict = field(default_factory=dict)
    data: object = None
    op_id: int | None = None
    lineno: int = 0
    const_key: str | None = None

    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        Node.live_count += 1
        return instance

    def __del__(self):
        Node.live_count -= 1


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
        types = {-i-1: t for i, t in enumerate(self.inputs)}
        types.update((n.id, n.type) for n in self.nodes)
        return types

    def summary(self):
        return dict(mode='dsl', profile=self.profile, cuts=self.cuts,
                    boots=sum(n.kind == 'bootstrap' for n in self.nodes),
                    inputs=[dict(scale=t.scale, level=t.level) for t in self.inputs],
                    values=[dict(id=n.id, kind=n.kind, scale=n.type.scale, level=n.type.level,
                                 op_id=n.op_id, lineno=n.lineno) for n in self.nodes],
                    outputs=self.outputs, attributes=self.attrs)
