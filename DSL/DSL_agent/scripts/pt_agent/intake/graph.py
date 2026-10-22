"""Static PT-Graph records and JSON/NPZ persistence without executable objects."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .core_ops import CORE_OPS


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    kind: str


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict
    src: str


@dataclass
class Graph:
    name: str
    nodes: list[Node]
    tensors: dict[str, TensorInfo]
    weights: dict[str, np.ndarray]
    inputs: list[str]
    outputs: list[str]

    def producer(self, tensor: str) -> Node | None:
        return next((node for node in self.nodes if tensor in node.outputs), None)

    def consumers(self, tensor: str) -> list[Node]:
        return [node for node in self.nodes if tensor in node.inputs]

    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(name=self.name, inputs=self.inputs, outputs=self.outputs,
                      tensors=[[t.name, t.shape, t.kind] for t in self.tensors.values()],
                      nodes=[[n.id, n.op, n.inputs, n.outputs, n.attrs, n.src] for n in self.nodes])
        (directory / "graph.json").write_text(json.dumps(record, indent=2) + "\n")
        np.savez(directory / "weights.npz", **self.weights)

    @staticmethod
    def load(directory) -> "Graph":
        directory = Path(directory)
        record = json.loads((directory / "graph.json").read_text())
        nodes = []
        for node_id, op, inputs, outputs, attrs, src in record["nodes"]:
            for key, value in list(attrs.items()):
                if CORE_OPS[op]["attrs"].get(key) in ("tuple", "pair"):
                    attrs[key] = tuple(value)
            nodes.append(Node(node_id, op, inputs, outputs, attrs, src))
        with np.load(directory / "weights.npz", allow_pickle=False) as archive:
            weights = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}
        for weight in weights.values():
            weight.setflags(write=False)
        tensors = {name: TensorInfo(name, tuple(shape), kind) for name, shape, kind in record["tensors"]}
        return Graph(record["name"], nodes, tensors, weights, record["inputs"], record["outputs"])
