"""The public R3 model intake interface."""

from .graph import Graph, Node, TensorInfo
from .torch_frontend import IntakeError, import_module as import_module_fx


def import_module(module, example_inputs: tuple, name: str, *, frontend="fx") -> Graph:
    if frontend == "fx":
        return import_module_fx(module, example_inputs, name)
    if frontend == "aten":
        from .aten_frontend import import_module_aten
        return import_module_aten(module, example_inputs, name)
    raise ValueError('frontend must be fx or aten')

__all__ = ["Graph", "Node", "TensorInfo", "IntakeError", "import_module"]
