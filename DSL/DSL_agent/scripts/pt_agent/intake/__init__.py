"""The public R3 model intake interface."""

from .graph import Graph, Node, TensorInfo
from .torch_frontend import IntakeError, import_module as import_module_fx


def import_module(module, example_inputs: tuple, name: str, *, frontend="fx",
                  validation_inputs=None, observer=None) -> Graph:
    if frontend == "fx":
        if validation_inputs is not None:
            raise ValueError('validation_inputs requires the aten frontend')
        return import_module_fx(module, example_inputs, name)
    if frontend == "aten":
        from .aten_frontend import import_module_aten
        options = {}
        if validation_inputs is not None:
            options['validation_inputs'] = validation_inputs
        if observer is not None:
            options['observer'] = observer
        return import_module_aten(module, example_inputs, name, **options)
    raise ValueError('frontend must be fx or aten')

__all__ = ["Graph", "Node", "TensorInfo", "IntakeError", "import_module"]
