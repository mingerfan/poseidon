"""Infer model information from an original model and original sample source."""

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path

from ..models.info import ModelInfo, infer_task, metric_for, source_kind


def load_callable(source):
    module_name, callable_name = str(source).rsplit(':', 1)
    if module_name.endswith('.py'):
        path = Path(module_name).resolve()
        definition = importlib.util.spec_from_file_location('pt_agent_user_module', path)
        module = importlib.util.module_from_spec(definition)
        definition.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, callable_name)
    model = factory()
    import torch
    if not isinstance(model, torch.nn.Module):
        raise ValueError('PyTorch callable must return torch.nn.Module')
    return model.eval(), callable_name


def rename_inputs(graph, names):
    if len(names) != len(graph.inputs):
        raise ValueError('Model and samples have different input counts')
    from .graph import TensorInfo
    mapping = dict(zip(graph.inputs, names))
    if len(set(names)) != len(names):
        raise ValueError('Model input names must be distinct')
    graph.tensors = {mapping.get(key, key): TensorInfo(mapping.get(key, key), value.shape, value.kind)
                     for key, value in graph.tensors.items()}
    for node in graph.nodes:
        node.inputs = [mapping.get(key, key) for key in node.inputs]
        node.outputs = [mapping.get(key, key) for key in node.outputs]
    graph.inputs = list(names)
    graph.outputs = [mapping.get(key, key) for key in graph.outputs]
    return graph


def model_info(source, graph, *, split, has_labels, handling=None, architectures=(), vocab_size=None,
               sequence=None, assumptions=(), dtypes=None):
    source = dict(source)
    if hasattr(graph, 'intake_stats'):
        source['intake'] = graph.intake_stats
    if len(graph.outputs) != 1:
        task, reason = 'regression', 'multiple output heads'
    else:
        task, reason = infer_task(graph.tensors[graph.outputs[0]].shape,
                                  architectures=architectures, vocab_size=vocab_size)
    metric, threshold = metric_for(task, has_labels=has_labels)
    kinds = ('image' if handling and handling.get('kind') == 'image' else 'features')
    inputs = [dict(name=name, kind='mask' if name == 'attention_mask' else kinds,
                   shape=list(graph.tensors[name].shape), dtype=(dtypes or {}).get(name, 'float64'))
              for name in graph.inputs]
    if handling and handling.get('embedding_boundary'):
        inputs[0]['kind'] = 'tokens'
    return ModelInfo(source, graph, handling or dict(mode='original tensors'), inputs, sequence,
                     task, metric, threshold, split, [*assumptions, dict(task_inference=reason)])


@dataclass
class ImportedModel:
    info: ModelInfo
    samples: object
    original: object = None


def import_torch(module, samples, *, name, source, split=None, preserve_input_names=False):
    import torch
    from ..data.samples import SampleSplits
    from . import import_module
    if samples.kind != 'tensor':
        raise ValueError('PyTorch models require the original tensor samples sent to module(x)')
    split = samples.split() if split is None else split
    tensors, _ = samples.tensors(split['calibration'][:1])
    example = tuple(torch.from_numpy(value) for value in tensors.values())
    graph = import_module(module, example, name, frontend='aten')
    if not preserve_input_names:
        # npy has no named keys; preserve the model's inferred argument name.
        if tuple(tensors) != ('input',):
            rename_inputs(graph, list(tensors))
    info = model_info(dict(type='torch', path=str(source)), graph, split=split,
                      has_labels=samples.labels is not None,
                      dtypes={key: str(value.dtype) for key, value in zip(graph.inputs, tensors.values())})
    return ImportedModel(info, SampleSplits(samples, info), module)


def import_model(source, samples):
    kind = source_kind(source)
    if kind == 'torch':
        module, name = load_callable(source)
        return import_torch(module, samples, name=name, source=source)
    if kind == 'hf':
        from .hf_frontend import import_local
        return import_local(source, samples)
    from .onnx_frontend import import_model as import_onnx
    return import_onnx(source, samples)
