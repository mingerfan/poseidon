"""Internally inferred model information; never a user-supplied description."""

from dataclasses import dataclass, field
import json
from pathlib import Path


def source_kind(source):
    source = str(source)
    if source.lower().endswith('.onnx'):
        return 'onnx'
    if (Path(source) / 'config.json').is_file():
        return 'hf'
    return 'torch' if ':' in source else 'hf'


@dataclass
class ModelInfo:
    source: dict
    graph: object
    input_handling: dict
    inputs: list[dict]
    sequence: dict | None
    task: str
    metric: str
    threshold: dict
    split: dict
    assumptions: list = field(default_factory=list)

    def __post_init__(self):
        if any('ciphertexts' not in item or 'layout' not in item for item in self.inputs):
            self.describe_layouts()

    def describe_layouts(self, slots=16384):
        from ..kernels.layout import KernelError, layout_blocks, layout_record, multi_mux_layout, multi_flat_layout

        for entry in self.inputs:
            shape = entry['shape']
            try:
                if len(shape) == 4 and shape[0] == 1:
                    lay = multi_mux_layout(slots, *shape[1:], 1)
                elif len(shape) == 2 and shape[0] == 1:
                    lay = multi_flat_layout(slots, shape[1])
                else:
                    entry.update(ciphertexts=None, layout=dict(slots=slots, limitation='token layout deferred'))
                    continue
                entry.update(ciphertexts=len(layout_blocks(lay)), layout=layout_record(lay))
            except KernelError as exc:
                entry.update(ciphertexts=None, layout=dict(slots=slots, limitation=str(exc)))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(vars(self), graph='graph' if self.graph is not None else None)
        if self.graph is not None:
            self.graph.save(directory / 'graph')
        (directory / 'model_info.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        path = directory if directory.is_file() else directory / 'model_info.json'
        record = json.loads(path.read_text())
        if record['graph'] is not None:
            from ..intake.graph import Graph
            record['graph'] = Graph.load(path.parent / record['graph'])
            if 'intake' in record['source']:
                record['graph'].intake_stats = record['source']['intake']
        return cls(**record)


def infer_task(output_shape, *, architectures=(), vocab_size=None):
    """Apply U1 §3.4's head/shape rules and retain the decision reason."""
    if any(name.endswith(('ForImageClassification', 'ForSequenceClassification')) for name in architectures):
        return 'classification', 'HF classification architecture'
    if any(name.endswith('ForCausalLM') for name in architectures):
        return 'causal_lm', 'HF causal LM architecture'
    if len(output_shape) == 2 and output_shape[0] == 1 and output_shape[1] >= 2:
        return 'classification', 'output shape [1,C], C>=2'
    if len(output_shape) == 3 and output_shape[0] == 1 and vocab_size == output_shape[2]:
        return 'causal_lm', 'output shape [1,L,V] matches known vocabulary'
    return 'regression', 'other output head'


def metric_for(task, *, has_labels):
    if task == 'classification':
        return ('classification_top1', {'acc_drop_pp': 0.5}) if has_labels else (
            'classification_top1', {'agreement_pct': 99.5})
    if task == 'causal_lm':
        return 'lm_next_token', {'agreement_pct': 99.0, 'logit_rel_err': 1e-2}
    if task == 'regression':
        return 'regression_relative', {'relative_error': 1e-2}
    raise ValueError(f'Unknown inferred task {task}')
