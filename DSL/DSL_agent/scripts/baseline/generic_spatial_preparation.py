"""Deterministic generic graph preparation before the only Agent output.

This emits a model graph and provenance, never DSL. External numerical and
compiler feedback have no entry here. Each step retains its independent limits.
"""
import copy
from pathlib import Path

from benchmark_graph import validate
from delivery_component import bound, canonical, sha, read_bytes, require
import spatial_channel_normalization
import spatial_window_normalization
import projection_fusion
import uniform_bias_normalization
import compact_graph_names

VERSION='poseidon-generic-spatial-preparation-v1'


def prepare(model):
    validate(model)
    original=copy.deepcopy(model)
    result=copy.deepcopy(model)
    steps=[]
    modules=[spatial_channel_normalization,spatial_window_normalization,
             projection_fusion,uniform_bias_normalization,compact_graph_names]
    for module in modules:
        result,receipt=module.normalize(result)
        validate(result)
        steps.append(dict(module=module.__name__,receipt=receipt,
                          source_sha256=sha(read_bytes(Path(module.__file__).absolute()))))
    require(model==original,'Preparation mutated caller model')
    require(result['inputs']==original['inputs']
            and [o['name']for o in result['outputs']]==[o['name']for o in original['outputs']],
            'Original public input/output signature changed')
    return result,bound(dict(format=VERSION,input_model_sha256=sha(canonical(original)),
        output_model_sha256=sha(canonical(result)),steps=steps,
        source_sha256=sha(read_bytes(Path(__file__).absolute())),
        dispatch='static operators, shapes and original public values',model_name_dispatch=False,
        response_or_numerical_feedback_used=False,DSL_generated=False,numeric_certificate=False))
