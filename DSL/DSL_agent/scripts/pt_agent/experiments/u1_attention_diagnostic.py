"""Read-only U1 §12 layer/ATen comparison on one frozen original input."""

import copy
import importlib
import json
import math
from pathlib import Path
from unittest.mock import patch


def run(args):
    import numpy as np
    import torch
    from torch.fx import Interpreter
    from ..data.samples import read
    from ..intake import aten_frontend
    from ..intake.hf_frontend import load_local, HFInputs, boundary_module
    from ..intake.graph_exec import forward
    from ..intake.probe import rename_inputs
    from ..models.info import ModelInfo

    info = ModelInfo.load(args.reference/'qwen_alternative')
    local = load_local(info.source['local_path'], attn_implementation='eager')
    dataset = HFInputs(local, read(args.inputs/'prompts.txt'), info)
    arrays, _ = dataset.select(info.split['calibration'][:1])
    tensors = tuple(torch.from_numpy(x) for x in arrays.values())
    wrapper = boundary_module(local)
    captured = {}
    original_make_fx = aten_frontend.make_fx

    def trace(module, *a, **kw):
        factory = original_make_fx(module, *a, **kw)
        def apply(*inputs):
            gm = factory(*inputs)
            captured.update(gm=gm, module=module)
            return gm
        return apply

    with patch.object(aten_frontend, 'make_fx', side_effect=trace):
        graph = aten_frontend.import_module_aten(wrapper, tensors, 'attention_diagnostic')
    rename_inputs(graph, list(arrays))
    graph.save(args.out/'graph')
    np.savez(args.out/'identical_inputs.npz', **arrays)
    fx_values = {}

    def clone(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, (tuple, list)):
            return type(value)(clone(v) for v in value)
        return value

    class Capture(Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            fx_values[node.name] = clone(result)
            return result

    native = {}
    handles = []
    for name, module in local.module.named_modules():
        def hook(mod, inputs, output, name=name):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(value):
                native[name] = clone(value)
        handles.append(module.register_forward_hook(hook))
        if name.endswith('o_proj'):
            def prehook(mod, inputs, name=name):
                native[name+'.input'] = clone(inputs[0])
            handles.append(module.register_forward_pre_hook(prehook))

    attention = next(module for name, module in local.module.named_modules() if name.endswith('self_attn'))
    definition = importlib.import_module(type(attention).__module__)
    rope = getattr(definition, 'apply_rotary_pos_emb')
    rope_rows = []
    rope_constants = []
    def observe_rope(*a, **kw):
        result = rope(*a, **kw)
        rope_rows.append(clone(result))
        rope_constants.append(clone(a[2:4]))
        return result
    try:
        with torch.no_grad(), patch.object(definition, 'apply_rotary_pos_emb', side_effect=observe_rope):
            expected = wrapper(*tensors).detach()
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        fx_result = Capture(captured['gm']).run(*tensors)
        actual = forward(graph, arrays, keep=True)

    def delta(a, b):
        if tuple(a.shape) != tuple(b.shape):
            return dict(shape_mismatch=[list(a.shape), list(b.shape)])
        a, b = a.to(torch.float64), b.to(torch.float64)
        diff = torch.where(a == b, torch.zeros_like(a), torch.abs(a-b))
        return dict(max_abs=float(diff.max()), max_normalized=float((diff/(1+torch.abs(b))).max()),
                    dtype_actual=str(a.dtype), shape=list(a.shape))

    if isinstance(fx_result, (tuple, list)):
        fx_result = fx_result[0]
    rows = []
    for node in graph.nodes:
        for output in node.outputs:
            if output in fx_values and torch.is_tensor(fx_values[output]):
                rows.append(dict(op_id=node.id, op=node.op, tensor=output, attrs=node.attrs,
                                 source=node.src, **delta(actual[output], fx_values[output])))
    first = next((row for row in rows if row.get('max_abs', 0) != 0), None)
    first_material = next((row for row in rows if row.get('max_normalized', 0) > 1e-6), None)
    layers = []
    for node in graph.nodes:
        key = node.attrs.get('weight', node.attrs.get('const', ''))
        if isinstance(key, str) and key.startswith('original.') and key.endswith('.weight'):
            module_name = key[len('original.'):-len('.weight')]
            if module_name in native:
                tensor = node.outputs[0]
                consumers = graph.consumers(tensor)
                if (actual[tensor].shape != native[module_name].shape and len(consumers) == 1
                        and consumers[0].op == 'reshape'
                        and actual[consumers[0].outputs[0]].shape == native[module_name].shape):
                    tensor = consumers[0].outputs[0]
                layers.append(dict(module=module_name, op_id=node.id,
                                   tensor=tensor, **delta(actual[tensor], native[module_name])))
                aliases = []
                if module_name.endswith('.self_attn.o_proj'):
                    aliases.append(module_name.removesuffix('.o_proj'))
                    input_name = module_name+'.input'
                    if input_name in native:
                        source = node.inputs[0]
                        before = graph.producer(source)
                        if before is not None and before.op == 'reshape':
                            source = before.inputs[0]
                        if actual[source].shape == native[input_name].shape:
                            layers.append(dict(module=input_name, op_id=node.id, tensor=source,
                                               **delta(actual[source], native[input_name])))
                elif module_name.endswith('.mlp.down_proj'):
                    aliases.append(module_name.removesuffix('.down_proj'))
                for alias in aliases:
                    layers.append(dict(module=alias, op_id=node.id, tensor=tensor,
                                       **delta(actual[tensor], native[alias])))
    # RoPE q/k are the residual additions immediately following rotate_half.
    rope_comparisons = []
    rope_outputs = [node for node in graph.nodes if node.op == 'add' and len(node.inputs) == 2
                    and graph.producer(node.inputs[1]) is not None
                    and graph.producer(node.inputs[1]).op == 'mul'
                    and any(graph.producer(name) is not None and graph.producer(name).op == 'concat'
                            for name in graph.producer(node.inputs[1]).inputs)]
    for index, (node, value) in enumerate(zip(rope_outputs, [v for pair in rope_rows for v in pair])):
        rope_comparisons.append(dict(index=index, op_id=node.id, **delta(actual[node.outputs[0]], value)))
    result = dict(eager=local.config._attn_implementation,
                  position_information=dict(source='original HF default position_ids',
                                            positions=list(range(arrays['attention_mask'].shape[1]))),
                  hf_vs_fx=delta(fx_result, expected), reference_vs_hf=delta(actual[graph.outputs[0]], expected),
                  first_nonzero=first, first_material_mismatch=first_material,
                  node_comparisons=rows, layer_comparisons=layers, rope_comparisons=rope_comparisons)
    saved = {name: value.cpu().numpy() for name, value in native.items()
             if name and name.endswith(('input_layernorm','q_proj','k_proj','v_proj','self_attn',
                                         'o_proj','o_proj.input','post_attention_layernorm','mlp','norm','lm_head'))}
    for index, pair in enumerate(rope_rows):
        saved[f'rope.{index}.q'], saved[f'rope.{index}.k'] = [v.cpu().numpy() for v in pair]
    for index, pair in enumerate(rope_constants):
        saved[f'rope.{index}.cos'], saved[f'rope.{index}.sin'] = [v.cpu().numpy() for v in pair]
    np.savez_compressed(args.out/'hook_outputs.npz', **saved)
    result['hook_tensors'] = {name: list(value.shape) for name, value in saved.items()}
    # Separate true/padded positions without excluding any output from success.
    length = int(arrays['attention_mask'].sum())
    from ..metrics import compare, passes
    metric = compare(dict(values=expected.cpu().numpy(), task=info.task,
                          lengths=np.asarray([length])),
                     actual[graph.outputs[0]].cpu().numpy(), None)
    result['unlabelled_metrics'] = dict(values=metric, passed=passes(metric, info.threshold),
                                       threshold=info.threshold)
    result['real_positions'] = delta(actual[graph.outputs[0]][:, :length], expected[:, :length])
    if length < arrays['attention_mask'].shape[1]:
        result['padded_positions'] = delta(actual[graph.outputs[0]][:, length:], expected[:, length:])
    def serializable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value
    result = serializable(result)
    (args.out/'DIAGNOSTIC.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('hf_vs_fx','reference_vs_hf','first_nonzero','first_material_mismatch')},indent=2),flush=True)
    return result
