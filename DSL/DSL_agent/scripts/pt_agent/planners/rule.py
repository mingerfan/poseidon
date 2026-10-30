"""Emit PT-DSL by walking supported graph nodes with the requested refresh mode."""

import json
import re

from ..intake import Graph


def plan(graph: Graph, calibration: dict, *, deg: int = 63, margin: float = 1.25,
         slots: int = 16384, placement: str = "depth-dp", refresh: str = "compiler",
         act_method: str = "cheb") -> str:
    if getattr(graph,'intake_stats',{}).get('transformer_normalized'):
        from .transformer_rule import plan as token_plan
        if refresh!='compiler':raise ValueError('token rule uses compiler refresh')
        return token_plan(graph,calibration,slots=slots,placement=placement)
    if margin < 1.1:
        raise ValueError("rule planner requires margin >= 1.1")
    if refresh not in ("compiler", "agent"):
        raise ValueError("refresh must be compiler or agent")
    param = graph.inputs[0]
    shape = graph.tensors[param].shape
    header = (f'@fhe.program(slots={slots}, refresh="agent", placement="depth-dp", l_max=5, boot_out_level=5)'
              if refresh == "agent" else f'@fhe.program(slots={slots}, refresh="compiler", placement={json.dumps(placement)})')
    lines = [header,
             f'def {graph.name}({param}: Cipher[{", ".join(map(str, shape))}]):']
    names, used = {}, {param}

    def name(tensor):
        if tensor not in names:
            base = re.sub(r"_+", "_", "v_"+re.sub(r"[^A-Za-z0-9]", "_", tensor)).rstrip("_")
            candidate, suffix = base, 2
            while candidate in used:
                candidate = f"{base}_{suffix}"
                suffix += 1
            names[tensor] = candidate
            used.add(candidate)
        return names[tensor]

    def wref(key):
        return "W["+json.dumps(key)+"]"

    if len(shape) == 2:
        lines.append(f'    {name(param)} = pack({param}, layout="flat")')
    else:
        lines.append(f'    {name(param)} = pack({param}, layout="mux", gap=1)')
    skipped = set()
    for node in graph.nodes:
        if node.id in skipped:
            continue
        a = node.attrs
        output = node.outputs[0]
        prev = name(node.inputs[0])
        if node.op == "conv2d":
            args = [prev, "w="+wref(a["weight"])]
            if a.get("bias") is not None:
                args.append("b="+wref(a["bias"]))
            consumers = graph.consumers(output)
            if len(consumers) == 1 and consumers[0].op == "batchnorm":
                bn = consumers[0]
                weight_name = bn.attrs["weight"]
                prefix = weight_name.removesuffix(".weight")
                if (not weight_name.endswith(".weight") or bn.attrs["bias"] != prefix+".bias"
                        or bn.attrs["mean"] != prefix+".running_mean" or bn.attrs["var"] != prefix+".running_var"):
                    raise ValueError(f"{bn.id}: inconsistent batchnorm weight names")
                args.extend(("bn="+json.dumps(prefix), "bn_eps="+repr(float(bn.attrs["eps"]))))
                output = bn.outputs[0]
                skipped.add(bn.id)
            if a['groups'] != 1:
                args.append('groups='+repr(a['groups']))
            args.extend(("stride="+repr(tuple(a["stride"])), "pad="+repr(tuple(a["padding"])), 'algo="mpcb"'))
            expression = "conv2d("+", ".join(args)+")"
        elif node.op == "act":
            if act_method == "minimax_comp" and a["fn"] == "relu":
                limits = calibration[node.inputs[0]]
                bound = margin*max(abs(float(limits["min"])), abs(float(limits["max"])))
                expression = (f'act({prev}, fn="relu", method="minimax_comp", '
                              f'degrees=(15, 15, 27), domain={(-bound, bound)!r})')
            else:
                domain = (float(calibration[node.inputs[0]]["min"])*margin,
                          float(calibration[node.inputs[0]]["max"])*margin)
                expression = f'act({prev}, fn={json.dumps(a["fn"])}, method="cheb", deg={deg}, domain={domain!r})'
        elif node.op in ('avgpool', 'maxpool'):
            args = [prev, 'k='+repr(tuple(a['kernel'])), 'stride='+repr(tuple(a['stride'])), 'pad='+repr(tuple(a['padding']))]
            if node.op == 'maxpool':
                limits = calibration[node.outputs[0]+'::differences']
                domain = (float(limits['min'])*1.5, float(limits['max'])*1.5)
                if act_method == 'minimax_comp':
                    bound = max(abs(domain[0]), abs(domain[1]))
                    args += ['method="minimax_comp"', 'degrees=(15, 15, 27)', 'domain='+repr((-bound, bound))]
                else:
                    args += ['method="cheb"', 'deg=127', 'domain='+repr(domain)]
            expression = node.op+'('+', '.join(args)+')'
        elif node.op == 'adaptive_avgpool':
            if tuple(a['output_size']) != (1, 1):
                raise ValueError(f'{node.id}: adaptive avgpool requires output (1,1)')
            expression = f'mean({prev}, axis=(2, 3))'
        elif node.op == 'concat':
            expression = 'concat('+', '.join(name(v) for v in node.inputs)+', axis='+repr(a['axis'])+')'
        elif node.op in ('flatten', 'reshape'):
            out_shape = graph.tensors[output].shape
            if len(out_shape) != 2 or out_shape[0] != 1:
                raise ValueError(f'{node.id}: flatten requires a (1,F) output')
            expression = f'flatten({prev}, start_dim=1)'
        elif node.op == 'batchnorm':
            prefix = a['weight'].removesuffix('.weight')
            if any(a[key] != prefix+'.'+suffix for key, suffix in
                   (('weight','weight'), ('bias','bias'), ('mean','running_mean'), ('var','running_var'))):
                raise ValueError(f'{node.id}: inconsistent batchnorm weight names')
            expression = f'batchnorm({prev}, bn={json.dumps(prefix)}, eps={float(a["eps"])!r})'
        elif node.op == "add":
            expression = f"add({prev}, {name(node.inputs[1])})"
        elif node.op == "slice":
            expression = (f'slice({prev}, axis={a["axis"]}, start={a["start"]}, '
                          f'end={a["end"]!r}, step={a["step"]})')
        elif node.op == "pad":
            expression = f'pad({prev}, pads={tuple(a["pads"])!r})'
        elif (node.op == "mean" and a['keepdim'] and set(a['axes']) == {2, 3}
              and output not in graph.outputs and len(graph.consumers(output)) == 1
              and graph.consumers(output)[0].op in ('flatten', 'reshape')):
            # ATen represents AdaptiveAvgPool(1) as a keepdim spatial mean.
            # The existing global-mean kernel already returns the flat vector.
            expression = f'mean({prev}, axis=(2, 3))'
        elif node.op == "mean":
            expression = f'mean({prev}, axis={tuple(a["axes"])!r})'
        elif node.op == "linear":
            flat = prev
            if graph.producer(node.inputs[0]) is not None and graph.producer(node.inputs[0]).op == "mean":
                flat = name(node.inputs[0]+"_flat")
                lines.append(f'    {flat} = relayout({prev}, layout="flat")')
            args = [flat, "w="+wref(a["weight"])]
            if a.get("bias") is not None:
                args.append("b="+wref(a["bias"]))
            expression = "linear("+", ".join(args)+', algo="bsgs")'
        else:
            raise ValueError(f"{node.id}: unsupported graph node {node.op}")
        lines.append(f"    {name(output)} = {expression}")
    lines.append("    return "+", ".join(name(output) for output in graph.outputs))
    return "\n".join(lines)+"\n"
