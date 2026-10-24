"""U1 §7.3–4: local exports and cross-source float64 semantic checks."""

from collections import Counter
import copy
import json


def run(args):
    import numpy as np
    import torch
    import onnxruntime as ort
    from ..data.cifar import normalize, read_rows
    from ..data.samples import read
    from ..intake import import_module
    from ..intake.graph_exec import forward
    from ..intake.hf_frontend import load_local, HFInputs, boundary_module
    from ..intake.onnx_frontend import import_graph, normalize_for_comparison
    from ..models.builders import load
    from ..models.info import ModelInfo
    from ..paths import RESULTS_DIR
    rows = []
    for name in ('resnet20','mlp','bert_tiny'):
        if args.only and name not in args.only.split(','):
            continue
        if name == 'bert_tiny':
            info = ModelInfo.load(args.reference/'bert_tiny')
            local = load_local(info.source['local_path'], attn_implementation='eager')
            samples = HFInputs(local, read(args.inputs/'prompts.txt'), info)
            arrays, _ = samples.select(info.split['calibration'][:4]+info.split['heldout'][:4])
            # The ONNX acceptance reference is explicitly float32. Keep the
            # HF float64 import proof separate; ORT has no float64 Erf kernel.
            arrays = {key: value.astype(np.float32) if value.dtype.kind == 'f' else value
                      for key, value in arrays.items()}
            path = args.out/'bert_tiny.onnx'
            torch.onnx.export(boundary_module(copy.deepcopy(local)).float().eval(),
                tuple(torch.from_numpy(value[:1]) for value in arrays.values()), str(path),
                opset_version=17, input_names=list(arrays), output_names=['logits'], do_constant_folding=True)
            original = None
        else:
            pixels, _ = read_rows('train', range(8))
            images = normalize(pixels).astype(np.float32)
            module = load(name, args.models_dir)
            if name == 'mlp':
                images = images.reshape(len(images), -1)  # original MLP user tensor
            original = import_module(module, (torch.from_numpy(images[:1]),), name, frontend='aten')
            arrays = {'input': images}
            path = args.out/(name+'.onnx')
            torch.onnx.export(copy.deepcopy(module).float().eval(), torch.from_numpy(images[:1]),
                              str(path), opset_version=17, input_names=['input'],
                              output_names=['logits'], do_constant_folding=True)
        graph = import_graph(path, {key: value[:1] for key, value in arrays.items()})
        graph.save(args.out/name)
        max_abs = max_ratio = 0.0
        if original is None:
            options = ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        for i in range(len(next(iter(arrays.values())))):
            batch = {key: value[i:i+1] for key, value in arrays.items()}
            actual = forward(graph, batch)[graph.outputs[0]]
            expected = (forward(original, {original.inputs[0]: batch['input']})[original.outputs[0]]
                        if original is not None else session.run(None, batch)[0])
            diff = np.abs(actual-expected)
            max_abs=max(max_abs,float(diff.max())); max_ratio=max(max_ratio,float((diff/(1+np.abs(expected))).max()))
            tolerance = 1e-5 if original is not None else 1e-4
            if not np.all(diff <= tolerance*(1+np.abs(expected))):
                raise AssertionError(f'{name}: cross-source numeric mismatch, max_abs={max_abs}, normalized={max_ratio}')
        row = dict(model=name,samples=len(next(iter(arrays.values()))),max_abs_diff=max_abs,max_normalized_diff=max_ratio)
        if original is not None:
            before=Counter(node.op for node in normalize_for_comparison(original).nodes)
            after=Counter(node.op for node in normalize_for_comparison(graph).nodes)
            if before != after:
                raise AssertionError(f'{name}: normalized op multisets differ: PyTorch={dict(before)}, ONNX={dict(after)}')
            row.update(normalized_ops=dict(before),op_multisets_equal=True)
        rows.append(row)
        (args.out/'CROSS.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print('CROSS '+name+' '+json.dumps(row),flush=True)
    return rows
