"""Locate original-Qwen rotary/normalization differences on one real prefix.

This diagnostic executes the saved first-block prefix and original framework
components, without evaluating the complete model or any candidate program.
"""

import argparse
import json
import os
from pathlib import Path
import resource

from .r10_paper_metrics import StageRecorder, write_json


def diagnose(args, recorder, result):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding, apply_rotary_pos_emb, rotate_half
    from .qwen_embedding_validation import map_stored_member, map_stored_array
    from ..intake.graph import Graph, Node, TensorInfo
    from ..intake import graph_exec
    from ..intake.mapped_reference import MappedReference
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    raw = json.loads((args.model_info / 'graph/graph.json').read_text())
    archive = args.model_info / 'graph/weights.npz'
    original = json.loads(args.reference.read_text())
    row = original['trajectories'][0]['steps'][0]
    ids = np.asarray(row['prefix_token_ids'], dtype=np.int64)
    embedding, _ = map_stored_member(archive, 'original.model.embed_tokens.weight.npy')
    embeddings = np.asarray(embedding[ids])[None, :, :]
    del embedding
    selected = []
    for node_id, op, inputs, outputs, attrs, source in raw['nodes']:
        node = Node(node_id, op, inputs, outputs, attrs, source)
        selected.append(node)
        if node.id == 'n12':
            break
    if [node.id for node in selected][-1] != 'n12':
        raise ValueError('The recorded first-layer rotary prefix differs')
    used = {value for node in selected for key, value in node.attrs.items()
            if key in ('weight', 'bias', 'const')}
    graph = Graph(raw['name'], selected,
                  {name: TensorInfo(name, tuple(shape), kind) for name, shape, kind in raw['tensors']},
                  {name: map_stored_array(archive, name + '.npy')[0] for name in used}, ['inputs_embeds'],
                  [selected[-1].outputs[0]])

    def compare(name, expected, observed):
        expected = expected.detach().cpu().numpy() if torch.is_tensor(expected) else np.asarray(expected)
        observed = observed.detach().cpu().numpy() if torch.is_tensor(observed) else np.asarray(observed)
        if expected.shape != observed.shape or not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise ValueError('Diagnostic shape/finite mismatch: ' + name)
        delta = np.abs(expected - observed)
        tolerance = 1e-12 + 1e-12 * np.abs(expected)
        result.setdefault('comparisons', []).append(dict(name=name, shape=list(expected.shape),
            max_absolute_difference=float(delta.max()), max_tolerance_ratio=float((delta / tolerance).max()),
            passed=bool(np.all(delta <= tolerance)), dtype=str(expected.dtype)))

    with recorder.span('original_interpreter_and_mapped_same_prefix'):
        values = graph_exec.forward(graph, {'inputs_embeds': embeddings}, keep=True)
        mapped = MappedReference(graph).forward({'inputs_embeds': embeddings}, keep=True)
        for node in selected:
            compare('mapped-versus-original:' + node.id, values[node.outputs[0]], mapped[node.outputs[0]])

    with recorder.span('original_framework_components_same_real_prefix'):
        config = AutoConfig.from_pretrained(str(args.local_model), local_files_only=True, trust_remote_code=False)
        config._attn_implementation = 'eager'
        norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).double().eval()
        norm.load_state_dict({'weight': torch.from_numpy(np.array(graph.weights['original.model.layers.0.input_layernorm.weight']))}, strict=True)
        x = torch.from_numpy(embeddings)
        h = norm(x)
        compare('original-rmsnorm-versus-normalized:n0', h, values['mul_4'])
        rotary = Qwen2RotaryEmbedding(config=config).double().eval()
        cos, sin = rotary(x, torch.arange(32)[None, :])

        def projection(which, normalized=False):
            if normalized:
                return values['addmm' if which == 'q' else 'addmm_1']
            root = 'original.model.layers.0.self_attn.' + which + '_proj.'
            return F.linear(h, torch.from_numpy(np.array(graph.weights[root + 'weight'])),
                            torch.from_numpy(np.array(graph.weights[root + 'bias'])))

        def rotate(q, k):
            q = q.reshape(1, 32, 14, 64).transpose(1, 2)
            k = k.reshape(1, 32, 2, 64).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            return q.transpose(1, 2).reshape(1, 32, 896), k.transpose(1, 2).reshape(1, 32, 128)

        q, k = rotate(projection('q', True), projection('k', True))
        raw_q = projection('q', True).reshape(1, 32, 14, 64).transpose(1, 2)
        raw_k = projection('k', True).reshape(1, 32, 2, 64).transpose(1, 2)
        for kind, value, heads, names in (
                ('q', raw_q, 14, ('mul_5', 'mul_6.shuffle', 'mul_6', 'add_2.shuffle')),
                ('k', raw_k, 2, ('mul_7', 'mul_8.shuffle', 'mul_8', 'add_3.shuffle'))):
            flatten = lambda x: x.transpose(1, 2).reshape(1, 32, heads * 64)
            compare('rotary-cos-term:' + kind, flatten(value * cos[:, None, :, :]), values[names[0]])
            compare('rotary-half-permutation:' + kind, flatten(rotate_half(value)), values[names[1]])
            compare('rotary-sin-term-before-second-shuffle:' + kind,
                    flatten(rotate_half(value) * sin[:, None, :, :]), values[names[2]])
            compare('rotary-sin-term-after-second-shuffle:' + kind,
                    flatten(rotate_half(value) * sin[:, None, :, :]), values[names[3]])
        compare('original-rotary-same-projection-versus-normalized:q', q, values['add_2'])
        compare('original-rotary-same-projection-versus-normalized:k', k, values['add_3'])
        compare('original-q-projection-versus-normalized:n1', projection('q'), values['addmm'])
        result['real_prefix'] = dict(sample_id=original['trajectories'][0]['sample_id'],
            prefix_length=row['prefix_length'], sequence_length=32, first_token=int(ids[0]),
            heldout_samples=0, confirm_samples=0, report_samples=0)
        result['scope'] = 'Original first-block real-input diagnostic; no full-model forward, approximation, or candidate selection'
    result['status'] = 'ok'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('model-info', 'reference', 'local-model', 'work-dir'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', api_calls=0, compiler_runs=0, keygen_runs=0, ckks_executions=0,
                  full_model_forwards=0, candidate_frozen=False, compiled_bootstraps=None,
                  predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None)
    recorder = StageRecorder(args.work_dir / 'events.jsonl')
    try:
        with recorder.span('qwen_first_rotary_diagnostic'):
            diagnose(args, recorder, result)
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc))
    result['worker_maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.work_dir / 'worker-result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
