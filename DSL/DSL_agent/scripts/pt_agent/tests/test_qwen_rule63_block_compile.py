"""Small source/AST and fake-process tests; no actual Qwen/fit/Count/compiler."""
import ast
from dataclasses import make_dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import qwen_rule63_block_compile as own


def metadata_graph():
    # Matching record fields keep this source-only fixture independent of the
    # intake package's eager numerical-framework imports.
    Graph = make_dataclass('Graph', ['name', 'nodes', 'tensors', 'weights', 'inputs', 'outputs'])
    Node = make_dataclass('Node', ['id', 'op', 'inputs', 'outputs', 'attrs', 'src'])
    TensorInfo = make_dataclass('TensorInfo', ['name', 'shape', 'kind'])
    kinds = ['rmsnorm', 'linear', 'mul', 'shuffle', 'mul', 'add', 'linear', 'mul',
             'shuffle', 'mul', 'add', 'matmul', 'softmax', 'linear', 'matmul', 'linear',
             'add', 'rmsnorm', 'linear', 'act', 'linear', 'mul', 'linear', 'add']
    linear = iter(('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                   'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'))
    norm = iter(('input_layernorm', 'post_attention_layernorm'))
    nodes = []; weights = {}; tensors = dict(inputs_embeds=TensorInfo('inputs_embeds', (1, 32, 896), 'act'),
        attention_mask=TensorInfo('attention_mask', (1, 32), 'mask'), sel=TensorInfo('sel', (1, 32), 'mask'))
    prior = 'inputs_embeds'; public = 0
    for index, kind in enumerate(kinds):
        attrs = {}; inputs = [prior]
        if kind == 'linear':
            name = next(linear); attrs['weight'] = 'original.model.layers.0.' + name + '.weight'
            if name.startswith('self_attn.') and name != 'self_attn.o_proj':
                attrs['bias'] = 'original.model.layers.0.' + name + '.bias'
        elif kind == 'rmsnorm':
            attrs['weight'] = 'original.model.layers.0.' + next(norm) + '.weight'
        elif kind == 'mul' and public < 4:
            attrs['const'] = 'u4.public.synthetic' + str(public); public += 1
        elif kind == 'softmax':
            inputs.append('attention_mask')
        for value in attrs.values(): weights[value] = object()
        output = 'add_7' if index == 23 else 'synthetic_value_' + str(index)
        tensors[output] = TensorInfo(output, (1, 32, 896), 'act')
        nodes.append(Node('synthetic_node_' + str(index), kind, inputs, [output], attrs, 'synthetic'))
        prior = output
    nodes.append(Node('next_layer', 'rmsnorm', ['add_7'], ['next'],
        dict(weight='original.model.layers.1.input_layernorm.weight'), 'synthetic'))
    nodes += [Node('unused_' + str(index), 'add', [], [], {}, 'synthetic') for index in range(554)]
    graph = Graph('synthetic_parent', nodes, tensors, weights,
                  ['inputs_embeds', 'attention_mask', 'sel'], ['synthetic_final_head'])
    graph.intake_stats = dict(transformer_normalized=True)
    return graph


class IndependentRule63Block(unittest.TestCase):
    def test_slice_preserves_original_objects_and_complete_first_block(self):
        graph = metadata_graph(); selected = own.slice_layer0(graph)
        self.assertEqual(len(selected.nodes), 24)
        self.assertEqual(selected.inputs, ['inputs_embeds', 'attention_mask'])
        self.assertEqual(selected.outputs, ['add_7'])
        self.assertEqual(len(selected.weights), 16)
        self.assertTrue(all(a is b for a, b in zip(selected.nodes, graph.nodes)))
        self.assertTrue(all(value is graph.weights[name] for name, value in selected.weights.items()))
        self.assertEqual(len(graph.nodes), 579)
        graph.nodes[24].attrs['weight'] = 'original.model.layers.2.input_layernorm.weight'
        with self.assertRaises(ValueError): own.slice_layer0(graph)

    def test_act63_only_leaves_original_R0_softmax_norm_and_domain(self):
        text = '''def original(x):
 n=rmsnorm(x,rsqrt_method="newton",rsqrt_deg0=7,rsqrt_iters=5)
 s=softmax(n,steps=8,exp_method="remez",exp_deg=63,inv_method="newton",inv_iters=5)
 y=act(s,fn="silu",method="cheb",deg=127,domain=(-4.,7.))
 return y
'''
        source, changes = own.rule63_source(text)
        self.assertEqual(changes[0]['domain'], (-4., 7.))
        transformed = ast.parse(source)
        calls = [node for node in ast.walk(transformed) if isinstance(node, ast.Call)]
        act = next(node for node in calls if node.func.id == 'act')
        degree = next(item for item in act.keywords if item.arg == 'deg')
        self.assertEqual(ast.literal_eval(degree.value), 63)
        degree.value = ast.Constant(value=127)
        self.assertEqual(ast.dump(transformed, include_attributes=False),
                         ast.dump(ast.parse(text), include_attributes=False))
        with self.assertRaises(ValueError): own.rule63_source(text.replace('fn="silu"', 'fn="gelu"'))

    def test_native_worker_argv_has_explicit_sources_and_one_intake_interpreter(self):
        args = own.arguments(['--out', '/tmp/synthetic-independent-qwen-rule63'])
        command, outer = own.phase_command(args, 'compile', 11000)
        self.assertEqual(command.count('--python'), 1)
        self.assertEqual(command.count(own.MODULE), 1)
        self.assertEqual(command[command.index('--phase-seconds') + 1], '7200')
        self.assertEqual(outer, 7320)
        for name in own.SOURCE_OPTIONS:
            self.assertEqual(command.count('--' + name.replace('_', '-')), 1)
        self.assertNotIn('--source-case', command)
        self.assertNotIn('--session', command)

    def test_fake_four_phase_success_or_first_R0_failure_closes_honest_case(self):
        from pt_agent import paths, registry
        from pt_agent.experiments import dacapo_compile_continuation as resources
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = own.arguments(['--out', str(root / 'case'), '--paper-runs', str(root / 'paper.jsonl')])
                for name in ('profile', 'metadata_template', 'calibration'):
                    path = root / (name + '.json'); path.write_text('{}'); setattr(args, name, path)
                calls = []
                def observation():
                    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[],
                                started_ns=10, ended_ns=125000010)
                def process(command, **kwargs):
                    phase = command[command.index('--worker-phase') + 1]; calls.append(phase)
                    value = dict(status='error' if failure else 'ok', error='Synthetic fixedR0 rejection' if failure else None)
                    if phase == 'compile':
                        value.update(native_process=observation(), compiled_bootstraps=12, execution_steps=45,
                            runtime_plan='synthetic-plan', prediction=dict(table_known_sum_us=1250000,
                            missing=[dict(op='relinearize')], by_op_us=dict(boot=1250000)))
                    (args.out / 'work' / phase / 'phase_result.json').write_text(json.dumps(value))
                    return observation()
                with patch.object(paths, 'RESULTS_DIR', root), patch.object(registry, 'git_rev', return_value='synthetic'),\
                        patch.object(own, 'source_gate', return_value={'synthetic': True}),\
                        patch.object(own, 'run_process', side_effect=process),\
                        patch.object(resources, 'available_ram', return_value=20 * own.GIB):
                    result = own.execute(args)
                self.assertEqual(calls, ['prepare'] if failure else list(own.PHASES))
                self.assertEqual(result['status'], 'error' if failure else 'ok')
                self.assertTrue(result['monitor_stopped'])
                marker = json.loads((args.out / 'completion.json').read_text())
                self.assertEqual(marker['status'], result['status'])
                rows = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 1); self.assertEqual(len(rows[0]), 8)
                paper = json.loads((root / 'paper.jsonl').read_text())
                self.assertEqual(paper['status'], 'driver_error' if failure else 'ok')
                self.assertIsNone(paper['generation_e2e_s']); self.assertFalse(paper['full_model'])
                self.assertEqual(paper['pred_latency_s'], None if failure else 1.25)
                self.assertAlmostEqual(sum(value for value in paper['stages_s'].values() if value is not None),
                                       paper['isolated_pipeline_wall_s'])


if __name__ == '__main__':
    unittest.main()
