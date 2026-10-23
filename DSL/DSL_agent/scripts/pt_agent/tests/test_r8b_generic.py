from collections import Counter
import copy
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pt_agent.analyzer import calibrate
from pt_agent.backends import MockBackend
from pt_agent.capabilities import CAPABILITIES, PATTERNS, render_capabilities
from pt_agent.data.cifar import calibration_set
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check
from pt_agent.intake.patterns import annotate
from pt_agent.lower import HANDLED_OPS, decode_outputs, lower, pack_inputs
from pt_agent.models.zoo import NAMES, get
from pt_agent.planners import prompts
from pt_agent.planners.agent_tools import FEASIBLE_MAX_U, build_context
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import DATA, require


MODELS = pt_result('models-v2')


@require(DATA, MODELS)
class GenericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.contexts = {}
        for name in NAMES:
            model = get(name, MODELS)
            graph = model.import_graph()
            images, _ = calibration_set(normalization=model.normalization)
            images = model.adapt(images[:8])
            calibration = calibrate(graph, {graph.inputs[0]: images})
            cls.contexts[name] = build_context(graph, calibration, n_val=50)

    def test_registry_matches_lowering(self):
        self.assertEqual({entry.op for entry in CAPABILITIES}, HANDLED_OPS)
        self.assertEqual([entry.op for entry in CAPABILITIES],
                         ['pack', 'conv2d', 'act', 'add', 'slice', 'pad', 'mean', 'relayout', 'linear', 'refresh'])
        rendered = render_capabilities()
        self.assertNotIn('refresh(x)', rendered)
        self.assertEqual([entry.op for entry in CAPABILITIES if not entry.in_prompt], ['refresh'])
        positions = [rendered.index('- '+entry.signature) for entry in CAPABILITIES if entry.in_prompt]
        self.assertEqual(positions, sorted(positions))
        for pattern in PATTERNS:
            self.assertIn(f'- {pattern.name}: {pattern.description} Write: {pattern.dsl_form}', rendered)

    def test_generic_prompt_layer(self):
        prompt = prompts.TEMPLATE.format(capabilities=render_capabilities(), max_u=FEASIBLE_MAX_U,
                                         n_val=50, examples='').lower()
        for text in ('resnet', 'layer1.', 'conv1.weight', 'linear.weight', 'fc1.weight', 'fc2.weight', 'cifar'):
            with self.subTest(text=text):
                self.assertNotIn(text, prompt)

    def test_four_model_prompts(self):
        for name, ctx in self.contexts.items():
            with self.subTest(model=name):
                prompt = prompts.build_system_prompt(ctx)
                for text in ('resnet', 'cifar', 'layer1.'):
                    self.assertNotIn(text, prompt.lower())
                self.assertIn('C -> C + 2p', prompt)
                self.assertIn('max |u| <= 0.9', prompt)
                self.assertIn('on the 50 validation images', prompt)
                self.assertEqual(ctx.model_name, name)

    def test_pattern_counts_order_and_no_graph_mutation(self):
        expected = {
            'resnet20': dict(conv_bn=19, downsample_pad=2, mean_head=1),
            'resnet20_silu': dict(conv_bn=19, downsample_pad=2, mean_head=1),
            'cnn_plain': dict(conv_bn=3, mean_head=1), 'mlp': {},
        }
        for name, ctx in self.contexts.items():
            with self.subTest(model=name):
                before = copy.deepcopy(ctx.graph.nodes)
                groups = annotate(ctx.graph)
                self.assertEqual(dict(Counter(group['pattern'] for group in groups)), expected[name])
                self.assertEqual([group['group'] for group in groups], list(range(len(groups))))
                positions = {node.id: index for index, node in enumerate(ctx.graph.nodes)}
                starts = [positions[group['nodes'][0]] for group in groups]
                self.assertEqual(starts, sorted(starts))
                self.assertEqual(ctx.graph.nodes, before)

    def test_pattern_matching_rejects_changed_bounds_and_branching(self):
        graph = copy.deepcopy(self.contexts['resnet20'].graph)
        original = annotate(graph)
        group = next(group for group in original if group['pattern'] == 'downsample_pad')
        first = next(node for node in graph.nodes if node.id == group['nodes'][0])
        first.attrs['end'] -= 1
        self.assertEqual(Counter(group['pattern'] for group in annotate(graph))['downsample_pad'], 1)
        graph = copy.deepcopy(self.contexts['resnet20'].graph)
        conv = next(node for node in graph.nodes if node.op == 'conv2d')
        extra = copy.deepcopy(graph.consumers(conv.outputs[0])[0])
        extra.id = 'extra-consumer'
        graph.nodes.append(extra)
        self.assertEqual(Counter(group['pattern'] for group in annotate(graph))['conv_bn'], 18)

    def test_initial_message_pattern_membership(self):
        for name, ctx in self.contexts.items():
            with self.subTest(model=name):
                message = json.loads(prompts.initial_message(ctx))
                groups = annotate(ctx.graph)
                self.assertEqual(message['model'], name)
                self.assertEqual(message['patterns'], groups)
                membership = {node: group for group in groups for node in group['nodes']}
                for node in message['nodes']:
                    if node['id'] in membership:
                        self.assertEqual(node['pattern'], membership[node['id']]['pattern'])
                        self.assertEqual(node['group'], membership[node['id']]['group'])
                    else:
                        self.assertNotIn('pattern', node)
                        self.assertNotIn('group', node)

    def test_examples_selected_by_features(self):
        expected = {
            'resnet20': ['residual_small', 'cnn_plain_small'],
            'resnet20_silu': ['residual_small', 'cnn_plain_small'],
            'cnn_plain': ['cnn_plain_small', 'residual_small'],
            'mlp': ['mlp_flat', 'cnn_plain_small'],
        }
        for name, ctx in self.contexts.items():
            with self.subTest(model=name):
                features = prompts.target_features(ctx.graph, annotate(ctx.graph))
                self.assertEqual(prompts.select_examples(features, name), expected[name])
                self.assertEqual('flat_input' in features, name == 'mlp')
                prompt = prompts.build_system_prompt(ctx)
                self.assertEqual([line.removeprefix('# example: ') for line in prompt.splitlines()
                                  if line.startswith('# example: ')], expected[name])

    def test_example_source_exclusion_ties_and_zero_similarity(self):
        examples = [dict(name='z', file='z.ptdsl', source_model='target', features=['linear']),
                    dict(name='b', file='b.ptdsl', source_model='other', features=['linear']),
                    dict(name='a', file='a.ptdsl', source_model='other', features=['linear']),
                    dict(name='c', file='c.ptdsl', source_model='other', features=['conv2d'])]
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'index.json').write_text(json.dumps(examples))
            with patch.object(prompts, 'EXAMPLES_DIR', directory):
                self.assertEqual(prompts.select_examples({'linear'}, 'target', k=4), ['a', 'b'])
                self.assertEqual(prompts.select_examples({'linear'}, 'other', k=4), ['z'])
                self.assertEqual(prompts.select_examples(set(), 'target'), [])

    def test_three_examples_parse_typecheck_and_exact_lower(self):
        rng = np.random.default_rng(20261026)
        for example in json.loads((prompts.EXAMPLES_DIR/'index.json').read_text()):
            with self.subTest(example=example['name']):
                source = (prompts.EXAMPLES_DIR/example['file']).read_text()
                program = parse(source)
                checked = check(program, example['weight_shapes'])
                self.assertTrue(checked.ok, checked.diags)
                weights = {key: (rng.uniform(.5, 1.5, size=shape) if key.endswith('.running_var')
                                 else rng.normal(scale=.05, size=shape))
                           for key, shape in example['weight_shapes'].items()}
                images = rng.normal(size=(2, *program.params[0].shape[1:]))
                backend = MockBackend(16384, batch=2)
                backend.bind(pack_inputs(program, [images]))
                outputs, layouts = lower(program, weights, backend, act_mode='exact')
                decoded = decode_outputs([output.value for output in outputs], layouts)
                self.assertEqual(decoded[0].shape, (2, 10))
                self.assertTrue(np.isfinite(decoded[0]).all())

    def test_context_default_and_feasibility_constant(self):
        self.assertEqual(FEASIBLE_MAX_U, 0.90)
        self.assertEqual(inspect.signature(build_context).parameters['n_val'].default, 1000)
        self.assertEqual(prompts.PROMPT_VERSION, 'r8-v4')
        self.assertEqual(self.contexts['mlp'].val_images.shape, (50, 3072))
        self.assertEqual(self.contexts['mlp'].val_exact_logits.shape, (50, 10))


if __name__ == '__main__':
    unittest.main()
