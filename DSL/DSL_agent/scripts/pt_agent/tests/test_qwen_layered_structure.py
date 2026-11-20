"""Small typed source and real Count/Emit boundaries; no model or native run."""

import ast
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from pt_agent.dsl.ir import ValueRef, WeightRef, canonical
from pt_agent.dsl.parser import parse
from pt_agent.dsl.typecheck import check, referenced_weights
from pt_agent.experiments import qwen_layered_structure as entry
from pt_agent.backends.streaming_count import StreamingCountBackend
from pt_agent.backends.streaming_emit import StreamingEmitBackend


def fixture():
    # Shape-only public metadata; no array with these model dimensions exists.
    source = ('def qwen_layer(x, am, w):\n'
              '    y=linear(x,w=W[w])\n'
              '    out=add(y,x)\n'
              '    return out\n'
              '@fhe.program(slots=16384,refresh="agent",'
              'placement="program-plan",l_max=5,boot_out_level=5)\n'
              'def qwen2(inputs_embeds: Cipher[1,32,896], attention_mask: Mask[1,32], sel: Mask[1,32]):\n'
              '    x=pack(inputs_embeds,layout="token")\n'
              '    am=pack(attention_mask,layout="flat")\n'
              '    sm=pack(sel,layout="flat")\n')
    current = 'x'
    shapes = {}
    for index in range(24):
        name = f'original.model.layers.{index}.fixture.weight'
        source += f'    h{index}=qwen_layer({current},am,"{name}")\n'
        shapes[name] = (896, 896)
        current = f'h{index}'
    source += (f'    n=rmsnorm({current},w=W["original.model.norm.weight"],eps=0.001,'
               'rsqrt_method="newton",rsqrt_domain=(0.01,0.2),rsqrt_deg0=7,rsqrt_iters=2)\n'
               '    selected=select(n,sm)\n'
               '    logits=linear(selected,w=W["original.model.embed_tokens.weight"])\n'
               '    return logits\n')
    shapes['original.model.norm.weight'] = (896,)
    shapes['original.model.embed_tokens.weight'] = (151936, 896)
    return parse(source), shapes


def raw_plan(calls, *, output_refresh=False):
    return dict(format='pt-agent-refresh-plan-v1', levels=dict(entry.staged.LEVELS), calls=calls,
        boots=int(output_refresh), substitutions=[], outputs=[[0, 1000]] if output_refresh else [], dead=[])


class TypedLayerStructure(unittest.TestCase):
    def setUp(self):
        self.program, self.shapes = fixture()
        self.partition = entry.split_program(self.program, self.shapes)

    def test_partition_preserves_all_source_ops_local_ids_and_weight_references(self):
        original = deepcopy(self.program)
        self.assertEqual(len(self.partition.layers), 24)
        covered = []
        for i, segment in enumerate((*self.partition.layers, self.partition.head)):
            self.assertEqual([op.id for op in segment.program.ops], list(range(len(segment.program.ops))))
            self.assertEqual(segment.local_to_original_ids,
                             tuple(enumerate(segment.original_op_ids)))
            covered.extend(segment.original_op_ids[3:])
            self.assertTrue(check(segment.program, self.shapes).ok)
            if i < 24:
                self.assertEqual(segment.program.outputs[1:], ('am', 'sm'))
                self.assertIn(f'original.model.layers.{i}.fixture.weight', referenced_weights(segment.program))
                self.assertEqual(segment.program.ops[3].inputs, ('x',))
        self.assertEqual(covered, list(range(3, len(self.program.ops))))
        self.assertEqual(self.program, original)
        self.assertEqual(canonical(self.program), canonical(original))
        self.assertEqual(set().union(*(referenced_weights(x.program)
                         for x in (*self.partition.layers, self.partition.head))), referenced_weights(self.program))

    def test_original_unused_rules_allow_internal_mask_passthrough(self):
        # This small layer deliberately uses no mask arithmetic; original warnings
        # remain visible and all inputs are still consumed by the lowerer ABI.
        result = check(self.partition.layers[0].program, self.shapes)
        self.assertTrue(result.ok)
        self.assertEqual(result.shapes[self.partition.layers[0].program.outputs[0]], (1, 32, 896))
        self.assertFalse(any(d.code == 'W_UNUSED' and d.msg.endswith(('am', 'sm')) for d in result.diags))
        self.assertTrue(any(d.code == 'W_UNUSED' and d.msg.endswith('am') for d in self.partition.diagnostics))

    def test_binding_contains_original_semantics_ids_and_weight_shapes_after_json_roundtrip(self):
        segment = self.partition.layers[1]
        source = entry.source_binding(segment)
        self.assertEqual(source, json.loads(json.dumps(source)))
        self.assertEqual(source['original_ops'][3]['id'], segment.original_op_ids[3])
        self.assertEqual(source['original_ops'][3]['attrs']['w'],
                         dict(weight='original.model.layers.1.fixture.weight'))
        self.assertEqual(source['weight_shapes'], [['original.model.layers.1.fixture.weight', [896, 896]]])
        bound = entry.bind_plan(segment, raw_plan(66), planned_source=source)
        self.assertIs(bound.segment, segment)
        self.assertEqual(bound.scope, 'semantic-object-binding-only')
        self.assertFalse(bound.planner_provenance_verified)
        self.assertFalse(bound.independent_replay_verified)
        bound.source['role'] = 'changed'
        self.assertEqual(entry.source_binding(segment)['role'], 'layer')

    def test_reject_wrong_source_layer_mapping_shape_or_missing_plan_fields(self):
        segment = self.partition.layers[0]
        for field, changed in (('role', 'head'), ('local_to_original_ids', []),
                               ('weight_shapes', []), ('original_ops', [])):
            source = entry.source_binding(segment); source[field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'original op IDs'):
                entry.bind_plan(segment, raw_plan(66), planned_source=source)
        with self.assertRaisesRegex(ValueError, 'schema'):
            entry.bind_plan(segment, dict(format='pt-agent-refresh-plan-v1'), planned_source=entry.source_binding(segment))
        plan = raw_plan(66); plan['calls'] = True
        with self.assertRaisesRegex(ValueError, 'schema'):
            entry.bind_plan(segment, plan, planned_source=entry.source_binding(segment))

    def test_reject_nonlocal_ids_wrong_layer_weights_head_or_cross_layer_dependency(self):
        ops = list(self.program.ops)
        for changed in (replace(ops[3], id=999),
                        replace(ops[3], attrs={**ops[3].attrs, 'w': WeightRef('original.model.layers.1.fixture.weight')}),
                        replace(ops[5], inputs=(ops[3].outputs[0],))):
            altered = list(ops); altered[changed.id if changed.id != 999 else 3] = changed
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                entry.split_program(replace(self.program, ops=tuple(altered)), self.shapes)
        altered = list(ops)
        altered[-1] = replace(ops[-1], attrs={**ops[-1].attrs, 'w': WeightRef('other_head')})
        with self.assertRaisesRegex(ValueError, 'tied vocabulary'):
            entry.split_program(replace(self.program, ops=tuple(altered)), {**self.shapes, 'other_head': (151936, 896)})

    def test_value_reference_rewrite_is_exact_and_original_attrs_stay_unchanged(self):
        value = {'mask': ValueRef('old'), 'w': WeightRef('old'), 'nested': (ValueRef('am'), [ValueRef('old')])}
        copied = deepcopy(value)
        changed = entry._rewrite(value, {'old': 'x'})
        self.assertEqual(changed['mask'], ValueRef('x'))
        self.assertEqual(changed['w'], WeightRef('old'))
        self.assertEqual(changed['nested'], (ValueRef('am'), [ValueRef('x')]))
        self.assertEqual(value, copied)

    def test_source_attr_mutation_is_detected_before_lowering_or_binding(self):
        segment = self.partition.layers[0]
        segment.program.ops[3].attrs['algo'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'changed after partition'):
            entry.source_binding(segment)
        with self.assertRaisesRegex(ValueError, 'changed after partition'):
            entry.bind_plan(segment, raw_plan(66), planned_source={})

    def test_callbacks_keep_original_mapping_and_call_original_lowering(self):
        weights = {name: SimpleNamespace(shape=shape) for name, shape in self.shapes.items()}
        layers, head = entry.callbacks(self.partition, weights)
        marker = object()
        with patch('pt_agent.experiments.qwen_prefill_compile._same_lowering', return_value=marker) as lowering:
            self.assertIs(layers[0](marker), marker)
        lowering.assert_called_once_with(self.partition.layers[0].program, weights, marker)
        self.assertTrue(all(cb.weights is weights for cb in (*layers, head)))
        weights['original.model.layers.0.fixture.weight'] = SimpleNamespace(shape=(1, 1))
        with self.assertRaisesRegex(ValueError, 'weight shape changed'):
            layers[0](marker)


class SymbolicLayerConnections(unittest.TestCase):
    def setUp(self):
        program, shapes = fixture()
        self.partition = entry.split_program(program, shapes)
        self.weights = {name: SimpleNamespace(shape=shape) for name, shape in shapes.items()}
        self.embedding = self.weights['original.model.embed_tokens.weight']
        self.layers, self.head = entry.callbacks(self.partition, self.weights)
        self.plans = tuple(entry.bind_plan(cb.segment, raw_plan(66), planned_source=entry.source_binding(cb.segment))
                           for cb in self.layers)
        self.head_plan = entry.bind_plan(self.head.segment, raw_plan(53, output_refresh=True),
                                        planned_source=entry.source_binding(self.head.segment))
        self.observed = []
        self.lowering_calls = []

    def fake_lowering(self, program, weights, active):
        self.assertIs(weights, self.weights)
        self.lowering_calls.append(program)
        values = [active.input(index) for index in range(34)]
        self.assertTrue(all(value.level == 5 for value in values))
        self.observed.append((program, tuple(value.value for value in values)))
        if program is self.head.segment.program:
            active.output([values[index]*.5 for index in range(19)])
        else:
            active.output([*[value*.5 for value in values[:32]], *values[32:]])

    def kernels(self, embedding, coefficients):
        self.assertIs(embedding, self.embedding)
        def first(active):
            values = [active.input(i) for i in range(19)]
            onehot = [value*.5 for value in values]
            active.output([*onehot, onehot[0]*.5])
        def terminal(active):
            values = [active.input(i) for i in range(19)]
            active.output([value*.5 for value in values])
        return first, terminal

    def run_fixture(self, backend, **overrides):
        arguments = dict(layers=self.layers, head=self.head, plans24=self.plans, head_plan=self.head_plan,
            first_kernel_plan=raw_plan(39), terminal_kernel_plan=raw_plan(38),
            original_embedding=self.embedding, prompt_length=9)
        arguments.update(overrides)
        with patch('pt_agent.experiments.qwen_prefill_compile._same_lowering', side_effect=self.fake_lowering), \
                patch.object(entry.staged, 'kernel_callbacks', side_effect=self.kernels):
            return entry.run_layered(backend, **arguments)

    def count(self):
        return StreamingCountBackend(16384, input_level=5, boot_out_level=5)

    def test_count_uses_real_boundary_bootstraps_and_reuses_all_24_sources_twice(self):
        backend = self.count()
        state = self.run_fixture(backend)
        self.assertEqual(entry.EXPLICIT_BOUNDARY_BOOTSTRAPS, 1577)
        self.assertEqual(state.metadata['explicit_boundary_bootstrap_calls'], 1577)
        self.assertEqual(state.metadata['internal_plan_bootstrap_calls'], 2)
        self.assertEqual(backend.stats()['counts']['boot'], 1579)
        self.assertEqual(len(backend._inputs), 34)
        self.assertEqual(len(backend._outputs), 38)
        self.assertEqual(len(state.metadata['segments']), 52)
        for index, cb in enumerate(self.layers):
            self.assertIs(self.lowering_calls[index], cb.segment.program)
            self.assertIs(self.lowering_calls[25+index], cb.segment.program)
        self.assertIs(self.lowering_calls[24], self.lowering_calls[49])
        self.assertTrue(state.metadata['handoff']['consumed_embedding'])
        second_input = self.observed[25][1][9]
        handoff = state._known[state.metadata['handoff']['normalized_embedding']]
        self.assertIs(second_input, handoff.value)
        self.assertIsNot(second_input, state.inputs[9].value)
        for forward in (0, 1):
            start = forward*25
            for row in self.observed[start:start+24]:
                self.assertIs(row[1][32], self.observed[start][1][32])
                self.assertIs(row[1][33], self.observed[start][1][33])
        self.assertEqual(state.metadata['scope'], 'symbolic-layer-composition-only')
        self.assertIsNone(state.metadata['native_compiled_bootstraps'])
        for key in ('C2_verified', 'compile_verified', 'whole_model_verified',
                    'planner_provenance_verified', 'independent_replay_verified'):
            self.assertIs(state.metadata[key], False)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)

    def test_emit_matches_count_inventory_without_loading_frontend_or_executing_generated_source(self):
        counted = self.run_fixture(self.count())
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            backend = StreamingEmitBackend(16384, directory=directory, frontend='/unused-public-fixture.py',
                                           max_constant_bytes=1024**2)
            try:
                emitted = self.run_fixture(backend)
                self.assertEqual(backend.stats()['counts'], counted.backend.stats()['counts'])
                self.assertEqual(emitted.metadata['boundary_bootstraps'], 1577)
                self.assertEqual(len(backend._outputs), 38)
                source = backend.write()
                text = source.read_text()
                self.assertEqual(text.count('hc.bootstrap('), 1579)
                self.assertEqual(backend.stats()['native_trace_executions'], 0)
                self.assertEqual(backend.stats()['compiler_runs'], 0)
                self.assertLess(backend.stats()['constant_payload_bytes'], 1024**2)
            finally:
                backend.close()

    def test_reject_wrong_layer_plan_or_parent_mapping_before_physical_inputs(self):
        plans = list(self.plans); plans[0] = self.plans[1]
        backend = self.count()
        with self.assertRaisesRegex(ValueError, 'exact segment'), \
                patch.object(entry.staged, 'kernel_callbacks', side_effect=AssertionError('No kernel admitted')):
            entry.run_layered(backend, layers=self.layers, head=self.head, plans24=plans, head_plan=self.head_plan,
                first_kernel_plan={}, terminal_kernel_plan={}, original_embedding=self.embedding, prompt_length=9)
        self.assertEqual(backend._inputs, [])
        head = replace(self.head, weights=dict(self.weights))
        with self.assertRaisesRegex(ValueError, 'same original source'), \
                patch.object(entry.staged, 'kernel_callbacks', side_effect=AssertionError('No kernel admitted')):
            entry.run_layered(backend, layers=self.layers, head=head, plans24=self.plans, head_plan=self.head_plan,
                first_kernel_plan={}, terminal_kernel_plan={}, original_embedding=self.embedding, prompt_length=9)
        self.assertEqual(backend._inputs, [])

    def test_real_call_replay_rejects_wrong_call_inventory_and_restores_backend(self):
        plans = list(self.plans)
        plans[0] = entry.bind_plan(self.layers[0].segment, raw_plan(65),
                                   planned_source=entry.source_binding(self.layers[0].segment))
        backend = self.count()
        with self.assertRaisesRegex(Exception, 'covers65calls|covers 65 calls'):
            self.run_fixture(backend, plans24=plans)
        self.assertEqual(len(backend._inputs), 34)
        self.assertIsNone(backend._outputs)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)

    def test_nonidentity_mask_output_is_rejected_without_changing_reference_or_plan(self):
        original = self.fake_lowering
        def wrong_mask(program, weights, active):
            if program is not self.layers[0].segment.program:
                return original(program, weights, active)
            values = [active.input(i) for i in range(34)]
            active.output([*[value*.5 for value in values[:32]], values[32]+0., values[33]])
        plans = list(self.plans)
        plans[0] = entry.bind_plan(self.layers[0].segment, raw_plan(67),
                                   planned_source=entry.source_binding(self.layers[0].segment))
        backend = self.count()
        with patch.object(self, 'fake_lowering', side_effect=wrong_mask), \
                self.assertRaisesRegex(ValueError, 'same two level5 mask'):
            self.run_fixture(backend, plans24=plans)
        self.assertIsNone(backend._outputs)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)

    def test_runtime_exception_keeps_original_inputs_and_cleans_refresh_aliases(self):
        backend = self.count()
        with patch.object(self, 'fake_lowering', side_effect=RuntimeError('fixture failure')), \
                self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            self.run_fixture(backend)
        self.assertEqual(len(backend._inputs), 34)
        self.assertTrue(all(value.is_input for value in backend._inputs))
        self.assertIsNone(backend._outputs)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)


class LayerArtifactExportChecks(unittest.TestCase):
    def setUp(self):
        program, shapes = fixture()
        self.partition = entry.split_program(program, shapes)
        self.weights = {name: SimpleNamespace(shape=shape) for name, shape in shapes.items()}
        self.callback = entry.callbacks(self.partition, self.weights)[0][0]
        self.bound = entry.bind_plan(self.callback.segment, raw_plan(66),
                                     planned_source=entry.source_binding(self.callback.segment))
        self.logical_outputs = []

    def lowering(self, program, weights, active):
        self.assertIs(program, self.callback.segment.program)
        self.assertIs(weights, self.weights)
        values = [active.input(index) for index in range(34)]
        self.assertTrue(all(value.level == 5 for value in values))
        output = [*[value*.5 for value in values[:32]], *values[32:]]
        self.logical_outputs.append(tuple(output))
        active.output(output)

    def export(self, backend, *, bound=None, lowering=None):
        with patch('pt_agent.experiments.qwen_prefill_compile._same_lowering',
                   side_effect=self.lowering if lowering is None else lowering):
            return entry.export_layer(backend, callback=self.callback,
                                      bound_plan=self.bound if bound is None else bound)

    def count(self):
        return StreamingCountBackend(16384, input_level=5, boot_out_level=5)

    def test_count_keeps34_logical_outputs_but_returns32_computed_values_and_mask_alias_manifest(self):
        backend = self.count()
        state = self.export(backend)
        self.assertEqual(len(self.logical_outputs[0]), 34)
        self.assertEqual(state.metadata['segments'][0]['logical_input_calls'], 34)
        self.assertEqual(len(state.metadata['segments'][0]['outputs']), 34)
        self.assertEqual(len(backend._inputs), 34)
        self.assertEqual(len(backend._outputs), 32)
        self.assertTrue(all(not value.is_input and value.level == 5 for value in backend._outputs))
        self.assertEqual(backend.stats()['counts']['boot'], 32)
        self.assertEqual(state.metadata['explicit_boundary_bootstrap_calls'], 32)
        self.assertEqual(state.metadata['internal_plan_bootstrap_calls'], 0)
        self.assertEqual([(row['logical_output'], row['input'], row['level'])
                          for row in state.metadata['mask_alias_outputs']], [(32, 32, 5), (33, 33, 5)])
        for row in state.metadata['mask_alias_outputs']:
            captured = state._known[row['captured_handle']]
            original = state._known[row['original_handle']]
            self.assertIs(captured.value, original.value)
            self.assertIs(original, state.inputs[row['input']])
            self.assertEqual(row['provenance'], 'exact-original-input-alias')
        self.assertEqual(state.metadata['symbolic_output_levels'], [5]*32)
        self.assertEqual(state.metadata['scope'], 'single-layer-trusted-export-preparation-only')
        for key in ('native_compiled_bootstraps', 'native_input_metadata', 'native_output_metadata'):
            self.assertIsNone(state.metadata[key])
        for key in ('native_metadata_verified', 'native_levels_verified', 'native_scales_verified',
                    'native_context_verified', 'compile_verified', 'C2_verified', 'whole_model_verified',
                    'planner_provenance_verified', 'independent_replay_verified'):
            self.assertIs(state.metadata[key], False)
        self.assertEqual(state.metadata['logical_to_physical_outputs'],
                         [dict(logical_output=i, physical_output=i) for i in range(32)])
        json.dumps(state.metadata)

    def test_streaming_emit_ast_retains34_arguments_and_only32_bootstrapped_outputs(self):
        with TemporaryDirectory() as temporary:
            backend = StreamingEmitBackend(16384, directory=Path(temporary),
                frontend='/unused-layer-artifact-frontend.py', max_constant_bytes=1024**2)
            try:
                state = self.export(backend)
                source = backend.write()
                tree = ast.parse(source.read_text())
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                self.assertEqual(len(functions), 1)
                golden = functions[0]
                self.assertEqual([arg.arg for arg in golden.args.args], [f'x{i}' for i in range(34)])
                self.assertEqual(golden.decorator_list[0].args[0].value, ','.join(['c']*34))
                returns = [node for node in ast.walk(golden) if isinstance(node, ast.Return)]
                self.assertEqual(len(returns), 1)
                self.assertIsInstance(returns[0].value, ast.List)
                self.assertEqual([value.id for value in returns[0].value.elts],
                                 [value.name for value in backend._outputs])
                self.assertEqual(len(returns[0].value.elts), 32)
                self.assertTrue(all(value.id not in ('x32', 'x33') for value in returns[0].value.elts))
                boot_calls = [node for node in ast.walk(golden) if isinstance(node, ast.Call)
                              and isinstance(node.func, ast.Attribute) and node.func.attr == 'bootstrap']
                self.assertEqual(len(boot_calls), 32)
                self.assertEqual(backend.stats()['counts']['boot'], 32)
                self.assertEqual(backend.stats()['native_trace_executions'], 0)
                self.assertEqual(backend.stats()['compiler_runs'], 0)
                self.assertFalse(state.metadata['native_metadata_verified'])
                self.assertLess(backend.stats()['constant_payload_bytes'], 1024**2)
            finally:
                backend.close()

    def test_internal_plan_refresh_is_counted_separately_from32_boundary_calls(self):
        bound = entry.bind_plan(self.callback.segment, raw_plan(66, output_refresh=True),
                                planned_source=entry.source_binding(self.callback.segment))
        state = self.export(self.count(), bound=bound)
        self.assertEqual(state.metadata['internal_plan_bootstrap_calls'], 1)
        self.assertEqual(state.metadata['explicit_boundary_bootstrap_calls'], 32)
        self.assertEqual(state.backend.stats()['counts']['boot'], 33)
        self.assertEqual(state.metadata['segments'][0]['output_refresh_positions'], [0])
        self.assertTrue(all(value.level == 5 for value in state.backend._outputs))

    def test_computed_same_level_mask_fails_exact_alias_provenance_and_cleans_backend(self):
        def changed_mask(program, weights, active):
            values = [active.input(i) for i in range(34)]
            active.output([*[value*.5 for value in values[:32]], values[32]+0., values[33]])
        bound = entry.bind_plan(self.callback.segment, raw_plan(67),
                                planned_source=entry.source_binding(self.callback.segment))
        backend = self.count()
        with self.assertRaisesRegex(ValueError, 'same original level5 input mask'):
            self.export(backend, bound=bound, lowering=changed_mask)
        self.assertEqual(len(backend._inputs), 34)
        self.assertEqual(backend.stats()['counts']['boot'], 0)
        self.assertIsNone(backend._outputs)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)

    def test_plan_refreshed_mask_is_rejected_even_when_actual_level_remains5(self):
        plan = raw_plan(66); plan.update(boots=1, outputs=[[32, 1000]])
        bound = entry.bind_plan(self.callback.segment, plan, planned_source=entry.source_binding(self.callback.segment))
        backend = self.count()
        with self.assertRaisesRegex(ValueError, 'same original level5 input mask'):
            self.export(backend, bound=bound)
        self.assertEqual(backend.stats()['counts']['boot'], 1)
        self.assertIsNone(backend._outputs)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)

    def test_missing_logical_output_and_wrong_plan_calls_do_not_publish_partial_artifact(self):
        def missing(program, weights, active):
            values = [active.input(i) for i in range(34)]
            active.output([*[value*.5 for value in values[:31]], *values[32:]])
        bound = entry.bind_plan(self.callback.segment, raw_plan(65),
                                planned_source=entry.source_binding(self.callback.segment))
        backend = self.count()
        with self.assertRaisesRegex(ValueError, 'all34 original logical'):
            self.export(backend, bound=bound, lowering=missing)
        self.assertIsNone(backend._outputs)
        self.assertEqual(backend.stats()['counts']['boot'], 0)
        bound = entry.bind_plan(self.callback.segment, raw_plan(65),
                                planned_source=entry.source_binding(self.callback.segment))
        backend = self.count()
        with self.assertRaisesRegex(Exception, 'covers 65 calls'):
            self.export(backend, bound=bound)
        self.assertIsNone(backend._outputs)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)

    def test_runtime_exception_preserves_failure_and_restores_original34_inputs(self):
        states = []
        original = entry.staged.Segments
        def ledger(backend):
            state = original(backend); states.append(state); return state
        backend = self.count()
        with patch.object(entry.staged, 'Segments', side_effect=ledger), \
                self.assertRaisesRegex(RuntimeError, 'bounded export fixture'):
            self.export(backend, lowering=RuntimeError('bounded export fixture'))
        self.assertEqual(states[0].metadata['status'], 'error')
        self.assertFalse(states[0]._busy)
        self.assertFalse(states[0].closed)
        self.assertEqual(len(backend._inputs), 34)
        self.assertTrue(all(value.is_input for value in backend._inputs))
        self.assertIsNone(backend._outputs)
        self.assertNotIn('input', backend.__dict__)
        self.assertNotIn('output', backend.__dict__)

    def test_wrong_head_or_layer_plan_is_rejected_before_physical_input_creation(self):
        backend = self.count()
        other = entry.bind_plan(self.partition.layers[1], raw_plan(66),
                                 planned_source=entry.source_binding(self.partition.layers[1]))
        with self.assertRaisesRegex(ValueError, 'exact segment'):
            entry.export_layer(backend, callback=self.callback, bound_plan=other)
        head = entry.callbacks(self.partition, self.weights)[1]
        with self.assertRaisesRegex(ValueError, 'one original typed Qwen layer'):
            entry.export_layer(backend, callback=head, bound_plan=self.bound)
        self.assertEqual(backend._inputs, [])


if __name__ == '__main__':
    unittest.main()
