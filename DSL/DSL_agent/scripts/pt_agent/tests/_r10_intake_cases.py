"""R10 real-tensor checks, executed only by the existing intake interpreter."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


INNER_CASE_COUNT = 13


class Recorder:
    def __init__(self):
        self.events = []

    @contextmanager
    def span(self, stage_id, **metadata):
        self.events.append(dict(stage_id=stage_id, event='begin', metadata=metadata))
        try:
            yield
        except BaseException:
            self.events.append(dict(stage_id=stage_id, event='end', status='error', metadata=metadata))
            raise
        else:
            self.events.append(dict(stage_id=stage_id, event='end', status='ok', metadata=metadata))


class RealProbeTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

        class Original(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[2., -1.], [1., 3.]], dtype=torch.float64))

            def forward(self, features, mask):
                return torch.nn.functional.linear(features, self.weight)*mask

        self.model = Original().eval()
        self.example = (torch.tensor([[1., 2.]]), torch.tensor([[1., 0.]]))
        self.probes = tuple((torch.tensor([[float(i+3), 2.]]), torch.tensor([[1., 1.]]))
                            for i in range(3))

    def test_three_real_probes_no_random_and_weight_binding(self):
        import numpy as np
        from pt_agent.intake import import_module
        from pt_agent.intake.graph_exec import forward
        recorder = Recorder()
        with patch.object(self.torch, 'randn', side_effect=AssertionError('Random probe called')):
            graph = import_module(self.model, self.example, 'real_probe', frontend='aten',
                                  validation_inputs=self.probes, observer=recorder)
        ends = [event for event in recorder.events
                if event['stage_id'] == 'intake.real_probe' and event['event'] == 'end']
        self.assertEqual([event['metadata']['probe_index'] for event in ends], [0, 1, 2])
        self.assertTrue(all(event['status'] == 'ok' for event in ends))
        self.assertTrue(graph.weights)
        np.testing.assert_array_equal(graph.weights['weight'], self.model.weight.detach().numpy())
        actual = forward(graph, {key: value.double().numpy() for key, value in zip(graph.inputs, self.example)})
        expected = self.model(*(value.double() for value in self.example)).detach().numpy()
        np.testing.assert_allclose(actual[graph.outputs[0]], expected, atol=1e-12, rtol=0)
        self.assertEqual([event['stage_id'] for event in recorder.events if event['event'] == 'begin'],
                         ['intake.deepcopy', 'intake.make_fx', 'intake.real_probes',
                          'intake.real_probe', 'intake.real_probe', 'intake.real_probe',
                          'intake.shape_prop', 'intake.map_weights'])

    def test_explicit_empty_probes_rejected_before_trace(self):
        from pt_agent.intake.aten_frontend import import_module_aten
        with patch('pt_agent.intake.aten_frontend.make_fx') as trace:
            with self.assertRaisesRegex(ValueError, 'at least one'):
                import_module_aten(self.model, self.example, 'empty', validation_inputs=[])
            trace.assert_not_called()

    def test_probe_input_count_rejected_before_trace(self):
        from pt_agent.intake.aten_frontend import import_module_aten
        with patch('pt_agent.intake.aten_frontend.make_fx') as trace:
            with self.assertRaisesRegex(ValueError, 'count'):
                import_module_aten(self.model, self.example, 'count', validation_inputs=[self.probes[0][:1]])
            trace.assert_not_called()

    def test_probe_shape_rejected_before_trace(self):
        from pt_agent.intake.aten_frontend import import_module_aten
        bad = (self.probes[0][0][:, :1], self.probes[0][1])
        with patch('pt_agent.intake.aten_frontend.make_fx') as trace:
            with self.assertRaisesRegex(ValueError, 'shapes'):
                import_module_aten(self.model, self.example, 'shape', validation_inputs=[bad])
            trace.assert_not_called()

    def test_each_probe_numeric_mismatch_uses_original_intake_error(self):
        from torch.fx.experimental.proxy_tensor import make_fx
        from pt_agent.intake.aten_frontend import CONTROL_FLOW_REASON, import_module_aten
        from pt_agent.intake import IntakeError
        for failed_index in range(3):
            with self.subTest(failed_index=failed_index):
                gm = make_fx(self.model)(*(value.double() for value in self.example))
                failed_value = float(self.probes[failed_index][0][0, 0])
                gm.register_forward_hook(lambda module, args, output, value=failed_value:
                                         output+1 if float(args[0][0, 0]) == value else output)
                recorder = Recorder()
                with patch('pt_agent.intake.aten_frontend.make_fx', return_value=lambda *args: gm):
                    with self.assertRaises(IntakeError) as caught:
                        import_module_aten(self.model, self.example, 'mismatch',
                                           validation_inputs=self.probes, observer=recorder)
                self.assertEqual(caught.exception.entries[0]['reason'], CONTROL_FLOW_REASON)
                ends = [event for event in recorder.events
                        if event['stage_id'] == 'intake.real_probe' and event['event'] == 'end']
                self.assertEqual(len(ends), failed_index+1)
                self.assertEqual(ends[-1]['status'], 'error')
                self.assertEqual(recorder.events[-1]['stage_id'], 'intake.real_probes')
                self.assertEqual(recorder.events[-1]['status'], 'error')

    def test_default_none_preserves_three_legacy_probes(self):
        from pt_agent.intake import import_module
        def fixed(shape, **kwargs):
            return self.torch.ones(shape, dtype=kwargs['dtype'])
        with patch.object(self.torch, 'randn', side_effect=fixed) as random:
            graph = import_module(self.model, self.example, 'legacy', frontend='aten')
        self.assertEqual(random.call_count, 3*len(self.example))
        self.assertTrue(graph.weights)

    def test_fx_rejects_real_probe_argument(self):
        from pt_agent.intake import import_module
        with patch('pt_agent.intake.import_module_fx') as trace:
            with self.assertRaisesRegex(ValueError, 'aten frontend'):
                import_module(self.model, self.example, 'fx', validation_inputs=self.probes)
            trace.assert_not_called()


class HFProbeTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        from pt_agent.data.samples import Samples
        from pt_agent.intake.graph import Graph, TensorInfo
        raw = Samples('text', ['original sentence '+str(i) for i in range(223)],
                      ['sample:'+str(i) for i in range(223)])
        config = SimpleNamespace(architectures=['Qwen2ForCausalLM'], model_type='qwen2',
                                 max_position_embeddings=131072, num_hidden_layers=24,
                                 hidden_size=2, vocab_size=7, num_labels=None, id2label=None)
        processor = MagicMock(return_value={'input_ids': [[1, 2, 3] for _ in range(43)]})
        local = SimpleNamespace(source='original-model', config=config, processor=processor,
                                module=MagicMock(), input_kind='tokens')
        graph = Graph('qwen2', [], {'inputs_embeds': TensorInfo('inputs_embeds', (1, 8, 2), 'input'),
                                  'attention_mask': TensorInfo('attention_mask', (1, 8), 'input'),
                                  'output': TensorInfo('output', (1, 1, 7), 'act')},
                      {'head': np.ones((7, 2))}, ['inputs_embeds', 'attention_mask'], ['output'])
        graph.intake_stats = dict(transformer_normalized=True)
        def selected(instance, indices):
            return {'inputs_embeds': np.full((1, 8, 2), float(indices[0])),
                    'attention_mask': np.ones((1, 8))}, None
        return raw, local, graph, selected

    def test_hf_three_real_samples_and_no_report_selection(self):
        from pt_agent.intake import hf_frontend as hf
        raw, local, graph, selected = self.fixture()
        split = raw.split()
        seen = []
        def tracked(instance, indices):
            seen.append(list(indices))
            return selected(instance, indices)
        observer = Recorder()
        with patch.object(hf, 'load_local', return_value=local), \
             patch.object(hf, 'resolve_local', return_value='original-model'), \
             patch.object(hf.HFInputs, 'select', new=tracked), \
             patch.object(hf, 'boundary_module', return_value=object()), \
             patch('pt_agent.intake.import_module', return_value=graph) as imported, \
             patch('pt_agent.intake.transformer.normalize', return_value=graph):
            result = hf.import_local('original-model', raw, observer=observer)
        self.assertEqual(seen, [[index] for index in split['calibration'][:4]])
        self.assertEqual(local.processor.call_args.args[0],
                         [raw.items[index] for index in split['calibration']+split['heldout']])
        self.assertEqual([len(split[key]) for key in ('calibration', 'heldout', 'confirm', 'report')],
                         [23, 20, 20, 160])
        probes = imported.call_args.kwargs['validation_inputs']
        self.assertEqual(len(probes), 3)
        self.assertEqual([float(values[0][0, 0, 0]) for values in probes],
                         [float(index) for index in split['calibration'][1:4]])
        self.assertTrue(all(tuple(value.shape) == shape for values in probes
                            for value, shape in zip(values, ((1, 8, 2), (1, 8)))))
        self.assertEqual(result.info.sequence['decision_samples'], 43)
        self.assertIs(imported.call_args.kwargs['observer'], observer)

    def test_hf_validation_name_order_checked_before_trace(self):
        from pt_agent.intake import hf_frontend as hf
        raw, local, graph, selected = self.fixture()
        trace_index = raw.split()['calibration'][0]
        def reordered(instance, indices):
            arrays, labels = selected(instance, indices)
            return (arrays if indices[0] == trace_index else dict(reversed(list(arrays.items())))), labels
        with patch.object(hf, 'load_local', return_value=local), \
             patch.object(hf, 'resolve_local', return_value='original-model'), \
             patch.object(hf.HFInputs, 'select', new=reordered), \
             patch('pt_agent.intake.import_module') as imported:
            with self.assertRaisesRegex(ValueError, 'names/order'):
                hf.import_local('original-model', raw)
        imported.assert_not_called()

    def test_loading_observer_records_local_only_and_exception_close(self):
        from pt_agent.intake import hf_frontend as hf
        raw, local, graph, selected = self.fixture()
        transformers = MagicMock()
        transformers.AutoConfig.from_pretrained.return_value = local.config
        transformers.AutoModelForCausalLM.from_pretrained.return_value = local.module
        local.module.double.return_value = local.module
        local.module.eval.return_value = local.module
        transformers.AutoTokenizer.from_pretrained.side_effect = RuntimeError('tokenizer failure')
        observer = Recorder()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'config.json').write_text('{}')
            (path/'model.safetensors').touch()
            with patch.dict('sys.modules', transformers=transformers):
                with self.assertRaisesRegex(ValueError, 'tokenizer failure'):
                    hf.load_local(path, observer=observer)
            transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                str(path), config=local.config, use_safetensors=True,
                trust_remote_code=False, local_files_only=True)
        self.assertEqual([event['stage_id'] for event in observer.events if event['event'] == 'begin'],
                         ['intake.config', 'intake.weights', 'intake.float64', 'intake.tokenizer'])
        self.assertEqual(observer.events[-1]['stage_id'], 'intake.tokenizer')
        self.assertEqual(observer.events[-1]['status'], 'error')

    def test_import_dispatch_forwards_observer(self):
        from pt_agent.intake.probe import import_model
        observer = Recorder()
        samples = object()
        with patch('pt_agent.intake.hf_frontend.import_local', return_value='imported') as imported:
            self.assertEqual(import_model('original/model', samples, observer=observer), 'imported')
        imported.assert_called_once_with('original/model', samples, observer=observer)


class IdentityAndObservationTests(unittest.TestCase):
    def test_identity_checks_actual_layers(self):
        import torch
        from pt_agent.experiments.r10_paper_worker import check_qwen_identity
        config = SimpleNamespace(num_hidden_layers=24, hidden_size=896, num_attention_heads=14,
                                 num_key_value_heads=2, intermediate_size=4864, vocab_size=151936,
                                 tie_word_embeddings=True, model_type='qwen2',
                                 architectures=['Qwen2ForCausalLM'], max_position_embeddings=131072,
                                 rope_theta=1000000., use_cache=True, use_sliding_window=False,
                                 torch_dtype='bfloat16')
        original = SimpleNamespace(config=config, module=SimpleNamespace(
            model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)]))))
        self.assertEqual(check_qwen_identity(SimpleNamespace(original=original))['actual_layers'], 24)
        original.module.model.layers = torch.nn.ModuleList(list(original.module.model.layers)[:-1])
        with self.assertRaises(ValueError):
            check_qwen_identity(SimpleNamespace(original=original))

    def test_optional_observer_preserves_exception(self):
        from pt_agent.intake.observation import observe
        with self.assertRaisesRegex(RuntimeError, 'preserved'):
            with observe(None, 'no_observer'):
                raise RuntimeError('preserved')
        recorder = Recorder()
        with self.assertRaisesRegex(RuntimeError, 'preserved'):
            with observe(recorder, 'observed', probe_index=2):
                raise RuntimeError('preserved')
        self.assertEqual(recorder.events[-1], dict(stage_id='observed', event='end', status='error',
                                                  metadata=dict(probe_index=2)))


if __name__ == '__main__':
    unittest.main()
