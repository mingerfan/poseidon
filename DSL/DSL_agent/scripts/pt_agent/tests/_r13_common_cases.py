"""R13 intake checks using deterministic protocol fixtures and tiny modules."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments.r13_common_models import FACTORIES
from pt_agent.tests._artifacts import require


INNER_CASE_COUNT = 25
WORKER_CASE_COUNT = 11
REQUIRED_MODELS = tuple(record[key] for record in FACTORIES.values()
                        for key in ('source', 'checkpoint', 'inference_preprocessing_source'))
CIFAR_RECORDS = paths.WORK_RESULTS / 'agent-cifar-offline-records-r1804/cifar10-records'
CIFAR_SELECTIONS = (paths.WORK_RESULTS /
                    'agent-resnet20-original-accuracy-r1805/frozen-selections.json')


@require(REQUIRED_MODELS, CIFAR_RECORDS, CIFAR_SELECTIONS)
class CommonCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from pt_agent.data.samples import Samples
        from pt_agent.experiments import r13_common_models
        from pt_agent.intake import probe
        from pt_agent.intake.graph import Graph, TensorInfo

        cls.np, cls.torch = np, torch
        cls.Samples = Samples
        cls.models, cls.probe = r13_common_models, probe
        cls.Graph, cls.TensorInfo = Graph, TensorInfo

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r13-common-cases-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def samples(self, count=5):
        np = self.np
        return self.Samples('tensor',
                            [{'input': np.array([[i + 0.5, i - 0.25]], dtype=np.float64)}
                             for i in range(count)],
                            ['fixture:' + str(i) for i in range(count)])

    def split(self, calibration):
        return dict(kind='protocol_fixture', calibration=list(calibration),
                    heldout=[], confirm=[], report=[])

    def graph(self, names=('x_1',)):
        tensors = {name: self.TensorInfo(name, (1, 2), 'input') for name in names}
        tensors['output'] = self.TensorInfo('output', (1, 2), 'act')
        return self.Graph('protocol_fixture', [], tensors, {}, list(names), ['output'])

    def tiny_module(self):
        torch = self.torch

        class Exact(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[2., -1.], [0.5, 3.]],
                                                              dtype=torch.float64))
                self.register_buffer('bias', torch.tensor([0.125, -0.25], dtype=torch.float64))

            def forward(self, features):
                return torch.nn.functional.linear(features, self.weight, self.bias)

        return Exact().eval()

    def test_calibration_order_and_at_most_three_real_tuples(self):
        raw = self.samples()
        split = self.split([4, 1, 3, 0, 2])
        with patch.object(raw, 'tensors', wraps=raw.tensors) as reads, \
             patch('pt_agent.intake.import_module', return_value=self.graph()) as imported:
            result = self.probe.import_torch(self.tiny_module(), raw, name='ordered',
                                             source='fixture:make', split=split)
        self.assertEqual([call.args[0] for call in reads.call_args_list], [[4], [1], [3]])
        probes = imported.call_args.kwargs['validation_inputs']
        self.assertIsInstance(probes, tuple)
        self.assertEqual(len(probes), 3)
        self.assertIs(imported.call_args.args[1], probes[0])
        self.assertEqual(imported.call_args.kwargs['frontend'], 'aten')
        for values, index in zip(probes, (4, 1, 3)):
            self.assertIsInstance(values, tuple)
            self.assertEqual(len(values), 1)
            self.assertEqual(tuple(values[0].shape), (1, 2))
            self.np.testing.assert_array_equal(values[0].numpy(), raw.items[index]['input'])
        self.assertIs(result.samples.samples, raw)
        self.assertIs(result.info.split, split)

    def test_short_calibration_preserves_all_available_samples(self):
        for indices in ([3], [3, 1]):
            with self.subTest(indices=indices):
                raw = self.samples()
                with patch.object(raw, 'tensors', wraps=raw.tensors) as reads, \
                     patch('pt_agent.intake.import_module', return_value=self.graph()) as imported:
                    self.probe.import_torch(self.tiny_module(), raw, name='short',
                                            source='fixture:make', split=self.split(indices))
                self.assertEqual([call.args[0] for call in reads.call_args_list],
                                 [[index] for index in indices])
                probes = imported.call_args.kwargs['validation_inputs']
                self.assertEqual(len(probes), len(indices))
                for values, index in zip(probes, indices):
                    self.np.testing.assert_array_equal(values[0].numpy(), raw.items[index]['input'])

    def test_empty_calibration_rejected_before_read_or_trace(self):
        raw = self.samples()
        with patch.object(raw, 'tensors', wraps=raw.tensors) as reads, \
             patch('pt_agent.intake.import_module') as imported:
            with self.assertRaisesRegex(ValueError, 'at least one calibration'):
                self.probe.import_torch(self.tiny_module(), raw, name='empty',
                                        source='fixture:make', split=self.split([]))
        reads.assert_not_called()
        imported.assert_not_called()

    def test_multiinput_probes_follow_first_selected_sample_key_order(self):
        np = self.np
        items = []
        for index in range(3):
            right = np.array([[index + 10., index + 11.]], dtype=np.float64)
            left = np.array([[index, index + 1.]], dtype=np.float64)
            items.append(dict(left=left, right=right) if index == 1 else dict(right=right, left=left))
        raw = self.Samples('tensor', items, ['fixture:' + str(i) for i in range(3)])
        with patch('pt_agent.intake.import_module', return_value=self.graph(('x_1', 'y_1'))) as imported:
            result = self.probe.import_torch(self.tiny_module(), raw, name='multi',
                                             source='fixture:make', split=self.split([2, 1, 0]))
        probes = imported.call_args.kwargs['validation_inputs']
        for values, index in zip(probes, (2, 1, 0)):
            self.assertEqual(len(values), 2)
            np.testing.assert_array_equal(values[0].numpy(), items[index]['right'])
            np.testing.assert_array_equal(values[1].numpy(), items[index]['left'])
        self.assertEqual(result.info.graph.inputs, ['right', 'left'])
        self.assertEqual([entry['name'] for entry in result.info.inputs], ['right', 'left'])

    def test_observer_and_none_pass_through_torch_import(self):
        for observer in (object(), None):
            with self.subTest(observed=observer is not None):
                with patch('pt_agent.intake.import_module', return_value=self.graph()) as imported:
                    self.probe.import_torch(self.tiny_module(), self.samples(), name='observed',
                                            source='fixture:make', split=self.split([0]),
                                            observer=observer)
                self.assertIs(imported.call_args.kwargs['observer'], observer)

    def test_import_model_torch_dispatch_forwards_observer(self):
        observer, raw, model = object(), object(), object()
        for current in (observer, None):
            with self.subTest(observed=current is not None):
                with patch.object(self.probe, 'load_callable', return_value=(model, 'make')) as load, \
                     patch.object(self.probe, 'import_torch', return_value='imported') as imported:
                    self.assertEqual(self.probe.import_model('fixture:make', raw, observer=current),
                                     'imported')
                load.assert_called_once_with('fixture:make')
                imported.assert_called_once_with(model, raw, name='make', source='fixture:make',
                                                 observer=current)

    def test_implicit_split_only_reads_its_calibration_items(self):
        raw = self.samples()
        split = dict(calibration=[3, 0, 2], heldout=[1], confirm=[4], report=[])
        with patch.object(raw, 'split', return_value=split) as partition, \
             patch.object(raw, 'tensors', wraps=raw.tensors) as reads, \
             patch('pt_agent.intake.import_module', return_value=self.graph()):
            result = self.probe.import_torch(self.tiny_module(), raw, name='implicit',
                                             source='fixture:make')
        partition.assert_called_once_with()
        self.assertEqual([call.args[0] for call in reads.call_args_list], [[3], [0], [2]])
        self.assertEqual(result.info.split, split)

    def test_tiny_exact_import_has_real_events_and_no_random_probes(self):
        from pt_agent.experiments.r10_paper_metrics import StageRecorder, read_events
        from pt_agent.intake.graph_exec import forward

        raw, model = self.samples(3), self.tiny_module()
        events_path = self.directory / 'events.jsonl'
        observer = StageRecorder(events_path)
        with patch.object(self.torch, 'randn', side_effect=AssertionError('Random probe requested')):
            imported = self.probe.import_torch(model, raw, name='tiny_exact', source='fixture:make',
                                               split=self.split([2, 0, 1]), observer=observer)
        events = read_events(events_path)
        ends = [event for event in events
                if event['type'] == 'end' and event['stage_id'] == 'intake.real_probe']
        self.assertEqual([event['metadata']['probe_index'] for event in ends], [0, 1, 2])
        self.assertTrue(all(event['status'] == 'ok' for event in ends))
        self.assertFalse(any(event['stage_id'] == 'intake.legacy_probes' for event in events))
        graph = imported.info.graph
        for index in (2, 0, 1):
            image = raw.items[index]['input']
            actual = forward(graph, {graph.inputs[0]: image})[graph.outputs[0]]
            expected = image @ model.weight.detach().numpy().T + model.bias.numpy()
            self.np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(imported.info.source['intake'], graph.intake_stats)

    def test_three_factories_keep_official_sources_and_round(self):
        prefix = 'pt_agent.experiments.r13_common_models:'
        expected = [('resnet20_silu', 'ResNet', 'resnet20.silu.model'),
                    ('vgg16_silu_avgpool', 'VGG16', 'vgg16_silu_avgpool_model'),
                    ('alexnet_silu_avgpool', 'AlexNet', 'alexNet_silu_avgpool_model')]
        self.assertEqual(list(self.models.FACTORIES), [prefix + item[0] for item in expected])
        for name, family, checkpoint in expected:
            with self.subTest(name=name):
                record = self.models.FACTORIES[prefix + name]
                self.assertEqual(record, dict(
                    name=name,
                    source=paths.DACAPO_UPSTREAM / ('python/poly/poly/models/' + family + '.py'),
                    checkpoint=paths.DACAPO_UPSTREAM / 'examples/data' / checkpoint,
                    inference_preprocessing_source=paths.DACAPO_UPSTREAM / ('examples/tests/' + family + '.py'),
                    spec_revision='R13'))
                self.assertTrue(callable(getattr(self.models, name)))

    def resnet_fixture(self):
        torch = self.torch

        class Original(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
                self.register_buffer('running_mean', torch.zeros(2, dtype=torch.float32))
                self.register_buffer('running_var', torch.ones(2, dtype=torch.float32))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.int64))

            def forward(self, features):
                raise AssertionError('Factory checks must not execute a model forward')

        model = Original()
        parallel = torch.nn.DataParallel(model)
        state = {'module.weight': torch.tensor([2., -1.], dtype=torch.float32),
                 'module.running_mean': torch.tensor([0.5, -0.25], dtype=torch.float32),
                 'module.running_var': torch.tensor([2., 3.], dtype=torch.float32),
                 'module.num_batches_tracked': torch.tensor(7, dtype=torch.int64)}
        return model, parallel, state

    def test_resnet_factory_strictly_loads_dataparallel_envelope(self):
        model, parallel, state = self.resnet_fixture()
        record = self.models.FACTORIES['pt_agent.experiments.r13_common_models:resnet20_silu']
        definition = SimpleNamespace(resnet20=lambda: model)
        with patch.object(self.models.dacapo_builders, '_load_definition', return_value=definition) as load, \
             patch.object(self.torch.nn, 'DataParallel', return_value=parallel) as wrap, \
             patch.object(self.torch, 'load', return_value={'epoch': 17, 'state_dict': state}) as checkpoint, \
             patch.object(parallel, 'load_state_dict', wraps=parallel.load_state_dict) as strict_load:
            returned = self.models.resnet20_silu()
        load.assert_called_once_with(record['source'], 'ResNet')
        wrap.assert_called_once_with(model)
        checkpoint.assert_called_once_with(record['checkpoint'], map_location='cpu', weights_only=True)
        strict_load.assert_called_once_with(state, strict=True)
        self.assertIs(returned, model)
        self.assertFalse(returned.training)
        self.assertEqual(set(returned.state_dict()), {key.removeprefix('module.') for key in state})
        for key, expected in state.items():
            actual = returned.state_dict()[key.removeprefix('module.')]
            self.assertEqual(actual.dtype, expected.dtype)
            self.torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_resnet_factory_rejects_missing_original_parameter(self):
        model, parallel, state = self.resnet_fixture()
        state.pop('module.weight')
        with patch.object(self.models.dacapo_builders, '_load_definition',
                          return_value=SimpleNamespace(resnet20=lambda: model)), \
             patch.object(self.torch.nn, 'DataParallel', return_value=parallel), \
             patch.object(self.torch, 'load', return_value={'state_dict': state}):
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                self.models.resnet20_silu()

    def test_resnet_factory_rejects_unknown_checkpoint_key(self):
        model, parallel, state = self.resnet_fixture()
        state['module.unrequested'] = self.torch.tensor([1.])
        with patch.object(self.models.dacapo_builders, '_load_definition',
                          return_value=SimpleNamespace(resnet20=lambda: model)), \
             patch.object(self.torch.nn, 'DataParallel', return_value=parallel), \
             patch.object(self.torch, 'load', return_value={'state_dict': state}):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
                self.models.resnet20_silu()

    def test_resnet_factory_requires_state_dict_envelope(self):
        model, parallel, state = self.resnet_fixture()
        with patch.object(self.models.dacapo_builders, '_load_definition',
                          return_value=SimpleNamespace(resnet20=lambda: model)), \
             patch.object(self.torch.nn, 'DataParallel', return_value=parallel), \
             patch.object(self.torch, 'load', return_value=state), \
             patch.object(parallel, 'load_state_dict', wraps=parallel.load_state_dict) as load:
            with self.assertRaisesRegex(KeyError, 'state_dict'):
                self.models.resnet20_silu()
        load.assert_not_called()

    def test_vgg_and_alex_factories_only_reuse_existing_wrappers(self):
        for name in ('vgg16_silu_avgpool', 'alexnet_silu_avgpool'):
            with self.subTest(name=name):
                original = object()
                with patch.object(self.models.dacapo_builders, name, return_value=original) as factory:
                    self.assertIs(getattr(self.models, name)(), original)
                factory.assert_called_once_with()


@require(REQUIRED_MODELS, CIFAR_RECORDS, CIFAR_SELECTIONS)
class WorkerCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from pt_agent.data import cifar
        from pt_agent.data.samples import LegacyCifar
        from pt_agent.experiments import r13_common_worker
        from pt_agent.intake.graph import Graph, Node, TensorInfo
        from pt_agent.intake.probe import model_info
        from pt_agent.models.info import ModelInfo

        cls.np, cls.torch = np, torch
        cls.cifar, cls.worker = cifar, r13_common_worker
        cls.LegacyCifar = LegacyCifar
        cls.Graph, cls.Node, cls.TensorInfo = Graph, Node, TensorInfo
        cls.model_info = staticmethod(model_info)
        cls.ModelInfo = ModelInfo

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r13-worker-cases-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def calibration_fixture(self):
        """Raw-dataset protocol metadata, with no real CIFAR pixels read."""
        records = self.directory / 'records'
        records.mkdir(exist_ok=True)
        selections = self.directory / 'selections.json'
        indices, labels = [2001, 5, 1007], [6, 9, 2]
        entries = [dict(index=index, label=label,
                        shard=f'train-{index // 1000 * 1000:05d}.records', row=index % 1000)
                   for index, label in zip(indices, labels)]
        metadata = dict(format='poseidon-cifar10-binary-records-v1', record_bytes=3073,
                        row_label_byte=0, preprocessing_applied=False,
                        train_count=50000, test_count=10000,
                        pixel_layout='uint8 planar RGB 3x32x32, C order', shards=[])
        for start in (0, 1000, 2000):
            name = f'train-{start:05d}.records'
            (records / name).touch()
            metadata['shards'].append(dict(file=name, split='train', start=start, count=1000))
        (records / 'dataset.json').write_text(json.dumps(metadata))
        selections.write_text(json.dumps(dict(
            calibration=entries + [dict(index='unused fourth entry')],
            encrypted_test=[dict(index=0, shard='test-00000.records')],
        )))
        pixels = (self.np.arange(3 * 3 * 32 * 32, dtype=self.np.uint32) % 256
                  ).astype(self.np.uint8).reshape(3, 3, 32, 32)
        return records, selections, metadata, entries, pixels, self.np.array(labels, dtype=self.np.int64)

    def semantic_fixture(self):
        """Small exact PT-Graph with declared names and deterministic values."""
        np = self.np
        input_name, output_name = 'camera_pixels', 'declared_logits'
        weights = np.zeros((10, 3072), dtype=np.float64)
        weights[np.arange(10), np.arange(10)] = 1.
        tensors = {
            input_name: self.TensorInfo(input_name, (1, 3, 32, 32), 'input'),
            'flat': self.TensorInfo('flat', (1, 3072), 'act'),
            'head_weight': self.TensorInfo('head_weight', (10, 3072), 'weight'),
            'head_bias': self.TensorInfo('head_bias', (10,), 'weight'),
            output_name: self.TensorInfo(output_name, (1, 10), 'act'),
        }
        nodes = [self.Node('flatten_original', 'flatten', [input_name], ['flat'],
                           dict(start_dim=1, end_dim=3), 'protocol fixture'),
                 self.Node('original_head', 'linear', ['flat'], [output_name],
                           dict(weight='head_weight', bias='head_bias'), 'protocol fixture')]
        graph = self.Graph('protocol_original', nodes, tensors,
                           dict(head_weight=weights, head_bias=np.zeros(10, dtype=np.float64)),
                           [input_name], [output_name])
        graph.intake_stats = dict(original_nodes=2, folded_nodes=0, original_unmapped={})
        images = np.arange(3 * 3072, dtype=np.float64).reshape(3, 3, 32, 32) / 4096
        references = [images[index:index + 1].reshape(1, -1)[:, :10].copy() for index in range(3)]
        split = dict(kind='legacy_cifar_intake_only', calibration=[0, 1, 2],
                     heldout=[], confirm=[], report=[], sample_ids=['train:2001', 'train:5', 'train:1007'])
        info = self.model_info(dict(type='torch', path='protocol:original', original_note='retain'),
                               graph, split=split, has_labels=True,
                               handling=dict(mode='original tensors', input_note='retain'),
                               assumptions=[dict(original_assumption='retain')])
        source = dict(records=str(self.directory / 'records'),
                      selections=str(self.directory / 'selections.json'), split='train',
                      indices=[2001, 5, 1007], sample_ids=list(split['sample_ids']), labels=[6, 9, 2],
                      normalization=dict(mean=list(self.cifar.MEAN), std=list(self.cifar.STD)))
        return SimpleNamespace(info=info, samples=object(), original=object()), images, references, source

    def test_read_calibration_only_uses_first_three_original_training_rows(self):
        records, selections, metadata, entries, pixels, labels = self.calibration_fixture()
        with patch.object(self.cifar, 'RECORDS', records), \
             patch.object(self.cifar, 'SELECTIONS', selections), \
             patch.object(self.cifar, 'read_rows', return_value=(pixels, labels)) as read:
            images, returned_labels, source = self.worker.read_calibration(records, selections)
        read.assert_called_once_with('train', [2001, 5, 1007])
        self.assertEqual(images.shape, (3, 3, 32, 32))
        self.assertEqual(images.dtype, self.np.float64)
        expected = ((pixels.astype(self.np.float64) / 255 -
                     self.np.array(self.cifar.MEAN).reshape(1, 3, 1, 1)) /
                    self.np.array(self.cifar.STD).reshape(1, 3, 1, 1))
        self.np.testing.assert_array_equal(images, expected)
        self.np.testing.assert_array_equal(returned_labels, labels)
        self.assertEqual(source['split'], 'train')
        self.assertEqual(source['indices'], [2001, 5, 1007])
        self.assertEqual(source['sample_ids'], ['train:2001', 'train:5', 'train:1007'])
        self.assertEqual(source['labels'], [6, 9, 2])
        self.assertEqual(source['entries'], entries)
        self.assertEqual(source['records'], str(records.resolve()))
        self.assertEqual(source['selections'], str(selections.resolve()))
        self.assertEqual(source['normalization'],
                         dict(mean=list(self.cifar.MEAN), std=list(self.cifar.STD)))

    def test_read_calibration_rejects_other_paths_before_any_pixel_read(self):
        records, selections, _, _, pixels, labels = self.calibration_fixture()
        for inputs, selected in ((self.directory / 'other-records', selections),
                                 (records, self.directory / 'other-selections.json')):
            with self.subTest(inputs=inputs, selected=selected):
                with patch.object(self.cifar, 'RECORDS', records), \
                     patch.object(self.cifar, 'SELECTIONS', selections), \
                     patch.object(self.cifar, 'read_rows', return_value=(pixels, labels)) as read:
                    with self.assertRaisesRegex(ValueError, 'existing CIFAR records'):
                        self.worker.read_calibration(inputs, selected)
                read.assert_not_called()

    def test_read_calibration_checks_dataset_and_selected_shard_metadata(self):
        records, selections, metadata, _, pixels, labels = self.calibration_fixture()
        malformed = []
        for key, value in (('format', 'other-format'), ('preprocessing_applied', True),
                           ('train_count', 49999), ('pixel_layout', 'interleaved RGB')):
            record = deepcopy(metadata)
            record[key] = value
            malformed.append(record)
        missing_shard = deepcopy(metadata)
        missing_shard['shards'] = missing_shard['shards'][:-1]
        malformed.append(missing_shard)
        for index, record in enumerate(malformed):
            with self.subTest(index=index):
                (records / 'dataset.json').write_text(json.dumps(record))
                with patch.object(self.cifar, 'RECORDS', records), \
                     patch.object(self.cifar, 'SELECTIONS', selections), \
                     patch.object(self.cifar, 'read_rows', return_value=(pixels, labels)) as read:
                    with self.assertRaises(ValueError):
                        self.worker.read_calibration(records, selections)
                read.assert_not_called()

    def test_read_calibration_checks_index_label_shard_and_row(self):
        records, selections, _, entries, pixels, labels = self.calibration_fixture()
        cases = [('index', -1), ('index', 50000), ('index', True),
                 ('label', -1), ('label', 10), ('label', True),
                 ('shard', 'test-02000.records'), ('row', 999)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                selected = deepcopy(entries)
                selected[0][key] = value
                selections.write_text(json.dumps(dict(calibration=selected)))
                with patch.object(self.cifar, 'RECORDS', records), \
                     patch.object(self.cifar, 'SELECTIONS', selections), \
                     patch.object(self.cifar, 'read_rows', return_value=(pixels, labels)) as read:
                    with self.assertRaises(ValueError):
                        self.worker.read_calibration(records, selections)
                read.assert_not_called()

    def test_read_calibration_rejects_label_mismatch_with_original_row(self):
        records, selections, _, _, pixels, labels = self.calibration_fixture()
        labels = labels.copy()
        labels[1] = 8
        with patch.object(self.cifar, 'RECORDS', records), \
             patch.object(self.cifar, 'SELECTIONS', selections), \
             patch.object(self.cifar, 'read_rows', return_value=(pixels, labels)) as read:
            with self.assertRaisesRegex(ValueError, 'label disagrees'):
                self.worker.read_calibration(records, selections)
        read.assert_called_once_with('train', [2001, 5, 1007])

    def test_finalize_info_retains_inferred_fields_and_returns_legacy_source(self):
        imported, _, _, source = self.semantic_fixture()
        original_source = deepcopy(imported.info.source)
        original_split = deepcopy(imported.info.split)
        model = next(iter(FACTORIES))
        with patch.object(self.cifar, 'read_rows', side_effect=AssertionError('Pixel read during metadata')) as read, \
             patch.object(self.LegacyCifar, 'get', side_effect=AssertionError('Split read during metadata')) as get:
            info, sample_source = self.worker.finalize_info(imported, model, FACTORIES[model], source)
        read.assert_not_called()
        get.assert_not_called()
        self.assertIsInstance(sample_source, self.LegacyCifar)
        self.assertIs(sample_source.graph, imported.info.graph)
        self.assertEqual(sample_source.normalization, (self.cifar.MEAN, self.cifar.STD))
        for name in ('inputs', 'task', 'metric', 'threshold', 'assumptions', 'sequence'):
            self.assertEqual(getattr(info, name), getattr(imported.info, name))
        for name, value in original_source.items():
            self.assertEqual(info.source[name], value)
        self.assertEqual(info.source['factory'], model)
        self.assertEqual(info.source['spec_revision'], 'R13')
        self.assertEqual(info.source['normalization'], source['normalization'])
        self.assertEqual(info.source['calibration_source'], source)
        self.assertEqual(info.source['preprocessing_provenance'], 'official inference test')
        self.assertIs(info.source['training_preprocessing_verified'], False)
        self.assertEqual(info.input_handling,
                         dict(imported.info.input_handling, internal_regression_source='legacy_cifar'))
        self.assertEqual(info.split, dict(kind='legacy_cifar',
                         calibration='frozen training selection', heldout=list(range(9000, 10000)),
                         confirm=list(range(8000, 9000)), report=list(range(0, 8000))))
        self.assertEqual(imported.info.split, original_split)
        info.source['intake']['original_nodes'] = -1
        info.source['calibration_source']['indices'][0] = -1
        self.assertEqual(imported.info.source, original_source)
        self.assertEqual(source['indices'], [2001, 5, 1007])

    def test_finalized_modelinfo_roundtrip_retains_metadata_and_graph_semantics(self):
        imported, images, references, source = self.semantic_fixture()
        model = next(iter(FACTORIES))
        info, _ = self.worker.finalize_info(imported, model, FACTORIES[model], source)
        directory = self.directory / 'model_info'
        info.save(directory)
        restored = self.ModelInfo.load(directory)
        for name in ('source', 'split', 'inputs', 'task', 'metric', 'threshold',
                     'assumptions', 'sequence', 'input_handling'):
            self.assertEqual(getattr(restored, name), getattr(info, name))
        self.assertEqual(restored.graph.intake_stats, imported.info.graph.intake_stats)
        self.assertEqual(restored.graph.inputs, ['camera_pixels'])
        self.assertEqual(restored.graph.outputs, ['declared_logits'])
        self.assertTrue(self.worker.compare_graph(restored.graph, images, references)['passed'])

    def test_compare_graph_uses_declared_input_and_output_names(self):
        imported, images, references, _ = self.semantic_fixture()
        observation = self.worker.compare_graph(imported.info.graph, images, references)
        self.assertTrue(observation['passed'])
        self.assertEqual(observation['criterion'], '1e-12+1e-12*abs(reference)')
        self.assertEqual(len(observation['samples']), 3)
        for index, sample in enumerate(observation['samples']):
            self.assertEqual(sample['probe_index'], index)
            self.assertEqual(sample['input_name'], 'camera_pixels')
            self.assertEqual(sample['output_name'], 'declared_logits')
            self.assertEqual(sample['reference_shape'], [1, 10])
            self.assertEqual(sample['output_shape'], [1, 10])
            self.assertTrue(sample['finite'])
            self.assertTrue(sample['shape_ok'])
            self.assertEqual(sample['max_absolute_error'], 0.)

    def test_compare_graph_records_math_failure_and_stops_later_probes(self):
        imported, images, references, _ = self.semantic_fixture()
        references[1][0, 7] += 1e-7
        observation = self.worker.compare_graph(imported.info.graph, images, references)
        self.assertFalse(observation['passed'])
        self.assertEqual(len(observation['samples']), 2)
        first, failed = observation['samples']
        self.assertTrue(first['passed'])
        self.assertFalse(failed['passed'])
        self.assertTrue(failed['finite'])
        self.assertTrue(failed['shape_ok'])
        self.assertGreater(failed['max_absolute_error'], 1e-8)
        self.assertEqual(failed['reference'], references[1].tolist())
        self.assertNotEqual(failed['reference'], failed['output'])

    def test_compare_graph_records_nonfinite_and_shape_failures_as_json(self):
        for case in ('output_nan', 'reference_inf', 'wrong_shape'):
            with self.subTest(case=case):
                imported, images, references, _ = self.semantic_fixture()
                graph = imported.info.graph
                if case == 'output_nan':
                    graph.weights['head_bias'][2] = self.np.nan
                elif case == 'reference_inf':
                    references[0][0, 4] = self.np.inf
                else:
                    graph.weights['head_weight'] = graph.weights['head_weight'][:9]
                    graph.weights['head_bias'] = graph.weights['head_bias'][:9]
                observation = self.worker.compare_graph(graph, images, references)
                self.assertFalse(observation['passed'])
                self.assertEqual(len(observation['samples']), 1)
                sample = observation['samples'][0]
                self.assertFalse(sample['passed'])
                self.assertIsNone(sample['max_absolute_error'])
                if case == 'wrong_shape':
                    self.assertTrue(sample['finite'])
                    self.assertFalse(sample['shape_ok'])
                    self.assertEqual(sample['output_shape'], [1, 9])
                else:
                    self.assertFalse(sample['finite'])
                    self.assertTrue(sample['shape_ok'])
                    values = sample['output'] if case == 'output_nan' else sample['reference']
                    self.assertIsNone(values[0][2 if case == 'output_nan' else 4])
                json.dumps(observation, allow_nan=False)

    def test_worker_persists_failed_math_without_compilation_or_split_reads(self):
        from pt_agent.intake import coverage, probe

        imported, images, _, source = self.semantic_fixture()
        imported.info.graph.weights['head_bias'][0] = 1e-6
        torch = self.torch

        class Original(torch.nn.Module):
            def forward(self, pixels):
                return pixels.flatten(1)[:, :10]

        model = next(iter(FACTORIES))
        work = self.directory / 'worker'
        cli = ['--model', model, '--inputs', str(self.directory / 'records'),
               '--selections', str(self.directory / 'selections.json'), '--work-dir', str(work)]
        # Intake completion is a protocol fixture here; no full model is imported.
        probe_observation = dict(passed=True, attempted=3, completed=3,
                                 criterion='1e-9*(1+abs(reference))')
        with patch.object(self.worker, 'read_calibration',
                          return_value=(images, self.np.array([6, 9, 2]), source)), \
             patch.object(probe, 'load_callable', return_value=(Original().eval(), 'protocol_original')), \
             patch.object(probe, 'import_torch', return_value=imported), \
             patch.object(self.worker, '_probe_observation', return_value=probe_observation), \
             patch.object(self.worker.signal, 'signal'), patch.object(self.worker.signal, 'alarm'), \
             patch.object(coverage, 'report', side_effect=AssertionError('Coverage after math failure')) as check, \
             patch.object(self.cifar, 'read_rows', side_effect=AssertionError('Unexpected CIFAR pixel read')) as read, \
             patch.object(self.LegacyCifar, 'get', side_effect=AssertionError('Unexpected split read')) as get:
            code = self.worker.main(cli)
        self.assertEqual(code, 1)
        check.assert_not_called()
        read.assert_not_called()
        get.assert_not_called()
        result = json.loads((work / 'worker_result.json').read_text())
        math_record = json.loads((work / 'graph_math.json').read_text())
        self.assertEqual(result['status'], 'math_failed')
        self.assertEqual(result['failed_phase'], 'graph_semantics')
        self.assertFalse(result['graph_math_passed'])
        self.assertEqual(result['graph_math'], math_record)
        self.assertEqual(result['artifacts']['graph_math'], str(work / 'graph_math.json'))
        self.assertFalse(math_record['passed'])
        self.assertGreater(math_record['samples'][0]['max_absolute_error'], 1e-7)
        for name in ('lowering_ready', 'compiled_bootstraps', 'predicted_latency', 'CKKS_latency'):
            self.assertIsNone(result[name])
        self.assertEqual(result['llm_calls'], 0)
        self.assertEqual(result['real_ckks_attempts'], 0)
        self.assertFalse((work / 'model_info').exists())


if __name__ == '__main__':
    unittest.main()
