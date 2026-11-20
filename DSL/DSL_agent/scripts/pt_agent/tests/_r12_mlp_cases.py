"""R12 arithmetic and real-client gates, selected separately by the worker.

Only MathCases runs before the native build. ClientGateCases always invokes the
supplied binary with --check-only and never requests encrypted execution.
"""

from contextlib import redirect_stderr
from copy import deepcopy
import gzip
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.tests._artifacts import require


OFFICIAL_MODEL = paths.DACAPO_UPSTREAM / 'examples/data/mlp.model'
OFFICIAL_SOURCE = paths.DACAPO_UPSTREAM / 'examples/benchmarks/MLP.py'
CONTEXT_ID = 'poseidon-agent-n15-q13-p1-tc128-v1'
MATH_CASE_COUNT = 26
CLIENT_GATE_CASE_COUNT = 13


def samples_record():
    return dict(format='pt-agent-r12-mlp-samples-v1',
                source=dict(model=str(OFFICIAL_MODEL.resolve()),
                            mnist=str((Path(__file__).parent / 'fixture-images.gz').resolve()),
                            split='train', normalization='float64(uint8)/255'),
                samples=[dict(index=i, input784=[0.0] * 784,
                              reference10=[0.0] * 10) for i in range(3)])


def plan_record():
    fixture = (paths.REPO / 'third_party/ckks-runtime/docs/runtime-plan/v1/testdata/valid'
               / 'v001_inline_encode_host_compute.json')
    record = json.loads(fixture.read_text())
    record['target']['device_counts'] = [0]
    for value in record['values']:
        value.update(context=CONTEXT_ID, level=12, scale_log2=45)
    return record


@require(OFFICIAL_MODEL, OFFICIAL_SOURCE)
class MathCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from pt_agent.experiments import r12_mlp_worker
        cls.np, cls.torch, cls.worker = np, torch, r12_mlp_worker

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r12-math-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def weights(self):
        """Finite deterministic protocol fixture, never calibration data."""
        np = self.np
        first = np.zeros((100, 784), dtype=np.float64)
        second = np.zeros((10, 100), dtype=np.float64)
        for row in range(100):
            first[row, (row * 7 + 781) % 784] = (1 if row % 2 else -1) / 8
        for row in range(10):
            second[row, (row * 9 + 97) % 100] = (1 if row % 2 else -1) / 4
        return {'linear1.weight': first,
                'linear1.bias': (np.arange(100, dtype=np.float64) % 5 - 2) / 32,
                'linear2.weight': second,
                'linear2.bias': (np.arange(10, dtype=np.float64) - 5) / 64}

    def checkpoint_state(self):
        """The four linear tensors plus the ten observed BN state entries."""
        torch = self.torch
        state = {key: torch.from_numpy(value.copy()).to(dtype=torch.float32)
                 for key, value in self.weights().items()}
        for prefix, width in (('bn1', 100), ('bn2', 10)):
            state[prefix + '.weight'] = torch.ones(width, dtype=torch.float32)
            state[prefix + '.bias'] = torch.arange(width, dtype=torch.float32) / 128
            state[prefix + '.running_mean'] = torch.zeros(width, dtype=torch.float32)
            state[prefix + '.running_var'] = torch.full((width,), 2., dtype=torch.float32)
            state[prefix + '.num_batches_tracked'] = torch.tensor(3, dtype=torch.int64)
        return state

    def expected_bn_metadata(self):
        records = {}
        for prefix, width in (('bn1', 100), ('bn2', 10)):
            for suffix in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'):
                counter = suffix == 'num_batches_tracked'
                name = prefix + '.' + suffix
                records[name] = dict(name=name, shape=[] if counter else [width],
                                     dtype='int64' if counter else 'float32',
                                     unused_by_artifact=True)
        return records

    def test_idx_keeps_first_three_original_indices_and_divides_by_255(self):
        np = self.np
        raw = (np.arange(4 * 784, dtype=np.uint32) % 256).astype(np.uint8).reshape(4, 784)
        raw[1, 0], raw[2, 0], raw[3, 0] = 53, 217, 101
        image_file = self.directory / 'images.gz'
        with gzip.open(image_file, 'wb') as stream:
            stream.write(struct.pack('>IIII', 2051, 4, 28, 28))
            stream.write(raw.tobytes())
        images, metadata = self.worker.read_mnist_images(image_file)
        self.assertEqual(images.shape, (3, 784))
        self.assertEqual(images.dtype, np.float64)
        np.testing.assert_array_equal(images, raw[:3].astype(np.float64) / 255)
        self.assertEqual(metadata['indices'], [0, 1, 2])
        self.assertEqual((metadata['magic'], metadata['rows'], metadata['cols']), (2051, 28, 28))
        self.assertEqual(Path(metadata['source']), image_file.resolve())

    def test_idx_rejects_wrong_header_dimensions_and_truncation(self):
        cases = [(999, 3, 28, 28, bytes(3 * 784)),
                 (2051, 3, 14, 56, bytes(3 * 784)),
                 (2051, 2, 28, 28, bytes(2 * 784)),
                 (2051, 3, 28, 28, bytes(3 * 784 - 1))]
        for index, (magic, count, rows, cols, body) in enumerate(cases):
            with self.subTest(index=index):
                path = self.directory / ('invalid-' + str(index) + '.gz')
                with gzip.open(path, 'wb') as stream:
                    stream.write(struct.pack('>IIII', magic, count, rows, cols))
                    stream.write(body)
                with self.assertRaises(ValueError):
                    self.worker.read_mnist_images(path)

    def test_input_padding_is_sixteen_zeros_then_an_800_period(self):
        np = self.np
        image = np.arange(784, dtype=np.float64) / 783
        packed = self.worker.pack_input(image)
        period = np.concatenate([image, np.zeros(16, dtype=np.float64)])
        self.assertEqual(packed.shape, (16384,))
        np.testing.assert_array_equal(packed, np.resize(period, 16384))
        np.testing.assert_array_equal(packed[784:800], np.zeros(16))
        np.testing.assert_array_equal(packed[800:1584], image)

    def test_plain_payloads_repeat_their_own_length_and_truncate(self):
        np = self.np
        for length in (800, 100, 10):
            with self.subTest(length=length):
                payload = np.arange(length, dtype=np.float64) / 8
                repeated = self.worker.repeat_slots(payload)
                np.testing.assert_array_equal(repeated, np.resize(payload, 16384))
                self.assertEqual(repeated[-1], payload[(16384 - 1) % length])
        with self.assertRaises(ValueError):
            self.worker.repeat_slots([])

    def test_matrix_reference_matches_independent_row_sums(self):
        np = self.np
        image = (np.arange(784, dtype=np.float64) % 256) / 255
        weights = self.weights()
        hidden = [sum(float(image[c]) * float(weights['linear1.weight'][r, c])
                      for c in range(784)) + float(weights['linear1.bias'][r])
                  for r in range(100)]
        expected = [sum(hidden[c] ** 2 * float(weights['linear2.weight'][r, c])
                        for c in range(100)) + float(weights['linear2.bias'][r])
                    for r in range(10)]
        result = self.worker.matrix_reference(image, weights)
        self.assertEqual(result.shape, (10,))
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)

    def test_original_packed_rotations_match_matrix_at_wrap_positions(self):
        np = self.np
        image = ((np.arange(784, dtype=np.float64) * 19 + 7) % 256) / 255
        weights = self.weights()
        packed = self.worker.packed_reference(OFFICIAL_SOURCE, image, weights)
        matrix = self.worker.matrix_reference(image, weights)
        self.assertEqual(packed.shape, (10,))
        np.testing.assert_allclose(packed, matrix, atol=1e-12, rtol=1e-12)
        # The fixture selects columns at both ends and hidden rows near 99;
        # reversing rotate(k) changes these selected matrix entries.
        reversed_input = np.roll(image, 1)
        self.assertGreater(float(np.max(np.abs(matrix - self.worker.matrix_reference(
            reversed_input, weights)))), 1e-8)

    def test_official_checkpoint_has_original_finite_values_and_safe_load(self):
        np, torch = self.np, self.torch
        original = torch.load(OFFICIAL_MODEL, map_location='cpu', weights_only=True)
        with patch.object(torch, 'load', wraps=torch.load) as load:
            weights = self.worker.load_checkpoint(OFFICIAL_MODEL)
        self.assertIs(load.call_args.kwargs.get('weights_only'), True)
        shapes = {'linear1.weight': (100, 784), 'linear1.bias': (100,),
                  'linear2.weight': (10, 100), 'linear2.bias': (10,)}
        self.assertEqual(len(original), 14)
        self.assertEqual(set(original), set(shapes) | set(self.expected_bn_metadata()))
        self.assertEqual(set(weights), set(shapes))
        for name, shape in shapes.items():
            with self.subTest(name=name):
                self.assertEqual(weights[name].shape, shape)
                self.assertEqual(weights[name].dtype, np.float64)
                self.assertTrue(np.isfinite(weights[name]).all())
                np.testing.assert_array_equal(weights[name], original[name].cpu().numpy())

    def test_checkpoint_rejects_wrong_shape_missing_key_and_nonfinite(self):
        torch = self.torch
        for case in ('shape', 'missing', 'nonfinite', 'unknown', 'unknown_same_count', 'missing_bn'):
            with self.subTest(case=case):
                state = self.checkpoint_state()
                if case == 'shape':
                    state['linear1.weight'] = state['linear1.weight'][:, :-1]
                elif case == 'missing':
                    del state['linear2.bias']
                elif case == 'nonfinite':
                    state['linear2.bias'][0] = float('nan')
                elif case == 'unknown':
                    state['bn3.weight'] = torch.ones(10, dtype=torch.float32)
                elif case == 'unknown_same_count':
                    state['bn3.running_mean'] = state.pop('bn1.running_mean')
                else:
                    del state['bn1.running_mean']
                checkpoint = self.directory / (case + '.model')
                torch.save(state, checkpoint)
                with self.assertRaises(ValueError):
                    self.worker.load_checkpoint(checkpoint)

    def test_known_fourteen_checkpoint_entries_preserve_values_and_unused_metadata(self):
        np, torch = self.np, self.torch
        state = self.checkpoint_state()
        checkpoint = self.directory / 'known-fourteen.model'
        torch.save(state, checkpoint)
        with patch.object(torch, 'load', wraps=torch.load) as load:
            weights, unused = self.worker._load_checkpoint(checkpoint)
        self.assertEqual(load.call_count, 1)
        self.assertIs(load.call_args.kwargs.get('weights_only'), True)
        self.assertEqual(len(state), 14)
        self.assertEqual(set(weights), set(self.weights()))
        self.assertEqual(len(unused), 10)
        self.assertEqual({entry['name']: entry for entry in unused}, self.expected_bn_metadata())
        for name, value in weights.items():
            with self.subTest(name=name):
                self.assertEqual(value.dtype, np.float64)
                np.testing.assert_array_equal(value, state[name].cpu().numpy())

    def test_checkpoint_rejects_bn_shape_dtype_and_nonfinite(self):
        torch = self.torch
        changes = [('bn1_shape', 'bn1.weight', torch.ones(99, dtype=torch.float32)),
                   ('bn2_shape', 'bn2.running_mean', torch.zeros(9, dtype=torch.float32)),
                   ('bn_float_dtype', 'bn1.bias', torch.zeros(100, dtype=torch.float64)),
                   ('count_dtype', 'bn2.num_batches_tracked', torch.tensor(3., dtype=torch.float32)),
                   ('count_shape', 'bn1.num_batches_tracked', torch.tensor([3], dtype=torch.int64)),
                   ('bn_nan', 'bn1.running_var', torch.full((100,), float('nan'), dtype=torch.float32)),
                   ('bn_inf', 'bn2.weight', torch.full((10,), float('inf'), dtype=torch.float32))]
        for label, name, replacement in changes:
            with self.subTest(label=label):
                state = self.checkpoint_state()
                state[name] = replacement
                checkpoint = self.directory / (label + '.model')
                torch.save(state, checkpoint)
                with self.assertRaises(ValueError):
                    self.worker.load_checkpoint(checkpoint)

    def test_bn_values_are_unused_by_original_linear_square_linear_arithmetic(self):
        np, torch = self.np, self.torch
        first, changed = self.checkpoint_state(), self.checkpoint_state()
        for name in self.expected_bn_metadata():
            changed[name] = changed[name] + (17 if changed[name].dtype == torch.int64 else 32.)
        first_path, changed_path = self.directory / 'first.model', self.directory / 'changed-bn.model'
        torch.save(first, first_path)
        torch.save(changed, changed_path)
        first_weights, first_metadata = self.worker._load_checkpoint(first_path)
        changed_weights, changed_metadata = self.worker._load_checkpoint(changed_path)
        self.assertEqual(first_metadata, changed_metadata)
        self.assertEqual(set(first_weights), set(self.weights()))
        for name in first_weights:
            np.testing.assert_array_equal(first_weights[name], changed_weights[name])
        image = (np.arange(784, dtype=np.float64) % 256) / 255
        reference = self.worker.matrix_reference(image, first_weights)
        np.testing.assert_array_equal(self.worker.matrix_reference(image, changed_weights), reference)
        np.testing.assert_allclose(self.worker.packed_reference(OFFICIAL_SOURCE, image, changed_weights),
                                   reference, atol=1e-12, rtol=1e-12)

    def test_sample_record_preserves_three_indices_and_metadata(self):
        record = samples_record()
        self.assertEqual(self.worker.validate_samples_record(record), record)
        self.assertEqual([sample['index'] for sample in record['samples']], [0, 1, 2])

    def test_sample_record_rejects_dimensions_indices_and_nonfinite(self):
        changes = [('shape', lambda r: r['samples'][0]['input784'].pop()),
                   ('reference', lambda r: r['samples'][1]['reference10'].pop()),
                   ('index', lambda r: r['samples'][2].update(index=3)),
                   ('count', lambda r: r['samples'].pop()),
                   ('nan_input', lambda r: r['samples'][0]['input784'].__setitem__(0, float('nan'))),
                   ('inf_reference', lambda r: r['samples'][1]['reference10'].__setitem__(0, float('inf'))),
                   ('range', lambda r: r['samples'][2]['input784'].__setitem__(0, 1.01)),
                   ('split', lambda r: r['source'].update(split='test')),
                   ('normalization', lambda r: r['source'].update(normalization='none'))]
        for label, mutate in changes:
            with self.subTest(label=label):
                record = samples_record()
                mutate(record)
                with self.assertRaises(ValueError):
                    self.worker.validate_samples_record(record)

    def test_plan_json_gate_accepts_only_observed_zero_boot_contract(self):
        result = self.worker.check_plan_contract(plan_record())
        self.assertIs(result['ok'], True)
        self.assertEqual(result['boots'], 0)
        self.assertEqual(result['context_id'], CONTEXT_ID)
        self.assertEqual(result['external_inputs'], 1)
        self.assertEqual(result['final_outputs'], 1)

    def test_plan_json_gate_rejects_bundle_boot_place_io_and_payload(self):
        changes = [('format', lambda r: r.update(format_version=2)),
                   ('bundle', lambda r: r.update(plaintext_bundle={})),
                   ('boot', lambda r: r['execution'][0].update(op='boot')),
                   ('place', lambda r: r['values'][0]['place'].update(rank=1)),
                   ('input', lambda r: r.update(external_inputs=[])),
                   ('output', lambda r: r.update(final_outputs=[])),
                   ('payload', lambda r: r['initialization'][0]['payload'].update(values=[])),
                   ('nonfinite', lambda r: r['initialization'][0]['payload'].update(values=[float('inf')]))]
        for label, mutate in changes:
            with self.subTest(label=label):
                record = plan_record()
                mutate(record)
                with self.assertRaises(ValueError):
                    self.worker.check_plan_contract(record)

    def observation_fixture(self):
        key = dict(op='add_cc', mode='compute', input_levels=[12, 12],
                   input_scale_log2=[45, 45], output_level=12,
                   output_scale_log2=45, rotation_step=None)
        missing = dict(op='rotate', mode='compute', input_levels=[12],
                       input_scale_log2=[45], output_level=12,
                       output_scale_log2=45, rotation_step=7)
        calls = [dict(sample_index=0, call_index=0, key=deepcopy(key), elapsed_ns=100, status='ok'),
                 dict(sample_index=0, call_index=1, key=deepcopy(key), elapsed_ns=300, status='ok'),
                 dict(sample_index=1, call_index=0, key=deepcopy(key), elapsed_ns=250, status='ok'),
                 dict(sample_index=1, call_index=1, key=deepcopy(key), elapsed_ns=250, status='ok'),
                 dict(sample_index=2, call_index=0, key=deepcopy(missing), elapsed_ns=80, status='ok')]
        profile = dict(format='pt-agent-r12-per-op-profile-v1', time_unit='ns',
                       context_id=CONTEXT_ID, calibration_sample_index=0, frozen=True,
                       entries=[dict(key=deepcopy(key), call_count=2, total_ns=400, mean_ns=200.0)],
                       bootstrap_latency_ns=None)
        record = dict(status='ok', encrypted_execution_attempts=3,
                      encrypted_execution_completed=3, ckks_tolerance_passed=True,
                      samples=[dict(index=i, encrypted_execution_attempted=True,
                                    encrypted_execution_completed=True, passed=True,
                                    runtime_run_ns=elapsed)
                               for i, elapsed in enumerate((500, 600, 110))],
                      predictions=[dict(sample_index=1, prediction_complete=True, missing_keys=[],
                                        predicted_compute_ns=400.0, measured_compute_ns=500,
                                        relative_abs_error=.2, runtime_run_ns=600),
                                   dict(sample_index=2, prediction_complete=False,
                                        missing_keys=[deepcopy(missing)], predicted_compute_ns=None,
                                        measured_compute_ns=80, relative_abs_error=None,
                                        runtime_run_ns=110)])
        return record, calls, profile

    def test_observation_verifies_frozen_means_and_preserves_inputs(self):
        record, calls, profile = self.observation_fixture()
        original = deepcopy((record, calls, profile))
        checked = self.worker.validate_observation(record, calls, profile)
        self.assertEqual(checked, dict(ok=True, profile_checked=True, predictions_checked=2))
        self.assertEqual((record, calls, profile), original)

    def test_canonical_rotate_name_preserves_key_and_other_operations_reject_steps(self):
        key = dict(op='Rotate', mode='compute', input_levels=[12], input_scale_log2=[45],
                   output_level=12, output_scale_log2=45, rotation_step=7)
        original = deepcopy(key)
        validated = json.loads(self.worker._key_string(key))
        self.assertEqual(validated['op'], 'Rotate')
        self.assertEqual(validated['rotation_step'], 7)
        self.assertEqual(key, original)
        non_rotation = dict(key, op='MulCP')
        with self.assertRaises(ValueError):
            self.worker._key_string(non_rotation)

    def test_observation_rejects_refit_table_and_different_full_keys(self):
        changes = [('refit', lambda r, c, p: p['entries'][0].update(
                        call_count=4, total_ns=900, mean_ns=225.0)),
                   ('mean', lambda r, c, p: p['entries'][0].update(mean_ns=201.0)),
                   ('frozen', lambda r, c, p: p.update(frozen=False)),
                   ('reuse_mode', lambda r, c, p: p['entries'][0]['key'].update(mode='compute_reuse')),
                   ('input_levels', lambda r, c, p: p['entries'][0]['key'].update(input_levels=[11, 12])),
                   ('bootstrap', lambda r, c, p: p.update(bootstrap_latency_ns=0))]
        for label, mutate in changes:
            with self.subTest(label=label):
                record, calls, profile = self.observation_fixture()
                mutate(record, calls, profile)
                with self.assertRaises(ValueError):
                    self.worker.validate_observation(record, calls, profile)

    def test_observation_missing_operation_keeps_null_instead_of_zero(self):
        for label, update in (('zero_prediction', dict(predicted_compute_ns=0)),
                              ('zero_error', dict(relative_abs_error=0)),
                              ('complete', dict(prediction_complete=True)),
                              ('omitted_key', dict(missing_keys=[])),
                              ('measured', dict(measured_compute_ns=0)),
                              ('runtime', dict(runtime_run_ns=0))):
            with self.subTest(label=label):
                record, calls, profile = self.observation_fixture()
                record['predictions'][1].update(update)
                with self.assertRaises(ValueError):
                    self.worker.validate_observation(record, calls, profile)

    def test_observation_zero_measured_time_keeps_relative_error_null(self):
        record, calls, profile = self.observation_fixture()
        for call in calls:
            if call['sample_index'] == 1:
                call['elapsed_ns'] = 0
        record['predictions'][0].update(measured_compute_ns=0, relative_abs_error=None)
        self.assertIs(self.worker.validate_observation(record, calls, profile)['ok'], True)
        record['predictions'][0]['relative_abs_error'] = 0
        with self.assertRaises(ValueError):
            self.worker.validate_observation(record, calls, profile)

    def test_failed_calibration_keeps_profile_absent_and_later_samples_unattempted(self):
        record, calls, profile = self.observation_fixture()
        record.update(status='failed', encrypted_execution_attempts=1,
                      encrypted_execution_completed=0, ckks_tolerance_passed=False)
        record['samples'][0].update(encrypted_execution_completed=False, passed=False)
        for sample in record['samples'][1:]:
            sample.update(encrypted_execution_attempted=False, encrypted_execution_completed=False,
                          passed=None, runtime_run_ns=None)
        record['predictions'] = []
        calls = calls[:1]
        calls[0]['status'] = 'error'
        checked = self.worker.validate_observation(record, calls, None)
        self.assertEqual(checked, dict(ok=True, profile_checked=False, predictions_checked=0))
        with self.assertRaises(ValueError):
            self.worker.validate_observation(record, calls, profile)
        record['samples'][1]['encrypted_execution_attempted'] = True
        with self.assertRaises(ValueError):
            self.worker.validate_observation(record, calls, None)


    def test_actual_commands_select_one_algorithm_and_preserve_common_parameters(self):
        from pt_agent import toolchain
        from pt_agent.experiments import r12_mlp_worker as worker
        from pt_agent.experiments.r10_paper_metrics import StageRecorder

        with TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / 'input.mlir'
            source.write_text('module {}\n')
            host = toolchain.host_paths()
            args = SimpleNamespace(work_dir=work, input=source,
                                   hecate_opt=Path(host['hecate_opt']),
                                   compiler_profile=Path(host['profile']),
                                   operator_spec=Path(host['spec']))
            fixture = plan_record()
            # Only a fixture protocol string is needed; no file-content digest
            # computation or integrity verification is performed by this test.
            protocol_string = fixture['target']['operator_spec']['source_sha256'].split(':', 1)[1]
            commands = []

            def trusted_fixture(run_dir, argv, log, *, env_extra, wall_seconds):
                self.assertEqual(run_dir, work)
                self.assertEqual(wall_seconds, 7200)
                self.assertEqual(env_extra, {'LD_LIBRARY_PATH': str(work / 'libraries')})
                commands.append(list(argv))
                (work / 'lowered._hecate_MLP.runtime-plan.json').write_text(json.dumps(fixture))
                (work / 'lowered.mlir').write_text('module {}\n')
                return dict(exit=0, timeout=False, seconds=0., peak_rss_mib=None, log_tail='fixture')

            with patch.object(toolchain, '_file_sha256', return_value=protocol_string), \
                 patch.object(toolchain, 'trusted_run', side_effect=trusted_fixture) as run, \
                 patch.dict(os.environ, {'HECATE_PYTHON_LIBRARY_PATH': str(work / 'libraries')}):
                for mode in ('dacapo-optimized', 'eva'):
                    if mode == 'eva':
                        args.compile_mode = mode
                    result = dict(artifacts={})
                    events = work / (mode + '-events.jsonl')
                    worker._compile(args, StageRecorder(events), result)
                    self.assertEqual(result['compile_mode'], mode)
                    self.assertEqual(result['spec_revision'], 'R12b')
                    self.assertEqual(result['compiled_bootstraps'], 0)
                    end_stages = [json.loads(line)['stage_id'] for line in events.read_text().splitlines()
                                  if json.loads(line)['type'] == 'end']
                    self.assertIn('compile.official_eva' if mode == 'eva' else 'compile.official_dacapo', end_stages)
                self.assertEqual(run.call_count, 2)
            default, eva = commands
            self.assertEqual(default.count('--dacapo'), 1)
            self.assertNotIn('--eva', default)
            self.assertEqual(eva.count('--eva'), 1)
            self.assertNotIn('--dacapo', eva)
            dacapo_parameters = {'--dacapo', '--boot-placement=optimized', '--output-val=10',
                                 '--threshold=0.5', '--dump-pass-pipeline'}
            self.assertTrue(dacapo_parameters <= set(default))
            self.assertFalse(any(arg.startswith(('--boot-placement=', '--output-val=', '--threshold='))
                                 for arg in eva))
            self.assertEqual([arg for arg in default if arg not in dacapo_parameters],
                             [arg for arg in eva if arg != '--eva'])
            common = ['--runtime-plan-format=json', '--runtime-plan-bundle-format=files',
                      '--ckks-config=' + str(args.compiler_profile.resolve()), '--waterline=45',
                      '--mlir-disable-threading', '--verify-each',
                      '--runtime-plan-context-id=poseidon-agent-n15-q13-p1-tc128-v1',
                      '--runtime-plan-device-count=0', '--runtime-plan-ntt=true',
                      '--runtime-plan-boot-implementation=native',
                      '--runtime-plan-inline-payload-max-bytes=131072']
            for command in commands:
                self.assertEqual(command[:2], [str(args.hecate_opt.resolve()), str(source.resolve())])
                self.assertEqual(command[-2:], ['-o', str(work / 'lowered.mlir')])
                self.assertFalse(any(arg.startswith('--boot-profile') for arg in command))
                for argument in common:
                    with self.subTest(argument=argument):
                        self.assertEqual(command.count(argument), 1)

    def test_unknown_mode_is_rejected_before_subprocess(self):
        from pt_agent import toolchain
        from pt_agent.experiments import r12_mlp_preflight as outer
        from pt_agent.experiments import r12_mlp_worker as worker

        with patch.object(toolchain, 'trusted_run') as run, \
             patch.object(toolchain, 'host_paths') as host:
            with self.assertRaises(worker.WorkerFailure) as caught:
                worker._compile(SimpleNamespace(compile_mode='unknown'), None, {'artifacts': {}})
            self.assertEqual(caught.exception.status, 'unsupported_contract')
            host.assert_not_called()
            run.assert_not_called()
        with patch.object(outer, 'run_process') as run:
            with self.assertRaises(ValueError):
                outer.execute(SimpleNamespace(compile_mode='unknown'))
            run.assert_not_called()
        with patch.object(sys, 'argv', ['r12', '--model', str(OFFICIAL_MODEL), '--mnist', '/fixture/images.gz',
                                       '--out', '/fixture/run', '--compile-mode', 'unknown']), \
             patch.object(outer, 'execute') as execute, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                outer.main()
            self.assertEqual(caught.exception.code, 2)
            execute.assert_not_called()
        argv = ['--phase', 'compile', '--work-dir', '/fixture/work', '--input', '/fixture/input.mlir',
                '--hecate-opt', '/fixture/hecate-opt', '--compiler-profile', '/fixture/profile.json',
                '--operator-spec', '/fixture/spec.json', '--compile-mode', 'unknown']
        with patch.object(worker, '_compile') as compile_worker, \
             patch.object(toolchain, 'trusted_run') as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                worker.main(argv)
            self.assertEqual(caught.exception.code, 2)
            compile_worker.assert_not_called()
            run.assert_not_called()

    def test_worker_cli_uses_string_default_and_explicit_eva(self):
        from pt_agent.experiments import r12_mlp_worker as worker

        for flags, expected in (([], 'dacapo-optimized'), (['--compile-mode', 'eva'], 'eva')):
            with self.subTest(expected=expected), TemporaryDirectory() as directory:
                work = Path(directory)
                argv = ['--phase', 'compile', '--work-dir', str(work), '--input', str(work / 'input.mlir'),
                        '--hecate-opt', str(work / 'hecate-opt'), '--compiler-profile', str(work / 'profile.json'),
                        '--operator-spec', str(work / 'spec.json'), *flags]
                with patch.object(worker, '_compile') as compile_worker, \
                     patch.object(worker.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)):
                    self.assertEqual(worker.main(argv), 0)
                args = compile_worker.call_args.args[0]
                self.assertIsInstance(args.compile_mode, str)
                self.assertEqual(args.compile_mode, expected)

    def build_cache_fixture(self, label):
        root = self.directory / label
        work = root / 'work'
        generated = root / 'configuration/generated'
        include_c, include_cxx = root / 'gmp C/include', root / 'gmp Cxx/include'
        lib_c, lib_cxx = root / 'lib/libgmp.so', root / 'lib/libgmpxx.so'
        native = root / 'lib/libposeidon_shared.so'
        for directory in (work, generated, include_c, include_cxx, lib_c.parent,
                          root / 'repo/third_party/ckks-runtime', root / 'client-source'):
            directory.mkdir(parents=True, exist_ok=True)
        for file in (lib_c, lib_cxx, native):
            file.write_text('R12 path fixture only\n')
        values = dict(GMP_INCLUDE_DIR=str(include_c.resolve()),
                      GMPXX_INCLUDE_DIR=str(include_cxx.resolve()),
                      GMP_LIBRARY=str(lib_c.resolve()), GMPXX_LIBRARY=str(lib_cxx.resolve()))
        cache = generated.parent / 'CMakeCache.txt'
        cache.write_text(''.join(name + (':PATH=' if 'INCLUDE' in name else ':FILEPATH=')
                                 + value + '\n' for name, value in values.items()))
        args = SimpleNamespace(work_dir=work, build_dir=work / 'native',
                               repo_root=root / 'repo', runtime_root=root / 'repo/third_party/ckks-runtime',
                               client_source_dir=root / 'client-source', generated_include_dir=generated,
                               native_library=native)
        return args, cache, values

    def test_build_forwards_four_existing_gmp_cache_values_and_records_source(self):
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        args, cache, values = self.build_cache_fixture('gmp-present')
        result = dict(artifacts={})
        commands = []

        def child_fixture(actual_args, command, label, timeout):
            self.assertIs(actual_args, args)
            self.assertEqual(timeout, 1800)
            commands.append((label, list(command)))
            if label == 'configure':
                args.build_dir.mkdir()
            else:
                self.assertEqual(label, 'build')
                # This placeholder checks worker path handling only. It is
                # never invoked as a native client or used by ClientGateCases.
                client = args.build_dir / 'mlp_client'
                client.write_text('R12 build-output fixture only\n')
                client.chmod(0o700)
            return dict(exit_code=0, timed_out=False, signal=None, remaining_pids=[])

        with patch.object(self.worker, '_run_child', side_effect=child_fixture) as child:
            self.worker._build(args, StageRecorder(args.work_dir / 'events.jsonl'), result)
        self.assertEqual(child.call_count, 2)
        self.assertEqual([label for label, _ in commands], ['configure', 'build'])
        configure = commands[0][1]
        for name, value in values.items():
            with self.subTest(name=name):
                self.assertEqual(configure.count('-D' + name + '=' + value), 1)
        self.assertEqual(result['gmp_cache_source'], str(cache.resolve()))
        self.assertEqual(result['gmp_cache_values'], values)
        self.assertEqual(commands[1][1], ['cmake', '--build', str(args.build_dir.resolve()),
                                          '--parallel', '1', '--target', 'mlp_client'])
        self.assertEqual(result['artifacts']['client'], str(args.build_dir.resolve() / 'mlp_client'))

    def test_build_missing_gmp_cache_value_stops_before_configure(self):
        from pt_agent.experiments.r10_paper_metrics import StageRecorder
        for missing in ('GMP_INCLUDE_DIR', 'GMPXX_INCLUDE_DIR', 'GMP_LIBRARY', 'GMPXX_LIBRARY'):
            with self.subTest(missing=missing):
                args, cache, values = self.build_cache_fixture('missing-' + missing)
                cache.write_text(''.join(name + (':PATH=' if 'INCLUDE' in name else ':FILEPATH=')
                                         + value + '\n' for name, value in values.items()
                                         if name != missing))
                result = dict(artifacts={})
                with patch.object(self.worker, '_run_child') as child:
                    with self.assertRaises(self.worker.WorkerFailure) as caught:
                        self.worker._build(args, StageRecorder(args.work_dir / 'events.jsonl'),
                                           result)
                    self.assertEqual(caught.exception.status, 'dependency_missing')
                    child.assert_not_called()
                self.assertEqual(result['gmp_cache_source'], str(cache.resolve()))
                self.assertIsNone(result['gmp_cache_values'])


_GATE_VARIABLES = ('PT_R12_CLIENT', 'PT_R12_PLAN', 'PT_R12_SPEC',
                   'PT_R12_INPUTS', 'PT_R12_TEST_TMP')
_GATE_ARTIFACTS = [Path(os.environ[name]) if os.environ.get(name)
                   else Path(__file__).parent / ('missing-' + name)
                   for name in _GATE_VARIABLES]


@require(*_GATE_ARTIFACTS)
class ClientGateCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client, cls.plan_path, cls.spec_path, cls.inputs_path, cls.tmp_path = (
            Path(os.environ[name]).resolve() for name in _GATE_VARIABLES)
        cls.plan = json.loads(cls.plan_path.read_text())
        cls.inputs = json.loads(cls.inputs_path.read_text())

    def setUp(self):
        temporary = TemporaryDirectory(prefix='r12-client-gate-', dir=self.tmp_path)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def check(self, *, plan=None, inputs=None, expected=2, raw_plan=None,
              missing_plan=False):
        plan_path, inputs_path = self.plan_path, self.inputs_path
        if plan is not None or raw_plan is not None:
            plan_path = self.directory / 'plan.json'
            plan_path.write_text(raw_plan if raw_plan is not None else json.dumps(plan))
        if missing_plan:
            plan_path = self.directory / 'absent-plan.json'
        if inputs is not None:
            inputs_path = self.directory / 'samples.json'
            inputs_path.write_text(json.dumps(inputs))
        output = self.directory / 'check-result.json'
        command = [str(self.client), '--plan', str(plan_path), '--spec', str(self.spec_path),
                   '--inputs', str(inputs_path), '--output', str(output), '--check-only']
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=60)
        self.assertTrue(output.is_file(), completed.stdout)
        result = json.loads(output.read_text())
        self.assertEqual(completed.returncode, expected, (result, completed.stdout))
        self.assertTrue({'status', 'failed_phase', 'error', 'check_only', 'context_started',
                         'keygen_started', 'encrypted_execution_attempts'} <= set(result))
        self.assertIs(result['check_only'], True)
        self.assertIs(result['context_started'], False)
        self.assertIs(result['keygen_started'], False)
        self.assertEqual(result['encrypted_execution_attempts'], 0)
        self.assertEqual(result['status'], {0: 'ok', 1: 'error', 2: 'unsupported_contract'}[expected])
        if expected:
            self.assertTrue(result['failed_phase'])
            self.assertTrue(result['error'])
        return result

    def test_valid_actual_plan_spec_and_samples(self):
        self.check(expected=0)

    def test_bundle_is_rejected_before_context(self):
        plan = deepcopy(self.plan)
        plan['plaintext_bundle'] = dict(id='r12-gate-bundle', version=1,
            manifest_sha256=plan['target']['operator_spec']['source_sha256'])
        self.check(plan=plan)

    def test_boot_implementations_are_rejected_before_keys(self):
        for implementation in ('native', 'decrypt_reencrypt'):
            with self.subTest(implementation=implementation):
                plan = deepcopy(self.plan)
                operation = next(op for op in plan['execution'] if op['kind'] == 'compute')
                output = next(v for v in plan['values'] if v['id'] == operation['output'])
                operation.update(op='boot', inputs=operation['inputs'][:1],
                                 attrs=dict(target_level=output['level'],
                                            target_scale_log2=output['scale_log2'],
                                            target_components=output['components'],
                                            operator_profile='r12-gate-unsupported',
                                            implementation=implementation))
                self.check(plan=plan)

    def test_nonzero_host_rank_is_rejected(self):
        plan = deepcopy(self.plan)
        plan['values'][0]['place']['rank'] = 1
        self.check(plan=plan)

    def test_missing_external_input_is_rejected(self):
        plan = deepcopy(self.plan)
        plan['external_inputs'] = []
        self.check(plan=plan)

    def test_missing_final_output_is_rejected(self):
        plan = deepcopy(self.plan)
        plan['final_outputs'] = []
        self.check(plan=plan)

    def test_context_mismatch_is_rejected(self):
        plan = deepcopy(self.plan)
        for value in plan['values']:
            value['context'] = 'r12-gate-unsupported-context'
        self.check(plan=plan)

    def test_level_outside_original_chain_is_rejected(self):
        plan = deepcopy(self.plan)
        plan['values'][0]['level'] = 13
        self.check(plan=plan)

    def test_nonfinite_input_is_a_parse_error_before_context(self):
        inputs = deepcopy(self.inputs)
        inputs['samples'][0]['input784'][0] = float('nan')
        self.check(inputs=inputs, expected=1)

    def test_finite_input_outside_unit_interval_is_rejected(self):
        inputs = deepcopy(self.inputs)
        inputs['samples'][0]['input784'][0] = 1.1
        self.check(inputs=inputs)

    def test_empty_inline_payload_is_rejected_by_original_reader(self):
        plan = deepcopy(self.plan)
        operation = next(op for op in plan['initialization'] if op['kind'] == 'encode')
        operation['payload']['values'] = []
        self.check(plan=plan, expected=1)

    def test_malformed_plan_writes_parse_error(self):
        self.check(raw_plan='{invalid-json', expected=1)

    def test_missing_plan_writes_io_error(self):
        self.check(missing_plan=True, expected=1)


if __name__ == '__main__':
    unittest.main()
