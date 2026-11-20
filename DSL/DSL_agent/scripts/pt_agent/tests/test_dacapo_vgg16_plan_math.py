"""Synthetic VGG physical-unit and provenance checks; all model jobs mocked."""
import copy
from io import StringIO
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from pt_agent.experiments import dacapo_vgg16_plan_math as entry


def clean():
    return dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])


def valid_worker():
    result = entry.replay.initial_result()
    result.update(round=entry.ROUND, model=entry.MODEL, source_factory=entry.FACTORY, status='ok',
        official_FHE_arithmetic_math_passed=True, original_model_modified=False, task_accuracy_evaluated=False,
        task_accuracy_acceptance=None, model_forwards=6, torch_model_builds=1, torch_model_forwards=6,
        checkpoint_reads=1, packed_forward_completed=3, math_check_completed=3,
        activation_completed=45, op_completed=128580, checkpoint_load_completed=1, physical_SiLU_modules=14, elapsed_ns=1,
        unused_convolution_bias_audit=[dict(used_by_official_FHE=False) for _ in range(13)],
        activation_math=[dict(passed=True) for _ in range(45)],
        probe_rows=[dict(sample_id='train:' + str(i), math=dict(passed=True)) for i in range(3)])
    return result


class VGGOfficialArithmeticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = entry.replay._np()

    def test_private_bias_masking_records_nonzero_values_and_preserves_original_parameters(self):
        np = self.np
        class Model:
            def __init__(self):
                self.layers = {'conv_1_1':SimpleNamespace(Conv2d=SimpleNamespace(
                    bias=np.array([.25, -.5], dtype=np.float64), out_channels=2))}
            def get_submodule(self, name):
                return self.layers[name]
        original = Model()
        private = copy.deepcopy(original)
        parameter = original.get_submodule('conv_1_1').Conv2d.bias
        rows = entry.audit_and_mask_bias(original, private, lambda x:x, conv_names=('conv_1_1',))
        self.assertIs(original.get_submodule('conv_1_1').Conv2d.bias, parameter)
        self.assertIsNone(private.get_submodule('conv_1_1').Conv2d.bias)
        self.assertEqual(rows[0]['shape'], [2]);self.assertEqual(rows[0]['count'], 2)
        self.assertEqual(rows[0]['nonzero_count'], 2);self.assertEqual(rows[0]['maximum_absolute'], .5)
        self.assertFalse(rows[0]['used_by_official_FHE'])
        self.assertEqual(len(entry.CONVS), 13)

    def test_physical32_polynomial_matches_independent_closed_formula(self):
        np = self.np
        coefficients = np.zeros(96, dtype=np.float64)
        coefficients[1] = .25
        x = np.array([-64., -16., 0., 32., 64.], dtype=np.float64)
        actual, checked = entry.physical_activation(x, lambda u:u * (.5 + .25 * u), coefficients)
        np.testing.assert_array_equal(actual, .5 * x + .25 * x * x / 32.)
        self.assertTrue(checked['passed'])
        self.assertEqual(checked['criterion'], entry.polynomial.CRITERION)

    def test_nonfinite_and_incorrect_original_polynomial_stop_without_relaxing_math(self):
        np = self.np
        coefficients = np.zeros(96, dtype=np.float64)
        for x in (np.array([np.inf], dtype=np.float64), np.array([1.], dtype=np.float32)):
            with self.assertRaises(ValueError):entry.physical_activation(x, lambda u:.5 * u, coefficients)
        with self.assertRaises(ValueError):
            entry.physical_activation(np.array([1.], dtype=np.float64), lambda u:.5 * u + 1e-6, coefficients)

    def test_source_facts_require_omitted_conv_bias_and_explicit_affine_scale32(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            func = root / 'python/poly/poly/Func.py'
            func.parent.mkdir(parents=True)
            func.write_text('def HE_ConvBN(close, mpp, conv, bn):\n return close["MPCB"](mpp, conv.weight, *MPCB.abstractBN(bn))\n'
                'def HE_SiLU(x):\n calculation = Poly.GenPoly()\n return x * (calculation(x) + 0.5)\n')
            script = root / 'examples/benchmarks/VGG16.py'
            script.parent.mkdir(parents=True)
            script.write_text('def VGG16(x):\n a=HE_Linear(None,x,None,scale=32.)\n b=HE_Linear(None,a,None,scale=32.)\n'
                ' c=HE_Linear(None,b,None,scale=32.)\n return HE_MPBN(c,None,scale=32.)\n')
            runtime = root / 'lib/Runtime/HEAAN_HEVM.cpp'
            runtime.parent.mkdir(parents=True)
            runtime.write_text('src[i % src.size()]\nsrc[i % src.size()]\n')
            facts = entry.source_facts(root)
            self.assertFalse(facts['convolution_bias_used_by_official_FHE'])
            self.assertIsNone(facts['task_accuracy_acceptance'])
            script.write_text(script.read_text().replace('scale=32.', 'scale=1.', 1))
            with self.assertRaises(ValueError):entry.source_facts(root)

    def test_program_uses_private_vgg_source_gate_and_original_lazy_replay(self):
        np = self.np
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files = entry.binding.write_public_fixture(root / 'fixture', bootstrap=True)
            source = dict(plan=files['plan'], bundle=files['bundle'],
                facts=dict(scalar_periodic_runtime_source='/synthetic/original/HEAAN_HEVM.cpp'))
            old_factory = entry.binding.SOURCE_FACTORY
            program, proof = entry.load_program(source)
            self.assertEqual(entry.binding.SOURCE_FACTORY, old_factory)
            self.assertEqual(proof['source_factory'], entry.FACTORY)
            self.assertTrue(program['metadata']['source_gates']['per_site_provenance_verified'])
            self.assertFalse(program['metadata']['source_gates']['original_training_math_verified'])
            raw = np.tile(np.array([.05 + .1 * i for i in range(8)], dtype=np.float64), 8192)
            constants = entry.replay.LazyPlaintexts(files['bundle'], program)
            result = entry.replay.initial_result()
            actual = entry.replay.execute_program(program, constants, raw, result)
            np.testing.assert_array_equal(actual, -(np.roll(raw, -1)**2 + .125 + raw))
            self.assertEqual(result['ideal_identity_nodes']['boot'], 1)
            self.assertLessEqual(constants.summary()['peak_decoded_bytes'], 65536 * 8)

    def test_original_pack_and_decoder_keep_channel_order_and_scale32(self):
        np = self.np
        image = np.arange(3072, dtype=np.float64).reshape(3, 32, 32)
        packed = entry.replay.reference.pack_input(image)
        np.testing.assert_array_equal(packed[:3072], image.reshape(-1, order='C') / 32.)
        np.testing.assert_array_equal(packed[3072:4096], np.zeros(1024))
        np.testing.assert_array_equal(packed[4096:8192], packed[:4096])
        decoded = entry.replay.reference.decode_output(packed)
        np.testing.assert_array_equal(decoded, np.arange(10, dtype=np.float64).reshape(1, 10))

    def test_final_one_by_one_channel_layout_keeps_first512_channels_in_order(self):
        np = self.np
        channels = np.arange(512, dtype=np.float64).reshape(1, 512, 1, 1)
        padded = np.pad(channels[0], ((0, 512), (0, 0), (0, 0)))
        # Original OutPack: (to ko1 ko2) h w -> (to h ko1 w ko2).
        block = padded.reshape(1, 32, 32, 1, 1).transpose(0, 3, 1, 4, 2).reshape(-1)
        packed = np.tile(block, 64)
        self.assertEqual(packed.shape, (65536,))
        np.testing.assert_array_equal(packed[:512], channels.reshape(-1))
        np.testing.assert_array_equal(packed[512:1024], np.zeros(512))

    def test_fused_affine_physical32_omits_conv_bias_with_separate_original_diagnostic(self):
        np = self.np
        x = np.array([2., -3.], dtype=np.float64)
        weight = np.array([[.5, -.25], [1., .75]], dtype=np.float64)
        conv_bias = np.array([.25, -.5], dtype=np.float64)
        gamma = np.array([1.5, .75], dtype=np.float64)
        var = np.array([2., .5], dtype=np.float64)
        mean = np.array([.125, -.25], dtype=np.float64)
        beta = np.array([-.5, .25], dtype=np.float64)
        G = gamma / np.sqrt(var + 1e-5)
        H = beta - G * mean
        physical = 32 * (G * (weight @ (x / 32)) + H / 32)
        private_reference = gamma * ((weight @ x) - mean) / np.sqrt(var + 1e-5) + beta
        original_Torch_formula = gamma * ((weight @ x + conv_bias) - mean) / np.sqrt(var + 1e-5) + beta
        self.assertTrue(entry.polynomial._check_math(private_reference, physical)['passed'])
        np.testing.assert_allclose(original_Torch_formula - private_reference, G * conv_bias, rtol=0, atol=1e-15)

    def test_terminal_protocol_requires_all_operations_math_and_zero_crypto(self):
        self.assertIs(entry.accepted_worker(valid_worker())['status'], 'ok')
        for key, changed in (('op_completed', 42860), ('activation_completed', 44), ('keygen_runs', 1),
                ('checkpoint_reads', 0), ('task_accuracy_evaluated', True), ('source_factory', entry.binding.SOURCE_FACTORY)):
            value = valid_worker();value[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):entry.accepted_worker(value)

    def fake_run(self, root, *, failed=False, interrupted=False):
        args = SimpleNamespace(out=root / 'new-case', compile_case=root / 'closed-compile', archive_case=root / 'closed-archive',
            seconds=1800, memory_gib=4, case_gib=.5, minimum_free_gib=10,
            intake_python=Path('/trusted/nix-python'), registry=root / 'registry.jsonl')
        calls = []
        def process(argv, **kw):
            calls.append(argv)
            self.assertEqual((kw['timeout'], kw['mem_gib'], kw['fsize_gib']), (1920, 4, 1))
            self.assertIn('--worker', argv)
            self.assertNotIn('--worker-trace', argv);self.assertNotIn('--worker-compile', argv)
            if interrupted:return dict(clean(), exit_code=-15, signal=signal.SIGTERM)
            value = valid_worker()
            if failed:value.update(status='error', official_FHE_arithmetic_math_passed=False, op_completed=3)
            work = args.out / 'worker';work.mkdir()
            (work / 'worker-result.json').write_text(json.dumps(value))
            return dict(clean(), exit_code=1 if failed else 0)
        with (patch.object(entry.paths, 'RESULTS_DIR', root),
              patch.object(entry, 'source_gate', return_value=(dict(source_factory=entry.FACTORY), {})),
              patch.object(entry.supervision, 'check_resources', return_value={}),
              patch.object(entry, 'run_process', side_effect=process)):
            result = entry.execute(args)
        return args, result, calls

    def test_one_bounded_fake_worker_persists_eight_field_registry_and_completion(self):
        with TemporaryDirectory() as temp:
            args, result, calls = self.fake_run(Path(temp))
            self.assertEqual(result['status'], 'ok');self.assertEqual(len(calls), 1)
            self.assertEqual(result['worker_completed'], 1)
            self.assertIsNone(result['accuracy_acceptance']);self.assertEqual(result['archive_restores'], 0)
            row = json.loads(args.registry.read_text())
            self.assertEqual(set(row), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})
            completed = json.loads((args.out / 'completion.json').read_text())
            self.assertEqual(completed['status'], 'ok')
            self.assertEqual(completed['case_artifact_bytes'], entry.artifact_bytes(args.out))

    def test_failed_or_interrupted_worker_preserves_partial_and_has_no_retry(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), TemporaryDirectory() as temp:
                args, result, calls = self.fake_run(Path(temp), failed=True, interrupted=interrupted)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result['status'], 'interrupted' if interrupted else 'error')
                self.assertEqual(result['worker_completed'], 0)
                self.assertTrue((args.out / 'completion.json').is_file())
                if not interrupted:self.assertEqual(result['worker']['op_completed'], 3)

    def test_cli_finite_limits_reject_invalid_without_model_or_subprocess(self):
        for option, value in (('--memory-gib', '5'), ('--case-gib', '1'), ('--seconds', '1801'), ('--minimum-free-gib', '3'), ('--memory-gib', 'nan')):
            with self.subTest(option=option), redirect_stderr(StringIO()), patch.object(entry, 'execute', side_effect=AssertionError('No job allowed')):
                with self.assertRaises(SystemExit):entry.main(['--compile-case', '/old', '--archive-case', '/archive', '--out', '/new', option, value])


if __name__ == '__main__':
    unittest.main()
