"""Profile provenance binding must never inherit unverified candidate choices."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from backend import digest, json_bytes
from generate_aligned import BOOT_DEFAULTS, bind_native_config, candidate, main, make_parser
from cerium_packing import Config
from test_cerium_lowering import CONTEXT, PROFILE


class ProfileBindingTests(unittest.TestCase):
    def setUp(self):
        self.context, self.profile, native = candidate('q52-p11', 32)
        self.native = dict(BOOT_DEFAULTS)
        self.native.update(native)
        self.native['eval_mod_full_coefficients'] = False
        self.fixture = dict(context=self.context, native_boot=self.native)
        self.raw = json_bytes(self.fixture)
        self.report = dict(fixture_sha256=digest(self.raw), native_boot_profile=dict(
            profile_id=self.profile['profile_id'], config={key: self.native[key] for key in (
                'bootstrap_ratio', 'eval_mod_log_scale', 'output_level', 'project_real')}))

    def test_incomplete_report_without_bound_fixture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'legacy probe omits'):
            bind_native_config(self.report, self.context)

    def test_bound_fixture_not_candidate_controls_full_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            path.write_bytes(self.raw)
            actual = bind_native_config(self.report, self.context, path)
        self.assertFalse(actual['eval_mod_full_coefficients'])
        self.assertEqual(actual['eval_mod_sine_degree'], 30)
        self.assertEqual(actual['logical_rescale_count'], 2)

    def test_modified_original_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            altered = deepcopy(self.fixture)
            altered['native_boot']['eval_mod_full_coefficients'] = True
            path.write_bytes(json_bytes(altered))
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                bind_native_config(self.report, self.context, path)

    def test_report_and_original_config_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            path.write_bytes(self.raw)
            altered = deepcopy(self.report)
            altered['native_boot_profile']['config']['bootstrap_ratio'] = 256
            with self.assertRaisesRegex(ValueError, 'disagree on bootstrap_ratio'):
                bind_native_config(altered, self.context, path)

    def test_complete_report_needs_no_candidate_defaults(self):
        report = deepcopy(self.report)
        report['native_boot_profile']['config'] = self.native
        actual = bind_native_config(report, self.context)
        self.assertFalse(actual['eval_mod_full_coefficients'])
        self.assertEqual(actual['eval_mod_k'], 16)

    def test_true_prime_pool_is_disjoint_and_ntt_compatible(self):
        for name, q_count, p_count in [('q52-p11', 52, 11), ('q49-p13', 49, 13)]:
            context, _, native = candidate(name)
            self.assertEqual(len(context['q_moduli']), q_count)
            self.assertEqual(len(context['p_moduli']), p_count)
            self.assertFalse(set(context['q_moduli']) & set(context['p_moduli']))
            self.assertTrue(all((q - 1) % 131072 == 0 for q in context['q_moduli'] + context['p_moduli']))
            self.assertEqual(native['bootstrap_ratio'], 256)
            self.assertEqual(context['secret_hamming_weight'], 32768)


class OptimizationFlagTests(unittest.TestCase):
    def test_parser_flags_are_explicit_and_default_off(self):
        parser = make_parser()
        args = parser.parse_args(['--output-dir', 'unused-output'])
        self.assertFalse(args.algebra_rewrite)
        self.assertFalse(args.shared_rotation_rescale)
        self.assertFalse(args.prepared_qk)
        self.assertFalse(args.qk_pack_inputs)
        self.assertFalse(args.prepared_av)
        self.assertFalse(args.av_pack_inputs)
        self.assertFalse(args.polynomial_reassociation)
        self.assertIsNone(args.data_from_plan)
        args = parser.parse_args(['--output-dir', 'unused-output', '--algebra-rewrite',
                                  '--shared-rotation-rescale', '--prepared-qk', '--polynomial-reassociation'])
        self.assertTrue(args.algebra_rewrite and args.shared_rotation_rescale and args.prepared_qk)
        self.assertTrue(args.polynomial_reassociation)

    def test_av_packing_cli_dependencies_and_physical_contract_fail_early(self):
        cases = [([], 'requires --prepared-av'),
                 (['--prepared-av'], 'requires --algebra-rewrite'),
                 (['--prepared-av', '--algebra-rewrite', '--scope', 'ffn-up'], 'cyclic selective'),
                 (['--prepared-av', '--algebra-rewrite', '--attention-layout', 'offset'], 'cyclic selective'),
                 (['--prepared-av', '--algebra-rewrite', '--strategy', 'budget'], 'cyclic selective'),
                 (['--prepared-av', '--algebra-rewrite', '--candidate', 'q49-p13'], 'cyclic selective'),
                 (['--prepared-av', '--algebra-rewrite', '--input-level', '10'], 'input L12')]
        with tempfile.TemporaryDirectory() as directory:
            for flags, message in cases:
                with self.subTest(flags=flags), patch('generate_aligned.create_weights') as factory:
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                        main(['--output-dir', str(Path(directory) / 'absent'), '--av-pack-inputs'] + flags)
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn(message, stderr.getvalue())
                    factory.assert_not_called()

    def test_tiny_cli_default_and_opt_in_metadata(self):
        config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        cases = [[], ['--algebra-rewrite'], ['--shared-rotation-rescale'],
                 ['--algebra-rewrite', '--shared-rotation-rescale'], ['--prepared-qk'],
                 ['--algebra-rewrite', '--shared-rotation-rescale', '--prepared-qk'],
                 ['--polynomial-reassociation'], ['--prepared-av'],
                 ['--algebra-rewrite', '--shared-rotation-rescale', '--prepared-qk', '--polynomial-reassociation'],
                 ['--algebra-rewrite', '--shared-rotation-rescale', '--prepared-qk', '--prepared-av', '--polynomial-reassociation'],
                 ['--prepared-qk', '--qk-pack-inputs'],
                 ['--algebra-rewrite', '--shared-rotation-rescale', '--prepared-qk', '--qk-pack-inputs',
                  '--prepared-av', '--polynomial-reassociation'],
                 ['--algebra-rewrite', '--prepared-av', '--av-pack-inputs'],
                 ['--algebra-rewrite', '--shared-rotation-rescale', '--prepared-qk', '--qk-pack-inputs',
                  '--prepared-av', '--av-pack-inputs', '--polynomial-reassociation']]
        with tempfile.TemporaryDirectory() as directory:
            for index, flags in enumerate(cases):
                with self.subTest(flags=flags):
                    output = Path(directory) / str(index)
                    with patch('generate_aligned.Config', return_value=config), \
                         patch('generate_aligned.candidate', return_value=(CONTEXT, PROFILE, {})), \
                         redirect_stdout(io.StringIO()):
                        main(['--output-dir', str(output)] + flags)
                    fixture = json.loads((output / 'fixture.json').read_bytes())
                    self.assertNotIn('data_binding', fixture['model'])
                    for field, flag in [('algebra_rewrite', '--algebra-rewrite'),
                                        ('shared_rotation_rescale', '--shared-rotation-rescale'),
                                        ('prepared_qk', '--prepared-qk'),
                                        ('qk_pack_inputs', '--qk-pack-inputs'),
                                        ('prepared_av', '--prepared-av'),
                                        ('av_pack_inputs', '--av-pack-inputs'),
                                        ('polynomial_reassociation', '--polynomial-reassociation')]:
                        self.assertEqual(field in fixture['model'], flag in flags)
                    self.assertEqual('prepared_qk_manifest' in fixture, '--prepared-qk' in flags)
                    if '--prepared-qk' in flags:
                        self.assertEqual(fixture['prepared_qk_manifest_sha256'],
                                         digest((output / 'prepared-qk.json').read_bytes()))
                        manifest = json.loads((output / 'prepared-qk.json').read_bytes())
                        self.assertEqual(manifest['format_version'], 2 if '--qk-pack-inputs' in flags else 1)
                        self.assertEqual(fixture.get('prepared_qk_required_capabilities', []),
                                         ['pack_inputs'] if '--qk-pack-inputs' in flags else [])
                        self.assertEqual('prepared_qk_required_capabilities' in fixture, '--qk-pack-inputs' in flags)
                        self.assertTrue(all(('pack_inputs' in row) == ('--qk-pack-inputs' in flags)
                                            for row in manifest['calls']))
                        self.assertTrue(all(row.get('pack_inputs', False) == ('--qk-pack-inputs' in flags)
                                            for row in manifest['calls']))
                    self.assertEqual('prepared_av_manifest' in fixture, '--prepared-av' in flags)
                    if '--prepared-av' in flags:
                        self.assertEqual(fixture['prepared_av_manifest_sha256'],
                                         digest((output / 'prepared-av.json').read_bytes()))
                        manifest = json.loads((output / 'prepared-av.json').read_bytes())
                        partial = '--av-pack-inputs' in flags
                        self.assertEqual(manifest['format_version'], 2 if partial else 1)
                        self.assertEqual(fixture.get('prepared_av_required_capabilities', []),
                                         ['packed_input2_partial'] if partial else [])
                        self.assertEqual('prepared_av_required_capabilities' in fixture, partial)
                        self.assertTrue(all(('algorithm' in row) == partial for row in manifest['calls']))
                        if partial:
                            self.assertTrue(all(row['algorithm'] == 'PackedInput2Partial' and
                                                row['callee'] == 'bert.compact_av_packed_partial'
                                                for row in manifest['calls']))


if __name__ == '__main__':
    unittest.main()
