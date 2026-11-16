"""Exact parameter/input binding must not trust cross-environment RNG seeds."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from backend import digest, json_bytes
from cerium_packing import Config, create_weights, unpack_outputs
from generate_aligned import main
from test_cerium_lowering import CONTEXT, PROFILE
from weight_binding import bind_frozen_weights, load_frozen_input


class FrozenWeightBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.golden = self.root / 'golden'
        self.config = Config(tokens=4, hidden=8, heads=4, tile=8, slots=32, intermediate=24)
        self.seed = 47
        self.run_cli(self.golden, [])
        self.fixture = json.loads((self.golden / 'fixture.json').read_bytes())

    def run_cli(self, target, flags):
        with patch('generate_aligned.Config', return_value=self.config), \
             patch('generate_aligned.candidate', return_value=(CONTEXT, PROFILE, {})), \
             redirect_stdout(io.StringIO()):
            main(['--output-dir', str(target), '--seed', str(self.seed)] + flags)

    def test_exact_dense_data_restores_changed_regeneration_without_mutating_it(self):
        original = create_weights(self.config, self.seed)
        regenerated = deepcopy(original)
        regenerated[0]['v_weight'][:] = .125
        regenerated[0]['ffn_up_bias'][:] = -.25
        result, provenance = bind_frozen_weights(regenerated, self.config, self.seed, self.golden)
        self.assertEqual(result[0]['v_weight'].tobytes(), original[0]['v_weight'].tobytes())
        self.assertEqual(result[0]['ffn_up_bias'].tobytes(), original[0]['ffn_up_bias'].tobytes())
        self.assertTrue((regenerated[0]['v_weight'] == .125).all())
        self.assertTrue(provenance['all_parameter_hashes_verified'])
        self.assertEqual(provenance['loaded_dense_matrices'], 6)

    def test_unavailable_non_dense_parameter_mismatch_fails_closed(self):
        regenerated = create_weights(self.config, self.seed)
        regenerated[0]['attention_ln_gamma'][0] += .1
        with self.assertRaisesRegex(ValueError, 'attention_ln_gamma'):
            bind_frozen_weights(regenerated, self.config, self.seed, self.golden)

    def test_config_seed_and_dense_file_sha_are_enforced(self):
        regenerated = create_weights(self.config, self.seed)
        with self.assertRaisesRegex(ValueError, 'identical model config and seed'):
            bind_frozen_weights(regenerated, self.config, self.seed + 1, self.golden)
        with (self.golden / 'linear-weights.bin').open('ab') as changed:
            changed.write(b'\x00')
        with self.assertRaisesRegex(ValueError, 'weight file SHA256 mismatch'):
            bind_frozen_weights(regenerated, self.config, self.seed, self.golden)

    def test_original_input_hash_is_verified_not_reconstructed_from_outputs(self):
        hidden, input_hash = load_frozen_input(self.config, self.golden)
        self.assertEqual(hidden.shape, (4, 8))
        self.assertEqual(input_hash, self.fixture['model']['input_sha256'])
        with self.assertRaisesRegex(ValueError, 'client input config mismatch'):
            load_frozen_input(replace(self.config, ln_epsilon=.01), self.golden)
        self.fixture['inputs'][0]['values'][0] += 1
        (self.golden / 'fixture.json').write_bytes(json_bytes(self.fixture))
        with self.assertRaisesRegex(ValueError, 'client input SHA256 mismatch'):
            load_frozen_input(self.config, self.golden)

    def test_changed_valid_fixture_cannot_mix_weight_and_input_snapshots(self):
        regenerated = create_weights(self.config, self.seed)
        _, binding = bind_frozen_weights(regenerated, self.config, self.seed, self.golden)
        self.fixture['inputs'][0]['values'][0] += 1
        packed = np.asarray([r['values'] for r in self.fixture['inputs']], dtype='<f8')
        changed = unpack_outputs(packed, self.config)
        self.fixture['model']['input_sha256'] = digest(changed.astype('<f8').tobytes())
        (self.golden / 'fixture.json').write_bytes(json_bytes(self.fixture))
        load_frozen_input(self.config, self.golden)  # independently valid new fixture
        with self.assertRaisesRegex(ValueError, 'fixture SHA256 changed'):
            load_frozen_input(self.config, self.golden,
                              expected_fixture_sha256=binding['fixture_sha256'])

    def test_malformed_manifest_schemas_and_parameter_hashes_fail_cleanly(self):
        manifest_path = self.golden / 'compact-linear.json'
        original = json.loads(manifest_path.read_bytes())
        cases = []
        missing = deepcopy(original); missing.pop('weight_sha256'); cases.append(missing)
        duplicate = deepcopy(original); duplicate['matrices'].append(duplicate['matrices'][0]); cases.append(duplicate)
        for offset in (-1, '0', 10**12):
            invalid = deepcopy(original); invalid['matrices'][0]['weight_offset'] = offset; cases.append(invalid)
        invalid = deepcopy(original); invalid['matrices'][0]['matrix_id'] = 5; cases.append(invalid)
        regenerated = create_weights(self.config, self.seed)
        for manifest in cases:
            with self.subTest(manifest=manifest):
                manifest_path.write_bytes(json_bytes(manifest))
                with self.assertRaises(ValueError):
                    bind_frozen_weights(regenerated, self.config, self.seed, self.golden)
        manifest_path.write_bytes(json_bytes(original))
        self.fixture['model']['weight_sha256']['0']['q_weight'] = 'sha256:' + '0' * 64
        (self.golden / 'fixture.json').write_bytes(json_bytes(self.fixture))
        with self.assertRaisesRegex(ValueError, 'layer0.q_weight'):
            bind_frozen_weights(regenerated, self.config, self.seed, self.golden)

    def test_explicit_prefix_uses_first_layers_and_same_original_input(self):
        prefix_config = self.config
        self.config = replace(prefix_config, layers=2)
        source = self.root / 'two-layers'
        self.run_cli(source, [])
        self.config = prefix_config
        regenerated = create_weights(prefix_config, self.seed)
        with self.assertRaises(ValueError):
            bind_frozen_weights(regenerated, prefix_config, self.seed, source)
        bound, metadata = bind_frozen_weights(regenerated, prefix_config, self.seed, source,
                                             allow_prefix_layers=True)
        with self.assertRaises(ValueError):
            bind_frozen_weights(regenerated, prefix_config, self.seed + 1, source, allow_prefix_layers=True)
        with self.assertRaises(ValueError):
            bind_frozen_weights(regenerated, replace(prefix_config, ln_epsilon=.01), self.seed,
                                source, allow_prefix_layers=True)
        changed_ln = deepcopy(regenerated)
        changed_ln[0]['attention_ln_gamma'][0] += .125
        with self.assertRaisesRegex(ValueError, 'attention_ln_gamma'):
            bind_frozen_weights(changed_ln, prefix_config, self.seed, source, allow_prefix_layers=True)
        for name, value in bound[0].items():
            self.assertEqual(value.tobytes(), regenerated[0][name].tobytes())
        self.assertEqual(metadata['source_layers'], 2)
        self.assertEqual(metadata['selected_prefix_layers'], 1)
        self.assertEqual(metadata['loaded_dense_matrices'], 6)
        self.assertFalse(metadata['all_source_parameter_hashes_verified'])
        a, _ = load_frozen_input(prefix_config, self.golden)
        b, _ = load_frozen_input(prefix_config, source, allow_prefix_layers=True,
                                  expected_fixture_sha256=metadata['fixture_sha256'])
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            bind_frozen_weights(create_weights(replace(prefix_config, layers=3), self.seed),
                                replace(prefix_config, layers=3), self.seed, source, allow_prefix_layers=True)
        target = self.root / 'bound-prefix'
        self.run_cli(target, ['--data-from-plan', str(source), '--data-prefix-layers'])
        self.assertEqual((target / 'linear-weights.bin').read_bytes(),
                         (self.golden / 'linear-weights.bin').read_bytes())
        selected = json.loads((target / 'fixture.json').read_bytes())
        self.assertEqual(selected['model']['config']['layers'], 1)
        self.assertEqual(set(selected['model']['weight_sha256']), {'0'})
        self.assertEqual(len(json.loads((target / 'compact-linear.json').read_bytes())['matrices']), 6)
        self.assertEqual(len(json.loads((target / 'layer-boundaries.json').read_bytes())['layers']), 1)

    def test_prefix_still_checks_unused_matrix_descriptors(self):
        prefix_config = self.config
        self.config = replace(prefix_config, layers=2)
        source = self.root / 'two-layers-invalid'
        self.run_cli(source, [])
        self.config = prefix_config
        manifest_path = source / 'compact-linear.json'
        manifest = json.loads(manifest_path.read_bytes())
        variants = []
        negative = deepcopy(manifest); negative['matrices'][-1]['weight_offset'] = -1; variants.append(negative)
        missing = deepcopy(manifest); missing['matrices'].pop(); variants.append(missing)
        duplicate = deepcopy(manifest); duplicate['matrices'].append(duplicate['matrices'][-1]); variants.append(duplicate)
        shape = deepcopy(manifest); shape['matrices'][-1]['input_width'] += 1; variants.append(shape)
        for variant in variants:
            with self.subTest(variant=variant):
                manifest_path.write_bytes(json_bytes(variant))
                with self.assertRaises(ValueError):
                    bind_frozen_weights(create_weights(prefix_config, self.seed), prefix_config,
                                        self.seed, source, allow_prefix_layers=True)

    def test_cli_binds_all_parameters_and_input_with_optimizer_flags(self):
        target = self.root / 'bound'
        self.run_cli(target, ['--data-from-plan', str(self.golden), '--algebra-rewrite',
                              '--shared-rotation-rescale', '--prepared-qk', '--prepared-av', '--polynomial-reassociation'])
        actual = json.loads((target / 'fixture.json').read_bytes())
        self.assertEqual(actual['model']['input_sha256'], self.fixture['model']['input_sha256'])
        self.assertEqual(actual['model']['weight_sha256'], self.fixture['model']['weight_sha256'])
        self.assertEqual((target / 'linear-weights.bin').read_bytes(),
                         (self.golden / 'linear-weights.bin').read_bytes())
        binding = actual['model']['data_binding']
        self.assertTrue(binding['original_client_input_loaded'])
        self.assertTrue(binding['all_parameter_hashes_verified'])


if __name__ == '__main__':
    unittest.main()
