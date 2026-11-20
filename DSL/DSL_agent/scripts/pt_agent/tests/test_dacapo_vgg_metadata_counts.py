"""Metadata-only regression for bundle references and independent blob files."""
from argparse import Namespace
from collections import Counter
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import dacapo_vgg16_plan_math as entry


class StopAfterInventory(RuntimeError):
    """A synthetic boundary before any model, probe or constant-array load."""


class VGGMetadataCountChecks(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_original_structure_counts_keep_references_separate_from_independent_files(self):
        self.assertEqual(entry.EXPECTED, dict(execution_count=42860, value_count=52860,
            initialization_count=9999, inline_scalar_count=179, bundle_encode_count=9820,
            unique_blob_count=9716, bootstrap_count=20))
        self.assertEqual(entry.EXPECTED['inline_scalar_count'] + entry.EXPECTED['bundle_encode_count'],
                         entry.EXPECTED['initialization_count'])
        self.assertEqual(entry.EXPECTED['bundle_encode_count'] - entry.EXPECTED['unique_blob_count'], 104)

    def test_original_inventory_algorithm_counts_repeated_blob_references(self):
        plan, manifest, _, _, _ = entry.binding.make_public_fixture()
        repeated = copy.deepcopy(next(row for row in plan['initialization'] if row['payload']['kind'] == 'bundle'))
        source = next(row for row in plan['values'] if row['id'] == repeated['output'])
        repeated['output'] = 'half-second-reference'
        plan['values'].append(dict(source, id=repeated['output']))
        plan['initialization'].append(repeated)
        for ordinal, row in enumerate(plan['initialization'] + plan['execution']):
            row['ordinal'] = ordinal
        metadata = entry.binding.plan_inventory(plan, manifest, public_fixture=True)
        self.assertEqual(metadata['initialization_count'], 3)
        self.assertEqual(metadata['inline_scalar_count'], 1)
        self.assertEqual(metadata['bundle_encode_count'], 2)
        self.assertEqual(metadata['unique_blob_count'], 1)
        self.assertEqual(len(manifest['blobs']), 1)
        self.assertEqual(Counter(row['payload']['content'] for row in plan['initialization']
                                if row['payload']['kind'] == 'bundle'), {'public:half': 2})

    def worker_inventory(self, counts):
        work = self.root/'worker'
        args = Namespace(work_dir=work, memory_gib=4, seconds=1800, case_gib=.5,
                         minimum_free_gib=10)
        source = dict(bundle=str(self.root/'no-bundle-opened'),
                      archive_manifest=str(self.root/'no-archive-opened'))
        program = dict(metadata=dict(counts))
        with ExitStack() as stack:
            stack.enter_context(patch.object(entry.resource, 'setrlimit'))
            stack.enter_context(patch.object(entry, 'source_gate', return_value=(source, {})))
            stack.enter_context(patch.object(entry, 'load_program', return_value=(program, {})))
            stop = stack.enter_context(patch.object(entry.replay, 'check_restored_bundle',
                side_effect=StopAfterInventory('synthetic stop before payloads')))
            stack.enter_context(patch.object(entry.replay, 'LazyPlaintexts',
                side_effect=AssertionError('No constant arrays in metadata tests')))
            stack.enter_context(patch.object(entry.replay.reference, '_load_probes',
                side_effect=AssertionError('No probes in metadata tests')))
            result = entry.worker(args)
        saved = json.loads((work/'worker-result.json').read_text())
        self.assertEqual(saved['error'], result['error'])
        self.assertEqual(saved['checkpoint_load_attempts'], 0)
        self.assertEqual(saved['packed_forward_attempts'], 0)
        self.assertEqual(saved['math_check_attempts'], 0)
        self.assertEqual(saved['constant_reads'], None)
        self.assertEqual(saved['model_forwards'], 0)
        self.assertFalse(saved['official_FHE_arithmetic_math_passed'])
        return result, stop

    def test_actual_worker_guard_admits_correct179_9820_9716_before_payload_boundary(self):
        counts = dict(execution_count=42860, value_count=52860, initialization_count=9999,
            inline_scalar_count=179, bundle_encode_count=9820, unique_blob_count=9716, bootstrap_count=20)
        result, stop = self.worker_inventory(counts)
        stop.assert_called_once()
        self.assertEqual(result['error_type'], 'StopAfterInventory')
        self.assertEqual(result['program_metadata'], counts)

    def test_actual_worker_guard_rejects_old283_9716_conflation_before_payload_boundary(self):
        counts = dict(entry.EXPECTED, inline_scalar_count=283, bundle_encode_count=9716)
        result, stop = self.worker_inventory(counts)
        stop.assert_not_called()
        self.assertEqual(result['error_type'], 'ValueError')
        self.assertEqual(result['error'], 'Original VGG complete plan inventory differs: inline_scalar_count')
        self.assertNotIn('program_metadata', result)

    def test_actual_worker_guard_rejects_bundle_reference_count_replaced_by_unique_count(self):
        result, stop = self.worker_inventory(dict(entry.EXPECTED, bundle_encode_count=9716))
        stop.assert_not_called()
        self.assertEqual(result['error'], 'Original VGG complete plan inventory differs: bundle_encode_count')

    def test_actual_worker_guard_keeps_independent_blob_count9716(self):
        result, stop = self.worker_inventory(dict(entry.EXPECTED, unique_blob_count=9820))
        stop.assert_not_called()
        self.assertEqual(result['error'], 'Original VGG complete plan inventory differs: unique_blob_count')

    def test_other_original_counts_are_still_enforced(self):
        for name in ('execution_count', 'value_count', 'initialization_count', 'bootstrap_count'):
            with self.subTest(name=name):
                self.root = self.root/name
                self.root.mkdir()
                result, stop = self.worker_inventory(dict(entry.EXPECTED, **{name: entry.EXPECTED[name]+1}))
                stop.assert_not_called()
                self.assertEqual(result['error'], 'Original VGG complete plan inventory differs: '+name)


if __name__ == '__main__':
    unittest.main()
