"""Two small metadata fixtures for the new selected-NPZ-only branch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import a_group_cold_closed_archive as own
from pt_agent.tests.test_a_group_cold_closed_archive import fixture, patched


def npz_fixture(root, name, *, raw=True):
    args, case, emit = fixture(root, name, status='error', raw=raw)
    args.out = root / (name + '-resource-archive-npz-only')
    args.npz_only_preserve_raw = True
    args.no_future_consumers = False
    args.no_future_selected_npz_consumers = True
    args.future_raw_consumers = True
    path = case / 'result.json'
    result = json.loads(path.read_text())
    result.update(placement='depth', monitor_stopped=True)
    path.write_text(json.dumps(result))
    return args, case, emit


class SelectedNPZOnlyArchiveTests(unittest.TestCase):
    def test_fake_selected_npz_archive_keeps_raw_and_clean_scientific_error(self):
        for name in own.NPZ_ONLY_CASES:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args, case, emit = npz_fixture(root, name)
                raw = emit / 'candidate_trace.mlir'
                original_raw = own.prior.snapshot(raw)
                original_result = own.prior.snapshot(case / 'result.json')
                selected = []
                def fake_archive(source, target, partial):
                    selected.append(str(source))
                    amount = source.stat().st_size
                    target.write_text('x')
                    source.unlink()
                    return dict(status='compressed', source=str(source), archive=str(target),
                        released_bytes=amount - 1, restore_argv=['gzip', '-dk', '--', str(target)])
                with patched(root, fake_archive, free=4 * own.GIB + 2 * 1024**2):
                    result = own.run(args)
                self.assertEqual(selected, [str(emit / 'constants.npz')])
                self.assertEqual(own.prior.snapshot(raw), original_raw)
                self.assertEqual(own.prior.snapshot(case / 'result.json'), original_result)
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(result['scientific_status'], 'error')
                self.assertFalse(result['scientific_status_upgraded'])
                self.assertEqual(result['completed_files'], 1)
                self.assertTrue(result['future_raw_consumers'])
                self.assertTrue(result['no_future_selected_NPZ_consumers'])
                admission = result['source_admission']
                self.assertNotIn('no_future_consumers', admission)
                self.assertTrue(any(row['path'] == str(raw) for row in admission['preserved']))
                self.assertFalse((emit / 'candidate_trace.mlir.gz').exists())
                self.assertTrue((emit / 'lowered._hecate_golden.runtime-plan.json').is_file())
                self.assertTrue((emit / 'lowered._hecate_golden.bundle/data/constant.bin').is_file())
                self.assertTrue((case / 'work/intake/model_info/model_info.json').is_file())
                self.assertEqual(json.loads((args.out / 'completion.json').read_text())['status'], 'ok')
                rows = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual(set(rows[-1]), own.prior.REGISTRY_FIELDS)

    def test_scope_live_source_and_partial_refused_missing_raw_skips_npz(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, case, emit = npz_fixture(root, own.NPZ_ONLY_CASES[0])
            with patched(root):
                for flag, value in (('no_future_consumers', True),
                                    ('no_future_selected_npz_consumers', False),
                                    ('future_raw_consumers', False)):
                    original = getattr(args, flag)
                    setattr(args, flag, value)
                    with self.assertRaises(ValueError): own.admit(args)
                    setattr(args, flag, original)
                with patch.object(own.prior, 'same_birth', return_value=True):
                    with self.assertRaises(ValueError): own.admit(args)
                partial = emit / 'constants.npz.gz.partial'
                partial.symlink_to(emit / 'absent-target')
                with self.assertRaises(ValueError): own.admit(args)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, case, emit = npz_fixture(root, own.NPZ_ONLY_CASES[1], raw=False)
            with patched(root), patch.object(own, 'archive_one') as archive:
                result = own.run(args)
            archive.assert_not_called()
            self.assertEqual(result['resource_status'], 'metadata_skip')
            self.assertEqual(result['scientific_status'], 'error')
            self.assertEqual(result['completed_files'], 0)
            self.assertTrue((emit / 'constants.npz').exists())


if __name__ == '__main__':
    unittest.main()
