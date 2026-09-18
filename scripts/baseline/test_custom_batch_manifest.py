"""Custom batches use user data, never a catalog name or a hidden live call."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from custom_batch_manifest import load_manifest
from run_agent_batch import summarize, load_failed_source
from audit_agent_lineage import catalog_for_report, lineage

BASE = Path(__file__).resolve().parent
MANIFEST = BASE/'cases/semantic-gap-manifest.json'


class CustomManifestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = json.loads(MANIFEST.read_text())

    def write(self, data):
        path = self.root/'models.json'
        path.write_text(json.dumps(data))
        return path

    def test_six_explicit_graphs_not_catalog_selection(self):
        rows, data, digest = load_manifest(MANIFEST)
        self.assertEqual(len(rows), 6)
        self.assertEqual(digest, hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
        self.assertEqual([r['descriptor'] for r in rows], data['cases'])
        self.assertEqual(summarize(rows)['planned'], 6)
        self.assertEqual(summarize(rows)['api_calls'], 0)

    def test_arbitrary_ids_weights_and_connections_remain_user_data(self):
        model = self.data['cases'][2]
        model['id'] = 'user-new-weights-not-in-catalog'
        model['constants']['weight'][0][0] = .173
        model['nodes'].append(dict(id='extra', op='negate', inputs=[model['output']]))
        model['output'] = 'extra'
        rows, data, _ = load_manifest(self.write(self.data))
        self.assertEqual(rows[2]['descriptor'], model)
        self.assertEqual(data['cases'][2]['constants']['weight'][0][0], .173)

    def test_duplicates_unsupported_ops_bad_shapes_and_extra_fields_rejected(self):
        invalid = []
        data = copy.deepcopy(self.data); data['cases'].append(data['cases'][0]); invalid.append(data)
        data = copy.deepcopy(self.data); data['cases'][0]['nodes'][0]['op'] = 'relu'; invalid.append(data)
        data = copy.deepcopy(self.data); data['cases'][0]['inputs'][0]['shape'] = [8]; invalid.append(data)
        invalid.extend([dict(self.data, commands=['anything']), dict(schema=True, cases=self.data['cases']),
                        dict(schema=1, cases=[]), dict(schema=1, cases=self.data['cases']*17)])
        for data in invalid:
            with self.subTest(data=str(data)[:80]), self.assertRaises(ValueError):
                load_manifest(self.write(data))

    def test_duplicate_json_nonfinite_and_oversize_rejected(self):
        path = self.root/'bad.json'
        for raw in ('{"schema":1,"schema":1,"cases":[]}', '{"schema":1,"cases":[NaN]}', ' '* (1024**2+1)):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_changed_manifest_and_execution_hash_rejected(self):
        path = self.write(self.data)
        _, _, digest = load_manifest(path)
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_manifest(path, digest)

    def test_plan_is_explicit_offline_and_exact_case_count(self):
        result = subprocess.run([sys.executable, '-B', str(BASE/'run_agent_batch.py'), '--plan',
                                 '--case-manifest', str(MANIFEST)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['mode'], 'plan_only')
        self.assertEqual(data['agent_calls'], 0)
        self.assertEqual(len(data['cases']), 6)
        self.assertEqual(data['summary']['not_completed'], 6)

    def test_manifest_and_extended_or_internal_hash_not_accepted(self):
        for extra in (['--extended'], ['--manifest-sha256', 'a'*64]):
            result = subprocess.run([sys.executable, '-B', str(BASE/'run_agent_batch.py'), '--plan',
                                     '--case-manifest', str(MANIFEST), *extra],
                                    capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('error:', result.stderr)

    def test_unsupported_manifest_rejected_before_credentials_or_provider_import(self):
        from unittest.mock import patch
        import run_agent_batch
        data = copy.deepcopy(self.data); data['cases'][0]['nodes'][0]['op'] = 'relu'
        path = self.write(data)
        with patch.object(sys, 'argv', ['run_agent_batch.py', '--live', '--case-manifest', str(path)]), \
             patch.dict(sys.modules, {'agent_credentials': None, 'deepseek_provider': None}):
            with self.assertRaisesRegex(ValueError, 'Unsupported graph'):
                run_agent_batch.main()


class CustomLineageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        initial_rows, self.data, self.digest = load_manifest(MANIFEST)
        self.catalog = [r['descriptor'] for r in initial_rows]

    def report(self, folder, rows, selection=None):
        folder = self.root/folder
        folder.mkdir()
        (folder/'custom-manifest.json').write_bytes(MANIFEST.read_bytes())
        report = dict(status='completed_with_failures', cases=rows,
                      custom_manifest=dict(file='custom-manifest.json', sha256=self.digest, case_count=6))
        if selection:
            report['selection'] = selection
        path = folder/'report.json'
        path.write_text(json.dumps(report))
        return path

    def rows(self):
        return [dict(descriptor=d, status='passed' if i == 0 else 'failed',
                     metrics=dict(passed=i == 0)) for i, d in enumerate(self.catalog)]

    def test_custom_cohort_and_failure_subset_preserve_all_descriptors(self):
        source = self.report('a', self.rows())
        prior, selected = load_failed_source(self.catalog, source, self.root)
        self.assertEqual(selected, self.catalog[1:])
        selection = dict(kind='previous_failures_only', source_report=str(source),
                         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        target = self.report('b', self.rows()[1:], selection)
        self.assertEqual(len(lineage(target, self.root)), 2)
        self.assertEqual(catalog_for_report(target, json.loads(target.read_text()), self.root), self.catalog)

    def test_modified_snapshot_and_nonlocal_filename_rejected(self):
        path = self.report('a', self.rows())
        report = json.loads(path.read_text())
        (path.parent/'custom-manifest.json').write_bytes(MANIFEST.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            catalog_for_report(path, report, self.root)
        report['custom_manifest']['file'] = '../outside.json'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            catalog_for_report(path, report, self.root)


if __name__ == '__main__':
    unittest.main()
