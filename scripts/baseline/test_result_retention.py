import json
import os
from pathlib import Path
import tempfile
import unittest
from result_retention import cleanup_run, plan_run, KEYS


class RetentionTests(unittest.TestCase):
    def fixture(self,root):
        run=root/'agent-deepseek-fixture'; run.mkdir()
        (run/'report.json').write_text(json.dumps(dict(status='passed')))
        (run/'parameters.json').write_text(json.dumps(dict(seal_version='4.0.0',polynomial_degree=32768,
            security_check='tc128',parameters_set=True,rotation_steps=[1,2])))
        (run/'private-keys').mkdir()
        for name in KEYS: (run/'private-keys'/name).write_bytes(b'fake-generated-key')
        (run/'model.json').write_text('{}')
        (run/'decrypted.npy').write_bytes(b'keep result')
        return run

    def test_only_keys_removed_and_reports_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); run=self.fixture(root)
            before={p.name:p.read_bytes() for p in run.iterdir() if p.is_file()}
            self.assertEqual(len(plan_run(run,root)['files']),5)
            self.assertTrue((run/'private-keys/sec.seal').exists())
            cleanup_run(run,root)
            for name,raw in before.items(): self.assertEqual((run/name).read_bytes(),raw)
            self.assertFalse((run/'private-keys').exists())
            self.assertTrue(json.loads((run/'key-cleanup-outcome.json').read_text())['complete'])
            self.assertEqual(cleanup_run(run,root)['bytes'],0)

    def test_running_unknown_and_symlink_targets_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); run=self.fixture(root)
            (run/'report.json').write_text('{"status":"running"}')
            with self.assertRaises(ValueError): cleanup_run(run,root)
            (run/'report.json').write_text('{"status":"passed"}')
            unexpected=run/'private-keys/unknown'; unexpected.write_bytes(b'keep')
            with self.assertRaises(ValueError): cleanup_run(run,root)
            unexpected.unlink()
            key=run/'private-keys/sec.seal'; key.unlink(); key.symlink_to(run/'model.json')
            with self.assertRaises(ValueError): cleanup_run(run,root)
            self.assertEqual((run/'model.json').read_text(),'{}')
            with self.assertRaises(ValueError): plan_run(root,root)

    def test_shared_hardlinked_keys_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); run=self.fixture(root)
            os.link(run/'private-keys/gal.seal',root/'shared-key')
            with self.assertRaises(ValueError): cleanup_run(run,root)
            self.assertTrue((root/'shared-key').exists())


if __name__=='__main__': unittest.main()
