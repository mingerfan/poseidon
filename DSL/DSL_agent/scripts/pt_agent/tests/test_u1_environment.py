"""Offline checks of the user-approved independent intake installation plan."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.tools import intake_python_env as environment


class IntakeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(environment.PROFILE.read_text())
        self.lock = json.loads((environment.PROFILE.parent / self.profile['lock']).read_text())

    def test_approved_version_families_and_removed_xet(self):
        versions = {row['name']: row['version'] for row in self.lock['wheels']}
        self.assertTrue(versions['protobuf'].startswith('4.25.'))
        self.assertTrue(versions['huggingface-hub'].startswith('0.24.'))
        self.assertNotIn('hf-xet', versions)
        self.assertEqual(versions['torch'], '2.0.1')
        self.assertEqual(versions['numpy'], '1.25.2')

    def test_complete_wheel_plan_within_approved_budget(self):
        self.assertEqual(len(self.lock['wheels']), 29)
        self.assertEqual(sum(row['bytes'] for row in self.lock['wheels']), self.lock['total_download_bytes'])
        self.assertLessEqual(self.lock['total_download_bytes'], environment.DOWNLOAD_LIMIT)
        for row in self.lock['wheels']:
            self.assertTrue({'name', 'version', 'filename', 'bytes', 'sha256', 'url'} <= row.keys())
            self.assertTrue(row['url'].startswith('https://files.pythonhosted.org/'))

    def test_install_rejects_excess_budget_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            lock = dict(self.lock, wheels=[dict(row) for row in self.lock['wheels']])
            lock['wheels'][0]['bytes'] = environment.DOWNLOAD_LIMIT + 1
            (directory / self.profile['lock']).write_text(json.dumps(lock))
            profile = directory / 'profile.json'
            profile.write_text(json.dumps(self.profile))
            args = SimpleNamespace(profile=profile, out=directory / 'output', install_approved=True)
            with self.assertRaisesRegex(ValueError, 'approved 133 MiB'):
                environment.install_or_verify(args)
            self.assertFalse(args.out.exists())

    def test_install_rejects_unapproved_family_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            lock = dict(self.lock, wheels=[dict(row) for row in self.lock['wheels']])
            next(row for row in lock['wheels'] if row['name'] == 'protobuf')['version'] = '7.36.2'
            (directory / self.profile['lock']).write_text(json.dumps(lock))
            profile = directory / 'profile.json'
            profile.write_text(json.dumps(self.profile))
            args = SimpleNamespace(profile=profile, out=directory / 'output', install_approved=True)
            with self.assertRaisesRegex(ValueError, 'user-approved'):
                environment.install_or_verify(args)
            self.assertFalse(args.out.exists())

    def test_no_mode_does_not_start_installation(self):
        with patch('sys.argv', ['intake_python_env']), patch.object(environment, 'install_or_verify') as install:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                environment.main()
            install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
