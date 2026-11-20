"""Finite parser scheduling contracts; no child process or model execution."""
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import autonomous_serial_batch as subject


class OrbitParserSerialAdmission(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / 'parser'
        self.argv = ['python3', '-B', '-m',
            subject.PREFIX + 'orbit_backend_parse_only', '--out', str(self.out),
            '--source-case', str(self.root / 'optimized'), '--shared-build-case',
            str(self.root / 'shared-build'), '--run']

    def manifest(self):
        return dict(schema_version=1, deadline_utc='2026-11-17T10:00:00+00:00',
            watch=dict(case=str(self.root / 'current-batch'), pid=42,
                       starttime=100, seconds=86400), jobs=[dict(
                id='official-parser-consumption', kind='experiment',
                argv=self.argv, out=str(self.out), seconds=1560,
                requires_case_statuses={str(self.root / 'shared-build'): 'ok',
                                       str(self.root / 'optimized'): 'ok'})])

    def test_known_parser_preserves_explicit_source_and_build_dependencies(self):
        manifest = self.manifest()
        self.assertIs(subject.validate(manifest, self.root), manifest)
        self.assertEqual(manifest['jobs'][0]['requires_case_statuses'],
            {str(self.root / 'shared-build'): 'ok', str(self.root / 'optimized'): 'ok'})

    def test_parser_uses_original_finite_parent_limits(self):
        self.assertEqual(subject.parent_limits(self.manifest()['jobs'][0]), (24, 16))
        expanded = dict(self.manifest()['jobs'][0], outer_address_space_gib=32,
                        outer_file_gib=64)
        with self.assertRaises(ValueError):
            subject.parent_limits(expanded)

    def test_detached_or_foreign_output_commands_are_refused(self):
        with self.assertRaises(ValueError):
            subject.check_command(self.argv + ['--detach'], self.out)
        with self.assertRaises(ValueError):
            subject.check_command(self.argv, self.root / 'foreign-output')

    def test_parser_preparation_is_not_admitted_as_actual_success(self):
        self.assertFalse(subject.accepted_status(self.argv, 'prepared'))
        self.assertTrue(subject.accepted_status(self.argv, 'ok'))


if __name__ == '__main__':
    unittest.main()
