"""Finite replay queue contracts; no experimental process or API execution."""
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import autonomous_serial_batch as subject


class QwenReplaySerialAdmission(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root/'guarded'
        self.argv = ['python3', '-B', '-m', subject.PREFIX+'qwen_guarded_replay',
            '--out', str(self.out), '--through', 'agent', '--resume-from',
            str(self.root/'nine'), '--partial-from', str(self.root/'partial'),
            '--memory-gib', '24', '--rss-gib', '16', '--minimum-available-gib', '1.25',
            '--case-gib', '32', '--seconds', '10800']

    def test_explicit_replay_is_admitted_with_original_default_parent(self):
        job = dict(argv=self.argv, out=str(self.out), seconds=11040)
        self.assertEqual(subject.check_command(self.argv, self.out), self.argv)
        self.assertEqual(subject.parent_limits(job), (24, 16))
        manifest = dict(schema_version=1, deadline_utc='2026-11-17T10:00:00+00:00',
            watch=dict(case=str(self.root/'previous'), pid=42, starttime=99, seconds=86400),
            jobs=[dict(job, id='retained-reply13')])
        self.assertIs(subject.validate(manifest, self.root), manifest)

    def test_replay_does_not_gain_prefill_parent_expansion(self):
        for pair in ((32, 16), (32, 64), (48, 16)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                subject.parent_limits(dict(argv=self.argv, outer_address_space_gib=pair[0],
                                           outer_file_gib=pair[1]))

    def test_replay_partial_preparation_is_not_an_actual_success(self):
        self.assertFalse(subject.accepted_status(self.argv, 'prepared'))
        self.assertFalse(subject.accepted_status(self.argv, 'quality_rejected'))
        self.assertTrue(subject.accepted_status(self.argv, 'ok'))

    def test_foreign_output_and_detached_entry_are_refused(self):
        with self.assertRaises(ValueError):
            subject.check_command(self.argv, self.root/'foreign')
        with self.assertRaises(ValueError):
            subject.check_command(self.argv+['--detach'], self.out)


if __name__ == '__main__':
    unittest.main()
