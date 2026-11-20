"""Finite scheduling admission; fixtures never start an experimental process."""
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments import autonomous_serial_batch as subject


class FreshColdSerialAdmission(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / 'fresh-alex'
        self.argv = ['python3', '-B', '-m', subject.PREFIX+'dacapo_fresh_cold',
            '--out', str(self.out), '--model', 'alexnet', '--memory-gib', '12',
            '--file-gib', '16', '--case-gib', '32', '--planning-gib', '24',
            '--waterline', '40', '--trace-seconds', '1800', '--compile-seconds',
            '7200', '--diagnostic-elide-final-ir']

    def job(self):
        return dict(id='original-alex-cold', argv=list(self.argv), out=str(self.out),
                    seconds=9960, outer_address_space_gib=32, outer_file_gib=16)

    def test_original_alex_exact_policy_admits_parent32_file16(self):
        job = self.job()
        self.assertEqual(subject.check_command(job['argv'], self.out), self.argv)
        self.assertEqual(subject.parent_limits(job), (32, 16))

    def test_each_explicit_value_is_required_for_expansion(self):
        for flag in ('--model', '--memory-gib', '--file-gib', '--case-gib',
                     '--planning-gib', '--waterline', '--trace-seconds', '--compile-seconds'):
            for replacement in (None, 'unexpected'):
                with self.subTest(flag=flag, replacement=replacement):
                    job = self.job(); index = job['argv'].index(flag)
                    if replacement is None:
                        del job['argv'][index:index+2]
                    else:
                        job['argv'][index+1] = replacement
                    with self.assertRaises(ValueError):
                        subject.parent_limits(job)

    def test_repeated_or_missing_diagnostic_flag_refuses_expansion(self):
        for diagnostic_count in (0, 2):
            job = self.job()
            job['argv'].remove('--diagnostic-elide-final-ir')
            job['argv'] += ['--diagnostic-elide-final-ir']*diagnostic_count
            with self.assertRaises(ValueError):
                subject.parent_limits(job)

    def test_repeated_resource_option_refuses_expansion(self):
        job = self.job(); job['argv'] += ['--memory-gib', '12']
        with self.assertRaises(ValueError):
            subject.parent_limits(job)

    def test_equals_form_cannot_override_or_duplicate_the_declared_policy(self):
        for flag in ('--model', '--memory-gib', '--file-gib', '--case-gib',
                     '--planning-gib', '--waterline', '--trace-seconds', '--compile-seconds'):
            with self.subTest(flag=flag):
                job = self.job(); index = job['argv'].index(flag)
                job['argv'].append(flag+'='+job['argv'][index+1])
                with self.assertRaises(ValueError):
                    subject.parent_limits(job)
        job = self.job(); job['argv'].append('--model=vgg16')
        with self.assertRaises(ValueError):
            subject.parent_limits(job)

    def test_abbreviated_or_undeclared_options_cannot_override_cold_policy(self):
        for extra in (['--mode', 'vgg16'], ['--mod=vgg16'],
                      ['--worker-compile'], ['--compiler', '/fixture/compiler'], ['--']):
            with self.subTest(extra=extra):
                job = self.job(); job['argv'] += extra
                with self.assertRaises(ValueError):
                    subject.parent_limits(job)

    def test_vgg_uses_default_parent_and_cannot_inherit_alex32(self):
        job = self.job(); job['argv'][job['argv'].index('--model')+1] = 'vgg16'
        with self.assertRaises(ValueError):
            subject.parent_limits(job)
        del job['outer_address_space_gib']; del job['outer_file_gib']
        self.assertEqual(subject.parent_limits(job), (24, 16))

    def test_other_parent_pairs_and_detached_or_foreign_commands_refused(self):
        for pair in ((32, 64), (48, 16), (32.0, 16), (True, 16)):
            with self.subTest(pair=pair):
                job = self.job()
                job.update(outer_address_space_gib=pair[0], outer_file_gib=pair[1])
                with self.assertRaises(ValueError):
                    subject.parent_limits(job)
        with self.assertRaises(ValueError):
            subject.check_command(self.argv+['--detach'], self.out)
        with self.assertRaises(ValueError):
            subject.check_command(self.argv, self.root/'foreign')


if __name__ == '__main__':
    unittest.main()
