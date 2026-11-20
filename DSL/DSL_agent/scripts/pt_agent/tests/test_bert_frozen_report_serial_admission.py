import unittest

from pt_agent.experiments import autonomous_serial_batch as batch


class FrozenReportSerialAdmission(unittest.TestCase):
    def test_isolated_report_keeps_ordinary_parent_limits(self):
        out = '/tmp/independent-frozen-report'
        argv = ['python3', '-B', '-m',
                'pt_agent.experiments.bert_frozen_fork_report', '--out', out]
        self.assertEqual(batch.check_command(argv, out), argv)
        self.assertEqual(batch.parent_limits({'argv': argv}), (24, 16))

    def test_isolated_report_output_stays_explicit(self):
        argv = ['python3', '-B', '-m',
                'pt_agent.experiments.bert_frozen_fork_report', '--out', '/tmp/first']
        with self.assertRaises(ValueError):
            batch.check_command(argv, '/tmp/second')


if __name__ == '__main__':
    unittest.main()
