"""Finite confirm-only recovery can enter the existing serial scheduler."""
import sys
import unittest
from pt_agent.experiments import autonomous_serial_batch as entry


class RetainedBudgetSerialAdmission(unittest.TestCase):
    def command(self):
        return [sys.executable, '-B', '-m', entry.PREFIX+'bert_retained_budget_confirm',
                '--source-case', '/fixture/original', '--continuation-case', '/fixture/closed',
                '--boundary-snapshot', '/fixture/closed/boundary.json', '--out', '/fixture/recovery']

    def test_reviewed_confirm_entry_with_default_parent_resources_is_admitted(self):
        argv=self.command()
        self.assertEqual(entry.check_command(argv, '/fixture/recovery'), argv)
        self.assertEqual(entry.parent_limits({'argv':argv}), (24,16))

    def test_foreign_interpreter_or_module_or_output_is_rejected(self):
        argv=self.command();argv[0]='/foreign/python'
        with self.assertRaises(ValueError):entry.check_command(argv, '/fixture/recovery')
        argv=self.command();argv[3]+='_unreviewed'
        with self.assertRaises(ValueError):entry.check_command(argv, '/fixture/recovery')
        with self.assertRaises(ValueError):entry.check_command(self.command(), '/fixture/other')


if __name__=='__main__':unittest.main()
