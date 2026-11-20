"""Admit only the reviewed frozen-fork and serial Alex compiler entries."""
import sys
import unittest

from pt_agent.experiments import autonomous_serial_batch as entry


class PriorityCompileAdmission(unittest.TestCase):
    def command(self, module):
        return [sys.executable, '-B', '-m', entry.PREFIX+module,
                '--out', '/fixture/independent-case']

    def test_reviewed_entries_preserve_default_parent_limits(self):
        for module in ('bert_budget_fork_compile', 'heir_alexnet_host_serial_json'):
            with self.subTest(module=module):
                argv=self.command(module)
                self.assertEqual(entry.check_command(argv, '/fixture/independent-case'), argv)
                self.assertEqual(entry.parent_limits({'argv':argv}), (24,16))

    def test_foreign_modules_interpreters_and_outputs_remain_rejected(self):
        for module in ('bert_budget_fork_compile', 'heir_alexnet_host_serial_json'):
            with self.subTest(module=module):
                argv=self.command(module+'_unreviewed')
                with self.assertRaises(ValueError):entry.check_command(argv, '/fixture/independent-case')
                argv=self.command(module);argv[0]='/foreign/python'
                with self.assertRaises(ValueError):entry.check_command(argv, '/fixture/independent-case')
                with self.assertRaises(ValueError):entry.check_command(self.command(module), '/fixture/other')


if __name__=='__main__':unittest.main()
