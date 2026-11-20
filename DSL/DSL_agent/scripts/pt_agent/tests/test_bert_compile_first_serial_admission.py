"""Admit the explicit trusted compile-first module with normal parent limits."""
from pathlib import Path
import unittest
from pt_agent.experiments import autonomous_serial_batch as entry


class CompileFirstAdmissionChecks(unittest.TestCase):
    def test_exact_own_module_uses_normal_finite_parent_limits(self):
        out=Path('/fixture-results/bert-compile-first')
        command=['python3','-B','-m',entry.PREFIX+'bert_compile_first_deferred_report','--out',str(out)]
        self.assertEqual(entry.check_command(command,out),command)
        self.assertEqual(entry.parent_limits(dict(argv=command)),(24,16))

    def test_detach_foreign_module_and_expanded_parent_are_rejected(self):
        out=Path('/fixture-results/bert-compile-first')
        command=['python3','-B','-m',entry.PREFIX+'bert_compile_first_deferred_report','--out',str(out)]
        with self.assertRaises(ValueError):entry.check_command(command+['--detach'],out)
        bad=command.copy();bad[3]+='_foreign'
        with self.assertRaises(ValueError):entry.check_command(bad,out)
        with self.assertRaises(ValueError):entry.parent_limits(dict(argv=command,outer_address_space_gib=32,outer_file_gib=64))


if __name__=='__main__':unittest.main()
