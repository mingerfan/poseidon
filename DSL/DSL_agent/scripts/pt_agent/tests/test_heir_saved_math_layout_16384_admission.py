"""Exact layout candidate scheduler admission with original parent limits."""
import sys
import unittest

from pt_agent.experiments import autonomous_serial_batch as serial


class HeirLayout16384Admission(unittest.TestCase):
    def command(self):
        out='/tmp/heir-layout-16384-new-case'
        return out,[sys.executable,'-B','-m','pt_agent.experiments.heir_saved_math_layout_16384',
            '--source-case','/tmp/original-closed-host-case','--source-variant','alex-serial-json',
            '--out',out,'--run']

    def test_exact_entry_keeps_original_parent_resources(self):
        out,argv=self.command()
        self.assertEqual(serial.check_command(argv,out),argv)
        self.assertEqual(serial.parent_limits({'argv':argv}),(24,16))

    def test_expanded_parent_resources_remain_rejected(self):
        _,argv=self.command()
        for limits in ({'outer_address_space_gib':32},{'outer_file_gib':64},
                {'outer_address_space_gib':32,'outer_file_gib':64}):
            with self.subTest(limits=limits),self.assertRaises(ValueError):
                serial.parent_limits(dict(argv=argv,**limits))

    def test_unknown_similar_entry_is_rejected(self):
        out,argv=self.command()
        for suffix in ('_unreviewed','_16384','_native'):
            unknown=list(argv);unknown[3]+=suffix
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):
                serial.check_command(unknown,out)


if __name__=='__main__':unittest.main()
