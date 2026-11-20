"""Explicit retained-math HEIR scheduler admission; no compiler is run."""
import sys
import unittest
from pt_agent.experiments import autonomous_serial_batch as serial

class HeirNoFusionAdmission(unittest.TestCase):
    def command(self):
        out='/tmp/heir-no-fusion-new-case'
        return out,[sys.executable,'-B','-m','pt_agent.experiments.heir_saved_math_no_linear_fusion',
                    '--source-case','/tmp/original-closed-case','--source-variant','alex-serial-json','--out',out,'--run']

    def test_explicit_entry_keeps_default_parent_resources(self):
        out,argv=self.command()
        self.assertEqual(serial.check_command(argv,out),argv)
        self.assertEqual(serial.parent_limits({'argv':argv}),(24,16))

    def test_expanded_parent_resources_remain_rejected(self):
        _,argv=self.command()
        for key,value in [('outer_address_space_gib',32),('outer_file_gib',64)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                serial.parent_limits({'argv':argv,key:value})

    def test_similarly_named_unknown_entry_is_rejected(self):
        out,argv=self.command();argv[3]+='_unreviewed'
        with self.assertRaises(ValueError):serial.check_command(argv,out)

if __name__=='__main__':unittest.main()
