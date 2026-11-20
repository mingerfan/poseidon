"""Exact new Go source admission; no gate or tool is executed."""
import sys
import unittest
from pt_agent.experiments import autonomous_serial_batch as serial

class HeirNoFusionGoAdmission(unittest.TestCase):
    def command(self):
        out='/tmp/heir-no-fusion-go-new-case'
        return out,[sys.executable,'-B','-m','pt_agent.experiments.heir_no_fusion_go_compile',
            '--model','alexnet','--source-case','/tmp/closed-accepted-native',
            '--out',out,'--backend-case','/tmp/backend','--cache-source','/tmp/cache',
            '--translate','/tmp/heir-translate','--go','/tmp/go',
            '--source-variant','alex-serial-json','--run']

    def test_exact_new_module_is_accepted_with_original_parent_limits(self):
        out,argv=self.command()
        self.assertEqual(serial.check_command(argv,out),argv)
        self.assertEqual(serial.parent_limits({'argv':argv}),(24,16))

    def test_parent_expansion_is_rejected(self):
        _,argv=self.command()
        with self.assertRaises(ValueError):serial.parent_limits({'argv':argv,'outer_address_space_gib':32})
        with self.assertRaises(ValueError):serial.parent_limits({'argv':argv,'outer_file_gib':64})

    def test_unknown_similar_module_is_rejected(self):
        out,argv=self.command();argv[3]+='_unreviewed'
        with self.assertRaises(ValueError):serial.check_command(argv,out)

if __name__=='__main__':unittest.main()
