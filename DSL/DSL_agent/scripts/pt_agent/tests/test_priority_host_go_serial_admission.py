"""The explicit host-source Go library bridge uses ordinary serial limits."""
import sys
import unittest
from pt_agent.experiments import autonomous_serial_batch as entry


class HostGoSerialAdmission(unittest.TestCase):
    def argv(self):
        return [sys.executable,'-B','-m',entry.PREFIX+'heir_host_go_compile',
                '--model','vgg16','--source-case','/fixture/accepted-native',
                '--out','/fixture/host-go']

    def test_exact_trusted_entry_preserves_default_limits(self):
        argv=self.argv()
        self.assertEqual(entry.check_command(argv,'/fixture/host-go'),argv)
        self.assertEqual(entry.parent_limits({'argv':argv}),(24,16))

    def test_unreviewed_entry_or_interpreter_or_output_is_rejected(self):
        argv=self.argv();argv[3]+='_unreviewed'
        with self.assertRaises(ValueError):entry.check_command(argv,'/fixture/host-go')
        argv=self.argv();argv[0]='/foreign/python'
        with self.assertRaises(ValueError):entry.check_command(argv,'/fixture/host-go')
        with self.assertRaises(ValueError):entry.check_command(self.argv(),'/fixture/other')


if __name__=='__main__':unittest.main()
