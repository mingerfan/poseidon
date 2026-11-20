"""The new archive entry retains default serial parent policy."""
import unittest
from pt_agent.experiments import autonomous_serial_batch as batch


class ClosedTraceArchiveSerialAdmission(unittest.TestCase):
    def command(self):
        return ['python3','-B','-m','pt_agent.experiments.closed_dacapo_trace_archive',
                '--source-case','/tmp/closed-original-Alex','--queue-case','/tmp/owner',
                '--out','/tmp/new-archive','--seconds','1800']

    def test_only_explicit_module_and_default_parent(self):
        argv=self.command()
        self.assertEqual(batch.check_command(argv,'/tmp/new-archive'),argv)
        self.assertEqual(batch.parent_limits({'argv':argv}),(24,16))

    def test_no_detach_or_expanded_parent(self):
        argv=self.command()
        with self.assertRaises(ValueError):
            batch.check_command(argv+['--detach'],'/tmp/new-archive')
        with self.assertRaises(ValueError):
            batch.parent_limits({'argv':argv,'outer_address_space_gib':32})


if __name__=='__main__':unittest.main()
