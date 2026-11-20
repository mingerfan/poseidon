import unittest

from pt_agent.experiments import autonomous_serial_batch as batch


class RSS16RecoveryAdmission(unittest.TestCase):
    def test_own_vgg_recovery_keeps_ordinary_parent(self):
        out = '/tmp/independent-vgg-cold'
        argv = ['python3', '-B', '-m',
                'pt_agent.experiments.dacapo_vgg_fresh_cold_rss16', '--out', out,
                '--model', 'vgg16']
        self.assertEqual(batch.check_command(argv, out), argv)
        self.assertEqual(batch.parent_limits({'argv': argv}), (24, 16))

    def test_resource_recovery_does_not_admit_wider_parent(self):
        argv = ['python3', '-B', '-m',
                'pt_agent.experiments.dacapo_vgg_fresh_cold_rss16', '--out', '/tmp/cold',
                '--model', 'vgg16']
        with self.assertRaises(ValueError):
            batch.parent_limits({'argv': argv, 'outer_address_space_gib': 32})


if __name__ == '__main__':
    unittest.main()
