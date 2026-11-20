import unittest

from pt_agent.experiments import autonomous_serial_batch as batch


class OrbitFreshE2EAdmission(unittest.TestCase):
    def test_continuous_orbit_outer_is_admitted(self):
        out = '/tmp/orbit-fresh-e2e'
        argv = ['python3', '-B', '-m',
                'pt_agent.experiments.orbit_fresh_compile_e2e', '--out', out,
                '--model', 'resnet20', '--run']
        self.assertEqual(batch.check_command(argv, out), argv)
        self.assertEqual(batch.parent_limits({'argv': argv}), (24, 16))

    def test_continuous_orbit_does_not_add_expanded_parent(self):
        argv = ['python3', '-B', '-m',
                'pt_agent.experiments.orbit_fresh_compile_e2e', '--out', '/tmp/orbit',
                '--model', 'alexnet', '--run']
        with self.assertRaises(ValueError):
            batch.parent_limits({'argv': argv, 'outer_file_gib': 64})


if __name__ == '__main__':
    unittest.main()
