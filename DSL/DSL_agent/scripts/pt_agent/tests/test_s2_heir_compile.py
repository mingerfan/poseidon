import tempfile
from pathlib import Path
import unittest

from pt_agent.experiments.s2_heir_compile import inventory, outcome, parameter_literal, parameter_pipeline


class NativeAdmissionTests(unittest.TestCase):
    def test_changes_are_parameter_only(self):
        original = '--pass-pipeline=builtin.module(early-bootstrap-placement{bootstrap-waterline=0 level-budget=10},secret-insert-mgmt-ckks{after-mul=true bootstrap-waterline=0 level-budget=10},generate-param-ckks{first-mod-bits=55 scaling-mod-bits=45 parameter-file=/old/config.json},scheme-to-lattigo)'
        modified = parameter_pipeline(original, Path('/new/config.json'))
        restored = modified.replace('level-budget=5', 'level-budget=10').replace('bootstrap-waterline=1', 'bootstrap-waterline=0').replace('first-mod-bits=52', 'first-mod-bits=55').replace('scaling-mod-bits=43', 'scaling-mod-bits=45').replace('/new/config.json', '/old/config.json')
        self.assertEqual(restored, original.removeprefix('--pass-pipeline='))

    def test_unexpected_pipeline_rejected(self):
        with self.assertRaises(ValueError):
            parameter_pipeline('builtin.module(canonicalize)', Path('/new/config.json'))

    def test_native_modulus_group_admission(self):
        p = dict(logN=16, log_default_scale=43, Q=[2**52-1] + [2**43-1]*8, P=[2**60-1]*3)
        accepted = parameter_literal(dict(parameters=[p]))
        self.assertEqual((accepted['logN'], accepted['logDefaultScale']), (16, 43))
        self.assertEqual(accepted['Q'], p['Q'])
        with self.assertRaises(ValueError):
            parameter_literal(dict(parameters=[dict(p, logN=15)]))

    def test_failure_classifications_remain_distinct(self):
        self.assertEqual(outcome(dict(exit_code=-15, timed_out=False), {'rss': 15}), 'resource_limit')
        self.assertEqual(outcome(dict(exit_code=-15, timed_out=True)), 'timeout')
        self.assertEqual(outcome(dict(exit_code=-11, timed_out=False)), 'compiler_crash')
        self.assertEqual(outcome(dict(exit_code=1, timed_out=False)), 'compiler_crash')
        self.assertEqual(outcome(dict(exit_code=0, timed_out=False), driver_error='counting failed'), 'driver_error')
        self.assertEqual(outcome(None, driver_error='missing adapter'), 'driver_error')
        self.assertEqual(outcome(dict(exit_code=0, timed_out=False)), 'ok')

    def test_straight_line_inventory_proves_single_invocation(self):
        source = 'module {\n  func.func @example__preprocessed() {\n    %0 = lattigo.ckks.bootstrap %x\n    return\n  }\n  func.func @example() {\n    call @example__preprocessed()\n    return\n  }\n}\n'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'fixture.mlir'
            p.write_text(source)
            value = inventory(p, 'example')
            self.assertEqual(value['boots_compiled'], 1)
            p.write_text(source.replace('    %0 = ', '    affine.for %i = 0 to 4 {\n    %0 = '))
            value = inventory(p, 'example')
            self.assertEqual(value['boot_sites_static'], 1)
            self.assertIsNone(value['boots_compiled'])
            self.assertEqual(value['boot_kind'], 'static')


if __name__ == '__main__':
    unittest.main()
