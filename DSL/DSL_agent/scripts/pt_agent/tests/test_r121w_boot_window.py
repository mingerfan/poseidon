"""Bootstrap segment rule: DaCapo 39c1fb2 (default) and the Qwen25 R121 compiler change."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pt_agent import toolchain
from pt_agent.backends import DagBackend
from pt_agent.cost import PROFILES, load_latency
from pt_agent.placement import PlacementError, depth_dp


def chain(length=8):
    dag = DagBackend(16)
    value = dag.input(0)
    for _ in range(length):
        value = value*.5
    dag.output([value])
    return dag


def place(dag, levels):
    config = PROFILES['qwen48']
    return depth_dp(dag, last_boot_level=config['boot_out_level']-config['boot_in_level'],
                    boot_in_level=config['boot_in_level'], boot_out_level=config['boot_out_level'],
                    latency_table=load_latency('qwen48'), boot_normalization_levels=levels)


class SegmentTests(unittest.TestCase):
    def test_default_rule_reserves_one_level(self):
        placement = place(chain(), 1)
        self.assertEqual((placement.cuts, placement.barriers, placement.boots, placement.max_depth),
                         ([4, 7], 2, 1, 8))

    def test_r121_rule_uses_the_full_segment(self):
        placement = place(chain(), 0)
        self.assertEqual((placement.cuts, placement.barriers, placement.boots, placement.max_depth),
                         ([4], 1, 1, 8))

    def test_r121_rule_never_places_more_boots(self):
        for length in range(1, 21):
            with self.subTest(length=length):
                self.assertLessEqual(place(chain(length), 0).boots, place(chain(length), 1).boots)

    def test_segment_must_leave_a_level(self):
        with self.assertRaises(PlacementError):
            place(chain(), 2)


class MarkerTests(unittest.TestCase):
    def build(self, rule=None):
        directory = Path(tempfile.mkdtemp())
        if rule is not None:
            (directory/toolchain.BOOT_WINDOW_MARKER).write_text(json.dumps(dict(rule=rule)))
        return mock.patch.object(toolchain, 'HECATE_BUILD', directory)

    def test_build_without_marker_keeps_the_default_rule(self):
        with self.build():
            self.assertEqual(toolchain.boot_window_rule(), 'dacapo-39c1fb2')
            self.assertEqual(toolchain.boot_normalization_levels(43, 43), 1)

    def test_r121_marker_drops_the_level_only_when_factor_equals_waterline(self):
        with self.build('r121'):
            self.assertEqual(toolchain.boot_window_rule(), 'r121')
            self.assertEqual(toolchain.boot_normalization_levels(43, 43), 0)
            self.assertEqual(toolchain.boot_normalization_levels(43, 45), 1)

    def test_unknown_marker_is_rejected(self):
        with self.build('other'), self.assertRaises(ValueError):
            toolchain.boot_window_rule()

    def test_profile_follows_the_configured_build(self):
        expected = 0 if toolchain.boot_window_rule() == 'r121' else 1
        self.assertEqual(PROFILES['qwen48']['boot_normalization_levels'], expected)
        self.assertEqual(toolchain.management_profile()['boot_window'], toolchain.boot_window_rule())

    def test_rule_is_evaluated_against_each_profile(self):
        from pt_agent.placement import boot_normalization_levels
        self.assertEqual(boot_normalization_levels('r121', 43, 43), 0)
        self.assertEqual(boot_normalization_levels('r121', 51, 43), 1)
        self.assertEqual(boot_normalization_levels('dacapo-39c1fb2', 43, 43), 1)
        with self.assertRaises(PlacementError):
            boot_normalization_levels('other', 43, 43)


if __name__ == '__main__':
    unittest.main()
