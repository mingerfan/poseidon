"""E11 core changes: changed lines between programs, the tuner focus and the configurable stop rule (inside Nix)."""
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.dsl.parser import parse
from pt_agent.planners import agent, tuner

SOURCE = '''@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")
def toy(v: Cipher[1, 2]):
    x = pack(v, layout="flat")
    h = linear(x, w=W["identity"], algo="bsgs")
    a = act(h, fn="relu", method="cheb", deg=127, domain=(-8.0, 8.0))
    g = linear(a, w=W["identity"], algo="bsgs")
    b = act(g, fn="relu", method="cheb", deg=127, domain=(-4.0, 4.0))
    out = linear(b, w=W["identity"], algo="bsgs")
    return out
'''
LINE_A, LINE_B = 5, 7


class ChangedLines(unittest.TestCase):
    def test_identical_programs_change_nothing(self):
        self.assertEqual(agent.changed_lines(SOURCE, SOURCE), set())

    def test_one_changed_attribute_marks_its_line(self):
        changed = SOURCE.replace('deg=127, domain=(-4.0', 'deg=31, domain=(-4.0')
        self.assertEqual(agent.changed_lines(changed, SOURCE), {LINE_B})

    def test_renamed_values_are_no_change(self):
        renamed = SOURCE.replace('    b = act', '    c = act').replace('linear(b,', 'linear(c,')
        self.assertEqual(agent.changed_lines(renamed, SOURCE), set())

    def test_a_renamed_and_edited_operation_marks_only_its_line(self):
        edited = SOURCE.replace('    b = act', '    c = act').replace('linear(b,', 'linear(c,').replace(
            'deg=127, domain=(-4.0', 'deg=31, domain=(-4.0')
        self.assertEqual(agent.changed_lines(edited, SOURCE), {LINE_B})


class NumericLock(unittest.TestCase):
    TUNED = SOURCE.replace('deg=127, domain=(-8.0, 8.0)', 'deg=15, domain=(-9.5, 9.5)')

    def degrees(self, source):
        return [(op.attrs['method'], op.attrs['deg'], op.attrs['domain']) for op in parse(source).ops if op.kind == 'act']

    def test_numeric_edits_take_the_tuned_values(self):
        reply = SOURCE.replace('deg=127, domain=(-4.0, 4.0)', 'deg=7, domain=(-1.0, 1.0)')
        locked, count = agent.lock_numeric(reply, self.TUNED)
        self.assertEqual(count, 2)
        self.assertEqual(self.degrees(locked), self.degrees(self.TUNED))

    def test_a_method_change_keeps_the_llm_settings(self):
        reply = self.TUNED.replace('method="cheb", deg=127, domain=(-4.0, 4.0)', 'method="remez", deg=31, domain=(-5.0, 5.0)')
        locked, count = agent.lock_numeric(reply, self.TUNED)
        self.assertEqual(count, 0)
        self.assertEqual(self.degrees(locked)[1], ('remez', 31, (-5.0, 5.0)))

    def test_renamed_values_still_lock(self):
        reply = SOURCE.replace('    a = act', '    z = act').replace('linear(a,', 'linear(z,')
        locked, count = agent.lock_numeric(reply, self.TUNED)
        self.assertEqual(count, 1)
        self.assertEqual(self.degrees(locked)[0], ('cheb', 15, (-9.5, 9.5)))

    def test_role_prefixed_settings_are_numeric(self):
        self.assertTrue(agent._numeric('rsqrt_domain'))
        self.assertTrue(agent._numeric('step_inv_deg'))
        self.assertTrue(agent._numeric('inv_budget'))
        self.assertFalse(agent._numeric('rsqrt_method'))
        self.assertFalse(agent._numeric('steps'))


class InFocus(unittest.TestCase):
    def test_lines_and_call_lines(self):
        op = SimpleNamespace(lineno=20, call_lines=(5, 9))
        self.assertTrue(tuner.in_focus(op, None))
        self.assertTrue(tuner.in_focus(op, {20}))
        self.assertTrue(tuner.in_focus(op, {9}))
        self.assertFalse(tuner.in_focus(op, {6, 7}))


def fake_cost(program, weights, profile, estimator, cache=True):
    return dict(latency_pred_us=float(sum(op.attrs['deg'] for op in program.ops if op.kind == 'act')),
                boot_nodes=[], hotspots=[], boots_pred=0)


def fake_validate(program, ctx, n=None, **kwargs):
    rows = [dict(obs_min=-1.0, obs_max=1.0) for op in program.ops if op.kind == 'act']
    return dict(feasible=True, acc_drop_pp=0.0, diverged=False, agree=1, total=1, max_u=0.5, top1=1.0,
                exact_top1=1.0, rows=rows, worst_acts=[])


class FocusedTune(unittest.TestCase):
    def tune(self, focus):
        ctx = SimpleNamespace(prompt_version='r8-v4', weights={}, profile=None, placement=None)
        with mock.patch.object(tuner, 'cached_estimate', side_effect=fake_cost), \
                mock.patch.object(tuner, 'validate', side_effect=fake_validate), \
                mock.patch.object(tuner, '_exact_check', return_value=dict(ok=True)):
            result = tuner.tune(parse(SOURCE), ctx, max_sweeps=8, focus=focus)
        return {op.lineno: op.attrs['deg'] for op in result.program.ops if op.kind == 'act'}

    def test_focus_moves_only_the_focus_line(self):
        self.assertEqual(self.tune({LINE_A}), {LINE_A: 7, LINE_B: 127})

    def test_no_focus_moves_every_point(self):
        self.assertEqual(self.tune(None), {LINE_A: 7, LINE_B: 7})

    def test_empty_focus_moves_nothing(self):
        self.assertEqual(self.tune(set()), {LINE_A: 127, LINE_B: 127})


class StopRule(unittest.TestCase):
    def test_limits_are_positive(self):
        with self.assertRaises(ValueError):
            agent._run_agent_once(SimpleNamespace(), None, trace_path='unused', stale_limit=0)
        with self.assertRaises(ValueError):
            agent._run_agent_once(SimpleNamespace(), None, trace_path='unused', tune_sweeps=0)


if __name__ == '__main__':
    unittest.main()
