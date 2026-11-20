"""Offline checks of the E10b driver: uniform programs, the descent and its stop rule, and the summary."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pt_agent.experiments import e10b_uniform_recipe as e10b

RULE = '''@fhe.program(slots=16384, refresh="compiler", placement="depth-dp", l_max=5, boot_out_level=5)
def m(img: Cipher[1, 3, 32, 32]):
    v1 = act(v0, fn="relu", method="cheb", deg=127, domain=(-1.0, 2.0))
    v2 = act(v1, fn="relu", method="cheb", deg=127, domain=(-3.0, 4.0))
'''


class Uniform(unittest.TestCase):
    def test_sets_every_degree(self):
        text = e10b.uniform(RULE, 31)
        self.assertEqual(text.count('deg=31'), 2)
        self.assertNotIn('deg=127', text)
        self.assertIn('domain=(-3.0, 4.0)', text)

    def test_rejects_mixed_degrees(self):
        with self.assertRaises(ValueError):
            e10b.uniform(RULE.replace('deg=127, domain=(-3.0', 'deg=63, domain=(-3.0'), 31)

    def test_degrees_below_the_rule_degree(self):
        self.assertEqual(e10b.degrees_below(127), [63, 31, 15, 7])
        self.assertEqual(e10b.degrees_below(63), [31, 15, 7])


def session(frozen, boots):
    return dict(outcome='frozen' if frozen else 'confirm_failed', frozen=frozen, seconds=1.0, api_calls=0,
                driver_error=None, agent_error=None, error=None,
                header='@fhe.program(refresh="agent", placement="program-plan")' if frozen else None,
                trace=[dict(feasible=frozen, feedback=dict(cost=dict(boots_pred=boots)))])


class Descent(unittest.TestCase):
    def run_model(self, outcomes):
        sessions = iter(outcomes)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(e10b, 'rule_program', return_value=Path(tmp)/'rule.ptdsl'), \
                mock.patch.object(e10b.e9b, 'session_child', side_effect=lambda *a, **k: next(sessions)), \
                mock.patch.object(e10b.e9b, 'recorded_calls', return_value=set()), \
                mock.patch.object(e10b.e9b, 'compiled', return_value=dict(complete=False)) as compiled, \
                mock.patch.object(e10b, 'ledger_rows', return_value=[]):
            (Path(tmp)/'rule.ptdsl').write_text(RULE)
            entry = e10b.run_model(Path(tmp), 'mlp', Path(tmp)/'ledger', dict(model='mlp'))
        return entry, compiled

    def test_stops_at_the_first_degree_that_is_not_frozen(self):
        entry, compiled = self.run_model([session(True, 20), session(True, 15), session(False, 12)])
        self.assertEqual([step['degree'] for step in entry['steps']], [63, 31, 15])
        self.assertEqual(entry['best_degree'], 31)
        self.assertTrue(str(compiled.call_args[0][1]).endswith('deg31/session/program.ptdsl'))

    def test_keeps_the_rule_degree_when_nothing_lower_freezes(self):
        entry, compiled = self.run_model([session(False, 20)])
        self.assertEqual(entry['best_degree'], 127)
        compiled.assert_not_called()

    def test_an_api_call_stops_the_round(self):
        bad = session(True, 20)
        bad['api_calls'] = 1
        with self.assertRaises(RuntimeError):
            self.run_model([bad])


class Summary(unittest.TestCase):
    def test_row_compares_with_the_rule_program_and_the_agent(self):
        data = dict(api_before=430, api_after=430, reference=dict(resnet20=dict(rule=209, agent=209, tuner_only=177)),
                    models=dict(resnet20=dict(rule_degree=127, best_degree=31, steps=[
                        dict(degree=63, outcome='frozen', boots_pred=190), dict(degree=31, outcome='frozen', boots_pred=170),
                        dict(degree=15, outcome='confirm_failed', boots_pred=150)],
                        compile=dict(boots=dict(predicted=170, exported=170, compiled=170)),
                        test=dict(top1_test=91.6, torch_top1_report=91.71, n_diverged=1))))
        with tempfile.TemporaryDirectory() as tmp:
            e10b.summarize(Path(tmp), data)
            text = (Path(tmp)/'E10b.md').read_text()
        self.assertIn('| resnet20 | 127 / 209 |', text)
        self.assertIn('| 31 |', text)
        self.assertIn('-18.7%', text)
        self.assertIn('+22.9%', text)


if __name__ == '__main__':
    unittest.main()
