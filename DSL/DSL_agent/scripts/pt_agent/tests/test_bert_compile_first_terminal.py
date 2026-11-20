"""Terminal acceptance matches finite resource policy, using no model data."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pt_agent.experiments import bert_compile_first_deferred_report as entry


class TerminalPolicyChecks(unittest.TestCase):
    def invoke(self,*,failures=(),expired=False,budget_error=False,available=None,status='ok'):
        args=SimpleNamespace(out=Path('/fixture-outer'),compile_out=Path('/fixture-native'))
        result=dict(status=status,error=None,termination=None,math_passed=True,compile_completed=True,
                    completed=dict(math=1,compile=1))
        with (patch.object(entry,'_budget',side_effect=RuntimeError('saved case reserve') if budget_error else None),
              patch.object(entry.time,'monotonic',return_value=100 if expired else 1),
              patch.object(entry.original,'_available_ram_bytes',return_value=50*entry.GIB if available is None else available)):
            entry.terminal_policy(args,result,100,list(failures))
        return result

    def test_clean_actual_completion_is_accepted(self):
        result=self.invoke();self.assertTrue(result['compile_accepted'])
        self.assertTrue(result['compile_completed']);self.assertEqual(result['terminal_rejections'],[])

    def test_late_resource_or_monitor_failure_revokes_acceptance_preserves_completed_fact(self):
        for kwargs in (dict(expired=True),dict(budget_error=True),dict(available=entry.GIB),
                       dict(failures=[dict(error='watcher failure after child')])):
            with self.subTest(kwargs=kwargs):
                result=self.invoke(**kwargs);self.assertEqual(result['status'],'error')
                self.assertFalse(result['compile_accepted']);self.assertTrue(result['compile_completed'])
                self.assertTrue(result['terminal_rejections']);self.assertEqual(result['termination']['cause'],'terminal_policy')

    def test_existing_error_stays_unaccepted(self):
        result=self.invoke(status='error');self.assertFalse(result['compile_accepted'])
        self.assertEqual(result['status'],'error')


if __name__=='__main__':unittest.main()
