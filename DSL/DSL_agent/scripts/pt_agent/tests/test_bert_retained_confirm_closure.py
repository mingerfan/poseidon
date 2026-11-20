"""New metadata-only closure regression; no original tests or numeric stages."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from pt_agent.experiments import bert_retained_budget_confirm as entry


class ClosureTests(unittest.TestCase):
    def test_completion_reserve_result_growth_and_actual_terminal_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out=Path(tmp))
            with mock.patch.object(entry,'artifact_bytes',return_value=entry.CASE_BYTES-entry.COMPLETION_RESERVE_BYTES+1):
                with self.assertRaisesRegex(RuntimeError,'artifact allowance'):entry.check_resources(args)
            result=dict(status='ok',outcome='frozen',admission_passed=True,attempted_confirm=1,
                        completed_confirm=1,frozen=True,frozen_accepted=True,**dict.fromkeys(entry.ZERO,0))
            actual=[0];saved=[]
            def save(path,value):
                saved.append(copy.deepcopy(value));actual[0]=entry.CASE_BYTES-entry.COMPLETION_RESERVE_BYTES+1
            with mock.patch.object(entry,'artifact_bytes',side_effect=lambda *a:actual[0]),mock.patch.object(entry,'save',side_effect=save):
                entry.terminal_case(args,result)
            self.assertEqual(result['status'],'error');self.assertTrue(result['frozen'])
            self.assertEqual(result['completed_confirm'],1);self.assertFalse(result['frozen_accepted'])
            self.assertFalse(saved[-1]['metrics']['frozen_accepted'])
            self.assertFalse(result['terminal_case_budget']['within_reserved_case_allowance'])
            self.assertEqual(len(saved),2)


if __name__=='__main__':unittest.main()
