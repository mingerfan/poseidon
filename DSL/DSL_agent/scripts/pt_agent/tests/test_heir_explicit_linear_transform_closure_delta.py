"""Actual fake-execute signal delivery during private HEIR finalization."""
import ast
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import heir_saved_math_explicit_linear_transform as entry
from pt_agent.tests import test_heir_saved_math_no_linear_fusion as fixtures


class ExplicitTargetClosureCases(unittest.TestCase):
    configure = fixtures.SavedMathCases.configure
    persist = fixtures.SavedMathCases.persist
    lease = fixtures.SavedMathCases.lease
    native_text = fixtures.SavedMathCases.native_text
    fake_run = fixtures.SavedMathCases.fake_run

    def setUp(self):
        fixtures.SavedMathCases.setUp(self)
        self.old = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
        def restore():
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in self.old.items():signal.signal(sig,handler)
        self.addCleanup(restore)

    def assert_masked(self):
        for sig in self.old:self.assertEqual(signal.getsignal(sig),signal.SIG_IGN)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0))

    def run_entry(self, thread, persist=None):
        base=entry.base
        with patch.object(base.paths,'RESULTS_DIR',self.root), \
                patch.object(base.serial,'lease_observation',side_effect=self.lease), \
                patch.object(base,'_available_ram_bytes',side_effect=lambda:self.available), \
                patch.object(base.shutil,'disk_usage',return_value=SimpleNamespace(free=20*entry.GIB)), \
                patch.object(base,'run_process',side_effect=self.fake_run), \
                patch.object(base.threading,'Thread',thread), \
                patch.object(base,'persist_case_completion',side_effect=persist or base.persist_case_completion):
            return entry.execute(self.args)

    def joined(self, signum):
        owner=self
        class Joined:
            def __init__(self,**kwargs):self.alive=False
            def start(self):self.alive=True
            def is_alive(self):return self.alive
            def join(self,timeout):
                owner.assertEqual(timeout,3);owner.assert_masked()
                signal.raise_signal(signum)
                self.alive=False
        return Joined

    def assert_durable_success_and_restored(self,result):
        self.assertEqual(result['status'],'ok');self.assertTrue(result['monitor_stopped'])
        self.assertTrue(result['model_compiled']);self.assertTrue(result['model_compiled_accepted'])
        self.assertEqual(result['attempted'],dict(native_compile=1));self.assertEqual(result['completed'],dict(native_compile=1))
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'ok')
        rows=[json.loads(line)for line in self.registry.read_text().splitlines()]
        self.assertEqual(len(rows),2);self.assertTrue(rows[-1]['metrics']['model_compiled_accepted'])
        self.assertEqual(set(rows[-1]),entry.base.host_sources.REGISTRY_FIELDS)
        for sig,handler in self.old.items():self.assertEqual(signal.getsignal(sig),handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.0,0.0))

    def test_SIGTERM_during_real_fake_execute_join_keeps_closure_and_reaps_monitor(self):
        self.assert_durable_success_and_restored(self.run_entry(self.joined(signal.SIGTERM)))

    def test_SIGALRM_during_real_fake_execute_join_keeps_closure_and_cancels_timer(self):
        self.assert_durable_success_and_restored(self.run_entry(self.joined(signal.SIGALRM)))

    def test_signals_during_completion_are_masked_until_persistence_finishes(self):
        original=entry.base.persist_case_completion;seen=[]
        def complete(*args,**kwargs):
            self.assert_masked();seen.append('before')
            for sig in self.old:signal.raise_signal(sig)
            value=original(*args,**kwargs)
            self.assert_masked();seen.append('after')
            return value
        result=self.run_entry(self.joined(signal.SIGTERM),complete)
        self.assertEqual(seen,['before','after']);self.assert_durable_success_and_restored(result)

    def test_unstopped_monitor_preserves_error_completion_and_ignores_late_signals(self):
        owner=self
        class Stuck:
            def __init__(self,**kwargs):pass
            def start(self):pass
            def is_alive(self):return True
            def join(self,timeout):
                owner.assertEqual(timeout,3);owner.assert_masked()
                signal.raise_signal(signal.SIGTERM);signal.raise_signal(signal.SIGALRM)
        result=self.run_entry(Stuck)
        self.assertEqual(result['status'],'error');self.assertFalse(result['monitor_stopped'])
        self.assertTrue(result['model_compiled']);self.assertFalse(result['model_compiled_accepted'])
        self.assertEqual(result['completed'],dict(native_compile=1))
        self.assertIn('Resource monitor did not stop',result['error'])
        self.assertEqual(json.loads((self.out/'completion.json').read_text())['status'],'error')
        self.assertFalse(json.loads(self.registry.read_text().splitlines()[-1])['metrics']['model_compiled_accepted'])
        self.assert_masked()
        for sig in self.old:signal.raise_signal(sig)
        self.assert_masked()

    def test_finally_inverse_AST_preserves_every_original_statement(self):
        tree=entry._closed_execute_tree();before=entry._execute_tree()
        def target(t):
            return next(n for n in ast.walk(t)if isinstance(n,ast.Try)and n.finalbody
                and any(isinstance(x,ast.Assign)and any(isinstance(y,ast.Subscript)
                    and isinstance(y.slice,ast.Constant)and y.slice.value=='monitor_stopped'
                    for y in x.targets)for x in n.finalbody))
        after_final=target(tree);before_final=target(before)
        self.assertEqual(len(after_final.finalbody),len(before_final.finalbody)+1)
        retained=after_final.finalbody[2:-1]
        original=[n for n in before_final.finalbody if not (
            isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)
            and n.value.func.attr=='setitimer')and not isinstance(n,ast.For)]
        self.assertEqual([ast.dump(n,include_attributes=False)for n in retained],
            [ast.dump(n,include_attributes=False)for n in original])
        after_final.finalbody=before_final.finalbody
        self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(before,include_attributes=False))


if __name__=='__main__':unittest.main()
