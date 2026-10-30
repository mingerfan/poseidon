"""U4 §7.9: complete legacy Function equality and diagnostic-only timing."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from pt_agent import toolchain
from pt_agent.manage import RecordingBackend, manage, check, emit
from pt_agent.manage.greedy import timing_log
from pt_agent.tests._artifacts import COMPILER, require
from pt_agent.tests._u4_legacy_manage import legacy_manage, compare_functions
from pt_agent.tests.test_s1_manage import record_case


@require(COMPILER)
class ManagementPerformanceTests(unittest.TestCase):
    def same(self, recorded, name):
        profile=toolchain.management_profile()
        old=legacy_manage(recorded,profile,name=name)
        new=manage(recorded,profile,name=name)
        self.assertEqual(compare_functions(old,new),dict(equal=True,nodes=len(new.nodes),
            boots=sum(n.kind=='bootstrap' for n in new.nodes),cuts=len(new.cuts),outputs=len(new.outputs)))
        check(new)
        return old,new

    def test_three_s1_small_programs(self):
        for name in ('chain','residual','constants'):
            with self.subTest(name=name):
                recorded,_=record_case(name)
                self.same(recorded,name.replace('-','_'))

    def test_fixed_seed_shared_graphs_and_multiple_outputs(self):
        rng=np.random.default_rng(20261027)
        for trial in range(12):
            b=RecordingBackend(1024,locations={1:SimpleNamespace(id=1,lineno=3)})
            b.op_context=1
            x,z=b.input(0),b.input(1)
            shared=x+z
            y=shared
            for index in range(12+trial):
                y=y*y
                if index%3==1:
                    y=y+shared.rotate(int(rng.integers(1,8)))
            b.output([y,y+shared,shared])
            with self.subTest(trial=trial):
                old,new=self.same(b,'random_'+str(trial))
                self.assertGreaterEqual(sum(n.kind=='bootstrap' for n in new.nodes),2)

    def test_timings_are_separate_and_emit_text_unchanged(self):
        recorded,_=record_case('constants')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with timing_log(root/'trace-stages.json'):
                old,new=self.same(recorded,'timed')
                before=emit(old,root/'old.mlir').read_text()
                after=emit(new,root/'new.mlir').read_text()
            self.assertEqual(before,after)
            rows=json.loads((root/'trace-stages.json').read_text())['stages']
            self.assertTrue({'schedule','depth_dp','main_loop','bootstrap_inputs',
                             'early_modswitch','check','managed_emit'} <= {r['stage'] for r in rows})
            self.assertTrue(all(r['complete'] and r['seconds']>=0 for r in rows))

    def test_standalone_emit_always_writes_segments(self):
        recorded,_=record_case('chain')
        function=manage(recorded,toolchain.management_profile(),name='standalone')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            emit(function,root/'managed.mlir')
            rows=json.loads((root/'trace-stages.json').read_text())['stages']
            self.assertIn('bootstrap_inputs',[r['stage'] for r in rows])
            self.assertTrue(all(r['complete'] for r in rows))


if __name__=='__main__':unittest.main()
