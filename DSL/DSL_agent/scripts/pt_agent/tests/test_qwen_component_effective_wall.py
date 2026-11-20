"""Actual private supervisor globals and generated launcher use the same wall."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from pt_agent.experiments import qwen_retained_component_replay as entry


class EffectiveComponentWallChecks(unittest.TestCase):
    def test_generated_launcher_and_inner_total_case_budget(self):
        args=entry.parser().parse_args(['--resume-from','/tmp/retained-wall-resume',
            '--partial-from','/tmp/retained-wall-partial','--grouped-from','/tmp/retained-wall-grouped'])
        for name in ('model','model_info','inputs','source_rows','metadata_template','frontend','profile',
                'linked_result','coefficients','calibration','session','codegen','ledger','allowance','intake_python'):
            setattr(args,name,Path('/tmp/retained-wall')/name)
        argv=entry.phase_command(args,'step',Path('/tmp/retained-wall/step'))
        outer=argv[:argv.index('--')]
        self.assertEqual(int(outer[outer.index('--seconds')+1]),9060)
        inner=entry.parser().parse_args(argv[argv.index(entry.MODULE)+1:])
        self.assertEqual(inner.seconds,10800)
        self.assertEqual(inner.memory_gib,24.)
        self.assertEqual(inner.rss_gib,16.)
        self.assertEqual(inner.minimum_available_gib,1.25)
        calibration=entry.phase_command(args,'calibration',Path('/tmp/retained-wall/cal'))
        self.assertEqual(int(calibration[calibration.index('--seconds')+1]),960)

    def test_actual_delegated_execute_uses_same_private_phase_table(self):
        namespace=entry._namespace()
        execute=namespace['execute']
        delegated=next(cell.cell_contents for cell in execute.__closure__
            if callable(cell.cell_contents)and getattr(cell.cell_contents,'__name__',None)=='execute')
        self.assertIs(delegated.__globals__,namespace)
        self.assertEqual(delegated.__globals__['PHASE_SECONDS']['step'],9000)
        self.assertEqual(delegated.__globals__['PHASE_SECONDS']['calibration'],900)
        original=entry.guarded.bridge._namespace()
        self.assertEqual(original['PHASE_SECONDS']['step'],3600)
        self.assertEqual(entry.guarded.original.PHASE_SECONDS['step'],3600)


if __name__=='__main__':unittest.main()
