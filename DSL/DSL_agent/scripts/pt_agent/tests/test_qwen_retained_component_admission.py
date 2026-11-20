"""New retained source admission and read-only saved-worker proof reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pt_agent.experiments import autonomous_serial_batch as serial
from pt_agent.experiments import qwen_prefill_compile as frozen
from pt_agent.experiments import qwen_retained_component_replay as entry
from pt_agent.tests import test_qwen_retained_component_source as source_fixtures


class RetainedComponentAdmission(unittest.TestCase):
    def command(self, case, module):
        return ['python3','-B','-m',module,'--out',str(case),
            '--resume-from',str(case.parent/'resume'),'--partial-from',str(case.parent/'partial'),
            '--grouped-from',str(case.parent/'grouped'),'--worker-phase','calibration',
            '--session',str(case/'session'),'--calibration',str(case/'calibration.json'),
            '--source-admission',str(case/'component-source-admission.json')]

    def test_saved_worker_keeps_explicit_inner_proof_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            case=Path(temporary)/'fresh';case.mkdir()
            (case/'command-1.json').write_text(json.dumps(self.command(case,entry.MODULE)))
            selected,args=frozen.source_arguments(case)
            self.assertIs(selected,entry)
            self.assertIsNone(args.out)
            self.assertEqual(args.worker_phase,'calibration')
            self.assertEqual(args.source_admission,case/'component-source-admission.json')
            self.assertEqual((args.memory_gib,args.rss_gib,args.minimum_available_gib),(24,16,1.25))

    def test_unknown_similar_saved_source_remains_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            case=Path(temporary)/'fresh';case.mkdir()
            (case/'command-1.json').write_text(json.dumps(self.command(case,entry.MODULE+'_foreign')))
            with self.assertRaisesRegex(ValueError,'trusted Qwen Agent worker'):
                frozen.source_arguments(case)

    def test_exact_scheduler_name_and_original_parent_limits(self):
        out='/tmp/new-retained-component-admission-only'
        argv=['python3','-B','-m',entry.MODULE,'--out',out,'--through','agent']
        self.assertEqual(serial.check_command(argv,out),argv)
        self.assertEqual(serial.parent_limits(dict(argv=argv)),(24,16))
        with self.assertRaises(ValueError):serial.parent_limits(dict(argv=argv,outer_address_space_gib=32))
        foreign=list(argv);foreign[3]+='_foreign'
        with self.assertRaises(ValueError):serial.check_command(foreign,out)

    def test_saved_calibration_revalidates_existing_proof_without_registry_or_creation(self):
        fixture=source_fixtures.ComponentSourceChecks();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args=fixture.args
        with patch.object(entry.guarded,'source_gate',side_effect=lambda unused:copy.deepcopy(fixture.base)), \
                patch.object(entry.grouped,'source_gate',side_effect=lambda unused:copy.deepcopy(fixture.expected)), \
                patch('pt_agent.experiments.r10_paper_metrics._process',return_value=None):
            outer=entry.source_gate(args)
            case=args.calibration.parent
            (case/'command-1.json').write_text(json.dumps(self.command(case,entry.MODULE)))
            selected,saved=frozen.source_arguments(case)
            original_open=Path.open
            def read_only(path,mode='r',*values,**keywords):
                if path==fixture.registry_path or 'x' in mode:
                    raise AssertionError('Saved worker revalidation must read its existing proof')
                return original_open(path,mode,*values,**keywords)
            with patch.object(Path,'open',read_only):
                actual=selected.source_gate(saved)
        self.assertEqual(actual,outer)
        self.assertTrue(actual['prior_grouped']['registry_checked_by_outer'])
        self.assertFalse(actual['prior_grouped']['strict_math_or_full_cost_completion_claimed'])


if __name__=='__main__':unittest.main()
