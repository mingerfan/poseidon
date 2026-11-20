"""Finite queue admission only: no model or compiler imports or execution."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from pt_agent.experiments import autonomous_serial_batch as entry


class RemainingPrefillSerialAdmission(unittest.TestCase):
    def setUp(self):
        self.job=dict(id='remaining-original-prefill',out='/tmp/results/remaining',
            outer_address_space_gib=24,outer_file_gib=64,seconds=43320,
            argv=['python3','-B','-m',entry.PREFIX+'qwen_remaining_prefill_compile',
                '--out','/tmp/results/remaining','--source-case','/tmp/results/frozen',
                '--probe-case','/tmp/results/probe','--intake-python','/tmp/intake-python',
                '--memory-gib','24','--case-gib','128','--seconds','43200',
                '--through','compile','--diagnostic-elide-final-ir'])

    def test_exact_master_has_explicit_parent_and_finite_guarded_argv(self):
        self.assertEqual(entry.check_command(self.job['argv'],self.job['out']),self.job['argv'])
        self.assertEqual(entry.parent_limits(self.job),(24,64))

    def test_changed_resources_or_incomplete_phase_are_rejected(self):
        for flag,value in (('--memory-gib','32'),('--case-gib','256'),
                ('--seconds','86400'),('--through','emit')):
            job=deepcopy(self.job);job['argv'][job['argv'].index(flag)+1]=value
            with self.subTest(flag=flag),self.assertRaises(ValueError):entry.parent_limits(job)
        for memory,file_size in ((32,64),(24,128),(4,64)):
            job=deepcopy(self.job);job.update(outer_address_space_gib=memory,outer_file_gib=file_size)
            with self.subTest(parent=(memory,file_size)),self.assertRaises(ValueError):entry.parent_limits(job)

    def test_internal_workers_abbreviation_equals_and_duplicate_limits_reject(self):
        for extra in (['--component','head'],['--worker-phase','compile'],
                ['--memo','24'],['--seconds=43200'],['--memory-gib','24']):
            job=deepcopy(self.job);job['argv'].extend(extra)
            with self.subTest(extra=extra),self.assertRaises(ValueError):entry.parent_limits(job)
        job=deepcopy(self.job);job['argv'].remove('--diagnostic-elide-final-ir')
        with self.assertRaises(ValueError):entry.parent_limits(job)

    def test_allowance_remains_private_to_exact_module(self):
        job=deepcopy(self.job);job['argv'][3]=entry.PREFIX+'heir_common_go_compile'
        with self.assertRaises(ValueError):entry.parent_limits(job)
        job=deepcopy(self.job);job['argv'][3]='foreign.qwen_remaining_prefill_compile'
        with self.assertRaises(ValueError):entry.check_command(job['argv'],job['out'])

    def test_configured_existing_numpy_outer_is_private_to_two_qwen_modules(self):
        configured=Path('/tmp/existing-paper-data/bin/python')
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',configured):
            for module in ('qwen_remaining_prefill_compile','qwen_layer_native_probe'):
                job=deepcopy(self.job);job['argv'][0]=str(configured);job['argv'][3]=entry.PREFIX+module
                self.assertIs(entry.check_command(job['argv'],job['out']),job['argv'])
            job['argv'][3]=entry.PREFIX+'heir_common_go_compile'
            with self.assertRaises(ValueError):entry.check_command(job['argv'],job['out'])
            job['argv'][3]=entry.PREFIX+'qwen_remaining_prefill_compile';job['argv'][0]='/tmp/other/bin/python'
            with self.assertRaises(ValueError):entry.check_command(job['argv'],job['out'])
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',None):
            job['argv'][0]=str(configured)
            with self.assertRaises(ValueError):entry.check_command(job['argv'],job['out'])


if __name__=='__main__':unittest.main()
