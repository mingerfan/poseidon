"""Exact finite parent resources for the trusted two-kernel compiler."""
from pathlib import Path
import unittest
from unittest.mock import patch
from pt_agent.experiments import autonomous_serial_batch as entry


class GenerationKernelSerialChecks(unittest.TestCase):
    def job(self):
        out='/fixture/qwen-kernels'
        command=['/fixture/numpy-venv/bin/python','-B','-m',entry.PREFIX+'qwen_generation_kernel_compile',
                 '--source-case','/fixture/original-source','--out',out,'--memory-gib','24',
                 '--case-gib','128','--seconds','21600','--through','compile','--diagnostic-elide-final-ir']
        return dict(argv=command,out=out,outer_address_space_gib=24,outer_file_gib=64)

    def test_exact_kernel_resources_and_explicit_numpy_host_are_admitted(self):
        job=self.job()
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',Path(job['argv'][0])):
            self.assertEqual(entry.check_command(job['argv'],job['out']),job['argv'])
        self.assertEqual(entry.parent_limits(job),(24,64))

    def test_unknown_python_or_module_is_rejected(self):
        job=self.job()
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',Path('/foreign/python')):
            with self.assertRaises(ValueError):entry.check_command(job['argv'],job['out'])
        job['argv'][3]+='_foreign'
        with self.assertRaises(ValueError):entry.parent_limits(job)

    def test_resource_options_cannot_drift_from_declared_finite_kernel_case(self):
        for flag,value in [('--memory-gib','32'),('--case-gib','256'),('--seconds','43200'),('--through','trace')]:
            with self.subTest(flag=flag):
                job=self.job();job['argv'][job['argv'].index(flag)+1]=value
                with self.assertRaises(ValueError):entry.parent_limits(job)
        job=self.job();job['argv'].remove('--diagnostic-elide-final-ir')
        with self.assertRaises(ValueError):entry.parent_limits(job)
        job=self.job();job['argv']+=['--unknown-resource','90000']
        with self.assertRaises(ValueError):entry.parent_limits(job)


if __name__=='__main__':unittest.main()
