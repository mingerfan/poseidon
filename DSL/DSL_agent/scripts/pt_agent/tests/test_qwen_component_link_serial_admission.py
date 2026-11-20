"""Only metadata command admission; no job/process/model/tool is launched."""
from pathlib import Path
import unittest
from unittest.mock import patch

from pt_agent.experiments import autonomous_serial_batch as entry


class ComponentLinkSerialAdmissionChecks(unittest.TestCase):
    def job(self):
        out = '/fixture/qwen-original27-link'
        return dict(out=out, seconds=1920,
            argv=['/fixture/metadata-numpy/bin/python','-B','-m',
                entry.PREFIX+'qwen_component_link_validation','--out',out,
                '--source-case','/fixture/frozen','--probe-case','/fixture/layer0',
                '--remaining-case','/fixture/remaining','--kernel-case','/fixture/kernels',
                '--library-case','/fixture/library','--initialization-audit','/fixture/init.json','--run'])

    def test_exact_link_entry_and_only_configured_numpy_python_are_admitted(self):
        job = self.job()
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',Path(job['argv'][0])):
            self.assertIs(entry.check_command(job['argv'],job['out']),job['argv'])
        self.assertEqual(entry.parent_limits(job),(24,16))

    def test_unknown_or_unconfigured_python_and_unknown_module_are_rejected(self):
        job = self.job()
        for configured in (None,Path('/foreign/python')):
            with self.subTest(configured=configured),patch.object(entry.paths,'PAPER_DATA_PYTHON',configured):
                with self.assertRaises(ValueError):entry.check_command(job['argv'],job['out'])
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',Path(job['argv'][0])):
            job['argv'][3] += '_foreign'
            with self.assertRaises(ValueError):entry.check_command(job['argv'],job['out'])

    def test_parent_limits_are_original24_16_and_cannot_expand_for_this_entry(self):
        job = self.job()
        self.assertEqual(entry.parent_limits(dict(job,outer_address_space_gib=24,outer_file_gib=16)),(24,16))
        for memory,file_size in ((32,64),(24,64),(32,16)):
            with self.subTest(memory=memory,file_size=file_size),self.assertRaises(ValueError):
                entry.parent_limits(dict(job,outer_address_space_gib=memory,outer_file_gib=file_size))

    def test_foreign_output_and_detached_variants_are_rejected(self):
        job = self.job()
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',Path(job['argv'][0])):
            with self.assertRaises(ValueError):entry.check_command(job['argv'],'/fixture/foreign')
            with self.assertRaises(ValueError):entry.check_command(job['argv']+['--detach'],job['out'])

    def test_manifest_accepts_exact_watch_deadline_and_five_closed_source_requirements(self):
        job = self.job();job['id']='one-finite27-link'
        cases=['/fixture/'+name for name in ('frozen','layer0','remaining','kernels','library')]
        job['requires_case_statuses']={case:'ok' for case in cases}
        manifest=dict(schema_version=1,deadline_utc='2026-11-17T10:00:00+00:00',
            watch=dict(case='/fixture/kernel-batch',pid=92537,starttime=2065138,seconds=57000),jobs=[job])
        with patch.object(entry.paths,'PAPER_DATA_PYTHON',Path(job['argv'][0])):
            self.assertIs(entry.validate(manifest,Path('/fixture')),manifest)
        self.assertEqual(len(job['requires_case_statuses']),5)
        self.assertEqual(entry.parent_limits(job),(24,16))


if __name__=='__main__':unittest.main()
