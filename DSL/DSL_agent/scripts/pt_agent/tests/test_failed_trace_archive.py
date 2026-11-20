import json
from pathlib import Path
import tempfile
import unittest

from pt_agent.experiments.failed_trace_archive import source_gate


class ClosedTraceGate(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.case=self.root/'case';self.case.mkdir()
        self.source=self.case/'work/trace/traced/VGG16.mlir';self.source.parent.mkdir(parents=True);self.source.write_text('partial trace fixture')
        self.path=self.case/'result.json'
        self.record=dict(mode='original-source-native-profile-compile',status='error',original_model_selector='vgg16',
            trace_attempts=1,trace_completed=0,compile_attempts=0,native_compile_attempts=0,trace_process=dict(exit_code=1,remaining_pids=[]))
        self.path.write_text(json.dumps(self.record));(self.case/'completion.json').write_text(json.dumps(dict(status='error',registry_persisted=True,required_artifacts_persisted=True)))

    def test_closed_partial_trace_path(self):
        source,record=source_gate(self.path,self.root)
        self.assertEqual(source,self.source);self.assertEqual(record['trace_completed'],0)

    def test_live_or_compiled_case_rejected(self):
        for key,value in [('status','running'),('compile_attempts',1),('trace_completed',1)]:
            with self.subTest(key=key):
                d=dict(self.record);d[key]=value;self.path.write_text(json.dumps(d))
                with self.assertRaises(ValueError):source_gate(self.path,self.root)

    def test_prior_archive_preserved(self):
        self.source.with_name('VGG16.mlir.gz').touch()
        with self.assertRaises(ValueError):source_gate(self.path,self.root)

    def test_parent_symlink_rejected(self):
        real=self.root/'other';self.source.parent.rename(real);self.source.parent.symlink_to(real,target_is_directory=True)
        with self.assertRaises(ValueError):source_gate(self.path,self.root)


if __name__=='__main__':unittest.main()
