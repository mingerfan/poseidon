"""Early recorded-command delegation; synthetic worker and no Provider calls."""
from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments import bert_sst2_recorded as entry
from pt_agent.tests.test_bert_frozen_compile import template,save


class EarlyBertMetadataChecks(unittest.TestCase):
    def test_context_precedes_worker_import_and_preserves_original_argv_without_new_digest(self):
        from pt_agent import toolchain
        with TemporaryDirectory() as temp:
            root=Path(temp);metadata=root/'template.json';save(metadata,template())
            config=root/'profiles';save(config/'compiler-config.json',dict(rescalingFactor=43))
            original_source=root/'worker.py';original_source.write_text('# synthetic trusted original worker\n')
            argv=['--phase','step','--work-dir',str(root/'work'),'--model','/same/model']
            received=[];original=toolchain.compile_argv
            def import_worker(name):
                self.assertEqual(name,entry.WORKER)
                self.assertIsNot(toolchain.compile_argv,original)
                self.assertEqual(toolchain.management_profile()['waterline'],43)
                return Namespace(__file__=str(original_source),main=lambda args:received.append(args) or 0)
            with (patch.object(entry.importlib,'import_module',side_effect=import_worker),
                  patch.object(toolchain,'BOOT_CONFIG_DIR',config),patch.object(toolchain,'boot_window_rule',return_value='dacapo-39c1fb2'),
                  patch.object(toolchain,'_file_sha256',side_effect=AssertionError('digest forbidden')),
                  patch.object(toolchain.hashlib,'sha256',side_effect=AssertionError('digest forbidden'))):
                self.assertEqual(entry.worker_delegate(argv,metadata),0)
            self.assertEqual(received,[argv]);self.assertIs(toolchain.compile_argv,original)
            note=json.loads((root/'work/recorded-command-delegation.json').read_text())
            self.assertTrue(note['early_context_installed_before_worker_import']);self.assertEqual(note['new_digest_computations'],0)
            self.assertTrue((root/'work/original-bert-phase-source.py').exists())

    def test_outer_replaces_only_worker_module_and_passes_template_explicitly(self):
        with TemporaryDirectory() as temp:
            metadata=Path(temp)/'template.json';save(metadata,template());received=[]
            original=lambda args,phase,work:['python','-m','nixrun','--','-m',entry.WORKER,'--phase',phase,'--work-dir',str(work)]
            outer=Namespace(WORKER=entry.WORKER,command=original)
            def main(argv):
                received.append(outer.command(None,'step',Path('/work')));self.assertEqual(argv,['--phase','step']);return 0
            outer.main=main
            with patch.object(entry.importlib,'import_module',return_value=outer):
                self.assertEqual(entry.outer_delegate(['--phase','step'],metadata),0)
            self.assertIs(outer.command,original);self.assertNotIn(entry.WORKER,received[0])
            self.assertIn(entry.__spec__.name,received[0]);self.assertEqual(received[0][-3:],['--delegate-worker','--metadata-template',str(metadata)])

    def test_original_worker_exception_restores_recorded_command_context(self):
        from pt_agent import toolchain
        with TemporaryDirectory() as temp:
            root=Path(temp);metadata=root/'template.json';save(metadata,template());source=root/'worker.py';source.write_text('# fixture\n')
            original=toolchain.compile_argv
            def fail(argv):raise RuntimeError('synthetic phase failure')
            with patch.object(entry.importlib,'import_module',return_value=Namespace(__file__=str(source),main=fail)):
                with self.assertRaises(RuntimeError):entry.worker_delegate(['--work-dir',str(root/'work')],metadata)
            self.assertIs(toolchain.compile_argv,original)


if __name__=='__main__':unittest.main()
