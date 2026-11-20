import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent.experiments.r34_archive_text import archive_one, archive_state, validate_files


class ArchiveTextTests(unittest.TestCase):
    def test_recoverable_json_and_explicit_restore_command(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); folder = root/'A1-historical'; folder.mkdir()
            source = folder/'sample.runtime-plan.json'
            source.write_text(json.dumps({'execution':[{'op':'add'}]*1000}))
            chosen = validate_files(root,[dict(path=str(source),bytes=source.stat().st_size)])
            result = archive_one(*chosen[0])
            self.assertFalse(source.exists()); self.assertGreater(result['released_bytes'],0)
            with gzip.open(result['archive'],'rt') as stream:
                restored = json.load(stream)
            self.assertEqual(len(restored['execution']),1000)
            self.assertEqual(restored['execution'][0]['op'],'add')
            self.assertEqual(result['restore_argv'],['gzip','-dk','--',result['archive']])

    def test_close_failure_keeps_original(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory).resolve()/'E-historical'; folder.mkdir()
            source = folder/'trace.mlir'; source.write_text('module { }\n'*1000)
            chosen = validate_files(Path(directory).resolve(),[dict(path=str(source),bytes=source.stat().st_size)])
            with patch('pt_agent.experiments.r34_archive_text.os.fsync',side_effect=OSError('test fsync failure')):
                with self.assertRaises(OSError):archive_one(*chosen[0])
            self.assertTrue(source.is_file()); self.assertFalse(chosen[0][1].exists())

    def test_active_root_or_symbolic_source_rejected(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve(); folder=root/'qwen-current'; folder.mkdir()
            source=folder/'x.runtime-plan.json';source.write_text('{}')
            with self.assertRaises(ValueError):validate_files(root,[dict(path=str(source),bytes=2)])
            old=root/'A1-historical';old.mkdir();link=old/'x.runtime-plan.json';link.symlink_to(source)
            with self.assertRaises(ValueError):validate_files(root,[dict(path=str(link),bytes=2)])

    def test_late_directory_failure_preserves_archive_recovery_record(self):
        with TemporaryDirectory() as directory:
            root=Path(directory).resolve(); folder=root/'E-historical'; folder.mkdir()
            source=folder/'trace.mlir';source.write_text('module { }\n'*1000)
            chosen=validate_files(root,[dict(path=str(source),bytes=source.stat().st_size)])[0]
            with patch('pt_agent.experiments.r34_archive_text.os.fsync',side_effect=[None,None,OSError('late fsync failure')]):
                with self.assertRaises(OSError):archive_one(*chosen)
            state=archive_state(*chosen)
            self.assertFalse(state['source_exists']);self.assertTrue(state['archive_exists'])
            self.assertFalse(state['temporary_exists'])
            self.assertEqual(state['restore_argv'],['gzip','-dk','--',state['archive']])
