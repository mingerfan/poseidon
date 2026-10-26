"""S1 §11 gzip equivalence and failure propagation at both pipe ends."""

import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent import toolchain
from pt_agent.manage import emit, emit_raw, read
from pt_agent.manage import streams
from pt_agent.tests._artifacts import COMPILER, require
from pt_agent.tests.test_s1_manage import record_case


class StreamTests(unittest.TestCase):
    def test_temporary_input_is_single_and_deleted_on_success_or_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'input.mlir.gz'
            with gzip.open(source,'wt') as dest:
                dest.write('module {}\n')
            script = ('import pathlib,sys; p=pathlib.Path(sys.argv[1]); '
                      'assert p.is_file() and len(list(p.parent.glob("s1-input-*.mlir")))==1; '
                      'sys.stdout.write(p.read_text()); sys.exit(int(sys.argv[2]))')
            for code in (0,7):
                target = root/('output-'+str(code)+'.mlir')
                with patch.object(streams,'TEMP_INPUT_LIMIT',0):
                    result = streams.run([sys.executable,'-c',script,str(source),str(code),
                        '--pass-pipeline=test','-o',str(target)],cwd=root,timeout=30)
                self.assertEqual(result.returncode,code)
                self.assertEqual(result.ir_io['route'],'temporary-file')
                self.assertTrue(result.ir_io['temporary_deleted'])
                self.assertFalse(list(root.glob('s1-input-*.mlir')))
                self.assertEqual(streams.resolve(target).exists(),code==0)
                if code==0:
                    with streams.open_text(target) as saved:
                        self.assertEqual(saved.read(),'module {}\n')

    def test_temporary_input_admission_reserves_the_full_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'input.mlir.gz'
            with gzip.open(source,'wt') as dest:
                dest.write('module {}\n')
            size = streams.uncompressed_size(source)
            usage = shutil.disk_usage(root)._replace(free=10*1024**3+size-1)
            with patch.object(streams,'TEMP_INPUT_LIMIT',0),patch.object(streams.shutil,'disk_usage',return_value=usage),\
                    patch.object(streams.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'temporary_bytes='+str(size)):
                    streams.run([sys.executable,'-c','pass',str(source)],cwd=root,timeout=30)
                launch.assert_not_called()
            self.assertFalse(list(root.glob('s1-input-*.mlir')))

    def test_python_driver_keeps_its_compressed_input_argument(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'input.mlir.gz'
            with gzip.open(source,'wt') as dest:
                dest.write('module {}\n')
            result = toolchain.trusted_run(root,[sys.executable,'-c','import sys; print(sys.argv[1])',
                str(source)],root/'run.log')
            self.assertEqual(result['exit'],0)
            self.assertEqual((root/'run.log').read_text().strip(),str(source))

    def test_raw_and_managed_text_and_reader_are_unchanged(self):
        recorded,function = record_case('constants')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for raw in (True,False):
                def write(path, compressed):
                    metrics = {}
                    if raw:
                        result = emit_raw(recorded,path,function.profile,metrics=metrics,compressed=compressed)
                    else:
                        result = emit(function,path,metrics=metrics,compressed=compressed)
                    with streams.open_text(result) as source:
                        self.assertEqual(len(source.read()),metrics['uncompressed_bytes'])
                    return result
                plain = write(root/'plain.mlir',False)
                zipped = write(root/'compressed.mlir',True)
                self.assertTrue(streams.same_text(plain,zipped))
                a,b = read(plain,function.profile),read(zipped,function.profile)
                self.assertEqual([(n.kind,n.args,n.type,n.attrs) for n in a.nodes],
                                 [(n.kind,n.args,n.type,n.attrs) for n in b.nodes])
            with patch.object(streams,'IR_LIMIT',1):
                path = emit(function,root/'automatic.mlir')
            self.assertEqual(path.suffix,'.gz')
            self.assertFalse((root/'automatic.mlir').exists())

    def test_chunked_pipe_keeps_diagnostics_out_of_ir(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source,target = root/'input.mlir.gz',root/'output.mlir'
            text = 'module {}\n'*(streams.CHUNK//4)
            with gzip.open(source,'wt') as dest:
                dest.write(text)
            script = 'import sys; sys.stderr.write("diagnostic\\n"); sys.stdout.buffer.write(sys.stdin.buffer.read())'
            command = [sys.executable,'-c',script,str(source),'--pass-pipeline=test','-o',str(target)]
            with (root/'run.log').open('w') as log:
                result = streams.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            self.assertEqual(result.returncode,0)
            self.assertEqual((root/'run.log').read_text(),'diagnostic\n')
            with streams.open_text(target) as saved:
                self.assertEqual(saved.read(),text)
            self.assertFalse(target.exists())
            self.assertFalse(Path(str(target)+'.gz.part').exists())

    def test_failed_process_does_not_publish_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source,target = root/'input.mlir.gz',root/'output.mlir'
            with gzip.open(source,'wt') as stream:
                stream.write('module {}\n')
            result = streams.run([sys.executable,'-c','import sys; sys.stdin.read(); print("partial"); sys.exit(7)',
                str(source),'--pass-pipeline=test','-o',str(target)],cwd=root,timeout=30)
            self.assertEqual(result.returncode,7)
            self.assertFalse(Path(str(target)+'.gz').exists())
            self.assertFalse(Path(str(target)+'.gz.part').exists())

    def test_corrupt_input_fails_the_whole_pipe(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source,target = root/'input.mlir.gz',root/'output.mlir'
            source.write_bytes(b'invalid gzip')
            with self.assertRaisesRegex(RuntimeError,'pipe failed'):
                streams.run([sys.executable,'-c','import sys; sys.stdout.write(sys.stdin.read())',
                    str(source),'--pass-pipeline=test','-o',str(target)],cwd=root,timeout=30)
            self.assertFalse(Path(str(target)+'.gz').exists())
            self.assertFalse(Path(str(target)+'.gz.part').exists())

    def test_frontend_compression_validates_format_and_preserves_small_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'trace.mlir'
            path.write_text('module {}\n')
            self.assertEqual(streams.compress_ir(path),path)
            with patch.object(streams.gzip,'open',side_effect=OSError('writer failed')):
                with self.assertRaisesRegex(OSError,'writer failed'):
                    streams.compress_ir(path,force=True)
            self.assertTrue(path.exists())
            actual = streams.compress_ir(path,force=True)
            self.assertFalse(path.exists())
            with streams.open_text(actual) as saved:
                self.assertEqual(saved.read(),'module {}\n')


@require(COMPILER)
class FixedLocationTests(unittest.TestCase):
    def test_same_raw_file_and_stdin_preprocessing_text(self):
        recorded,function = record_case('constants')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = emit_raw(recorded,root/'raw.earth.mlir',function.profile)
            text = raw.read_text()
            self.assertIn('loc(unknown)',text)
            self.assertIn('loc("ptdsl-op-12":7:0)',text)
            outputs = []
            for mode in ('file','stdin','temporary-file'):
                directory = root/mode
                directory.mkdir()
                shutil.copyfile(toolchain.BOOT_CONFIG_DIR/'compiler-config.json',directory/'boot-compiler-config.json')
                command = toolchain.to_host_argv(toolchain.frontend_preprocess_argv('dacapo-depth-dp','qwen48',directory),directory)
                command[1] = str(raw)
                if mode!='file':
                    streams.compress_ir(raw,force=True)
                limit = 0 if mode=='temporary-file' else 1024**3
                with patch.object(streams,'TEMP_INPUT_LIMIT',limit):
                    status = toolchain.trusted_run(directory,command,directory/'preprocess.log',
                        env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
                self.assertEqual(status['exit'],0,status['log_tail'])
                if mode!='file':
                    self.assertEqual(status['ir_io']['route'],mode)
                outputs.append(directory/'preprocessed.earth.mlir')
            self.assertTrue(all(streams.same_text(outputs[0],other) for other in outputs[1:]))


def compression_suite(root):
    @require(COMPILER, root)
    class CompressionModels(unittest.TestCase):
        def compare(self,model):
            reports = [json.loads((root/model/mode/'REPORT.json').read_text())
                       for mode in ('plain','gzip','temporary-file')]
            self.assertTrue(all(r['complete'] for r in reports))
            for label in ('preprocessed','managed','tail'):
                paths = [r['text_outputs'][label] for r in reports]
                self.assertTrue(all(streams.same_text(paths[0],other) for other in paths[1:]),model+' '+label)
            self.assertTrue(all(r['plan']['equal'] for r in reports))
            status = reports[2]['frontend_preprocess']['status']['ir_io']
            self.assertEqual(status['route'],'temporary-file')
            self.assertTrue(status['temporary_deleted'])

        def test_mlp(self):
            self.compare('mlp')

        def test_cnn_plain(self):
            self.compare('cnn_plain')

    return unittest.defaultTestLoader.loadTestsFromTestCase(CompressionModels)
