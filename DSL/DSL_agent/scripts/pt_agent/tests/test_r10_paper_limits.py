"""R10 wall-clock forwarding, location overrides and Nix interpreter quoting."""

from contextlib import ExitStack, contextmanager
import importlib.util
import gzip
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from pt_agent import paths, toolchain
from pt_agent.tools import nixrun


@contextmanager
def stream_runner():
    """Load the standard-library stream runner without unrelated NumPy planners."""
    source = Path(toolchain.__file__).parent/'manage/streams.py'
    definition = importlib.util.spec_from_file_location('pt_agent.manage.streams', source)
    streams = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(streams)
    package = ModuleType('pt_agent.manage')
    package.__path__ = [str(source.parent)]
    package.streams = streams
    with patch.dict(sys.modules, {'pt_agent.manage': package, 'pt_agent.manage.streams': streams}):
        yield streams


@contextmanager
def stream_wait():
    with stream_runner() as streams, patch.object(streams.subprocess, 'run') as wait:
        yield wait


class TrustedWallClockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def run_through_wait(self, executable, **kwargs):
        """Keep both runner routes intact until their subprocess wait boundary."""
        with patch.object(toolchain.shutil, 'disk_usage') as usage, \
                stream_wait() as wait:
            usage.return_value.free = 20 * 1024**3
            wait.return_value = subprocess.CompletedProcess([], 0)
            result = toolchain.trusted_run(self.directory, [executable],
                                           self.directory/'run.log', **kwargs)
        self.assertEqual(result['exit'], 0)
        self.assertFalse(result['timeout'])
        self.assertEqual(wait.call_count, 1)
        return wait.call_args

    def test_explicit_7200_reaches_both_subprocess_wait_routes(self):
        for executable in ('/unused/hecate-opt', '/unused/python'):
            with self.subTest(executable=executable):
                call = self.run_through_wait(executable, wall_seconds=7200)
                self.assertEqual(call.kwargs['timeout'], 7200)
                self.assertEqual(call.args[0][-1], executable)
                self.assertEqual(call.args[0][0], '/usr/bin/time')
                self.assertTrue(callable(call.kwargs['preexec_fn']))

    def test_default_remains_3600_for_both_wait_routes(self):
        for executable in ('/unused/hecate-opt', '/unused/python'):
            with self.subTest(executable=executable):
                call = self.run_through_wait(executable)
                self.assertEqual(call.kwargs['timeout'], 3600)
        self.assertEqual(toolchain.TRUSTED_LIMITS,
                         dict(mem_gib=6, fsize_gib=8, wall_seconds=3600, min_free_gib=10))

    def test_compressed_ir_route_passes_7200_to_popen_wait(self):
        source = self.directory/'input.mlir.gz'
        with gzip.open(source, 'wt') as stream:
            stream.write('module {}\n')
        with stream_runner() as streams, \
                patch.object(toolchain.shutil, 'disk_usage') as usage, \
                patch.object(streams.subprocess, 'Popen') as popen:
            usage.return_value.free = 20 * 1024**3
            process = popen.return_value
            process.stdin = io.BytesIO()
            process.wait.return_value = 0
            process.poll.return_value = 0
            result = toolchain.trusted_run(self.directory,
                ['/unused/hecate-opt', str(source)], self.directory/'compressed.log', wall_seconds=7200)
        process.wait.assert_called_once_with(timeout=7200)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(result['exit'], 0)
        self.assertEqual(result['ir_io']['route'], 'stdin')

    def test_explicit_wall_clock_is_a_positive_integer_at_most_7200(self):
        for invalid in (0, -1, 7201, True, False, 1.5, '7200'):
            with self.subTest(invalid=invalid), patch.object(toolchain.shutil, 'disk_usage') as usage:
                with self.assertRaises(ValueError):
                    toolchain.trusted_run(self.directory, ['/unused/python'],
                                          self.directory/'invalid.log', wall_seconds=invalid)
                usage.assert_not_called()
        self.assertEqual(self.run_through_wait('/unused/python', wall_seconds=1).kwargs['timeout'], 1)

    def test_timeout_flag_preserves_missing_measurements(self):
        for executable in ('/unused/hecate-opt', '/unused/python'):
            with self.subTest(executable=executable), \
                    patch.object(toolchain.shutil, 'disk_usage') as usage, \
                    stream_wait() as wait:
                usage.return_value.free = 20 * 1024**3
                wait.side_effect = subprocess.TimeoutExpired([executable], 7200)
                result = toolchain.trusted_run(self.directory, [executable],
                                               self.directory/'timeout.log', wall_seconds=7200)
                self.assertEqual(result['exit'], -9)
                self.assertTrue(result['timeout'])
                self.assertIsNone(result['peak_rss_mib'])

    def compile_through_wait(self, **kwargs):
        with ExitStack() as stack:
            stack.enter_context(patch.object(toolchain, 'setup_env'))
            stack.enter_context(patch.object(toolchain, 'compile_argv', return_value=['/unused/hecate-opt']))
            stack.enter_context(patch.object(toolchain, 'to_host_argv', side_effect=lambda argv, _: argv))
            usage = stack.enter_context(patch.object(toolchain.shutil, 'disk_usage'))
            usage.return_value.free = 20 * 1024**3
            wait = stack.enter_context(stream_wait())
            wait.return_value = subprocess.CompletedProcess([], 0)
            stack.enter_context(patch.dict(os.environ, {'HECATE_PYTHON_LIBRARY_PATH': '/unused/lib'}))
            result = toolchain.compile(self.directory, 'pars', **kwargs)
        self.assertEqual(result['exit'], 0)
        self.assertFalse(result['plan'])
        self.assertEqual(wait.call_count, 1)
        return wait.call_args.kwargs['timeout']

    def test_trusted_compile_default_ignores_old_seconds_parameter(self):
        self.assertEqual(self.compile_through_wait(trusted=True), 3600)
        self.assertEqual(self.compile_through_wait(seconds=17, trusted=True), 3600)

    def test_trusted_compile_explicit_7200_reaches_wait(self):
        self.assertEqual(self.compile_through_wait(trusted=True, trusted_wall_seconds=7200), 7200)

    def test_nontrusted_compile_retains_seconds(self):
        sandbox = ModuleType('poseidon_candidate_sandbox')
        calls = []

        def run(*args):
            calls.append(args)
            Path(args[3]).write_text('ordinary compiler log\n')
            return 0

        sandbox.run = run
        with patch.object(toolchain, 'setup_env'), \
                patch.object(toolchain, '_legacy_sdk_env'), \
                patch.object(toolchain, 'compile_argv', return_value=['/unused/hecate-opt']), \
                patch.dict(sys.modules, {'poseidon_candidate_sandbox': sandbox}), \
                patch.object(toolchain, 'trusted_run') as trusted:
            result = toolchain.compile(self.directory, 'pars', seconds=17,
                                       trusted=False, trusted_wall_seconds=7200)
        trusted.assert_not_called()
        self.assertEqual(calls[0][-1], 17)
        self.assertEqual(result['exit'], 0)


class PaperLocationTests(unittest.TestCase):
    def locations(self, overrides):
        names = ('PT_ORION_ROOT', 'PT_ORBIT_ROOT', 'PT_INTAKE_PYTHON', 'PT_PAPER_DATA_PYTHON')
        environment = {key: value for key, value in os.environ.items() if key not in names}
        environment.update(overrides)
        code = ("import json; from pt_agent import paths; "
                "keys=('ORION_ROOT','ORBIT_ROOT','INTAKE_PYTHON','PAPER_DATA_PYTHON'); "
                "print(json.dumps({key:None if getattr(paths,key) is None else str(getattr(paths,key)) "
                "for key in keys}))")
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(__file__).resolve().parents[2], env=environment,
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def test_defaults_derive_from_existing_location_roots(self):
        actual = self.locations({})
        self.assertEqual(actual['ORION_ROOT'], str(paths.WORKSPACE/'baselines/orion'))
        self.assertEqual(actual['ORBIT_ROOT'], str(paths.WORKSPACE/'baselines/orbit'))
        self.assertEqual(actual['INTAKE_PYTHON'], str(paths.PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'))
        self.assertIsNone(actual['PAPER_DATA_PYTHON'])

    def test_all_four_overrides_and_empty_data_interpreter(self):
        names = {'PT_ORION_ROOT': '/paper/orion checkout', 'PT_ORBIT_ROOT': '/paper/orbit checkout',
                 'PT_INTAKE_PYTHON': '/paper/intake python', 'PT_PAPER_DATA_PYTHON': '/paper/data python'}
        actual = self.locations(names)
        self.assertEqual(actual, {key.removeprefix('PT_'): value for key, value in names.items()})
        self.assertIsNone(self.locations({'PT_PAPER_DATA_PYTHON': '   '})['PAPER_DATA_PYTHON'])
        self.assertTrue(set(names).issubset(paths.ENV_VARS))


class NixInterpreterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.default_python = self.directory/'venv/bin/python'
        self.default_python.parent.mkdir(parents=True)
        self.default_python.touch()
        self.explicit_python = self.directory/"intake ' $(literal) python"
        self.explicit_python.touch()

    def launch(self, options):
        captured = []
        module = ModuleType('hecate_python_env')
        module.ROOT = self.directory
        module.VENV = self.default_python.parents[1]
        module.enter_nix = lambda command, **kwargs: captured.append((command, kwargs)) or 11
        worker = ['-m', 'pt_agent.experiments.r10_paper_worker', '--inputs',
                  "a path with ' $() `literal` and\nnewlines"]
        with patch.dict(sys.modules, {'hecate_python_env': module}), \
                patch.object(sys, 'argv', ['nixrun', *options, '--', *worker]), \
                patch.object(sys, 'path', list(sys.path)), patch.object(nixrun.os, 'chdir') as chdir, \
                patch.dict(os.environ, {'PT_ORION_ROOT': "/orion ' $() path",
                                       'PT_ORBIT_ROOT': '/orbit path',
                                       'PT_INTAKE_PYTHON': str(self.explicit_python),
                                       'PT_PAPER_DATA_PYTHON': ''}):
            code = nixrun.main()
        self.assertEqual(code, 11)
        self.assertEqual(len(captured), 1)
        chdir.assert_called_once_with(self.directory)
        command, kwargs = captured[0]
        tokens = shlex.split(command)
        self.assertEqual(tokens[-len(worker):], worker)
        self.assertIn("PT_ORION_ROOT=/orion ' $() path", tokens)
        self.assertIn('PT_ORBIT_ROOT=/orbit path', tokens)
        self.assertIn('PT_INTAKE_PYTHON='+str(self.explicit_python), tokens)
        self.assertFalse(any(token.startswith('PT_PAPER_DATA_PYTHON=') for token in tokens))
        self.assertEqual(tokens[-len(worker)-1], '-B')
        if '--python' in options:
            self.assertEqual(tokens[3:7], ['test', '-x', tokens[-len(worker)-2], '&&'])
        else:
            self.assertNotIn('test', tokens)
        return tokens[-len(worker)-2], kwargs

    def test_default_interpreter_and_seconds_are_unchanged(self):
        interpreter, kwargs = self.launch([])
        self.assertEqual(interpreter, str(self.default_python))
        self.assertEqual(kwargs, {'seconds': 600})

    def test_explicit_interpreter_and_extended_budget_are_quoted(self):
        interpreter, kwargs = self.launch(['--python', str(self.explicit_python), '--seconds', '10920'])
        self.assertEqual(interpreter, str(self.explicit_python))
        self.assertEqual(kwargs, {'seconds': 10920})

    def test_explicit_venv_symlink_keeps_the_venv_entry_path(self):
        interpreter_path = self.directory/'intake-venv/bin/python'
        interpreter_path.parent.mkdir(parents=True)
        interpreter_path.symlink_to(sys.executable)
        interpreter, _ = self.launch(['--python', str(interpreter_path)])
        self.assertEqual(interpreter, str(interpreter_path))
        self.assertNotEqual(interpreter, str(interpreter_path.resolve()))

    def test_nix_store_symlink_is_checked_inside_the_pure_environment(self):
        interpreter_path = self.directory/'nix-intake/bin/python'
        interpreter_path.parent.mkdir(parents=True)
        interpreter_path.symlink_to('/nix/store/r10-placeholder-python/bin/python3')
        self.assertFalse(interpreter_path.is_file())
        interpreter, _ = self.launch(['--python', str(interpreter_path)])
        self.assertEqual(interpreter, str(interpreter_path))
        probe = subprocess.run(['/bin/sh', '-c', 'test -x '+shlex.quote(interpreter)], check=False)
        self.assertEqual(probe.returncode, 1)

    def test_missing_interpreter_is_rejected_before_environment_import(self):
        missing = self.directory/'absent python'
        with patch.object(sys, 'argv', ['nixrun', '--python', str(missing), '--', '-m', 'worker']), \
                patch.object(nixrun.os, 'chdir') as chdir, \
                patch('sys.stderr', new_callable=io.StringIO) as stderr, \
                self.assertRaises(SystemExit) as error:
            nixrun.main()
        self.assertEqual(error.exception.code, 2)
        chdir.assert_not_called()
        self.assertIn('existing interpreter', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
