"""Offline host/guest portability tests. No SSH, credentials or paid API calls."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import workspace_paths as paths

spec = importlib.util.spec_from_file_location('portable_agent', paths.ROOT/'scripts/agent.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PortablePathsTests(unittest.TestCase):
    def test_root_is_code_location_not_cwd(self):
        self.assertEqual(paths.ROOT, Path(__file__).resolve().parents[2])
        code = 'import sys;sys.path.insert(0,sys.argv[1]);import workspace_paths as p;print(p.ROOT)'
        with tempfile.TemporaryDirectory(prefix='agent path space ') as folder:
            result = subprocess.run([sys.executable,'-B','-c',code,str(Path(__file__).parent)],
                                    cwd=folder,capture_output=True,text=True,check=True,timeout=10)
        self.assertEqual(result.stdout.strip(), str(paths.ROOT))

    def test_different_checkout_and_user_do_not_require_source_edits(self):
        with tempfile.TemporaryDirectory(prefix='new user space ') as folder:
            checkout = Path(folder)/'different checkout'
            module = checkout/'scripts/baseline/workspace_paths.py'
            module.parent.mkdir(parents=True)
            module.write_bytes(Path(paths.__file__).read_bytes())
            for name in ('platform_config.py', 'platform-profiles.json'):
                (module.parent/name).write_bytes((Path(paths.__file__).parent/name).read_bytes())
            work = Path(folder)/'native disk work'
            env = dict(os.environ, POSEIDON_WORK_ROOT=str(work), PYTHONDONTWRITEBYTECODE='1')
            env.pop('POSEIDON_PLATFORM', None)  # Exercise the retained x86 default.
            code = 'import workspace_paths as p;print(p.ROOT);print(p.WORK)'
            result = subprocess.run([sys.executable,'-B','-c',code],cwd=module.parent,env=env,
                                    capture_output=True,text=True,check=True,timeout=10)
            self.assertEqual(result.stdout.splitlines(), [str(checkout.resolve()),str(work.resolve())])
            self.assertFalse(work.exists(), 'Import must not create work dirs')

    def test_work_override_and_unsafe_locations(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            home = Path(folder).resolve()
            self.assertEqual(paths.work_root(home=home), home/'poseidon-work')
            self.assertEqual(paths.work_root(str(home/'disk with spaces'),home=home),home/'disk with spaces')
            for value in ('relative/work', str(home), home.anchor):
                with self.assertRaises(ValueError):
                    paths.work_root(value,home=home)

    def test_nix_only_keeps_explicit_work_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.nix_environment_options(), [])
            os.environ['POSEIDON_WORK_ROOT'] = '/data/agent work'
            os.environ['DEEPSEEK_API_KEY'] = 'synthetic-not-a-secret'
            self.assertEqual(paths.nix_environment_options(), ['--keep','POSEIDON_WORK_ROOT'])

    @patch.dict(os.environ, {}, clear=True)
    def test_backend_architecture_is_explicit_not_guessed(self):
        for system, machine in [('win32','AMD64'),('darwin','arm64'),('linux','aarch64')]:
            with patch.object(paths.sys,'platform',system), patch.object(paths.platform,'machine',return_value=machine):
                with self.assertRaisesRegex(RuntimeError,'Linux x86_64'):
                    paths.require_linux_backend()
        with patch.object(paths.sys,'platform','linux'), patch.object(paths.platform,'machine',return_value='x86_64'):
            paths.require_linux_backend()

    def test_no_personal_runtime_paths_in_agent_sources(self):
        forbidden = ('/home/'+'lhy/', '/mnt/d/'+'Code Space/Poseidon', '/mnt/d/'+'CodeSpace/Poseidon')
        sources = list(Path(__file__).parent.glob('*.py')) + list(Path(__file__).parent.glob('*.sh'))
        for source in sources:
            for text in forbidden:
                self.assertNotIn(text, source.read_text(encoding='utf-8'), str(source))


class LauncherTests(unittest.TestCase):
    def args(self, *values):
        with patch.dict(os.environ, {}, clear=True):
            return launcher.parse_args(values)

    def test_windows_uses_selected_or_default_wsl(self):
        args = self.args('--backend-root','/mnt/e/Team Project','candidate','--','--case','cases/a.json','--prepare')
        command = launcher.build_command(args,host_platform='win32')
        self.assertEqual(command[:2],['wsl.exe','--exec'])
        self.assertIn('/mnt/e/Team Project', command)
        self.assertEqual(command[-3:],['--case','cases/a.json','--prepare'])
        args.wsl_distro = 'Team-Ubuntu'
        self.assertEqual(launcher.build_command(args,host_platform='win32')[:3],
                         ['wsl.exe','--distribution','Team-Ubuntu'])

    def test_mac_ssh_quotes_each_argument_and_preserves_deadline(self):
        args = self.args('--ssh-host','team-ubuntu','--backend-root',"/srv/Team's Project",
                         '--work-root','/data/work space','--timeout','123','candidate','--',
                         '--case','cases/a;$(never-run).json','--prepare')
        command = launcher.build_command(args,host_platform='darwin')
        self.assertEqual(command[0],'ssh')
        self.assertIn('StrictHostKeyChecking=yes',command)
        self.assertIn('BatchMode=yes',command)
        remote = shlex.split(command[-1])
        self.assertEqual(remote[-3:],['--case','cases/a;$(never-run).json','--prepare'])
        self.assertIn("/srv/Team's Project", remote)
        self.assertIn('123',remote)
        self.assertIn('/data/work space',remote)
        self.assertIn('os.execvp("timeout"',remote[2])

    def test_no_platform_guess_or_inside_bypass(self):
        cases = [(['--backend','local','doctor'],'darwin'),
                 (['--backend','wsl','doctor'],'linux'),
                 (['--backend','wsl','doctor'],'win32'),
                 (['--backend','ssh','--ssh-host','-evil','doctor'],'darwin')]
        for argv,host in cases[:3]:
            with self.assertRaises(ValueError):
                launcher.build_command(self.args(*argv),host_platform=host)
        args = self.args('--backend','local','candidate','--','--inside')
        with self.assertRaises(ValueError):
            launcher.build_command(args,host_platform='linux')

    def test_host_paths_and_ssh_options_are_rejected(self):
        for path in ('D:\\repo','../repo','~/repo','/','/data/../repo'):
            with self.assertRaises(ValueError):
                launcher.posix_absolute(path,'root')
        args = self.args('--backend','ssh','--ssh-host=-oProxyCommand=evil',
                         '--backend-root','/srv/repo','doctor')
        with self.assertRaises(ValueError):
            launcher.build_command(args,host_platform='darwin')

    def test_dry_run_never_spawns_or_reads_credentials(self):
        with patch.object(launcher.subprocess,'run') as run, patch('builtins.print'):
            code = launcher.main(['--backend','ssh','--ssh-host','ubuntu',
                                  '--backend-root','/srv/repo','--dry-run','doctor'])
        self.assertEqual(code,0)
        run.assert_not_called()

    def test_local_launch_works_from_other_cwd(self):
        args = self.args('--backend','local','--backend-root','/srv/checkout','doctor')
        command = launcher.build_command(args,host_platform='linux')
        self.assertEqual(command[-1],'scripts/baseline/agent_doctor.py')
        self.assertIn('/srv/checkout',command)


if __name__ == '__main__':
    unittest.main()
