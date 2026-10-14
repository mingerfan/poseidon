"""Offline policy tests; mocked architecture/ELF fixtures are not native execution."""
import copy
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import platform_config as config
import hecate_python_env as python_env
import hevm_abi
import candidate_sandbox
import workspace_paths
from test_portability import launcher


class PlatformTests(unittest.TestCase):
    def test_explicit_platform_matrix_and_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.configuration()['id'], 'x86_64-linux')
        for name, machine in [('x86_64-linux', 'x86_64'), ('aarch64-linux', 'aarch64')]:
            self.assertEqual(config.require_platform(name, system='linux', machine=machine)['id'], name)
            for wrong in ['riscv64', 'x86_64' if machine == 'aarch64' else 'aarch64']:
                with self.assertRaises(RuntimeError):
                    config.require_platform(name, system='linux', machine=wrong)
            with self.assertRaises(RuntimeError):
                config.require_platform(name, system='darwin', machine=machine)
        with self.assertRaises(ValueError):
            config.configuration('arbitrary-linux')

    def test_arm_work_is_separate_and_import_does_not_create_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()/'work'
            code = 'import workspace_paths as p; print(p.WORK_BASE); print(p.WORK)'
            for name, suffix in [('x86_64-linux', ''), ('aarch64-linux', 'platforms/aarch64-linux')]:
                env = dict(os.environ, POSEIDON_PLATFORM=name, POSEIDON_WORK_ROOT=str(root))
                result = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).parent,
                                        env=env, capture_output=True, text=True, check=True, timeout=10)
                self.assertEqual(result.stdout.splitlines(), [str(root), str(root/suffix)])
                self.assertFalse(root.exists())

    def test_unpinned_arm_launcher_cannot_execute(self):
        with patch.dict(os.environ, POSEIDON_PLATFORM='aarch64-linux'), \
                patch.object(config.sys, 'platform', 'linux'), \
                patch.object(config.platform, 'machine', return_value='aarch64'):
            if config.configuration()['nix_portable']['sha256'] is None:
                with self.assertRaisesRegex(RuntimeError, 'fingerprint'):
                    config.launcher_sha256()
            else:
                self.assertEqual(len(config.launcher_sha256()), 64)

    def test_exact_torch_versions_and_cpu_only(self):
        for name, machine in [('x86_64-linux', 'x86_64'), ('aarch64-linux', 'aarch64')]:
            with patch.dict(os.environ, POSEIDON_PLATFORM=name), \
                    patch.object(config.sys, 'platform', 'linux'), \
                    patch.object(config.platform, 'machine', return_value=machine):
                expected = config.configuration()['torch']
                torch = SimpleNamespace(__version__=expected, version=SimpleNamespace(cuda=None, hip=None))
                numpy = SimpleNamespace(__version__='1.25.2')
                config.require_python_packages(torch, numpy)
                torch.__version__ = '2.0.1' if expected.endswith('+cpu') else '2.0.1+cpu'
                with self.assertRaises(RuntimeError):
                    config.require_python_packages(torch, numpy)
                torch.__version__ = expected
                torch.version.cuda = '11.7'
                with self.assertRaises(RuntimeError):
                    config.require_python_packages(torch, numpy)

    def test_arm_selection_is_forwarded_without_shell_interpolation(self):
        args = launcher.parse_args(['--backend', 'ssh', '--ssh-host', 'ubuntu',
            '--backend-root', '/srv/project', '--platform', 'aarch64-linux', '--dry-run', 'doctor'])
        cmd = launcher.build_command(args, host_platform='darwin')
        self.assertEqual(shlex.split(cmd[-1])[:2], ['env', 'POSEIDON_PLATFORM=aarch64-linux'])

    def test_architecture_and_toolchain_profiles_are_independent(self):
        arm = config.configuration('aarch64-linux')
        self.assertEqual(arm['system'], 'aarch64-linux')
        self.assertEqual(arm['llvm_codegen_targets']['hecate'], [])
        self.assertEqual(arm['llvm_codegen_targets']['full'], ['AArch64'])

    def test_pure_shell_keeps_platform_but_not_secrets(self):
        with patch.dict(os.environ, {'POSEIDON_PLATFORM':'aarch64-linux',
                'POSEIDON_WORK_ROOT':'/data/work', 'DEEPSEEK_API_KEY':'synthetic'}, clear=True):
            self.assertEqual(workspace_paths.nix_environment_options(),
                             ['--keep','POSEIDON_WORK_ROOT','--keep','POSEIDON_PLATFORM'])
            self.assertEqual(config.nix_platform_options(), ['--argstr','platform','aarch64-linux'])

    def test_sandbox_preserves_base_path_and_mounts_only_platform_policy_files(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder).resolve()
            work = base/'platforms/aarch64-linux'
            output = work/'results/case'; output.mkdir(parents=True)
            payload = work/'results/payload.json'; payload.write_text('{}')
            with patch.object(candidate_sandbox, 'WORK', work), \
                    patch.object(candidate_sandbox, 'WORK_BASE', base), \
                    patch.dict(os.environ, POSEIDON_PLATFORM='aarch64-linux',
                               HECATE_PYTHON_LIBRARY_PATH='/nix/store/synthetic/lib'):
                cmd = candidate_sandbox.command(payload, output, ['fixed-worker'])
            values = {cmd[i+1]:cmd[i+2] for i,x in enumerate(cmd) if x == '--setenv'}
            self.assertEqual(values['POSEIDON_WORK_ROOT'], str(base))
            self.assertEqual(values['POSEIDON_PLATFORM'], 'aarch64-linux')
            self.assertNotIn('DEEPSEEK_API_KEY', values)
            self.assertIn('--unshare-all', cmd)
            self.assertIn('--clearenv', cmd)
            self.assertIn('/app/platform-profiles.json', cmd)


class WheelPlatformTests(unittest.TestCase):
    def test_both_locks_and_cross_architecture_rejection(self):
        base = Path(__file__).resolve().parents[2]/'src/poseidon/tools/dacapo'
        for name in ('x86_64-linux', 'aarch64-linux'):
            profile = config.configuration(name)
            pins = dict(python_env.PINS, torch=profile['torch'])
            lock = json.loads((base/profile['wheel_lock']).read_text())
            with patch.object(python_env, 'PLATFORM', profile), patch.object(python_env, 'PINS', pins):
                self.assertEqual(len(python_env.validate_lock(lock)), 9)
                wrong = copy.deepcopy(lock)
                wrong['wheels'][0]['filename'] = wrong['wheels'][0]['filename'].replace(
                    profile['machine'], 'aarch64' if name == 'x86_64-linux' else 'x86_64')
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    python_env.validate_lock(wrong)
                wrong = copy.deepcopy(lock)
                wrong['target'] = 'cp311-linux-aarch64'
                with self.assertRaisesRegex(ValueError, 'platform/Python'):
                    python_env.validate_lock(wrong)


class ABIGuardTests(unittest.TestCase):
    def test_cross_arch_and_wrong_endian_libraries_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'synthetic.so'
            raw = bytearray(20); raw[:6] = b'\x7fELF\x02\x01'
            for name, machine in [('x86_64-linux', 62), ('aarch64-linux', 183)]:
                struct.pack_into('<H', raw, 18, machine); path.write_bytes(raw)
                hevm_abi.check_elf(path, name)
                with self.assertRaises(RuntimeError):
                    hevm_abi.check_elf(path, 'aarch64-linux' if machine == 62 else 'x86_64-linux')
            raw[5] = 2; path.write_bytes(raw)
            with self.assertRaises(RuntimeError):
                hevm_abi.check_elf(path, 'aarch64-linux')

    def test_layout_alignment_padding_and_byte_order_are_checked(self):
        measured = dict(header_hex=struct.pack('<IIQQ', 0x4845564D, 24, 1, 2).hex(),
            body_hex=struct.pack('<5Q', 104, 3, 4, 5, 13).hex(),
            operation_hex=struct.pack('<4H', 1, 2, 3, 65533).hex(),
            sizes=[24,40,8], alignments=[8,8,2], pointer_bytes=8, int_bytes=4, double_bytes=8)
        hevm_abi.validate_layout(measured)
        for key, wrong in [('sizes', [32,40,8]), ('alignments', [4,8,2]),
                           ('operation_hex', struct.pack('>4H', 1,2,3,65533).hex())]:
            with self.assertRaises(RuntimeError):
                hevm_abi.validate_layout(dict(measured, **{key:wrong}))
