"""Cross-platform launcher for the existing Ubuntu Agent backend (Python >=3.10).

No implicit upload, installation, credential forwarding or paid test. All paths
after -- belong to the backend checkout. --dry-run never starts a process.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {'candidate': 'run_candidate.py', 'batch': 'run_agent_batch.py',
           'doctor': 'agent_doctor.py', 'inventory': 'dsl_semantic_inventory.py'}


def posix_absolute(value, name):
    if not value or not value.startswith('/') or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError(name + ' must be an absolute Linux path, not a host path or ~/ abbreviation')
    if '..' in PurePosixPath(value).parts or value == '/':
        raise ValueError(name + ' must identify a dedicated directory')
    return value


def build_command(args, *, host_platform=None):
    host = sys.platform if host_platform is None else host_platform
    backend = args.backend
    if backend == 'auto':
        backend = 'wsl' if host == 'win32' else 'local' if host == 'linux' else 'ssh'
    root = args.backend_root
    if backend == 'local':
        if host != 'linux':
            raise ValueError('Local FHE backend requires Linux; use WSL or SSH to Ubuntu')
        root = root or str(ROOT)
    elif backend == 'wsl':
        if host != 'win32':
            raise ValueError('WSL transport is only available from Windows; use SSH on macOS')
        if not root:
            raise ValueError('Set --backend-root to the checkout path inside WSL')
    elif not args.ssh_host or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:-]*', args.ssh_host):
        raise ValueError('SSH requires --ssh-host (a configured SSH host alias is recommended)')
    root = posix_absolute(root, '--backend-root')
    if args.work_root:
        posix_absolute(args.work_root, '--work-root')
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if '--inside' in forwarded:
        raise ValueError('--inside is an internal backend option, not a host launcher option')
    script = 'scripts/baseline/' + ENTRIES[args.command]
    platform_env = ['POSEIDON_PLATFORM=' + args.platform] if args.platform else []
    # Encode all user arguments as data, including whitespace/quotes/metacharacters.
    # The timeout executes INSIDE Linux so an SSH disconnect cannot remove it.
    code = ('import os,sys; os.chdir(sys.argv[1]); '
            'os.execvp("timeout", ["timeout", "-k", "10s", sys.argv[2]+"s", '
            '"env", "PYTHONDONTWRITEBYTECODE=1"] + '
            '(["POSEIDON_WORK_ROOT="+sys.argv[3]] if sys.argv[3] else []) + '
            '["python3", sys.argv[4]] + sys.argv[5:])')
    command = ['python3', '-c', code, root, str(args.timeout), args.work_root or '', script, *forwarded]
    if platform_env:
        command = ['env', *platform_env, *command]
    if backend == 'local':
        return command
    if backend == 'wsl':
        return ['wsl.exe', *(['--distribution', args.wsl_distro] if args.wsl_distro else []),
                '--exec', *command]
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3', args.ssh_host, shlex.join(command)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('auto','local','wsl','ssh'), default='auto')
    parser.add_argument('--backend-root', default=os.environ.get('POSEIDON_BACKEND_ROOT'))
    parser.add_argument('--work-root', default=None, help='Linux work root; otherwise backend environment/home default')
    parser.add_argument('--platform', choices=('x86_64-linux', 'aarch64-linux'), default=None,
                        help='Explicit backend architecture; otherwise use the backend environment/default x86')
    parser.add_argument('--wsl-distro', default=os.environ.get('POSEIDON_WSL_DISTRO'),
                        help='Omit to use the configured default WSL distribution')
    parser.add_argument('--ssh-host', default=os.environ.get('POSEIDON_SSH_HOST'))
    parser.add_argument('--timeout', type=int, default=43200, help='Whole job deadline, not API timeout')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('command', choices=tuple(ENTRIES))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 604800:
        parser.error('--timeout must be 1..604800 seconds')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        command = build_command(args)
        if args.dry_run:
            print(json.dumps({'argv': command, 'paid_calls': 0, 'files_transferred': 0}, indent=2))
            return 0
        # Credentials stay in the backend checkout's existing .env. No reading,
        # copying, SSH SendEnv or command-line key arguments are introduced here.
        return subprocess.run(command, timeout=args.timeout+30, check=False).returncode
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        print('Agent launcher: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
