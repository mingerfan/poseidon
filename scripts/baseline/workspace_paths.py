"""Machine-independent paths; imports never read credentials or create directories."""
import os
from pathlib import Path
import platform
import sys


def work_root(value=None, *, home=None):
    home = Path.home() if home is None else Path(home)
    value = os.environ.get('POSEIDON_WORK_ROOT') if value is None else value
    path = Path(value).expanduser() if value else home / 'poseidon-work'
    if not path.is_absolute():
        raise ValueError('POSEIDON_WORK_ROOT must be an absolute path in the execution OS')
    path = path.resolve()
    if path == Path(path.anchor) or path == home.resolve():
        raise ValueError('POSEIDON_WORK_ROOT must be a dedicated directory, not root or home')
    return path


ROOT = Path(__file__).resolve().parent.parent.parent
WORK = work_root()
RESULTS = WORK / 'results'


def require_linux_backend():
    """Pinned binaries/wheels are Linux x86_64, not portable Python dependencies."""
    if sys.platform != 'linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError(
            'The pinned Dacapo/SEAL backend requires Linux x86_64. '
            'Use scripts/agent.py --backend wsl on Windows or --backend ssh '
            'to an Ubuntu x86_64 host/VM from macOS. ARM64 backend is not validated; '
            'no unsafe native or unsandboxed fallback is permitted.')


def nix_environment_options():
    # Only non-secret path configuration crosses the pure-shell boundary here.
    return ['--keep', 'POSEIDON_WORK_ROOT'] if 'POSEIDON_WORK_ROOT' in os.environ else []


if __name__ == '__main__':
    print(WORK)
