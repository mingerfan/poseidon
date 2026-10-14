"""Machine-independent paths; imports never read credentials or create directories."""
import os
from pathlib import Path
import platform
import sys
from platform_config import configuration, require_platform


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
WORK_BASE = work_root()
WORK = WORK_BASE / configuration()['work_subdirectory']
RESULTS = WORK / 'results'


def require_linux_backend():
    """Only explicitly configured, matching Linux architectures may execute."""
    return require_platform(system=sys.platform, machine=platform.machine())


def nix_environment_options():
    # Only non-secret path configuration crosses the pure-shell boundary here.
    return [item for name in ('POSEIDON_WORK_ROOT', 'POSEIDON_PLATFORM')
            if name in os.environ for item in ('--keep', name)]


if __name__ == '__main__':
    print(WORK_BASE if sys.argv[1:] == ['--base'] else WORK)
