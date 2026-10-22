"""Explicit backend identity. Importing this module never provisions an environment."""
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys

MANIFEST = Path(__file__).with_name('platform-profiles.json')
DEFAULT = 'x86_64-linux'


def configuration(name=None):
    name = os.environ.get('POSEIDON_PLATFORM', DEFAULT) if name is None else name
    profiles = json.loads(MANIFEST.read_text())['platforms']
    if name not in profiles:
        raise ValueError('Unconfigured POSEIDON_PLATFORM: ' + str(name))
    return dict(profiles[name], id=name)


def require_platform(name=None, *, system=None, machine=None):
    selected = configuration(name)
    system = sys.platform if system is None else system
    machine = platform.machine().lower() if machine is None else machine.lower()
    if system != 'linux' or machine not in selected['machine_aliases']:
        raise RuntimeError('Backend requires Linux ' + selected['machine'] +
                           '; selected ' + selected['id'] + ', actual ' + system + '/' + machine)
    if sys.byteorder != 'little' or struct.calcsize('P') != 8:
        raise RuntimeError('Backend requires a little-endian 64-bit process')
    return selected


def identity():
    selected = configuration()
    return dict(configured_platform=selected['id'], system=sys.platform,
                machine=platform.machine(), byteorder=sys.byteorder,
                pointer_bits=struct.calcsize('P') * 8,
                profile_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                validation_status=selected['validation_status'])


def loaded_libraries():
    """Paths actually mapped in this process, not merely LD_LIBRARY_PATH hints."""
    maps = Path('/proc/self/maps')
    if sys.platform != 'linux' or not maps.is_file():
        raise RuntimeError('Linux process mappings are required for library evidence')
    return sorted({fields[5] for line in maps.read_text().splitlines()
                   if len(fields := line.split(maxsplit=5)) == 6
                   and fields[5].startswith('/') and '.so' in fields[5]})


def require_python_packages(torch, numpy):
    selected = require_platform()
    if (torch.__version__ != selected['torch'] or numpy.__version__ != selected['numpy']
            or torch.version.cuda is not None or getattr(torch.version, 'hip', None) is not None):
        raise RuntimeError('Python packages do not match the exact CPU platform pins')


def nix_platform_options():
    return ['--argstr', 'platform', configuration()['id']]


def launcher_sha256():
    selected = require_platform()
    digest = selected['nix_portable']['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise RuntimeError('Launcher fingerprint not established for ' + selected['id'] +
                           '; approve provisioning and review its full hash before execution')
    return digest


if __name__ == '__main__':
    if sys.argv[1:] == ['--launcher-sha256']:
        print(launcher_sha256())
    elif sys.argv[1:] == ['--require']:
        print(require_platform()['id'])
    else:
        print(json.dumps(identity(), indent=2))
