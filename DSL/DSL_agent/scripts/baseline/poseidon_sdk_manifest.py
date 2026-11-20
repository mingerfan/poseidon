"""Opt-in, source/binary-bound SDK records for backend migration.

An explicit host-selected staging record permits acceptance testing. Its
qualification field remains pending until the independent acceptance succeeds.
No credential, download, installation or backend reconfiguration occurs here.
"""
import hashlib
import json
from pathlib import Path

# The original hamming=0 sampler produced a zero secret. B's r2513
# certificate rejects this exact compiled library; historical records stay intact.
REJECTED_KEYGEN_LIBRARY_SHA256 = '11e6233cc39359edd54924d7201433671b892cdf758990f2a8477a2696f9fc4c'

FORMAT = 'poseidon-runtime-toolchain-v2'
FIELDS = {'format', 'platform', 'backend_source_root', 'runtime_source', 'compiler_build',
          'executor_build', 'poseidon_build', 'source_lock', 'source_lock_sha256',
          'resource_policy', 'resource_policy_sha256', 'source_trees', 'artifacts',
          'cmake_caches', 'build_receipts', 'environment_files', 'qualification', 'binding'}


def require(ok, message):
    if not ok:
        raise ValueError('SDK binding: ' + message)


def reject_known_unsafe_key_generator(value):
    artifacts = value.get('artifacts')
    if type(artifacts) is dict:
        require(artifacts.get('poseidon_library') != REJECTED_KEYGEN_LIBRARY_SHA256,
                'rejected zero-hamming key generator (r2513)')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular artifact required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''):
            h.update(part)
    return h.hexdigest()


def path(value, *, within=None, directory=False):
    require(type(value) is str, 'path must be a string')
    p = Path(value)
    require(p.is_absolute() and '..' not in p.parts, 'relative path')
    require(within is None or p.is_relative_to(Path(within).resolve()), 'artifact outside selected work root')
    require(p.is_dir() if directory else p.is_file(), 'missing SDK input')
    return p


def read_json(p, maximum=256*1024):
    p = Path(p)
    require(p.suffix == '.json' and p.is_file() and not p.is_symlink() and p.stat().st_size <= maximum,
            'JSON size/type')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(p.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, 'nonfinite JSON'))


def tree(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'source directory required')
    entries = {}
    for item in root.rglob('*'):
        rel = item.relative_to(root)
        if set(rel.parts).intersection({'.git', '__pycache__'}) or item.suffix == '.pyc':
            continue
        require(not item.is_symlink(), 'indirect source entry')
        if item.is_file():
            entries[str(rel)] = file_hash(item)
    require(bool(entries), 'empty source tree')
    return digest(entries)


def load(filename, work, platform):
    filename = path(str(filename), within=work)
    value = read_json(filename)
    require(type(value) is dict and set(value) == FIELDS, 'descriptor fields')
    require(value['format'] == FORMAT and value['platform'] == platform, 'format/platform')
    require(value['qualification'] in ('pending_agent_acceptance', 'minimal_cpu_agent_verified'), 'qualification scope')
    reject_known_unsafe_key_generator(value)
    source_root = path(value['backend_source_root'], directory=True)
    runtime = path(value['runtime_source'], directory=True)
    require(runtime == source_root / 'third_party/ckks-runtime', 'runtime/source root association')
    for name in ('compiler_build', 'executor_build', 'poseidon_build'):
        path(value[name], within=work, directory=True)
    for name in ('source_lock', 'resource_policy'):
        p = path(value[name], within=work)
        read_json(p)
    require(set(value['source_trees']) == {'runtime', 'api', 'runner'}, 'source tree fields')
    require(set(value['artifacts']) == {'compiler', 'frontend', 'executor', 'poseidon_library'}, 'artifact fields')
    require(set(value['cmake_caches']) == {'compiler', 'executor', 'poseidon'}, 'build configuration fields')
    require(set(value['build_receipts']) == {'compiler', 'executor'}, 'build receipt fields')
    require(type(value['environment_files']) is dict and 1 <= len(value['environment_files']) <= 16,
            'environment source fields')
    return value


def attest(value, work, agent_root):
    """Hash attestation removed on 2026-10-20 at the user's request; paths are checked by load().
    The known-unsafe key generator rejection is a security check and stays."""
    reject_known_unsafe_key_generator(value)
    return value
