"""Exact tool and source binding for reusable Poseidon packages."""
import hashlib,json
from pathlib import Path
from hecate_python_env import ROOT,WORK,digest
from poseidon_runtime_paths import SOURCE,DACAPO,BUILD,CPU_BUILD,POSEIDON_BUILD,PROFILE,SPEC,SOURCE_LOCK,BACKEND_SOURCE_ROOT,checked_sdk
def tree(path):
    path=Path(path)
    if not path.is_dir():
        raise ValueError("Missing backend source directory: "+str(path))
    entries={str(p.relative_to(path)):digest(p) for p in path.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.suffix!=".pyc"}
    if not entries:
        raise ValueError("Empty backend source directory: "+str(path))
    return hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def checked_hash(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024**2:
        raise ValueError("Missing or invalid backend configuration file: "+str(path))
    return digest(path)

def build_cache(directory, *, allow_parent=False):
    directory=Path(directory)
    candidates=[directory/"CMakeCache.txt"]
    if allow_parent:candidates.append(directory.parent/"CMakeCache.txt")
    for path in candidates:
        if path.exists():
            if not path.resolve().is_relative_to(WORK.resolve()):
                raise ValueError("Build configuration outside platform work root")
            return checked_hash(path)
    raise ValueError("Missing CMake configuration for configured backend: "+str(directory))

def configuration_identity():
    from platform_config import configuration
    tools="src/poseidon/tools/dacapo/"
    names=["scripts/baseline/platform-profiles.json",
           tools+"dependency-lock.json",tools+"hecate-toolchain-components.json",
           tools+"dacapo-dependencies.nix",tools+"poseidon-cpu-shell.nix",
           tools+configuration()["wheel_lock"]]
    return dict(environment_files={name:checked_hash(ROOT/name) for name in names},
                cmake_cache_sha256=dict(compiler=build_cache(BUILD),
                    executor=build_cache(CPU_BUILD),
                    poseidon=build_cache(POSEIDON_BUILD,allow_parent=True)))

def identity():
    from platform_config import identity as platform_identity
    sdk=checked_sdk()
    if sdk is not None and sdk.get('format')=='poseidon-runtime-toolchain-native-v5':
        from native_sdk_manifest_v5 import backend_identity
        return backend_identity(sdk,platform_identity())
    lock=SOURCE_LOCK if sdk is not None else ROOT/"scripts/baseline/poseidon-runtime-sources.lock.json"
    backend_root=BACKEND_SOURCE_ROOT if sdk is not None else ROOT
    value=dict(format="poseidon-backend-identity-v2",platform=platform_identity(),
        build_configuration=configuration_identity(),
        source_lock_sha256=digest(lock),source_lock=json.loads(lock.read_text()),
        runtime_dacapo_tree_sha256=tree(SOURCE),
        api_tree_sha256=tree(backend_root/"src/poseidon/runtime_api"),
        runner_tree_sha256=tree(backend_root/"src/poseidon/tools/agent_runtime"),
        compiler_sha256=digest(BUILD/"bin/hecate-opt"),frontend_sha256=digest(BUILD/"lib/libHecateFrontend.so"),
        executor_sha256=digest(CPU_BUILD/"poseidon_agent_cpu"),poseidon_library_sha256=digest(POSEIDON_BUILD/"libposeidon_shared.so"),
        compiler_profile_sha256=digest(PROFILE),operator_spec_sha256=digest(SPEC))
    if sdk is not None and 'native'in sdk:
        from native_sdk_manifest import execution_fingerprint
        native=sdk['native']
        value.update(executor_sha256=native['runtime']['sha256'],
            execution_profile='original-epoch-public-native-v1',
            native_sdk_execution_fingerprint=execution_fingerprint(sdk),
            native_runtime_sha256=native['runtime']['sha256'],native_client_sha256=native['client']['sha256'],
            native_capability_binding=native['capability']['binding'],
            native_key_epoch=read_native_epoch(native),
            inherited_base_executor_sha256=sdk['artifacts']['executor'])
    return value


def read_native_epoch(native):
    from native_sdk_manifest import EPOCH
    return EPOCH
