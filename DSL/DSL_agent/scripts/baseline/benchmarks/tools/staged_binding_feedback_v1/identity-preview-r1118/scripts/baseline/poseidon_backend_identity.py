"""Exact tool and source binding for reusable Poseidon packages."""
import hashlib,json
from pathlib import Path
from hecate_python_env import ROOT,WORK,digest
from poseidon_runtime_paths import SOURCE,DACAPO,BUILD,CPU_BUILD,POSEIDON_BUILD,PROFILE,SPEC
def tree(path):
    entries={str(p.relative_to(path)):digest(p) for p in path.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.suffix!=".pyc"}
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
    lock=ROOT/"scripts/baseline/poseidon-runtime-sources.lock.json"
    return dict(format="poseidon-backend-identity-v2",platform=platform_identity(),
        build_configuration=configuration_identity(),
        source_lock_sha256=digest(lock),source_lock=json.loads(lock.read_text()),
        runtime_dacapo_tree_sha256=tree(SOURCE),
        api_tree_sha256=tree(ROOT/"src/poseidon/runtime_api"),
        runner_tree_sha256=tree(ROOT/"src/poseidon/tools/agent_runtime"),
        compiler_sha256=digest(BUILD/"bin/hecate-opt"),frontend_sha256=digest(BUILD/"lib/libHecateFrontend.so"),
        executor_sha256=digest(CPU_BUILD/"poseidon_agent_cpu"),poseidon_library_sha256=digest(POSEIDON_BUILD/"libposeidon_shared.so"),
        compiler_profile_sha256=digest(PROFILE),operator_spec_sha256=digest(SPEC))
