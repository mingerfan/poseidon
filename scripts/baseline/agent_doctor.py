"""Read-only, credential-free platform/path preflight; never runs Nix or downloads."""
import json
import platform
import shutil
from workspace_paths import ROOT, WORK, require_linux_backend
from platform_config import identity, launcher_sha256


def inspect():
    errors = []
    try:
        require_linux_backend()
        launcher_sha256()
    except RuntimeError as error:
        errors.append(str(error))
    files = {
        'dacapo_frontend': ROOT/'third_party/dacapo/python/hecate/hecate/__init__.py',
        'nix_portable': WORK/'deps/nix-portable-v012/nix-portable',
        'python_venv': WORK/'venvs/hecate-2.0.1-cpu/bin/python',
        'hecate_opt': WORK/'build-dacapo/hecate-18.1.2-nix/bin/hecate-opt',
        'hevm_abi': WORK/'build-dacapo/hecate-18.1.2-nix/hevm-abi.json',
        'seal_metadata': WORK/'build-dacapo/seal-golden-keys/libseal_golden_metadata.so',
    }
    for name, path in files.items():
        # nix-portable's /nix/store is visible only inside its mount namespace.
        # A venv link may be unresolved here; real loading is a separate gate.
        if not path.is_file() and not (name == 'python_venv' and path.is_symlink()):
            errors.append('Missing ' + name + ': ' + str(path))
    commands = {name: shutil.which(name) for name in ('bash','timeout','flock','bwrap','git')}
    errors.extend('Missing command: '+name for name,value in commands.items() if not value)
    return dict(platform_identity=identity(), source_root=str(ROOT), work_root=str(WORK), platform=platform.system(),
                machine=platform.machine(), files={k:str(v) for k,v in files.items()},
                commands=commands, path_prerequisites_present=not errors, errors=errors,
                compiler_execution_validated=False, encrypted_execution_validated=False,
                note='Presence check only; Nix-backed symlinks require the isolated shell to resolve.',
                credentials_read=False, downloads=0, paid_calls=0)


if __name__ == '__main__':
    report = inspect()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['path_prerequisites_present'] else 2)
