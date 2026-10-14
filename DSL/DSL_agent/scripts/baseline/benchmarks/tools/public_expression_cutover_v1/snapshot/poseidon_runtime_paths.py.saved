"""Platform-local configured tool paths; source archives are version locked."""
import json
from pathlib import Path
from hecate_python_env import ROOT,WORK
SOURCE=ROOT/"third_party/ckks-runtime-agent"
DACAPO=SOURCE/"third_party/dacapo"
PROFILE=ROOT/"scripts/baseline/poseidon-compiler-profile-v1.json"
SPEC=ROOT/"scripts/baseline/poseidon-operator-spec-v1.json"
MANIFEST=WORK/"deps/poseidon-runtime-v1.json"
def read_paths():
    if not MANIFEST.exists():
        return dict(compiler_build=WORK/"build-dacapo/runtime-plan-v1",executor_build=WORK/"build-poseidon/agent-runtime-v1",poseidon_build=WORK/"build-poseidon/cpu-v1/poseidon_build")
    if not MANIFEST.is_file() or MANIFEST.is_symlink() or MANIFEST.stat().st_size>16384:
        raise RuntimeError("Configure the platform-local Poseidon runtime toolchain: "+str(MANIFEST))
    data=json.loads(MANIFEST.read_text())
    from platform_config import configuration
    if set(data)!={"format","platform","compiler_build","executor_build","poseidon_build"} or data["format"]!="poseidon-runtime-toolchain-v1" or data["platform"]!=configuration()["id"]:
        raise RuntimeError("Poseidon toolchain platform/fields mismatch")
    paths={}
    for key in ("compiler_build","executor_build","poseidon_build"):
        p=Path(data[key])
        if not p.is_absolute() or not p.resolve().is_relative_to(WORK.resolve()):
            raise RuntimeError("Poseidon binary path outside selected platform work root")
        paths[key]=p
    return paths
_paths=read_paths()
BUILD=_paths["compiler_build"];CPU_BUILD=_paths["executor_build"];POSEIDON_BUILD=_paths["poseidon_build"]
