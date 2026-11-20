"""Platform-local configured tool paths; source archives are version locked."""
import json,os
from pathlib import Path
from hecate_python_env import ROOT,WORK
SOURCE=ROOT/"third_party/ckks-runtime-agent"
DACAPO=SOURCE/"third_party/dacapo"
PROFILE=ROOT/"scripts/baseline/poseidon-compiler-profile-v1.json"
SPEC=ROOT/"scripts/baseline/poseidon-operator-spec-v1.json"
MANIFEST=Path(os.environ["POSEIDON_SDK_MANIFEST"]) if "POSEIDON_SDK_MANIFEST" in os.environ else WORK/"deps/poseidon-runtime-v1.json"
_EXPLICIT_SDK = "POSEIDON_SDK_MANIFEST" in os.environ
_SDK_DATA = None

def selected_sdk():
    if not _EXPLICIT_SDK:return None
    from poseidon_sdk_manifest import load,read_json,path
    from platform_config import configuration
    raw=read_json(path(str(MANIFEST),within=WORK))
    if raw.get('format')in('poseidon-runtime-toolchain-native-v4','poseidon-runtime-toolchain-native-v5'):
        from native_sdk_manifest import load as native_load
        data=native_load(MANIFEST,WORK,configuration()["id"],ROOT)
    else:
        data=load(MANIFEST,WORK,configuration()["id"])
    if _SDK_DATA is not None and data!=_SDK_DATA:
        raise ValueError("SDK binding: descriptor changed during worker")
    return data

def checked_sdk():
    data=selected_sdk()
    if data is not None:
        from poseidon_sdk_manifest import attest
        if 'native'in data:
            from native_sdk_manifest import attest as native_attest
            native_attest(data,WORK,ROOT)
        else:
            attest(data,WORK,ROOT)
    return data

def read_paths():
    selected=selected_sdk()
    if selected is not None:
        return {name:Path(selected[name]) for name in ("compiler_build","executor_build","poseidon_build")}
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

_SDK_DATA=selected_sdk()
SOURCE_LOCK=ROOT/"scripts/baseline/poseidon-runtime-sources.lock.json"
BACKEND_SOURCE_ROOT=ROOT
if _SDK_DATA is not None:
    SOURCE=Path(_SDK_DATA["runtime_source"])
    DACAPO=SOURCE/"third_party/dacapo"
    SOURCE_LOCK=Path(_SDK_DATA["source_lock"])
    BACKEND_SOURCE_ROOT=Path(_SDK_DATA["backend_source_root"])
    if 'native'in _SDK_DATA:
        PROFILE=Path(_SDK_DATA['native']['compiler_profile']['path'])
        SPEC=Path(_SDK_DATA['native']['operator_spec']['path'])
