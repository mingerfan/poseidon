"""Optional upstream helper dependency verification; never installs implicitly.

The approved wheel lives outside the tracing venv. Verification compares every
installed member to the hash-locked wheel; installation.json is informational.
Candidates cannot import this module or arbitrary poly helpers.
"""
import hashlib
import json
from pathlib import Path
import zipfile
from workspace_paths import ROOT, WORK

LOCK=ROOT/"src/poseidon/tools/dacapo/poly-python-wheels.lock.json"
TARGET=WORK/"deps/poly-python/einops-0.6.1"
CACHE=WORK/"cache/poly-python/einops-0.6.1-py3-none-any.whl"

def verify():
    spec=json.loads(LOCK.read_text())["wheels"][0]
    raw=CACHE.read_bytes()
    if len(raw)!=spec["bytes"] or hashlib.sha256(raw).hexdigest()!=spec["sha256"]:
        raise ValueError("Optional helper wheel hash/size mismatch")
    members={}
    with zipfile.ZipFile(CACHE) as wheel:
        for item in wheel.infolist():
            relative=Path(item.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe wheel path")
            if item.is_dir():continue
            path=TARGET/relative
            if path.is_symlink() or not path.is_file() or path.read_bytes()!=wheel.read(item):
                raise ValueError("Installed helper dependency changed: "+item.filename)
            members[item.filename]=hashlib.sha256(path.read_bytes()).hexdigest()
    actual={str(p.relative_to(TARGET)) for p in TARGET.rglob("*") if p.is_file()}
    if actual-set(members)-{"installation.json"}:
        raise ValueError("Unexpected files in optional helper dependency")
    return dict(version=spec["version"],target=str(TARGET),wheel_sha256=spec["sha256"],
                files_sha256=members,lock_sha256=hashlib.sha256(LOCK.read_bytes()).hexdigest())

if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
