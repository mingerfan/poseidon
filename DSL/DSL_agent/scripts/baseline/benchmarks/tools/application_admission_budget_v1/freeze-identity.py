import sys,json
from pathlib import Path
sys.path.insert(0,str(Path("scripts/baseline").resolve()))
from application_identity import current_identity_manifest
value=current_identity_manifest()
path=Path("scripts/baseline/benchmarks/tools/application_admission_budget_v1/current-sdk-r1382.json")
with path.open("x") as stream:json.dump(value,stream,sort_keys=True,indent=2)
print(value["manifest_sha256"])
