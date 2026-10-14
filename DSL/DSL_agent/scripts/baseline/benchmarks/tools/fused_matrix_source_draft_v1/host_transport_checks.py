"""Data-only FX checks against actual captured source records."""
import copy,hashlib,importlib.abc,json,platform,sys,time
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline";HERE=Path(__file__).resolve().parent
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
if len(sys.argv)!=3:raise SystemExit("Usage: host_transport_checks.py SOURCE_RESULT FRESH_HOST_RESULT")
source_name,out_name=sys.argv[1:]
for value,prefix in ((source_name,"agent-fused-matrix-intake-r"),(out_name,"agent-fused-matrix-host-checks-r")):
    if not value.startswith(prefix) or not value.removeprefix(prefix).isdigit():raise SystemExit("Numeric result names")
out=WORK/"results"/out_name
if out.exists():raise SystemExit("Fresh host result required")
class DenyModelLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in ("torch","numpy","source_models","source_helpers"):
            raise AssertionError("Host check attempted source or numerical-library import")
sys.meta_path.insert(0,DenyModelLibraries())
sys.path.insert(0,str(HERE/"targets"));sys.path.insert(1,str(BASE))
from pytorch_fused_matrix_normalization import check_transport
from benchmark_graph import digest
start=time.monotonic();positive=[];negative=[]
files=sorted((WORK/"results"/source_name).glob("source-worker-*/capture-reference.json"))
if len(files)!=9:raise SystemExit("Expected nine retained source examples")
source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
for path in files:
    if path.is_symlink() or path.stat().st_size>4*1024**2:raise SystemExit("Retained regular source record")
    report=json.loads(path.read_text())
    record=report["capture_metadata"]["ordered_alias_capture"]["capture_adapter"]["fused_matrix_normalization"]
    check_transport(record,inputs=record["input_signature"],resource_profile=record["resource_profile"])
    positive.append(dict(case=path.parent.name,record_sha256=digest(record),status="passed"))
    for mode in ("operation","axis","missing_step","coefficient","bad_type"):
        bad=copy.deepcopy(record)
        if mode=="operation":
            for n in bad["normalized_inventory"]:
                if n["target"] in ("torch.mm","torch.bmm"):n["target"]="torch.add";break
        elif mode=="axis":bad["entries"][0]["plan"]["reduction_axis"]=1
        elif mode=="missing_step":bad["entries"][0]["emitted_nodes"].pop(0)
        elif mode=="coefficient":bad["entries"][0]["plan"]["alpha"]=.875
        else:bad["input_signature"][0]["shape"]=[True]
        bad["binding_sha256"]=digest({k:v for k,v in bad.items() if k!="binding_sha256"})
        try:check_transport(bad,inputs=record["input_signature"],resource_profile=record["resource_profile"])
        except ValueError:negative.append(dict(case=path.parent.name,mutation=mode,status="rejected"))
        else:raise AssertionError("Invalid record accepted")
if any(n.split(".")[0] in ("torch","numpy","source_models","source_helpers") for n in sys.modules):
    raise AssertionError("Host library boundary")
if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=w for p,w in source_hashes.items()):
    raise AssertionError("Retained evidence changed")
receipt=dict(format="poseidon-fused-matrix-data-host-checks-v1",status="passed",
             tests=dict(passed=len(positive)+len(negative),failed=0,errors=0,skipped=0),
             positive_records=positive,rejections=negative,python=platform.python_version(),
             source_records=source_hashes,model_source_imports=0,numerical_library_imports=0,
             compiler_calls=0,encrypted_executions=0,provider_calls=0,
             implementation_sha256=hashlib.sha256((HERE/"targets/pytorch_fused_matrix_normalization.py").read_bytes()).hexdigest(),
             checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.monotonic()-start)
receipt["binding"]=digest(receipt)
out.mkdir(mode=0o700);(out/"report.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(result=str(out),tests=receipt["tests"],seconds=receipt["seconds"])))
