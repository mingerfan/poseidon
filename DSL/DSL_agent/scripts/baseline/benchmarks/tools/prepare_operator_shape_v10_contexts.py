"""Create explicit v10 manifests with unchanged owned models, weights and mathematics."""
import hashlib,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from pytorch_model_intake import OPERATOR_SHAPE_FORMAT,prepare
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run():
    old=ROOT/"scripts/baseline/cases/trusted-pytorch-operator-shape-v1"
    new=ROOT/"scripts/baseline/cases/trusted-pytorch-operator-shape-v10"
    require(not new.exists(),"Fresh v10 source inventory")
    previous=json.loads((old/"catalog.json").read_text())
    require(previous["binding"]==digest({k:v for k,v in previous.items() if k!="binding"}),"Original context inventory")
    new.mkdir();rows=[];preserved={str(p):sha(p) for p in old.rglob("*") if p.is_file()}
    for case in previous["cases"]:
        source=old/Path(case["manifest"]).parent;destination=new/Path(case["manifest"]).parent
        destination.mkdir()
        for path in source.iterdir():
            require(path.is_file() and not path.is_symlink(),"Flat owned source package")
            (destination/path.name).write_bytes(path.read_bytes())
        spec=json.loads((destination/"manifest.json").read_text())
        require(spec["format"]=="poseidon-trusted-pytorch-v9" and digest(spec)==case["manifest_sha256"],"Original manifest binding")
        spec["format"]=OPERATOR_SHAPE_FORMAT
        (destination/"manifest.json").write_bytes(canonical(spec));prepare(destination/"manifest.json")
        row=dict(case);row["manifest_sha256"]=digest(spec);rows.append(row)
        require(sha(new/case["graph"])==sha(old/case["graph"]),"Independent mathematical graph unchanged")
    require(all(sha(Path(p))==h for p,h in preserved.items()),"Original public-entry failures retained")
    catalog={k:v for k,v in previous.items() if k not in ("binding","cases")}
    catalog.update(format="poseidon-pytorch-operator-shape-contexts-v10",cases=rows,
        original_inventory_binding=previous["binding"],input_format=OPERATOR_SHAPE_FORMAT,
        mathematical_changes=0,source_changes=0,weight_changes=0,manifest_change="explicit version only")
    catalog["binding"]=digest(catalog);(new/"catalog.json").write_bytes(canonical(catalog))
    print(json.dumps(dict(path=str(new),source_variants=39,mathematical_groups=12,binding=catalog["binding"])))
if __name__=="__main__":run()
