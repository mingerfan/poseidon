"""Reproduce versioned static-matrix examples into a fresh directory."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import require
from pytorch_source_contract import MATRIX_FORMAT
from pytorch_weight_package import export_state

def create(output):
    output=Path(output)
    require(not output.exists() and not output.is_symlink(),"Fresh matrix fixture directory required")
    source_path=ROOT/"scripts/baseline/cases/trusted-pytorch-matrix-v7/models.py"
    source=source_path.read_bytes();source_hash=hashlib.sha256(source).hexdigest()
    spec=importlib.util.spec_from_file_location("matrix_fixture_models",source_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output.mkdir(mode=0o700)
    (output/"models.py").write_bytes(source)
    rows=[]
    for partition in module.PARTITIONS:
        for index in range(3):
            structure="dict" if partition=="cipher_batch" and index==1 else "tuple" if partition=="public_batch_l" and index==1 else "tensor"
            model=module.make_model(partition,index,structure)
            case=output/(partition+"_"+str(index));case.mkdir()
            weights=export_state(model,case,resource_profile="general-tensors-v1") if partition.startswith("public_") else {}
            (case/"models.py").write_bytes(source)
            manifest=dict(format=MATRIX_FORMAT,files={"models.py":source_hash},entry="models.make_model",
                factory=dict(args=[partition,index],kwargs=dict(structure=structure)),
                inputs=module.signature(partition,index),outputs=["out","extra"] if structure!="tensor" else ["out"],
                approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
                dtype_policy="strict-float64-v1",weights=weights)
            (case/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            rows.append(dict(id=case.name,manifest_sha256=hashlib.sha256((case/"manifest.json").read_bytes()).hexdigest()))
    return dict(format="poseidon-matrix-fixture-generation-v1",models=len(rows),rows=rows,source_sha256=source_hash,
                paid_calls=0,encrypted_execution=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(create(args.output),indent=2))
if __name__=="__main__":main()
