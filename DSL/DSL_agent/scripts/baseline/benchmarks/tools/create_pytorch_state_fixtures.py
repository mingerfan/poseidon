"""Reproduce small, public state-weight fixtures. Refuse existing destinations."""
import hashlib, importlib.util, json, sys
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[6] if Path(__file__).resolve().parents[4].name == "DSL_agent" and Path(__file__).resolve().parents[5].name == "DSL" else Path(__file__).resolve().parents[4])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from pytorch_weight_package import export_state
from benchmark_graph import require

MATRIX_SOURCE = '''
class WeightedMatrixFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(.125,dtype=torch.float64))
    def forward(self,x,y):
        product=torch.matmul(x,y)*self.scale
        return {"product":product,"branches":[x+y,x-y]}
def make_weighted_matrix(*,scale):
    if scale!=1: raise ValueError("The fixture uses a unit configured scale")
    return WeightedMatrixFusion()
'''

def create():
    import torch
    before=ROOT/"scripts/baseline/cases/trusted-pytorch-general-v2"
    target=ROOT/"scripts/baseline/cases/trusted-pytorch-state-v3"
    require(not target.exists(),"Preserve existing state fixtures")
    source=(before/"models.py").read_text()+MATRIX_SOURCE
    spec=importlib.util.spec_from_file_location("state_fixture_models",before/"models.py")
    models=importlib.util.module_from_spec(spec);spec.loader.exec_module(models)
    target.mkdir()
    rows=[]
    for index,case in enumerate(("mlp","conv_pool","residual","batchnorm","matrix_fusion","shape_linear")):
        manifest=json.loads((before/(case+".json")).read_text())
        entry=manifest["entry"].split(".")[1]
        if case=="matrix_fusion":entry="make_weighted_matrix"
        factory=manifest["factory"]
        if case=="matrix_fusion":
            class WeightedMatrix(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.125,dtype=torch.float64))
                def forward(self,x,y):
                    return {"product":torch.matmul(x,y)*self.scale,"branches":[x+y,x-y]}
            module=WeightedMatrix().eval()
        else:module=getattr(models,entry)(*factory["args"],**factory["kwargs"]).eval()
        generator=torch.Generator().manual_seed(9000+index)
        changed=0
        with torch.no_grad():
            for name,tensor in module.state_dict().items():
                if tensor.is_floating_point():
                    random=torch.rand(tensor.shape,generator=generator,dtype=tensor.dtype)
                    if name.endswith("running_var"):value=random*.25+.75
                    elif name.endswith("running_mean"):value=(random-.5)*.125
                    elif case=="batchnorm" and name.endswith("weight"):value=random*.125+.875
                    else:value=(random-.5)*.25
                    if not torch.equal(tensor,value):changed+=tensor.numel()
                    tensor.copy_(value)
                elif name.endswith("num_batches_tracked"):tensor.fill_(7)
        require(changed>0,"External fixture weights must differ from factory")
        folder=target/case
        records=export_state(module,folder,resource_profile="general-tensors-v1")
        (folder/"models.py").write_text(source)
        manifest.update(format="poseidon-trusted-pytorch-v3",entry="models."+entry,
            files={"models.py":hashlib.sha256(source.encode()).hexdigest()},weights=records)
        (folder/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        rows.append(dict(id=case,state_tensors=len(records),changed_elements=changed,
                         declared_bytes=sum(v["bytes"] for v in records.values())))
    (target/"index.json").write_text(json.dumps(dict(format="poseidon-pytorch-state-fixtures-v1",cases=rows),indent=2)+"\n")
    print(json.dumps(rows))
if __name__=="__main__":create()
