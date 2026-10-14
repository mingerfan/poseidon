import sys,json,tempfile
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
import torch
from pytorch_static_trace import trace
from pytorch_public_value_ops import CAST_POLICY
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("std",torch.tensor([.5,.75,1.25],dtype=torch.float64))
        self.register_buffer("mean",torch.tensor([.1,-.2,.05],dtype=torch.float64))
    def forward(self,x):
        return (x-self.mean)/(self.std+.01)
try:
    result=trace(Model().eval(),[dict(name="x",shape=[2,3])],
            resource_profile="general-tensors-v1",public_expression_policy=CAST_POLICY,
            common_calls=True,ordered_effects=True,public_tensor_division=True,public_boolean_arithmetic=True)
    value=dict(status="accepted",graph=str(result.graph))
except Exception as error:
    import traceback
    value=dict(status="rejected",type=type(error).__name__,reason=str(error),trace=traceback.format_exc())
Path(__file__).with_suffix(".json").write_text(json.dumps(value,indent=2))
print(json.dumps(value,indent=2))
