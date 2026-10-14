import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
import numpy as np
from pytorch_public_value_ops import plan
out=[]
for dtype,target in [("<f4","double"),("<f8","float"),("<i8","double")]:
 value=np.arange(1,7,dtype=dtype).reshape(2,3)
 try:plan(value,"call_method",target,("public",),{})
 except ValueError as e:
  assert str(e)=="public_expression_bool_cast"
  out.append(dict(dtype=dtype,target=target,rejected=str(e)))
 else:raise AssertionError("Original contract unexpectedly permits cast")
data=dict(format="poseidon-public-cast-original-rejection-v1",observations=out,
 provider_calls=0,compiler_calls=0,encrypted_calls=0)
with(HERE/"baseline_probe.json").open("x")as f:json.dump(data,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(data))
