import os,sys,json,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])/"scripts/baseline"
sys.path.insert(0,str(BASE));os.environ["POSEIDON_PLATFORM"]="aarch64-linux";os.environ["POSEIDON_WORK_ROOT"]="/home/lhohy/poseidon-work"
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=60,keep_env=(),environment="poseidon-cpu"))
import torch
from test_pytorch_scalar_partitions import model,APPROX,PROFILE,SCALAR_POLICY
from pytorch_static_trace import trace,ALIAS_POLICY,INDEX_POLICY
from pytorch_einsum_extension import normalize as einsum
from pytorch_partition_normalization import normalize as partition
from pytorch_fused_matrix_normalization import normalize as fused
from pytorch_arithmetic_normalization import normalize_common,normalize
from pytorch_model_capture import capture
m=model("a,b=self.weight.unbind()"+chr(10)+"return x*a+b")
m.register_buffer("weight",torch.tensor([.2,.3],dtype=torch.float64))
sig=[dict(name="x",shape=[2])]
fx=trace(m,sig,resource_profile=PROFILE,policy=ALIAS_POLICY,fused_matrix_policy="poseidon-fx-fused-matrix-draft-v1",mha_source_policy="poseidon-mha-ordered-source-v1",partition_source_policy=SCALAR_POLICY,einsum_source_policy="poseidon-pytorch-static-einsum-v2")
fx,_=einsum(fx,sig,resource_profile=PROFILE)
fx,_=partition(fx,sig,resource_profile=PROFILE,source_policy=SCALAR_POLICY)
fx,_=fused(fx,sig,resource_profile=PROFILE)
fx,_=normalize_common(fx,sig,resource_profile=PROFILE,ordered_effects=True)
fx,arith=normalize(fx,resource_profile=PROFILE,preserve_unused_state=True,allow_ordered_mutations=True)
high,low,binding=capture(fx,sig,["out"],approximations=APPROX,resource_profile=PROFILE,dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True,capture_policy=INDEX_POLICY,matrix_policy="static-tensor-matmul-v1",approximation_policy="node-module-default-v1",functional_policy="static-adaptive-average-v1",capture_aliases=True,gelu_source_policy="poseidon-gelu-source-variants-v1",einsum_view_policy="poseidon-pytorch-static-einsum-v2",mha_source_policy="poseidon-mha-ordered-source-v1")
v=dict(arithmetic=arith,graph=binding["pytorch_capture"]["ordered_alias_capture"]["capture_emissions"])
with(HERE/"public-view-diagnostic.json").open("x")as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
