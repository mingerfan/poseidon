"""Create deterministic test-only mixed-type fixtures in a fresh draft directory."""
import hashlib,importlib.util,json,os,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE));os.environ.update(POSEIDON_PLATFORM="aarch64-linux",POSEIDON_WORK_ROOT="/home/lhohy/poseidon-work")
def inside():
 from hecate_python_env import VENV
 from pytorch_weight_package import export_state
 assert os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV
 fixtures=HERE/"fixtures-v2";assert not fixtures.exists();fixtures.mkdir()
 definition=HERE/"fixture_model.py";raw=definition.read_bytes()
 spec=importlib.util.spec_from_file_location("controlled_fixture",definition)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 cases=json.loads((HERE.parent/"public_tensor_source_v1/cases.json").read_text())
 for name in ["batch_norm_module","batch_norm_function","index_select_function","index_method_nested","bool_multiply","mask_index_residual","divided_relu_residual","divided_gelu_linear","divided_sigmoid_tuple","float32_vector","float32_batch_norm","float32_gelu_projection"]:
  case=fixtures/name;case.mkdir()
  instance=module.make(name);weights=export_state(instance,case,resource_profile="general-tensors-v1")
  (case/"model.py").write_bytes(raw)
  manifest=dict(format="poseidon-trusted-pytorch-v21",entry="model.make",factory=dict(args=[name],kwargs={}),
   files={"model.py":hashlib.sha256(raw).hexdigest()},inputs=[dict(name="x",shape=[2,3])],
   outputs=["selected","masked"] if name=="mask_index_residual" else ["quotient","activated"] if name=="divided_sigmoid_tuple" else ["out"],
   dtype_policy="convert-float64-v1"if name.startswith("float32_")else"strict-float64-v1",weights=weights,
   approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]))
  nonlinear={"divided_relu_residual":("relu",[.25,.5,.25]),
   "divided_gelu_linear":("gelu",[0.,.5,.3989422804014327]),
   "divided_sigmoid_tuple":("sigmoid",[.5,.25,0.,-1./48.]),
   "float32_gelu_projection":("gelu",[0.,.5,.3989422804014327])}
  if name in nonlinear:
   function,coefficients=nonlinear[name]
   manifest["approximations"]["defaults"][function]=dict(method="polynomial",basis="power",
    domain=[-4.,4.],coefficients=coefficients)
  path=case/"manifest.json";path.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False))
  cases.append(dict(case=name,manifest=str(path.relative_to(ROOT)),
   manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),purpose="mixed-public-state source compatibility"))
 (HERE/"cases.json").write_text(json.dumps(cases,indent=2,sort_keys=True,allow_nan=False))
 print(json.dumps(dict(fixtures_created=12,captures_planned=len(cases),production_activation=False,provider_calls=0)))
 return 0
def outer():
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 return enter_nix(cmd,seconds=117,keep_env=(),environment="poseidon-cpu")
if __name__=="__main__":raise SystemExit(inside() if sys.argv[1:]==["--inside"] else outer())
