"""Read-only normalization reconstruction; no compiler/runtime/API."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(HERE.parent/"staged_mha_ordered_graph_v1/overlay"))
from pytorch_alias_capture_bridge import normalize_emissions
from benchmark_graph import canonical,digest
from candidate_bundle import load_model_provenance
name="boolean_padding_residual__declared_shallow_v1"
base=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
capture=base/"agent-mha-ordered-capture-v2"/name
saved=base/"agent-mha-replicated-scalar-backend-v1"/name/"program/model-provenance.json"
pairs=[("in_memory_reloaded_capture",load_model_provenance(capture,json.loads((capture/"model.json").read_bytes()),max_bytes=4*1024**2)),(str(saved),json.loads(saved.read_bytes()))]
for path,value in pairs:
 metadata=value["reference_report"]["capture_metadata"];alias=metadata["ordered_alias_capture"]
 rebuilt,lowered,_,regenerated=normalize_emissions(alias["capture_emissions"],alias["capture_source_origins"],resource_profile="general-tensors-v1",constant_dtypes=alias["capture_emission_constant_dtypes"],state_constant_sources=alias["capture_emission_state_constant_sources"],public_constant_views=alias["capture_public_constant_views"])
 a={k:v for k,v in alias.items() if k not in ("capture_adapter","source_capture_policy","binding")}
 b={k:v for k,v in regenerated.items() if k!="binding"}
 print("FILE",path,"projection_match",canonical(a)==canonical(b),"original_match",canonical(rebuilt)==canonical(value["original"]),"sha",digest(value),flush=True)
 differences=[]
 def diff(x,y,path):
  if type(x)!=type(y):differences.append([path,"type",str(type(x)),str(type(y))]);return
  if isinstance(x,dict):
   for k in sorted(set(x)|set(y)):
    if k not in x or k not in y:differences.append([path+"/"+k,"key",k in x,k in y])
    else:diff(x[k],y[k],path+"/"+k)
  elif isinstance(x,list):
   if len(x)!=len(y):differences.append([path,"length",len(x),len(y)])
   for i,(u,v) in enumerate(zip(x,y)):diff(u,v,path+"/"+str(i))
  elif x!=y:differences.append([path,"value",str(x)[:350],str(y)[:350]])
 diff(a,b,"capture");diff(value["original"],rebuilt,"graph")
 print(json.dumps({"count":len(differences),"differences":differences[:45]},sort_keys=True),flush=True)
