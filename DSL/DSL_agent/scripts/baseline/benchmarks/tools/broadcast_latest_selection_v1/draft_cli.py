#!/usr/bin/env python3
"""Create a trusted-local PyTorch source manifest from files and a static signature."""
import argparse,ast,hashlib,json,math,os,re,struct,sys
from pathlib import Path
BASE=Path("/home/lhohy/Code/Poseidon/scripts/baseline")
sys.path.insert(0,str(BASE))
from benchmark_graph import NAME,require,canonical
from model_resources import resolve,PROFILES
from model_weight_package import regular_bytes
from pytorch_ordered_contract import SOURCE_FORMATS,shim,validate_package,implementation_paths,RESERVED
from pytorch_source_contract import FLOAT_POLICIES
from pytorch_weight_package import declarations,DTYPES
import trusted_pytorch_import as legacy
DEFAULT_FORMAT="poseidon-trusted-pytorch-v18"
LATEST_FORMAT="poseidon-trusted-pytorch-v28"
SOURCE_SELECTIONS=(*SOURCE_FORMATS,"latest")
def resolve_source_format(value):
 require(type(value)is str,"Source version selection")
 result=LATEST_FORMAT if value=="latest" else value
 require(result in SOURCE_FORMATS,"Source version selection")
 return result
EMPTY_APPROXIMATIONS=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[])
def strict_json(raw,limit=131072):
 require(type(raw)is str and len(raw.encode())<=limit,"JSON data byte limit")
 def pairs(items):
  result={}
  for name,value in items:
   require(name not in result,"Duplicate JSON field");result[name]=value
  return result
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError("Nonfinite JSON")))
def regular_path(value):
 path=Path(os.path.abspath(value))
 require(path==path.resolve()and path.is_file()and
  not any(p.is_symlink()for p in(path,*path.parents)),"Regular source/data file required")
 return path
def input_signature(values,profile):
 limits=resolve(profile);inputs=[]
 for text in values:
  require(type(text)is str and len(text)<=512 and text.count(":")==1,"Input syntax is name:d1,d2")
  name,raw=text.split(":")
  require(NAME.fullmatch(name)is not None,"Input name")
  require(re.fullmatch(r"[0-9]+(?:,[0-9]+)*",raw)is not None,"Static positive input dimensions")
  dims=[int(v)for v in raw.split(",")]
  require(1<=len(dims)<=limits.rank and all(0<v<=limits.axis_size for v in dims),"Input rank/axis budget")
  inputs.append(dict(name=name,shape=dims))
 require(1<=len(inputs)<=limits.inputs and len({v["name"]for v in inputs})==len(inputs),"Input inventory")
 require(sum(math.prod(v["shape"])for v in inputs)<=limits.input_elements,"Input element budget")
 return inputs
def npy_record(path,profile):
 limits=resolve(profile);path=regular_path(path)
 require(re.fullmatch(r"[A-Za-z0-9_-]+\.npy",path.name)is not None,"Flat NPY weight filename")
 raw=regular_bytes(path,8*limits.constant_elements+65536)
 require(raw[:6]==b"\x93NUMPY"and raw[6:8]in(b"\x01\x00",b"\x02\x00"),"NPY version")
 width=2 if raw[6:8]==b"\x01\x00"else 4
 require(len(raw)>=8+width,"Truncated NPY header")
 size=int.from_bytes(raw[8:8+width],"little");start=8+width;end=start+size
 require(0<size<=65536 and end<=len(raw),"NPY header budget")
 try:header=ast.literal_eval(raw[start:end].decode("latin1").strip())
 except(SyntaxError,ValueError,RecursionError,UnicodeError)as error:raise ValueError("NPY literal header")from error
 require(type(header)is dict and set(header)=={"descr","fortran_order","shape"},"NPY header fields")
 dims=header["shape"];dtype=header["descr"]
 require(type(dims)is tuple and len(dims)<=limits.rank and
  all(type(n)is int and 0<n<=limits.axis_size for n in dims),"NPY shape")
 require(type(dtype)is str and dtype in DTYPES and header["fortran_order"]is False,"NPY dtype/order")
 count=math.prod(dims)
 require(count<=limits.constant_elements and len(raw)-end==count*DTYPES[dtype],"NPY payload/element budget")
 code={"<f4":"f","<f8":"d","<i4":"i","<i8":"q","|b1":"?"}[dtype]
 for item in struct.iter_unpack("<"+code,memoryview(raw)[end:]):
  value=item[0]
  if dtype.startswith("<f"):require(math.isfinite(value)and abs(value)<=1024,"State floating values")
  elif dtype.startswith("<i"):require(-(2**53)<=value<=2**53,"State integer precision")
 return dict(file=path.name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),shape=list(dims),dtype=dtype)
def build(model_file,*,entry,inputs,outputs,helpers=(),weights=(),factory_args=(),factory_kwargs=None,
          dtype_policy="strict-float64-v1",approximations=None,source_format=DEFAULT_FORMAT,
          resource_profile="general-tensors-v1",call=None):
 model=regular_path(model_file);root=model.parent;limits=resolve(resource_profile)
 require(type(helpers)in(list,tuple)and len(helpers)<8,"Source inventory")
 require(type(weights)in(list,tuple)and len(weights)<=limits.constants,"State weight inventory")
 require(type(factory_args)in(list,tuple),"Factory args type")
 selected_latest=source_format=="latest"
 source_format=resolve_source_format(source_format)
 require(dtype_policy in FLOAT_POLICIES,"Source version/float policy")
 files={};text_files={};total=0;reserved=set(implementation_paths())|RESERVED
 for supplied in(model,*helpers):
  path=regular_path(supplied)
  require(path.parent==root and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.py",path.name)is not None,
   "Source/helper files must be flat Python files in the model directory")
  require(path.name not in files and path.name not in reserved,"Duplicate/reserved source filename")
  raw=regular_bytes(path,65536-total);total+=len(raw)
  require(total<=65536,"Source package byte budget")
  text=raw.decode("utf-8");ast.parse(text,filename=path.name)
  files[path.name]=hashlib.sha256(raw).hexdigest();text_files[path.name]=text
 require(1<=len(files)<=8,"Source inventory")
 signature=input_signature(inputs,resource_profile)
 output_names=list(outputs)
 require(1<=len(output_names)<=limits.outputs and len(set(output_names))==len(output_names)and
  all(type(n)is str and NAME.fullmatch(n)is not None for n in output_names),"Output names/inventory")
 state={};total_weights=0
 for text in weights:
  require(type(text)is str and "="in text,"Weight syntax is state_name=filename.npy")
  name,_,filename=text.rpartition("=")
  require(name not in state,"Duplicate state weight name")
  path=regular_path(root/filename);require(path.parent==root,"Weights must be in the model directory")
  state[name]=npy_record(path,resource_profile)
  total_weights+=math.prod(state[name]["shape"])
  require(total_weights<=limits.total_constant_elements,"State weight element budget")
 declarations(state,resource_profile=resource_profile)
 require(dtype_policy!="strict-float64-v1"or all(v["dtype"]!="<f4"for v in state.values()),
  "Float32 weights require convert-float64-v1")
 spec=dict(format=source_format,entry=entry,files=files,inputs=signature,outputs=output_names,
  factory=dict(args=list(factory_args),kwargs={}if factory_kwargs is None else factory_kwargs),
  dtype_policy=dtype_policy,weights=state,
  approximations=json.loads(json.dumps(EMPTY_APPROXIMATIONS if approximations is None else approximations)))
 from pytorch_input_binding import validate_call
 from pytorch_spatial_policy import supports_input_call
 if supports_input_call(dict(format=source_format)):
  if call is None and selected_latest:
   call=dict(format="poseidon-pytorch-call-v1",
    args=[dict(kind="tensor",name=item["name"])for item in signature],kwargs={})
  require(call is not None,"Structured input source requires --call JSON")
  spec["call"]=validate_call(call,signature)
 else:require(call is None,"Structured calls require an explicit v26+ input binding")
 legacy.validate_manifest(shim(spec))
 validate_package(spec,text_files)
 require(len(canonical(spec))<=131072,"Source manifest byte budget")
 return spec,root
def save(spec,path,root):
 output=Path(os.path.abspath(path))
 require(output.parent==root and output==output.resolve()and output.suffix==".json",
  "Manifest must be a JSON file in the model directory")
 require(not output.exists(),"Manifest already exists")
 data=json.dumps(spec,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
 with output.open("xb")as stream:stream.write(data)
 return output
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("model_file",type=Path);parser.add_argument("--entry",required=True)
 parser.add_argument("--input",action="append",required=True,help="Static input, e.g. x:1,3,32,32")
 parser.add_argument("--output",action="append",required=True,help="Tensor output name in declaration order")
 parser.add_argument("--helper",action="append",type=Path,default=[])
 parser.add_argument("--weight",action="append",default=[],help="State name and NPY, e.g. linear.weight=weight.npy")
 parser.add_argument("--factory-args",default="[]");parser.add_argument("--factory-kwargs",default="{}")
 parser.add_argument("--dtype-policy",choices=FLOAT_POLICIES,default="strict-float64-v1")
 parser.add_argument("--approximations",type=Path)
 parser.add_argument("--call",type=Path,help="Versioned structured forward args/kwargs JSON (v26+)")
 parser.add_argument("--source-format",choices=SOURCE_SELECTIONS,default=DEFAULT_FORMAT,
                     help="Explicit contract version; latest resolves to the current registered contract")
 parser.add_argument("--resource-profile",choices=tuple(PROFILES),default="general-tensors-v1")
 parser.add_argument("--manifest",type=Path,help="Create manifest; omitted prints the configuration")
 args=parser.parse_args(argv)
 try:
  factory_args=strict_json(args.factory_args,8192);factory_kwargs=strict_json(args.factory_kwargs,8192)
  require(type(factory_args)is list and type(factory_kwargs)is dict,"Factory args/kwargs types")
  approximations=None if args.approximations is None else strict_json(
   regular_bytes(regular_path(args.approximations),131072).decode("utf-8"))
  spec,root=build(args.model_file,entry=args.entry,inputs=args.input,outputs=args.output,helpers=args.helper,
   weights=args.weight,factory_args=factory_args,factory_kwargs=factory_kwargs,dtype_policy=args.dtype_policy,
   approximations=approximations,source_format=args.source_format,resource_profile=args.resource_profile,
   call=None if args.call is None else strict_json(regular_bytes(regular_path(args.call),8192).decode("utf-8"),8192))
  if args.manifest:
   output=save(spec,args.manifest,root)
   print(json.dumps(dict(status="manifest_created",manifest=str(output),
    manifest_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),resource_profile=args.resource_profile,
    source_format=spec["format"],source_executed=False,compiler_calls=0,encrypted_calls=0,provider_calls=0),
    sort_keys=True))
  else:print(json.dumps(spec,sort_keys=True,indent=2,allow_nan=False))
 except(OSError,ValueError,SyntaxError,UnicodeError)as error:parser.exit(2,"source_prepare_error: "+str(error)+"\n")
 return 0
if __name__=="__main__":raise SystemExit(main())
