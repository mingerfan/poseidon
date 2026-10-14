"""Bounded non-pickle input arrays for local, explicit dataset assessment."""
import hashlib,io,json,math,re
from pathlib import Path
import numpy as np
from benchmark_graph import digest,validate,canonical
from model_output_contract import snapshot,is_hash,OutputError
from model_weight_package import regular_bytes
FORMAT="poseidon-assessment-array-dataset-v1"
MAX_SAMPLES=65536
MAX_MANIFEST_BYTES=4*1024**2
MAX_ARRAY_BYTES=256*1024**2
FLOAT_DTYPES={"<f4":4,"<f8":8}
LABEL_DTYPES={"<i4":4,"<i8":8}
NAME=re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")
FILE=re.compile(r"[A-Za-z0-9_-]+[.]npy")
CODES={
 "dataset_file_rejected":"Dataset file, signature, shape, dtype or labels are invalid.",
 "dataset_file_binding_changed":"Dataset content does not match its trusted binding.",
 "dataset_export_rejected":"Dataset export requires validated arrays and a fresh directory.",
}
class DatasetFileError(ValueError):
    def __init__(self,code):
        if code not in CODES:raise ValueError("Unknown dataset file diagnostic")
        self.code=code;super().__init__(CODES[code])
    def feedback(self):return dict(code=self.code,layer="input",message=CODES[self.code],retry_policy="never_automatic")
def require(ok,code="dataset_file_rejected"):
    if not ok:raise DatasetFileError(code)
def guard(function):
    def call(*args,**kwargs):
        try:return function(*args,**kwargs)
        except DatasetFileError:raise
        except (ValueError,TypeError,KeyError,OSError,OverflowError):
            raise DatasetFileError("dataset_file_rejected") from None
    return call
def input_signature(request):
    validate(request["model"])
    return digest(request["model"]["inputs"])
def labels_spec(request,labels):
    if labels is None:return None
    require(type(labels)is dict and set(labels)=={"output","class_axis","values"})
    name=labels["output"];axis=labels["class_axis"]
    checked=validate(request["model"]);shapes={o["name"]:list(checked["shapes"][o["value"]]) for o in request["model"]["outputs"]}
    require(type(name)is str and name in shapes and type(axis)is int and -len(shapes[name])<=axis<len(shapes[name]))
    axis%=len(shapes[name]);shape=shapes[name]
    return dict(output=name,class_axis=axis,label_shape=shape[:axis]+shape[axis+1:],classes=shape[axis])
def array_record(value,filename,dtypes):
    require(type(value)is np.ndarray and value.dtype.str in dtypes and not value.dtype.hasobject)
    value=np.array(value,copy=True,order="C")
    if value.dtype.kind=="f":require(np.isfinite(value).all())
    stream=io.BytesIO();np.save(stream,value,allow_pickle=False);raw=stream.getvalue()
    require(len(raw)<=MAX_ARRAY_BYTES)
    return dict(file=filename,shape=list(value.shape),dtype=value.dtype.str,bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest()),raw
def decode(raw,item,dtypes):
    require(type(item)is dict and set(item)=={"file","shape","dtype","bytes","sha256"} and
        type(item["file"])is str and FILE.fullmatch(item["file"]) and type(item["shape"])is list and
        all(type(n)is int and n>0 for n in item["shape"]) and
        type(item["dtype"])is str and item["dtype"]in dtypes and
        type(item["bytes"])is int and 0<item["bytes"]<=MAX_ARRAY_BYTES and is_hash(item["sha256"]))
    require(len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["sha256"],
            "dataset_file_binding_changed")
    stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream);require(version in ((1,0),(2,0)))
    read=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    dims,fortran,dtype=read(stream,max_header_size=65536)
    require(list(dims)==item["shape"] and not fortran and dtype.str==item["dtype"] and
            len(raw)-stream.tell()==math.prod(dims)*dtype.itemsize)
    value=np.frombuffer(raw,dtype=dtype,offset=stream.tell()).reshape(dims)
    if dtype.kind=="f":require(np.isfinite(value).all())
    return value
@guard
def export_dataset(request,inputs,directory,*,sample_ids=None,labels=None):
    """Export explicit host arrays; input values and original float dtypes stay unchanged."""
    require(type(inputs)is dict and set(inputs)=={i["name"] for i in request["model"]["inputs"]})
    specs=request["model"]["inputs"];input_signature(request);count=None;records={};payloads={}
    for index,spec in enumerate(specs):
        array=inputs[spec["name"]]
        require(type(array)is np.ndarray and array.ndim==len(spec["shape"])+1 and list(array.shape[1:])==spec["shape"])
        count=array.shape[0] if count is None else count
        require(type(count)is int and 1<=count<=MAX_SAMPLES and array.shape[0]==count)
        item,raw=array_record(array,"input_"+str(index)+".npy",FLOAT_DTYPES)
        records[spec["name"]]=item;payloads[item["file"]]=raw
    if sample_ids is None:sample_ids=["sample_"+str(i) for i in range(count)]
    sample_ids=snapshot(sample_ids,limit=MAX_MANIFEST_BYTES)
    require(type(sample_ids)is list and len(sample_ids)==count and len(set(sample_ids))==count and
            all(type(x)is str and NAME.fullmatch(x) for x in sample_ids))
    label_record=None
    if labels is not None:
        info=labels_spec(request,labels);array=labels["values"]
        require(type(array)is np.ndarray and list(array.shape)==[count]+info["label_shape"] and
                array.dtype.str in LABEL_DTYPES and (array>=0).all() and (array<info["classes"]).all())
        item,raw=array_record(array,"labels.npy",LABEL_DTYPES)
        label_record=dict(output=info["output"],class_axis=info["class_axis"],array=item)
        payloads[item["file"]]=raw
    require(sum(len(x) for x in payloads.values())<=MAX_ARRAY_BYTES)
    manifest=dict(format=FORMAT,input_signature_sha256=input_signature(request),samples=count,
        sample_ids=sample_ids,inputs=records,labels=label_record)
    manifest["binding"]=digest(manifest);require(len(canonical(manifest))<=MAX_MANIFEST_BYTES)
    path=Path(directory)
    require(path.is_absolute() and path==path.resolve() and not path.exists() and not path.is_symlink(),
            "dataset_export_rejected")
    path.mkdir(parents=True,mode=0o700)
    for name,raw in payloads.items():
        with (path/name).open("xb") as f:f.write(raw)
    with (path/"dataset.json").open("xb") as f:f.write(canonical(manifest))
    return manifest
class ArrayDataset:
    """Immutable opened-inode bytes, partitioned into detached 1..16 sample batches."""
    def __init__(self,manifest,arrays,labels):
        self._manifest=snapshot(manifest,limit=MAX_MANIFEST_BYTES)
        self._arrays=arrays;self._labels=labels
    @property
    def manifest(self):return snapshot(self._manifest,limit=MAX_MANIFEST_BYTES)
    def batches(self,part_samples=16):
        require(type(part_samples)is int and 1<=part_samples<=16)
        result=[]
        for start in range(0,self._manifest["samples"],part_samples):
            end=min(start+part_samples,self._manifest["samples"])
            result.append(dict(id="part_"+str(start//part_samples),
                sample_ids=list(self._manifest["sample_ids"][start:end]),
                samples=[{n:a[i].copy() for n,a in self._arrays.items()} for i in range(start,end)]))
        return result
    def criterion(self,requirements=None):
        if requirements is None:return None
        value=snapshot(requirements,limit=131072)
        if self._labels is None:return value
        require(type(value)is dict and set(value)=={"kind","max_drop"} and value["kind"]=="top1_accuracy_drop")
        info=self._manifest["labels"]
        return dict(kind="top1_accuracy_drop",output=info["output"],class_axis=info["class_axis"],
                    labels=self._labels.tolist(),max_drop=value["max_drop"])
@guard
def load_dataset(request,manifest_path,*,expected_binding):
    path=Path(manifest_path)
    require(path.is_absolute() and path==path.resolve() and path.name=="dataset.json" and not path.is_symlink())
    raw=regular_bytes(path,MAX_MANIFEST_BYTES)
    def pairs(items):
        d={}
        for k,v in items:require(k not in d);d[k]=v
        return d
    def nonfinite(value):raise DatasetFileError("dataset_file_rejected")
    manifest=json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite)
    require(type(manifest)is dict and set(manifest)=={"format","input_signature_sha256","samples","sample_ids","inputs","labels","binding"} and
        manifest["format"]==FORMAT and is_hash(expected_binding) and manifest["binding"]==expected_binding and
        digest({k:v for k,v in manifest.items() if k!="binding"})==expected_binding,"dataset_file_binding_changed")
    require(manifest["input_signature_sha256"]==input_signature(request) and
        type(manifest["samples"])is int and 1<=manifest["samples"]<=MAX_SAMPLES and
        type(manifest["sample_ids"])is list and len(manifest["sample_ids"])==manifest["samples"] and
        all(type(x)is str and NAME.fullmatch(x) for x in manifest["sample_ids"]) and
        len(set(manifest["sample_ids"]))==manifest["samples"])
    require(type(manifest["inputs"])is dict and set(manifest["inputs"])=={i["name"] for i in request["model"]["inputs"]})
    arrays={};total=0;files=set()
    for spec in request["model"]["inputs"]:
        item=manifest["inputs"][spec["name"]];require(type(item)is dict and type(item.get("file"))is str and FILE.fullmatch(item["file"]))
        require(item["file"]not in files);files.add(item["file"])
        total+=item["bytes"];require(total<=MAX_ARRAY_BYTES)
        value=decode(regular_bytes(path.parent/item["file"],item["bytes"]),item,FLOAT_DTYPES)
        require(list(value.shape)==[manifest["samples"]]+spec["shape"])
        arrays[spec["name"]]=value
    labels=None;declared=manifest["labels"]
    if declared is not None:
        require(type(declared)is dict and set(declared)=={"output","class_axis","array"})
        info=labels_spec(request,dict(output=declared["output"],class_axis=declared["class_axis"],values=None))
        item=declared["array"];require(type(item)is dict and type(item.get("file"))is str and FILE.fullmatch(item["file"]) and item["file"]not in files)
        total+=item["bytes"];require(total<=MAX_ARRAY_BYTES)
        labels=decode(regular_bytes(path.parent/item["file"],item["bytes"]),item,LABEL_DTYPES)
        require(list(labels.shape)==[manifest["samples"]]+info["label_shape"] and
                (labels>=0).all() and (labels<info["classes"]).all())
    return ArrayDataset(manifest,arrays,labels)
