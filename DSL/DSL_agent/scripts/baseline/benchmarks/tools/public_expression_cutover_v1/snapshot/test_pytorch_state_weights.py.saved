"""State weights: data-only NPY parsing, exact module loads and inert provenance."""
import copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from benchmark_graph import digest
from pytorch_weight_package import declarations,decode,read,load_state,check_binding,binding,export_state
from pytorch_source_contract import WEIGHT_FORMAT,validate_manifest
from pytorch_model_capture import capture
from trusted_pytorch_import import check_capture_metadata,source_identity

def record(value,name="tensor.npy",**options):
    stream=io.BytesIO();np.save(stream,value,allow_pickle=options.get("pickle",False))
    raw=stream.getvalue()
    return raw,dict(file=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                    shape=list(value.shape),dtype=value.dtype.str)

class StateWeightTests(unittest.TestCase):
    def test_supported_exact_types_including_scalar_buffers(self):
        for dtype in ("<f4","<f8","<i4","<i8","|b1"):
            value=np.asarray(1,dtype=dtype);raw,item=record(value)
            declarations({"norm.counter":item})
            np.testing.assert_array_equal(decode(raw,item),value)

    def test_object_fortran_trailing_and_mismatched_headers_rejected(self):
        cases=[]
        raw,item=record(np.asarray([1,2],dtype=object),pickle=True);item["dtype"]="<f8";cases.append((raw,item))
        raw,item=record(np.asfortranarray(np.arange(6,dtype=np.float64).reshape(2,3)));cases.append((raw,item))
        raw,item=record(np.ones((2,),dtype=np.float64));bad=dict(item,shape=[1,2]);cases.append((raw,bad))
        raw2=raw+b"\0";bad=dict(item,bytes=len(raw2),sha256=hashlib.sha256(raw2).hexdigest());cases.append((raw2,bad))
        for raw,item in cases:
            with self.assertRaises((ValueError,TypeError)):decode(raw,item)

    def test_invalid_declarations_and_value_domains(self):
        raw,item=record(np.asarray([1.],dtype=np.float64))
        for field,value in (("file","../tensor.npy"),("file","state.pt"),("shape",[True]),("shape",[0]),
                            ("bytes",True),("dtype",">f8"),("sha256","bad")):
            bad=copy.deepcopy(item);bad[field]=value
            with self.assertRaises(ValueError):declarations({"weight":bad})
        for values in (np.asarray([np.nan]),np.asarray([1025.]),
                       np.asarray([-(2**63)],dtype=np.int64),np.asarray([2**53+1],dtype=np.int64)):
            raw,item=record(values)
            with self.assertRaises(ValueError):decode(raw,item)

    def test_regular_file_and_symlink_integrity(self):
        raw,item=record(np.asarray([.25],dtype=np.float64))
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/item["file"]).write_bytes(raw)
            arrays,files=read(root,{"w":item})
            self.assertEqual(files[item["file"]],raw)
            bad=dict(item,sha256="0"*64)
            with self.assertRaises(ValueError):read(root,{"w":bad})
            (root/"alias.npy").symlink_to(root/item["file"])
            with self.assertRaises(OSError):read(root,{"w":dict(item,file="alias.npy")})

    def test_exact_float32_module_load_before_normalization(self):
        module=torch.nn.Linear(2,2)
        arrays={"weight":np.asarray([[.125,-.25],[.375,.0625]],dtype=np.float32),
                "bias":np.asarray([.03125,-.125],dtype=np.float32)}
        records={name:record(value,name+".npy")[1] for name,value in arrays.items()}
        loaded=load_state(module,arrays,records)
        self.assertEqual(loaded,binding(arrays,records))
        np.testing.assert_array_equal(module.weight.detach().numpy(),arrays["weight"])
        original,_,meta=capture(module,[dict(name="x",shape=[2])],["out"],
            dtype_policy="convert-float64-v1",preserve_output_structure=True,record_state_sources=True)
        meta=meta["pytorch_capture"];meta["weight_binding"]=loaded
        spec=dict(format=WEIGHT_FORMAT,files={"model.py":"0"*64},entry="model.make",
            inputs=[dict(name="x",shape=[2])],outputs=["out"],approximations={},
            factory=dict(args=[],kwargs={}),dtype_policy="convert-float64-v1",weights=records)
        validate_manifest(spec);check_capture_metadata(meta,original,spec)
        self.assertEqual(set(meta["state_constant_sources"].values()),{"weight","bias"})
        bad=copy.deepcopy(original);key=next(iter(meta["state_constant_sources"]));bad["constants"][key][0][0]+=.125
        meta=copy.deepcopy(meta);meta["graph_sha256"]=digest(bad)
        with self.assertRaises(ValueError):check_capture_metadata(meta,bad,spec)

    def test_export_round_trip_and_no_overwrite(self):
        module=torch.nn.BatchNorm1d(2)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"weights"
            records=export_state(module,path)
            self.assertEqual(records["num_batches_tracked"]["shape"],[])
            arrays,_=read(path,records)
            other=torch.nn.BatchNorm1d(2)
            self.assertEqual(load_state(other,arrays,records),binding(arrays,records))
            for name,tensor in module.state_dict().items():
                np.testing.assert_array_equal(tensor.detach().numpy(),other.state_dict()[name].numpy())
            before={p.name:p.read_bytes() for p in path.iterdir()}
            with self.assertRaises(ValueError):export_state(module,path)
            self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})
            second=export_state(module,Path(root)/"second")
            self.assertEqual(records,second)

    def test_export_rejects_invalid_state_before_writing(self):
        module=torch.nn.Linear(2,2)
        with torch.no_grad():module.weight[0,0]=float("nan")
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"weights"
            with self.assertRaises(ValueError):export_state(module,path)
            self.assertFalse(path.exists())

    def test_loaded_dtype_metadata_cannot_be_forged(self):
        module=torch.nn.Linear(2,2)
        arrays={k:v.detach().numpy().copy() for k,v in module.state_dict().items()}
        records={k:record(v,k+".npy")[1] for k,v in arrays.items()}
        loaded=load_state(module,arrays,records)
        original,_,bound=capture(module,[dict(name="x",shape=[2])],["out"],
            dtype_policy="convert-float64-v1",preserve_output_structure=True,record_state_sources=True)
        meta=bound["pytorch_capture"];meta["weight_binding"]=loaded
        spec=dict(format=WEIGHT_FORMAT,files={"model.py":"0"*64},entry="model.make",
            inputs=[dict(name="x",shape=[2])],outputs=["out"],approximations={},
            factory=dict(args=[],kwargs={}),dtype_policy="convert-float64-v1",weights=records)
        meta["source_state_dtypes"]["weight"]="float64"
        with self.assertRaises(ValueError):check_capture_metadata(meta,original,spec)

    def test_batchnorm_without_affine_uses_float64_public_ones(self):
        module=torch.nn.Sequential(torch.nn.BatchNorm1d(2,affine=False).double()).eval()
        graph,_,bound=capture(module,[dict(name="x",shape=[1,2,2])],["out"],
            dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True)
        meta=bound["pytorch_capture"]
        self.assertNotIn("float32",meta["constant_dtypes"].values())
        self.assertEqual(set(meta["state_constant_sources"].values()),{"0.running_mean","0.running_var"})

    def test_names_dtype_and_shape_are_not_silently_coerced(self):
        module=torch.nn.Linear(2,2)
        arrays={k:v.detach().numpy().copy() for k,v in module.state_dict().items()}
        records={k:record(v,k+".npy")[1] for k,v in arrays.items()}
        with self.assertRaises(ValueError):load_state(module,{"weight":arrays["weight"]},{"weight":records["weight"]})
        bad={**arrays,"weight":arrays["weight"].astype(np.float64)}
        changed={**records,"weight":record(bad["weight"],"weight.npy")[1]}
        with self.assertRaises(ValueError):load_state(module,bad,changed)
        bad={**arrays,"weight":arrays["weight"].reshape(4)}
        changed={**records,"weight":record(bad["weight"],"weight.npy")[1]}
        with self.assertRaises(ValueError):load_state(module,bad,changed)

    def test_tied_state_conflicting_values_rejected(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.a=torch.nn.Linear(1,1,bias=False);self.b=self.a
        arrays={"a.weight":np.asarray([[.125]],dtype=np.float32),
                "b.weight":np.asarray([[.25]],dtype=np.float32)}
        records={k:record(v,k.replace(".","_")+".npy")[1] for k,v in arrays.items()}
        with self.assertRaises(ValueError):load_state(Model(),arrays,records)

    def test_forged_or_incomplete_binding_rejected(self):
        raw,item=record(np.ones(2,dtype=np.float64));records={"a":item};good=binding({"a":decode(raw,item)},records)
        self.assertEqual(check_binding(good,records),good)
        for mutate in (lambda v:v.update(declarations_sha256="0"*64),
                       lambda v:v["normalized_float64_sha256"].clear(),
                       lambda v:v["normalized_float64_sha256"].update(a="../private")):
            bad=copy.deepcopy(good);mutate(bad)
            with self.assertRaises(ValueError):check_binding(bad,records)

if __name__=="__main__":unittest.main()
