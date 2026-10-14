"""NPY input-file integrity and scope tests; no model API or FHE."""
import copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from benchmark_graph import canonical,digest
from assessment_array_dataset import export_dataset,load_dataset,DatasetFileError
from test_model_output_contract import model
class ArrayDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.request={"model":model()}
    def exported(self,n=17,dtype=np.float64,labels=None):
        values=np.arange(n*2,dtype=dtype).reshape(n,2)/100
        folder=self.root/("set_"+str(len(list(self.root.iterdir()))))
        info=export_dataset(self.request,{"x":values},folder,labels=labels)
        return folder,info,values
    def opened(self,folder,manifest):
        return load_dataset(self.request,folder/"dataset.json",expected_binding=manifest["binding"])
    def rewrite(self,folder,manifest):
        manifest=copy.deepcopy(manifest)
        manifest["binding"]=digest({k:v for k,v in manifest.items() if k!="binding"})
        (folder/"dataset.json").write_bytes(canonical(manifest));return manifest
    def replace_array(self,folder,manifest,array,**kwargs):
        stream=io.BytesIO();np.save(stream,array,allow_pickle=True,**kwargs);raw=stream.getvalue()
        item=manifest["inputs"]["x"];(folder/item["file"]).write_bytes(raw)
        item.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        return self.rewrite(folder,manifest)
    def reject(self,call,code=None):
        with self.assertRaises(DatasetFileError) as error:call()
        if code:self.assertEqual(error.exception.code,code)
        self.assertNotIn("PRIVATE",str(error.exception.feedback()))
    def test_roundtrip_float32_float64_keeps_dtype_and_17_sample_partition(self):
        for dtype in (np.float32,np.float64):
            folder,manifest,values=self.exported(dtype=dtype)
            data=self.opened(folder,manifest);batches=data.batches()
            self.assertEqual([len(b["samples"]) for b in batches],[16,1])
            actual=np.stack([s["x"] for b in batches for s in b["samples"]])
            np.testing.assert_array_equal(actual,values);self.assertEqual(actual.dtype,dtype)
            self.assertEqual(batches[1]["sample_ids"],["sample_16"])
    def test_multiple_different_shape_inputs_preserve_names_and_shapes(self):
        graph=model();graph["inputs"].append(dict(name="other",shape=[1,2,1]))
        request={"model":graph};folder=self.root/"multi"
        m=export_dataset(request,{"x":np.ones((3,2)), "other":np.zeros((3,1,2,1),np.float32)},folder)
        b=load_dataset(request,folder/"dataset.json",expected_binding=m["binding"]).batches()
        self.assertEqual(set(b[0]["samples"][0]),{"x","other"})
        self.assertEqual(b[0]["samples"][0]["other"].shape,(1,2,1))
    def test_noncontiguous_export_is_explicit_c_copy(self):
        x=np.arange(12,dtype=np.float64).reshape(2,6)[:,::3];folder=self.root/"strided"
        m=export_dataset(self.request,{"x":x},folder)
        np.testing.assert_array_equal(self.opened(folder,m).batches()[0]["samples"][0]["x"],x[0])
    def test_batches_and_manifest_are_detached_from_callers(self):
        folder,m,x=self.exported();data=self.opened(folder,m);first=data.batches()
        first[0]["samples"][0]["x"][0]=999;first[0]["sample_ids"][0]="changed"
        external=data.manifest;external["samples"]=999
        np.testing.assert_array_equal(data.batches()[0]["samples"][0]["x"],x[0])
        self.assertEqual(data.manifest["samples"],17)
    def test_loaded_bytes_stay_frozen_after_disk_replacement(self):
        folder,m,x=self.exported();data=self.opened(folder,m)
        (folder/m["inputs"]["x"]["file"]).write_bytes(b"PRIVATE")
        np.testing.assert_array_equal(data.batches()[0]["samples"][0]["x"],x[0])
        self.reject(lambda:self.opened(folder,m))
    def test_labels_bind_output_axis_and_top1_scope(self):
        labels=dict(output="out",class_axis=-1,values=np.array([0,1,0],np.int64))
        folder,m,_=self.exported(3,labels=labels);data=self.opened(folder,m)
        self.assertEqual(data.criterion(dict(kind="top1_accuracy_drop",max_drop=.01)),
            dict(kind="top1_accuracy_drop",output="out",class_axis=0,labels=[0,1,0],max_drop=.01))
        self.assertEqual(data.criterion(dict(kind="max_absolute_error",limit=.1)),
            dict(kind="max_absolute_error",limit=.1))
    def test_rank_three_output_label_shape_uses_selected_axis(self):
        self.request={"model":model((1,2,1))}
        folder=self.root/"labels_rank";x=np.ones((3,1,2,1))
        m=export_dataset(self.request,{"x":x},folder,labels=dict(output="out",class_axis=1,values=np.zeros((3,1,1),np.int32)))
        self.assertEqual(self.opened(folder,m).criterion(dict(kind="top1_accuracy_drop",max_drop=0.))["labels"],[[[0]],[[0]],[[0]]])
    def test_fresh_destination_prevents_overwrite(self):
        folder,m,x=self.exported()
        self.reject(lambda:export_dataset(self.request,{"x":x},folder),"dataset_export_rejected")
        self.assertEqual(self.opened(folder,m).manifest,m)
    def test_input_integer_bool_object_complex_are_rejected(self):
        for dtype in (np.int64,np.bool_,object,np.complex128):
            self.reject(lambda d=dtype:export_dataset(self.request,{"x":np.ones((1,2),dtype=d)},self.root/str(dtype)))
    def test_nonfinite_input_is_rejected_on_export_and_load(self):
        for value in (np.nan,np.inf,-np.inf):
            self.reject(lambda v=value:export_dataset(self.request,{"x":np.array([[v,0.]])},self.root/"invalid"))
        folder,m,_=self.exported(1);m=self.replace_array(folder,m,np.array([[np.nan,0.]]))
        self.reject(lambda:self.opened(folder,m))
    def test_signature_shape_and_sample_count_are_enforced(self):
        for x in (np.ones((2,3)),np.empty((0,2)),np.ones((2,2,1))):
            self.reject(lambda a=x:export_dataset(self.request,{"x":a},self.root/"invalid"))
        folder,m,_=self.exported(1);self.request={"model":model((1,2))}
        self.reject(lambda:self.opened(folder,m))
    def test_ids_reject_duplicates_nontext_and_pathlike_names(self):
        for ids in (["same","same"],["valid",{}],["valid","../PRIVATE"],["valid",""]):
            self.reject(lambda v=ids:export_dataset(self.request,{"x":np.ones((2,2))},self.root/"ids",sample_ids=v))
    def test_labels_reject_range_dtype_axis_and_shape(self):
        for value in (dict(output="out",class_axis=0,values=np.array([2],np.int64)),
            dict(output="out",class_axis=0,values=np.array([-1],np.int64)),
            dict(output="out",class_axis=0,values=np.array([0.],np.float64)),
            dict(output="out",class_axis=1,values=np.array([0],np.int64)),
            dict(output="wrong",class_axis=0,values=np.array([0],np.int64)),
            dict(output="out",class_axis=0,values=np.zeros((1,1),np.int64))):
            self.reject(lambda v=value:export_dataset(self.request,{"x":np.ones((1,2))},self.root/"label_bad",labels=v))
    def test_trusted_manifest_pin_and_content_hash_both_required(self):
        folder,m,_=self.exported()
        self.reject(lambda:load_dataset(self.request,folder/"dataset.json",expected_binding="0"*64),
            "dataset_file_binding_changed")
        item=m["inputs"]["x"];raw=(folder/item["file"]).read_bytes()
        (folder/item["file"]).write_bytes(raw[:-1]+bytes([raw[-1]^1]))
        self.reject(lambda:self.opened(folder,m),"dataset_file_binding_changed")
    def test_duplicate_json_keys_and_nonfinite_json_are_rejected(self):
        folder,m,_=self.exported()
        for raw in (b'{"format":"PRIVATE","format":"again"}',b'{"samples":NaN}'):
            (folder/"dataset.json").write_bytes(raw);self.reject(lambda:self.opened(folder,m))
    def test_manifest_and_array_symlinks_are_rejected(self):
        folder,m,_=self.exported();target=folder/m["inputs"]["x"]["file"]
        saved=folder/"saved.npy";target.rename(saved);target.symlink_to(saved)
        self.reject(lambda:self.opened(folder,m))
        path=folder/"dataset.json";original=folder/"saved.json";path.rename(original);path.symlink_to(original)
        self.reject(lambda:self.opened(folder,m))
    def test_unsafe_file_path_cannot_escape_dataset(self):
        folder,m,_=self.exported();m["inputs"]["x"]["file"]="../PRIVATE.npy";m=self.rewrite(folder,m)
        self.reject(lambda:self.opened(folder,m))
    def test_object_npy_header_never_reaches_pickle_loader(self):
        folder,m,_=self.exported(1);m=self.replace_array(folder,m,np.array([["PRIVATE","x"]],dtype=object))
        with patch("numpy.load",side_effect=AssertionError("pickle loader")):
            self.reject(lambda:self.opened(folder,m))
    def test_fortran_layout_shape_and_truncated_payload_are_rejected(self):
        folder,m,_=self.exported(2);m=self.replace_array(folder,m,np.asfortranarray(np.ones((2,2))))
        self.reject(lambda:self.opened(folder,m))
        folder,m,_=self.exported(2);m=self.replace_array(folder,m,np.ones((1,4)))
        self.reject(lambda:self.opened(folder,m))
        folder,m,_=self.exported(2);item=m["inputs"]["x"]
        raw=(folder/item["file"]).read_bytes()[:-8];(folder/item["file"]).write_bytes(raw)
        item.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());m=self.rewrite(folder,m)
        self.reject(lambda:self.opened(folder,m))
    def test_storage_and_partition_limits_do_not_increase_execution_capacity(self):
        folder,m,_=self.exported()
        with patch("assessment_array_dataset.MAX_ARRAY_BYTES",32):
            self.reject(lambda:self.opened(folder,m))
        data=self.opened(folder,m)
        for n in (0,17,True):self.reject(lambda v=n:data.batches(v))
if __name__=="__main__":unittest.main()
