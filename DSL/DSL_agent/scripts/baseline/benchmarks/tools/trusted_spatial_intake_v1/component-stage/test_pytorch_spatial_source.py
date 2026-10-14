"""Spatial source semantics, provenance rejection, and legacy version boundaries."""
import copy,unittest
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_spatial_source import metadata,select,check_record,POLICY
from pytorch_spatial_policy import SOURCE_FORMAT,INTERNAL_FORMATS,supports_input_call
from pytorch_ordered_capture import prepare_module
from pytorch_public_arithmetic_capture import CAPTURE_OPTIONS,TRACE_OPTIONS
from pytorch_ordered_contract import validate_package,SOURCE_FORMATS
from pytorch_static_trace import trace
PROFILE="general-tensors-v1"
OPTIONS=dict(CAPTURE_OPTIONS,approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},nodes=[],modules=[]))
def meta(args,kwargs,rank=1):
    x=torch.empty((1,1,3)if rank==1 else (1,1,2,3),dtype=torch.float64,device="meta")
    return metadata("call_function",F.interpolate,(x,*args),kwargs,child=None,
                    resolve_value=lambda v:v,resource_profile=PROFILE)
class SpatialSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Resize(nn.Module):
            def __init__(self):
                super().__init__();self.up=nn.Upsample(size=5,mode="linear",align_corners=False)
            def forward(self,x):return self.up(x)
        high,lower,bound,context=prepare_module(Resize().eval(),[dict(name="x",shape=[1,1,3])],["output"],
            resource_profile=PROFILE,spatial_source_policy=POLICY,**OPTIONS)
        cls.captured=bound["pytorch_capture"];cls.alias=cls.captured["ordered_alias_capture"]
    def check(self,record=None,emissions=None,descriptions=None,origins=None):
        return check_record(record or copy.deepcopy(self.captured["spatial_source_capture"]),
            emissions or self.alias["capture_emissions"],
            descriptions or self.alias["normalization"]["source_value_descriptions"],
            source_origins=origins or self.alias["capture_source_origins"],resource_profile=PROFILE)
    def test_registered_record_is_complete_and_reconstructible(self):
        self.check()
    def test_plain_static_size_and_scale_are_supported(self):
        self.assertEqual(tuple(meta((),dict(size=5,mode="nearest")).shape),(1,1,5))
        self.assertEqual(tuple(meta((),dict(scale_factor=1.7,mode="linear",align_corners=False)).shape),(1,1,5))
    def test_unknown_and_duplicate_arguments_are_rejected(self):
        for args,kwargs in [((5,),dict(size=6)),((),dict(size=5,unknown=True))]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):meta(args,kwargs)
    def test_modes_antialias_and_rank_are_not_silently_changed(self):
        for kwargs,rank in [(dict(size=(3,4),mode="bicubic"),2),
                            (dict(size=(3,4),mode="bilinear",align_corners=False,antialias=True),2),
                            (dict(size=5,mode="trilinear"),1)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):meta((),kwargs,rank)
    def test_encrypted_or_nonfinite_sizes_are_rejected(self):
        for size in [float("nan"),torch.empty((),device="meta"),True,0,100000]:
            with self.subTest(size=str(size)),self.assertRaises(ValueError):meta((),dict(size=size))
    def test_cpu_float64_and_contiguity_are_explicit(self):
        for x in [torch.empty((1,1,3),dtype=torch.float32,device="meta"),
                  torch.empty((1,1,2,3),dtype=torch.float64,device="meta").transpose(-1,-2)]:
            with self.subTest(dtype=x.dtype),self.assertRaises(ValueError):
                metadata("call_function",F.interpolate,(x,),dict(size=5),child=None,
                    resolve_value=lambda v:v,resource_profile=PROFILE)
    def test_names_do_not_replace_callable_identity(self):
        def interpolate(x,**kwargs):return x
        interpolate.__module__="torch.nn.functional"
        self.assertIsNone(select("call_function",interpolate,(),{}))
    def test_module_metadata_does_not_add_public_reads_to_retrace(self):
        for rank in (1,2):
            class Transpose(nn.Module):
                def __init__(self):
                    super().__init__();klass=nn.ConvTranspose1d if rank==1 else nn.ConvTranspose2d
                    self.up=klass(1,1,2,dtype=torch.float64)
                def forward(self,x):return self.up(x)
            inputs=[dict(name="x",shape=[1,1,3]if rank==1 else [1,1,2,3])]
            one=trace(Transpose().eval(),inputs,resource_profile=PROFILE,spatial_source_policy=POLICY,**TRACE_OPTIONS)
            two=trace(one,inputs,resource_profile=PROFILE,spatial_source_policy=POLICY,**TRACE_OPTIONS)
            self.assertEqual(str(one.graph),str(two.graph))
            self.assertFalse(any(n.op=="get_attr"for n in two.graph.nodes))
    def test_cipher_weights_are_rejected(self):
        class Dynamic(nn.Module):
            def forward(self,x,w):return F.conv_transpose1d(x,w)
        with self.assertRaisesRegex(ValueError,"public_weight"):
            prepare_module(Dynamic().eval(),[dict(name="x",shape=[1,1,3]),dict(name="w",shape=[1,1,2])],
                ["output"],resource_profile=PROFILE,spatial_source_policy=POLICY,**OPTIONS)
    def test_noncontiguous_source_is_rejected(self):
        class NonContiguous(nn.Module):
            def forward(self,x):return F.interpolate(x.transpose(-1,-2),size=(3,4),mode="nearest")
        with self.assertRaisesRegex(ValueError,"contiguous"):
            prepare_module(NonContiguous().eval(),[dict(name="x",shape=[1,1,2,3])],["output"],
                resource_profile=PROFILE,spatial_source_policy=POLICY,**OPTIONS)
    def test_unsupported_format_does_not_gain_spatial_calls(self):
        class Resize(nn.Module):
            def forward(self,x):return F.interpolate(x,size=5)
        with self.assertRaises(ValueError):
            prepare_module(Resize().eval(),[dict(name="x",shape=[1,1,3])],["output"],resource_profile=PROFILE,**OPTIONS)
    def test_private_transports_are_not_public_packages(self):
        self.assertIn(SOURCE_FORMAT,SOURCE_FORMATS)
        for version in INTERNAL_FORMATS:
            self.assertNotIn(version,SOURCE_FORMATS)
            with self.assertRaisesRegex(ValueError,"Public trusted source format"):validate_package(dict(format=version),{})
    def test_structured_calls_are_versioned(self):
        self.assertTrue(supports_input_call(dict(format=SOURCE_FORMAT)))
        self.assertTrue(supports_input_call(dict(format="poseidon-trusted-pytorch-v26")))
        self.assertFalse(supports_input_call(dict(format="poseidon-trusted-pytorch-v25")))
        for version in INTERNAL_FORMATS:self.assertFalse(supports_input_call(dict(format=version)))
    def test_source_record_missing_duplicate_or_extra_rows_are_rejected(self):
        for entries in [[],self.captured["spatial_source_capture"]["entries"]*2]:
            record=copy.deepcopy(self.captured["spatial_source_capture"]);record["entries"]=entries
            with self.assertRaises(ValueError):self.check(record)
    def test_source_record_fields_and_scope_are_bound(self):
        for key,value in [("version","future"),("graph_format","poseidon-operator-graph-v1"),
                          ("original_function_accuracy_certified",True),("unexpected",None)]:
            record=copy.deepcopy(self.captured["spatial_source_capture"]);record[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(record)
    def test_source_entry_attributes_shapes_and_identity_are_bound(self):
        for key,value in [("fx_node","unknown"),("operator","square"),("source","module:ConvTranspose1d"),
                         ("input_shape",[3]),("output_shape",[5]),("attributes",{}),
                         ("fixed_semantics","CUDA"),("unexpected",None)]:
            record=copy.deepcopy(self.captured["spatial_source_capture"]);record["entries"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(record)
    def test_source_kind_is_tied_to_real_origin(self):
        record=copy.deepcopy(self.captured["spatial_source_capture"])
        record["entries"][0]["source"]="function:torch.nn.functional.interpolate"
        with self.assertRaisesRegex(ValueError,"origin"):self.check(record)
    def test_typed_dependency_is_reconstructed(self):
        descriptions=copy.deepcopy(self.alias["normalization"]["source_value_descriptions"])
        descriptions["x"]["kind"]="plain"
        with self.assertRaisesRegex(ValueError,"encrypted_dependency"):self.check(descriptions=descriptions)
if __name__=="__main__":unittest.main()
