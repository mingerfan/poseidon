"""Source metadata and freeze controls; no Agent or encrypted execution."""
import copy,unittest
import torch
from source_fx_observer import FXObserver,graph_description
from source_fx_metadata import checked_source,TRANSPARENT_ATTRIBUTES
from source_calibration_contract import CalibrationError,FX_MODULE_FORMAT
import test_source_module_calibration as fixtures

def modules():
    return [torch.nn.Conv1d(1,1,1),torch.nn.Conv2d(1,1,1),
            torch.nn.BatchNorm1d(1),torch.nn.BatchNorm2d(1),
            torch.nn.AvgPool1d(2),torch.nn.AvgPool2d(2),
            torch.nn.AdaptiveAvgPool1d(1),torch.nn.AdaptiveAvgPool2d(1)]
def node_for(module):
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.before=module;self.activation=torch.nn.GELU()
        def forward(self,x):return self.activation(self.before(x))
    m=Model().eval()
    o=FXObserver(m,fixtures.selection(),tensor_elements=64,max_compute_nodes=64)
    node=next(n for n in o.description["nodes"]if n["target"]=="before")
    return m,o,node

class MetadataTests(unittest.TestCase):
    def test_eight_exact_types_bind_complete_attributes(self):
        for m in modules():
            with self.subTest(kind=type(m).__name__):
                model,o,node=node_for(m)
                kind="torch.nn."+type(m).__name__
                self.assertEqual(node["source"]["module"]["type"],kind)
                self.assertEqual(set(node["source"]["module"]["attributes"]),TRANSPARENT_ATTRIBUTES[kind])
                self.assertEqual(node["source"]["module_paths"],["before"])
                self.assertEqual(checked_source(node),node["source"])
    def test_three_native_shapes_per_family_preserve_original_execution(self):
        for m in modules():
            for length in(2,4,6):
                rank=1 if type(m).__name__.endswith("1d")else 2
                shape=(1,1)+((length,)*rank)
                model,o,node=node_for(copy.deepcopy(m).double())
                x=torch.linspace(-.2,.3,torch.tensor(shape).prod().item(),dtype=torch.float64).reshape(shape)
                with o:o.begin();got=o.execute(x)
                torch.testing.assert_close(got,model(x),rtol=0,atol=0)
    def test_convolution_attributes_drift_changes_identity(self):
        for attr,value in(("stride",(2,2)),("padding",(1,1)),("dilation",(2,2)),("padding_mode","reflect")):
            model,o,node=node_for(torch.nn.Conv2d(1,1,1))
            setattr(model.before,attr,value)
            self.assertNotEqual(o.description,graph_description(o.traced,source_locations=True))
            with self.assertRaises(CalibrationError):
                with o:pass
    def test_batchnorm_and_pool_flags_are_bound(self):
        for module,attr,value in((torch.nn.BatchNorm1d(1),"eps",.02),
                                  (torch.nn.BatchNorm2d(1),"track_running_stats",False),
                                  (torch.nn.AvgPool2d(2),"count_include_pad",False),
                                  (torch.nn.AvgPool1d(2),"ceil_mode",True),
                                  (torch.nn.AdaptiveAvgPool1d(1),"output_size",2)):
            model,o,node=node_for(module)
            setattr(model.before,attr,value)
            self.assertNotEqual(o.description,graph_description(o.traced,source_locations=True))
    def test_unknown_module_keeps_explicit_unknown_type(self):
        _,_,node=node_for(torch.nn.MaxPool1d(2))
        self.assertEqual(node["source"]["module"],{"type":None,"attributes":{}})
    def test_invalid_shapes_and_private_fields_rejected(self):
        _,_,node=node_for(torch.nn.Conv2d(2,2,1))
        for values in(dict(stride=[True,1]),dict(groups=3),dict(padding=[-1,0]),
                      dict(kernel_size=[1]),dict(extra="PRIVATE"),dict(padding_mode="PRIVATE")):
            changed=copy.deepcopy(node);changed["source"]["module"]["attributes"].update(values)
            with self.assertRaises(CalibrationError):checked_source(changed)
    def test_native_one_dimension_pool_attributes_normalize(self):
        _,_,node=node_for(torch.nn.AvgPool1d(2))
        attrs=node["source"]["module"]["attributes"]
        self.assertEqual(attrs["kernel_size"],[2]);self.assertEqual(attrs["stride"],[2])
        self.assertIsNone(attrs["divisor_override"])
    def test_legacy_graph_remains_without_module_fields(self):
        o=FXObserver(torch.nn.Sequential(torch.nn.Conv2d(1,1,1)),fixtures.selection(fixtures.FX_FORMAT),tensor_elements=64,max_compute_nodes=64)
        self.assertTrue(all(set(n)=={"name","op","target","args","kwargs"}for n in o.description["nodes"]))

class TransparentFreezeTests(unittest.TestCase):
    def test_all_transparent_types_coexist_with_explicit_activation_profile(self):
        for module in modules():
            base=fixtures.ModuleFreezeTests();base.setUp()
            try:
                _,_,before=node_for(module)
                base.graph["nodes"].insert(0,before)
                result=base.call()
                self.assertEqual([s["function"]for s in result["approximation_selection"]["selections"]],["gelu"])
            finally:base.tearDown()
    def test_unknown_compute_module_is_not_silently_skipped(self):
        base=fixtures.ModuleFreezeTests();base.setUp()
        try:
            _,_,before=node_for(torch.nn.MaxPool1d(2))
            base.graph["nodes"].insert(0,before)
            with self.assertRaises(ValueError):base.call()
        finally:base.tearDown()
if __name__=="__main__":unittest.main()
