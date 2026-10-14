"""Unpaid structured package forwarding checks; no backend calls."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from package_checks import forward_samples
from pytorch_input_binding import flatten_samples
def leaf(name):return dict(kind="tensor",name=name)
def literal(value):return dict(kind="literal",value=value)
class PackageCallTests(unittest.TestCase):
    def meta(self,call):
        return dict(model_provenance=dict(trusted_pytorch=dict(manifest=dict(
            format="poseidon-trusted-pytorch-v27",call=call))))
    def test_dictionary_binding_follows_names_not_input_order(self):
        call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="dict",
            items=[["right",leaf("y")],["left",leaf("x")]])],kwargs={})
        sample=dict(x=[[.1,.2]],y=[[.3,.4]])
        self.assertEqual(forward_samples(self.meta(call),[sample]),
            [dict(args=[dict(right=sample["y"],left=sample["x"])],kwargs={})])
    def test_nested_containers_keep_tuple_list_and_public_flag(self):
        call=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tuple",items=[
            leaf("x"),dict(kind="list",items=[leaf("y"),literal(None)])])],
            kwargs=dict(flag=literal(True)))
        result=forward_samples(self.meta(call),[dict(x=[1.],y=[2.])])[0]
        self.assertIs(type(result["args"][0]),tuple)
        self.assertIs(type(result["args"][0][1]),list)
        self.assertIsNone(result["args"][0][1][1])
        self.assertEqual(result["kwargs"],dict(flag=True))
    def test_forwarding_does_not_mutate_frozen_manifest(self):
        call=dict(format="poseidon-pytorch-call-v1",args=[leaf("x")],
                  kwargs=dict(settings=dict(kind="dict",items=[["gain",literal(.5)]])))
        meta=self.meta(call);before=copy.deepcopy(meta)
        result=forward_samples(meta,[dict(x=[1.])])[0]
        result["kwargs"]["settings"]["gain"]=123
        self.assertEqual(meta,before)
    def test_old_source_kind_rejected(self):
        meta=self.meta(dict(args=[],kwargs={}))
        meta["model_provenance"]["trusted_pytorch"]["manifest"]["format"]="poseidon-trusted-pytorch-v25"
        with self.assertRaisesRegex(ValueError,"structured_package_source"):
            forward_samples(meta,[{}])
if __name__=="__main__":unittest.main()
