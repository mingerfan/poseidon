"""Typed source options and output structure are data, never import hooks."""
import copy,hashlib,unittest
from pytorch_source_contract import (FORMAT,validate_manifest,factory_parameters,
                                     output_tree,validate_output_tree,restore_outputs)
from trusted_pytorch_import import validate_package,package_format,PACKAGE_V2

class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.files={"model.py":"def make(width, *, gain): pass\n"}
        self.spec=dict(format=FORMAT,files={k:hashlib.sha256(v.encode()).hexdigest() for k,v in self.files.items()},
            entry="model.make",inputs=[dict(name="x",shape=[2])],outputs=["a"],
            approximations={},factory=dict(args=[2],kwargs=dict(gain=.25)),
            dtype_policy="convert-float64-v1")
    def test_factory_data_and_package_version(self):
        self.assertEqual(validate_package(self.spec,self.files),self.spec)
        self.assertEqual(factory_parameters(self.spec),self.spec["factory"])
        self.assertEqual(package_format(self.spec),PACKAGE_V2)
    def test_nonfinite_function_object_and_huge_integer(self):
        for value in (float("nan"),float("inf"),lambda:None,2**53+1):
            bad=copy.deepcopy(self.spec);bad["factory"]["args"]=[value]
            with self.assertRaises(ValueError):validate_manifest(bad)
    def test_unknown_policy_and_fields(self):
        for field,value in (("dtype_policy","auto"),("unrelated",True)):
            bad=copy.deepcopy(self.spec);bad[field]=value
            with self.assertRaises(ValueError):validate_manifest(bad)
    def test_depth_size_and_keyword_names(self):
        for factory in (dict(args=list(range(65)),kwargs={}),dict(args=[],kwargs={"bad-key":0})):
            bad=copy.deepcopy(self.spec);bad["factory"]=factory
            with self.assertRaises(ValueError):validate_manifest(bad)
        nested=0
        for _ in range(10):nested=[nested]
        bad=copy.deepcopy(self.spec);bad["factory"]["args"]=[nested]
        with self.assertRaises(ValueError):validate_manifest(bad)
    def test_old_manifest_keeps_noarg_factory(self):
        old=copy.deepcopy(self.spec);old["format"]="poseidon-trusted-pytorch-v1"
        del old["factory"];del old["dtype_policy"]
        self.assertEqual(factory_parameters(old),dict(args=[],kwargs={}))
        validate_package(old,self.files)
    def test_static_output_tree_keeps_tuple_list_and_dict(self):
        a,b,c=object(),object(),object()
        raw={"prediction":a,"features":(b,[c])}
        values,tree=output_tree(raw,lambda v:v in (a,b,c),["p","f0","f1"])
        self.assertEqual(values,[a,b,c])
        reconstructed=restore_outputs(dict(p=1,f0=2,f1=3),tree)
        self.assertEqual(reconstructed,{"prediction":1,"features":(2,[3])})
        validate_output_tree(tree,["p","f0","f1"])
    def test_output_signature_count_types_and_empty(self):
        tensor=object()
        for raw,names in (([tensor],["a","b"]),([tensor,tensor],["a"]),({},["a"]),
                          ({0:tensor},["a"]),([0],["a"])):
            with self.assertRaises(ValueError):output_tree(raw,lambda v:v is tensor,names)
    def test_tree_duplicates_extras_and_order(self):
        bad=dict(kind="dict",items=[dict(key="x",value=dict(kind="tensor",name="a")),
                                   dict(key="x",value=dict(kind="tensor",name="b"))])
        with self.assertRaises(ValueError):validate_output_tree(bad,["a","b"])
        bad=dict(kind="tensor",name="a",path="/private")
        with self.assertRaises(ValueError):validate_output_tree(bad,["a"])
        with self.assertRaises(ValueError):validate_output_tree(dict(kind="tensor",name="a"),["b"])

if __name__=="__main__":unittest.main()



class ExplicitNoneOutputTests(unittest.TestCase):
    def test_none_requires_explicit_policy(self):
        value=object()
        with self.assertRaises(ValueError):output_tree((value,None),lambda x:x is value,["out"])
        leaves,tree=output_tree({"result":value,"aux":[None]},lambda x:x is value,["out"],allow_none=True)
        self.assertEqual(leaves,[value])
        self.assertEqual(restore_outputs({"out":7},tree,allow_none=True),{"result":7,"aux":[None]})
        with self.assertRaises(ValueError):restore_outputs({"out":7},tree)
    def test_none_is_not_a_cipher_leaf_and_policy_is_typed(self):
        with self.assertRaises(ValueError):output_tree(None,lambda x:False,["out"],allow_none=True)
        with self.assertRaises(ValueError):validate_output_tree({"kind":"none"},[],allow_none="true")
    def test_malformed_none_node_rejected(self):
        tree={"kind":"list","items":[{"kind":"tensor","name":"out"},{"kind":"none","name":"extra"}]}
        with self.assertRaises(ValueError):validate_output_tree(tree,["out"],allow_none=True)
