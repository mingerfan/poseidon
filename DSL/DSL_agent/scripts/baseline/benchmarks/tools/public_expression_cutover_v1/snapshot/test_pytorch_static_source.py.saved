"""Version four changes only explicitly selected trusted local source policy."""
import copy,hashlib,unittest
from pytorch_source_contract import STATIC_FORMAT,WEIGHT_FORMAT,validate_manifest,validate_output_tree,restore_outputs
from trusted_pytorch_import import package_format,PACKAGE_V4

class StaticSourceContractTests(unittest.TestCase):
    def setUp(self):
        text="def make(): pass\n"
        self.spec=dict(format=STATIC_FORMAT,files={"model.py":hashlib.sha256(text.encode()).hexdigest()},
            entry="model.make",inputs=[dict(name="x",shape=[2])],outputs=["out"],approximations={},
            factory=dict(args=[],kwargs={}),dtype_policy="strict-float64-v1",weights={})
    def test_explicit_v4_and_empty_complete_state(self):
        self.assertEqual(validate_manifest(self.spec),self.spec)
        self.assertEqual(package_format(self.spec),PACKAGE_V4)
    def test_weights_required_and_legacy_fields_preserved(self):
        bad=copy.deepcopy(self.spec);del bad["weights"]
        with self.assertRaises(ValueError):validate_manifest(bad)
        for version in ("poseidon-trusted-pytorch-v2",WEIGHT_FORMAT):
            old=copy.deepcopy(self.spec);old["format"]=version
            if version.endswith("v2"):del old["weights"]
            self.assertEqual(validate_manifest(old),old)
    def test_no_user_trace_callbacks_or_automatic_modes(self):
        for field in ("capture_policy","tracer","execute_in_host"):
            bad=copy.deepcopy(self.spec);bad[field]="automatic"
            with self.assertRaises(ValueError):validate_manifest(bad)
    def test_scalar_tree_is_version_opt_in(self):
        tree=dict(kind="tensor",name="out",shape=[])
        with self.assertRaises(ValueError):validate_output_tree(tree,["out"])
        self.assertEqual(validate_output_tree(tree,["out"],allow_scalars=True),tree)
        self.assertEqual(restore_outputs(dict(out=[2.5]),tree),2.5)
    def test_scalar_tree_rejects_nonzero_shape_and_wrong_abi(self):
        for shape in ([1],[2],None):
            with self.assertRaises(ValueError):
                validate_output_tree(dict(kind="tensor",name="out",shape=shape),["out"],allow_scalars=True)
        for value in (2.,[[2.]],[],[2.,3.]):
            with self.assertRaises(ValueError):restore_outputs(dict(out=value),dict(kind="tensor",name="out",shape=[]))
    def test_registered_capability_is_shared(self):
        from application_capabilities import input_sources
        value=input_sources()["trusted_pytorch"]
        self.assertIn(STATIC_FORMAT,value["formats"])
        self.assertEqual(value["static_shape_policy"]["version"],STATIC_FORMAT)
        self.assertNotIn("encrypted_execution",value)

if __name__=="__main__":unittest.main()
