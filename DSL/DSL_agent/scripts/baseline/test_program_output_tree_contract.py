"""Versioned static output-tree compatibility; no compiler or FHE calls."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest import mock
from program_execution import package_execution_metadata
from pytorch_source_contract import restore_outputs
from pytorch_ordered_contract import allows_none_output

TREE=dict(kind="dict",items=[
    dict(key="primary",value=dict(kind="tensor",name="y")),
    dict(key="saved",value=dict(kind="list",items=[dict(kind="tensor",name="saved"),dict(kind="none")]))])
VALUES=dict(y=[[.1,.2],[.3,.4]],saved=[[1.,2.],[3.,4.]])

class PackageOutputTreeTests(unittest.TestCase):
    def metadata(self,format):
        model=dict(model="bound_fixture")
        provenance=dict(original=model, trusted_pytorch=dict(manifest=dict(format=format)),
            reference_report=dict(capture_metadata=dict(output_structure=copy.deepcopy(TREE))))
        raw=json.dumps(provenance,sort_keys=True).encode()
        with tempfile.TemporaryDirectory()as name:
            folder=Path(name);(folder/"model-provenance.json").write_bytes(raw)
            manifest=dict(format="poseidon-dsl-bundle-v6",model_provenance="included",
                          files={"model-provenance.json":hashlib.sha256(raw).hexdigest()})
            with mock.patch("candidate_bundle.check_provenance")as checked:
                result=package_execution_metadata(folder,manifest,model)
            checked.assert_called_once()
            self.assertEqual(checked.call_args.args,(provenance,model))
        return result
    def test_inherited_contracts_restore_bound_none_and_tensor_names(self):
        for version in range(16,25):
            with self.subTest(version=version):
                spec=dict(format=f"poseidon-trusted-pytorch-v{version}")
                metadata=self.metadata(spec["format"])
                self.assertTrue(allows_none_output(spec))
                self.assertEqual(metadata["output_structure_policy"],"poseidon-mha-ordered-source-v1")
                actual=restore_outputs(VALUES,metadata["output_structure"],allow_none=True)
                self.assertEqual(list(actual),["primary","saved"])
                self.assertEqual(actual,dict(primary=VALUES["y"],saved=[VALUES["saved"],None]))
                self.assertEqual(metadata["output_structure"],TREE)
    def test_v16_metadata_preserves_existing_policy_identity(self):
        self.assertEqual(self.metadata("poseidon-trusted-pytorch-v16")["output_structure_policy"],
                         "poseidon-mha-ordered-source-v1")
    def test_v15_keeps_none_rejection(self):
        metadata=self.metadata("poseidon-trusted-pytorch-v15")
        self.assertNotIn("output_structure_policy",metadata)
        self.assertFalse(allows_none_output(dict(format="poseidon-trusted-pytorch-v15")))
        with self.assertRaisesRegex(ValueError,"Explicit static None output"):
            restore_outputs(VALUES,metadata["output_structure"])
    def test_unknown_and_future_contracts_do_not_inherit_policy(self):
        for format in ("poseidon-trusted-pytorch-unknown","poseidon-trusted-pytorch-v1000",None):
            self.assertFalse(allows_none_output(dict(format=format)))
    def test_unrelated_graph_package_has_no_source_tree_policy(self):
        model=dict(model="graph")
        metadata=package_execution_metadata(Path("/unused"),dict(model_provenance="absent"),model)
        self.assertEqual(metadata,dict(original_model=model,output_structure=None))
    def test_none_is_still_explicit_and_missing_tensor_rejects(self):
        with self.assertRaisesRegex(ValueError,"Explicit static None output"):
            restore_outputs(VALUES,TREE)
        with self.assertRaises((ValueError,KeyError)):
            restore_outputs({"y":VALUES["y"]},TREE,allow_none=True)
    def test_public_structure_does_not_mutate_input_arrays(self):
        values=copy.deepcopy(VALUES);tree=copy.deepcopy(TREE)
        actual=restore_outputs(values,tree,allow_none=True)
        self.assertEqual(values,VALUES);self.assertEqual(tree,TREE)
        self.assertIsNone(actual["saved"][1])
if __name__=="__main__":unittest.main()
