"""Actual FX, original Module and two independent graph references."""
import copy,json,importlib.util,unittest
from pathlib import Path
import numpy as np,torch
from benchmark_graph import samples,digest
from benchmark_math import evaluate as math_ref
from benchmark_torch import evaluate as torch_ref
from operator_reference import evaluate as high_ref
from pytorch_model_capture import capture
from pytorch_source_contract import output_tree
from pytorch_static_index import check_record
from hecate_python_env import ROOT
CASES=ROOT/"scripts/baseline/cases/trusted-pytorch-index-v6"
def fixture():
    sp=importlib.util.spec_from_file_location("owned_static_index_fixture",CASES/"models.py")
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module);return module
def captured(module,dims,outputs):
    return capture(module,[dict(name="x",shape=dims)],outputs,resource_profile="general-tensors-v1",
        dtype_policy="strict-float64-v1",preserve_output_structure=True,record_state_sources=True,
        capture_policy="signature-static-index-v1",approximation_policy="node-module-default-v1",
        approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}))

class IndexCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records={};models=fixture()
        for p in sorted(CASES.glob("*.json")):
            spec=json.loads(p.read_text());module=models.make(p.stem)
            original,lowered,binding=captured(module,spec["inputs"][0]["shape"],spec["outputs"])
            cls.records[p.stem]=(spec,module,original,lowered,binding)
    def test_fifteen_models_have_original_module_and_dual_reference_checks(self):
        self.assertEqual(len(self.records),15)
        for spec,module,original,lowered,binding in self.records.values():
            check_record(binding["pytorch_capture"]["index_capture"],original,resource_profile="general-tensors-v1")
            for values in samples(lowered,16):
                raw,tree=output_tree(module(torch.from_numpy(values["x"])),lambda v:isinstance(v,torch.Tensor),
                    spec["outputs"],scalar_tensor=lambda v:v.ndim==0)
                a=math_ref(lowered,values);b=torch_ref(lowered,values);high=high_ref(original,values)
                for name,t in zip(spec["outputs"],raw):
                    arr=t.detach().numpy().reshape(1) if t.ndim==0 else t.detach().numpy()
                    for other in (b[name],high[name],arr):np.testing.assert_allclose(a[name],other,atol=1e-12,rtol=1e-12)
                self.assertEqual(tree,binding["pytorch_capture"]["output_structure"])
    def test_three_contexts_per_selection_partition(self):
        for partition in ("slicing","axes","narrow","vector","index_select"):
            names=[n for n in self.records if n.startswith(partition+"_")]
            self.assertEqual(len(names),3)
            for name in names:self.assertTrue(self.records[name][-1]["pytorch_capture"]["index_capture"]["entries"])
    def test_static_shape_queries_after_selection(self):
        class Module(torch.nn.Module):
            def forward(self,x):
                y=x[...,::2]
                return y.reshape(y.size(0),y.numel()//y.size(0))
        original,lowered,binding=captured(Module(),[2,4],["out"])
        self.assertEqual(binding["pytorch_capture"]["output_structure"],dict(kind="tensor",name="out"))
        self.assertEqual(math_ref(lowered,samples(lowered,4)[0])["out"].shape,(2,2))
    def test_scalar_selection_and_nested_outputs(self):
        record=self.records["axes_0"][-1]["pytorch_capture"]
        self.assertEqual(record["output_structure"]["kind"],"dict")
        scalar=record["output_structure"]["items"][1]["value"]
        self.assertEqual(scalar,dict(kind="tensor",name="score",shape=[]))
    def test_source_shape_plan_and_node_mapping_tampering_rejected(self):
        original=self.records["vector_0"][2];record=self.records["vector_0"][-1]["pytorch_capture"]["index_capture"]
        changes=(lambda v:v["entries"][0]["source_shape"].__setitem__(0,1),
            lambda v:v["entries"][0]["plan"]["result_shape"].__setitem__(0,1),
            lambda v:v["entries"][0]["plan"]["steps"][0]["attrs"]["indices"].__setitem__(0,2),
            lambda v:v["entries"][0]["node_ids"].reverse(),
            lambda v:v["entries"][0].update(source="missing"))
        for edit in changes:
            bad=copy.deepcopy(record);edit(bad)
            with self.subTest(edit=edit),self.assertRaises(ValueError):check_record(bad,original,resource_profile="general-tensors-v1")
    def test_index_mutation_changes_graph_and_source_binding(self):
        spec,module,original,_,_=self.records["vector_1"]
        with torch.no_grad():module.idx.copy_(torch.tensor([0,1,0]))
        changed,_,_=captured(module,spec["inputs"][0]["shape"],spec["outputs"])
        self.assertNotEqual(digest(original),digest(changed))
    def test_old_static_capture_keeps_tensor_indexing_closed(self):
        class Module(torch.nn.Module):
            def forward(self,x):return x[:,1]
        with self.assertRaisesRegex(ValueError,"Unsupported PyTorch call"):
            capture(Module(),[dict(name="x",shape=[2,3])],["out"],resource_profile="general-tensors-v1",
              dtype_policy="strict-float64-v1",preserve_output_structure=True,capture_policy="signature-static-v1")
    def test_private_index_and_mask_are_rejected(self):
        class Module(torch.nn.Module):
            def forward(self,x):return x[x>0]
        with self.assertRaises(ValueError):captured(Module(),[4],["out"])
        class Dynamic(torch.nn.Module):
            def forward(self,x):return x[x.long()]
        with self.assertRaises((ValueError,RuntimeError)):captured(Dynamic(),[4],["out"])
    def test_old_position_index_select_graph_has_no_v6_metadata(self):
        module=fixture().make("index_select_2")
        old,_,binding=capture(module,[dict(name="x",shape=[2,3,2])],["out"],resource_profile="general-tensors-v1",
            dtype_policy="strict-float64-v1",preserve_output_structure=True,capture_policy="signature-static-v1")
        self.assertNotIn("index_capture",binding["pytorch_capture"])
        self.assertEqual(old["nodes"][0]["op"],"public_gather")
    def test_duplicate_keywords_and_empty_narrow_rejected(self):
        class Empty(torch.nn.Module):
            def forward(self,x):return x.narrow(0,0,0)
        with self.assertRaisesRegex(ValueError,"Static narrow interval"):captured(Empty(),[4],["out"])

if __name__=="__main__":unittest.main()
