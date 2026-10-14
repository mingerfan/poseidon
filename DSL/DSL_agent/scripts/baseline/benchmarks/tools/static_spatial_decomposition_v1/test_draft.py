"""Shape, resource, semantics and provenance checks for the isolated spatial draft."""
import copy, unittest
from unittest.mock import patch
import numpy as np
import torch
from benchmark_graph import digest, samples, validate
from benchmark_math import evaluate as lower_math
from benchmark_torch import evaluate as lower_torch
from cases import models, resize, transpose
from draft import decompose
from reference import evaluate

class SpatialTests(unittest.TestCase):
    def test_model_references(self):
        for model in models():
            with self.subTest(model=model["id"]):
                before=copy.deepcopy(model)
                lowered,binding=decompose(model)
                again,identity=decompose(model)
                self.assertEqual((lowered,binding),(again,identity))
                self.assertEqual(model,before)
                self.assertEqual(binding["original_sha256"],digest(model))
                self.assertEqual(binding["lowered_sha256"],digest(lowered))
                self.assertEqual(set(binding["source_map"]),{n["id"] for n in lowered["nodes"]})
                self.assertTrue(all(v in {n["id"] for n in model["nodes"]} for v in binding["source_map"].values()))
                self.assertFalse(binding["active_agent_capability"])
                self.assertFalse(binding["runtime_layout_validated"])
                for batch in samples(model,16):
                    original=evaluate(model,batch)
                    original_torch=evaluate(model,batch,torch_reference=True)
                    core=lower_math(lowered,batch)
                    core_torch=lower_torch(lowered,batch)
                    for name in original:
                        for computed in (original_torch,core,core_torch):
                            np.testing.assert_allclose(computed[name],original[name],atol=2e-12,rtol=2e-12)

    def test_resize_rejections(self):
        base=resize("guard",[1,1,3],size=[5],mode="linear",align=False)
        changes=[dict(scale_factor=[2.]),dict(size=None),dict(size=[0]),dict(size=[2.5]),
                 dict(size=[2,3]),dict(mode="bilinear"),dict(mode="cubic"),dict(antialias=True),
                 dict(align_corners=None),dict(recompute_scale_factor=True)]
        for change in changes:
            model=copy.deepcopy(base);model["nodes"][0]["attrs"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):decompose(model)
        for factor in (0.,-1.,float("inf"),True):
            model=resize("factor",[1,1,3],factor=[factor])
            with self.subTest(factor=factor),self.assertRaises(ValueError):decompose(model)
        with self.assertRaises(ValueError):
            decompose(resize("rank",[1,1,2,2,2],size=[3,3,3]))

    def test_matrix_budget_precedes_allocation(self):
        model=resize("budget",[1,1,1024],size=[1024])
        with patch("draft.row_weights",side_effect=AssertionError("allocated before guard")):
            with self.assertRaisesRegex(ValueError,"spatial_matrix_budget"):decompose(model)

    def test_transpose_rejections(self):
        base=transpose("guard",[1,2,3],2,[3],[1],[1],[1])
        for change in (dict(groups=True),dict(groups=3),dict(output_padding=[1]),
                       dict(output_size=[4]),dict(output_size="dynamic"),
                       dict(stride=[0]),dict(dilation=[0]),dict(padding=[-1])):
            model=copy.deepcopy(base);model["nodes"][0]["attrs"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):decompose(model)
        model=copy.deepcopy(base);model["constants"]["weight"]=model["constants"]["weight"][:1]
        with self.assertRaisesRegex(ValueError,"weight"):decompose(model)
        model=copy.deepcopy(base);model["nodes"][0]["inputs"][1]="x"
        with self.assertRaisesRegex(ValueError,"public_weights"):decompose(model)

    def test_explicit_factor_and_cpu_dispatch(self):
        for mode in ("nearest","nearest-exact","linear"):
            model=resize("factor",[1,1,4],factor=[1.24],mode=mode,
                         align=False if mode=="linear" else None,recompute=False)
            x=dict(x=np.asarray([[[0.,1.,4.,9.]]]))
            lowered,_=decompose(model)
            for expected in (evaluate(model,x),evaluate(model,x,torch_reference=True)):
                np.testing.assert_allclose(lower_math(lowered,x)["output"],expected["output"],atol=2e-12,rtol=2e-12)
        for mode in ("nearest","nearest-exact"):
            model=resize("double",[1,1,3],factor=[2.2],mode=mode,recompute=False)
            x=dict(x=np.asarray([[[1.,3.,7.]]]))
            lowered,_=decompose(model)
            np.testing.assert_array_equal(lower_math(lowered,x)["output"],evaluate(model,x,torch_reference=True)["output"])

    def test_nearest_generic_and_vectorized_2d_contexts(self):
        for shape,size,factor in (([1,1,3,2],[127,2],None),
                                  ([1,1,3,2],None,[2.2,2.2]),
                                  ([1,4,1,1],[127,2],None)):
            for mode in ("nearest","nearest-exact"):
                model=resize("dispatch",shape,size=size,factor=factor,mode=mode,recompute=False)
                # Only the centre channel is asymmetric in the singleton context.
                values=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)+.25
                lowered,_=decompose(model)
                for expected in (evaluate(model,dict(x=values)),evaluate(model,dict(x=values),torch_reference=True)):
                    np.testing.assert_array_equal(lower_math(lowered,dict(x=values))["output"],expected["output"])

    def test_public_weight_change_changes_binding(self):
        model=next(m for m in models() if m["id"]=="transpose_2d_1")
        _,before=decompose(model)
        altered=copy.deepcopy(model);altered["constants"]["weight"][0][0][0][0]+=.01
        _,after=decompose(altered)
        self.assertNotEqual(before["original_sha256"],after["original_sha256"])
        self.assertNotEqual(before["lowered_sha256"],after["lowered_sha256"])
        self.assertNotEqual(before["binding_sha256"],after["binding_sha256"])

    def test_undefined_and_duplicate_nodes(self):
        model=models()[0]
        model["nodes"][0]["inputs"]=["y"]
        with self.assertRaisesRegex(ValueError,"undefined"):decompose(model)
        model=models()[-1];model["nodes"][1]["id"]=model["nodes"][0]["id"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):decompose(model)

if __name__=="__main__":unittest.main(verbosity=2)
