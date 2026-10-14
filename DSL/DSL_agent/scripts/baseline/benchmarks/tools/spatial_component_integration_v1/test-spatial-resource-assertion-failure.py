"""Spatial graph semantics, resource checks and prior-format compatibility."""
import copy,unittest
from unittest.mock import patch
import numpy as np
from benchmark_graph import digest,samples
from benchmark_math import evaluate as core
from operator_arithmetic_graph import decompose,evaluate_reference,preparation_options
from operator_spatial_graph import FORMAT

def resize(shape=(1,1,3),size=(5,),mode="linear",align=False):
    return dict(format=FORMAT,id="spatial_model",inputs=[dict(name="x",shape=list(shape))],
        constants={},nodes=[dict(id="resize",op="resize",inputs=["x"],outputs=["y"],
        attrs=dict(size=list(size),scale_factor=None,mode=mode,align_corners=align,
            recompute_scale_factor=None,antialias=False))],outputs=[dict(name="output",value="y")])

def transpose():
    return dict(format=FORMAT,id="transpose_model",inputs=[dict(name="x",shape=[1,2,3])],
        constants=dict(w=[[[.1,-.2,.3]],[[.4,.5,-.6]]],b=[.03]),
        nodes=[dict(id="conv_t",op="conv_transpose1d",inputs=["x","w","b"],outputs=["y"],
            attrs=dict(stride=[2],padding=[1],dilation=[1],groups=1,
                output_padding=[1],output_size=None))],outputs=[dict(name="output",value="y")])

class SpatialGraphTests(unittest.TestCase):
    def test_resize_exact_reference_and_source_map(self):
        m=resize();snapshot=copy.deepcopy(m);low,binding=decompose(m,resource_profile="general-tensors-v1")
        self.assertEqual(m,snapshot)
        self.assertEqual((low,binding),decompose(m,resource_profile="general-tensors-v1"))
        self.assertEqual(binding["original_sha256"],digest(m))
        self.assertEqual(binding["lowered_sha256"],digest(low))
        self.assertEqual(set(binding["source_map"]),{n["id"]for n in low["nodes"]})
        self.assertTrue(all(n=="resize"for n in binding["source_map"].values()))
        self.assertTrue(binding["no_dsl_answer"])
        self.assertFalse(binding["runtime_layout_validated"])
        for batch in samples(m,16):
            np.testing.assert_allclose(core(low,batch)["output"],evaluate_reference(m,batch)["output"],rtol=2e-12,atol=2e-12)
    def test_weight_content_binding_and_transpose_reference(self):
        m=transpose();low,before=decompose(m)
        for batch in samples(m,16):
            np.testing.assert_allclose(core(low,batch)["output"],evaluate_reference(m,batch)["output"],rtol=2e-12,atol=2e-12)
        changed=copy.deepcopy(m);changed["constants"]["w"][0][0][0]+=.01
        low2,after=decompose(changed)
        self.assertNotEqual(before["binding_sha256"],after["binding_sha256"])
        self.assertNotEqual(before["lowered_sha256"],after["lowered_sha256"])
        e=after["normalization"]["spatial_entries"][0]
        self.assertNotEqual(e["original_weight_sha256"],e["transformed_weight_sha256"])
    def test_bounded_linear_map_has_exact_shape_and_bias(self):
        model=transpose();low,binding=decompose(model)
        evidence=binding["normalization"]["spatial_entries"][0]
        self.assertEqual(evidence["axis_binding"],"exact_public_linear_map")
        self.assertEqual(evidence["output_shape"],[1,1,6])
        self.assertEqual(evidence["matrix_elements"],36)
        self.assertEqual([n["op"]for n in low["nodes"]],["reshape","linear","reshape"])
        inputs={"x":np.array([[[-.8,.35,.2],[.4,-.6,.1]]],dtype=np.float64)}
        np.testing.assert_allclose(core(low,inputs)["output"],evaluate_reference(model,inputs)["output"],rtol=2e-12,atol=2e-12)

    def test_large_linear_map_keeps_bounded_sparse_axis_route(self):
        model=transpose()
        model["inputs"][0]["shape"]=[1,1,257]
        model["constants"]={"w":[[[.3]]]}
        node=model["nodes"][0]
        node["inputs"]=["x","w"]
        node["attrs"]=dict(stride=[1],padding=[0],dilation=[1],groups=1,output_padding=[0],output_size=None)
        low,binding=decompose(model,resource_profile="medium-v1")
        evidence=binding["normalization"]["spatial_entries"][0]
        self.assertEqual(evidence["axis_binding"],"static_gather_and_public_zero_mask")
        self.assertEqual(evidence["matrix_elements"],0)
        self.assertEqual(evidence["public_index_elements"],257)
        inputs={"x":np.linspace(-.9,.7,257,dtype=np.float64).reshape(1,1,257)}
        np.testing.assert_allclose(core(low,inputs)["output"],.3*inputs["x"],rtol=2e-12,atol=2e-12)
        self.assertEqual(binding["resource_profile"],"medium-v1")

    def test_json_guard_precedes_copying_objects(self):
        class NotData:
            def __deepcopy__(self,memo):raise AssertionError("Non-data copied")
        model=resize();model["constants"]["bad"]=NotData()
        with self.assertRaises(TypeError):decompose(model)
        model=resize();model["extra"]=NotData()
        with self.assertRaises(ValueError):decompose(model)

    def test_axis_matrix_capacity_before_allocation(self):
        m=resize(shape=(1,1,1024),size=(1024,))
        with patch("operator_spatial_graph.row_weights",side_effect=AssertionError("allocated before guard")):
            with self.assertRaisesRegex(ValueError,"spatial_matrix_budget"):decompose(m,resource_profile="general-tensors-v1")
    def test_malformed_and_unsupported_resize(self):
        m=resize()
        for attrs in [dict(mode="bicubic"),dict(antialias=True),dict(size=[0]),
                dict(scale_factor=[2.]),dict(align_corners=None),dict(recompute_scale_factor=True)]:
            x=copy.deepcopy(m);x["nodes"][0]["attrs"].update(attrs)
            with self.subTest(attrs=attrs),self.assertRaises(ValueError):decompose(x)
        x=resize(shape=(1,1,2,2,2),size=(3,3,3))
        with self.assertRaisesRegex(ValueError,"resize_rank"):decompose(x,resource_profile="general-tensors-v1")
    def test_transpose_rejects_dynamic_weight_and_output_padding(self):
        m=transpose();m["nodes"][0]["inputs"][1]="x"
        with self.assertRaisesRegex(ValueError,"public_weights"):decompose(m)
        m=transpose();m["nodes"][0]["attrs"]["output_padding"]=[2]
        with self.assertRaisesRegex(ValueError,"output_padding"):decompose(m)
        m=transpose();m["nodes"][0]["attrs"]["groups"]=True
        with self.assertRaises(ValueError):decompose(m)
    def test_old_versions_are_not_widened(self):
        for version in (1,2,3):
            m=resize();m["format"]=f"poseidon-operator-graph-v{version}"
            with self.subTest(version=version),self.assertRaisesRegex(ValueError,"Unregistered"):decompose(m)
    def test_composes_with_general_matrix_and_public_left_arithmetic(self):
        m=resize();m["constants"]=dict(left=.125,w=[[.5,-.25,.125,.3,-.1]])
        m["nodes"] += [dict(id="shift",op="subtract",inputs=["left","y"],attrs={},outputs=["z"]),
            dict(id="project",op="matmul",inputs=["z","w"],attrs={},outputs=["r"])]
        # Matrix weight is [5,1]; batch matmul output retains the declared axes.
        m["constants"]["w"]=[[.5],[-.25],[.125],[.3],[-.1]]
        m["outputs"]=[dict(name="output",value="r")]
        low,b=decompose(m,resource_profile="general-tensors-v1")
        self.assertTrue(b["normalization"]["matrix_entries"])
        self.assertTrue(b["normalization"]["arithmetic_entries"])
        for batch in samples(m,16):
            np.testing.assert_allclose(core(low,batch)["output"],evaluate_reference(m,batch)["output"],rtol=2e-12,atol=2e-12)
    def test_graph_plan_routes_and_capabilities(self):
        from application_capabilities import describe,assess_model
        caps=describe()
        self.assertIn(FORMAT,caps["graph_formats"])
        self.assertEqual(set(caps["spatial_decomposition_registry"]["operators"]),{"resize","conv_transpose1d","conv_transpose2d"})
        m=resize();low,b=decompose(m,resource_profile="general-tensors-v1")
        options=preparation_options(m,low)
        self.assertIn("capacity_packing_version",options)
        assessment=assess_model(m,resource_profile="general-tensors-v1")
        self.assertEqual(assessment["stages"]["decomposition"]["status"],"accepted")
        self.assertEqual(assessment["decomposition"]["binding_sha256"],b["binding_sha256"])

if __name__=="__main__":unittest.main()
