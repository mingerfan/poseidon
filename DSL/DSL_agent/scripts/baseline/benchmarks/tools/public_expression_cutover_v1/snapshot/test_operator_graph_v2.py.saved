"""Versioned JSON matrix normalization, references and component admission."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from benchmark_graph import canonical,samples,digest
from benchmark_math import evaluate as core_reference
from benchmark_torch import evaluate as tensor_reference
from model_decomposition import decompose as old_decompose,FORMAT as V1
from operator_graph import FORMAT,decompose,evaluate_reference,preparation_options
from application_capabilities import assess_model,input_sources
from candidate_bundle import check_provenance
from component_contract import prepare_task,reconstruct_request
from trusted_pytorch_import import source_identity

CASES=Path(__file__).parent/"cases/matrix-entry-equivalence-v1"
OPTIONS=dict(chunk_period=8,capacity_packing_version="capacity-public-tensors-v9",
             generation_guidance="explicit-v10",public_weight_policy="descriptors-v1")

def original(name="matrix_matrix_0"):
    model=json.loads((CASES/name/"operator-model.json").read_text())
    model["format"]=FORMAT
    return model

def compare(model,count=16):
    lowered,binding=decompose(model,resource_profile="general-tensors-v1")
    for inputs in samples(model,count):
        expected=evaluate_reference(model,inputs)
        for actual in (core_reference(lowered,inputs,resource_profile="general-tensors-v1"),
                       tensor_reference(lowered,inputs,resource_profile="general-tensors-v1")):
            for name,truth in expected.items():
                if actual[name].shape!=truth.shape:
                    raise AssertionError((actual[name].shape,truth.shape))
                np.testing.assert_allclose(actual[name],truth,atol=1e-12,rtol=1e-12)
    return lowered,binding

class OperatorGraphV2(unittest.TestCase):
    def test_all_24_contexts_and_unchanged_original_models(self):
        catalog=json.loads((CASES/"catalog.json").read_text())
        for row in catalog["cases"]:
            with self.subTest(case=row["id"]):
                model=original(row["id"])
                before=canonical(model)
                lowered,binding=compare(model)
                self.assertEqual(canonical(model),before)
                self.assertEqual(binding["original_sha256"],digest(model))
                self.assertEqual(binding["lowered_sha256"],digest(lowered))
                self.assertEqual(set(binding["source_map"].values()),{"product"})
                self.assertEqual(len(binding["normalization"]["entries"]),1)

    def test_v1_delegation_preserves_all_accepted_binding_bytes(self):
        catalog=json.loads((CASES/"catalog.json").read_text())
        accepted=0
        for row in catalog["cases"]:
            model=json.loads((CASES/row["operator_model"]).read_text())
            try:legacy=old_decompose(model,resource_profile="general-tensors-v1")
            except ValueError:
                with self.assertRaises(ValueError):decompose(model,resource_profile="general-tensors-v1")
                continue
            actual=decompose(model,resource_profile="general-tensors-v1")
            self.assertEqual(canonical(actual),canonical(legacy))
            accepted+=1
        self.assertEqual(accepted,9)

    def test_all_24_admission_and_request_reconstruction(self):
        catalog=json.loads((CASES/"catalog.json").read_text())
        for row in catalog["cases"]:
            with self.subTest(case=row["id"]):
                model=original(row["id"])
                assessment=assess_model(model,resource_profile="general-tensors-v1",options=OPTIONS)
                self.assertEqual(assessment["admission"],"accepted_pending_validation")
                lowered,_=decompose(model,resource_profile="general-tensors-v1")
                request=prepare_task(lowered,OPTIONS)
                self.assertEqual(assessment["identity"]["request_id"],request["request_id"])
                self.assertEqual(canonical(reconstruct_request(request)),canonical(request))
                for stage in ("compiler","encrypted","agent"):
                    self.assertEqual(assessment["stages"][stage]["status"],"unverified")

    def test_public_left_and_right_use_existing_linear_or_product(self):
        for name,strategy in [("public_left_0","plain_left_linear"),
                              ("public_right_0","plain_right_linear"),
                              ("public_left_1","plain_tensor_product"),
                              ("public_right_1","plain_tensor_product")]:
            _,binding=compare(original(name),4)
            self.assertEqual(binding["normalization"]["entries"][0]["plan"]["strategy"],strategy)

    def test_scalar_matmul_and_downstream_consumers(self):
        for size in (2,3,4):
            model=dict(format=FORMAT,id="scalar_"+str(size),
                inputs=[dict(name="left",shape=[size]),dict(name="right",shape=[size])],constants={},
                nodes=[dict(id="dot",op="matmul",inputs=["left","right"],attrs={},outputs=["scalar"]),
                       dict(id="square",op="square",inputs=["scalar"],attrs={},outputs=["squared"]),
                       dict(id="join",op="stack",inputs=["scalar","squared"],attrs=dict(axis=0),outputs=["joined"])],
                outputs=[dict(name="dot",value="scalar"),dict(name="pair",value="joined")])
            lowered,binding=compare(model)
            self.assertEqual(binding["normalization"]["entries"][0]["plan"]["result_shape"],[])
            self.assertEqual(binding["normalization"]["entries"][0]["plan"]["core_shape"],[1])

    def test_public_dot_both_sides_and_cipher_dot(self):
        for side in (None,"left","right"):
            model=dict(format=FORMAT,id="vector_dot",inputs=[],constants={},
                nodes=[dict(id="product",op="dot",inputs=["left","right"],attrs={},outputs=["result"])],
                outputs=[dict(name="out",value="result")])
            for name in ("left","right"):
                if side==name:model["constants"][name]=[.125,-.25,.5]
                else:model["inputs"].append(dict(name=name,shape=[3]))
            compare(model)

    def test_matrix_can_follow_shared_cipher_node_and_feed_multiple_outputs(self):
        for size in (2,3,4):
            model=dict(format=FORMAT,id="matrix_chain_"+str(size),
                inputs=[dict(name="x",shape=[2,size]),dict(name="y",shape=[size,2])],
                constants={"weight":[[.125],[-.375]]},
                nodes=[dict(id="dot",op="matmul",inputs=["x","y"],attrs={},outputs=["product"]),
                       dict(id="power",op="square",inputs=["product"],attrs={},outputs=["squared"]),
                       dict(id="project",op="matmul",inputs=["squared","weight"],attrs={},outputs=["projection"]),
                       dict(id="mean",op="mean",inputs=["projection"],attrs=dict(axes=[0],keepdims=False),outputs=["reduced"])],
                outputs=[dict(name="out",value="projection"),dict(name="mean",value="reduced"),
                         dict(name="shared",value="product")])
            _,binding=compare(model)
            self.assertEqual(len(binding["normalization"]["entries"]),2)
            self.assertEqual(set(binding["source_map"].values()),{"dot","power","project","mean"})

    def test_reference_does_not_use_matrix_plans_or_lowering(self):
        model=original("public_left_1")
        inputs=samples(model,4)[2]
        with (patch("operator_graph.plan",side_effect=AssertionError("plan")),
              patch("operator_graph.decompose",side_effect=AssertionError("decompose"))):
            expected=evaluate_reference(model,inputs)
        self.assertTrue(np.isfinite(expected["out"]).all())

    def test_plain_plain_is_rejected(self):
        model=original("public_right_0")
        model["constants"]["left"]=[[.1,.2,.3],[.4,.5,.6]]
        model["inputs"]=[dict(name="unused",shape=[1])]
        with self.assertRaisesRegex(ValueError,"encrypted operand"):decompose(model,resource_profile="general-tensors-v1")

    def test_other_operators_cannot_read_plain_first_operands(self):
        model=original("public_left_0")
        model["nodes"][0].update(op="square",inputs=["public_weight"])
        with self.assertRaisesRegex(ValueError,"First operand"):decompose(model,resource_profile="general-tensors-v1")

    def test_bad_contraction_and_batch_rejected(self):
        for shapes in [([2,3],[4,2]),([2,2,3],[3,3,2])]:
            model=original()
            model["inputs"][0]["shape"]=shapes[0]
            model["inputs"][1]["shape"]=shapes[1]
            with self.assertRaises(ValueError):decompose(model,resource_profile="general-tensors-v1")

    def test_rank_33_and_original_profile_bound_remain_rejected(self):
        model=original("higher_rank_cipher_2")
        model["inputs"][0]["shape"]=[1]+model["inputs"][0]["shape"]
        with self.assertRaises(ValueError):decompose(model,resource_profile="general-tensors-v1")
        with self.assertRaises(ValueError):decompose(original("higher_rank_cipher_1"),resource_profile="legacy-v1")

    def test_materialized_product_budget_rejected(self):
        model=original()
        model["inputs"]=[dict(name="left",shape=[17,256]),dict(name="right",shape=[256,17])]
        # Inputs fit the medium profile, but product materialization exceeds 4096.
        model["inputs"]=[dict(name="left",shape=[16,32]),dict(name="right",shape=[32,16])]
        with self.assertRaisesRegex(ValueError,"product tensor budget"):decompose(model,resource_profile="medium-v1")

    def test_attrs_version_arity_and_dynamic_values_rejected(self):
        for change in [lambda m:m.update(format="poseidon-operator-graph-v3"),
                       lambda m:m["nodes"][0].update(attrs={"transpose":True}),
                       lambda m:m["nodes"][0].update(inputs=["left"]),
                       lambda m:m["nodes"][0].update(inputs=["left","result"]),
                       lambda m:m["inputs"][0].update(shape=[True,3])]:
            model=original();change(model)
            with self.assertRaises((ValueError,TypeError,KeyError)):decompose(model,resource_profile="general-tensors-v1")

    def test_source_aliases_and_multiple_output_bindings_remain_strict(self):
        model=original()
        model["nodes"][0]["outputs"]=["left"]
        with self.assertRaises(ValueError):decompose(model,resource_profile="general-tensors-v1")
        model=original();model["outputs"]=[dict(name="out",value="unknown")]
        with self.assertRaises(ValueError):decompose(model,resource_profile="general-tensors-v1")

    def test_provenance_rejects_altered_weights_and_normalization_records(self):
        model=original("public_left_1")
        lowered,binding=decompose(model,resource_profile="general-tensors-v1")
        report=dict(original_sha256=digest(model),model_sha256=digest(lowered),
                    reference_status="passed",approximation_accuracy_certified=False)
        provenance=dict(original=model,decomposition=binding,reference_report=report)
        self.assertEqual(check_provenance(provenance,lowered),binding)
        changed=copy.deepcopy(provenance)
        changed["decomposition"]["normalization"]["entries"][0]["plan"]["public_side"]="right"
        with self.assertRaisesRegex(ValueError,"provenance mismatch"):check_provenance(changed,lowered)
        changed=copy.deepcopy(provenance)
        changed["original"]["constants"]["public_weight"][0][0][0]+=.125
        with self.assertRaises(ValueError):check_provenance(changed,lowered)

    def test_auto_packing_is_only_for_explicit_v2_capacity_import(self):
        model=original();lowered,_=decompose(model,resource_profile="general-tensors-v1")
        self.assertEqual(preparation_options(model,lowered,{})["capacity_packing_version"],
                         "capacity-public-tensors-v9")
        legacy=copy.deepcopy(model);legacy["format"]=V1
        self.assertEqual(preparation_options(legacy,lowered,{}),{})
        supplied={"capacity_packing_version":"capacity-public-tensors-v8"}
        self.assertEqual(preparation_options(model,lowered,supplied),supplied)

    def test_v2_cli_plan_enters_existing_poseidon_environment_once(self):
        import sys
        import model_import_cli
        argv=["prepare_model.py","--graph",str(CASES.parent/"operator-matrix-v2/public_left_0.json"),
              "--resource-profile","general-tensors-v1"]
        with (patch.object(sys,"argv",argv),
              patch("hecate_python_env.enter_nix",return_value=0) as entering):
            self.assertEqual(model_import_cli.main(),0)
        self.assertEqual(entering.call_count,1)
        self.assertEqual(entering.call_args.kwargs["environment"],"poseidon-cpu")
        self.assertIn("--inside",entering.call_args.args[0])

    def test_v2_json_guard_precedes_copying_non_data_objects(self):
        class NotData:
            called=False
            def __deepcopy__(self,memo):
                self.called=True
                raise AssertionError("Object executed")
        model=original();value=NotData();model["constants"]["bad"]=value
        with self.assertRaises((ValueError,TypeError)):decompose(model,resource_profile="general-tensors-v1")
        self.assertFalse(value.called)

    def matrix_domain_graph(self, width):
        return dict(format=FORMAT,id="matrix_domain",
            inputs=[dict(name="left",shape=[width]),dict(name="right",shape=[width])],
            constants={},nodes=[
                dict(id="dot_product",op="matmul",inputs=["left","right"],attrs={},outputs=["scalar"]),
                dict(id="collect",op="stack",inputs=["scalar","scalar"],attrs=dict(axis=0),outputs=["pair"]),
                dict(id="reduce",op="mean",inputs=["pair"],attrs=dict(axes=[1],keepdims=False),outputs=["vector"]),
                dict(id="activation",op="exp",inputs=["vector"],attrs=dict(approximation=dict(
                    method="taylor",degree=1,domain=[-1.,1.])),outputs=["output"])],
            outputs=[dict(name="result",value="output")])

    def test_v2_domain_preflight_uses_scalar_carriers_at_each_consumer(self):
        from execution_input_domain import check_approximation_domains
        for width in (1,2,3):
            model=self.matrix_domain_graph(width)
            decompose(model,resource_profile="general-tensors-v1")
            values=dict(left=np.full(width,.1),right=np.full(width,.2))
            with self.subTest(width=width):
                with patch("operator_graph.decompose",side_effect=AssertionError("No DSL/reference lowering")):
                    report=check_approximation_domains(model,[values])
                self.assertEqual(report["status"],"passed")
                self.assertEqual(report["node_count"],1)

    def test_v2_domain_rejection_keeps_location_and_private_values_hidden(self):
        from execution_input_domain import check_approximation_domains,ApproximationDomainRejected
        for width in (1,2,3):
            model=self.matrix_domain_graph(width)
            decompose(model,resource_profile="general-tensors-v1")
            values=dict(left=np.full(width,2.123456789),right=np.ones(width))
            with self.subTest(width=width):
                with self.assertRaises(ApproximationDomainRejected) as caught:
                    check_approximation_domains(model,[values])
                failure=caught.exception.public_failure
                self.assertEqual(failure["location"]["id"],"activation")
                self.assertEqual(failure["location"]["index"],3)
                self.assertNotIn("2.123456789",json.dumps(failure))

    def test_capabilities_register_input_without_claiming_execution(self):
        report=input_sources()["graph"]
        self.assertIn(FORMAT,report["formats"])
        self.assertEqual(report["matrix_normalization"]["scalar_carrier_shape"],[1])
        for stage in ("compiler_status","encrypted_status","agent_status"):
            self.assertEqual(report["matrix_normalization"][stage],"unverified")

if __name__=="__main__":
    unittest.main()
