"""Pure preparation/static/reference controls; no SDK, compiler, FHE or API."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parents[2]
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("replicated_rule_tested", HERE/"replicated_scalar_lowering.py")
rule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule)
from benchmark_math import evaluate
from benchmark_graph import validate
from unified_graph_contract import prepare, validate_request, validate_candidate
from unified_request_storage import restore_public_constants

RESULTS = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-ordered-backend-v1")
NAMES = ("self_post_linear_residual__declared_shallow_v1",
         "linear_into_separate_cross_none__declared_shallow_v1",
         "boolean_padding_residual__declared_shallow_v1")


def original(name):
    return json.loads((RESULTS/name/"request.json").read_bytes())


def plain_source(source, request, sample):
    """Independent bounded AST interpreter over plaintext periodic vectors."""
    q = copy.deepcopy(request)
    restore_public_constants(q)
    period = q["layout"]["input_slot_period"]
    env = {name:np.asarray(value if type(value) is list else [value], dtype=np.float64)
           for name, value in q["public_constants"].items()}
    for binding in q["layout"]["inputs"]:
        flat = np.asarray(sample[binding["name"]], dtype=np.float64).reshape(-1)
        start = binding.get("offset", 0)
        values = flat[start:start+binding.get("elements", flat.size)]
        env[binding["dsl_name"]] = np.pad(values, (0, period-values.size))
    env["zero_ct"] = np.zeros(period)
    tree = ast.parse(source)
    assert len(tree.body) == 1 and tree.body[0].name == "golden"
    def expression(node):
        if type(node) is ast.Name: return env[node.id]
        if type(node) is ast.Constant and type(node.value) in (int, float): return node.value
        if type(node) is ast.UnaryOp and type(node.op) is ast.USub: return -expression(node.operand)
        if type(node) is ast.BinOp:
            left, right = expression(node.left), expression(node.right)
            if type(node.op) is ast.Add: return left+right
            if type(node.op) is ast.Sub: return left-right
            if type(node.op) is ast.Mult: return left*right
        if type(node) is ast.Call and type(node.func) is ast.Attribute and node.func.attr == "rotate":
            return np.roll(expression(node.func.value), -node.args[0].value)
        if type(node) in (ast.List, ast.Tuple): return [expression(x) for x in node.elts]
        raise AssertionError("Unexpected generated AST "+type(node).__name__)
    for statement in tree.body[0].body:
        if type(statement) is ast.Assign:
            env[statement.targets[0].id] = expression(statement.value)
        elif type(statement) is ast.Return:
            return expression(statement.value)
        else: raise AssertionError("Unexpected generated statement")
    raise AssertionError("No return")


def sample(model, group):
    result = {}
    for index, item in enumerate(model["inputs"]):
        n = math.prod(item["shape"])
        values = np.sin(np.arange(n, dtype=np.float64)+.17+group*.31+index*.43)*.23
        result[item["name"]] = values.reshape(item["shape"])
    return result


def generic_request():
    q = original(NAMES[0])
    graph = dict(format="poseidon-model-graph-v2", id="generic_cell_layout",
        resource_profile="general-tensors-v1", inputs=[dict(name="a", shape=[2,3]), dict(name="b", shape=[2,2])],
        constants=dict(w=[[.21,-.37,.43],[-.19,.29,.53]], bias=[.07,-.11]),
        nodes=[dict(id="n0",op="linear",inputs=["a","w","bias"],outputs=["l"],attrs={}),
               dict(id="n1",op="permute",inputs=["l"],outputs=["p"],attrs=dict(dims=[1,0])),
               dict(id="n2",op="sum",inputs=["p"],outputs=["r"],attrs=dict(axes=[1],keepdims=True)),
               dict(id="n3",op="multiply",inputs=["p","r"],outputs=["m"],attrs={}),
               dict(id="n4",op="add",inputs=["m","b"],outputs=["z"],attrs={}),
               dict(id="n5",op="split",inputs=["z"],outputs=["u","v"],attrs=dict(axis=1,sections=[1,1])),
               dict(id="n6",op="concat",inputs=["v","u"],outputs=["joined"],attrs=dict(axis=1))],
        outputs=[dict(name="result",value="joined"),dict(name="rows",value="r"),dict(name="part",value="u")])
    return prepare(graph, q["compiler_profile_sha256"], q["compiler_configuration"],
                   chunk_period=4, public_weight_policy="descriptors-math-roles-v2")


class ReplicatedScalarLowering(unittest.TestCase):
    def verify_math(self, request, groups=4):
        before = copy.deepcopy(request)
        text, record = rule.lower(request)
        self.assertEqual(request, before)
        q = copy.deepcopy(request)
        validate_request(q)
        validate_candidate(dict(schema=1,request_id=q["request_id"],hecate_source=text), q)
        self.assertEqual(record["physical_output_chunks"], q["layout"]["output_ciphertexts"])
        self.assertEqual(record["expanded_native_work"] <= 1024, True)
        self.assertEqual(record["operations"]["ast_nodes"] <= 4096, True)
        for group in range(groups):
            inputs = sample(q["model"], group)
            actual = plain_source(text, q, inputs)
            expected = evaluate(q["model"], inputs)
            observed = [actual[chunk][slot] for chunk, slot in q["layout"]["output_selectors"]]
            want = np.concatenate([expected[o["name"]].reshape(-1) for o in q["model"]["outputs"]])
            np.testing.assert_allclose(observed, want, atol=2e-12, rtol=2e-12)
        return text, record, q

    def test_original_self_request_four_physical_outputs(self):
        self.verify_math(original(NAMES[0]))

    def test_original_cross_separate_dimensions_and_none_cipher_boundary(self):
        text, record, q = self.verify_math(original(NAMES[1]))
        self.assertEqual(record["physical_input_chunks"], 7)
        self.assertEqual(record["physical_output_chunks"], 4)
        self.assertEqual([x["name"] for x in q["model"]["outputs"]], ["result","projected"])
        self.assertNotIn("None", text)

    def test_original_boolean_mask_padding_and_two_element_tail(self):
        text, record, q = self.verify_math(original(NAMES[2]))
        self.assertEqual(record["physical_output_chunks"], 3)
        self.assertEqual(record["outputs"][-1]["elements"], 2)
        self.assertEqual(record["outputs"][-1]["offset"], 4)

    def test_generic_linear_views_partial_reduction_broadcast_split_multioutput(self):
        self.verify_math(generic_request())

    def test_immutable_complete_binding_and_real_module_hashes(self):
        q = generic_request()
        text, record = rule.lower(q)
        self.assertEqual(record["input_request_sha256"], rule.digest(q))
        self.assertEqual(record["candidate_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(record["binding"], rule.digest({k:v for k,v in record.items() if k != "binding"}))
        for item in record["implementation"].values():
            self.assertEqual(item["sha256"], hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest())
        for field in ("request_changed","model_changed","public_registry_changed","compiler_profile_changed",
                      "reference_changed","provider_candidate_rewritten","new_agent_generation",
                      "compiler_run","encrypted_execution","depth_is_capacity_proof"):
            self.assertFalse(record[field])

    def test_every_fallback_literal_has_original_exact_coordinate_and_source_location(self):
        q = generic_request()
        text, record = rule.lower(q)
        self.assertTrue(record["missing_literals"])
        self.assertEqual(record["literal_source_map_sha256"], rule.digest(record["missing_literals"]))
        for row in record["missing_literals"]:
            self.assertEqual(row["origins_sha256"], rule.digest(row["origins"]))
            self.assertEqual(row["source_locations_sha256"], rule.digest(row["source_locations"]))
            self.assertTrue(row["source_locations"])
            for origin in row["origins"]:
                self.assertEqual(origin["kind"], "model_constant")
                value = q["model"]["constants"][origin["logical_name"]]
                self.assertEqual(origin["constant_sha256"], rule.digest(value))
                for index in origin["coordinate"]: value = value[index]
                self.assertEqual(float(value).hex(), row["value_hex"])
        self.assertNotIn("pv", json.dumps(record["missing_literals"]))

    def test_rule_only_authority_rejects_provider_and_replay(self):
        for kind in ("provider", "real_agent", "replay", None):
            with self.assertRaisesRegex(ValueError, "Rule-only"):
                rule.lower(generic_request(), source_kind=kind)

    def test_missing_literal_not_authorized_by_an_arbitrary_pv(self):
        q = generic_request()
        em = rule.ReplicatedEmitter(q, {})
        with self.assertRaisesRegex(ValueError, "original public provenance"):
            em.literal(.987654321)

    def test_exact64_signed_zero_and_finite_boundary(self):
        self.assertNotEqual(rule.exact(0.), rule.exact(-0.))
        for value in (float("nan"),float("inf"),1024.01,True,"1"):
            with self.assertRaises(ValueError): rule.exact(value)

    def test_changed_public_registry_rejected(self):
        q = generic_request()
        q["public_constants"]["forged"] = .23
        with self.assertRaises(ValueError): rule.lower(q)

    def test_changed_physical_output_and_input_layout_rejected(self):
        for direction in ("inputs","outputs"):
            q = generic_request()
            if direction == "inputs": q["layout"]["inputs"][0]["offset"] = 1
            else: q["layout"]["outputs"][0]["chunks"][0]["elements"] = 3
            with self.assertRaises(ValueError): rule.lower(q)

    def test_unbound_derived_number_rejected_despite_finite_magnitude(self):
        q = generic_request()
        em = rule.ReplicatedEmitter(q, rule.literal_sources(q["model"],validate(q["model"])))
        with self.assertRaisesRegex(ValueError,"original public provenance"):
            em.literal(.887654321)

    def test_registered_coefficients_use_original_plain_names(self):
        q = original(NAMES[0]); restore_public_constants(q)
        em = rule.ReplicatedEmitter(q, rule.literal_sources(q["model"],validate(q["model"])))
        names = {float(v).hex():k for k,v in q["public_constants"].items() if type(v) in (int,float)}
        for node in q["model"]["nodes"]:
            if node["op"] == "polynomial":
                for value in q["model"]["constants"][node["inputs"][1]]:
                    self.assertEqual(em.literal(value), names[float(value).hex()])
        self.assertFalse(em.fallbacks)

    def test_derived_mean_divisor_rebuilt_from_original_axes(self):
        q = generic_request()
        model = copy.deepcopy(q["model"])
        model["nodes"][2] = dict(id="n2",op="mean",inputs=["p"],outputs=["r"],attrs=dict(axes=[1],keepdims=True))
        entries = rule.literal_sources(model, validate(model))
        origins = entries[float(.5).hex()]
        row = next(row for row in origins if row["kind"] == "mean_divisor")
        self.assertEqual(row["count"],2)
        self.assertEqual(row["node_id"],"n2")
        self.assertEqual(row["attributes_sha256"],rule.digest(model["nodes"][2]["attrs"]))

    def test_original_tensor_digest_is_computed_once_not_per_coordinate(self):
        q = generic_request()
        tensor = q["model"]["constants"]["w"]
        calls = []
        actual = rule.digest
        def observed(value):
            if value is tensor: calls.append(value)
            return actual(value)
        with mock.patch.object(rule, "digest", side_effect=observed):
            rule.literal_sources(q["model"], validate(q["model"]))
        self.assertEqual(len(calls), 1)

    def test_large_input_floor_rejects_before_weight_authority_is_built(self):
        template = original(NAMES[0])
        model = dict(format="poseidon-model-graph-v2",id="large_input_work_floor",resource_profile="general-tensors-v1",
                     inputs=[dict(name="a",shape=[256])],constants={},
                     nodes=[dict(id="n",op="reshape",inputs=["a"],outputs=["v"],attrs=dict(shape=[256]))],
                     outputs=[dict(name="result",value="v")])
        q = prepare(model,template["compiler_profile_sha256"],template["compiler_configuration"],
                    chunk_period=256,public_weight_policy="descriptors-math-roles-v2")
        with mock.patch.object(rule,"literal_sources",side_effect=AssertionError("Too late")):
            with self.assertRaisesRegex(ValueError,"primitive lower bound"):
                rule.lower(q)

    def test_final_pack_lower_bound_does_not_guess_nonzero_for_zero_model(self):
        q = generic_request()
        model = copy.deepcopy(q["model"])
        model["constants"]["zero"] = 0.
        model["nodes"].append(dict(id="zero",op="multiply",inputs=["joined","zero"],outputs=["all_zero"],attrs={}))
        model["outputs"] = [dict(name="result",value="all_zero")]
        template = original(NAMES[0])
        q = prepare(model,template["compiler_profile_sha256"],template["compiler_configuration"],
                    chunk_period=4,public_weight_policy="descriptors-math-roles-v2")
        floor = rule.budget_floor(q,validate(model))
        self.assertEqual(floor["final_pack"],0)
        self.assertEqual(floor["proven_nonzero_output_cells"],0)
        self.assertGreater(floor["input_extraction"],0)

    def test_no_global_emitter_monkeypatch_or_dynamic_candidate_execution(self):
        tree = ast.parse((HERE/"replicated_scalar_lowering.py").read_text())
        calls = [n.func.id for n in ast.walk(tree) if type(n) is ast.Call and type(n.func) is ast.Name]
        self.assertFalse(set(calls)&{"exec","eval","getattr","setattr"})
        for node in ast.walk(tree):
            if type(node) is ast.Assign:
                self.assertFalse(any(type(target) is ast.Attribute and type(target.value) is ast.Name and
                                     target.value.id in ("Emitter","Scalar","graph_lowering","capacity_lowering")
                                     for target in node.targets))


if __name__ == "__main__":
    unittest.main()
