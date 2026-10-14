"""Pure slot-math/static controls: no SDK/compiler/FHE/API entry points."""
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
spec = importlib.util.spec_from_file_location("mixed_rule_tested", HERE/"mixed_packed_scalar_lowering.py")
rule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rule)
from benchmark_math import evaluate
from unified_graph_contract import prepare, validate_request, validate_candidate
from unified_request_storage import restore_public_constants

RESULTS = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-ordered-backend-v1")
NAMES = ("self_post_linear_residual__declared_shallow_v1", "linear_into_separate_cross_none__declared_shallow_v1",
         "boolean_padding_residual__declared_shallow_v1")


def original(index=0):
    return json.loads((RESULTS/NAMES[index]/"request.json").read_bytes())


def periodic_source(source, request, sample, *, tail=0.):
    """Independent non-executing AST interpreter with deliberately poisoned tails."""
    q = copy.deepcopy(request)
    restore_public_constants(q)
    period = q["layout"]["input_slot_period"]
    public = {name:np.asarray(value if type(value) is list else [value], dtype=np.float64)
              for name, value in q["public_constants"].items()}
    env = dict(public)
    for binding in q["layout"]["inputs"]:
        flat = np.asarray(sample[binding["name"]], dtype=np.float64).reshape(-1)
        start = binding.get("offset", 0)
        values = flat[start:start+binding.get("elements", flat.size)]
        env[binding["dsl_name"]] = np.pad(values, (0, period-values.size), constant_values=tail)
    env["zero_ct"] = np.zeros(period)
    functions = {fn.name:fn for fn in ast.parse(source).body}
    def invoke(name, arguments):
        fn = functions[name]
        local = dict(public)
        local.update(zip((arg.arg for arg in fn.args.args), arguments))
        def expression(node):
            if type(node) is ast.Name: return local[node.id]
            if type(node) is ast.Constant and type(node.value) in (int, float): return node.value
            if type(node) is ast.UnaryOp and type(node.op) is ast.USub: return -expression(node.operand)
            if type(node) is ast.BinOp:
                left, right = expression(node.left), expression(node.right)
                if type(node.op) is ast.Add: return left+right
                if type(node.op) is ast.Sub: return left-right
                if type(node.op) is ast.Mult: return left*right
            if type(node) is ast.Call and type(node.func) is ast.Attribute and node.func.attr == "rotate":
                return np.roll(expression(node.func.value), -expression(node.args[0]))
            if type(node) is ast.Call and type(node.func) is ast.Name and node.func.id in functions:
                return invoke(node.func.id, [expression(arg) for arg in node.args])
            if type(node) in (ast.List, ast.Tuple): return [expression(item) for item in node.elts]
            raise AssertionError("Unexpected generated AST "+type(node).__name__)
        for statement in fn.body:
            if type(statement) is ast.Assign:
                local[statement.targets[0].id] = expression(statement.value)
            elif type(statement) is ast.Return:
                return expression(statement.value)
            else: raise AssertionError("Unexpected generated statement")
        raise AssertionError("No return")
    return invoke("golden", [env[arg.arg] for arg in functions["golden"].args.args])


def sample(model, group):
    result = {}
    for index, item in enumerate(model["inputs"]):
        values = np.sin(np.arange(math.prod(item["shape"]), dtype=np.float64)+.17+group*.31+index*.43)*.23
        result[item["name"]] = values.reshape(item["shape"])
    return result


def prepared(model):
    q = original()
    return prepare(model, q["compiler_profile_sha256"], q["compiler_configuration"],
                   chunk_period=4, public_weight_policy="descriptors-math-roles-v2")


def generic_request(*, late_linear=False):
    graph = dict(format="poseidon-model-graph-v2", id="mixed_generic_graph",resource_profile="general-tensors-v1",
        inputs=[dict(name="a",shape=[2,3]),dict(name="b",shape=[2,2])],
        constants=dict(w=[[.21,-.37,.43],[-.19,.29,.53]], bias=[.07,-.11]),
        nodes=[dict(id="n0",op="linear",inputs=["a","w","bias"],outputs=["l"],attrs={}),
               dict(id="n1",op="permute",inputs=["l"],outputs=["p"],attrs=dict(dims=[1,0])),
               dict(id="n2",op="sum",inputs=["p"],outputs=["r"],attrs=dict(axes=[1],keepdims=True)),
               dict(id="n3",op="multiply",inputs=["p","r"],outputs=["m"],attrs={}),
               dict(id="n4",op="add",inputs=["m","b"],outputs=["z"],attrs={}),
               dict(id="n5",op="split",inputs=["z"],outputs=["u","v"],attrs=dict(axis=1,sections=[1,1])),
               dict(id="n6",op="concat",inputs=["v","u"],outputs=["joined"],attrs=dict(axis=1))],
        outputs=[dict(name="result",value="joined"),dict(name="rows",value="r"),dict(name="part",value="u")])
    if late_linear:
        graph["constants"].update(late_w=[[.3,-.1],[.2,.4]],late_b=[.02,-.03])
        graph["nodes"].append(dict(id="late",op="linear",inputs=["joined","late_w","late_b"],outputs=["late_y"],attrs={}))
        graph["outputs"][0]["value"] = "late_y"
    return prepared(graph)


def view_request():
    return prepared(dict(format="poseidon-model-graph-v2",id="views_only",resource_profile="general-tensors-v1",
        inputs=[dict(name="a",shape=[2,2])],constants={},
        nodes=[dict(id="v",op="reshape",inputs=["a"],outputs=["r"],attrs=dict(shape=[4]))],
        outputs=[dict(name="result",value="r")]))


def unit_materializer(public=None):
    q = dict(layout=dict(input_slot_period=4), public_constants=dict(mask0=[1.,0.,0.,0.]), model=dict(constants={},nodes=[]))
    if public: q["public_constants"].update(public)
    em = rule.MixedEmitter(q, {"shapes":{}})
    return em, rule.Materializer(em)


class MixedPackedScalarLowering(unittest.TestCase):
    def verify_math(self, request, *, tail=0.):
        before = copy.deepcopy(request)
        text, record = rule.lower(request)
        self.assertEqual(request, before)
        q = copy.deepcopy(request)
        validate_request(q)
        validate_candidate(dict(schema=1,request_id=q["request_id"],hecate_source=text),q)
        self.assertLessEqual(record["expanded_native_work"],1024)
        self.assertLessEqual(record["operations"]["ast_nodes"],4096)
        self.assertLessEqual(record["operations"]["source_bytes"],65536)
        for group in range(4):
            inputs = sample(q["model"],group)
            actual = periodic_source(text,q,inputs,tail=tail)
            expected = evaluate(q["model"],inputs)
            got = [actual[chunk][slot] for chunk,slot in q["layout"]["output_selectors"]]
            want = np.concatenate([expected[o["name"]].reshape(-1) for o in q["model"]["outputs"]])
            np.testing.assert_allclose(got,want,rtol=2e-12,atol=2e-12)
        return text,record,q

    def test_original_self_math_and_all_five_frozen_linears(self):
        _,record,q = self.verify_math(original(0))
        actual = [row["node_id"] for row in record["transitions"] if row["representation"] == "bound_packed_linear"]
        self.assertEqual(actual,[node["id"] for node in q["model"]["nodes"] if node["op"] == "linear"])
        self.assertEqual(len(actual),5)
        self.assertEqual(record["physical_output_chunks"],4)

    def test_original_cross_dimensions_none_and_named_projected_packed_output(self):
        _,record,q = self.verify_math(original(1))
        self.assertEqual(record["physical_input_chunks"],7)
        self.assertEqual(record["physical_output_chunks"],4)
        self.assertEqual([o["name"] for o in q["model"]["outputs"]],["result","projected"])
        self.assertFalse(any(row["boundary"].get("output_name") == "projected" for row in record["packs"]))

    def test_original_boolean_masks_padding_and_tail2(self):
        _,record,_ = self.verify_math(original(2),tail=7.3)
        self.assertEqual(record["physical_output_chunks"],3)
        self.assertEqual(record["outputs"][-1]["elements"],2)
        self.assertEqual(record["outputs"][-1]["offset"],4)

    def test_generic_views_partial_reduction_cipher_broadcast_split_multioutput(self):
        self.verify_math(generic_request(),tail=-8.2)

    def test_uniform_public_tail_cannot_leak_through_split_slice_or_late_linear(self):
        q = prepared(dict(format="poseidon-model-graph-v2",id="uniform_tail_views",resource_profile="general-tensors-v1",
            inputs=[dict(name="a",shape=[2,3])], constants=dict(c=.73,w=[[.2,-.3,.4],[-.1,.5,.7]],bias=[.03,-.02]),
            nodes=[dict(id="uniform",op="add",inputs=["a","c"],outputs=["a_bias"],attrs={}),
                   dict(id="split",op="split",inputs=["a_bias"],outputs=["u","v"],attrs=dict(axis=1,sections=[1,2])),
                   dict(id="join",op="concat",inputs=["v","u"],outputs=["joined"],attrs=dict(axis=1)),
                   dict(id="slice",op="slice",inputs=["a_bias"],outputs=["narrow"],attrs=dict(axis=1,start=0,stop=2,step=1)),
                   dict(id="late",op="linear",inputs=["joined","w","bias"],outputs=["y"],attrs={})],
            outputs=[dict(name="result",value="y"),dict(name="part",value="u"),dict(name="slice",value="narrow")]))
        _,record,_ = self.verify_math(q,tail=91.)
        self.assertTrue(any(row["boundary"].get("node_id") == "late" for row in record["packs"]))
        self.assertTrue(any(row["representation"] == "packed_pointwise" for row in record["transitions"]))

    def test_all_public_helper_arithmetic_explicitly_rejected(self):
        em,m = unit_materializer()
        with self.assertRaisesRegex(ValueError,"all-public"):
            rule.packed_binary("add",[np.array([.2]),np.array([.3])],[1],dict(id="public"),{},m)

    def test_uncovered_batch_norm_is_explicitly_rejected(self):
        q = prepared(dict(format="poseidon-model-graph-v2",id="bn_reject",resource_profile="general-tensors-v1",
            inputs=[dict(name="a",shape=[2,2])], constants=dict(mean=[.1,.2],var=[.7,.8],gamma=[.3,.4],beta=[.01,.02]),
            nodes=[dict(id="bn",op="batch_norm",inputs=["a","mean","var","gamma","beta"],outputs=["y"],attrs=dict(eps=.001))],
            outputs=[dict(name="result",value="y")]))
        with self.assertRaisesRegex(ValueError,"Unsupported staged mixed"):
            rule.lower(q)

    def test_early_statement_byte_budget_does_not_wait_for_final_candidate_check(self):
        em,_ = unit_materializer()
        with self.assertRaisesRegex(ValueError,"statement source budget"):
            em.emit("x + "+"a"*65536)

    def test_original_primitive_budget_is_not_expanded(self):
        em,_ = unit_materializer()
        for _ in range(1024): em.emit("x + x")
        with self.assertRaisesRegex(ValueError,"operation budget"):
            em.emit("x + x")

    def test_late_linear_packs_before_existing_helper(self):
        _,record,_ = self.verify_math(generic_request(late_linear=True))
        rows = [row for row in record["packs"] if row["boundary"].get("node_id") == "late"]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["packing"],"canonical_C_order_prefixes")
        self.assertFalse(any(row["boundary"].get("output_name") == "result" for row in record["packs"]))

    def test_pure_canonical_views_emit_no_cipher_operations_or_cells(self):
        _,record,_ = self.verify_math(view_request())
        self.assertEqual(record["materialized_cells"],[])
        self.assertEqual(record["packs"],[])
        self.assertEqual(record["emitted_operations"],0)

    def test_lazy_cell_cache_uses_immutable_producer_slot_period(self):
        em,m = unit_materializer(dict(mask2=[0.,0.,1.,0.]))
        value = m.packed_value([rule.Scalar(em,"x")],[4],producer={})
        again = m.packed_value([rule.Scalar(em,"x")],[2,2],producer={})
        a = m.cell(value.entries[2]); n = len(em.lines)
        self.assertIs(a,m.cell(again.entries[1,0]))
        self.assertEqual(len(em.lines),n)
        self.assertEqual(len(m.materializations),1)
        self.assertEqual(m.materializations[0]["route"],"bound_slot_mask")
        self.assertEqual(em.lines[0],"    v0 = x * mask2")

    def test_one_cell_does_not_extract_other_logical_slots(self):
        em,m = unit_materializer()
        value = m.packed_value([rule.Scalar(em,"x")],[2],producer={})
        m.cell(value.entries[0])
        self.assertEqual(len(m.cells),1)
        self.assertEqual([row["logical_index"] for row in m.materializations],[0])

    def test_tail_slot_and_foreign_emitter_rejected(self):
        em,m = unit_materializer()
        value = m.packed_value([rule.Scalar(em,"x")],[2],producer={})
        with self.assertRaisesRegex(ValueError,"invalid logical cell"):
            m.cell(rule.BoundCell(value.entries[0].storage,2))
        other,_ = unit_materializer()
        with self.assertRaisesRegex(ValueError,"foreign"):
            m.cell(rule.Scalar(other,"x"))

    def test_mask0_fallback_is_bound_rotation_then_replication(self):
        em,m = unit_materializer()
        value = m.packed_value([rule.Scalar(em,"x")],[4],producer={})
        m.cell(value.entries[3])
        self.assertEqual(m.materializations[0]["route"],"bound_mask0_rotated_source")
        self.assertIn("x.rotate(1).rotate(2)",em.lines[0])

    def test_cell_mask_direction_and_full_period_are_verified(self):
        for wrong in ([0.,0.,.5,0.],[0.,1.,0.,0.],[0.,0.,1.]):
            em,m = unit_materializer(dict(mask2=wrong))
            value = m.packed_value([rule.Scalar(em,"x")],[4],producer={})
            with self.assertRaisesRegex(ValueError,"one-hot"):
                m.cell(value.entries[2])

    def test_mask0_rejects_short_and_nonbinary(self):
        for mask in ([1.,0.],[.5,0.,0.,0.]):
            with self.assertRaisesRegex(ValueError,"full-period"):
                unit_materializer(dict(mask0=mask))

    def test_reshape_is_canonical_but_head_permute_is_only_an_index_map(self):
        em,m = unit_materializer()
        value = m.packed_value([rule.Scalar(em,"x")],[2,2],producer={})
        reshaped = rule.view("reshape",[value],dict(shape=[4]))
        permuted = rule.view("permute",[value],dict(dims=[1,0]))
        self.assertIs(reshaped.canonical_storage(),value.canonical_storage())
        self.assertIsNone(permuted.canonical_storage())
        self.assertEqual([ref.index for ref in permuted.entries.flat],[0,2,1,3])
        self.assertEqual(em.lines,[])

    def test_output_pack_cache_reuses_exact_cell_sequence(self):
        em,m = unit_materializer()
        value = m.packed_value([rule.Scalar(em,"x")],[2,2],producer={})
        permuted = rule.view("permute",[value],dict(dims=[1,0]))
        first = m.chunks(permuted,boundary={}); n = len(em.lines)
        second = m.chunks(permuted,boundary={})
        self.assertEqual(first.parts,second.parts)
        self.assertEqual(len(em.lines),n)
        self.assertEqual(len(m.packs),1)

    def test_complete_immutable_source_binding_and_actual_implementation_hashes(self):
        text,record,_ = self.verify_math(generic_request())
        self.assertEqual(record["binding"],rule.digest({k:v for k,v in record.items() if k != "binding"}))
        self.assertEqual(record["candidate_sha256"],hashlib.sha256(text.encode()).hexdigest())
        for item in record["implementation"].values():
            self.assertEqual(item["sha256"],hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest())
        for key in ("request_changed","model_changed","public_registry_changed","compiler_profile_changed",
                    "reference_changed","provider_candidate_rewritten","new_agent_generation","compiler_run",
                    "encrypted_execution","depth_is_capacity_proof","peak_allocation_is_known"):
            self.assertFalse(record[key])

    def test_packed_linears_do_not_walk_every_original_weight_literal(self):
        with mock.patch.object(rule.scalar_rule,"literal_sources",side_effect=AssertionError("Eager weight walk")):
            _,record = rule.lower(original(0))
        self.assertFalse(record["public_origin_authority_loaded"])
        self.assertFalse(record["missing_literals"])

    def test_unregistered_scalar_only_uses_existing_original_coordinate_authority(self):
        em,m = unit_materializer(dict(pv0=[.987654321]*4))
        with self.assertRaisesRegex(ValueError,"original public provenance"):
            em.literal(.987654321)
        self.assertTrue(em.authority_loaded)
        self.assertFalse(em.fallbacks)

    def test_rule_authority_rejects_provider_and_replay(self):
        for kind in ("provider","real_agent","replay",None):
            with self.assertRaisesRegex(ValueError,"Rule-only"):
                rule.lower(view_request(),source_kind=kind)

    def test_changed_registry_or_capacity_binding_is_rejected(self):
        for field in ("registry","input","output"):
            q = generic_request()
            if field == "registry": q["public_constants"]["forged"] = .17
            elif field == "input": q["layout"]["inputs"][0]["offset"] = 1
            else: q["layout"]["outputs"][0]["chunks"][0]["elements"] = 3
            with self.assertRaises(ValueError): rule.lower(q)

    def test_no_model_or_family_dispatch_dynamic_execution_or_global_monkeypatch(self):
        tree = ast.parse((HERE/"mixed_packed_scalar_lowering.py").read_text())
        calls = [node.func.id for node in ast.walk(tree) if type(node) is ast.Call and type(node.func) is ast.Name]
        self.assertFalse(set(calls)&{"exec","eval","getattr","setattr"})
        literals = [node.value for node in ast.walk(tree) if type(node) is ast.Constant and type(node.value) is str]
        self.assertFalse(any(name in text for name in NAMES for text in literals))
        for node in ast.walk(tree):
            if type(node) is ast.Assign:
                self.assertFalse(any(type(target) is ast.Attribute and type(target.value) is ast.Name and
                                     target.value.id in ("Chunks","Scalar","capacity_lowering","scalar_rule")
                                     for target in node.targets))


if __name__ == "__main__":
    unittest.main()
