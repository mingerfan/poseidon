"""Independent slot arithmetic, immutable layout and explicit baseline selection."""
import ast, copy, hashlib, json, math, unittest
from pathlib import Path
from unittest import mock
import numpy as np
import unified_mixed_layout as rule
from benchmark_math import evaluate
from benchmark_graph import samples, canonical
from unified_graph_contract import validate_request, validate_candidate
from unified_request_storage import restore_public_constants
from component_contract import prepare_task
from unified_capacity_constants import INDEX_VERSION
from unified_graph_lowering import candidate_source
from unified_layout_strategies import MIXED, describe

def prepared(model):
    return prepare_task(model, dict(chunk_period=4, capacity_packing_version=INDEX_VERSION,
                                    public_weight_policy="descriptors-math-roles-v2"))


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

class MixedLayoutComponentTests(unittest.TestCase):
    def compare(self, request, tail=0.):
        before=canonical(request)
        source,record=candidate_source(request,strategy=MIXED)
        self.assertEqual(before,canonical(request))
        self.assertFalse(record["provider_candidate_rewritten"])
        self.assertFalse(record["depth_is_capacity_proof"])
        self.assertFalse(record["peak_allocation_is_known"])
        for group in range(4):
            probe=sample(request["model"],group)
            expected=evaluate(request["model"],probe)
            parts=periodic_source(source,request,probe,tail=tail)
            actual=np.asarray([parts[c][slot] for c,slot in request["layout"]["output_selectors"]])
            target=np.concatenate([expected[o["name"]].reshape(-1) for o in request["model"]["outputs"]])
            np.testing.assert_allclose(actual,target,atol=1e-12,rtol=1e-12)
        return source,record
    def test_routed_broadcast_split_multioutput(self):
        self.compare(generic_request())
    def test_late_linear_has_single_pack_boundary(self):
        _,r=self.compare(generic_request(late_linear=True))
        self.assertTrue(any(p["boundary"]["kind"]=="linear_input" for p in r["packs"]))
    def test_canonical_views_are_cipher_free(self):
        _,r=self.compare(view_request())
        self.assertEqual(r["materialized_cells"],[])
        self.assertEqual(r["packs"],[])
    def test_poisoned_tail_is_not_logical_input(self):
        self.compare(generic_request(late_linear=True),tail=91.)
    def test_model_and_node_names_do_not_select_strategy(self):
        q=generic_request();m=copy.deepcopy(q["model"]);m["id"]="renamed_generic"
        for i,n in enumerate(m["nodes"]):n["id"]="other_"+str(i)
        self.compare(prepared(m))
    def test_periods_and_shape_partitions(self):
        for p in (4,8,16):
            q=prepare_task(generic_request()["model"],dict(chunk_period=p,capacity_packing_version=INDEX_VERSION,
                             public_weight_policy="descriptors-math-roles-v2"))
            self.compare(q,tail=37.)
    def test_candidate_source_preserves_pair_interface(self):
        q=generic_request()
        source,record=candidate_source(q,strategy=MIXED)
        self.assertIsInstance(source,str);self.assertIsInstance(record,dict)
    def test_unknown_strategy_and_provider_source_rejected(self):
        from unified_layout_strategies import lower
        for mode in ("auto","provider",True,1,{}):
            with self.assertRaises(ValueError):lower(generic_request(),mode)
        with self.assertRaises(ValueError):lower(generic_request(),MIXED,source_kind="provider")
    def test_default_strategy_remains_the_existing_capacity_lowerer(self):
        q=generic_request()
        import unified_capacity_lowering
        with mock.patch.object(unified_capacity_lowering,"lower",return_value=("legacy",{})) as selected:
            self.assertEqual(candidate_source(q),("legacy",{}))
            selected.assert_called_once()
    def test_public_binding_tamper_fails_closed(self):
        q=generic_request();key=next(k for k in q["public_constants"] if k.startswith("pv"))
        q["public_constants"][key][0]+=.125
        with self.assertRaises(ValueError):candidate_source(q,strategy=MIXED)
    def test_foreign_cell_and_canonical_storage_are_rejected(self):
        q=generic_request();checked=rule.validate(q["model"])
        a=rule.MixedEmitter(q,checked);b=rule.MixedEmitter(q,checked)
        ma,mb=rule.Materializer(a),rule.Materializer(b)
        from unified_graph_lowering import Scalar
        x=mb.packed_value([Scalar(b,q["layout"]["inputs"][0]["dsl_name"])],[4],producer={"test":True})
        with self.assertRaises(ValueError):ma.cell(x.entries[0])
        with self.assertRaises(ValueError):ma.chunks(x,boundary={"test":True})
    def test_registry_is_detached_and_not_an_execution_score(self):
        entry=describe()[MIXED]
        self.assertEqual(entry["source_kind"],"deterministic_rule_baseline")
        self.assertEqual(entry["encrypted_status"],"unverified")
        entry["operators"].append("unregistered")
        self.assertNotIn("unregistered",describe()[MIXED]["operators"])
    def test_runtime_component_has_no_experiment_import(self):
        import replicated_cell_emitter
        for module in (rule,replicated_cell_emitter):
            source=Path(module.__file__).read_text();tree=ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or "").startswith("benchmarks."))
                if isinstance(node,ast.Import):
                    self.assertFalse(any(a.name.startswith("benchmarks.") for a in node.names))
            self.assertNotIn("agent-mha-",source)
