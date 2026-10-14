"""Formal provenance JSON transport controls; no compiler, runtime or Provider.

The small representation tests check transport only. Retained capture tests use
its complete original source verifier; they do not create fresh capture/FHE
results or claim production MHA intake activation.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
MHA_STAGE = HERE.parent/"staged_mha_ordered_graph_v1"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(MHA_STAGE/"overlay"))
import candidate_bundle
from benchmark_graph import canonical, digest
from candidate_contract import request_file_bytes
from candidate_bundle import check_provenance, load_model_provenance

RESULTS = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
CAPTURE = RESULTS/"agent-mha-ordered-capture-v2"
NAMES = ("self_post_linear_residual__declared_shallow_v1",
         "linear_into_separate_cross_none__declared_shallow_v1",
         "boolean_padding_residual__declared_shallow_v1")
SMALL = 131072
LARGE = 4*1024**2


def transport(value, limit=LARGE):
    return candidate_bundle.serialize_model_provenance(value, limit)


def constant_orders(value):
    result = []
    def visit(node, path):
        if type(node) is dict:
            for name, child in node.items():
                location = path+[name]
                if name == "constants" and type(child) is dict:
                    result.append((location, list(child)))
                visit(child, location)
        elif type(node) is list:
            for index, child in enumerate(node):
                visit(child, path+[index])
    visit(value, [])
    return result


class RepresentationControls(unittest.TestCase):
    def test_nested_constant_declaration_order_and_value_are_preserved(self):
        value = dict(original=dict(constants={"weight_2":[2.],"weight_10":[10.],"weight_1":[1.]}),
            reference_report=dict(capture_metadata=dict(source=dict(constants={"p_9":9.,"p_11":11.}))))
        before = copy.deepcopy(value)
        raw = transport(value)
        self.assertIs(type(raw), bytes)
        restored = json.loads(raw)
        self.assertEqual(constant_orders(restored), constant_orders(before))
        self.assertEqual(canonical(restored), canonical(before))
        self.assertEqual(digest(restored), digest(before))
        self.assertEqual(value, before)
        self.assertEqual(constant_orders(value), constant_orders(before))

    def test_input_node_and_output_sequences_and_none_are_retained(self):
        value = dict(inputs=[dict(name="z",shape=[2,1]),dict(name="a",shape=[1])],
            nodes=[dict(id="n10",inputs=["z","a"]),dict(id="n2",inputs=["n10"])],
            output_structure=dict(kind="dict",items=[dict(key="result",value=dict(kind="tensor",name="output0")),
                  dict(key="weights",value=dict(kind="none"))]),
            payload=[None, True, -0., .25, "Chinese text 中文"])
        restored = json.loads(transport(value))
        self.assertEqual(restored, value)
        self.assertEqual(list(restored), list(value))
        self.assertEqual(restored["inputs"][0]["name"], "z")
        self.assertEqual(restored["nodes"][0]["id"], "n10")
        self.assertEqual(restored["output_structure"]["items"][1]["key"], "weights")
        self.assertIsNone(restored["payload"][0])
        self.assertEqual(float(restored["payload"][2]).hex(), float(-0.).hex())

    def test_nonfinite_numbers_are_rejected_at_json_boundary(self):
        for value in (float("nan"),float("inf"),-float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):transport(dict(constants=dict(w=[value])))

    def test_exact_small_encoded_byte_limit_and_one_byte_over(self):
        overhead = len(transport(dict(text="")))
        value = dict(text="a"*(SMALL-overhead))
        self.assertEqual(len(transport(value, SMALL)), SMALL)
        value["text"] += "a"
        with self.assertRaises(ValueError):transport(value, SMALL)
        self.assertEqual(len(transport(value, LARGE)), SMALL+1)

    def test_real_pretty_encoded_size_not_only_canonical_size_is_enforced(self):
        value = dict(constants=dict(w=[0]*32768))
        self.assertLess(len(canonical(value)), SMALL)
        self.assertGreater(len(transport(value, LARGE)), SMALL)
        with self.assertRaises(ValueError):transport(value, SMALL)

    def test_four_mib_encoded_boundary_is_unchanged(self):
        overhead = len(transport(dict(text="")))
        value = dict(text="b"*(LARGE-overhead))
        self.assertEqual(len(transport(value, LARGE)), LARGE)
        value["text"] += "b"
        with self.assertRaises(ValueError):transport(value, LARGE)

    def test_export_calls_full_precheck_and_reloaded_full_bundle_checker(self):
        module = ast.parse(Path(candidate_bundle.__file__).read_text())
        function = next(n for n in module.body if type(n) is ast.FunctionDef and n.name == "export_bundle")
        calls = [n for n in ast.walk(function) if type(n) is ast.Call]
        def positions(name):
            return [n.lineno for n in calls if type(n.func) is ast.Name and n.func.id == name]
        audit, encode, reload = positions("audit_provenance"), positions("serialize_model_provenance"), positions("check_bundle")
        self.assertTrue(audit and encode and reload)
        self.assertLess(min(audit), min(encode))
        self.assertLess(max(encode), min(reload))
        writers = [n.lineno for n in calls if type(n.func) is ast.Attribute and n.func.attr == "write_bytes"
                   and type(n.func.value) is ast.BinOp and type(n.func.value.op) is ast.Div
                   and type(n.func.value.right) is ast.Name and n.func.value.right.id == "PROVENANCE"]
        self.assertTrue(writers)
        self.assertLess(max(encode), min(writers))
        self.assertLess(max(writers), min(reload))

    def test_shared_sorted_json_writer_and_candidate_request_writer_remain_separate(self):
        import strict_json_files
        tree = ast.parse(Path(strict_json_files.__file__).read_text())
        dumps = [n for n in ast.walk(tree) if type(n) is ast.Call and type(n.func) is ast.Attribute and n.func.attr == "dumps"]
        self.assertTrue(any(any(k.arg == "sort_keys" and type(k.value) is ast.Constant and k.value.value is True
                                for k in n.keywords) for n in dumps))
        tree = ast.parse(Path(candidate_bundle.__file__).read_text())
        calls = [n for n in ast.walk(tree) if type(n) is ast.Call and type(n.func) is ast.Name]
        self.assertTrue(any(n.func.id == "request_file_bytes" for n in calls))


class RetainedSourceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {}
        for name in NAMES:
            model = json.loads((CAPTURE/name/"model.json").read_bytes())
            cls.fixtures[name] = (model, load_model_provenance(CAPTURE/name, model, max_bytes=LARGE))

    def fixture(self, name=NAMES[2]):
        return copy.deepcopy(self.fixtures[name])

    def verify_source(self, name):
        model, provenance = self.fixture(name)
        check_provenance(provenance, model)
        raw = transport(provenance)
        reread = json.loads(raw)
        check_provenance(reread, model)
        self.assertEqual(canonical(provenance), canonical(reread))
        self.assertEqual(constant_orders(provenance), constant_orders(reread))
        request = json.loads((RESULTS/"agent-mha-ordered-backend-v1"/name/"request.json").read_bytes())
        before = request_file_bytes(request)
        transport(provenance)
        self.assertEqual(request_file_bytes(request), before)
        self.assertEqual(request["request_id"], json.loads(before)["request_id"])

    def test_full_self_source_validator_before_and_after_transport(self):
        self.verify_source(NAMES[0])

    def test_full_cross_source_validator_before_and_after_transport(self):
        self.verify_source(NAMES[1])

    def test_full_boolean_source_validator_before_and_after_transport(self):
        self.verify_source(NAMES[2])

    def test_same_canonical_value_with_sorted_declarations_remains_rejected(self):
        model, provenance = self.fixture()
        reordered = json.loads(json.dumps(provenance, sort_keys=True))
        self.assertEqual(canonical(provenance), canonical(reordered))
        self.assertNotEqual(constant_orders(provenance), constant_orders(reordered))
        with self.assertRaisesRegex(ValueError, "Ordered source normalization/provenance"):
            check_provenance(json.loads(transport(reordered)), model)

    def test_changed_weight_or_source_provenance_is_not_made_valid_by_serialization(self):
        for field in ("original_weight", "capture_origin"):
            model, provenance = self.fixture()
            if field == "original_weight":
                constants = provenance["original"]["constants"]
                name = next(iter(constants))
                constants[name] = [0.]
            else:
                alias = provenance["reference_report"]["capture_metadata"]["ordered_alias_capture"]
                alias["capture_source_origins"] = {}
            with self.subTest(field=field):
                with self.assertRaises(ValueError):check_provenance(json.loads(transport(provenance)), model)


if __name__ == "__main__":
    unittest.main()
