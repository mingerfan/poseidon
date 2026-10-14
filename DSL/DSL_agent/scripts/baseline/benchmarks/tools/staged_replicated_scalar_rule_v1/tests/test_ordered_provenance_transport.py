"""Complete provenance/package controls without compiler, execution or API calls."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parents[2]
MHA_STAGE = HERE.parent/"staged_mha_ordered_graph_v1"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(MHA_STAGE/"overlay"))
spec = importlib.util.spec_from_file_location("ordered_transport_tested", HERE/"ordered_provenance_transport.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
from benchmark_graph import canonical, digest
from candidate_bundle import check_bundle, check_provenance, load_model_provenance

RESULTS = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
CAPTURE = RESULTS/"agent-mha-ordered-capture-v2"
NAMES = ("self_post_linear_residual__declared_shallow_v1",
         "linear_into_separate_cross_none__declared_shallow_v1",
         "boolean_padding_residual__declared_shallow_v1")
SOURCE = RESULTS/"agent-mha-replicated-scalar-backend-v1"/NAMES[2]/"program"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_scalar(value):
    if type(value) is list:
        value[0] = changed_scalar(value[0])
        return value
    return float(value)+.000123


def update_manifest(folder, change):
    path = folder/"manifest.json"
    value = json.loads(path.read_bytes())
    change(value)
    value.pop("binding")
    value["binding"] = digest(value)
    path.write_bytes(canonical(value))


class OrderedProvenanceTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {}
        for name in NAMES:
            model = json.loads((CAPTURE/name/"model.json").read_bytes())
            cls.fixtures[name] = (model, load_model_provenance(CAPTURE/name, model, max_bytes=4*1024**2))

    def fixture(self, name=NAMES[2]):
        return copy.deepcopy(self.fixtures[name])

    def copy_source(self, temporary):
        source = Path(temporary)/"source"
        shutil.copytree(SOURCE, source)
        return source

    def verify_roundtrip(self, name):
        model, provenance = self.fixture(name)
        before = copy.deepcopy(provenance)
        blob, record = transport.serialize(provenance, model)
        actual = json.loads(blob)
        self.assertEqual(canonical(actual), canonical(before))
        self.assertEqual(transport.declaration_orders(actual), transport.declaration_orders(before))
        self.assertEqual(provenance, before)
        self.assertEqual(record["canonical_sha256"], digest(before))
        self.assertEqual(record["serialized_sha256"], hashlib.sha256(blob).hexdigest())
        self.assertEqual(record["declaration_order_sha256"], digest(transport.declaration_orders(before)))
        self.assertEqual(record["implementation"]["sha256"], sha(HERE/"ordered_provenance_transport.py"))
        self.assertEqual(record["binding"], digest({k:v for k,v in record.items() if k != "binding"}))
        self.assertEqual(record["full_provenance_checks"], 2)
        self.assertFalse(record["production_export_changed"])

    def test_serialize_original_self_full_source_roundtrip(self):
        self.verify_roundtrip(NAMES[0])

    def test_serialize_original_cross_full_source_roundtrip(self):
        self.verify_roundtrip(NAMES[1])

    def test_serialize_original_boolean_full_source_roundtrip(self):
        self.verify_roundtrip(NAMES[2])

    def test_actual_sorted_serialization_reproduces_normalization_failure(self):
        model, provenance = self.fixture()
        reordered = json.loads(json.dumps(provenance, sort_keys=True))
        self.assertEqual(canonical(provenance), canonical(reordered))
        self.assertNotEqual(transport.declaration_orders(provenance), transport.declaration_orders(reordered))
        with self.assertRaisesRegex(ValueError, "Ordered source normalization/provenance"):
            check_provenance(reordered, model)
        with self.assertRaisesRegex(ValueError, "Ordered source normalization/provenance"):
            transport.serialize(reordered, model)

    def test_altered_constant_rejected_by_original_full_checker(self):
        model, provenance = self.fixture()
        constants = provenance["original"]["constants"]
        name = next(iter(constants))
        constants[name] = changed_scalar(constants[name])
        with self.assertRaises(ValueError):
            transport.serialize(provenance, model)

    def test_altered_source_provenance_rejected(self):
        model, provenance = self.fixture()
        alias = provenance["reference_report"]["capture_metadata"]["ordered_alias_capture"]
        alias["capture_source_origins"] = {}
        with self.assertRaises(ValueError):
            transport.serialize(provenance, model)

    def test_repack_boolean_retains_every_other_byte_and_old_failed_package(self):
        _, provenance = self.fixture()
        manifest = json.loads((SOURCE/"manifest.json").read_bytes())
        before = {name:sha(SOURCE/name) for name in ["manifest.json", *manifest["files"]]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"repaired"
            repaired, record = transport.repack(SOURCE, output, provenance)
            self.assertEqual(check_bundle(output), repaired)
            self.assertFalse(repaired["agent_generated"])
            self.assertEqual(record["old_package_binding"], manifest["binding"])
            self.assertNotEqual(record["old_package_binding"], record["new_package_binding"])
            self.assertEqual(record["canonical_sha256"], digest(provenance))
            self.assertNotEqual(record["old_provenance_sha256"], record["new_provenance_sha256"])
            self.assertEqual(set(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()),
                             set(manifest["files"])|{"manifest.json"})
            for name, expected in manifest["files"].items():
                if name != transport.PROVENANCE:
                    self.assertEqual(sha(output/name), expected)
            self.assertEqual(record["unchanged_file_sha256"],
                             {name:expected for name,expected in manifest["files"].items() if name != transport.PROVENANCE})
            self.assertEqual(record["binding"], digest({k:v for k,v in record.items() if k != "binding"}))
        self.assertEqual(before, {name:sha(SOURCE/name) for name in before})

    def test_existing_destination_preserved(self):
        _, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"existing"; output.mkdir()
            sentinel = output/"preserved"; sentinel.write_bytes(b"untouched")
            with self.assertRaisesRegex(ValueError, "Preserve destination"):
                transport.repack(SOURCE, output, provenance)
            self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_source_file_hash_tampering_rejected_before_destination_creation(self):
        _, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            source = self.copy_source(temporary)
            (source/"candidate.py").write_bytes((source/"candidate.py").read_bytes()+b"\n")
            output = Path(temporary)/"repaired"
            with self.assertRaisesRegex(ValueError, "source file integrity"):
                transport.repack(source, output, provenance)
            self.assertFalse(output.exists())

    def test_changed_stored_canonical_value_rejected_even_with_valid_byte_binding(self):
        _, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            source = self.copy_source(temporary)
            path = source/transport.PROVENANCE
            stored = json.loads(path.read_bytes())
            name = next(iter(stored["original"]["constants"]))
            stored["original"]["constants"][name] = changed_scalar(stored["original"]["constants"][name])
            path.write_bytes(canonical(stored))
            update_manifest(source, lambda value:value["files"].update({transport.PROVENANCE:sha(path)}))
            output = Path(temporary)/"repaired"
            with self.assertRaisesRegex(ValueError, "original provenance value mismatch"):
                transport.repack(source, output, provenance)
            self.assertFalse(output.exists())

    def test_source_symlink_rejected(self):
        _, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            source = self.copy_source(temporary)
            candidate = source/"candidate.py"; candidate.unlink(); candidate.symlink_to(SOURCE/"candidate.py")
            with self.assertRaisesRegex(ValueError, "artifact escape|artifact symlink"):
                transport.repack(source, Path(temporary)/"repaired", provenance)

    def test_source_manifest_binding_tampering_rejected(self):
        _, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            source = self.copy_source(temporary)
            manifest = json.loads((source/"manifest.json").read_bytes())
            manifest["origin"]["request_id"] = "f"*64
            (source/"manifest.json").write_bytes(canonical(manifest))
            with self.assertRaisesRegex(ValueError, "source manifest binding"):
                transport.repack(source, Path(temporary)/"repaired", provenance)

    def test_path_traversal_rejected_despite_rebound_manifest(self):
        _, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            source = self.copy_source(temporary)
            update_manifest(source, lambda value:value["files"].update({"artifacts/../outside": "0"*64}))
            with self.assertRaisesRegex(ValueError, "source artifact path"):
                transport.repack(source, Path(temporary)/"repaired", provenance)

    def test_agent_generated_package_cannot_be_relabelled_as_rule_transport(self):
        _, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            source = self.copy_source(temporary)
            update_manifest(source, lambda value:value.update(agent_generated=True))
            with self.assertRaisesRegex(ValueError, "rule-only"):
                transport.repack(source, Path(temporary)/"repaired", provenance)

    def test_duplicate_keys_and_nonfinite_transport_rejected(self):
        for blob in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                transport._parse(blob, 100)

    def test_repair_contains_no_dynamic_candidate_execution_or_checker_patch(self):
        tree = ast.parse((HERE/"ordered_provenance_transport.py").read_text())
        calls = [node.func.id for node in ast.walk(tree)
                 if type(node) is ast.Call and type(node.func) is ast.Name]
        self.assertFalse(set(calls)&{"eval", "exec", "setattr", "getattr"})
        for node in ast.walk(tree):
            if type(node) is ast.Assign:
                self.assertFalse(any(type(target) is ast.Attribute and
                    type(target.value) is ast.Name and target.value.id in
                    ("candidate_bundle", "pytorch_ordered_contract", "pytorch_alias_composition")
                    for target in node.targets))


if __name__ == "__main__":
    unittest.main()
