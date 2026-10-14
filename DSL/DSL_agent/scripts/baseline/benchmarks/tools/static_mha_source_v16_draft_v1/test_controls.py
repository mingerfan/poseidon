"""Standard-library-only admission and output integrity controls."""
import copy
import unittest
from output_contract import SOURCE_FORMAT, OLD_FORMAT, digest, flatten, validate, restore
from source_plan import check_request_header, check_approximation, source_identity

class OutputControls(unittest.TestCase):
    def record(self):
        leaf = object()
        values, record = flatten((leaf, None), lambda x: x is leaf, ["output"],
                                 source_format=SOURCE_FORMAT)
        return leaf, values, record
    def test_none_is_static_and_has_no_tensor_slot(self):
        leaf, values, record = self.record()
        self.assertEqual(values, [leaf])
        self.assertEqual(record["static_none_count"], 1)
        self.assertEqual(restore({"output": 7}, record), (7, None))
    def test_v15_still_rejects_none(self):
        leaf = object()
        with self.assertRaisesRegex(ValueError, "static_none_requires_v16"):
            flatten((leaf, None), lambda x: x is leaf, ["output"], source_format=OLD_FORMAT)
    def test_v15_tensor_tree_is_preserved(self):
        leaf = object()
        _, r = flatten([leaf], lambda x: x is leaf, ["output"], source_format=OLD_FORMAT)
        self.assertEqual(restore({"output": 3}, r), [3])
    def test_resigned_none_under_v15_rejected(self):
        _, _, r = self.record()
        r["source_format"] = OLD_FORMAT
        r["binding"] = digest({k:v for k,v in r.items() if k != "binding"})
        with self.assertRaises(ValueError): validate(r)
    def test_resigned_none_counter_rejected(self):
        _, _, r = self.record()
        r["static_none_count"] = 0
        r["binding"] = digest({k:v for k,v in r.items() if k != "binding"})
        with self.assertRaisesRegex(ValueError, "output_signature"): validate(r)
    def test_named_tensor_order_is_binding(self):
        leaf = object()
        _, r = flatten({"first": leaf, "second": None}, lambda x:x is leaf,
                       ["output"], source_format=SOURCE_FORMAT)
        self.assertEqual(list(restore({"output": 1}, r)), ["first", "second"])
        with self.assertRaises(ValueError): restore({"other": 1}, r)
    def test_no_dynamic_output_objects(self):
        for bad in (1, False, "tensor", lambda: None):
            with self.subTest(type=type(bad).__name__):
                with self.assertRaises(ValueError):
                    flatten((object(), bad), lambda x:type(x) is object, ["out"], source_format=SOURCE_FORMAT)
    def test_depth_and_extra_fields_rejected(self):
        leaf, _, r = self.record()
        r["tree"]["items"][1]["value"] = None
        r["binding"] = digest({k:v for k,v in r.items() if k != "binding"})
        with self.assertRaises(ValueError): validate(r)
        value = leaf
        for _ in range(10): value = [value]
        with self.assertRaises(ValueError):
            flatten(value, lambda x:x is leaf, ["output"], source_format=SOURCE_FORMAT)
    def test_closed_request_header(self):
        with self.assertRaisesRegex(ValueError, "source_request_fields"): check_request_header({})
    def test_missing_approximation_is_not_defaulted(self):
        for value in ({}, {"exp":{}}, {"exp":{}, "reciprocal":{}, "extra":1}):
            with self.assertRaises(ValueError): check_approximation(value)
    def test_source_identity_is_explicit_closed_files(self):
        files = source_identity()
        self.assertGreaterEqual(len(files), 10)
        self.assertTrue(all(len(v) == 64 for v in files.values()))
        self.assertTrue(any(k.endswith("mha_source_reference.py") for k in files))
    def test_output_tamper_requires_original_binding(self):
        _, _, r = self.record()
        r["tree"]["items"].reverse()
        with self.assertRaisesRegex(ValueError, "output_binding"): validate(r)

if __name__ == "__main__":
    unittest.main()
