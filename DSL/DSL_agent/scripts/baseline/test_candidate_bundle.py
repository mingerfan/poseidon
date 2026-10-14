import copy,json,tempfile,unittest
from pathlib import Path
from candidate_bundle import check_bundle,sha
from benchmark_graph import digest
from compiler_configuration import configuration
class BundleTests(unittest.TestCase):
    def bundle(self,p):
        (p/"model.json").write_text("{}")
        (p/"candidate.py").write_text("def golden(x): return x\n")
        m=dict(format="poseidon-dsl-bundle-v1",files={n:sha(p/n) for n in ("model.json","candidate.py")},
               compiler_configuration=configuration("seal-cpu-eva-w45-v1"),origin={},
               agent_generated=False)
        m["binding"]=digest(m);(p/"manifest.json").write_text(json.dumps(m));return m
    def test_valid_manifest_is_integrity_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p)
            self.assertFalse(check_bundle(p)["agent_generated"])
    def test_tampered_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.bundle(p);(p/"candidate.py").write_text("changed")
            with self.assertRaisesRegex(ValueError,"integrity"):check_bundle(p)
    def test_extra_executable_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p);m["files"]["../outside.py"]="x"
            m["binding"]=digest({k:v for k,v in m.items() if k!="binding"});(p/"manifest.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"Unexpected"):check_bundle(p)
    def test_changed_profile_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=self.bundle(p);m["compiler_configuration"]["waterline"]=5
            m["binding"]=digest({k:v for k,v in m.items() if k!="binding"});(p/"manifest.json").write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,"profile"):check_bundle(p)


class ProvenanceSerializationTests(unittest.TestCase):
    """JSON transport controls, separate from source and numerical acceptance."""
    SMALL = 131072
    LARGE = 4*1024**2

    def encode(self, value, limit=None):
        from candidate_bundle import serialize_model_provenance
        return serialize_model_provenance(value, self.LARGE if limit is None else limit)

    def test_nested_declaration_order_and_structural_sequences_are_preserved(self):
        value = dict(original=dict(constants={"weight_2":[2.],"weight_10":[10.],"weight_1":[1.]},
                                   inputs=[dict(name="z",shape=[2]),dict(name="a",shape=[1])],
                                   nodes=[dict(id="n10"),dict(id="n2")]),
                     reference_report=dict(capture_metadata=dict(source=dict(constants={"p_9":9.,"p_11":11.}),
                       output_structure=dict(kind="dict",items=[dict(key="z",value=None),dict(key="a",value=[-0.])]))))
        before = copy.deepcopy(value)
        raw = self.encode(value)
        self.assertIs(type(raw), bytes)
        restored = json.loads(raw)
        self.assertEqual(list(restored), list(before))
        self.assertEqual(list(restored["original"]["constants"]), ["weight_2","weight_10","weight_1"])
        self.assertEqual(list(restored["reference_report"]["capture_metadata"]["source"]["constants"]),
                         ["p_9","p_11"])
        self.assertEqual(restored, before)
        self.assertEqual(value, before)
        self.assertEqual(list(value["original"]["constants"]), ["weight_2","weight_10","weight_1"])
        self.assertEqual(float(restored["reference_report"]["capture_metadata"]["output_structure"]
                               ["items"][1]["value"][0]).hex(), float(-0.).hex())

    def test_canonical_identity_and_legacy_request_hash_are_unchanged(self):
        from benchmark_graph import canonical
        from benchmark_suite import Builder
        from component_contract import prepare_task, reconstruct_request
        from candidate_contract import request_file_bytes
        builder = Builder([(2,)])
        model = builder.finish(builder.node("square", ["input0"]))
        # This is the existing legacy contract fixture; no compiler is invoked.
        request = prepare_task(model, dict(compiler_configuration="seal-cpu-eva-w45-v1"))
        request_id = request["request_id"]
        original_request = canonical(request)
        wire = request_file_bytes(request)
        value = dict(original=request["model"], reference_report=dict(constants={"w_2":2.,"w_10":10.}))
        original = canonical(value)
        restored = json.loads(self.encode(value))
        self.assertEqual(canonical(restored), original)
        self.assertEqual(digest(restored), digest(value))
        self.assertEqual(canonical(request), original_request)
        self.assertEqual(request_file_bytes(request), wire)
        self.assertEqual(reconstruct_request(request)["request_id"], request_id)

    def test_pretty_encoded_size_is_checked_in_addition_to_canonical_size(self):
        from benchmark_graph import canonical
        value = dict(constants=dict(weight=[0]*32768))
        self.assertLess(len(canonical(value)), self.SMALL)
        self.assertGreater(len(self.encode(value, self.LARGE)), self.SMALL)
        with self.assertRaisesRegex(ValueError, "Serialized model provenance size"):
            self.encode(value, self.SMALL)

    def test_exact_128k_encoded_boundary_and_one_byte_over(self):
        overhead = len(self.encode(dict(text="")))
        value = dict(text="a"*(self.SMALL-overhead))
        self.assertEqual(len(self.encode(value, self.SMALL)), self.SMALL)
        value["text"] += "a"
        with self.assertRaises(ValueError): self.encode(value, self.SMALL)
        self.assertEqual(len(self.encode(value, self.LARGE)), self.SMALL+1)

    def test_existing_four_mib_encoded_boundary_is_kept(self):
        overhead = len(self.encode(dict(text="")))
        value = dict(text="b"*(self.LARGE-overhead))
        self.assertEqual(len(self.encode(value, self.LARGE)), self.LARGE)
        value["text"] += "b"
        with self.assertRaises(ValueError): self.encode(value, self.LARGE)

    def test_invalid_value_and_limit_types_are_rejected(self):
        for value in (None, [], "provenance", 1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): self.encode(value)
        for limit in (True, None, "131072", 131072., 0, -1):
            with self.subTest(limit=limit):
                from candidate_bundle import serialize_model_provenance
                with self.assertRaises(ValueError): serialize_model_provenance({}, limit)

    def test_nan_and_infinity_are_rejected(self):
        for value in (float("nan"),float("inf"),-float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError): self.encode(dict(constants=dict(weight=[value])))

    def test_ordinary_shared_json_dump_still_sorts_mapping_keys(self):
        from strict_json_files import dump
        value = dict(constants={"weight_2":2.,"weight_10":10.,"weight_1":1.})
        self.encode(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ordinary.json"
            dump(path, value)
            self.assertEqual(path.read_bytes(),
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()+b"\n")
            self.assertEqual(list(json.loads(path.read_bytes())["constants"]),
                             ["weight_1","weight_10","weight_2"])
            self.assertFalse(path.with_name(path.name+".partial").exists())
        self.assertEqual(list(value["constants"]), ["weight_2","weight_10","weight_1"])

if __name__=="__main__":unittest.main()

class CompactProvenanceSerializationTests(unittest.TestCase):
    def test_selected_compact_roundtrip_preserves_order_types_identity_and_immutability(self):
        from candidate_bundle import serialize_model_provenance
        from benchmark_graph import canonical,digest
        value=dict(constants={"weight_2":[-0.,1.,True,None],"weight_10":[0]*32768},nodes=[{"id":"n2"},{"id":"n1"}])
        before=copy.deepcopy(value);limit=131072
        with self.assertRaises(ValueError):serialize_model_provenance(value,limit)
        raw=serialize_model_provenance(value,limit,compact_fallback=True)
        self.assertLessEqual(len(raw),limit)
        restored=json.loads(raw)
        self.assertEqual(list(restored["constants"]),["weight_2","weight_10"])
        self.assertEqual(canonical(restored),canonical(before))
        self.assertEqual(digest(restored),digest(before))
        self.assertEqual(value,before)
        self.assertEqual(restored["constants"]["weight_2"][0].hex(),(-0.).hex())
        self.assertIs(type(restored["constants"]["weight_2"][1]),float)
        self.assertIs(type(restored["constants"]["weight_2"][2]),bool)

    def test_ordinary_encoding_remains_byte_identical(self):
        from candidate_bundle import serialize_model_provenance
        value=dict(constants={"w2":2.,"w10":10.},nodes=[{"id":"z"}])
        legacy=serialize_model_provenance(value,131072)
        self.assertEqual(serialize_model_provenance(value,131072,compact_fallback=True),legacy)

    def test_compact_exact_boundary_and_actual_one_byte_over(self):
        from candidate_bundle import serialize_model_provenance
        from benchmark_graph import canonical
        value={"text":"x"*100};limit=len(canonical(value))
        raw=serialize_model_provenance(value,limit,compact_fallback=True)
        self.assertEqual(len(raw),limit)
        self.assertEqual(json.loads(raw),value)
        with self.assertRaisesRegex(ValueError,"Model provenance size budget"):
            serialize_model_provenance(value,limit-1,compact_fallback=True)

    def test_compact_selection_and_nonfinite_still_reject(self):
        from candidate_bundle import serialize_model_provenance
        for selection in (1,0,None,"compact"):
            with self.assertRaises(ValueError):serialize_model_provenance({},128,compact_fallback=selection)
        for x in (float("nan"),float("inf"),-float("inf")):
            with self.assertRaises(ValueError):serialize_model_provenance({"x":x},128,compact_fallback=True)
