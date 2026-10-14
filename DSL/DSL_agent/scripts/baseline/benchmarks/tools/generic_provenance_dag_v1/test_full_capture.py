"""Guard tests for the inactive complete-provenance analysis tool."""
import copy, hashlib, json, tempfile, unittest
from pathlib import Path
from codec import raw
from full_capture import assemble, analyze, parse

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

class RetainedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "trusted-python"
        source.mkdir()
        script = b"def make():\n    return None\n"
        # Intentional non-NPY data: this tool proves byte retention, not decoding.
        weight = b"bound-public-weight-bytes"
        spec = dict(format="poseidon-trusted-pytorch-v34",
                    files={"model.py":hash_bytes(script)},
                    weights={"w":dict(file="state_0.npy",sha256=hash_bytes(weight),
                                      bytes=len(weight),shape=[1],dtype="<f8")})
        self.spec = spec
        (source / "model.py").write_bytes(script)
        (source / "state_0.npy").write_bytes(weight)
        self.write_spec()
        self.parts = {}
        self.values = {
            "original": {"shape":[1],"value":-0.0},
            "decomposition":{"manifest_sha256":hash_bytes(raw(spec, sorted_keys=True))},
            "reference_report":{"manifest_sha256":hash_bytes(raw(spec, sorted_keys=True))},
        }
        for key, name in (("original","original.json"),
                          ("decomposition","decomposition.json"),
                          ("reference_report","reference-report.json")):
            data = raw(self.values[key]); (self.root / name).write_bytes(data)
            self.parts[key] = hash_bytes(data)

    def write_spec(self):
        (self.root / "trusted-python/manifest.json").write_bytes(raw(self.spec))

    def rebind_manifest(self):
        self.write_spec()
        for key in ("decomposition", "reference_report"):
            self.values[key]["manifest_sha256"] = hash_bytes(raw(self.spec, sorted_keys=True))
            name = "decomposition.json" if key=="decomposition" else "reference-report.json"
            data=raw(self.values[key]);(self.root/name).write_bytes(data)
            self.parts[key]=hash_bytes(data)

    def test_full_content_lossless(self):
        value=assemble(self.root,self.parts);packet,record=analyze(value)
        self.assertTrue(record["exact_roundtrip"])
        self.assertFalse(record["production_active"])
        self.assertFalse(record["trusted_npy_decoding_checked"])
        self.assertEqual(value["trusted_pytorch"]["files"]["model.py"],
                         "def make():\n    return None\n")
        self.assertTrue(value["trusted_pytorch"]["weight_files"])
        self.assertLess(record["wire_bytes"],4*1024**2)

    def test_part_hash_change_rejected(self):
        (self.root/"original.json").write_bytes(b'{"shape":[2]}')
        with self.assertRaisesRegex(ValueError,"retained_file_hash"):
            assemble(self.root,self.parts)

    def test_source_hash_change_rejected(self):
        (self.root/"trusted-python/model.py").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"retained_file_hash"):
            assemble(self.root,self.parts)

    def test_weight_hash_change_rejected(self):
        path=self.root/"trusted-python/state_0.npy"
        path.write_bytes(b"x"*len(path.read_bytes()))
        with self.assertRaisesRegex(ValueError,"retained_file_hash"):
            assemble(self.root,self.parts)

    def test_weight_size_change_rejected(self):
        (self.root/"trusted-python/state_0.npy").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"retained_file_size"):
            assemble(self.root,self.parts)

    def test_source_path_escape_rejected(self):
        self.spec["files"]={"../model.py":"0"*64};self.rebind_manifest()
        with self.assertRaisesRegex(ValueError,"retained_source_filename"):
            assemble(self.root,self.parts)

    def test_weight_path_escape_rejected(self):
        self.spec["weights"]["w"]["file"]="../state_0.npy";self.rebind_manifest()
        with self.assertRaisesRegex(ValueError,"retained_weight_filename"):
            assemble(self.root,self.parts)

    def test_symlink_rejected(self):
        p=self.root/"trusted-python/state_0.npy";p.rename(p.with_name("other.npy"))
        p.symlink_to("other.npy")
        with self.assertRaisesRegex(ValueError,"retained_regular_file"):
            assemble(self.root,self.parts)

    def test_manifest_binding_rejected(self):
        self.spec["extra"]="changed";self.write_spec()
        with self.assertRaisesRegex(ValueError,"retained_manifest_binding"):
            assemble(self.root,self.parts)

    def test_wrong_source_version_rejected(self):
        self.spec["format"]="poseidon-trusted-pytorch-v99";self.rebind_manifest()
        with self.assertRaisesRegex(ValueError,"draft_v34_source_scope"):
            assemble(self.root,self.parts)

    def test_duplicate_json_key_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate_retained_key"):
            parse(b'{"a":1,"a":2}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaisesRegex(ValueError,"nonfinite_retained_json"):
            parse(b'{"a":NaN}')

if __name__ == "__main__":
    unittest.main()
