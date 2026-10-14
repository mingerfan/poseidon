import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from audit_upstream_helpers import verify_manifest
from upstream_adapters.scenarios import supplemental_rows, BN_SHAPES
from upstream_adapters.batch_norm import bind

class EvidenceTests(unittest.TestCase):
    def test_supplemental_scope_and_binding(self):
        rows=supplemental_rows()
        self.assertEqual(len({r["model_sha256"] for r in rows}),8)
        for r,shape in zip(rows,BN_SHAPES):
            self.assertEqual(bind(r["model"])["input_shape"],list(shape))
            self.assertEqual(r["metadata"]["scope"],"supplemental-asymmetric-v1")
    def test_manifest_tampering_and_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/"report.json";report.write_text("{}")
            record=dict(schema=1,private_keys_retained=False,files={"report.json":hashlib.sha256(b"{}").hexdigest()})
            manifest=root/"evidence-manifest.json"
            manifest.write_text(json.dumps(record));verify_manifest(root)
            report.write_text("{ }")
            with self.assertRaisesRegex(ValueError,"Changed"):verify_manifest(root)
            report.write_text("{}")
            for name in ("../bad","/bad","private-keys/bad"):
                changed=dict(record,files=dict(record["files"],**{name:"0"*64}))
                manifest.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,"Unsafe"):verify_manifest(root)
            manifest.write_text(json.dumps(record))
            (root/"private-keys").mkdir()
            with self.assertRaisesRegex(ValueError,"cleaned"):verify_manifest(root)

if __name__=="__main__":unittest.main()
