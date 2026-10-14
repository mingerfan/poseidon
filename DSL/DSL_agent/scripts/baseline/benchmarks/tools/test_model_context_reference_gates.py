"""Pure integrity gates; actual NumPy/Torch execution is a separate acceptance."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location("verify_contexts",HERE/"verify_model_semantic_contexts.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SUITE=module.BASE/"benchmarks/semantic-v1-chunk-helpers-r29"
BUNDLE=module.BASE/"benchmarks/model-contexts-draft-v1/tasks.json"

class ContextReferenceGateTests(unittest.TestCase):
    def changed(self,edit,rehash=True):
        b=json.loads(BUNDLE.read_text());edit(b)
        if rehash:b["bundle_sha256"]=module.digest({k:v for k,v in b.items() if k!="bundle_sha256"})
        return b
    def check_bad(self,b,message):
        with tempfile.TemporaryDirectory(prefix="poseidon-context-gate-") as d:
            p=Path(d)/"bundle.json";p.write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,message):module.checked_bundle(p,SUITE)
    def test_valid_bundle_rebuilt_exactly(self):
        b,parents=module.checked_bundle(BUNDLE,SUITE)
        self.assertEqual(len(b["supplemental_models"]),92)
        self.assertEqual(len(b["tasks"]),459)
        self.assertEqual(len(parents),1200)
    def test_unbound_modification_rejected(self):
        b=self.changed(lambda b:b["tasks"][0].update(state="passed"),rehash=False)
        self.check_bad(b,"bundle hash")
    def test_rehashed_false_task_rejected(self):
        b=self.changed(lambda b:b["tasks"][0].update(state="passed"))
        self.check_bad(b,"bound generator")
    def test_arbitrary_source_path_rejected_before_read(self):
        b=self.changed(lambda b:b["source_binding"]["generator_sources"].update({".env":"0"*64}))
        self.check_bad(b,"generator binding mismatch")
    def test_parent_provenance_forgery_rejected(self):
        b=self.changed(lambda b:b["supplemental_models"][0]["provenance"].update(recipe="unknown_recipe"))
        self.check_bad(b,"bound generator")
    def test_symlink_and_size_gates(self):
        with tempfile.TemporaryDirectory(prefix="poseidon-context-gate-") as d:
            p=Path(d)/"link";p.symlink_to(BUNDLE)
            with self.assertRaisesRegex(ValueError,"file gate"):module.checked_bundle(p,SUITE)
            p=Path(d)/"large";p.write_bytes(b" "*(4*1024**2+1))
            with self.assertRaisesRegex(ValueError,"file gate"):module.checked_bundle(p,SUITE)
    def test_composition_uses_declared_transform(self):
        original={"x":2.0,"y":-1.0}
        self.assertEqual(module.transformed_inputs(original,"input_affine"),{"x":-.875,"y":.625})
        self.assertEqual(module.transformed_inputs(original,"input_negate"),{"x":-2.0,"y":1.0})
        self.assertEqual(module.transformed_inputs(original,"output_residual"),original)
        self.assertEqual(module.transformed_first(2.0,"output_residual"),6.0)
        self.assertEqual(module.transformed_first(2.0,"output_negate"),-2.0)
        self.assertEqual(module.transformed_first(2.0,"input_affine"),2.0)
        self.assertEqual(original,{"x":2.0,"y":-1.0})
if __name__=="__main__":unittest.main()
