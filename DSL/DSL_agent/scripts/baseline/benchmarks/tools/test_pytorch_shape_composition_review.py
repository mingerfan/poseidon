"""Precise refinement of immutable capture rejections."""
import unittest
from benchmarks.tools.review_pytorch_shape_composition import refine
class ShapeReviewGuards(unittest.TestCase):
    def row(self,reason):return dict(kind="rejection",rejection=reason,style="function",query="literal")
    def log(self,message):return 'File "/app/pytorch_arithmetic_normalization.py", line 75\nValueError: '+message
    def test_actual_rejection_phrases_require_stage_and_fixture(self):
        for reason,message,code in (
            ("inplace","In-place arithmetic is unsupported","INPLACE_REJECTED"),
            ("out_mutation","Arithmetic out mutation is unsupported","OUT_MUTATION_REJECTED")):
            self.assertEqual(refine(self.row(reason),self.log(message))["code"],code)
            self.assertIsNone(refine(self.row(reason),"ValueError: "+message))
            self.assertIsNone(refine(self.row(reason),self.log(message)+"\nRuntimeError: environment"))
            self.assertIsNone(refine(self.row("encrypted_branch"),self.log(message)))
    def test_substring_or_service_errors_are_not_semantic_evidence(self):
        self.assertIsNone(refine(self.row("inplace"),self.log("In-place arithmetic is unsupported trailing")))
        self.assertIsNone(refine(self.row("out_mutation"),self.log("Pinned capture runtime")))
if __name__=="__main__":unittest.main()
