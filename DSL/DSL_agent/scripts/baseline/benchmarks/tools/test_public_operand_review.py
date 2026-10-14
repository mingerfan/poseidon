"""Guard exact capture-cause classification and immutable audit bindings."""
import copy,unittest
from benchmarks.tools.review_public_operand_capture import classify_rejection,checked_binding
from benchmark_graph import digest

class ReviewGuards(unittest.TestCase):
    def row(self,op="add",kind="tensor",side="left"):
        return dict(operator=op,public_kind=kind,public_side=side)
    def test_exact_tensor_methods_and_scalar_subtract(self):
        for op,target in (("add","add"),("subtract","sub"),("multiply","mul")):
            self.assertEqual(classify_rejection(self.row(op),"Traceback\nValueError: Unsupported PyTorch call: "+target)["code"],
                             "UNMAPPED_TENSOR_ARITHMETIC_METHOD")
        self.assertEqual(classify_rejection(self.row("subtract","scalar"),"ValueError: First operand must be encrypted")["code"],
                         "PUBLIC_FIRST_OPERAND")
    def test_wrapped_fixture_node_error_has_exact_operator_context(self):
        row=self.row("subtract","scalar")
        self.assertEqual(classify_rejection(row,
            "ValueError: node fx_sub (subtract): First operand must be encrypted")["code"],"PUBLIC_FIRST_OPERAND")
        self.assertIsNone(classify_rejection(row,
            "ValueError: node fx_sub (multiply): First operand must be encrypted"))
    def test_environment_fault_is_not_operand_rejection(self):
        self.assertIsNone(classify_rejection(self.row(),"RuntimeError: Pinned capture runtime"))
        self.assertIsNone(classify_rejection(self.row(),"ValueError: Unsupported PyTorch call: div"))
    def test_other_position_or_operand_role_is_not_reclassified(self):
        self.assertIsNone(classify_rejection(self.row(side="right"),"ValueError: Unsupported PyTorch call: add"))
        self.assertIsNone(classify_rejection(self.row(kind="scalar"),"ValueError: Unsupported PyTorch call: add"))
    def test_suffix_or_nonterminal_error_cannot_match(self):
        for log in ("ValueError: Unsupported PyTorch call: add malicious",
                    "ValueError: Unsupported PyTorch call: add\nRuntimeError: different failure"):
            self.assertIsNone(classify_rejection(self.row(),log))
    def test_changed_audit_body_rejected(self):
        record=dict(planned=12,passed=8);record["binding"]=digest(record)
        checked_binding(record)
        changed=copy.deepcopy(record);changed["passed"]=12
        with self.assertRaisesRegex(ValueError,"Audit report binding"):checked_binding(changed)
