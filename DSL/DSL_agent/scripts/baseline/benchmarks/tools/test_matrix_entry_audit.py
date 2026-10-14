"""Boundary checks for paired matrix inputs and their logical planning."""
import math
import unittest
from generate_matrix_entry_sources import SHAPES, output_shape
from pytorch_matrix_plan import plan

class MatrixInputContexts(unittest.TestCase):
    def test_all_contexts_fit_small_budget_and_exact_shapes(self):
        self.assertEqual(sum(map(len,SHAPES.values())),24)
        self.assertEqual(len(SHAPES),8)
        for group,shapes in SHAPES.items():
            self.assertEqual(len(shapes),3)
            for first,second in shapes:
                with self.subTest(group=group,first=first,second=second):
                    side="right" if group=="public_right" else "left" if group=="public_left" else None
                    result=plan(first,second,public_side=side,resource_profile="general-tensors-v1")
                    self.assertEqual(result["result_shape"],output_shape(first,second))
                    self.assertLessEqual(math.prod(result["core_shape"]),16)
                    self.assertLessEqual(result["product_tensor_elements"] or 0,32)

    def test_high_ranks_need_explicit_general_profile(self):
        for first,second in SHAPES["higher_rank_cipher"][1:]:
            with self.assertRaisesRegex(ValueError,"shape budget"):
                plan(first,second,resource_profile="legacy-v1")
            self.assertEqual(plan(first,second,resource_profile="general-tensors-v1")["source_shapes"],
                             [first,second])

    def test_vector_results_follow_matmul_promotion(self):
        self.assertEqual(plan([3],[3,2])["result_shape"],[2])
        self.assertEqual(plan([2,3],[3])["result_shape"],[2])
        scalar=plan([3],[3])
        self.assertEqual(scalar["result_shape"],[])
        self.assertEqual(scalar["core_shape"],[1])

    def test_bad_contraction_and_batch_rejected(self):
        with self.assertRaisesRegex(ValueError,"contraction"):
            plan([2,3],[2,1])
        with self.assertRaises(ValueError):
            plan([2,2,3],[3,3,1])

    def test_rank_33_not_admitted(self):
        with self.assertRaisesRegex(ValueError,"shape budget"):
            plan([1]*31+[2,2],[2,2],resource_profile="general-tensors-v1")

    def test_intermediate_budget_enforced(self):
        with self.assertRaisesRegex(ValueError,"product tensor budget"):
            plan([17,256],[256,17],resource_profile="general-tensors-v1")

    def test_invalid_operation_and_public_side_rejected(self):
        for arguments in ({"operation":"einsum"},{"public_side":"automatic"}):
            with self.assertRaisesRegex(ValueError,"operation/public side"):
                plan([2,2],[2,2],**arguments)

    def test_public_side_selects_existing_lowering(self):
        self.assertEqual(plan([2,3],[3,2],public_side="right")["strategy"],"plain_right_linear")
        self.assertEqual(plan([2,3],[3,2],public_side="left")["strategy"],"plain_left_linear")
        self.assertEqual(plan([1,2,3],[2,3,1],public_side="right")["strategy"],"plain_tensor_product")

if __name__=="__main__":
    unittest.main()
