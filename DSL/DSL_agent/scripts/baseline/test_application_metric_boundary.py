"""Accurate application classification boundaries without changing CKKS tolerances."""
import unittest
import numpy as np
from task_numerical_assessment import application_metric,NATIVE_POLICY,ATOL,RTOL
class CountBoundaryTests(unittest.TestCase):
    def metric(self,original_correct,encrypted_correct,limit=.01):
        def logits(correct):
            out=np.zeros((100,2),dtype=np.float64)
            out[:correct,0]=1.;out[correct:,1]=1.;return [dict(out=out)]
        return application_metric(dict(kind="top1_accuracy_drop",output="out",class_axis=1,
            labels=[[0]*100],max_drop=limit),logits(original_correct),logits(encrypted_correct),
            logits(encrypted_correct),{})
    def test_exact_one_percentage_point_is_accepted_in_three_contexts(self):
        for original,actual in ((96,95),(95,94),(100,99)):
            with self.subTest(original=original):
                result=self.metric(original,actual)
                self.assertTrue(result["passed"]);self.assertEqual(result["accuracy_drop"],.01)
    def test_over_limit_rejected_and_improvement_accepted(self):
        self.assertFalse(self.metric(96,94)["passed"])
        self.assertFalse(self.metric(96,95,.009999)["passed"])
        self.assertTrue(self.metric(90,95,0.)["passed"])
    def test_native_and_mathematical_metric_keep_the_same_decision(self):
        original=[dict(out=np.array([[1.,0.]]))]
        profile=dict(kind="top1_accuracy_drop",output="out",class_axis=1,labels=[[0]],max_drop=0.)
        native=application_metric(profile,original,original,original,{},scope=NATIVE_POLICY)
        self.assertTrue(native["passed"]);self.assertEqual(native["original_source_top1"],1.)
    def test_return_fields_and_ckks_threshold_stay_compatible(self):
        result=self.metric(96,95)
        self.assertEqual(set(result),{"kind","status","passed","output","class_axis","original_mathematical_top1",
            "approximate_mathematical_top1","encrypted_top1","accuracy_drop","max_drop","labels_compared","tie_policy","scope"})
        self.assertEqual((ATOL,RTOL),(1e-5,1e-4))
if __name__=="__main__":unittest.main(verbosity=2)
