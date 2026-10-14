"""Distinct application/CKKS outcomes, exact limits, ties, shape/type and binding."""
import copy,unittest
from application_metrics import calculation,check_calculation,digest
class AccuracyTests(unittest.TestCase):
    def test_zero_loss_classifier(self):
        rows=[[3.,1.],[1.,3.],[4.,0.]]
        result=calculation(rows,rows,rows,dict(kind="top1_drop",labels=[0,1,0],maximum_drop_percentage_points=1.))
        self.assertTrue(result["application_metric"]["decrypted_metric_passed"])
        self.assertFalse(result["application_accuracy_certified"])
    def test_ckks_pass_does_not_hide_approximation_accuracy_loss(self):
        original=[[2.,0.],[0.,2.]]
        approx=[[0.,2.],[0.,2.]]
        r=calculation(original,approx,approx,dict(kind="top1_drop",labels=[0,1],maximum_drop_percentage_points=1.))
        self.assertTrue(r["execution_error"]["elementwise_threshold_passed"])
        self.assertEqual(r["application_metric"]["approximation_drop_percentage_points"],50.)
        self.assertFalse(r["application_metric"]["approximation_metric_passed"])
    def test_near_tie_label_change_separate_from_ckks_threshold(self):
        original=[[.10000001,.1]]
        decrypted=[[.1,.10000001]]
        r=calculation(original,original,decrypted,dict(kind="top1_drop",labels=[0],maximum_drop_percentage_points=1.))
        self.assertTrue(r["execution_error"]["elementwise_threshold_passed"])
        self.assertFalse(r["application_metric"]["decrypted_metric_passed"])
    def test_first_index_tie_matches_standard_argmax(self):
        r=calculation([[1.,1.]],[[1.,1.]],[[1.,1.]],dict(kind="top1_drop",labels=[0],maximum_drop_percentage_points=0.))
        self.assertEqual(r["application_metric"]["original"]["correct"],1)
    def test_exact_percentage_point_boundary(self):
        original=[[1.,0.]]*16
        approximate=[[0.,1.]]+[[1.,0.]]*15
        for limit,passed in ((6.25,True),(6.2499,False)):
            r=calculation(original,approximate,approximate,dict(kind="top1_drop",labels=[0]*16,maximum_drop_percentage_points=limit))
            self.assertEqual(r["application_metric"]["approximation_metric_passed"],passed)
    def test_mae_boundaries(self):
        for maximum,expected in ((.5,True),(.499,False),(1.,True)):
            r=calculation([[0.,0.]],[[1.,0.]],[[1.,0.]],dict(kind="mae",maximum_error=maximum))
            self.assertEqual(r["application_metric"]["approximation_metric_passed"],expected)
    def test_maximum_error_not_just_average(self):
        r=calculation([[0.,0.,0.]],[[0.,0.,3.]],[[0.,0.,3.]],dict(kind="maximum_absolute_error",maximum_error=1.))
        self.assertFalse(r["application_metric"]["approximation_metric_passed"])
        self.assertEqual(r["approximation_error"]["mae"],1.)
    def test_data_change_and_boolean_identity_change_invalidates_binding(self):
        metric=dict(kind="mae",maximum_error=0.)
        r=calculation([[1.]],[[1.]],[[1.]],metric)
        check_calculation(r,[[1.]],[[1.]],[[1.]],metric)
        with self.assertRaises(ValueError):check_calculation(r,[[2.]],[[1.]],[[1.]],metric)
        changed=copy.deepcopy(r);changed["samples"]=True
        changed["binding"]=digest({k:v for k,v in changed.items()if k!="binding"})
        with self.assertRaises(ValueError):check_calculation(changed,[[1.]],[[1.]],[[1.]],metric)
    def test_invalid_shapes_labels_and_types(self):
        good=[[1.,0.]]
        invalid=[([],good,good,dict(kind="mae",maximum_error=0.)),
            ([[True,0.]],good,good,dict(kind="mae",maximum_error=0.)),
            ([[float("nan"),0.]],good,good,dict(kind="mae",maximum_error=0.)),
            (good,[[1.]],good,dict(kind="mae",maximum_error=0.)),
            (good,good,good,dict(kind="top1_drop",labels=[True],maximum_drop_percentage_points=1.)),
            (good,good,good,dict(kind="top1_drop",labels=[2],maximum_drop_percentage_points=1.)),
            (good,good,good,dict(kind="top1_drop",labels=[0],maximum_drop_percentage_points=True)),
            (good,good,good,dict(kind="mae",maximum_error=float("inf"))),
            (good,good,good,dict(kind="unknown",maximum_error=0.)),
            (good*17,good*17,good*17,dict(kind="mae",maximum_error=0.))]
        for original,approx,actual,metric in invalid:
            with self.subTest(metric=metric),self.assertRaises(ValueError):calculation(original,approx,actual,metric)
if __name__=="__main__":unittest.main(verbosity=2)
