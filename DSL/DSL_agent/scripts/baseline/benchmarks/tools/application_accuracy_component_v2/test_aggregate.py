"""Dataset coverage, identity and exact full-test-set classification threshold."""
import copy,unittest
from application_metrics import calculation,aggregate
IDENTITY=dict(model_sha256="a"*64,package_binding="b"*64,dataset_sha256="c"*64)
def chunk(indices,*,bad=0,limit=1.):
    original=[[1.,0.]for _ in indices]
    approx=[[0.,1.]if i<bad else[1.,0.]for i in indices]
    return dict(indices=list(indices),calculation=calculation(original,approx,approx,
        dict(kind="top1_drop",labels=[0]*len(indices),maximum_drop_percentage_points=limit)))
class AggregateTests(unittest.TestCase):
    def test_full_10000_dataset_exact_one_percentage_point(self):
        chunks=[chunk(range(start,start+16),bad=100)for start in range(0,10000,16)]
        r=aggregate(chunks,sample_count=10000,identity=IDENTITY)
        self.assertEqual(r["application_metric"]["approximation_drop_percentage_points"],1.)
        self.assertTrue(r["application_metric"]["approximation_metric_passed"])
        self.assertFalse(r["application_accuracy_certified"])
        self.assertEqual(r["per_execution_sample_limit"],16)
    def test_duplicate_and_missing_positions_rejected(self):
        for chunks,count in (([chunk([0,1]),chunk([1,2])],3),([chunk([0,1])],3)):
            with self.assertRaises(ValueError):aggregate(chunks,sample_count=count,identity=IDENTITY)
    def test_mixed_metric_and_invalid_identity_rejected(self):
        with self.assertRaises(ValueError):aggregate([chunk([0],limit=1.),chunk([1],limit=2.)],sample_count=2,identity=IDENTITY)
        with self.assertRaises(ValueError):aggregate([chunk([0])],sample_count=1,identity=dict(IDENTITY,extra="unexpected"))
        with self.assertRaises(ValueError):aggregate([chunk([0])],sample_count=1,identity=dict(IDENTITY,dataset_sha256=True))
    def test_regression_is_weighted_by_values(self):
        rows=[dict(indices=[0,1],calculation=calculation([[0.],[0.]],[[0.],[0.]],[[0.],[0.]],dict(kind="mae",maximum_error=1.))),
            dict(indices=[2],calculation=calculation([[0.]],[[3.]],[[3.]],dict(kind="mae",maximum_error=1.)))]
        r=aggregate(rows,sample_count=3,identity=IDENTITY)
        self.assertEqual(r["application_metric"]["approximation_value"],1.)
        self.assertTrue(r["application_metric"]["approximation_metric_passed"])
if __name__=="__main__":unittest.main(verbosity=2)
