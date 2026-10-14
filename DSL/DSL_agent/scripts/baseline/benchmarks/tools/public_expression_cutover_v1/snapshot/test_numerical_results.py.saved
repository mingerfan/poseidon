import json,tempfile,unittest
from pathlib import Path
import numpy as np
from numerical_results import compare,dump
class NumericalResultsTests(unittest.TestCase):
    def test_frozen_threshold_is_elementwise(self):
        report=compare([[0.,1.00010,-1.00012]],[[0.,1.,-1.]],1e-5,1e-4)
        self.assertFalse(report["passed"])
        self.assertEqual(report["elementwise_pass"],[[True,True,False]])
        self.assertEqual(report["compared_values"],3)
    def test_zero_reference_has_no_relative_or_cosine_claim(self):
        report=compare([[0.,1e-6]],[[0.,0.]],1e-5,1e-4)
        self.assertTrue(report["passed"])
        self.assertIsNone(report["max_nonzero_reference_relative_error"])
        self.assertIsNone(report["cosine_similarity"])
        self.assertEqual(report["relative_error"],[[None,None]])
    def test_invalid_arrays_rejected(self):
        for a,b in (([[1.]],[[1.,2.]]),([1.],[1.]),([[]],[[]]),([[np.nan]],[[1.]]),([[1.]],[[np.inf]])):
            with self.subTest(a=a,b=b),self.assertRaises(ValueError):compare(a,b,1e-5,1e-4)
    def test_historical_metrics_preserved(self):
        from seal_cpu_golden import compare as historical
        rng=np.random.RandomState(235);reference=rng.uniform(-1,1,(4,16));reference[0,0]=0
        for magnitude in (0.,1e-7,1e-3):
            actual=reference+rng.uniform(-magnitude,magnitude,reference.shape)
            self.assertEqual(compare(actual,reference,1e-5,1e-4),historical(actual,reference,1e-5,1e-4))
    def test_reports_are_strict_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"report.json";dump(p,{"passed":True})
            self.assertEqual(json.loads(p.read_text()),{"passed":True})
            with self.assertRaises(ValueError):dump(p,{"value":float("nan")})
