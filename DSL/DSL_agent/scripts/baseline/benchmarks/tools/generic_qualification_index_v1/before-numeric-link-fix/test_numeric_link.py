import io,struct,unittest,zipfile
from numeric_link import read_f64,select,reference_from_npz

def npy(shape=(2,2),descr="<f8",order=False,data=(1.,2.,3.,4.)):
    header=repr(dict(descr=descr,fortran_order=order,shape=shape)).encode()+b"\n"
    return b"\x93NUMPY\x01\x00"+len(header).to_bytes(2,"little")+header+struct.pack("<"+"d"*len(data),*data)

class Tests(unittest.TestCase):
    def test_exact_f64_decode(self):
        self.assertEqual(read_f64(npy()),((2,2),(1.,2.,3.,4.)))
    def test_wrong_dtype(self):
        with self.assertRaisesRegex(ValueError,"npy_f64_c_order"):read_f64(npy(descr="<i8"))
    def test_fortran_order(self):
        with self.assertRaisesRegex(ValueError,"npy_f64_c_order"):read_f64(npy(order=True))
    def test_payload_length(self):
        with self.assertRaisesRegex(ValueError,"npy_body_size"):read_f64(npy()[:-1])
    def test_nonfinite(self):
        with self.assertRaisesRegex(ValueError,"npy_finite_data"):read_f64(npy(data=(1.,2.,3.,float("nan"))))
    def test_selector_order_and_repetition(self):
        self.assertEqual(select([[[1.,2.]],[[3.,4.]]],[[0,1],[0,0],[0,1]]),(2.,1.,2.,4.,3.,4.))
    def test_negative_selector_rejected(self):
        with self.assertRaisesRegex(ValueError,"decode_selector_bounds"):select([[[1.]]],[[0,-1]])
    def test_boolean_selector_rejected(self):
        with self.assertRaisesRegex(ValueError,"decode_selector_bounds"):select([[[1.]]],[[False,0]])
    def test_duplicate_reference_rejected(self):
        raw=io.BytesIO()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning)
            with zipfile.ZipFile(raw,"w")as z:z.writestr("reference.npy",npy());z.writestr("reference.npy",npy())
        with self.assertRaisesRegex(ValueError,"npz_unique_reference"):reference_from_npz(raw.getvalue())

if __name__=="__main__":
    unittest.main(verbosity=2)
