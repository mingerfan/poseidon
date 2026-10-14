import hashlib,io,struct,tempfile,unittest,zipfile
from pathlib import Path
from numeric_link import read_f64,select,reference_from_npz,evidence_file,Reader

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

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name).resolve();self.reader=Reader((self.folder,))
    def put(self,name,raw=b"{}"):
        p=self.folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);return p
    def test_explicit_supplement_is_freshly_bound(self):
        p=self.put("attempt-00/output/execution.json")
        self.assertEqual(evidence_file(self.reader,{"files":{}},self.folder,"attempt-00/output/execution.json",True),{})
        self.assertEqual(self.reader.known()[str(p)],hashlib.sha256(b"{}").hexdigest())
    def test_original_files_must_keep_index_binding(self):
        self.put("report.json")
        with self.assertRaisesRegex(ValueError,"numeric_link_index_file"):
            evidence_file(self.reader,{"files":{}},self.folder,"report.json")
    def test_indexed_supplement_drift_rejected(self):
        p=self.put("attempt-00/output/decrypted.npy")
        with self.assertRaisesRegex(ValueError,"evidence_content_drift"):
            evidence_file(self.reader,{"files":{str(p):"0"*64}},self.folder,"attempt-00/output/decrypted.npy")
    def test_unknown_or_traversal_name_rejected(self):
        for name in ("attempt-00/output/other.json","../report.json","/tmp/report.json","attempt-00/output/../execution.json"):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"numeric_link_file_name"):
                evidence_file(self.reader,{"files":{}},self.folder,name)
    def test_supplement_symlink_rejected(self):
        target=self.put("target.json");p=self.folder/"attempt-00/output/execution.json"
        p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(target)
        with self.assertRaisesRegex(ValueError,"evidence_regular_file"):
            evidence_file(self.reader,{"files":{}},self.folder,"attempt-00/output/execution.json")

if __name__=="__main__":
    unittest.main(verbosity=2)
