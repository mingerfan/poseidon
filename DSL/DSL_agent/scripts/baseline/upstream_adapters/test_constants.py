import math
import struct
import unittest
from upstream_adapters.constants import canonicalize, SLOTS

def cst(*vectors):
    return struct.pack("<q", len(vectors))+b"".join(struct.pack("<q",len(v))+struct.pack("<%sd"%len(v),*v) for v in vectors)

class ConstantLayoutTests(unittest.TestCase):
    def test_exact_nonuniform_expansion(self):
        for period in (4,8,16,32,64,128,256):
            block=[(-1)**i*(i+0.25) for i in range(period)]
            original=cst(block*(SLOTS//period),[2.0],block)
            result,record=canonicalize(original,period)
            self.assertEqual(result,cst(block,[2.0],block))
            self.assertEqual(record["changes"],[dict(index=0,original_size=SLOTS,canonical_size=period)])
            self.assertEqual(canonicalize(result,period)[0],result)
    def test_tail_and_signed_zero_rejected(self):
        for bad in (1.0,-0.0):
            values=[0.0]*SLOTS;values[-1]=bad
            with self.assertRaisesRegex(ValueError,"Non-periodic"):
                canonicalize(cst(values),4)
    def test_nonfinite_rejected(self):
        for bad in (math.nan,math.inf,-math.inf):
            with self.assertRaisesRegex(ValueError,"Non-finite"):
                canonicalize(cst([bad]*SLOTS),4)
    def test_malformed_rejected(self):
        for data in (b"",struct.pack("<q",257),struct.pack("<q",-1),cst([1.0])*2,
                     cst([1.0]*4)[:-1],cst([1.0]*8),cst([])):
            with self.assertRaises(ValueError):canonicalize(data,4)
        for p in (True,0,2,3,512):
            with self.assertRaises(ValueError):canonicalize(cst([1.0]),p)

if __name__=="__main__":unittest.main()
