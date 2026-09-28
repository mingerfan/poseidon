import json,tempfile,unittest
from pathlib import Path
from campaign_request_files import read_request
class RequestFiles(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)/"request.json"
 def write(self,x):self.p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
 def test_large_frozen_request(self):
  x={"request_id":"locked","values":[0]*20000};self.write(x)
  self.assertGreater(self.p.stat().st_size,131072);self.assertEqual(read_request(self.p,x),x)
 def test_same_size_tampering(self):
  self.write({"a":2})
  with self.assertRaisesRegex(ValueError,"changed"):read_request(self.p,{"a":1})
 def test_padding_cannot_increase_boundary(self):
  self.write({"a":1});self.p.write_text(self.p.read_text()+" "*100)
  with self.assertRaisesRegex(ValueError,"file gate"):read_request(self.p,{"a":1})
 def test_symlink_rejected(self):
  target=self.p.with_name("real");target.write_text("{}");self.p.symlink_to(target)
  with self.assertRaises(ValueError):read_request(self.p,{})
 def test_missing_rejected(self):
  with self.assertRaises(ValueError):read_request(self.p,{})
 def test_frozen_oversize_rejected(self):
  with self.assertRaisesRegex(ValueError,"shard boundary"):read_request(self.p,{"a":"x"*(8*1024**2)})
if __name__=="__main__":unittest.main()
