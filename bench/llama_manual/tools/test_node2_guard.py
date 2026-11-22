#!/usr/bin/env python3
"""Pure CPU registration parsing regressions, no SSH/CUDA/signals."""
import importlib.util,json,pathlib,tempfile,unittest
spec=importlib.util.spec_from_file_location('guard',pathlib.Path(__file__).with_name('run_node2_guard.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class Mapping(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=pathlib.Path(self.tmp.name)/'map.json';self.identity={'pid':101,'pgid':101,'start_ticks':'12345'}
  self.m={'namespace_pid':101,'pgid':101,'start_ticks':'12345','supervisor_pid':99,'gpu_uuid':g.UUID,'nvml_host_pid':10001}
 def tearDown(self):self.tmp.cleanup()
 def test_absent(self):self.assertIsNone(g.read_registered_map(self.p,self.identity,99))
 def test_postexit_map_accepted(self):
  self.p.write_text(json.dumps(self.m));self.assertEqual(g.read_registered_map(self.p,self.identity,99),self.m)
 def test_all_ownership_fields_rejected(self):
  for k in ('namespace_pid','pgid','start_ticks','supervisor_pid','gpu_uuid','nvml_host_pid'):
   with self.subTest(k=k):
    m=self.m.copy();m[k]='wrong';self.p.write_text(json.dumps(m))
    with self.assertRaises(RuntimeError):g.read_registered_map(self.p,self.identity,99)
 def test_nonpositive_host_pid(self):
  for pid in (0,-1,True):
   self.m['nvml_host_pid']=pid;self.p.write_text(json.dumps(self.m))
   with self.assertRaises(RuntimeError):g.read_registered_map(self.p,self.identity,99)
 def test_no_owned_identity(self):
  self.p.write_text(json.dumps(self.m))
  with self.assertRaises(RuntimeError):g.read_registered_map(self.p,None,99)
 def test_duplicate_json(self):
  self.p.write_text('{"namespace_pid":101,"namespace_pid":101}')
  with self.assertRaises(ValueError):g.read_registered_map(self.p,self.identity,99)
if __name__=='__main__':unittest.main()
