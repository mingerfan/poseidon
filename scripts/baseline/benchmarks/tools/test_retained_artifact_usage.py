"""Deterministic regression for key-directory cleanup during byte accounting."""
import errno,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from stage2_agent_campaign import ROOT
from retained_artifact_usage import size_bytes
class UsageTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix=".size-accounting-test-",dir=ROOT)
  self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_nested_files_and_duplicate_root(self):
  (self.root/"a").write_bytes(b"123");(self.root/"sub").mkdir();(self.root/"sub"/"b").write_bytes(b"4567")
  self.assertEqual(size_bytes([self.root,self.root]),7)
 def test_missing_future_directory_is_zero(self):
  self.assertEqual(size_bytes([self.root/"not-created"]),0)
 def test_key_directory_removed_between_listing_and_descent(self):
  keys=self.root/"private-keys";keys.mkdir();(self.root/"retained").write_bytes(b"evidence")
  real=os.scandir
  class ClosingScan:
   def __init__(self,path):self.path=Path(path);self.scan=real(path)
   def __enter__(self):return self.scan.__enter__()
   def __exit__(self,*a):
    result=self.scan.__exit__(*a)
    if self.path==keys.parent:keys.rmdir()
    return result
  with patch("retained_artifact_usage.os.scandir",side_effect=ClosingScan):
   self.assertEqual(size_bytes([self.root]),8)
  self.assertFalse(keys.exists())
 def test_disappearing_file_does_not_hide_retained_file(self):
  class GoneEntry:
   path="deleted";name="deleted"
   def stat(self,**kwargs):raise FileNotFoundError(errno.ENOENT,"gone")
  real=os.scandir
  class Scan:
   def __init__(self,path):self.scan=real(path)
   def __enter__(self):return iter([GoneEntry(),*list(self.scan.__enter__())])
   def __exit__(self,*args):return self.scan.__exit__(*args)
  (self.root/"retained").write_bytes(b"12345")
  with patch("retained_artifact_usage.os.scandir",side_effect=Scan):self.assertEqual(size_bytes([self.root]),5)
 def test_permission_failure_is_not_ignored(self):
  with patch("retained_artifact_usage.os.scandir",side_effect=PermissionError(errno.EACCES,"denied")):
   with self.assertRaises(PermissionError):size_bytes([self.root])
 def test_other_io_failure_is_not_ignored(self):
  with patch("retained_artifact_usage.os.scandir",side_effect=OSError(errno.EIO,"I/O")):
   with self.assertRaises(OSError):size_bytes([self.root])
 def test_file_links_keep_previous_byte_accounting(self):
  f=self.root/"retained";f.write_bytes(b"1234");(self.root/"link").symlink_to(f)
  self.assertEqual(size_bytes([self.root]),8)
 def test_directory_link_not_recursed(self):
  d=self.root/"sub";d.mkdir();(d/"retained").write_bytes(b"1234");(d/"cycle").symlink_to(self.root,target_is_directory=True)
  self.assertEqual(size_bytes([self.root]),4)
if __name__=="__main__":unittest.main()
