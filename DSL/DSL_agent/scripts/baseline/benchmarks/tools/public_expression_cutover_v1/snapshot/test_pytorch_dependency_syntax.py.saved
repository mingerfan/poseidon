"""Syntax reuse never replaces current dependency existence and byte hashing."""
import unittest,tempfile,hashlib
from pathlib import Path
from pytorch_ordered_contract import _dependency_names,_implementation_paths_at
class DependencySyntaxTests(unittest.TestCase):
 def setUp(self):_dependency_names.cache_clear()
 def test_identical_bytes_reuse_syntax(self):
  code=b"import a\nfrom b import c\n"
  self.assertEqual(_dependency_names(code),frozenset({"a.py","b.py"}))
  _dependency_names(code)
  self.assertEqual(_dependency_names.cache_info().hits,1)
 def test_changed_bytes_are_parsed_again(self):
  self.assertEqual(_dependency_names(b"import old\n"),frozenset({"old.py"}))
  self.assertEqual(_dependency_names(b"import new\n"),frozenset({"new.py"}))
  self.assertEqual(_dependency_names.cache_info().misses,2)
 def test_adjacent_locked_data(self):
  code=b"p.with_name('config.json')\np.with_name('guard.py')\np.with_name('private.txt')\n"
  self.assertEqual(_dependency_names(code),frozenset({"config.json","guard.py"}))
 def test_dynamic_path_is_not_mistaken_for_literal(self):
  self.assertEqual(_dependency_names(b"p.with_name(variable)\n"),frozenset())
 def test_results_are_immutable(self):
  names=_dependency_names(b"import a\n")
  with self.assertRaises(AttributeError):names.add("fake.py")
 def test_invalid_replacement_source_is_not_cached_as_valid(self):
  _dependency_names(b"import a\n")
  with self.assertRaises(SyntaxError):_dependency_names(b"import !!!\n")

class LiveDependencyTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.addCleanup(self.temp.cleanup)
 def paths(self):return _implementation_paths_at(self.root,self.root,["entry.py"])
 def test_newly_present_local_import_is_discovered(self):
  (self.root/"entry.py").write_text("import added\n")
  self.assertEqual(set(self.paths()),{"entry.py"})
  (self.root/"added.py").write_text("VALUE = 1\n")
  self.assertEqual(set(self.paths()),{"entry.py","added.py"})
 def test_changed_import_bytes_change_closure(self):
  (self.root/"entry.py").write_text("import first\n")
  (self.root/"first.py").write_text("VALUE = 1\n")
  (self.root/"second.py").write_text("VALUE = 2\n")
  self.assertEqual(set(self.paths()),{"entry.py","first.py"})
  (self.root/"entry.py").write_text("import second\n")
  self.assertEqual(set(self.paths()),{"entry.py","second.py"})
 def test_dependency_content_is_rehashed_after_cached_syntax(self):
  (self.root/"entry.py").write_text("import child\n")
  child=self.root/"child.py";child.write_text("VALUE = 1\n")
  first={n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in self.paths().items()}
  child.write_text("VALUE = 9\n")
  second={n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in self.paths().items()}
  self.assertNotEqual(first["child.py"],second["child.py"])
 def test_deleted_required_entry_rejected(self):
  (self.root/"entry.py").write_text("VALUE = 1\n")
  self.paths();(self.root/"entry.py").unlink()
  with self.assertRaisesRegex(ValueError,"file missing"):self.paths()
 def test_locked_json_content_is_not_cached_as_verified(self):
  (self.root/"entry.py").write_text("p.with_name('parameters.json')\n")
  data=self.root/"parameters.json";data.write_text('{"value":1}')
  first=hashlib.sha256(self.paths()["parameters.json"].read_bytes()).hexdigest()
  data.write_text('{"value":2}')
  self.assertNotEqual(first,hashlib.sha256(self.paths()["parameters.json"].read_bytes()).hexdigest())
