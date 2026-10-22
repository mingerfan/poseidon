"""Syntax memoization must never cache trust decisions or current file identities."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import ast
import application_identity as identity

class SyntaxCacheTests(unittest.TestCase):
    def setUp(self):
        identity._SYNTAX_CACHE.clear()
        identity._SYNTAX_CACHE_BYTES=0
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_cached_syntax_still_hashes_changed_child(self):
        (self.root/"entry.py").write_text("import child\n")
        (self.root/"child.py").write_text("value=1\n")
        with patch.object(identity.ast,"parse",wraps=ast.parse) as parser:
            first=identity.source_closure(self.root,["entry.py"])
            parsed=parser.call_count
            self.assertEqual(identity.source_closure(self.root,["entry.py"]),first)
            self.assertEqual(parser.call_count,parsed)
            (self.root/"child.py").write_text("value=2\n")
            changed=identity.source_closure(self.root,["entry.py"])
            self.assertGreater(parser.call_count,parsed)
        self.assertNotEqual(first["sources"]["child.py"],changed["sources"]["child.py"])

    def test_cached_import_resolves_new_local_file(self):
        (self.root/"entry.py").write_text("import callback\n")
        first=identity.source_closure(self.root,["entry.py"])
        self.assertIn("callback",first["external_imports"])
        (self.root/"callback.py").write_text("value=1\n")
        second=identity.source_closure(self.root,["entry.py"])
        self.assertIn("callback.py",second["sources"])
        self.assertNotIn("callback",second["external_imports"])

    def test_cached_dynamic_call_rechecks_current_authority(self):
        (self.root/"entry.py").write_text("import importlib\nname='module'\nimportlib.import_module(name)\n")
        key=("entry.py","importlib.import_module(name)")
        with patch.dict(identity.DYNAMIC,{key:"request-bound source"}):
            identity.source_closure(self.root,["entry.py"])
        with self.assertRaisesRegex(ValueError,"Unbounded dynamic dependency"):
            identity.source_closure(self.root,["entry.py"])

    def test_cache_has_explicit_entry_and_payload_bounds(self):
        for n in range(identity.SYNTAX_CACHE_ENTRIES+8):
            identity._syntax_facts(("value="+str(n)+"\n").encode(),"entry.py")
        self.assertLessEqual(len(identity._SYNTAX_CACHE),identity.SYNTAX_CACHE_ENTRIES)
        self.assertLessEqual(identity._SYNTAX_CACHE_BYTES,identity.SYNTAX_CACHE_BYTES)
        self.assertEqual(identity._SYNTAX_CACHE_BYTES,sum(size for facts,size in identity._SYNTAX_CACHE.values()))


    def test_large_closure_and_repository_scan_reuse_only_syntax(self):
        base=self.root/"scripts/baseline";base.mkdir(parents=True)
        names=["part_%03d"%i for i in range(600)]
        (base/"entry.py").write_text("import "+", ".join(names)+"\n")
        for name in names:(base/(name+".py")).write_text("VALUE=1\n")
        with patch.object(identity.ast,"parse",wraps=ast.parse) as parser, \
                patch.object(identity,"_read",wraps=identity._read) as reader:
            first=identity.source_closure(base,["entry.py"])
            parsed=parser.call_count;read_count=reader.call_count
            self.assertEqual(parsed,601)
            identity._repository_references(self.root,first["sources"])
            self.assertEqual(parser.call_count,parsed)
            second=identity.source_closure(base,["entry.py"])
            identity._repository_references(self.root,second["sources"])
            self.assertEqual(second,first)
            self.assertEqual(parser.call_count,parsed)
            self.assertGreater(reader.call_count,read_count)
        self.assertLessEqual(identity._SYNTAX_CACHE_BYTES,identity.SYNTAX_CACHE_BYTES)

    def test_same_size_and_timestamp_source_change_gets_fresh_imports(self):
        entry=self.root/"entry.py"
        entry.write_text("import alpha\n")
        for name in ("alpha","bravo"):(self.root/(name+".py")).write_text("value=1\n")
        first=identity.source_closure(self.root,["entry.py"])
        stat=entry.stat()
        entry.write_text("import bravo\n")
        import os
        os.utime(entry,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        self.assertEqual(entry.stat().st_size,stat.st_size)
        self.assertEqual(entry.stat().st_mtime_ns,stat.st_mtime_ns)
        second=identity.source_closure(self.root,["entry.py"])
        self.assertIn("alpha.py",first["sources"])
        self.assertNotIn("alpha.py",second["sources"])
        self.assertIn("bravo.py",second["sources"])
        self.assertNotEqual(first["sources"]["entry.py"],second["sources"]["entry.py"])

    def test_cached_repository_asset_is_reread_and_rehashed(self):
        base=self.root/"scripts/baseline";base.mkdir(parents=True)
        (base/"entry.py").write_text("CONFIG='settings.json'\n")
        asset=self.root/"settings.json";asset.write_text('{"value":1}')
        closure=identity.source_closure(base,["entry.py"])
        with patch.object(identity.ast,"parse",wraps=ast.parse) as parser:
            before=identity._repository_references(self.root,closure["sources"])
            asset.write_text('{"value":2}')
            after=identity._repository_references(self.root,closure["sources"])
            self.assertEqual(parser.call_count,0)
        self.assertNotEqual(before["settings.json"],after["settings.json"])

    def test_warm_cache_rejects_new_source_symlink(self):
        entry=self.root/"entry.py";entry.write_text("VALUE=1\n")
        identity.source_closure(self.root,["entry.py"])
        other=self.root/"other.py";other.write_text(entry.read_text())
        entry.unlink();entry.symlink_to(other)
        with self.assertRaisesRegex(ValueError,"symlink"):
            identity.source_closure(self.root,["entry.py"])

    def test_fact_byte_bound_evicts_and_does_not_retain_oversized_value(self):
        with patch.object(identity,"SYNTAX_CACHE_BYTES",48):
            for i in range(8):identity._syntax_facts(("import m"+str(i)+"\n").encode(),"entry.py")
            self.assertLessEqual(identity._SYNTAX_CACHE_BYTES,48)
            self.assertLessEqual(len(identity._SYNTAX_CACHE),2)
            retained=list(identity._SYNTAX_CACHE)
            raw=("import "+"long_name_"*10+"\n").encode()
            with patch.object(identity.ast,"parse",wraps=ast.parse) as parser:
                first=identity._syntax_facts(raw,"entry.py")
                second=identity._syntax_facts(raw,"entry.py")
                self.assertEqual(first,second)
                self.assertEqual(parser.call_count,2)
            self.assertEqual(list(identity._SYNTAX_CACHE),retained)
            self.assertEqual(identity._SYNTAX_CACHE_BYTES,sum(s for _,s in identity._SYNTAX_CACHE.values()))

    def test_evicted_facts_preserve_dependency_result(self):
        for name in ("a","b","c"):(self.root/(name+".py")).write_text("VALUE=1\n")
        (self.root/"entry.py").write_text("import a,b,c\n")
        expected=identity.source_closure(self.root,["entry.py"])
        identity._SYNTAX_CACHE.clear();identity._SYNTAX_CACHE_BYTES=0
        with patch.object(identity,"SYNTAX_CACHE_ENTRIES",1):
            actual=identity.source_closure(self.root,["entry.py"])
        self.assertEqual(expected,actual)
        self.assertLessEqual(len(identity._SYNTAX_CACHE),1)

if __name__=="__main__":unittest.main()
