"""Capture source closure, shadowing and immutable-byte cache boundaries."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from trusted_capture_implementation import implementation_paths_at


class CaptureImplementationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix="capture-deps-unit-")
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.local=self.root/'local';self.local.mkdir()
        self.retained=self.root/'retained';self.retained.mkdir()

    def write(self,name,text,*,retained=False):
        root=self.retained if retained else self.local
        (root/name).write_text(text)

    def paths(self,initial=('entry.py',)):
        return implementation_paths_at(self.local,self.retained,initial)

    def test_recursive_eager_lazy_and_adjacent_json_dependencies_are_closed(self):
        self.write('entry.py','import child\ndef lazy():\n import deferred\n')
        self.write('child.py','from pathlib import Path\nimport entry\nvalue=Path(__file__).with_name("rules.json")\n')
        self.write('deferred.py','import retained_only\n')
        self.write('retained_only.py','',retained=True)
        self.write('rules.json','{}')
        self.write('unused.py','raise RuntimeError("must remain unmounted")\n')
        self.assertEqual(set(self.paths()),{'entry.py','child.py','deferred.py','retained_only.py','rules.json'})
        self.assertEqual(self.paths()['retained_only.py'],self.retained/'retained_only.py')

    def test_scanning_never_executes_source_and_never_inventories_private_files(self):
        marker=self.root/'sentinel'
        self.write('entry.py',f'from pathlib import Path\nPath({str(marker)!r}).write_text("wrong")\n')
        self.write('.env','FAKE-NOT-A-CREDENTIAL')
        self.assertEqual(set(self.paths()),{'entry.py'})
        self.assertFalse(marker.exists())

    def test_source_byte_change_is_seen_without_metadata_cache_assumptions(self):
        self.write('entry.py','import first\n');self.write('first.py','');self.write('other.py','')
        self.assertIn('first.py',self.paths())
        self.write('entry.py','import other\n')
        self.assertIn('other.py',self.paths());self.assertNotIn('first.py',self.paths())

    def test_missing_initial_file_and_unsafe_names_fail_closed(self):
        for initial in [('missing.py',),('../private.py',),('/private.py',),('.env',)]:
            with self.subTest(initial=initial),self.assertRaises(ValueError):self.paths(initial)

    def test_local_source_precedes_retained_source(self):
        self.write('entry.py','value=1');self.write('entry.py','value=2',retained=True)
        self.assertEqual(self.paths()['entry.py'],self.local/'entry.py')

    def test_indirect_trusted_dependency_is_rejected(self):
        self.write('entry.py','import child\n')
        external=self.root/'external.py';external.write_text('')
        (self.local/'child.py').symlink_to(external)
        with self.assertRaisesRegex(ValueError,'indirect'):self.paths()

    def test_all_production_capture_versions_hash_the_same_files_they_mount(self):
        import trusted_pytorch_import as legacy
        import pytorch_model_intake as arithmetic
        import pytorch_ordered_contract as ordered
        for module in [legacy,arithmetic,ordered]:
            with self.subTest(module=module.__name__):
                paths=module.implementation_paths()
                identities=module.source_identity()
                self.assertEqual(identities,{n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in paths.items()})
                self.assertIn('platform-profiles.json',paths)
                self.assertIn('trusted_capture_implementation.py',paths)
                self.assertTrue(all(Path(n).name==n and Path(n).suffix in('.py','.json')for n in paths))
        self.assertIn('pytorch_tensor_contraction_source.py',arithmetic.implementation_paths())

    def test_legacy_source_cannot_shadow_a_transitive_trusted_dependency(self):
        from trusted_pytorch_import import validate_package
        source='def make():\n return None\n'
        name='platform_config.py'
        spec=dict(format='poseidon-trusted-pytorch-v1',files={name:hashlib.sha256(source.encode()).hexdigest()},
            entry='platform_config.make',inputs=[dict(name='x',shape=[2])],
            outputs=['y'],approximations={})
        with self.assertRaisesRegex(ValueError,'conflicts'):validate_package(spec,{name:source})


if __name__=='__main__':unittest.main()
