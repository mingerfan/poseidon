"""Package integrity counterexamples; fixtures contain no executable program."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from delivery_component import bound,canonical,sha
import normal_cpu_overlay_package as package


class PackageInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)/'package';self.root.mkdir()
        (self.root/'program').mkdir();(self.root/'program/candidate.py').write_bytes(b'inert exact source\r\n')
        (self.root/'program/request.json').write_bytes(b'{"fixture":true}')
        (self.root/'program/manifest.json').write_bytes(b'{"inert_original_manifest":true}')
        (self.root/'adapter-identity.json').write_bytes(b'{"inert_identity":true}')
        self.value=bound(dict(format=package.FORMAT,files=package.inventory(self.root),
            inner_manifest=dict(path='program/manifest.json',sha256=sha((self.root/'program/manifest.json').read_bytes())),
            source_sha256=sha((self.root/'program/candidate.py').read_bytes()),request_sha256=sha((self.root/'program/request.json').read_bytes()),
            qualification_level='numerical',full_SDK_or_network_qualified=False))

    def test_exact_source_bytes_and_all_package_files_are_frozen(self):
        self.assertEqual(package.verify_files(self.root,self.value),self.value)
        self.assertEqual((self.root/'program/candidate.py').read_bytes(),b'inert exact source\r\n')

    def test_LF_normalization_cannot_replace_frozen_CRLF_source(self):
        (self.root/'program/candidate.py').write_bytes(b'inert exact source\n')
        with self.assertRaises(ValueError):package.verify_files(self.root,self.value)

    def test_mutated_compiler_artifact_or_request_cannot_reload(self):
        (self.root/'program/request.json').write_bytes(b'{"changed":true}')
        with self.assertRaises(ValueError):package.verify_files(self.root,self.value)

    def test_unlisted_new_file_cannot_be_ignored(self):
        (self.root/'unlisted.py').write_bytes(b'inert unlisted file')
        with self.assertRaises(ValueError):package.verify_files(self.root,self.value)

    def test_reference_dump_cannot_be_added_by_resigning_the_outer_inventory(self):
        (self.root/'reference.json').write_bytes(b'{"inert_private_reference":true}')
        v=copy.deepcopy(self.value);v.pop('binding');v['files']=package.inventory(self.root)
        with self.assertRaises(ValueError):package.verify_files(self.root,bound(v))

    def test_missing_program_file_cannot_reload(self):
        (self.root/'program/manifest.json').unlink()
        with self.assertRaises(ValueError):package.verify_files(self.root,self.value)

    def test_file_symlink_cannot_redirect_runtime_artifacts(self):
        p=self.root/'program/candidate.py';p.unlink();outside=self.root.parent/'outside.py';outside.write_bytes(b'inert exact source\r\n');p.symlink_to(outside)
        with self.assertRaises(ValueError):package.verify_files(self.root,self.value)

    def test_directory_symlink_cannot_escape_complete_inventory(self):
        outside=self.root.parent/'elsewhere';outside.mkdir();(outside/'extra').write_text('fixture')
        (self.root/'escape').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):package.verify_files(self.root,self.value)

    def test_resigned_outer_descriptor_cannot_select_another_inner_package(self):
        v=copy.deepcopy(self.value);v.pop('binding');v['inner_manifest']['path']='other/manifest.json'
        with self.assertRaises(ValueError):package.verify_files(self.root,bound(v))

    def test_source_identity_cannot_be_resigned_to_differ_from_inventory(self):
        v=copy.deepcopy(self.value);v.pop('binding');v['source_sha256']='0'*64
        with self.assertRaises(ValueError):package.verify_files(self.root,bound(v))

    def test_package_cannot_be_resigned_as_full_SDK_or_network_qualification(self):
        v=copy.deepcopy(self.value);v.pop('binding');v['full_SDK_or_network_qualified']=True
        with self.assertRaises(ValueError):package.verify_files(self.root,bound(v))

    def test_invalid_qualification_grade_cannot_be_used_as_a_program(self):
        v=copy.deepcopy(self.value);v.pop('binding');v['qualification_level']='API_success'
        with self.assertRaises(ValueError):package.verify_files(self.root,bound(v))

    def test_forged_outer_binding_is_rejected_before_legacy_reader(self):
        v=copy.deepcopy(self.value);v['binding']='0'*64
        with self.assertRaises(ValueError):package.verify_files(self.root,v)


if __name__=='__main__':unittest.main()
