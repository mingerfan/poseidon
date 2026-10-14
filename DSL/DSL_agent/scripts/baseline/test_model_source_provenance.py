"""Source/package origin checks. Fixtures never compile or execute FHE."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_graph import digest,canonical
from benchmark_suite import Builder
from component_contract import prepare_task
from model_source_provenance import build,verify,check_package,reuse_identity,FILE,FORMAT
from application_model_source import own_identity


class SourceOriginTests(unittest.TestCase):
    def setUp(self):
        b=Builder([(2,)])
        self.model=b.finish(b.node("square",["input0"]))
        self.request=prepare_task(self.model)
        raw=canonical(self.model)
        self.snapshot=dict(kind="graph",resource_profile="legacy-v1",
            binding="a"*64,files={"model.json":dict(bytes=len(raw),sha256=digest(self.model))},
            implementation=own_identity())
        self.origin=build(self.snapshot,self.request)

    def rebind(self,value):
        value["binding"]=digest({k:v for k,v in value.items() if k!="binding"})
        return value

    def test_reuse_identity_preserves_semantics_but_excludes_task_snapshot(self):
        other=self.rebind(dict(self.origin,snapshot_binding="b"*64))
        verify(other,self.request)
        self.assertEqual(reuse_identity(self.origin),reuse_identity(other))
        changed=self.rebind(dict(self.origin,implementation=dict(
            self.origin["implementation"],controller_sha256="0"*64)))
        self.assertNotEqual(reuse_identity(self.origin),reuse_identity(changed))
        changed=self.rebind(dict(self.origin,files={}))
        self.assertNotEqual(reuse_identity(self.origin),reuse_identity(changed))
        with self.assertRaises(ValueError):
            reuse_identity(dict(self.origin,private_key="rejected"))

    def test_origin_is_separate_from_validation_and_request(self):
        self.assertEqual(verify(self.origin,self.request),self.origin)
        self.assertNotIn("model_source",self.request)
        self.assertNotIn("numerically_validated",self.origin)
        self.assertNotIn("agent_generated",self.origin)

    def test_rehashed_different_model_or_source_inventory_rejected(self):
        for key,value in [("model_sha256","b"*64),("files",{}),
                          ("resource_profile","general-tensors-v1"),("provenance_sha256","c"*64)]:
            changed=self.rebind(dict(copy.deepcopy(self.origin),**{key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify(changed,self.request)

    def test_invalid_source_kind_and_fields_rejected(self):
        for changed in [dict(self.origin,kind="pickle"),dict(self.origin,private_key="bad"),
                        dict(self.origin,snapshot_binding=".."),dict(self.origin,implementation={})]:
            with self.subTest(change=list(changed)),self.assertRaises(ValueError):
                verify(self.rebind(changed),self.request)

    def test_identity_is_checked_against_package_generation_stage(self):
        with self.assertRaisesRegex(ValueError,"controller/dependency"):
            verify(self.origin,self.request,controller_sha256="0"*64)

    def test_python_claim_requires_full_model_provenance(self):
        changed=self.rebind(dict(self.origin,kind="trusted_pytorch"))
        with self.assertRaisesRegex(ValueError,"provenance required"):
            verify(changed,self.request)

    def test_missing_descriptor_rejects_unreferenced_origin_file(self):
        manifest=dict(origin={},files={FILE:"a"*64},format="poseidon-dsl-bundle-v5")
        with self.assertRaisesRegex(ValueError,"Unreferenced"):
            check_package("/not-accessed",manifest,self.request)

    def test_old_packages_without_origin_remain_optional(self):
        self.assertIsNone(check_package("/not-accessed",
            dict(origin={},files={},format="poseidon-dsl-bundle-v5"),self.request))

    def test_descriptor_version_and_path_rejection(self):
        for fmt,file in [("unknown",FILE),(FORMAT,"../source.json")]:
            manifest=dict(origin={"model_source":dict(format=fmt,file=file,sha256="a"*64)},
                files={file:"a"*64},format="poseidon-dsl-bundle-v5")
            with self.assertRaisesRegex(ValueError,"descriptor"):
                check_package("/not-accessed",manifest,self.request)

    def test_actual_record_file_read_checks_generation_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/"artifacts").mkdir()
            raw=canonical(self.origin);(folder/FILE).write_bytes(raw)
            h=hashlib.sha256(raw).hexdigest()
            manifest=dict(format="poseidon-dsl-bundle-v5",
                files={FILE:h},origin={"model_source":dict(format=FORMAT,file=FILE,sha256=h)})
            identity={"generation":{"sources":{"application_model_source.py":
                         self.origin["implementation"]["controller_sha256"]}}}
            with patch("candidate_bundle.package_identities",return_value=identity):
                self.assertEqual(check_package(folder,manifest,self.request),self.origin)
                identity["generation"]["sources"]["application_model_source.py"]="0"*64
                with self.assertRaisesRegex(ValueError,"controller/dependency"):
                    check_package(folder,manifest,self.request)


if __name__=="__main__":
    unittest.main()
