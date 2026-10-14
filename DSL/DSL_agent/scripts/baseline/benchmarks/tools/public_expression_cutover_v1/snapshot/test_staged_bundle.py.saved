"""Versioned package checks with synthetic manifests; no FHE execution."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from benchmark_graph import canonical, digest
from benchmark_suite import Builder
from component_contract import prepare_task
from candidate_bundle import check_bundle, check_execution_dependencies, sha
import test_application_identity as identity_fixtures

class StagedBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        identity_fixture=identity_fixtures.IdentityTests()
        identity_fixture.setUp()
        self.addCleanup(identity_fixture.doCleanups)
        self.identities=identity_fixture.manifest()
        b=Builder([(2,)])
        self.request=prepare_task(b.finish(b.node("square",["input0"])))
        p=self.folder
        (p/"artifacts").mkdir()
        (p/"model.json").write_bytes(canonical(self.request["model"]))
        (p/"request.json").write_bytes(canonical(self.request))
        (p/"candidate.py").write_text('@hc.func("c,c")\ndef golden(x,zero_ct):\n    return [x*x]\n')
        (p/"artifacts/application-identities.json").write_bytes(canonical(self.identities))
        self.manifest=dict(format="poseidon-dsl-bundle-v5",
            files={str(f.relative_to(p)):sha(f) for f in p.rglob("*") if f.is_file()},
            compiler_configuration=self.request["compiler_configuration"],
            origin=dict(request_id=self.request["request_id"]), agent_generated=False,
            dependencies=dict(compiler_sha256="0"*64,helper_environment=None,implementation={},
                validation_sources={},backend_identity=self.identities["backend"],
                application_identities=dict(file="artifacts/application-identities.json",
                    sha256=sha(p/"artifacts/application-identities.json"))),
            model_provenance="not_supplied",
            validation=dict(level="numerical",encrypted_execution=True,numerically_validated=True))
        self.save()
    def save(self):
        m=self.manifest
        m["binding"]=digest({k:v for k,v in m.items() if k!="binding"})
        (self.folder/"manifest.json").write_bytes(canonical(m))
    def test_v5_identity_file_is_hash_bound(self):
        self.assertEqual(check_bundle(self.folder),self.manifest)
        (self.folder/"artifacts/application-identities.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"integrity"):
            check_bundle(self.folder)
    def test_descriptor_escape_or_mismatch_rejected(self):
        m=copy.deepcopy(self.manifest)
        self.manifest["dependencies"]["application_identities"]["file"]="../outside.json"
        self.save()
        with self.assertRaisesRegex(ValueError,"descriptor"):check_bundle(self.folder)
        self.manifest=m
        self.manifest["dependencies"]["application_identities"]["sha256"]="1"*64
        self.save()
        with self.assertRaisesRegex(ValueError,"descriptor"):check_bundle(self.folder)
    def test_resigned_malformed_stage_still_rejected(self):
        bad=copy.deepcopy(self.identities)
        bad["execution"]["stage"]="generation"
        from application_identity import _digest,STAGES
        for stage in STAGES:
            bad[stage]["sha256"]=_digest({k:v for k,v in bad[stage].items() if k!="sha256"})
        bad["manifest_sha256"]=_digest({k:v for k,v in bad.items() if k!="manifest_sha256"})
        path=self.folder/"artifacts/application-identities.json"
        path.write_bytes(canonical(bad))
        self.manifest["files"]["artifacts/application-identities.json"]=sha(path)
        self.manifest["dependencies"]["application_identities"]["sha256"]=sha(path)
        self.save()
        with self.assertRaisesRegex(ValueError,"stage format"):check_bundle(self.folder)
    def test_execution_restores_compact_public_vectors_before_hash_checks(self):
        from general_tensor_cases import models
        from candidate_contract import request_file_bytes
        from compiler_configuration import request_configuration
        from unified_graph_lowering import candidate_source
        from poseidon_runtime_paths import BUILD
        self.request=prepare_task(models()[1],dict(chunk_period=4,generation_guidance="explicit-v10"))
        source,_=candidate_source(self.request)
        (self.folder/"model.json").write_bytes(canonical(self.request["model"]))
        raw=request_file_bytes(self.request)
        self.assertNotEqual(canonical(json.loads(raw)),canonical(self.request))
        (self.folder/"request.json").write_bytes(raw)
        (self.folder/"candidate.py").write_text(source)
        self.manifest["origin"]["request_id"]=self.request["request_id"]
        self.manifest["compiler_configuration"]=self.request["compiler_configuration"]
        self.manifest["files"]={str(p.relative_to(self.folder)):sha(p) for p in self.folder.rglob("*")
                                if p.is_file() and p.name!="manifest.json"}
        self.save()
        actual_sha=sha
        with patch("poseidon_backend_identity.identity",return_value=self.identities["backend"]), \
             patch("application_identity.current_identity_manifest",return_value=self.identities), \
             patch("candidate_bundle.sha",side_effect=lambda p:"0"*64 if Path(p)==BUILD/"bin/hecate-opt" else actual_sha(p)):
            manifest,restored=check_execution_dependencies(self.folder)
        self.assertEqual(manifest,self.manifest)
        self.assertEqual(canonical(restored),canonical(self.request))
        request_configuration(restored)

    def test_v4_fields_remain_separate(self):
        m=self.manifest
        m["format"]="poseidon-dsl-bundle-v4"
        del m["dependencies"]["application_identities"]
        self.save()
        self.assertEqual(check_bundle(self.folder)["format"],"poseidon-dsl-bundle-v4")
        m["dependencies"]["application_identities"]={}
        self.save()
        with self.assertRaisesRegex(ValueError,"dependency fields"):check_bundle(self.folder)


    def include_generation(self):
        from application_generation import FILE
        from test_application_generation import receipt_fixture
        source=(self.folder/"candidate.py").read_text()
        receipt=receipt_fixture(self.request,source,self.identities["generation"]["sha256"])
        path=self.folder/FILE;path.write_bytes(canonical(receipt))
        self.manifest["files"][FILE]=sha(path)
        self.manifest["origin"]["generation_provenance"]=dict(file=FILE,sha256=sha(path))
        self.manifest["agent_generated"]=True
        self.save()
        return path
    def test_generation_receipt_is_separate_from_backend_validation(self):
        self.include_generation()
        value=check_bundle(self.folder)
        self.assertTrue(value["agent_generated"])
        self.assertTrue(value["validation"]["numerically_validated"])
        self.assertIn("generation_provenance",value["origin"])
    def test_changed_returned_source_is_rejected_after_resigning(self):
        self.include_generation()
        source=self.folder/"candidate.py"
        source.write_text(source.read_text().replace("x*x","x*x+zero_ct"))
        self.manifest["files"]["candidate.py"]=sha(source);self.save()
        with self.assertRaisesRegex(ValueError,"source identity"):
            check_bundle(self.folder)
    def test_unreferenced_generation_receipt_is_rejected(self):
        self.include_generation()
        del self.manifest["origin"]["generation_provenance"];self.save()
        with self.assertRaisesRegex(ValueError,"Unreferenced generation"):
            check_bundle(self.folder)
    def test_generation_flag_is_not_enough_to_override_receipt(self):
        self.include_generation()
        self.manifest["agent_generated"]=False;self.save()
        with self.assertRaisesRegex(ValueError,"generation descriptor"):
            check_bundle(self.folder)

    def test_v6_weight_package_keeps_old_formats_separate(self):
        from public_weight_cases import models,period
        from public_weight_context import POLICY
        from candidate_contract import request_file_bytes
        from unified_graph_lowering import candidate_source
        model=models()[-1]
        self.request=prepare_task(model,dict(chunk_period=period(model),public_weight_policy=POLICY,generation_guidance="explicit-v10"))
        source,_=candidate_source(self.request)
        (self.folder/"model.json").write_bytes(canonical(model))
        (self.folder/"request.json").write_bytes(request_file_bytes(self.request))
        (self.folder/"candidate.py").write_text(source)
        self.manifest["format"]="poseidon-dsl-bundle-v6"
        self.manifest["compiler_configuration"]=self.request["compiler_configuration"]
        self.manifest["origin"]["request_id"]=self.request["request_id"]
        self.manifest["files"]={str(p.relative_to(self.folder)):sha(p) for p in self.folder.rglob("*")
                                 if p.is_file() and p.name!="manifest.json"}
        self.save()
        self.assertEqual(check_bundle(self.folder)["format"],"poseidon-dsl-bundle-v6")
        self.manifest["format"]="poseidon-dsl-bundle-v5";self.save()
        with self.assertRaisesRegex(ValueError,"integrity|policy/version"):
            check_bundle(self.folder)

if __name__=="__main__":unittest.main()
