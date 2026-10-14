"""Source-origin bundle wiring with synthetic artifacts; no compiler or FHE."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application_identity import STAGES, _digest
from application_model_source import own_identity
from benchmark_graph import canonical, digest
from candidate_bundle import check_bundle, export_bundle, sha
from model_source_provenance import build, FILE, FORMAT
import test_staged_bundle as staged_fixture


class SourceBundleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = staged_fixture.StagedBundleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.folder = self.fixture.folder
        identities = self.fixture.identities
        identities["generation"]["sources"]["application_model_source.py"] = own_identity()["controller_sha256"]
        for stage in STAGES:
            identities[stage]["sha256"] = _digest({k:v for k,v in identities[stage].items() if k != "sha256"})
        identities["manifest_sha256"] = _digest({k:v for k,v in identities.items() if k != "manifest_sha256"})
        p = self.folder/"artifacts/application-identities.json"
        p.write_bytes(canonical(identities))
        self.fixture.manifest["files"][str(p.relative_to(self.folder))] = sha(p)
        self.fixture.manifest["dependencies"]["application_identities"]["sha256"] = sha(p)
        model = self.fixture.request["model"]
        raw = canonical(model)
        snapshot = dict(kind="graph",resource_profile="legacy-v1",binding="a"*64,
            files={"model.json":dict(bytes=len(raw),sha256=digest(model))},
            implementation=own_identity())
        self.origin = build(snapshot,self.fixture.request)
        self.fixture.save()

    def include_origin(self, value=None):
        value = self.origin if value is None else value
        path = self.folder/FILE
        path.write_bytes(canonical(value))
        h = sha(path)
        self.fixture.manifest["files"][FILE] = h
        self.fixture.manifest["origin"]["model_source"] = dict(format=FORMAT,file=FILE,sha256=h)
        self.fixture.save()

    def changed_origin(self, **changes):
        value = dict(copy.deepcopy(self.origin),**changes)
        value["binding"] = digest({k:v for k,v in value.items() if k != "binding"})
        return value

    def test_full_manifest_reload_keeps_origin_separate_from_generation(self):
        self.include_origin()
        manifest = check_bundle(self.folder)
        self.assertFalse(manifest["agent_generated"])
        self.assertNotIn("generation_provenance",manifest["origin"])
        self.assertEqual(json.loads((self.folder/FILE).read_text()),self.origin)
        self.assertEqual(check_bundle(self.folder),manifest)

    def test_resigned_source_for_different_model_rejected(self):
        self.include_origin(self.changed_origin(model_sha256="b"*64))
        with self.assertRaisesRegex(ValueError,"model/provenance"):
            check_bundle(self.folder)

    def test_resigned_different_source_controller_rejected(self):
        changed = dict(self.origin["implementation"],controller_sha256="0"*64)
        self.include_origin(self.changed_origin(implementation=changed))
        with self.assertRaisesRegex(ValueError,"controller/dependency"):
            check_bundle(self.folder)

    def test_deleted_descriptor_and_tampered_origin_rejected(self):
        self.include_origin()
        (self.folder/FILE).write_text("{}")
        with self.assertRaisesRegex(ValueError,"integrity"):
            check_bundle(self.folder)
        self.include_origin()
        del self.fixture.manifest["origin"]["model_source"]
        self.fixture.save()
        with self.assertRaisesRegex(ValueError,"Unreferenced"):
            check_bundle(self.folder)

    def test_pre_source_packages_are_optional_and_v4_cannot_claim_intake(self):
        self.assertEqual(check_bundle(self.folder),self.fixture.manifest)
        self.include_origin()
        self.fixture.manifest["format"]="poseidon-dsl-bundle-v4"
        del self.fixture.manifest["dependencies"]["application_identities"]
        self.fixture.save()
        with self.assertRaisesRegex(ValueError,"source descriptor"):
            check_bundle(self.folder)

    def export_fixture(self, value=None):
        base = self.folder/"synthetic"
        base.mkdir()
        evidence = base/"evidence";evidence.mkdir()
        attempt = evidence/"attempt-00";attempt.mkdir()
        out = attempt/"output";out.mkdir()
        request = self.fixture.request
        (evidence/"request.json").write_bytes(canonical(request))
        (evidence/"model.json").write_bytes(canonical(request["model"]))
        (attempt/"candidate.py").write_bytes((self.folder/"candidate.py").read_bytes())
        for name in ("candidate_trace.mlir","lowered.earth.mlir","lowered.ckks.mlir",
                     "lowered._hecate_golden.runtime-plan.json","candidate_trace.entry.mlir",
                     "entry-selection.json","candidate_trace.compact.mlir"):
            (out/name).write_text("synthetic fixture; never executed\n")
        sdk = base/"sdk";(sdk/"bin").mkdir(parents=True)
        compiler = sdk/"bin/hecate-opt";compiler.write_bytes(b"non-executable synthetic fixture")
        spec = sdk/"spec.json";spec.write_text("{}")
        profile = sdk/"profile.json";profile.write_text("{}")
        report = dict(backend="poseidon_cpu",compiler_sha256=sha(compiler),
            frontend_sha256="1"*64,runtime_sha256="2"*64,platform_identity=dict(id="fixture-only"),
            source_hashes={},backend_identity=self.fixture.identities["backend"],
            attempts=[dict(index=0,status="passed",artifact_gate={"fixture":True},
                           runtime_plan_entry="fixture",compile_command=[])])
        (evidence/"report.json").write_bytes(canonical(report))
        output = base/"program"
        # Synthetic verifier and tools isolate serialization wiring. These
        # fixtures provide no tracing, compiled or encrypted acceptance evidence.
        with patch("workspace_paths.RESULTS",base), \
             patch("audit_unified_candidate.verify_candidate",return_value={"model_sha256":digest(request["model"])}), \
             patch("poseidon_runtime_paths.BUILD",sdk), \
             patch("poseidon_runtime_paths.SPEC",spec), \
             patch("poseidon_runtime_paths.PROFILE",profile), \
             patch("application_identity.current_identity_manifest",return_value=self.fixture.identities):
            manifest = export_bundle(evidence,output,validation_level="compiled",
                application_identities=self.fixture.identities,model_source=self.origin if value is None else value)
        return output,manifest

    def test_export_and_reload_bind_source_record(self):
        output,manifest=self.export_fixture()
        self.assertEqual(manifest["format"],"poseidon-dsl-bundle-v5")
        self.assertEqual(check_bundle(output),manifest)
        self.assertEqual(json.loads((output/FILE).read_text()),self.origin)
        self.assertEqual(manifest["validation"],dict(level="compiled",encrypted_execution=False,numerically_validated=False))
        self.assertFalse(manifest["agent_generated"])

    def test_export_rejects_source_from_different_controller(self):
        value=self.changed_origin(implementation=dict(self.origin["implementation"],controller_sha256="0"*64))
        with self.assertRaisesRegex(ValueError,"controller/dependency"):
            self.export_fixture(value)


if __name__ == "__main__":
    unittest.main()
