"""Source package routing controls; no real capture, Provider or FHE."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmark_graph import canonical
from pytorch_source_package import SourcePackageError
from pytorch_source_package_policy import (SOURCE_FORMAT,PACKAGE_FORMAT,REFERENCE_FORMAT,
                                          CAPTURE_FORMAT,mount_directories)
from pytorch_ordered_contract import prepare,validate_package,package_version,reference_version
from pytorch_spatial_policy import supports_input_call,supports_spatial
from pytorch_static_broadcast import supports_source
sys.path.insert(0,str(Path(__file__).parents[1]))
import prepare_pytorch_source as cli

CALL=dict(format="poseidon-pytorch-call-v1",args=[dict(kind="tensor",name="x")],kwargs={})
APPROX=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[])


def manifest(files,entry="models.network.build"):
    return dict(format=SOURCE_FORMAT,files={name:hashlib.sha256(raw).hexdigest()for name,raw in files.items()},
                entry=entry,inputs=[dict(name="x",shape=[2,2])],outputs=["out"],
                factory=dict(args=[],kwargs={"gain":0.25}),dtype_policy="strict-float64-v1",
                weights={},approximations=copy.deepcopy(APPROX),call=copy.deepcopy(CALL))


def sources():
    return {"models/__init__.py":b"",
            "models/layers.py":b"from torch import nn\nclass Block(nn.Module):\n    def forward(self,x): return x.square()\n",
            "models/network.py":b"from torch import nn\nfrom .layers import Block\nclass Model(nn.Module):\n    def __init__(self,gain):\n        super().__init__()\n        self.block=Block()\n        self.gain=gain\n    def forward(self,x): return self.block(x)+self.gain\ndef build(gain=0.25): return Model(gain)\n"}


class PackageIntakeControls(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.files=sources()
        self.spec=manifest(self.files)
        self.path=self.root/"manifest.json"
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        for name,raw in self.files.items():
            target=self.root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(raw)
        self.path.write_bytes(canonical(self.spec))

    def test_full_manifest_prepares_nested_source_paths(self):
        checked,paths=prepare(self.path)
        self.assertEqual(checked,self.spec)
        self.assertEqual(set(paths),set(self.files))
        for name,path in paths.items():
            self.assertEqual(path,self.root/name)
            self.assertEqual(path.read_bytes(),self.files[name])

    def test_source_text_and_original_bytes_validate_identically(self):
        self.assertEqual(validate_package(self.spec,self.files),self.spec)
        self.assertEqual(validate_package(self.spec,{name:raw.decode()for name,raw in self.files.items()}),self.spec)

    def test_source_package_has_own_provenance_and_reference_versions(self):
        from pytorch_model_intake import package_format,ordered
        from pytorch_public_expression_capture import capture_format
        self.assertTrue(ordered(self.spec))
        self.assertEqual(package_version(self.spec),PACKAGE_FORMAT)
        self.assertEqual(package_format(self.spec),PACKAGE_FORMAT)
        self.assertEqual(reference_version(self.spec),REFERENCE_FORMAT)
        self.assertEqual(capture_format(self.spec),CAPTURE_FORMAT)

    def test_package_inherits_structured_spatial_and_broadcast_computation(self):
        for check in(supports_input_call,supports_spatial,supports_source):
            self.assertTrue(check(self.spec))
            self.assertFalse(check({"format":"poseidon-trusted-pytorch-v999"}))

    def test_missing_call_does_not_use_flat_fallback(self):
        self.spec.pop("call")
        with self.assertRaises(ValueError):
            validate_package(self.spec,self.files)

    def test_legacy_v28_still_rejects_package_paths(self):
        self.spec["format"]="poseidon-trusted-pytorch-v28"
        with self.assertRaises(ValueError):
            validate_package(self.spec,self.files)

    def test_source_hash_change_after_manifest_is_rejected(self):
        (self.root/"models/network.py").write_bytes(b"def build(): return 1\n")
        with self.assertRaisesRegex(SourcePackageError,"source_hash_mismatch"):
            prepare(self.path)

    def test_unlisted_parent_initializer_is_rejected(self):
        self.spec["files"].pop("models/__init__.py")
        with self.assertRaisesRegex(SourcePackageError,"source_package_parent_missing"):
            validate_package(self.spec,{name:raw for name,raw in self.files.items()if name!="models/__init__.py"})

    def test_internal_adapter_root_cannot_be_supplied_by_model(self):
        files={"pytorch_ordered_intake.py":b"def build(): pass\n"}
        with self.assertRaisesRegex(SourcePackageError,"source_module_reserved"):
            validate_package(manifest(files,"pytorch_ordered_intake.build"),files)

    def test_json_rule_files_remain_hashed_without_becoming_module_roots(self):
        from pytorch_ordered_contract import source_package_reserved_roots,source_identity
        from pytorch_source_package import identifier
        roots=source_package_reserved_roots()
        self.assertTrue(all(identifier(value)for value in roots))
        self.assertIn("pytorch_ordered_intake",roots)
        self.assertNotIn("platform-profiles",roots)
        self.assertIn("platform-profiles.json",source_identity())

    def test_manifest_factory_and_float_rules_remain_enforced(self):
        for change in({"factory":{"args":[],"kwargs":{"not valid":1}}},
                      {"dtype_policy":"accept-anything"},{"unexpected":True}):
            spec=copy.deepcopy(self.spec);spec.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_package(spec,self.files)

    def test_parent_symlink_is_rejected_before_capture(self):
        actual=self.root/"actual"
        (self.root/"models").rename(actual)
        (self.root/"models").symlink_to(actual,target_is_directory=True)
        with self.assertRaisesRegex(SourcePackageError,"source_path_link"):
            prepare(self.path)

    def test_mount_arguments_create_parent_directories_in_order(self):
        nested={"models/__init__.py":b"","models/nested/__init__.py":b"",
                "models/nested/network.py":b"def build(): pass\n"}
        self.assertEqual(mount_directories(manifest(nested,"models.nested.network.build")),
                         ["--dir","/model/models","--dir","/model/models/nested"])
        self.assertEqual(mount_directories(manifest({"model.py":b""},"model.build")),[])

    def test_mount_arguments_reject_path_escape(self):
        self.spec["files"]={"../../private.py":"0"*64}
        with self.assertRaises(SourcePackageError):
            mount_directories(self.spec)

    def test_source_builder_handles_empty_initializer_and_explicit_root(self):
        spec,root=cli.build(self.root/"models/network.py",entry=self.spec["entry"],inputs=["x:2,2"],
            outputs=["out"],helpers=[self.root/"models/__init__.py",self.root/"models/layers.py"],
            source_format=SOURCE_FORMAT,source_root=self.root,call=CALL,factory_kwargs={"gain":0.25})
        self.assertEqual(root,self.root)
        self.assertEqual(canonical(spec),canonical(self.spec))

    def test_cli_save_keeps_manifest_in_package_root(self):
        spec,root=cli.build(self.root/"models/network.py",entry=self.spec["entry"],inputs=["x:2,2"],
            outputs=["out"],helpers=[self.root/"models/__init__.py",self.root/"models/layers.py"],
            source_format=SOURCE_FORMAT,source_root=self.root,call=CALL,factory_kwargs={"gain":0.25})
        output=cli.save(spec,self.root/"new.json",root)
        self.assertEqual(json.loads(output.read_text()),self.spec)

    def test_builder_rejects_helpers_outside_declared_root(self):
        outside=self.root/"other.py";outside.write_bytes(b"")
        with self.assertRaises(ValueError):
            cli.build(self.root/"models/network.py",entry=self.spec["entry"],inputs=["x:2,2"],outputs=["out"],
                helpers=[outside],source_format=SOURCE_FORMAT,source_root=self.root/"models",call=CALL)

    def test_source_root_is_not_silently_enabled_for_legacy_version(self):
        flat=self.root/"model.py";flat.write_bytes(b"def build(): pass\n")
        with self.assertRaisesRegex(ValueError,"Source root requires"):
            cli.build(flat,entry="model.build",inputs=["x:2,2"],outputs=["out"],
                      source_format="poseidon-trusted-pytorch-v28",source_root=self.root,call=CALL)

    def test_source_package_builder_requires_all_parent_initializers(self):
        with self.assertRaisesRegex(SourcePackageError,"source_package_parent_missing"):
            cli.build(self.root/"models/network.py",entry=self.spec["entry"],inputs=["x:2,2"],outputs=["out"],
                      source_format=SOURCE_FORMAT,source_root=self.root,call=CALL)

    def test_default_and_latest_keep_their_prior_contracts(self):
        self.assertEqual(cli.DEFAULT_FORMAT,"poseidon-trusted-pytorch-v18")
        self.assertEqual(cli.resolve_source_format("latest"),"poseidon-trusted-pytorch-v34")
        self.assertEqual(cli.resolve_source_format("poseidon-trusted-pytorch-v35"),"poseidon-trusted-pytorch-v35")
        self.assertEqual(cli.resolve_source_format(SOURCE_FORMAT),SOURCE_FORMAT)

    def test_v28_flat_manifest_bytes_are_unchanged(self):
        raw=b"from torch import nn\ndef build(gain=0.25): return nn.Identity()\n"
        flat=self.root/"model.py";flat.write_bytes(raw)
        expected=manifest({"model.py":raw},"model.build");expected["format"]="poseidon-trusted-pytorch-v28"
        actual,_=cli.build(flat,entry="model.build",inputs=["x:2,2"],outputs=["out"],
                          source_format="poseidon-trusted-pytorch-v28",call=CALL,factory_kwargs={"gain":0.25})
        self.assertEqual(canonical(actual),canonical(expected))

    def test_factory_loader_resolves_full_dotted_module_once(self):
        import torch
        import trusted_pytorch_import as legacy
        from pytorch_ordered_contract import source_identity
        from pytorch_ordered_intake import capture_request
        request=self.root/"request.json"
        request.write_bytes(canonical(dict(manifest=self.spec,resource_profile="general-tensors-v1",
                                           runtime=legacy.runtime_identity(),source_hashes=source_identity())))
        factory=unittest.mock.Mock(side_effect=lambda gain:torch.nn.Identity())
        module=types.SimpleNamespace(build=factory)
        with patch("importlib.import_module",return_value=module)as importer,\
             patch("pytorch_public_expression_capture.capture_loaded_module",return_value="capture_stub")as capture:
            result=capture_request(request,self.root,self.root/"output")
        self.assertEqual(result,"capture_stub")
        importer.assert_called_once_with("models.network")
        factory.assert_called_once_with(gain=0.25)
        self.assertEqual(capture.call_count,1)

    def test_application_snapshot_preserves_nested_bytes_without_execution(self):
        from application_model_source import _snapshot,FORMAT
        source=dict(format=FORMAT,kind="trusted_pytorch",manifest=str(self.path),trust_local_code=True)
        kind,files=_snapshot(source,"general-tensors-v1")
        self.assertEqual(kind,"trusted_pytorch")
        self.assertEqual(files["manifest.json"],canonical(self.spec))
        for name,raw in self.files.items():self.assertEqual(files[name],raw)

    def test_task_submission_creates_nested_snapshot_without_provider(self):
        from test_application_model_source import Backend,SourceApplication
        from application_model_source import FORMAT
        backend=Backend();app=SourceApplication(self.root/"tasks",backend)
        task=app.submit_source(dict(format=FORMAT,kind="trusted_pytorch",manifest=str(self.path),
                                   trust_local_code=True),resource_profile="general-tensors-v1")
        state=app.status(task)
        self.assertEqual((state["status"],state["phase"]),("queued","intake_queued"))
        self.assertEqual((state["generations"],state["http_attempts"],backend.calls),(0,0,0))
        for name,raw in self.files.items():self.assertEqual((app._dir(task)/"source"/name).read_bytes(),raw)

    def test_capabilities_describe_registration_without_numerical_claim(self):
        from application_capabilities import input_sources
        trusted=input_sources()["trusted_pytorch"]
        value=trusted["source_package"]
        self.assertEqual(value["version"],SOURCE_FORMAT)
        self.assertEqual((value["files"],value["bytes"],value["path_depth"]),(8,65536,8))
        self.assertEqual(value["input_status"],"registered")
        for stage in("compiler","encrypted","agent"):self.assertEqual(value[stage+"_status"],"unverified")
        for group in("input_call","spatial_calls","static_broadcast"):
            self.assertIn(SOURCE_FORMAT,trusted[group]["versions"])

    def test_provenance_reader_retains_paths_and_hashes(self):
        from candidate_bundle import load_model_provenance
        from pytorch_weight_package import binding
        from benchmark_suite import Builder
        b=Builder([(2,2)]);model=b.finish(b.node("square",["input0"]))
        original={}
        bound=binding({}, {})
        report={"capture_metadata":{"weight_binding":bound}}
        target=self.root/"import";target.mkdir()
        for name,value in(("original.json",original),("decomposition.json",{}),("reference-report.json",report)):
            (target/name).write_bytes(canonical(value))
        retained=target/"trusted-python";retained.mkdir()
        (retained/"manifest.json").write_bytes(canonical(self.spec))
        for name,raw in self.files.items():
            path=retained/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        with patch("candidate_bundle.check_provenance")as verify:
            result=load_model_provenance(target,model)
        verify.assert_called_once()
        package=result["trusted_pytorch"]
        self.assertEqual(package["format"],PACKAGE_FORMAT)
        self.assertEqual(package["manifest"],self.spec)
        self.assertEqual(package["files"],{name:raw.decode()for name,raw in self.files.items()})
        self.assertEqual(package["weight_files"],{})


if __name__=="__main__":
    unittest.main()
