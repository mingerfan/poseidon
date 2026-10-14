"""Source package preparation controls; uploaded Python is never imported."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytorch_source_package as package


def specification(files, entry="model.build"):
    return dict(format=package.FORMAT,
                files={name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
                entry=entry)


def model_files():
    return {"model.py": b"import torch\nfrom torch import nn\n\ndef build():\n    return nn.Identity()\n"}


class PackageControls(unittest.TestCase):
    def error(self, reason, function, *args, **kwargs):
        with self.assertRaises(package.SourcePackageError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def validate(self, files, entry="model.build", **kwargs):
        return package.validate(specification(files, entry), files, **kwargs)

    def test_flat_source_is_prepared_without_factory_execution(self):
        record = self.validate(model_files())
        self.assertEqual(record["entry"], {"module": "model", "factory": "build"})
        self.assertEqual(record["status"], "prepared_not_executed")
        for key in ("factory_execution", "graph_capture", "backend_execution"):
            self.assertEqual(record[key], "not_run")

    def test_inert_user_source_is_not_imported(self):
        files = {"model.py": b"import torch\nraise RuntimeError('this must never execute')\ndef build(): pass\n"}
        original_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in ("model", "torch"):
                raise AssertionError("user source or external import")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded_import), \
                patch("importlib.import_module", side_effect=AssertionError("dynamic source import")):
            record = self.validate(files)
        self.assertEqual(record["status"], "prepared_not_executed")

    def test_regular_package_supports_empty_initializer_and_relative_helper(self):
        files = {"models/__init__.py": b"",
                 "models/layers.py": b"class Block: pass\n",
                 "models/network.py": b"from .layers import Block\ndef build(): return Block()\n"}
        record = self.validate(files, "models.network.build")
        self.assertEqual(record["source_files"]["models/__init__.py"]["bytes"], 0)
        self.assertEqual(record["modules"]["models"], {"file": "models/__init__.py", "kind": "package"})
        self.assertEqual(record["imports"][0]["module"], "models.layers")
        self.assertEqual(record["imports"][0]["category"], "local")

    def test_nested_package_relative_parent_import(self):
        files = {"models/__init__.py": b"", "models/layers.py": b"class Block: pass\n",
                 "models/cnn/__init__.py": b"",
                 "models/cnn/network.py": b"from ..layers import Block\ndef build(): return Block()\n"}
        record = self.validate(files, "models.cnn.network.build")
        self.assertEqual(record["imports"][0]["module"], "models.layers")

    def test_package_initializer_can_declare_reexported_factory(self):
        files = {"models/__init__.py": b"from .network import build\n",
                 "models/network.py": b"def build(): pass\n"}
        record = self.validate(files, "models.build")
        self.assertEqual(record["entry"]["module"], "models")
        self.assertEqual(record["factory_execution"], "not_run")

    def test_from_package_import_and_alias_are_retained(self):
        files = {"models/__init__.py": b"", "models/layers.py": b"class Block: pass\n",
                 "models/network.py": b"from . import layers as local\nimport models.layers as layers\ndef build(): pass\n"}
        record = self.validate(files, "models.network.build")
        imports = record["imports"]
        self.assertEqual(imports[0]["module"], "models")
        self.assertEqual(imports[0]["alias"], {"name": "layers", "asname": "local"})
        self.assertEqual(imports[1]["module"], "models.layers")

    def test_private_helper_module_is_valid(self):
        files = {"_helpers.py": b"def scale(x): return x\n",
                 "model.py": b"from _helpers import scale\ndef build(): pass\n"}
        self.assertEqual(self.validate(files)["imports"][0]["module"], "_helpers")

    def test_registered_external_imports_are_recorded_not_executed(self):
        files = {"model.py": b"import torch.nn as nn\nimport numpy as np\nfrom math import sqrt\ndef build(): pass\n"}
        record = self.validate(files)
        self.assertEqual({row["module"] for row in record["imports"]}, {"torch.nn", "numpy", "math"})
        self.assertEqual({row["category"] for row in record["imports"]}, {"registered_external"})

    def test_source_order_does_not_change_binding(self):
        files = {"model.py": b"from helper import Block\ndef build(): pass\n",
                 "helper.py": b"class Block: pass\n"}
        first = self.validate(files)
        reverse = dict(reversed(tuple(files.items())))
        self.assertEqual(first, self.validate(reverse))

    def test_weightless_source_bytes_are_retained_exactly(self):
        files = {"model.py": b"\xef\xbb\xbf# utf-8\n# \xe4\xb8\xad\xe6\x96\x87\ndef build(): pass\n"}
        record = self.validate(files)
        self.assertEqual(record["source_files"]["model.py"]["sha256"], hashlib.sha256(files["model.py"]).hexdigest())

    def test_source_change_changes_binding(self):
        first = self.validate({"model.py": b"def build(): pass\n"})
        second = self.validate({"model.py": b"def build(): pass\n# changed\n"})
        self.assertNotEqual(first["binding"], second["binding"])

    def test_manifest_or_source_inventory_mismatch_is_rejected(self):
        files = model_files()
        for raw in ({}, {**files, "helper.py": b""}):
            with self.subTest(inventory=tuple(raw)):
                self.error("source_inventory_mismatch", package.validate, specification(files), raw)

    def test_hash_mismatch_is_rejected(self):
        files = model_files()
        spec = specification(files)
        spec["files"]["model.py"] = "0" * 64
        self.error("source_hash_mismatch", package.validate, spec, files)

    def test_invalid_hash_and_nonbyte_source_are_rejected(self):
        files = model_files()
        spec = specification(files)
        spec["files"]["model.py"] = "ABC"
        self.error("source_hash_invalid", package.validate, spec, files)
        self.error("source_bytes_invalid", package.validate, specification(files), {"model.py": "source"})

    def test_manifest_requires_exact_fields_and_version(self):
        for changed in ({"format": "future"}, {"weights": {}}, {"entry": None}):
            spec = specification(model_files())
            spec.update(changed)
            reason = "source_entry_invalid" if changed.get("entry", "valid") is None else "source_package_fields"
            with self.subTest(change=changed):
                self.error(reason, package.validate, spec, model_files())

    def test_file_count_and_empty_inventory_are_bounded(self):
        for files in ({}, {f"mod{index}.py": b"" for index in range(9)}):
            with self.subTest(count=len(files)):
                self.error("source_file_capacity", package.validate, specification(files), files)

    def test_byte_limit_accepts_boundary_and_rejects_one_extra_byte(self):
        prefix = b"def build(): pass\n#"
        files = {"model.py": prefix + b"x" * (package.MAX_BYTES - len(prefix))}
        self.assertEqual(self.validate(files)["counts"]["bytes"], package.MAX_BYTES)
        files["model.py"] += b"x"
        self.error("source_byte_capacity", package.validate, specification(files), files)

    def test_ast_budget_is_checked_without_factory_execution(self):
        files = {"model.py": b"x=1\n" * 8200 + b"def build(): pass\n"}
        self.assertLess(len(files["model.py"]), package.MAX_BYTES)
        self.error("source_ast_capacity", package.validate, specification(files), files)

    def test_invalid_syntax_and_null_source_are_structured(self):
        for raw in (b"def build(:\n", b"x=\x00\n"):
            with self.subTest(raw=raw):
                self.error("source_syntax_invalid", package.validate, specification({"model.py": raw}), {"model.py": raw})

    def test_non_utf8_encoding_and_invalid_bytes_are_rejected(self):
        for raw, reason in ((b"# coding: latin-1\n#\xff\n", "source_encoding_unsupported"),
                            (b"# coding: utf-8\n#\xff\n", "source_encoding_invalid")):
            with self.subTest(reason=reason):
                self.error(reason, package.validate, specification({"model.py": raw}), {"model.py": raw})

    def test_absolute_traversal_and_noncanonical_paths_are_rejected(self):
        for name in ("/secret/model.py", "../model.py", "pkg/../model.py", "./model.py",
                     "pkg//model.py", "pkg/./model.py", r"pkg\model.py"):
            with self.subTest(name=name):
                self.error("source_path_invalid", package.validate, specification({name: b""}), {name: b""})

    def test_nonpython_keyword_and_invalid_module_names_are_rejected(self):
        for name in ("model.txt", "class.py", "pkg-name/model.py", "1model.py", "模型.py", "pkg/class/model.py"):
            with self.subTest(name=name):
                self.error("source_path_invalid", package.validate, specification({name: b""}), {name: b""})

    def test_dunder_paths_are_rejected_but_initializer_is_allowed(self):
        for name in ("__secret.py", "__cache/model.py"):
            with self.subTest(name=name):
                self.error("source_path_reserved", package.validate, specification({name: b""}), {name: b""})

    def test_root_initializer_has_no_implicit_package_name(self):
        files = {"__init__.py": b""}
        self.error("source_root_initializer_unbound", package.validate, specification(files), files)

    def test_path_depth_and_identifier_length_are_bounded(self):
        for name in ("/".join(["pkg"] * 8 + ["model.py"]), "x" * 65 + ".py"):
            with self.subTest(name=name):
                self.error("source_path_invalid", package.validate, specification({name: b""}), {name: b""})

    def test_package_parent_must_be_explicit_and_not_a_plain_module(self):
        variants = [
            {"models/network.py": b"def build(): pass\n"},
            {"models.py": b"", "models/network.py": b"def build(): pass\n"}]
        for files in variants:
            with self.subTest(files=tuple(files)):
                self.error("source_package_parent_missing", package.validate, specification(files, "models.network.build"), files)

    def test_module_and_package_identity_collision_is_rejected(self):
        files = {"models.py": b"", "models/__init__.py": b""}
        self.error("source_module_collision", package.validate, specification(files, "models.build"), files)

    def test_stdlib_and_backend_modules_cannot_be_shadowed(self):
        for name in ("torch.py", "numpy.py", "math.py", "ast.py", "hecate.py", "poly.py", "mlir.py", "einops.py"):
            with self.subTest(name=name):
                self.error("source_module_reserved", package.validate, specification({name: b""}, name[:-3] + ".build"), {name: b""})

    def test_caller_sdk_reserved_roots_are_honored(self):
        files = {"candidate_worker.py": b"def build(): pass\n"}
        self.error("source_module_reserved", package.validate, specification(files, "candidate_worker.build"),
                   files, reserved_module_roots=("candidate_worker",))

    def test_entry_is_explicit_dotted_and_bounded(self):
        files = model_files()
        for entry in ("model", "model.", ".build", "model.__dict__", "model.class", "x." * 300 + "build"):
            with self.subTest(entry_length=len(entry)):
                self.error("source_entry_invalid", package.validate, specification(files, entry), files)
        self.error("source_entry_module_missing", package.validate, specification(files, "missing.build"), files)

    def test_unlisted_local_import_is_rejected(self):
        files = {"model.py": b"import models.unlisted\ndef build(): pass\n",
                 "models/__init__.py": b""}
        self.error("source_import_not_declared", package.validate, specification(files), files)

    def test_unregistered_external_import_is_rejected(self):
        files = {"model.py": b"import local_not_listed\ndef build(): pass\n"}
        self.error("source_import_not_registered", package.validate, specification(files), files)

    def test_relative_import_cannot_escape_package(self):
        for files, entry in (
            ({"model.py": b"from .helper import Block\ndef build(): pass\n"}, "model.build"),
            ({"models/__init__.py": b"", "models/network.py": b"from ..helper import Block\ndef build(): pass\n"},
             "models.network.build")):
            with self.subTest(entry=entry):
                self.error("source_relative_import_escape", package.validate, specification(files, entry), files)

    def test_dynamic_import_and_direct_execution_calls_are_rejected(self):
        for expression in ("__import__('torch')", "eval('1')", "exec('x=1')", "compile('1','x','eval')",
                           "open('secret')", "input()"):
            files = {"model.py": (expression + "\ndef build(): pass\n").encode()}
            with self.subTest(expression=expression):
                error = self.error("source_dynamic_execution", package.validate, specification(files), files)
                self.assertEqual(error.location["file"], "model.py")
                self.assertEqual(error.location["line"], 1)

    def test_forbidden_external_roots_cannot_be_opted_in(self):
        for root in ("os", "sys", "socket", "importlib", "ctypes", "pickle"):
            with self.subTest(root=root):
                self.error("source_external_roots_forbidden", package.validate, specification(model_files()),
                           model_files(), external_roots=(root,))

    def test_error_omits_source_body_and_absolute_path(self):
        secret = b"raise ValueError('PRIVATE_REFERENCE_123')\nx=\x00\n"
        error = self.error("source_syntax_invalid", package.validate,
                           specification({"model.py": secret}), {"model.py": secret})
        public = str(error) + json.dumps(error.location) + json.dumps(error.context)
        self.assertNotIn("PRIVATE_REFERENCE_123", public)
        self.assertNotIn(str(Path.cwd()), public)

    def test_preparation_is_not_advertised_as_active_capture(self):
        from pytorch_model_intake import SOURCE_FORMATS
        self.assertNotIn(package.FORMAT, SOURCE_FORMATS)
        self.assertIn("poseidon-trusted-pytorch-v28", SOURCE_FORMATS)
        script = Path(__file__).parents[1]/"prepare_pytorch_source.py"
        latest = next(node.value.value for node in ast.parse(script.read_text()).body
                      if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == "LATEST_FORMAT" for target in node.targets))
        self.assertEqual(latest, "poseidon-trusted-pytorch-v34")


DEFAULT_ROOT = object()


class PackageReader(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.files = {"models/__init__.py": b"",
                      "models/layers.py": b"class Block: pass\n",
                      "models/network.py": b"from .layers import Block\ndef build(): return Block()\n"}
        self.spec = specification(self.files, "models.network.build")
        for name, raw in self.files.items():
            path = self.root/name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(raw)

    def tearDown(self):
        self.temporary.cleanup()

    def error(self, reason, root=DEFAULT_ROOT):
        with self.assertRaises(package.SourcePackageError) as caught:
            package.prepare(self.root if root is DEFAULT_ROOT else root, self.spec)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def test_explicit_files_are_read_and_hashed_without_source_execution(self):
        record, files = package.prepare(self.root, self.spec)
        self.assertEqual(files, self.files)
        self.assertEqual(record, package.validate(self.spec, self.files))
        self.assertEqual(record["backend_execution"], "not_run")

    def test_unlisted_file_is_not_read(self):
        (self.root/"secret.py").write_text("PRIVATE_ENV_VALUE")
        record, files = package.prepare(self.root, self.spec)
        self.assertNotIn("secret.py", files)
        self.assertEqual(record["counts"]["files"], 3)

    def test_empty_initializer_is_permitted(self):
        record, _ = package.prepare(self.root, self.spec)
        self.assertEqual(record["source_files"]["models/__init__.py"]["bytes"], 0)

    def test_final_symlink_is_rejected(self):
        target = self.root/"models/network.py"
        target.unlink()
        target.symlink_to(self.root/"models/layers.py")
        error = self.error("source_path_link")
        self.assertEqual(error.location["file"], "models/network.py")

    def test_parent_directory_symlink_is_rejected(self):
        original = self.root/"models"
        renamed = self.root/"actual"
        original.rename(renamed)
        original.symlink_to(renamed, target_is_directory=True)
        self.error("source_path_link")

    def test_root_symlink_is_rejected(self):
        link = self.root/"root-link"
        link.symlink_to(self.root, target_is_directory=True)
        error = self.error("source_root_unavailable", link)
        self.assertIsNone(error.location)

    def test_hardlinked_source_is_rejected(self):
        os.link(self.root/"models/network.py", self.root/"other.py")
        self.error("source_file_type")

    def test_fifo_does_not_block_or_read(self):
        path = self.root/"models/network.py"
        path.unlink()
        os.mkfifo(path)
        self.error("source_file_type")

    def test_directory_instead_of_source_is_rejected(self):
        path = self.root/"models/network.py"
        path.unlink()
        path.mkdir()
        self.error("source_file_type")

    def test_missing_source_has_relative_diagnostic(self):
        (self.root/"models/network.py").unlink()
        error = self.error("source_file_unavailable")
        self.assertEqual(error.location["file"], "models/network.py")

    def test_invalid_root_is_structured_and_private(self):
        for root in (None, "", "/private\0value", self.root/"not-present"):
            with self.subTest(root_type=type(root).__name__):
                error = self.error("source_root_unavailable" if isinstance(root, Path) else "source_root_invalid", root)
                self.assertNotIn("private", str(error))
                self.assertIsNone(error.location)

    def test_changed_hash_is_rejected_after_secure_read(self):
        (self.root/"models/network.py").write_bytes(b"def build(): return 2\n")
        self.error("source_hash_mismatch")

    def test_files_over_byte_limit_are_rejected_before_read(self):
        (self.root/"models/network.py").write_bytes(b"x" * (package.MAX_BYTES + 1))
        self.error("source_byte_capacity")

    def test_mutation_during_read_is_rejected(self):
        original_read = package.os.read
        target = self.root/"models/network.py"
        changed = False
        def read(descriptor, count):
            nonlocal changed
            raw = original_read(descriptor, count)
            info = os.fstat(descriptor)
            if info.st_ino == target.stat().st_ino and not changed:
                changed = True
                with target.open("ab") as stream:
                    stream.write(b"# changed\n")
            return raw
        with patch.object(package.os, "read", side_effect=read):
            self.error("source_changed_during_read")
        self.assertTrue(changed)

    def test_descriptors_are_closed_after_success_and_rejection(self):
        def descriptors():
            return set(os.listdir("/proc/self/fd"))
        before = descriptors()
        package.prepare(self.root, self.spec)
        self.assertEqual(descriptors(), before)
        os.link(self.root/"models/network.py", self.root/"linked.py")
        self.error("source_file_type")
        self.assertEqual(descriptors(), before)


class PackageSnapshot(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root/"snapshot"
        self.files = {"models/__init__.py": b"",
                      "models/layers.py": b"\xef\xbb\xbfclass Block: pass\n",
                      "models/network.py": b"from .layers import Block\ndef build(): return Block()\n"}
        self.spec = specification(self.files, "models.network.build")
        self.record = package.validate(self.spec, self.files)
        self.binding = self.record["binding"]

    def tearDown(self):
        self.temporary.cleanup()

    def freeze(self):
        return package.snapshot(self.destination, self.spec, self.files, expected_binding=self.binding)

    def reload(self):
        return package.read_snapshot(self.destination, expected_binding=self.binding)

    def error(self, reason, function, *args, **kwargs):
        with self.assertRaises(package.SourcePackageError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception

    def test_nested_package_roundtrip_preserves_original_bytes_and_preparation_only(self):
        written = self.freeze()
        reloaded, files = self.reload()
        self.assertEqual(written, self.record)
        self.assertEqual(reloaded, self.record)
        self.assertEqual(files, self.files)
        self.assertEqual(reloaded["factory_execution"], "not_run")

    def test_binding_change_is_rejected_before_creating_snapshot(self):
        files = dict(self.files)
        files["models/network.py"] += b"# changed\n"
        self.error("source_preparation_binding_mismatch", package.snapshot, self.destination,
                   specification(files, "models.network.build"), files, expected_binding=self.binding)
        self.assertFalse(self.destination.exists())

    def test_existing_snapshot_is_not_overwritten(self):
        self.destination.mkdir()
        (self.destination/"retained.txt").write_bytes(b"retained")
        self.error("source_snapshot_exists", self.freeze)
        self.assertEqual((self.destination/"retained.txt").read_bytes(), b"retained")
        self.assertEqual(list(self.destination.iterdir()), [self.destination/"retained.txt"])

    def test_snapshot_parent_symlink_is_rejected(self):
        link = self.root/"linked-root"
        link.symlink_to(self.root, target_is_directory=True)
        self.error("source_root_unavailable", package.snapshot, link/"copy", self.spec,
                   self.files, expected_binding=self.binding)
        self.assertFalse((self.root/"copy").exists())

    def test_binding_is_required_and_well_formed(self):
        for value in (None, "", "0" * 63, "A" * 64):
            with self.subTest(value=value):
                self.error("source_binding_invalid", package.snapshot, self.destination, self.spec,
                           self.files, expected_binding=value)
        self.assertFalse(self.destination.exists())

    def test_source_tampering_after_snapshot_is_rejected(self):
        self.freeze()
        (self.destination/"models/network.py").write_bytes(b"def build(): return 2\n")
        self.error("source_hash_mismatch", self.reload)

    def test_self_rebound_files_cannot_override_trusted_binding(self):
        self.freeze()
        files = dict(self.files)
        files["models/network.py"] += b"# changed\n"
        spec = specification(files, "models.network.build")
        prepared = package.validate(spec, files)
        (self.destination/"models/network.py").write_bytes(files["models/network.py"])
        (self.destination/package.SNAPSHOT_MANIFEST).write_bytes(package.canonical(spec))
        (self.destination/package.SNAPSHOT_RECORD).write_bytes(package.canonical(prepared))
        self.error("source_preparation_binding_mismatch", self.reload)

    def test_record_cannot_promote_preparation_to_capture_or_backend_success(self):
        self.freeze()
        tampered = copy.deepcopy(self.record)
        tampered["factory_execution"] = "passed"
        tampered["backend_execution"] = "passed"
        (self.destination/package.SNAPSHOT_RECORD).write_bytes(package.canonical(tampered))
        self.error("source_snapshot_record_mismatch", self.reload)

    def test_snapshot_source_symlink_is_rejected(self):
        self.freeze()
        path = self.destination/"models/network.py"
        path.unlink()
        path.symlink_to(self.destination/"models/layers.py")
        self.error("source_path_link", self.reload)

    def test_missing_completion_record_rejects_partial_snapshot(self):
        self.freeze()
        (self.destination/package.SNAPSHOT_RECORD).unlink()
        self.error("source_file_unavailable", self.reload)

    def test_write_failure_retains_unusable_partial_snapshot(self):
        original = package.os.write
        count = 0
        def fail(descriptor, data):
            nonlocal count
            count += 1
            if count == 1:
                raise OSError("test failure")
            return original(descriptor, data)
        with patch.object(package.os, "write", side_effect=fail):
            self.error("source_snapshot_write", self.freeze)
        self.assertTrue(self.destination.exists())
        self.assertFalse((self.destination/package.SNAPSHOT_RECORD).exists())
        self.error("source_file_unavailable", self.reload)

    def test_duplicate_metadata_keys_are_rejected(self):
        self.freeze()
        path = self.destination/package.SNAPSHOT_MANIFEST
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + b',"format":"duplicate"}')
        self.error("source_snapshot_json", self.reload)

    def test_nonfinite_metadata_is_rejected(self):
        self.freeze()
        path = self.destination/package.SNAPSHOT_RECORD
        path.write_bytes(b'{"value":NaN}')
        self.error("source_snapshot_json", self.reload)

    def test_snapshot_files_use_private_modes(self):
        self.freeze()
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o700)
        for name in (*self.files, package.SNAPSHOT_MANIFEST, package.SNAPSHOT_RECORD):
            self.assertEqual((self.destination/name).stat().st_mode & 0o777, 0o600)

    def test_repeated_reload_is_identical_and_closes_descriptors(self):
        self.freeze()
        before = set(os.listdir("/proc/self/fd"))
        first = self.reload()
        self.assertEqual(first, self.reload())
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)
        os.link(self.destination/package.SNAPSHOT_RECORD, self.root/"linked.json")
        self.error("source_file_type", self.reload)
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)


if __name__ == "__main__":
    unittest.main()
