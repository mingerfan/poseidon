"""Structured package contracts and comparisons with frozen arithmetic paths."""

import ast
from contextlib import contextmanager
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pt_agent import run, toolchain
from pt_agent.backends import EmitBackend, MockBackend, RefreshBackend
from pt_agent.cost import PROFILES
from pt_agent.data.cifar import heldout_set
from pt_agent.dsl.parser import parse
from pt_agent.export import export_package
from pt_agent.export.codegen import generate
from pt_agent.export.package import compile_template
from pt_agent.lower import LoweringError, decode_outputs, lower, pack_inputs
from pt_agent.models.zoo import get
from pt_agent.paths import pt_result
from pt_agent.tests._artifacts import DATA, WEIGHTS, require


FROZEN = pt_result("r8e7-reconfirm-20261022T122558")
R6B = pt_result("r6b-refresh-20261021T025529/REPORT.json")
MODELS = pt_result("models-v2")


def cases():
    result = [(name, name, FROZEN/"models"/name/"program.ptdsl")
              for name in ("resnet20", "resnet20_silu", "cnn_plain", "mlp")]
    result.extend((name, "resnet20", FROZEN/"baselines"/name/"program.ptdsl") for name in ("R6c", "R7c-B"))
    report = json.loads(R6B.read_text())
    item = next(item for item in report["cases"] if item["mode"] == "agent" and item["deg"] == 127)
    result.append(("rule-agent", "resnet20", Path(item["program_path"])))
    return result


@contextmanager
def package_modules(package):
    name = json.loads((package/"config/model.json").read_text())["name"]
    previous = {key: value for key, value in sys.modules.items() if key == name or key.startswith(name+".")}
    for key in previous:
        del sys.modules[key]
    sys.path.insert(0, str(package))
    try:
        yield importlib.import_module(name+".runtime")
    finally:
        sys.path.pop(0)
        for key in list(sys.modules):
            if key == name or key.startswith(name+"."):
                del sys.modules[key]
        sys.modules.update(previous)


def emit_pair(text, weights, package):
    program = parse(text)
    first, second = EmitBackend(program.header.slots), EmitBackend(program.header.slots)
    config = PROFILES["qwen48"]
    active = (RefreshBackend(first, **{key: config[key] for key in ("input_level", "boot_in_level", "boot_out_level")})
              if program.header.refresh == "agent" else first)
    lower(program, weights, active)
    with package_modules(package) as runtime:
        runtime.invoke(second, weights)
    return first.render(), second.render()


def mock_reference(program, weights, images):
    backend = MockBackend(program.header.slots, batch=len(images))
    backend.bind(pack_inputs(program, [images]))
    values, layouts = lower(program, weights, backend)
    return decode_outputs([value.value for value in values], layouts)[0].copy()


class ExportUnitTests(unittest.TestCase):
    text = ('@fhe.program(slots=8192)\ndef small(x: Cipher[1,4]):\n'
            ' a = pack(x, layout="flat")\n b = act(a, fn="square", method="exact")\n return b\n')

    def export(self, directory, text=None):
        text = self.text if text is None else text
        return export_package(text, parse(text), {}, directory)

    def test_standalone_pad_error_matches_lowering(self):
        text = ('@fhe.program(slots=8192)\ndef bad(x: Cipher[1,3,8,8]):\n'
                ' a = pack(x, layout="mux")\n b = pad(a, pads=(0,0))\n return b\n')
        from pt_agent.backends import CountBackend
        with self.assertRaises(LoweringError) as original:
            lower(parse(text), {}, CountBackend(8192))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(LoweringError) as exported:
            self.export(directory, text)
        self.assertEqual(str(exported.exception), str(original.exception))
        self.assertEqual(exported.exception.op_id, original.exception.op_id)
        self.assertEqual(exported.exception.lineno, original.exception.lineno)

    def test_compile_template_and_dry_run(self):
        for refresh, mode in (("compiler", "dacapo-depth-dp"), ("agent", "pars")):
            with self.subTest(refresh=refresh), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                text = self.text.replace('slots=8192', f'slots=8192, refresh="{refresh}", l_max=5, boot_out_level=5')
                package = self.export(directory, text)
                source, output = directory/"trace/trace_model.mlir", directory/"compiled"
                proc = subprocess.run([sys.executable, "-B", str(package/"compile_model.py"), "--dry-run",
                    "--hecate-opt", str(toolchain.host_paths()["hecate_opt"]), "--input", str(source),
                    "--output-dir", str(output)], check=True, capture_output=True, text=True)
                original = toolchain.to_host_argv(toolchain.compile_argv(mode, "qwen48", output), output)
                expected = [arg.replace(str(output/"candidate_trace.mlir"), str(source))
                            .replace(str(output/"boot-compiler-config.json"), str(package/"config/compiler-config.json"))
                            for arg in original]
                self.assertEqual(json.loads(proc.stdout), expected)
                self.assertEqual(json.loads((package/"config/compile_argv.json").read_text()), compile_template(mode))
                self.assertFalse(output.exists())

    def test_first_lowering_error_precedes_later_pad(self):
        text = ('@fhe.program(slots=8192)\ndef bad(x: Cipher[1,3,8,8]):\n'
                ' a = pack(x, layout="mux")\n'
                ' c = conv2d(a, w=W["conv.weight"], stride=(1,2), pad=(1,1))\n'
                ' d = pad(c, pads=(0,0))\n return d\n')
        weights = {"conv.weight": np.ones((3, 3, 3, 3), dtype=np.float64)}
        from pt_agent.backends import CountBackend
        with self.assertRaises(LoweringError) as original:
            lower(parse(text), weights, CountBackend(8192))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(LoweringError) as exported:
            export_package(text, parse(text), weights, directory)
        self.assertEqual(str(exported.exception), str(original.exception))
        self.assertEqual(exported.exception.op_id, original.exception.op_id)
        self.assertEqual(exported.exception.lineno, original.exception.lineno)

    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.export(directory)
            argv = [sys.executable, "-B", str(package/"verify_manifest.py")]
            self.assertEqual(subprocess.run(argv, capture_output=True).returncode, 0)
            target = package/"small/model.py"
            data = target.read_bytes()
            target.write_bytes(bytes([data[0] ^ 1])+data[1:])
            self.assertNotEqual(subprocess.run(argv, capture_output=True).returncode, 0)

    def test_core_copies_and_hecate_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.export(directory)
            root = Path(toolchain.__file__).parent
            from pt_agent.export.package import CORE_SOURCES
            for name, source in CORE_SOURCES.items():
                self.assertEqual((package/"small/core"/name).read_text(), (root/source).read_text())
            with package_modules(package) as runtime:
                from pt_agent.tests.fixtures import fake_hecate_expr as hc
                module = importlib.import_module("small.hecate_backend")
                # The fake frontend uses the same expression operators as native Hecate.
                backend = module.HecateBackend(8192, hc, [hc.Expr(np.arange(8192, dtype=float))])
                runtime.invoke(backend, {})
                self.assertEqual(len(backend._outputs), 1)

    def test_run_compile_requires_export(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory), planner="rule", model="mlp", models_dir=MODELS,
                                   stages=["compile"], deg=31, margin=1.5, refresh="compiler")
            self.assertEqual(run._inside(args), 1)
            report = json.loads((args.out/"REPORT.json").read_text())
            self.assertIn("compile requires the export stage", report["error"])

    def test_server_job_contains_package_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            package = self.export(directory/"export")
            script = run._server_job(directory, package, directory/"export/weights/small.npz")
            text = script.read_text()
            self.assertIn(toolchain.server_env_exports(), text)
            self.assertNotIn("from pt_agent", text)
            self.assertNotIn("pt_agent.tools", text)
            self.assertTrue((script.parent/package.name/"trace_model.py").is_file())
            self.assertTrue((script.parent/"small.npz").is_file())


@require(FROZEN, R6B, DATA, WEIGHTS, MODELS)
class ExportArtifactTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_codegen_all_programs(self):
        for label, _, path in cases():
            with self.subTest(program=label):
                program = parse(path.read_text())
                text = generate(program)
                tree = ast.parse(text)
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                removed = sum(line.count("# slice ")*2 for line in text.splitlines())
                self.assertEqual(len(calls), len(program.ops)-removed)
                self.assertFalse(any(isinstance(node, ast.List) and any(isinstance(item, ast.Constant) for item in node.elts)
                                     for node in ast.walk(tree)))
                acts = [op for op in program.ops if op.kind == "act"]
                reads = [node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
                         and isinstance(node.value, ast.Name) and node.value.id == "A"]
                self.assertCountEqual(reads, [op.outputs[0] for op in acts])

    def test_A_emit_all_programs(self):
        for label, model, path in cases():
            with self.subTest(program=label), tempfile.TemporaryDirectory() as directory:
                weights = get(model, MODELS).import_graph().weights
                text = path.read_text()
                package = export_package(text, parse(text), weights, directory)
                (expected, ec), (actual, ac) = emit_pair(text, weights, package)
                self.assertEqual(actual, expected)
                self.assertEqual(list(ac), list(ec))
                self.assertTrue(all(np.array_equal(ac[name], ec[name]) for name in ec))
                print(f"P2 A {label}: render identical; constants={len(ec)}", flush=True)
                del expected, actual, ac, ec, weights
                gc.collect()

    def test_B_numpy_heldout_20(self):
        for name in ("mlp", "cnn_plain", "resnet20"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                zm = get(name, MODELS)
                weights = zm.import_graph().weights
                images = zm.adapt(heldout_set(20, normalization=zm.normalization)[0])
                text = (FROZEN/"models"/name/"program.ptdsl").read_text()
                program = parse(text)
                package = export_package(text, program, weights, directory)
                expected = mock_reference(program, weights, images)
                with package_modules(package) as runtime:
                    actual = runtime.run_numpy(weights, [images])[0]
                self.assertTrue(np.array_equal(actual, expected))
                print(f"P2 B {name}: 20 heldout, array_equal=True", flush=True)

    def test_self_contained_and_relocated(self):
        name = "mlp"
        zm = get(name, MODELS)
        weights = zm.import_graph().weights
        images = zm.adapt(heldout_set(2, normalization=zm.normalization)[0])
        text = (FROZEN/"models"/name/"program.ptdsl").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            package = export_package(text, parse(text), weights, directory/"original",
                run_info=dict(program_path=FROZEN/"models/mlp/program.ptdsl", run_dir=FROZEN/"models/mlp"))
            for path in package.rglob("*"):
                if not path.is_file():
                    continue
                content = path.read_text()
                self.assertNotIn("WORKSPACE", content)
                self.assertNotIn("REPO", content)
                self.assertNotRegex(content, r'''["']/(?:[^"']*)["']''')
                if path.suffix == ".py":
                    for node in ast.walk(ast.parse(content)):
                        if isinstance(node, ast.ImportFrom) and node.level:
                            continue
                        modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                                   [node.module] if isinstance(node, ast.ImportFrom) else [])
                        for module in modules:
                            self.assertIn(module.split(".")[0], sys.stdlib_module_names | {"numpy", "hecate", name})
            relocated = directory/"renamed"
            shutil.copytree(package, relocated)
            inputs, output = directory/"inputs.npy", directory/"output.npy"
            np.save(inputs, images)
            env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            planned = subprocess.run([sys.executable, "-B", str(relocated/"trace_model.py"), "--plan"],
                                     cwd=directory, env=env, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(planned.stdout)["function"], "_hecate_mlp")
            subprocess.run([sys.executable, "-B", str(relocated/"run_numpy.py"), "--weights",
                str(directory/"original/weights/mlp.npz"), "--inputs", str(inputs), "--output", str(output)],
                cwd=directory, env=env, check=True)
            self.assertTrue(np.array_equal(np.load(output), mock_reference(parse(text), weights, images)))

    def test_run_plan_check_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            args = SimpleNamespace(out=directory, planner="rule", model="mlp", models_dir=MODELS,
                stages=["plan", "check", "export"], deg=31, margin=1.5, refresh="compiler", act_method="cheb")
            # Keep all output inside the test's new directory, with the real exporter and planner.
            def new_directory(tag):
                path = directory/tag
                path.mkdir()
                return path
            with patch.object(toolchain, "new_run_dir", side_effect=new_directory):
                self.assertEqual(run._inside(args), 0)
            report = json.loads((directory/"REPORT.json").read_text())
            self.assertTrue((Path(report["export"]["package"])/"manifest.json").is_file())
            self.assertTrue(Path(report["export"]["weights"]).is_file())
            self.assertIsNone(report["emit"])


if __name__ == "__main__":
    unittest.main()
