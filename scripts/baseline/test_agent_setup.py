"""Offline setup safety/regression tests; no dependency installation or downloads."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import agent_setup as setup
import platform_config

ROOT = setup.ROOT


class PlanTests(unittest.TestCase):
    def test_auto_and_explicit_platforms(self):
        self.assertEqual(setup.select_platform("auto", system="linux", machine="arm64")["id"], "aarch64-linux")
        self.assertEqual(setup.select_platform("auto", system="linux", machine="amd64")["id"], "x86_64-linux")
        for system, machine in [("darwin", "arm64"), ("linux", "riscv64")]:
            with self.assertRaises(ValueError):
                setup.select_platform("auto", system=system, machine=machine)
        with self.assertRaises(ValueError):
            setup.select_platform("arbitrary")

    def test_plan_is_offline_and_does_not_create_work(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "absent"
            with patch.object(setup.subprocess, "run", side_effect=AssertionError("No subprocess in plan")):
                for name in ("aarch64-linux", "x86_64-linux"):
                    selected = platform_config.configuration(name)
                    plan = setup.make_plan(selected, work)
                    self.assertEqual(plan["platform"], name)
                    self.assertEqual(len(plan["python_wheels"]), 9)
                    self.assertEqual(len(plan["patches"]), 11)
                    self.assertFalse(plan["encrypted_execution_validated"])
                    self.assertEqual(plan["work_root"], str(work / selected["work_subdirectory"]))
            self.assertFalse(work.exists())

    def test_export_is_exact_and_architecture_specific(self):
        for name in ("aarch64-linux", "x86_64-linux"):
            selected = platform_config.configuration(name)
            plan = setup.make_plan(selected, Path("/tmp/poseidon-requirements-test"))
            requirements = setup.requirements_text(plan)
            records = [line for line in requirements.splitlines() if not line.startswith("#")]
            self.assertEqual(len(records), 9)
            for record, wheel in zip(records, plan["python_wheels"]):
                self.assertEqual(record, f"{wheel['name']} @ {wheel['url']} --hash=sha256:{wheel['sha256']}")
            self.assertEqual(plan["versions"]["torch"], "2.0.1" if name == "aarch64-linux" else "2.0.1+cpu")

    def test_work_directory_cannot_contain_or_be_inside_source(self):
        for work in (ROOT, ROOT / "build", ROOT.parent):
            with self.assertRaisesRegex(ValueError, "outside"):
                setup.make_plan(platform_config.configuration("aarch64-linux"), work)

    def test_apply_requires_explicit_budgets_and_submodule_flag_is_separate(self):
        for argv in (["--apply"], ["--apply", "--download-budget-mib", "10", "--disk-budget-gib", "-1"],
                     ["--init-submodule"], ["--self-test"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                setup.parse_args(argv)
        args = setup.parse_args(["--apply", "--download-budget-mib", "1", "--disk-budget-gib", "1"])
        self.assertFalse(args.init_submodule)
        self.assertFalse(args.self_test)

    def test_wrong_platform_apply_stops_before_work_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "absent"
            with patch.dict(os.environ, clear=False), \
                    patch.object(platform_config.sys, "platform", "linux"), \
                    patch.object(platform_config.platform, "machine", return_value="aarch64"), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = setup.main(["--platform", "x86_64-linux", "--work-root", str(work),
                                   "--apply", "--download-budget-mib", "1", "--disk-budget-gib", "1"])
            self.assertEqual(code, 2)
            self.assertFalse(work.exists())

    def test_cli_in_fresh_work_root(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "fresh"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/setup_agent.py"),
                "--platform", "aarch64-linux", "--work-root", str(work), "--plan", "--json"],
                capture_output=True, text=True, timeout=20, check=True)
            self.assertEqual(json.loads(result.stdout)["work_root"], str(work / "platforms/aarch64-linux"))
            self.assertFalse(work.exists())


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        setup.checked(["git", "init", "-q"], self.repo)
        self.file = self.repo / "example.txt"
        self.file.write_text("first\n")
        setup.checked(["git", "add", "example.txt"], self.repo)
        # Commit only a disposable test fixture, never the user's checkout.
        setup.checked(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                       "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], self.repo)
        self.commit = setup.checked(["git", "rev-parse", "HEAD"], self.repo)
        self.patches = []
        for index, (old, new) in enumerate([("first", "second"), ("second", "third")]):
            p = self.base / f"{index}.patch"
            p.write_text(f"diff --git a/example.txt b/example.txt\n--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-{old}\n+{new}\n")
            self.patches.append(p)

    def test_clean_partial_and_complete_prefixes(self):
        self.assertEqual(setup.patched_prefix(self.repo, self.patches, self.commit), 0)
        setup.checked(["git", "apply", str(self.patches[0])], self.repo)
        self.assertEqual(setup.patched_prefix(self.repo, self.patches, self.commit), 1)
        setup.checked(["git", "apply", str(self.patches[1])], self.repo)
        self.assertEqual(setup.patched_prefix(self.repo, self.patches, self.commit), 2)
        self.assertEqual(self.file.read_text(), "third\n")

    def test_conflicting_local_edit_is_preserved(self):
        self.file.write_text("user edit\n")
        with self.assertRaisesRegex(RuntimeError, "exact patch prefix"):
            setup.patched_prefix(self.repo, self.patches, self.commit)
        self.assertEqual(self.file.read_text(), "user edit\n")

    def test_staged_changes_are_preserved(self):
        self.file.write_text("user edit\n")
        setup.checked(["git", "add", "example.txt"], self.repo)
        with self.assertRaisesRegex(RuntimeError, "staged"):
            setup.patched_prefix(self.repo, self.patches, self.commit)
        self.assertEqual(setup.checked(["git", "show", ":example.txt"], self.repo), "user edit")

    def test_wrong_commit_and_uninitialized_directory(self):
        with self.assertRaisesRegex(RuntimeError, "gitlink"):
            setup.patched_prefix(self.repo, self.patches, "0" * 40)
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            setup.patched_prefix(self.base, self.patches, self.commit)

    def test_bad_patch_never_modifies_actual_checkout(self):
        self.patches[1].write_text("not a patch")
        with self.assertRaises(RuntimeError):
            setup.patched_prefix(self.repo, self.patches, self.commit)
        self.assertEqual(self.file.read_text(), "first\n")


class ExistingStateTests(unittest.TestCase):
    def test_frontend_link_is_idempotent_and_preserves_other_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            setup.ensure_frontend_link(work)
            setup.ensure_frontend_link(work)
            link = work / "build-dacapo/hecate-python-root/build"
            link.unlink()
            link.symlink_to(work / "user-build")
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                setup.ensure_frontend_link(work)
            self.assertEqual(link.readlink(), work / "user-build")

    def test_cross_architecture_cmake_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            build = work / "build-dacapo/hecate-18.1.2-nix"
            build.mkdir(parents=True)
            (build / "CMakeCache.txt").write_text("existing cache")
            selected = platform_config.configuration("aarch64-linux")
            with self.assertRaisesRegex(RuntimeError, "identify"):
                setup.check_build_caches(work, selected)
            record = build / "CMakeFiles/3.28.3/CMakeSystem.cmake"
            record.parent.mkdir(parents=True)
            record.write_text('set(CMAKE_SYSTEM_NAME "Linux")\nset(CMAKE_SYSTEM_PROCESSOR "x86_64")\nset(CMAKE_CROSSCOMPILING "FALSE")\n')
            with self.assertRaisesRegex(RuntimeError, "mismatched"):
                setup.check_build_caches(work, selected)
            record.write_text(record.read_text().replace("x86_64", "aarch64"))
            setup.check_build_caches(work, selected)
            self.assertEqual((build / "CMakeCache.txt").read_text(), "existing cache")

    def test_bad_existing_launcher_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            target = work / "deps/nix-portable-v012/nix-portable"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"user content")
            run = SimpleNamespace(work=work, run=lambda *args: self.fail("No download expected"))
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                setup.provision_launcher(run, platform_config.configuration("aarch64-linux"))
            self.assertEqual(target.read_bytes(), b"user content")

    def test_budget_counts_all_network_and_preserves_disk_reserve(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(setup, "network_bytes", return_value={"eth0": 10}) as network, \
                patch.object(setup.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * setup.GIB)) as disk:
            budget = setup.Budget(Path(directory), 1, 1)
            budget.check()
            network.return_value = {"eth0": 10 + setup.MIB}
            with self.assertRaisesRegex(RuntimeError, "Download budget"):
                budget.check()
            network.return_value = {"eth0": 10}
            disk.return_value = SimpleNamespace(free=8 * setup.GIB)
            with self.assertRaisesRegex(RuntimeError, "Disk budget"):
                budget.check()
            network.return_value = {"eth0": 9}
            with self.assertRaisesRegex(RuntimeError, "reset"):
                budget.check()

    def test_failed_child_and_source_change_stop_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "results").mkdir()
            budget = SimpleNamespace(usage={}, check=lambda: None)
            plan = {"platform": "aarch64-linux", "work_root_base": str(work)}
            with patch.dict(os.environ, DEEPSEEK_API_KEY="synthetic-secret"), \
                    patch.object(setup, "source_fingerprint", return_value={"a": "first"}) as fingerprints:
                runner = setup.Runner(ROOT, work, plan, budget)
                self.assertNotIn("DEEPSEEK_API_KEY", runner.env)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "failed"):
                    runner.run("failure", [sys.executable, "-c", "raise SystemExit(7)"], 5)
                self.assertEqual(runner.state["steps"][0]["exit_code"], 7)
                fingerprints.return_value = {"a": "changed"}
                with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                    runner.run("must_not_run", [sys.executable, "-c", "pass"], 5)
                self.assertEqual(len(runner.state["steps"]), 1)


class SequenceTests(unittest.TestCase):
    def test_build_sequence_reuses_existing_venv_and_never_calls_provider(self):
        selected = platform_config.configuration("aarch64-linux")
        commit = setup.read_json(ROOT / "src/poseidon/tools/dacapo/dependency-lock.json")["dacapo_commit"]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "venvs/hecate-2.0.1-cpu").mkdir(parents=True)
            calls = []
            def run_step(name, command, seconds):
                calls.append((name, command, seconds))
                if name == "nix_metadata":
                    return json.dumps({"platform_id": selected["id"], "toolchain_profile": "hecate",
                                       "shell_bash": "/nix/store/" + "a" * 32 + "-bash-5.2/bin/bash"})
                return ""
            run = SimpleNamespace(root=ROOT, work=work, folder=work, run=run_step,
                                  state={"plan": {"dacapo_commit": commit}})
            with patch.object(setup, "checked", return_value="160000 commit " + commit + "\tthird_party/dacapo"), \
                    patch.object(setup, "patched_prefix", return_value=11), \
                    patch.object(setup, "submodule_fingerprint", return_value={"fixture.cpp": "hash"}), \
                    patch.object(setup, "provision_launcher"), patch.object(setup, "check_build_caches"):
                setup.build_sequence(run, selected)
            stages = [x[0] for x in calls]
            self.assertEqual(stages, ["bubblewrap", "nix_metadata", "nix_dry_run", "build_sources",
                                     "build_seal", "build_toolchain", "build_shell", "build_native",
                                     "python_environment", "doctor"])
            python_command = next(cmd for name, cmd, _ in calls if name == "python_environment")
            self.assertIn("--verify-existing", python_command)
            self.assertNotIn("--install-approved", python_command)
            self.assertTrue(all("--live" not in cmd for _, cmd, _ in calls))
            self.assertTrue(all(seconds <= 43200 for _, _, seconds in calls))
            native = next(cmd for name, cmd, _ in calls if name == "build_native")
            self.assertIn("--pure", native)
            self.assertEqual(native[native.index("--max-jobs") + 1], "0")

    def test_submodule_fingerprint_reads_separate_tracked_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.cpp").write_text("a")
            (root / "b.py").write_text("b")
            (root / ".env").write_text("synthetic")
            with patch.object(setup, "checked", return_value="a.cpp\0b.py\0.env\0"):
                actual = setup.submodule_fingerprint(root)
            self.assertEqual(set(actual), {"a.cpp", "b.py"})

    def test_budget_interrupt_stops_child(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "results").mkdir()
            checks = []
            def check():
                checks.append(1)
                if len(checks) > 1:
                    raise RuntimeError("test budget exceeded")
            budget = SimpleNamespace(usage={}, check=check)
            with patch.object(setup, "source_fingerprint", return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                run = setup.Runner(ROOT, work, {"platform": "aarch64-linux", "work_root_base": str(work)}, budget)
                with self.assertRaisesRegex(RuntimeError, "test budget"):
                    run.run("interrupted", [sys.executable, "-c", "import time; time.sleep(60)"], 10)
            report = json.loads((run.folder / "report.json").read_text())
            self.assertEqual(report["steps"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
