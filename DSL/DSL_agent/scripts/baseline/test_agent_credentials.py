"""Credential persistence tests use temporary files and fake keys only."""
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_credentials import CredentialError, credential_root, load_api_key, parse_env


class CredentialTests(unittest.TestCase):
    def test_linked_checkout_loads_primary_file_only_with_clean_git_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "primary"
            linked = Path(directory) / "linked"
            primary.mkdir(); linked.mkdir()
            (primary / ".git").mkdir()
            (linked / ".git").write_text("gitdir: fixture metadata\n")
            (primary / ".env").write_text("DEEPSEEK_API_KEY=FAKE-PRIMARY")
            (linked / ".env").write_text("DEEPSEEK_API_KEY=FAKE-NOT-USED")
            def git(argv, **options):
                self.assertEqual(options["env"], {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
                self.assertNotIn("FAKE", " ".join(argv))
                self.assertIn("--no-optional-locks", argv)
                return SimpleNamespace(stdout=str(linked)+"\n"+str(primary / ".git")+"\n")
            with patch("agent_credentials.subprocess.run", side_effect=git):
                self.assertEqual(load_api_key(linked, {}), "FAKE-PRIMARY")
            self.assertTrue((primary / ".env").is_file() and (linked / ".env").is_file())

    def test_invalid_git_worktree_never_falls_back_to_another_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").write_text("gitdir: fixture\n")
            (root / ".env").write_text("DEEPSEEK_API_KEY=FAKE-NOT-USED")
            with patch("agent_credentials.subprocess.run", return_value=SimpleNamespace(stdout="invalid\n")):
                with self.assertRaises(CredentialError):
                    load_api_key(root, {})

    def test_actual_feature_checkout_resolves_integration_without_reading_credentials(self):
        root = Path(__file__).resolve().parents[4]
        if root.name != "Poseidon-generalization":
            self.skipTest("Feature checkout metadata fixture is unavailable")
        self.assertEqual(credential_root(root), root.parent / "Poseidon")

    def test_literal_quoted_and_windows_lines(self):
        for text in ("DEEPSEEK_API_KEY=FAKE-KEY", 'DEEPSEEK_API_KEY="FAKE-KEY"',
                     "# comment\r\n DEEPSEEK_API_KEY = 'FAKE-KEY'\r\n"):
            self.assertEqual(parse_env(text), "FAKE-KEY")

    def test_no_shell_execution_or_unknown_variables(self):
        cases = ["", "DEEPSEEK_API_KEY=", "DEEPSEEK_API_KEY=$(touch SECRET)",
                 "DEEPSEEK_API_KEY=`SECRET`", "export DEEPSEEK_API_KEY=SECRET",
                 "OTHER=SECRET", "DEEPSEEK_API_KEY=SECRET\nDEEPSEEK_API_KEY=SECRET",
                 'DEEPSEEK_API_KEY="SECRET', "DEEPSEEK_API_KEY=SECRET\x00",
                 "DEEPSEEK_API_KEY=SECRET value"]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(CredentialError) as error:
                parse_env(text)
            self.assertNotIn("SECRET", str(error.exception))

    def test_environment_wins_without_accessing_file(self):
        self.assertEqual(load_api_key("/does-not-exist", {"DEEPSEEK_API_KEY": "FAKE-OVERRIDE"}),
                         "FAKE-OVERRIDE")

    def test_file_survives_independent_loads_without_modifying_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / ".env"
            file.write_bytes(b"\xef\xbb\xbfDEEPSEEK_API_KEY=FAKE-PERSISTED\r\n")
            before = file.read_bytes()
            for _ in range(2):
                env = {}
                self.assertEqual(load_api_key(directory, env), "FAKE-PERSISTED")
                self.assertEqual(env, {})
            self.assertEqual(file.read_bytes(), before)

    def test_missing_oversized_and_nonregular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / ".env"
            with self.assertRaises(CredentialError):
                load_api_key(directory, {})
            file.write_bytes(b"x" * 4097)
            with self.assertRaises(CredentialError):
                load_api_key(directory, {})
            file.unlink()
            file.mkdir()
            with self.assertRaises(CredentialError):
                load_api_key(directory, {})

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("DEEPSEEK_API_KEY=FAKE-KEY")
            try:
                (Path(directory) / ".env").symlink_to(target)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(CredentialError):
                load_api_key(directory, {})

    def test_removed_credential_variables_are_rejected(self):
        for name in ("OPENCODE_GO_API_KEY", "CMD_API_KEY"):
            with self.subTest(name=name):
                with self.assertRaises(CredentialError):
                    parse_env("DEEPSEEK_API_KEY=FAKE-DIRECT\n" + name + "=FAKE-OLD")
                with self.assertRaises(CredentialError):
                    parse_env(name + "=FAKE-OLD", name)

    def test_duplicate_or_invalid_other_provider_is_rejected_without_leaking(self):
        for suffix in ("CMD_API_KEY=SECRET\nCMD_API_KEY=SECRET",
                       "CMD_API_KEY=$(SECRET)", 'CMD_API_KEY="SECRET'):
            with self.assertRaises(CredentialError) as error:
                parse_env("DEEPSEEK_API_KEY=FAKE-DIRECT\n" + suffix)
            self.assertNotIn("SECRET", str(error.exception))

    def test_legacy_files_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, name in ((".env.local", "DEEPSEEK_API_KEY"),
                                   (".env.opencode-go.local", "OPENCODE_GO_API_KEY"),
                                   (".env.commandcode-goat.local", "CMD_API_KEY")):
                (root / filename).write_text(name + "=FAKE-LEGACY")
            with self.assertRaisesRegex(CredentialError, "Missing .env"):
                load_api_key(root, {})

    def test_removed_providers_cannot_use_environment_override(self):
        for provider, name in (("opencode-go", "OPENCODE_GO_API_KEY"),
                               ("commandcode-goat", "CMD_API_KEY")):
            with self.assertRaisesRegex(CredentialError, "Unsupported credential provider"):
                load_api_key("/does-not-exist", {name: "FAKE-OVERRIDE",
                             "DEEPSEEK_API_KEY": "FAKE-DIRECT"}, provider=provider)


if __name__ == "__main__":
    unittest.main()
