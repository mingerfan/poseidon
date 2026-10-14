import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import poseidon_backend_identity as binding
from platform_config import configuration

class BackendConfigurationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.root=root;work=root/"work";work.mkdir()
        self.compiler=work/"compiler";self.executor=work/"executor";self.poseidon=work/"poseidon"/"library"
        for p in (self.compiler,self.executor,self.poseidon):p.mkdir(parents=True)
        for p in (self.compiler/"CMakeCache.txt",self.executor/"CMakeCache.txt",self.poseidon.parent/"CMakeCache.txt"):
            p.write_text("CMAKE_BUILD_TYPE:STRING=Release\n")
        paths=["scripts/baseline/platform-profiles.json"]
        paths+=["src/poseidon/tools/dacapo/"+name for name in (
            "dependency-lock.json","hecate-toolchain-components.json","dacapo-dependencies.nix",
            "poseidon-cpu-shell.nix",configuration()["wheel_lock"])]
        for name in paths:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("original")
        self.scope=patch.multiple(binding,ROOT=root,WORK=work,BUILD=self.compiler,CPU_BUILD=self.executor,POSEIDON_BUILD=self.poseidon)
        self.scope.start();self.addCleanup(self.scope.stop)

    def test_environment_change_invalidates_configuration(self):
        initial=binding.configuration_identity()
        for name in initial["environment_files"]:
            p=self.root/name;p.write_text("changed")
            self.assertNotEqual(initial,binding.configuration_identity(),name)
            p.write_text("original")
        self.assertEqual(initial,binding.configuration_identity())

    def test_each_actual_cache_changes_identity(self):
        initial=binding.configuration_identity()
        for p in (self.compiler/"CMakeCache.txt",self.executor/"CMakeCache.txt",self.poseidon.parent/"CMakeCache.txt"):
            original=p.read_text();p.write_text(original+"CMAKE_CXX_FLAGS:STRING=-O3\n")
            self.assertNotEqual(initial,binding.configuration_identity())
            p.write_text(original)

    def test_missing_build_evidence_rejected(self):
        (self.compiler/"CMakeCache.txt").unlink()
        with self.assertRaisesRegex(ValueError,"Missing CMake"):binding.configuration_identity()

    def test_symlink_configuration_rejected(self):
        p=self.compiler/"CMakeCache.txt";p.unlink();p.symlink_to(self.executor/"CMakeCache.txt")
        with self.assertRaises(ValueError):binding.configuration_identity()

    def test_parent_cache_is_limited_to_work_root(self):
        outside=self.root/"outside";outside.mkdir();(outside/"CMakeCache.txt").write_text("outside")
        with self.assertRaisesRegex(ValueError,"outside platform"):binding.build_cache(outside)

    def test_old_identity_refused_before_package_execution(self):
        import poseidon_package
        with patch.object(poseidon_package,"identity",return_value={"format":"poseidon-backend-identity-v2"}):
            with self.assertRaisesRegex(ValueError,"dependency mismatch"):
                poseidon_package.replay(self.root,{"dependencies":{"backend_identity":{"format":"poseidon-backend-identity-v1"}}})
