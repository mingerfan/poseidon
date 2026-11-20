"""Identity/DSO-mount controls. Inert files are never libraries to execute."""
import copy
from pathlib import Path
import tempfile
import unittest

from delivery_component import bound, canonical, sha
import normal_cpu_overlay_sdk as overlay
from normal_cpu_overlay_sandbox import rewrite_command, loaded_library, run, TARGET
import test_poseidon_sdk_manifest as fixtures


class OverlaySelectorTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ManifestTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        self.ancestor=copy.deepcopy(f.value)
        candidate=f.work/'approved-library';candidate.mkdir();(candidate/'libposeidon_shared.so').write_bytes(b'inert approved-library fixture')
        (candidate/'CMakeCache.txt').write_bytes((f.library/'CMakeCache.txt').read_bytes())
        digest=sha((candidate/'libposeidon_shared.so').read_bytes())
        self.parent=dict(format='poseidon-runtime-toolchain-native-v5',poseidon_build=str(candidate),
            artifacts=dict(poseidon_library=digest),native=dict(base_sdk=dict(binding=self.ancestor['binding'])),
            library_overlay=dict(candidate=dict(path=str(candidate/'libposeidon_shared.so'),sha256=digest)))

    def test_selector_retains_current_compiler_executor_and_registered_sources(self):
        value=overlay.selector(self.ancestor,self.parent)
        overlay.check_selector(value,self.ancestor,self.parent)
        for key in ('compiler_build','executor_build','runtime_source','source_lock','source_lock_sha256','environment_files','source_trees','build_receipts'):
            self.assertEqual(value[key],self.ancestor[key])
        self.assertEqual(value['artifacts']['executor'],self.ancestor['artifacts']['executor'])

    def test_repaired_library_retains_the_original_configuration_ancestry(self):
        value=overlay.selector(self.ancestor,self.parent)
        self.assertEqual(value['artifacts']['poseidon_library'],self.parent['artifacts']['poseidon_library'])
        self.assertEqual(value['cmake_caches']['poseidon'],self.ancestor['cmake_caches']['poseidon'])
        self.assertEqual(value['cmake_caches']['compiler'],self.ancestor['cmake_caches']['compiler'])

    def test_pending_normal_scope_never_imports_native_or_bootstrap_acceptance(self):
        value=overlay.selector(self.ancestor,self.parent)
        self.assertEqual(value['qualification'],'pending_agent_acceptance')
        self.assertNotIn('native',value)
        self.assertNotIn('library_overlay',value)
        self.assertEqual(set(value),overlay.base.FIELDS)

    def test_changed_compiler_executor_or_source_ancestry_is_rejected(self):
        for field in ('compiler_build','executor_build','source_lock_sha256'):
            value=overlay.selector(self.ancestor,self.parent);value.pop('binding');value[field]='changed'
            with self.assertRaises(ValueError):overlay.check_selector(bound(value),self.ancestor,self.parent)

    def test_changed_artifact_or_wrong_overlay_digest_cannot_be_resigned(self):
        value=overlay.selector(self.ancestor,self.parent);value.pop('binding');value['artifacts']['executor']='0'*64
        with self.assertRaises(ValueError):overlay.check_selector(bound(value),self.ancestor,self.parent)
        self.parent['artifacts']['poseidon_library']='0'*64
        with self.assertRaises(ValueError):overlay.selector(self.ancestor,self.parent)

    def test_rejected_zero_key_library_remains_forbidden(self):
        self.parent['artifacts']['poseidon_library']=overlay.base.REJECTED_KEYGEN_LIBRARY_SHA256
        self.parent['library_overlay']['candidate']['sha256']=overlay.base.REJECTED_KEYGEN_LIBRARY_SHA256
        with self.assertRaises(ValueError):overlay.selector(self.ancestor,self.parent)

    def test_compiler_ancestor_binding_cannot_be_replaced(self):
        self.parent['native']['base_sdk']['binding']='0'*64
        with self.assertRaises(ValueError):overlay.selector(self.ancestor,self.parent)

    def test_retained_compatibility_alias_is_bound_to_actual_parameter_files(self):
        actual=self.fixture.agent/'DSL/DSL_agent/scripts/baseline';actual.mkdir(parents=True)
        (self.fixture.agent/'scripts').mkdir()
        (self.fixture.agent/'scripts/baseline').symlink_to(actual,target_is_directory=True)
        names=('poseidon-compiler-profile-v1.json','poseidon-operator-spec-v1.json')
        for name in names:(actual/name).write_text('inert parameter fixture')
        paths=overlay.ordinary_parameter_files(self.fixture.agent)
        self.assertEqual(paths,tuple(actual/name for name in names))
        self.assertTrue(all(p.resolve()==p and not p.is_symlink()for p in paths))


class OverlaySandboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.library=self.root/'libposeidon_shared.so';self.library.write_bytes(b'inert loader-identity fixture')
        self.original=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv',
            '--ro-bind','/nix/store','/nix/store','--bind','/output','/out',
            '--ro-bind',str(self.library),str(self.library),
            '--setenv','LD_LIBRARY_PATH','/nix/store/fixture/lib','--setenv','PATH','/nonexistent',
            '--chdir','/out','--','/poseidon-agent-cpu','/out/plan.json']

    def test_only_attested_readonly_library_mount_and_search_path_change(self):
        cmd=rewrite_command(self.original,self.library)
        self.assertIn(TARGET,cmd);self.assertIn('/selected-poseidon:/nix/store/fixture/lib',cmd)
        self.assertIn('--unshare-all',cmd);self.assertIn('--clearenv',cmd)
        self.assertEqual(cmd[cmd.index('--'):],self.original[self.original.index('--'):])
        self.assertEqual([cmd[i:i+3]for i in range(len(cmd)-2)if cmd[i]=='--bind'],[['--bind','/output','/out']])

    def test_original_command_is_not_mutated(self):
        before=copy.deepcopy(self.original);rewrite_command(self.original,self.library,trace_loader=True)
        self.assertEqual(self.original,before)

    def test_loader_trace_is_explicit_and_not_network_or_key_access(self):
        cmd=rewrite_command(self.original,self.library,trace_loader=True)
        index=cmd.index('LD_DEBUG');self.assertEqual(cmd[index+1],'libs')
        self.assertNotIn('--share-net',cmd);self.assertNotIn('--unshare-net-try',cmd)

    def test_missing_or_ambiguous_library_mount_is_rejected(self):
        for cmd in [self.original[:-15],self.original+['--ro-bind',str(self.library),str(self.library)]]:
            with self.assertRaises(ValueError):rewrite_command(cmd,self.library)

    def test_non_Nix_or_ambiguous_library_environment_is_rejected(self):
        cmd=list(self.original);cmd[cmd.index('LD_LIBRARY_PATH')+1]='/tmp/arbitrary'
        with self.assertRaises(ValueError):rewrite_command(cmd,self.library)
        with self.assertRaises(ValueError):rewrite_command(self.original+['--setenv','LD_LIBRARY_PATH','/nix/store/other'],self.library)

    def test_actual_loader_evidence_binds_selected_DSO_and_log_bytes(self):
        log=self.root/'execute.log';log.write_text('    1234: calling init: '+TARGET+'\n')
        receipt=loaded_library(log,self.library)
        self.assertEqual(receipt['selected_library_sha256'],sha(self.library.read_bytes()))
        self.assertEqual(receipt['loader_log_sha256'],sha(log.read_bytes()))
        self.assertFalse(receipt['secret_key_contents_or_hashes_read'])

    def test_success_text_without_actual_loader_path_is_rejected(self):
        log=self.root/'execute.log';log.write_text('program finished successfully\n')
        with self.assertRaises(ValueError):loaded_library(log,self.library)

    def test_old_or_multiple_loaded_Poseidon_libraries_are_rejected(self):
        log=self.root/'execute.log'
        for lines in ['    123: calling init: /old/libposeidon_shared.so\n',
                '    123: calling init: '+TARGET+'\n    123: calling init: /old/libposeidon_shared.so\n']:
            log.write_text(lines)
            with self.assertRaises(ValueError):loaded_library(log,self.library)

    def test_persistent_key_directories_are_not_accepted_by_ordinary_client(self):
        with self.assertRaises(ValueError):run(self.root,self.root,[],self.root/'execute.log',
            keys=self.root/'new-private-epoch',selected_library=self.library)

    def test_selected_library_name_is_fixed(self):
        with self.assertRaises(ValueError):rewrite_command(self.original,self.root/'other.so')


if __name__=='__main__':unittest.main()
