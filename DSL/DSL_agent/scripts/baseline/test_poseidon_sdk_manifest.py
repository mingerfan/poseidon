"""SDK source/build binding and request-version regressions; no CKKS simulation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import poseidon_sdk_manifest as sdk


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / 'work'
        self.agent = self.root / 'agent'
        self.backend = self.root / 'backend'
        for p in (self.work, self.agent, self.backend):
            p.mkdir()
        for name in ('third_party/ckks-runtime/third_party/dacapo/compiler.cpp',
                     'src/poseidon/runtime_api/cpu.cpp', 'src/poseidon/tools/agent_runtime/runner.cpp'):
            p = self.backend / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('// Identity-only fixture; never compiled or executed.\n')
        (self.agent / 'environment.nix').write_text('identity fixture')
        self.compiler, self.executor, self.library = (self.work / name for name in ('compiler', 'executor', 'library'))
        for p in (self.compiler, self.executor, self.library):
            p.mkdir()
            (p / 'CMakeCache.txt').write_text('fixture-cache')
        for parent, name in ((self.compiler, 'bin/hecate-opt'), (self.compiler, 'lib/libHecateFrontend.so'),
                             (self.executor, 'poseidon_agent_cpu'), (self.library, 'libposeidon_shared.so')):
            p = parent / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'identity-only non-executable fixture')
        trees = dict(runtime=sdk.tree(self.backend / 'third_party/ckks-runtime'),
                     api=sdk.tree(self.backend / 'src/poseidon/runtime_api'),
                     runner=sdk.tree(self.backend / 'src/poseidon/tools/agent_runtime'))
        artifacts = dict(compiler=sdk.file_hash(self.compiler / 'bin/hecate-opt'),
                         frontend=sdk.file_hash(self.compiler / 'lib/libHecateFrontend.so'),
                         executor=sdk.file_hash(self.executor / 'poseidon_agent_cpu'),
                         poseidon_library=sdk.file_hash(self.library / 'libposeidon_shared.so'))
        lock = dict(format='poseidon-current-runtime-sources-v1', source_trees=trees,
                    compiler_source_binding='a'*64, executor_plan_binding='b'*64)
        write_json(self.work / 'source-lock.json', lock)
        write_json(self.work / 'policy.json', {'fixture': 'source-policy'})
        c = dict(status='succeeded', source_binding='a'*64,
                 artifacts={'bin/hecate-opt': {'sha256': artifacts['compiler']},
                            'lib/libHecateFrontend.so': {'sha256': artifacts['frontend']}})
        e = dict(status='succeeded', plan_binding='b'*64, executor_sha256=artifacts['executor'])
        write_json(self.work / 'compiler-report.json', c)
        write_json(self.work / 'executor-report.json', e)
        self.value = dict(format=sdk.FORMAT, platform='aarch64-linux', backend_source_root=str(self.backend),
            runtime_source=str(self.backend / 'third_party/ckks-runtime'), compiler_build=str(self.compiler),
            executor_build=str(self.executor), poseidon_build=str(self.library),
            source_lock=str(self.work / 'source-lock.json'), source_lock_sha256=sdk.file_hash(self.work / 'source-lock.json'),
            resource_policy=str(self.work / 'policy.json'), resource_policy_sha256=sdk.file_hash(self.work / 'policy.json'),
            source_trees=trees, artifacts=artifacts,
            cmake_caches={name: sdk.file_hash(p / 'CMakeCache.txt') for name, p in
                          (('compiler', self.compiler), ('executor', self.executor), ('poseidon', self.library))},
            build_receipts={name: dict(path=str(self.work / (name+'-report.json')),
                           sha256=sdk.file_hash(self.work / (name+'-report.json'))) for name in ('compiler', 'executor')},
            environment_files={'environment.nix': sdk.file_hash(self.agent / 'environment.nix')},
            qualification='pending_agent_acceptance')
        self.filename = self.work / 'sdk.json'
        self.save()

    def save(self):
        self.value['binding'] = sdk.digest({k: v for k, v in self.value.items() if k != 'binding'})
        write_json(self.filename, self.value)

    def checked(self):
        return sdk.attest(sdk.load(self.filename, self.work, 'aarch64-linux'), self.work, self.agent)

    def test_staging_record_preserves_pending_scope(self):
        self.assertEqual(self.checked(), self.value)
        self.assertEqual(self.checked()['qualification'], 'pending_agent_acceptance')

    def test_platform_and_descriptor_hash_cannot_be_reinterpreted(self):
        with self.assertRaises(ValueError):
            sdk.load(self.filename, self.work, 'x86_64-linux')
        self.value['compiler_build'] = str(self.executor)
        write_json(self.filename, self.value)
        with self.assertRaisesRegex(ValueError, 'descriptor hash'):
            self.checked()

    def test_every_binary_and_actual_cache_is_checked(self):
        paths = (self.compiler / 'bin/hecate-opt', self.compiler / 'lib/libHecateFrontend.so',
                 self.executor / 'poseidon_agent_cpu', self.library / 'libposeidon_shared.so',
                 self.compiler / 'CMakeCache.txt', self.executor / 'CMakeCache.txt', self.library / 'CMakeCache.txt')
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.checked()
            path.write_bytes(original)
        self.checked()

    def test_source_and_environment_drift_are_rejected(self):
        paths = [p for p in self.backend.rglob('*.cpp')] + [self.agent / 'environment.nix']
        for p in paths:
            original = p.read_bytes()
            p.write_bytes(original + b'changed')
            with self.subTest(path=str(p)), self.assertRaises(ValueError):
                self.checked()
            p.write_bytes(original)

    def test_rebinding_source_tree_label_cannot_replace_build_lock(self):
        p = self.backend / 'src/poseidon/runtime_api/cpu.cpp'
        p.write_text('changed source')
        self.value['source_trees']['api'] = sdk.tree(p.parent)
        self.save()
        with self.assertRaisesRegex(ValueError, 'source/build lock mismatch'):
            self.checked()

    def test_failed_or_mismatched_build_receipt_is_rejected_even_after_resigning(self):
        p = self.work / 'compiler-report.json'
        report = json.loads(p.read_text())
        report['status'] = 'failed'
        write_json(p, report)
        self.value['build_receipts']['compiler']['sha256'] = sdk.file_hash(p)
        self.save()
        with self.assertRaisesRegex(ValueError, 'unverified build'):
            self.checked()

    def test_empty_sources_symlinks_and_duplicate_fields_are_rejected(self):
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaises(ValueError):
            sdk.tree(empty)
        link = self.work / 'linked-sdk.json'
        link.symlink_to(self.filename)
        with self.assertRaises(ValueError):
            sdk.load(link, self.work, 'aarch64-linux')
        self.filename.write_text('{"format":1,"format":2}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            sdk.load(self.filename, self.work, 'aarch64-linux')


class ConfigurationTests(unittest.TestCase):
    def test_new_request_binds_current_sources_and_explicit_export(self):
        import compiler_configuration as cfg
        from poseidon_runtime_compile import command
        value = cfg.configuration(cfg.CURRENT_CONFIGURATION)
        lock = Path(cfg.__file__).with_name('poseidon-runtime-sources-current.lock.json')
        self.assertEqual(value['source_lock_sha256'], sdk.file_hash(lock))
        self.assertEqual(value['schema'], 5)
        self.assertEqual(value['runtime_commit'], 'f7e9115eaec22e44c8e12ff9f53528e421691688')
        argv = command(value['waterline'], entry_policy=value['runtime_plan_entry_policy'],
                       serialization_policy=value['runtime_plan_serialization_policy'])
        self.assertIn('--runtime-plan-format=json', argv)
        self.assertIn('--runtime-plan-bundle-format=files', argv)
        self.assertNotIn('--runtime-plan-format=json', command())
        self.assertNotEqual(value['identity_sha256'], cfg.configuration('poseidon-cpu-eva-w45-v1')['identity_sha256'])
        with self.assertRaises(ValueError):
            command(serialization_policy='unregistered')

    def test_current_resource_formula_rejects_reused_storage(self):
        from application_planning.compiled_resources import _measure_plan
        value = lambda name: dict(id=name, kind='ciphertext', level=0, components=2, scale_log2=45)
        plan = dict(values=[value('x'), value('y')], external_inputs=['x'], final_outputs=['y'],
                    initialization=[], finalization=[], execution=[dict(kind='compute', ordinal=0, op='negate',
                    inputs=['x'], output='y', reuse_input=True)])
        with self.assertRaisesRegex(ValueError, 'reuse_requires_lifetime_policy'):
            _measure_plan(plan, 8, 10000, {})
        plan['execution'][0]['reuse_input'] = False
        self.assertEqual(_measure_plan(plan, 8, 10000, {})['retained_coefficient_payload_bytes'], 256)


if __name__ == '__main__':
    unittest.main()
