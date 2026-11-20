"""Four small staged checks for optional R30 archiving; no model or toolchain."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pt_agent import paths
from pt_agent.experiments import r30_common_model_e2e as entry
from pt_agent.experiments import r34_archive_text as gzip_archive


class R30TraceArchiveChecks(unittest.TestCase):
    def test_cli_defaults_and_explicit_finite_resource_option(self):
        original_phases = ('intake', 'calibration', 'agent', 'placement', 'emit', 'trace', 'compile')
        self.assertEqual(entry.PHASES, original_phases)
        for extra, enabled, memory, case in [([], False, 6, 8),
                (['--archive-emit-constants-after-trace', '--memory-gib', '12', '--case-gib', '32'], True, 12, 32)]:
            with patch.object(entry, 'execute', return_value={'status': 'prepared'}) as execute:
                self.assertEqual(entry.main(['--model', 'vgg16_silu_avgpool', '--out', '/tmp/staged-r30', *extra]), 0)
            args = execute.call_args.args[0]
            self.assertEqual(args.archive_emit_constants_after_trace, enabled)
            self.assertEqual((args.memory_gib, args.case_gib), (memory, case))
            self.assertEqual(args.stop_after, 'compile')
        for option, value in [('--memory-gib', '13'), ('--case-gib', '33')]:
            with patch.object(entry, 'execute') as execute, self.assertRaises(SystemExit):
                entry.main(['--model', 'resnet20_silu', '--out', '/tmp/staged-r30', option, value])
            execute.assert_not_called()

    def trace_fixture(self, root):
        emit = root / 'post-freeze/emit'
        emit.mkdir(parents=True)
        source = emit / 'constants.npz'
        source.write_text('synthetic public generated NPZ fixture\n' * 2000)
        trace_source = emit / 'candidate_trace.py'
        trace_source.write_text('# synthetic trusted trace source\n')
        mlir = emit / 'candidate_trace.mlir'
        mlir.write_text('module { /* synthetic dense constant */ }\n')
        clean = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])
        result = dict(model='resnet20_silu', attempted=dict(emit=1, trace=1), completed=dict(emit=1, trace=1),
            phases=dict(emit=dict(status='ok', trace_source=str(trace_source)),
                        trace=dict(status='ok', process=clean, mlir=dict(path=str(mlir), bytes=mlir.stat().st_size))),
            processes=[dict(phase='trace', observation=deepcopy(clean))])
        return source, result

    def test_clean_trace_and_case_paths_are_required_before_archive(self):
        mutations = [lambda r: r['completed'].pop('trace'),
                     lambda r: r['phases']['emit'].update(status='error'),
                     lambda r: r['phases']['trace']['process'].update(exit_code=1),
                     lambda r: r['phases']['trace']['process'].update(timed_out=True),
                     lambda r: r['phases']['trace']['process'].update(signals_sent=['TERM']),
                     lambda r: r['processes'][0]['observation'].update(signal=15),
                     lambda r: r['processes'][0]['observation'].update(remaining_pids=[111]),
                     lambda r: r['processes'].append(deepcopy(r['processes'][0])),
                     lambda r: r['attempted'].update(compile=1),
                     lambda r: r['phases']['trace']['mlir'].update(path='/foreign/candidate_trace.mlir'),
                     lambda r: r['phases']['trace']['mlir'].update(bytes=0),
                     lambda r: r['phases']['emit'].update(trace_source='/foreign/candidate_trace.py')]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, original = self.trace_fixture(root)
            for mutation in mutations:
                result = deepcopy(original)
                mutation(result)
                with patch.object(gzip_archive, 'archive_one') as archive, self.assertRaises(ValueError):
                    entry.archive_emit_constants_after_trace(result, root)
                archive.assert_not_called()
                self.assertTrue(source.is_file())
            target = source.with_name(source.name + '.gz.partial')
            target.write_text('retained interrupted output\n')
            with patch.object(gzip_archive, 'archive_one') as archive, self.assertRaises(ValueError):
                entry.archive_emit_constants_after_trace(original, root)
            archive.assert_not_called()
            target.unlink()
            with patch.object(entry.shutil, 'disk_usage', return_value=Namespace(free=10 * entry.GIB - 1)), \
                    patch.object(gzip_archive, 'archive_one') as archive, self.assertRaises(RuntimeError):
                entry.archive_emit_constants_after_trace(original, root)
            archive.assert_not_called()
            saved = source.with_name('source-retained.npz')
            source.rename(saved)
            source.symlink_to(saved)
            with patch.object(gzip_archive, 'archive_one') as archive, self.assertRaises(ValueError):
                entry.archive_emit_constants_after_trace(original, root)
            archive.assert_not_called()

    def run_case(self, root, *, enabled, archive_fails=False):
        from pt_agent.experiments import r13_common_models
        original = root / 'original'
        original.mkdir()
        inputs = original / 'records'
        inputs.mkdir()
        (inputs / 'dataset.json').write_text('{}')
        files = {key: original / key for key in ('source', 'checkpoint', 'inference_preprocessing_source')}
        for path in files.values():
            path.write_text('synthetic original-source metadata\n')
        args = Namespace(out=root / 'new-e2e', model='vgg16_silu_avgpool', inputs=inputs,
            selections=original / 'selections', frontend=original / 'frontend', profile=original / 'profile',
            metadata_template=original / 'metadata', ledger=root / 'calls.jsonl', allowance=root / 'allowance.json',
            seconds=28800, agent_seconds=7200, trace_seconds=1800, compile_seconds=7200, memory_gib=6,
            case_gib=32, calls=6, tuner_seconds=600, stop_after='compile',
            archive_emit_constants_after_trace=enabled)
        for path in [args.selections, args.frontend, args.profile, args.metadata_template]:
            path.write_text('{}\n')
        order, prepared = [], []
        clean = dict(exit_code=0, timed_out=False, signal=None, signals_sent=[], remaining_pids=[])

        def process(argv, **kwargs):
            phase = argv[argv.index('--worker-phase') + 1]
            work = Path(argv[argv.index('--work-dir') + 1])
            order.append(phase)
            value = dict(status='ok', phase=phase)
            emit = args.out / 'post-freeze/emit'
            if phase == 'agent':
                (work / 'worker-result.json').write_text(json.dumps(dict(status='ok')))
            elif phase == 'emit':
                emit.mkdir()
                (emit / 'constants.npz').write_text('synthetic generated constants\n' * 2000)
                (emit / 'candidate_trace.py').write_text('# synthetic trusted source\n')
                value['trace_source'] = str(emit / 'candidate_trace.py')
            elif phase == 'trace':
                mlir = emit / 'candidate_trace.mlir'
                mlir.write_text('module { /* synthetic dense constant */ }\n')
                value.update(process=deepcopy(clean), mlir=dict(path=str(mlir), bytes=mlir.stat().st_size))
            elif phase == 'compile':
                self.assertEqual((emit / 'constants.npz').exists(), not enabled)
                if enabled:
                    recovery = json.loads((args.out / 'post-freeze/emit-constants-recovery.json').read_text())
                    self.assertEqual(recovery['status'], 'compressed')
                    self.assertEqual(recovery['restore_argv'], ['gzip', '-dk', '--', str(emit / 'constants.npz.gz')])
                    self.assertTrue((emit / 'constants.npz.gz').is_file())
                value.update(compiled_bootstraps=2, predicted_latency_us=123)
            (work / 'phase_result.json').write_text(json.dumps(value))
            return deepcopy(clean)

        archive_impl = entry.archive_emit_constants_after_trace
        compress_impl = gzip_archive.archive_one

        def archive(result, case_root):
            order.append('archive_emit_constants')
            self.assertEqual(result['completed']['trace'], 1)
            self.assertNotIn('compile', result['attempted'])
            return archive_impl(result, case_root)

        def compress(source, target, temporary):
            manifest = args.out / 'post-freeze/emit-constants-recovery.json'
            recovery = json.loads(manifest.read_text())
            self.assertEqual(recovery['status'], 'prepared')
            self.assertTrue(source.is_file())
            prepared.append(recovery)
            if archive_fails:
                with patch.object(gzip_archive.os, 'fsync', side_effect=OSError('synthetic archive fsync failure')):
                    return compress_impl(source, target, temporary)
            return compress_impl(source, target, temporary)

        with ExitStack() as stack:
            stack.enter_context(patch.object(paths, 'RESULTS_DIR', root))
            stack.enter_context(patch.dict(r13_common_models.FACTORIES, {entry.FACTORY_PREFIX + args.model: files}))
            stack.enter_context(patch('pt_agent.registry.git_rev', return_value='synthetic'))
            stack.enter_context(patch.object(entry, 'run_process', side_effect=process))
            stack.enter_context(patch.object(entry, 'audit_agent', return_value={'status': 'quality_accepted'}))
            stack.enter_context(patch.object(entry, 'archive_emit_constants_after_trace', side_effect=archive))
            stack.enter_context(patch.object(gzip_archive, 'archive_one', side_effect=compress))
            result = entry.execute(args)
        return args, result, order, prepared

    def test_optional_stage_preserves_default_and_records_timing_and_recovery(self):
        for enabled in [False, True]:
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory:
                args, result, order, prepared = self.run_case(Path(directory), enabled=enabled)
                expected = list(entry.PHASES)
                if enabled:
                    expected.insert(expected.index('compile'), 'archive_emit_constants')
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(order, expected)
                self.assertEqual(list(result['attempted']), expected)
                self.assertEqual(result['completed'], {name: 1 for name in expected})
                timing = json.loads((args.out / 'timing.json').read_text())
                self.assertEqual(timing['status'], 'ok')
                spans = [row for row in timing['spans'] if row['stage_id'] == 'archive_emit_constants']
                self.assertEqual(len(spans), int(enabled))
                self.assertEqual(len(prepared), int(enabled))
                if enabled:
                    self.assertEqual(spans[0]['status'], 'ok')
                    self.assertGreaterEqual(spans[0]['inclusive_ns'], 0)
                completion = json.loads((args.out / 'completion.json').read_text())
                self.assertTrue(completion['registry_persisted'])
                rows = [json.loads(line) for line in (Path(directory) / 'registry.jsonl').read_text().splitlines()]
                self.assertEqual(set(rows[-1]), {'run_id', 'round', 'model', 'mode', 'git_rev', 'metrics', 'artifacts_dir', 'ts'})

    def test_failed_archive_retains_source_manifest_and_partial_and_halts_compile(self):
        with TemporaryDirectory() as directory:
            args, result, order, prepared = self.run_case(Path(directory), enabled=True, archive_fails=True)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(order, list(entry.PHASES[:-1]) + ['archive_emit_constants'])
            self.assertNotIn('compile', result['attempted'])
            self.assertNotIn('archive_emit_constants', result['completed'])
            self.assertEqual(len(prepared), 1)
            self.assertTrue((args.out / 'post-freeze/emit/constants.npz').is_file())
            self.assertFalse((args.out / 'post-freeze/emit/constants.npz.gz').exists())
            self.assertTrue((args.out / 'post-freeze/emit/constants.npz.gz.partial').is_file())
            recovery = json.loads((args.out / 'post-freeze/emit-constants-recovery.json').read_text())
            self.assertEqual(recovery['status'], 'error')
            self.assertTrue(recovery['interrupted_state']['source_exists'])
            self.assertTrue(recovery['interrupted_state']['temporary_exists'])
            timing = json.loads((args.out / 'timing.json').read_text())
            self.assertEqual(timing['status'], 'ok')
            stage = next(row for row in timing['spans'] if row['stage_id'] == 'archive_emit_constants')
            self.assertEqual(stage['status'], 'error')


if __name__ == '__main__':
    unittest.main()
