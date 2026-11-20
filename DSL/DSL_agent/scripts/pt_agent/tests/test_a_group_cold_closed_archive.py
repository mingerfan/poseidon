from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pt_agent.experiments import a_group_cold_closed_archive as own
from pt_agent.experiments.r34_archive_text import archive_one as original_archive_one


def fixture(root, name, *, status='ok', raw=True):
    case = root / name
    emit = case / 'work/emit'
    emit.mkdir(parents=True)
    record = dict(round='COLD-RULE63', run_id=name, model=own.CASES[name],
        status=status, error=None if status == 'ok' else 'Synthetic closed compiler error',
        processes=[dict(observation=dict(remaining_pids=[], timed_out=False))])
    (case / 'result.json').write_text(json.dumps(record))
    (case / 'completion.json').write_text(json.dumps(dict(status=status,
        registry_persisted=True, required_artifacts_persisted=True)))
    registry = dict(run_id=name, round='COLD-RULE63', model=own.CASES[name],
        mode='synthetic', git_rev=None, metrics=dict(status=status),
        artifacts_dir=str(case), ts='synthetic')
    (root / 'registry.jsonl').write_text(json.dumps(registry) + '\n')
    (case / 'phase-resources.jsonl').write_text(json.dumps(dict(
        processes=[dict(pid=90001, starttime=42)])) + '\n')
    (emit / 'candidate_trace.py').write_text('synthetic retained source')
    (emit / 'lowered._hecate_golden.runtime-plan.json').write_text('{"execution":[]}')
    bundle = emit / 'lowered._hecate_golden.bundle'
    (bundle / 'data').mkdir(parents=True)
    (bundle / 'manifest.json').write_text('{}')
    (bundle / 'data/constant.bin').write_text('synthetic retained payload')
    info = case / 'work/intake/model_info'
    info.mkdir(parents=True)
    (info / 'model_info.json').write_text('{"source":"synthetic"}')
    if raw:
        (emit / 'candidate_trace.mlir').write_text('Synthetic raw text only\n' * 4)
    (emit / 'constants.npz').write_text('Synthetic redundant NPZ only\n' * 4)
    args = SimpleNamespace(case=name, out=root / 'archive-resource-only', run=True,
        seconds=1140, no_future_consumers=True,
        host_pid_namespace=own.os.readlink('/proc/self/ns/pid'))
    return args, case, emit


def patched(root, archive=None, *, free=None):
    stack = ExitStack()
    stack.enter_context(patch.object(own.paths, 'RESULTS_DIR', root))
    stack.enter_context(patch.object(own.prior, 'same_birth', return_value=False))
    stack.enter_context(patch.object(own.resource, 'setrlimit'))
    stack.enter_context(patch.object(own.signal, 'setitimer'))
    if archive is not None:
        stack.enter_context(patch.object(own, 'archive_one', side_effect=archive))
    if free is not None:
        stack.enter_context(patch.object(own.shutil, 'disk_usage',
            return_value=SimpleNamespace(free=free)))
    return stack


class ColdClosedArchiveTests(unittest.TestCase):
    def test_five_exact_root_contracts_and_closed_metadata_refusals(self):
        self.assertIs(own.archive_one, original_archive_one)
        self.assertEqual(len(own.CASES), 5)
        for name in own.CASES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args, case, emit = fixture(root, name)
                with patched(root):
                    admitted = own.admit(args)
                    self.assertEqual([Path(row['source']).name for row in admitted['selected']],
                                     ['candidate_trace.mlir', 'constants.npz'])
                    self.assertEqual(admitted['source_closure']['run_id'], name)
                    self.assertEqual(admitted['source_case'], str(case))
                    with patch.object(own.prior, 'same_birth', return_value=True):
                        with self.assertRaises(ValueError): own.admit(args)
                    args.host_pid_namespace = 'synthetic-nonhost'
                    with self.assertRaises(ValueError): own.admit(args)
                    args.host_pid_namespace = own.os.readlink('/proc/self/ns/pid')
                    record = json.loads((case / 'result.json').read_text())
                    record['processes'][0]['observation']['remaining_pids'] = [7]
                    (case / 'result.json').write_text(json.dumps(record))
                    with self.assertRaises(ValueError): own.admit(args)

    def test_fake_archive_clean_scientific_error_keeps_all_other_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, case, emit = fixture(root, list(own.CASES)[3], status='error')
            source_result = own.prior.snapshot(case / 'result.json')
            selected = []
            def fake_archive(source, target, temporary):
                selected.append(str(source))
                amount = source.stat().st_size
                target.write_text('x')
                source.unlink()
                return dict(status='compressed', source=str(source), archive=str(target),
                            released_bytes=amount-1, restore_argv=['gzip', '-dk', '--', str(target)])
            with patched(root, fake_archive, free=4*own.GIB + 2*1024**2):
                result = own.run(args)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['scientific_status'], 'error')
            self.assertFalse(result['scientific_status_upgraded'])
            self.assertEqual(result['completed_files'], 2)
            self.assertEqual([Path(p).name for p in selected], ['candidate_trace.mlir', 'constants.npz'])
            self.assertEqual(own.prior.snapshot(case / 'result.json'), source_result)
            self.assertTrue((case / 'work/intake/model_info/model_info.json').is_file())
            self.assertTrue((emit / 'lowered._hecate_golden.bundle/data/constant.bin').is_file())
            self.assertTrue((emit / 'lowered._hecate_golden.runtime-plan.json').is_file())
            self.assertTrue((args.out / 'completion.json').is_file())
            rows = [json.loads(line) for line in (root / 'registry.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(set(rows[1]), own.prior.REGISTRY_FIELDS)

    def test_absent_raw_skips_redundant_npz_without_scientific_upgrade(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, case, emit = fixture(root, list(own.CASES)[0], status='error', raw=False)
            with patched(root), patch.object(own, 'archive_one') as archive:
                result = own.run(args)
            archive.assert_not_called()
            self.assertEqual(result['resource_status'], 'metadata_skip')
            self.assertEqual(result['scientific_status'], 'error')
            self.assertEqual(result['completed_files'], 0)
            self.assertTrue((emit / 'constants.npz').exists())

    def test_disk_or_first_compression_failure_stops_and_persists_resource_error(self):
        for failure in ('disk', 'compressor'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args, case, emit = fixture(root, list(own.CASES)[-1])
                calls = []
                def fail(source, target, temporary):
                    calls.append(str(source))
                    temporary.write_text('Synthetic interrupted output')
                    raise OSError('Synthetic archive failure')
                free = 4*own.GIB if failure == 'disk' else 5*own.GIB
                with patched(root, fail, free=free):
                    result = own.run(args)
                self.assertEqual(result['status'], 'error')
                self.assertEqual(result['completed_files'], 0)
                self.assertEqual(len(calls), 0 if failure == 'disk' else 1)
                self.assertTrue((emit / 'candidate_trace.mlir').exists())
                self.assertTrue((emit / 'constants.npz').exists())
                self.assertTrue((args.out / 'result.json').is_file())
                self.assertTrue((args.out / 'completion.json').is_file())
                self.assertFalse((emit / 'constants.npz.gz.partial').exists())


if __name__ == '__main__':
    unittest.main()
