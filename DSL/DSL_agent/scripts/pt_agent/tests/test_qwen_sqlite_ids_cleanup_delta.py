"""Targeted interruption/owned cleanup delta; original scalar tests stay separate."""
import gc
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import weakref

from pt_agent.experiments import qwen_sqlite_ids as storage


KEY = ('neg', (('cipher', 1),), None)


class SQLiteCleanupDeltaChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)

    def mapping(self):
        value = storage.SQLiteIds(self.parent)
        self.addCleanup(value.close)
        value[KEY] = 1
        return value

    def assert_interruption(self, expected, callback):
        with self.assertRaises(InterruptedError) as caught:
            callback()
        self.assertIs(caught.exception, expected)

    def test_flush_and_mutate_keep_exact_interruption(self):
        value = self.mapping()
        interrupted = InterruptedError('flush fixture')
        with patch.object(value, '_disk', side_effect=interrupted):
            self.assert_interruption(interrupted, value.flush)
        self.assertIsNone(value.failure)
        self.assertFalse(value.snapshot()['closed'])
        interrupted = InterruptedError('mutation fixture')
        with patch.object(value, '_connection', side_effect=interrupted):
            self.assert_interruption(interrupted, lambda: value.__setitem__(KEY, 2))
        self.assertEqual(value[KEY], 1)
        self.assertIsNone(value.failure)

    def test_iteration_and_lookup_keep_fetch_interruption_when_close_also_fails(self):
        value = self.mapping()
        real_connection = value._state['connection']
        for operation in ('iteration', 'lookup'):
            for cleanup in (OSError('cursor close'), InterruptedError('later interruption')):
                with self.subTest(operation=operation, cleanup=type(cleanup).__name__):
                    interrupted = InterruptedError('original fetch')
                    class Cursor:
                        def fetchone(self):raise interrupted
                        def close(self):raise cleanup
                    class Connection:
                        def execute(self, *args):return Cursor()
                    value._state['connection'] = Connection()
                    try:
                        action = (lambda: next(iter(value))) if operation == 'iteration' else (lambda: value[KEY])
                        self.assert_interruption(interrupted, action)
                    finally:
                        value._state['connection'] = real_connection
                    self.assertIsNone(value.failure)
                    last = value.snapshot()['cleanup_errors'][-1]
                    self.assertEqual(last['operation'], ('iterator' if operation == 'iteration' else operation) + '.cursor.close')
                    self.assertEqual(last['error'], type(cleanup).__name__)

    def test_release_interruption_preserves_identity_paths_and_retry(self):
        for operation in ('connection', 'unlink', 'directory'):
            with self.subTest(operation=operation):
                value = self.mapping()
                directory = value.directory
                path = value._state['path']
                interrupted = InterruptedError(operation + ' fixture')
                if operation == 'connection':
                    real_connection = value._state['connection']
                    class Connection:
                        def close(self):raise interrupted
                    value._state['connection'] = Connection()
                    try:self.assert_interruption(interrupted, value.close)
                    finally:value._state['connection'] = real_connection
                    self.assertTrue(value.snapshot()['connection_pending_close'])
                elif operation == 'unlink':
                    real_unlink = Path.unlink
                    def unlink(target, *args, **kwargs):
                        if target == path:raise interrupted
                        return real_unlink(target, *args, **kwargs)
                    with patch.object(Path, 'unlink', unlink):
                        self.assert_interruption(interrupted, value.close)
                    self.assertIn(str(path), value.snapshot()['pending_owned_paths'])
                    self.assertEqual(value._state['path'], path)
                else:
                    real_rmdir = Path.rmdir
                    def rmdir(target):
                        if target == directory:raise interrupted
                        return real_rmdir(target)
                    with patch.object(Path, 'rmdir', rmdir):
                        self.assert_interruption(interrupted, value.close)
                    self.assertTrue(value.snapshot()['directory_pending'])
                self.assertTrue(value.snapshot()['cleanup_incomplete'])
                self.assertTrue(value._finalizer.alive)
                value.close()
                self.assertFalse(directory.exists())
                self.assertFalse(value.snapshot()['cleanup_incomplete'])
                self.assertEqual(value.snapshot()['pending_owned_paths'], [])
                self.assertFalse(value._finalizer.alive)

    def test_clear_retains_only_failed_owned_sidecar_until_explicit_retry(self):
        value = self.mapping()
        value.flush()
        path = value._state['path']
        sidecar = Path(str(path) + '-wal')
        sidecar.write_text('owned scalar fixture')
        foreign = self.parent / 'foreign-metadata'
        foreign.write_text('keep')
        real_unlink = Path.unlink
        def unlink(target, *args, **kwargs):
            if target == sidecar:raise PermissionError('fixture denied')
            return real_unlink(target, *args, **kwargs)
        with patch.object(Path, 'unlink', unlink):
            with self.assertRaises(storage.SQLiteStorageError):value.clear()
        observed = value.snapshot()
        self.assertTrue(observed['closed'])
        self.assertTrue(observed['cleanup_incomplete'])
        self.assertEqual(observed['pending_owned_paths'], [str(sidecar)])
        self.assertEqual(value._state['path'], path)
        self.assertEqual(observed['cleanup_errors'][-1], dict(operation='unlink', path=str(sidecar), error='PermissionError'))
        self.assertFalse(path.exists())
        self.assertTrue(sidecar.exists())
        value.close()
        self.assertFalse(value.directory.exists())
        self.assertEqual(foreign.read_text(), 'keep')
        self.assertEqual(value.snapshot()['cleanup_errors'], observed['cleanup_errors'])
        self.assertFalse(value.snapshot()['cleanup_incomplete'])

    def test_sql_failure_discloses_owned_cleanup_failure_and_close_retries(self):
        value = self.mapping()
        path = value._state['path']
        failure = sqlite3.OperationalError('fixture SQL failure')
        real_unlink = Path.unlink
        def unlink(target, *args, **kwargs):
            if target == path:raise PermissionError('fixture denied')
            return real_unlink(target, *args, **kwargs)
        with patch.object(Path, 'unlink', unlink):
            with self.assertRaisesRegex(storage.SQLiteStorageError, 'cleanup incomplete: PermissionError') as caught:
                value._fail(failure)
        self.assertIs(caught.exception.__cause__, failure)
        observed = value.snapshot()
        self.assertEqual(observed['failure'], 'OperationalError')
        self.assertTrue(observed['closed'])
        self.assertTrue(observed['cleanup_incomplete'])
        self.assertEqual(observed['pending_owned_paths'], [str(path)])
        self.assertEqual(value._state['path'], path)
        value.close()
        self.assertFalse(value.directory.exists())
        self.assertFalse(value.snapshot()['cleanup_incomplete'])

    def test_failed_close_leaves_finalizer_retry_without_cyclic_gc(self):
        value = storage.SQLiteIds(self.parent)
        value[KEY] = 1
        directory = value.directory
        state = value._state
        path = state['path']
        real_unlink = Path.unlink
        def unlink(target, *args, **kwargs):
            if target == path:raise PermissionError('fixture denied')
            return real_unlink(target, *args, **kwargs)
        with patch.object(Path, 'unlink', unlink):
            with self.assertRaises(storage.SQLiteStorageError):value.close()
        self.assertTrue(value._finalizer.alive)
        self.assertEqual(state['owned_paths'], {path})
        enabled = gc.isenabled()
        gc.disable()
        try:
            reference = weakref.ref(value)
            del value
            self.assertIsNone(reference())
            self.assertFalse(directory.exists())
            self.assertEqual(state['owned_paths'], set())
            self.assertIsNone(state['path'])
            self.assertFalse(state['cleanup_incomplete'])
        finally:
            if enabled:gc.enable()

    def test_context_keeps_original_interruption_and_discloses_cleanup_failure(self):
        value = self.mapping()
        path = value._state['path']
        interrupted = InterruptedError('original body')
        real_unlink = Path.unlink
        def unlink(target, *args, **kwargs):
            if target == path:raise PermissionError('fixture denied')
            return real_unlink(target, *args, **kwargs)
        def body():
            with value:raise interrupted
        with patch.object(Path, 'unlink', unlink):
            self.assert_interruption(interrupted, body)
        observed = value.snapshot()
        self.assertTrue(observed['closed'])
        self.assertTrue(observed['cleanup_incomplete'])
        self.assertEqual(observed['pending_owned_paths'], [str(path)])
        self.assertEqual(observed['cleanup_errors'][-1]['operation'], 'interrupted.context.close')
        value.close()
        self.assertFalse(value.directory.exists())


if __name__ == '__main__':unittest.main()
