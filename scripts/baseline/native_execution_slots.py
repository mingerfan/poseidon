"""Cross-process native-stage limit; no slot is held during model API calls.

Lock files are reusable cache, not evidence or credentials. Kernel flock releases
ownership on process exit; files must not be deleted while any runner is active.
"""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import stat
import time

NATIVE_CONCURRENCY = 2


@contextmanager
def native_slot(directory, *, timeout=600, metrics=None):
    import fcntl

    if not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Native slot timeout must be finite and positive')
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ValueError('Unsafe native slot directory')
    started = time.monotonic()
    handle = None
    try:
        while handle is None:
            for index in range(NATIVE_CONCURRENCY):
                fd = os.open(directory / f'slot-{index}.lock',
                             os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o022:
                        raise ValueError('Unsafe native slot file')
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    handle = fd
                    break
                finally:
                    if handle != fd:
                        os.close(fd)
            if handle is None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError('Native execution slots busy; no unbounded wait')
                time.sleep(min(0.05, remaining))
        if metrics is not None:
            metrics['acquisitions'] = metrics.get('acquisitions', 0) + 1
            metrics['wait_seconds'] = metrics.get('wait_seconds', 0.0) + time.monotonic() - started
        yield
    finally:
        if handle is not None:
            os.close(handle)
