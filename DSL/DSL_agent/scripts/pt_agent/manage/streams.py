"""S1 §11: bounded gzip streams for large compiler intermediates."""

from contextlib import contextmanager
import gzip
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading


IR_LIMIT = 256 * 1024**2
TEMP_INPUT_LIMIT = 1024**3
CHUNK = 1024**2
_sizes = {}
_temporary_lock = threading.Lock()


def admit(directory, min_free_gib=10, *, extra_bytes=0):
    free = shutil.disk_usage(directory).free
    required = min_free_gib * 1024**3 + extra_bytes
    if free < required:
        raise RuntimeError('Trusted runner requires at least 10 GiB free disk space'
                           f' (free_bytes={free}, temporary_bytes={extra_bytes}, required_bytes={required})')
    return free


def remember_size(path, size):
    path = Path(path).resolve()
    stat = path.stat()
    _sizes[path] = (stat.st_size,stat.st_mtime_ns,size)


def uncompressed_size(path):
    """Use the writer's original byte count; read bounded chunks for a fresh process."""
    path = Path(path).resolve()
    stat = path.stat()
    cached = _sizes.get(path)
    if cached is not None and cached[:2]==(stat.st_size,stat.st_mtime_ns):
        return cached[2]
    size = 0
    with gzip.open(path,'rb') as stream:
        while chunk := stream.read(CHUNK):
            size += len(chunk)
    remember_size(path,size)
    return size


def resolve(path):
    path = Path(path)
    zipped = Path(str(path)+'.gz')
    return zipped if not path.exists() and zipped.exists() else path


def open_text(path):
    path = resolve(path)
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()


def output_path(path, size=0, *, compressed=False):
    path = Path(path)
    if (compressed or size >= IR_LIMIT) and path.suffix != '.gz':
        path = Path(str(path)+'.gz')
    return path


@contextmanager
def text_writer(path):
    """Publish a gzip file only after its writer closes successfully."""
    path = Path(path)
    if path.suffix != '.gz':
        with path.open('w') as stream:
            yield stream
        return
    admit(path.parent)
    partial = Path(str(path)+'.part')
    try:
        with gzip.open(partial, 'wt', compresslevel=6) as stream:
            yield stream
            stream.flush()
            size = stream.buffer.tell()
        partial.replace(path)
        remember_size(path,size)
    finally:
        partial.unlink(missing_ok=True)


def compress_ir(path, *, force=False):
    """Validate gzip format before deleting a completed frontend output."""
    path = resolve(path)
    if path.suffix == '.gz' or (not force and path.stat().st_size < IR_LIMIT):
        return path
    admit(path.parent)
    target = Path(str(path)+'.gz')
    size = path.stat().st_size
    partial = Path(str(target)+'.part')
    try:
        with path.open('rb') as source, gzip.open(partial, 'wb', compresslevel=6) as dest:
            shutil.copyfileobj(source, dest, CHUNK)
        with gzip.open(partial, 'rb') as source:
            while source.read(CHUNK):
                pass
        partial.replace(target)
        remember_size(target,size)
        path.unlink()
    finally:
        partial.unlink(missing_ok=True)
    return target


def same_text(a, b):
    """Compare decompressed text as required by S1 §11.1.6."""
    with open_text(a) as left, open_text(b) as right:
        while True:
            x, y = left.read(CHUNK), right.read(CHUNK)
            if x != y:
                return False
            if not x:
                return True


@contextmanager
def _temporary_input(path, directory, size):
    """§11.4: reserve 10 GiB plus this one file before decompression."""
    with _temporary_lock:
        free = admit(directory,extra_bytes=size)
        temporary = None
        info = dict(route='temporary-file',uncompressed_bytes=size,free_before_bytes=free)
        try:
            with tempfile.NamedTemporaryFile(mode='wb',prefix='s1-input-',suffix='.mlir',
                    dir=directory,delete=False) as dest:
                temporary = Path(dest.name)
                info['temporary_path'] = str(temporary)
                with gzip.open(path,'rb') as source:
                    shutil.copyfileobj(source,dest,CHUNK)
            yield temporary,info
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            info['temporary_deleted'] = temporary is None or not temporary.exists()


def run(argv, *, cwd=None, env=None, stdout=None, stderr=None, preexec_fn=None, timeout=None):
    """Choose stdin or the §11.4 temporary-file route for the compressed input."""
    command = list(map(str,argv))
    outputs = [i+1 for i,arg in enumerate(command) if arg=='-o']
    output_index = outputs[-1] if outputs else None
    inputs = [resolve(arg) for i,arg in enumerate(command) if i!=output_index and
              arg.endswith(('.mlir','.mlir.gz')) and resolve(arg).suffix=='.gz' and resolve(arg).is_file()]
    if len(inputs)>1:
        raise ValueError('Expected one compressed IR input')
    keywords = dict(cwd=cwd,env=env,stdout=stdout,stderr=stderr,preexec_fn=preexec_fn,timeout=timeout)
    if not inputs:
        return _run(command,**keywords)
    source = inputs[0]
    directory = cwd or source.parent
    admit(directory)
    try:
        size = uncompressed_size(source)
    except (OSError,EOFError) as exc:
        raise RuntimeError('Compressed IR pipe failed: '+str(exc)) from exc
    if size>TEMP_INPUT_LIMIT:
        with _temporary_input(source,directory,size) as (temporary,info):
            result = _run(command,file_input=temporary,**keywords)
        result.ir_io = info
    else:
        result = _run(command,**keywords)
        result.ir_io = dict(route='stdin',uncompressed_bytes=size)
    return result


def _run(argv, *, cwd=None, env=None, stdout=None, stderr=None, preexec_fn=None, timeout=None, file_input=None):
    """Pipe gzip input/output without buffering a complete IR in memory.

    Explicit pass pipelines produce text through stdout. Named full pipelines
    retain -o because optimizer.cpp derives debug/RuntimePlan paths from it.
    Both pipe workers must finish successfully before publishing an output.
    """
    command = list(map(str, argv))
    input_path = output = None
    outputs = [i+1 for i,arg in enumerate(command) if arg=='-o']
    output_index = outputs[-1] if outputs else None
    for i, arg in enumerate(command):
        if i == output_index or not arg.endswith(('.mlir', '.mlir.gz')):
            continue
        path = resolve(arg)
        if path.suffix == '.gz' and path.is_file():
            if input_path is not None:
                raise ValueError('Expected one compressed IR input')
            input_path, command[i] = path, str(file_input) if file_input is not None else '-'
    if input_path is not None and output_index is not None and '--emit-bytecode' not in command and any(
            arg.startswith('--pass-pipeline=') for arg in command):
        output = output_path(command[output_index], compressed=True)
        command[output_index] = '-'
    if input_path is None and output is None:
        return subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                              preexec_fn=preexec_fn, timeout=timeout)
    admit(cwd or input_path.parent)
    partial = Path(str(output)+'.part') if output else None
    errors = []
    workers = []
    pipe_stderr = stdout if output and stderr == subprocess.STDOUT else stderr
    process = subprocess.Popen(command, cwd=cwd, env=env, preexec_fn=preexec_fn, start_new_session=True,
                               stdin=subprocess.PIPE if input_path and file_input is None else None,
                               stdout=subprocess.PIPE if output else stdout, stderr=pipe_stderr)

    def kill():
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass

    def feed():
        try:
            with gzip.open(input_path, 'rb') as source, process.stdin as dest:
                shutil.copyfileobj(source, dest, CHUNK)
        except Exception as exc:
            errors.append(exc)
            kill()

    def receive():
        try:
            size = 0
            with process.stdout as source, gzip.open(partial, 'wb', compresslevel=6) as dest:
                while chunk := source.read(CHUNK):
                    dest.write(chunk)
                    size += len(chunk)
            remember_size(partial,size)
        except Exception as exc:
            errors.append(exc)
            kill()

    try:
        for action in ((feed,) if input_path and file_input is None else ()) + ((receive,) if output else ()):
            thread = threading.Thread(target=action, daemon=True)
            thread.start()
            workers.append(thread)
        code = process.wait(timeout=timeout)
        for thread in workers:
            thread.join()
        if errors:
            raise RuntimeError('Compressed IR pipe failed: '+str(errors[0])) from errors[0]
        if code == 0 and output:
            size = uncompressed_size(partial)
            partial.replace(output)
            remember_size(output,size)
        return subprocess.CompletedProcess(command, code)
    finally:
        if process.poll() is None:
            kill()
            process.wait()
        for thread in workers:
            thread.join()
        if partial:
            partial.unlink(missing_ok=True)
