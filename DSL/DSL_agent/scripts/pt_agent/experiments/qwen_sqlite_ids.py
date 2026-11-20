"""Standalone exact scalar CSE storage. No Agent/worker admission is provided.

Keys contain only the original CostDag typed fields; immutable operand sharing
does not affect their canonical semantic encoding. SQLite holds scalar metadata,
never model arrays. An owned database is temporary and is never resumed.
"""
from collections.abc import MutableMapping
from pathlib import Path
import sqlite3
import sys
import tempfile
import weakref

PREFIX = b'QSI\x01'
PAGE_BYTES = 4096
CACHE_KIB = 2048
CHECKPOINT_MUTATIONS = 256
DEFAULT_MAX_BYTES = 64*1024**2


class SQLiteStorageError(RuntimeError):
    pass


def _uint(value):
    if type(value) is not int or value < 0:raise ValueError('Nonnegative exact int required')
    result = bytearray()
    while value >= 128:
        result.append((value&127)|128);value >>= 7
    result.append(value)
    return bytes(result)


def _read_uint(blob,position):
    value,shift,start = 0,0,position
    while position < len(blob):
        byte = blob[position];position += 1;value |= (byte&127)<<shift
        if byte < 128:
            if position-start > 1 and byte == 0:raise ValueError('Nonminimal integer')
            return value,position
        shift += 7
    raise ValueError('Truncated integer')


def _text(value):
    if type(value) is not str:raise ValueError('Exact string required')
    data = value.encode('utf-8','surrogatepass')
    return _uint(len(data))+data


def _read_text(blob,position):
    length,position = _read_uint(blob,position);end = position+length
    if end > len(blob):raise ValueError('Truncated string')
    return blob[position:end].decode('utf-8','surrogatepass'),end


def encode_key(key):
    if type(key) is not tuple or len(key) != 3:raise ValueError('Original three-field tuple required')
    kind,operands,step = key
    if type(operands) is not tuple:raise ValueError('Ordered operand tuple required')
    result = bytearray(PREFIX+_text(kind)+_uint(len(operands)))
    for operand in operands:
        if type(operand) is not tuple or len(operand) != 2:raise ValueError('Tagged operand pair required')
        tag,value = operand
        if type(tag) is not str:raise ValueError('Exact operand tag required')
        if tag == 'cipher':result += b'C'+_uint(value)
        elif tag == 'plain':result += b'P'+_text(value)
        else:raise ValueError('Original cipher/plain operand tag required')
    if step is None:result += b'N'
    elif type(step) is int:result += b'I'+bytes((int(step<0),))+_uint(abs(step))
    else:raise ValueError('Exact int or None step required')
    return bytes(result)


def decode_key(blob):
    if type(blob) is not bytes or not blob.startswith(PREFIX):raise ValueError('Encoded key version required')
    kind,position = _read_text(blob,len(PREFIX));count,position = _read_uint(blob,position);operands = []
    for _ in range(count):
        if position == len(blob):raise ValueError('Truncated operand')
        tag = blob[position:position+1];position += 1
        if tag == b'C':value,position = _read_uint(blob,position);name = 'cipher'
        elif tag == b'P':value,position = _read_text(blob,position);name = 'plain'
        else:raise ValueError('Invalid encoded operand tag')
        operands.append((name,value))
    if position == len(blob):raise ValueError('Missing step')
    tag = blob[position:position+1];position += 1
    if tag == b'N':step = None
    elif tag == b'I':
        if position == len(blob) or blob[position] not in (0,1):raise ValueError('Integer sign required')
        sign = blob[position];value,position = _read_uint(blob,position+1)
        if sign and value == 0:raise ValueError('Negative zero is noncanonical')
        step = -value if sign else value
    else:raise ValueError('Invalid step tag')
    if position != len(blob):raise ValueError('Trailing semantic fields')
    return kind,tuple(operands),step


def _cleanup_error(state,operation,path,exc):
    state.setdefault('cleanup_errors',[]).append(dict(operation=operation,path=str(path),error=type(exc).__name__))
    state['cleanup_incomplete'] = True


def _release(state,*,directory=False):
    errors = [];connection = state.get('connection')
    if connection is not None:
        try:connection.close();state['connection'] = None
        except InterruptedError as exc:
            _cleanup_error(state,'connection.close',state.get('path'),exc);raise
        except (sqlite3.Error,OSError) as exc:
            _cleanup_error(state,'connection.close',state.get('path'),exc);errors.append(type(exc).__name__)
    if state.get('connection') is None:
        path = state.get('path')
        if path is not None:
            owned = state.setdefault('owned_paths',set((path,Path(str(path)+'-journal'),Path(str(path)+'-wal'),Path(str(path)+'-shm'))))
            for target in tuple(owned):
                try:target.unlink(missing_ok=True)
                except InterruptedError as exc:
                    _cleanup_error(state,'unlink',target,exc);raise
                except OSError as exc:
                    _cleanup_error(state,'unlink',target,exc);errors.append(type(exc).__name__)
                else:owned.remove(target)
            if not owned:state['path'] = None
        if directory and not state.get('owned_paths'):
            state['directory_pending'] = True
            try:state['directory'].rmdir()
            except InterruptedError as exc:
                _cleanup_error(state,'rmdir',state['directory'],exc);raise
            except FileNotFoundError:state['directory_pending'] = False
            except OSError as exc:
                _cleanup_error(state,'rmdir',state['directory'],exc);errors.append(type(exc).__name__)
            else:state['directory_pending'] = False
    state['cleanup_incomplete'] = bool(state.get('connection') is not None or state.get('owned_paths') or state.get('directory_pending'))
    return errors


def _finalize(state):
    # Explicit close/finally is primary; this callback never retains the owner.
    _release(state,directory=True)


class SQLiteIds(MutableMapping):
    """Exact typed key -> Python int mapping, ordered and bounded on disk.

    The main DB has a hard page cap. Combined DB/journal sizes are checked every
    256 mutations and at flush; temporary inter-checkpoint journal growth is not
    a hard combined-file bound. Any SQL/storage failure closes and discards this
    entire ephemeral mapping. No reconnect/recovery operation exists.
    """
    def __init__(self,parent,*,max_bytes=DEFAULT_MAX_BYTES):
        if type(max_bytes) is not int or not 4*PAGE_BYTES <= max_bytes <= 16*1024**3:
            raise ValueError('Finite exact integer disk allowance required')
        parent = Path(parent).resolve(strict=True)
        if not parent.is_dir():raise ValueError('Existing local parent directory required')
        self.directory = Path(tempfile.mkdtemp(prefix='qwen-cse-',dir=parent))
        self._state = dict(directory=self.directory,connection=None,path=None,owned_paths=set(),
                           cleanup_errors=[],cleanup_incomplete=False,directory_pending=False)
        self._finalizer = weakref.finalize(self,_finalize,self._state)
        self.max_bytes = max_bytes;self._length = self._next = self._version = self._generation = 0
        self._pending = self._reads = self._writes = self._commits = self._peak_disk = 0
        self._closed = False;self.failure = None

    def _alive(self):
        if self._closed:raise SQLiteStorageError('Ephemeral mapping is closed; create a new mapping')

    def _connection(self):
        self._alive()
        if self._state['connection'] is None:
            self._generation += 1;path = self.directory/('ids-'+str(self._generation)+'.sqlite')
            with path.open('xb'):pass
            self._state['path'] = path
            self._state['owned_paths'] = set((path,Path(str(path)+'-journal'),Path(str(path)+'-wal'),Path(str(path)+'-shm')))
            connection = sqlite3.connect(path,isolation_level=None,cached_statements=8)
            self._state['connection'] = connection
            for pragma in ('page_size='+str(PAGE_BYTES),'max_page_count='+str(self.max_bytes//PAGE_BYTES),
                           'cache_size=-'+str(CACHE_KIB),'mmap_size=0','temp_store=FILE','journal_mode=DELETE'):
                cursor = connection.execute('PRAGMA '+pragma);cursor.close()
            connection.execute('CREATE TABLE ids(seq INTEGER PRIMARY KEY,key BLOB UNIQUE NOT NULL,nid BLOB NOT NULL)').close()
        return self._state['connection']

    def _fail(self,exc):
        self.failure = type(exc).__name__;self._closed = True;self._length = 0;self._version += 1
        errors = _release(self._state,directory=True)
        suffix = '; cleanup incomplete: '+','.join(errors) if errors else ''
        raise SQLiteStorageError('Ephemeral CSE storage failed: '+type(exc).__name__+suffix) from exc

    def _disk(self):
        total = sum(path.stat().st_size for path in self.directory.iterdir() if path.is_file())
        self._peak_disk = max(self._peak_disk,total)
        if total > self.max_bytes:raise OSError('Combined CSE disk allowance exceeded')
        return total

    def flush(self):
        self._alive()
        try:
            self._disk();connection = self._state['connection']
            if connection is not None and connection.in_transaction:
                connection.execute('COMMIT').close();self._commits += 1
            self._pending = 0;self._disk()
        except InterruptedError:raise
        except (sqlite3.Error,OSError) as exc:self._fail(exc)

    def _lookup(self,encoded):
        self._alive();self._reads += 1
        if self._state['connection'] is None:return None
        try:
            cursor = self._state['connection'].execute('SELECT nid FROM ids WHERE key=?',(encoded,))
            interruption = None
            try:return cursor.fetchone()
            except InterruptedError as exc:interruption = exc;raise
            finally:
                try:cursor.close()
                except BaseException as exc:
                    if interruption is None:raise
                    _cleanup_error(self._state,'lookup.cursor.close',self._state.get('path'),exc)
        except sqlite3.Error as exc:self._fail(exc)

    def __contains__(self,key):return self._lookup(encode_key(key)) is not None

    def __getitem__(self,key):
        row = self._lookup(encode_key(key))
        if row is None:raise KeyError(key)
        value,end = _read_uint(row[0],0)
        if end != len(row[0]):raise SQLiteStorageError('Invalid scalar nid representation')
        return value

    def _mutate(self,operation,key,value=None):
        encoded = encode_key(key);nid = _uint(value) if operation=='set' else None
        try:
            connection = self._connection()
            if not connection.in_transaction:connection.execute('BEGIN IMMEDIATE').close()
            if operation=='set':
                cursor = connection.execute('UPDATE ids SET nid=? WHERE key=?',(nid,encoded))
                found = cursor.rowcount;cursor.close()
                if not found:
                    connection.execute('INSERT INTO ids(seq,key,nid) VALUES(?,?,?)',(self._next,encoded,nid)).close()
                    self._next += 1;self._length += 1;self._version += 1
            else:
                cursor = connection.execute('DELETE FROM ids WHERE key=?',(encoded,))
                found = cursor.rowcount;cursor.close()
                if not found:raise KeyError(key)
                self._length -= 1;self._version += 1
            self._writes += 1;self._pending += 1
            if self._pending >= CHECKPOINT_MUTATIONS:self.flush()
        except InterruptedError:raise
        except (sqlite3.Error,OSError) as exc:self._fail(exc)

    def __setitem__(self,key,value):self._mutate('set',key,value)
    def __delitem__(self,key):self._mutate('delete',key)
    def __len__(self):return self._length

    def __iter__(self):
        self._alive();version = self._version
        def rows():
            cursor = None
            try:
                if version != self._version:raise RuntimeError('CSE mapping changed during iteration')
                if self._state['connection'] is not None:
                    cursor = self._state['connection'].execute('SELECT key FROM ids ORDER BY seq')
                    while True:
                        if version != self._version:raise RuntimeError('CSE mapping changed during iteration')
                        row = cursor.fetchone()
                        if row is None:break
                        yield decode_key(row[0])
                if version != self._version:raise RuntimeError('CSE mapping changed during iteration')
            except InterruptedError:raise
            except (sqlite3.Error,OSError) as exc:self._fail(exc)
            finally:
                if cursor is not None:
                    interruption = sys.exc_info()[1]
                    try:cursor.close()
                    except InterruptedError as exc:
                        _cleanup_error(self._state,'iterator.cursor.close',self._state.get('path'),exc)
                        if not isinstance(interruption,InterruptedError):raise
                    except (sqlite3.Error,OSError) as exc:
                        _cleanup_error(self._state,'iterator.cursor.close',self._state.get('path'),exc)
                        if not isinstance(interruption,InterruptedError) and not isinstance(exc,sqlite3.Error):raise
        return rows()

    def clear(self):
        self._length = self._next = self._pending = 0;self._version += 1
        try:errors = _release(self._state)
        except InterruptedError:self._closed = True;raise
        if errors:self._closed = True;raise SQLiteStorageError('CSE clear failed: '+','.join(errors))

    def close(self):
        self.clear();self._closed = True;errors = _release(self._state,directory=True)
        if errors:raise SQLiteStorageError('CSE close failed: '+','.join(errors))
        self._finalizer.detach()

    def snapshot(self):
        return dict(entries=self._length,generation=self._generation,closed=self._closed,failure=self.failure,
            cache_target_kib=CACHE_KIB,mmap_bytes=0,max_main_DB_bytes=(self.max_bytes//PAGE_BYTES)*PAGE_BYTES,
            combined_disk_checkpoint_allowance_bytes=self.max_bytes,checkpoint_mutations=CHECKPOINT_MUTATIONS,
            peak_combined_disk_observed_bytes=self._peak_disk,reads=self._reads,writes=self._writes,commits=self._commits,
            cleanup_incomplete=self._state['cleanup_incomplete'],cleanup_errors=list(self._state['cleanup_errors']),
            pending_owned_paths=sorted(str(path)for path in self._state['owned_paths']),
            connection_pending_close=self._state['connection'] is not None and self._closed,
            directory_pending=self._state['directory_pending'])

    def __enter__(self):self._alive();return self
    def __exit__(self,*error):
        if isinstance(error[1],InterruptedError):
            try:self.close()
            except BaseException as exc:_cleanup_error(self._state,'interrupted.context.close',self.directory,exc)
            return False
        self.close()
