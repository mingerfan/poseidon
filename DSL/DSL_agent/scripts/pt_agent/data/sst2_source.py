"""Lazy SST-2 inputs with fixed training partitions and a frozen-only report.

The dataset's native validation partition supplies the labelled report.  Only
Parquet metadata is inspected before an explicit selection; the hidden-label
native test partition is never opened by this adapter.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys


VERSION = 'sst2-native-v1'
SEED = 20261024
INTERNAL_SAMPLES = 1000
REPORT_SAMPLES = 872
SPLITS = ('calibration', 'heldout', 'confirm', 'report')


def reader_environment():
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(('LD_', 'NIX_', '__NIX_'))
                   and name not in ('IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    return environment


class ExternalReader:
    """Use the existing Parquet environment without changing either venv."""
    def __init__(self, python):
        self.python = Path(python).expanduser().absolute()
        if not (self.python.is_file() or self.python.is_symlink()):
            raise ValueError('The installed independent Parquet interpreter is required')

    def _request(self, operation, path, indices=()):
        completed = subprocess.run([str(self.python), '-B', '-m', 'pt_agent.data.sst2_source',
            '--operation', operation, '--source', str(path)], input=json.dumps(list(indices)),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
            cwd=Path(__file__).resolve().parents[2], env=reader_environment())
        if completed.returncode:
            raise ValueError('Independent SST-2 reader failed: ' + completed.stderr[-2000:])
        return json.loads(completed.stdout)

    def metadata(self, path):
        return self._request('metadata', path)

    def rows(self, path, indices):
        return {int(key): tuple(value) for key, value in self._request('rows', path, indices).items()}


def resolve_files(source):
    source = Path(source).expanduser().resolve()
    data = source / 'data' if (source / 'data').is_dir() else source
    files = {}
    for split in ('train', 'validation'):
        matches = sorted(data.glob(split + '-*.parquet'))
        if len(matches) != 1 or not matches[0].is_file():
            raise ValueError('SST-2 requires one original ' + split + ' Parquet shard')
        files[split] = matches[0]
    return files


def parquet_metadata(path):
    import pyarrow.parquet as parquet

    reader = parquet.ParquetFile(path)
    names = set(reader.schema_arrow.names)
    if not {'sentence', 'label'}.issubset(names):
        raise ValueError('SST-2 requires the native sentence and label columns')
    return dict(path=str(Path(path).resolve()), rows=reader.metadata.num_rows,
                columns=['sentence', 'label'], bytes=Path(path).stat().st_size)


def make_partition(train_rows, validation_rows, *, internal_samples=INTERNAL_SAMPLES, seed=SEED):
    import numpy as np

    if (type(internal_samples) is not int or internal_samples < 4
            or type(train_rows) is not int or train_rows < 3 * internal_samples
            or type(validation_rows) is not int or validation_rows <= 0):
        raise ValueError('SST-2 partition requires enough original labelled rows')
    chosen = np.random.default_rng(seed).permutation(train_rows)[:3 * internal_samples].tolist()
    ids = ['train:' + str(index) for index in chosen]
    ids += ['validation:' + str(index) for index in range(validation_rows)]
    return dict(kind='sst2_native', version=VERSION, seed=seed,
                calibration=list(range(internal_samples)),
                heldout=list(range(internal_samples, 2 * internal_samples)),
                confirm=list(range(2 * internal_samples, 3 * internal_samples)),
                report=list(range(3 * internal_samples, len(ids))), sample_ids=ids,
                training_rows=train_rows, unused_training_rows=train_rows - len(chosen),
                report_source='native validation',
                policy='fixed seeded training selection without replacement; native validation after freeze')


def read_rows(path, indices):
    """Read requested native rows without materializing any other partition."""
    import pyarrow.parquet as parquet

    requested = set(indices)
    if any(type(index) is not int or index < 0 for index in requested):
        raise ValueError('SST-2 row indices must be nonnegative integers')
    if not requested:
        return {}
    reader = parquet.ParquetFile(path)
    if max(requested) >= reader.metadata.num_rows:
        raise ValueError('SST-2 requested row is outside its original shard')
    found, offset = {}, 0
    for batch in reader.iter_batches(batch_size=4096, columns=['sentence', 'label']):
        selected = sorted(index for index in requested if offset <= index < offset + len(batch))
        if selected:
            for index in selected:
                row = index - offset
                text = batch.column(0)[row].as_py()
                label = batch.column(1)[row].as_py()
                if not isinstance(text, str) or not text.strip() or type(label) is not int or label not in (0, 1):
                    raise ValueError('SST-2 selected row requires real text and a binary native label')
                found[index] = (text, label)
        offset += len(batch)
        if len(found) == len(requested):
            break
    if set(found) != requested:
        raise ValueError('SST-2 selected rows are missing')
    return found


def require_frozen(marker, program, *, split_version=VERSION):
    marker, program = Path(marker).resolve(), Path(program).resolve()
    record = json.loads(marker.read_text())
    original = Path(record.get('program', '')).resolve()
    if (not original.is_file() or not program.is_file()
            or original.read_text() != program.read_text()
            or record.get('split_version') != split_version):
        raise ValueError('SST-2 report requires the unchanged program frozen on this partition')
    return record


class SST2Samples:
    kind = 'text'

    def __init__(self, source, *, record=None, internal_samples=INTERNAL_SAMPLES,
                 expected_report=REPORT_SAMPLES, data_python=None):
        import numpy as np

        self.source = Path(source).expanduser().resolve()
        self.files = resolve_files(self.source)
        configured = data_python if data_python is not None else (record or {}).get('reader_python')
        self.reader = ExternalReader(configured) if configured is not None else None
        metadata = {name: (self.reader.metadata(path) if self.reader else parquet_metadata(path))
                    for name, path in self.files.items()}
        if metadata['validation']['rows'] != expected_report:
            raise ValueError('SST-2 native validation row count differs from the fixed source')
        if record is None:
            self.partition = make_partition(metadata['train']['rows'], expected_report,
                                            internal_samples=internal_samples)
        else:
            if Path(record['source']).resolve() != self.source or record.get('metadata') != metadata:
                raise ValueError('Saved SST-2 source metadata changed')
            saved = record['partition']
            self.partition = make_partition(metadata['train']['rows'], expected_report,
                internal_samples=len(saved['calibration']), seed=saved['seed'])
            if self.partition != saved:
                raise ValueError('Saved SST-2 partition changed')
        self.metadata = metadata
        self.ids = list(self.partition['sample_ids'])
        # Availability follows the native schema; unretrieved labels stay -1.
        self.labels = np.full(len(self.ids), -1, dtype=np.int64)
        self._cache = {}
        self.report_authorized = False
        self.accesses = []

    def __len__(self):
        return len(self.ids)

    def split(self):
        return copy.deepcopy(self.partition)

    def record(self):
        return dict(source=str(self.source), metadata=copy.deepcopy(self.metadata),
                    partition=self.split(), native_test_read=False,
                    reader_python=str(self.reader.python) if self.reader else None)

    def authorize_report(self, marker, program):
        require_frozen(marker, program)
        self.report_authorized = True

    def select(self, indices):
        import numpy as np

        indices = list(indices)
        if any(type(index) is not int or not 0 <= index < len(self) for index in indices):
            raise ValueError('SST-2 selected sample index is invalid')
        if any(self.ids[index].startswith('validation:') for index in indices) and not self.report_authorized:
            raise ValueError('SST-2 validation content is available after freezing')
        for split in ('train', 'validation'):
            missing = [index for index in indices if index not in self._cache
                       and self.ids[index].startswith(split + ':')]
            if missing:
                rows = [int(self.ids[index].split(':')[1]) for index in missing]
                values = self.reader.rows(self.files[split], rows) if self.reader else read_rows(self.files[split], rows)
                self.accesses.append(dict(split=split, rows=list(rows)))
                for index, row in zip(missing, rows):
                    self._cache[index] = values[row]
                    self.labels[index] = values[row][1]
        return [self._cache[index][0] for index in indices], np.asarray([self._cache[index][1] for index in indices], dtype=np.int64)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Read selected original SST-2 data in the installed data environment')
    parser.add_argument('--operation', choices=('metadata', 'rows'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source.name.startswith('test-'):
        raise ValueError('The hidden-label SST-2 native test shard is outside this source')
    value = parquet_metadata(args.source) if args.operation == 'metadata' else read_rows(args.source, json.loads(sys.stdin.read()))
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
