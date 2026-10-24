"""Original samples, deterministic partitions and client-side sequence handling."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from ..config import (MIN_INPUT_SAMPLES, SAMPLE_SPLIT_SEED, SEQUENCE_BUCKET_PERCENTILES,
                      SEQUENCE_MAX_BUCKETS, SEQUENCE_MULTIPLE, SEQUENCE_RATIO_TRIGGER,
                      SEQUENCE_TRUNCATION_TRIGGER)


def split_indices(count):
    if count < MIN_INPUT_SAMPLES:
        raise ValueError(f'至少需要 223 个输入样本，当前 {count} 个')
    order = np.random.default_rng(SAMPLE_SPLIT_SEED).permutation(count)
    calibration = math.ceil(count / 10)
    heldout = (count - calibration) // 10
    return dict(seed=SAMPLE_SPLIT_SEED, calibration=order[:calibration].tolist(),
                heldout=order[calibration:calibration+heldout].tolist(),
                confirm=order[calibration+heldout:calibration+2*heldout].tolist(),
                report=order[calibration+2*heldout:].tolist())


@dataclass
class Samples:
    kind: str
    items: list
    ids: list[str]
    labels: object = None

    def __len__(self):
        return len(self.items)

    def split(self):
        return dict(split_indices(len(self)), sample_ids=list(self.ids))

    def select(self, indices):
        indices = list(indices)
        labels = None if self.labels is None else np.asarray(self.labels)[indices]
        return [self.items[i] for i in indices], labels

    def tensors(self, indices, *, names=None):
        if self.kind != 'tensor':
            raise ValueError('Raw image/text samples require the repository processor/tokenizer')
        items, labels = self.select(indices)
        keys = tuple(items[0])
        result = {name: np.concatenate([item[name] for item in items], axis=0) for name in keys}
        if names is not None:
            if len(names) == 1 and keys == ('input',):
                result = {names[0]: result['input']}
            elif set(result) != set(names):
                raise ValueError(f'Input sample keys {sorted(result)} differ from model inputs {sorted(names)}')
        return result, labels


def _tensor_file(path):
    if path.suffix.lower() == '.npy':
        tensors, labels = {'input': np.load(path, allow_pickle=False)}, None
    else:
        with np.load(path, allow_pickle=False) as archive:
            tensors = {name: np.array(archive[name]) for name in archive.files if name != 'labels'}
            labels = np.array(archive['labels']) if 'labels' in archive.files else None
    if not tensors or any(value.ndim == 0 or value.dtype.kind not in 'biuf' for value in tensors.values()):
        raise ValueError('Tensor samples require numeric arrays with a leading sample dimension')
    count = len(next(iter(tensors.values())))
    if any(len(value) != count for value in tensors.values()) or labels is not None and len(labels) != count:
        raise ValueError('Input/label arrays have different sample counts')
    return [{name: value[i:i+1] for name, value in tensors.items()} for i in range(count)], labels


def read(source):
    path = Path(source).resolve()
    extensions = {'.png', '.jpg', '.jpeg', '.txt', '.jsonl', '.npy', '.npz'}
    files = sorted((p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in extensions),
                   key=lambda p: str(p.relative_to(path))) if path.is_dir() else [path]
    if not files or any(not p.is_file() for p in files):
        raise ValueError(f'No input samples found in {source}')
    kind = lambda p: ('image' if p.suffix.lower() in ('.png', '.jpg', '.jpeg') else
                      'text' if p.suffix.lower() in ('.txt', '.jsonl') else
                      'tensor' if p.suffix.lower() in ('.npy', '.npz') else 'unknown')
    kinds = {kind(p) for p in files}
    if len(kinds) != 1 or 'unknown' in kinds:
        raise ValueError(f'Input directory mixes incompatible sample kinds: {sorted(kinds)}')
    selected_kind = kinds.pop()
    items, ids, labels = [], [], []
    if selected_kind == 'image':
        relative = [p.relative_to(path) for p in files] if path.is_dir() else [Path(files[0].name)]
        classes = sorted({p.parts[0] for p in relative if len(p.parts) > 1})
        labelled = all(len(p.parts) > 1 for p in relative)
        items, ids = files, [str(p) for p in relative]
        labels = [classes.index(p.parts[0]) for p in relative] if labelled else [None] * len(files)
    else:
        for file in files:
            stem = str(file.relative_to(path)) if path.is_dir() else file.name
            if selected_kind == 'tensor':
                values, file_labels = _tensor_file(file)
                items.extend(values)
                ids.extend(f'{stem}:{i:08d}' for i in range(len(values)))
                labels.extend([None] * len(values) if file_labels is None else file_labels.tolist())
            else:
                for lineno, line in enumerate(file.read_text().splitlines()):
                    if not line.strip():
                        continue
                    value = json.loads(line) if file.suffix.lower() == '.jsonl' else line
                    label = value.get('label') if isinstance(value, dict) else None
                    text = value.get('text') if isinstance(value, dict) else value
                    if not isinstance(text, str):
                        raise ValueError('JSONL samples require a string or a text field')
                    items.append(text); ids.append(f'{stem}:{lineno:08d}'); labels.append(label)
    if not items:
        raise ValueError('No nonempty input samples found')
    if any(label is None for label in labels) and not all(label is None for label in labels):
        raise ValueError('Input samples have partially missing labels')
    if selected_kind == 'tensor':
        signatures = {tuple((key, tuple(value.shape)) for key, value in sorted(item.items())) for item in items}
        if len(signatures) != 1:
            raise ValueError(f'Input sample shapes differ: {sorted(signatures)}')
    return Samples(selected_kind, items, ids, None if all(v is None for v in labels) else np.asarray(labels))


def choose_sequence(lengths, max_position_embeddings):
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or not len(lengths) or np.any(lengths < 0):
        raise ValueError('Sequence statistics require nonnegative calibration/heldout lengths')
    cap = int(max_position_embeddings) // SEQUENCE_MULTIPLE * SEQUENCE_MULTIPLE
    if cap < SEQUENCE_MULTIPLE:
        raise ValueError('Model maximum length is smaller than the required length multiple')
    rounded = lambda value: min(cap, max(SEQUENCE_MULTIPLE, math.ceil(float(value)/SEQUENCE_MULTIPLE)*SEQUENCE_MULTIPLE))
    p50, p95 = np.percentile(lengths, (50, 95))
    length = rounded(p95)
    ratio = float(p95/p50) if p50 else (float('inf') if p95 else 1.0)
    truncation = float(np.mean(lengths > length))
    trigger = truncation > SEQUENCE_TRUNCATION_TRIGGER or ratio > SEQUENCE_RATIO_TRIGGER
    buckets = (sorted({rounded(v) for v in np.percentile(lengths, SEQUENCE_BUCKET_PERCENTILES)})
               if trigger else [length])
    buckets = buckets[:SEQUENCE_MAX_BUCKETS]
    assignments = [next((b for b in buckets if n <= b), buckets[-1]) for n in lengths]
    truncated = int(np.sum(lengths > np.asarray(assignments)))
    return dict(length=length, buckets=buckets, bucketed=bool(trigger), p50=float(p50), p95=float(p95),
                padding_side='right', truncation_count=truncated, truncation_fraction=truncated/len(lengths),
                decision_samples=len(lengths), source_splits=['calibration', 'heldout'])


def encode_texts(tokenizer, texts, sequence, *, task):
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'left' if task == 'causal_lm' else 'right'
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer supplies neither pad nor eos token')
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(texts, padding='max_length', truncation=True,
                     max_length=max(sequence['buckets']), return_attention_mask=True, return_tensors='np')


@dataclass
class SampleSplits:
    """Lazy split access: confirmation/report are only loaded on explicit use."""
    samples: Samples
    info: object

    def get(self, split, n=None):
        indices = self.info.split[split]
        if n is not None:
            indices = indices[:n]
        return self.samples.tensors(indices, names=self.info.graph.inputs)


@dataclass
class LegacyCifar:
    """Internal frozen-regression source; deliberately absent from the CLI."""
    graph: object
    normalization: object = None

    def get(self, split, n=None):
        from . import cifar
        normalization = self.normalization or (cifar.MEAN, cifar.STD)
        readers = dict(calibration=cifar.calibration_set, heldout=cifar.heldout_set,
                       confirm=cifar.confirm_set, report=cifar.report_test_set)
        images, labels = (readers[split](n, normalization=normalization)
                          if split == 'heldout' and n is not None else readers[split](normalization=normalization))
        if n is not None:
            images, labels = images[:n], labels[:n]
        name = self.graph.inputs[0]
        images = images.reshape((len(images), *self.graph.tensors[name].shape[1:]))
        return {name: images}, labels


def legacy_cifar(graph, *, normalization=None):
    from . import cifar
    from ..models.info import ModelInfo, metric_for
    metric, threshold = metric_for('classification', has_labels=True)
    info = ModelInfo(dict(type='torch', callable='pt_agent.models.builders:'+graph.name), graph,
                     dict(mode='original tensors', internal_regression_source='legacy_cifar'),
                     [dict(name=name, kind='features', shape=list(graph.tensors[name].shape), dtype='float64')
                      for name in graph.inputs], None, 'classification', metric, threshold,
                     dict(kind='legacy_cifar', heldout=list(cifar.HELDOUT), confirm=list(cifar.CONFIRM),
                          report=list(cifar.REPORT), calibration='frozen training selection'), [])
    return info, LegacyCifar(graph, normalization)


@dataclass
class StoredSplits:
    """Read the numeric exchange format, without importing an intake package."""
    directory: object
    info: object
    prepare_missing: object = None

    def get(self, split, n=None):
        path = Path(self.directory) / (split+'.npz')
        if not path.exists() and self.prepare_missing is not None:
            self.prepare_missing(split)
        if not path.exists():
            raise ValueError(f'Original samples for {split} have not been prepared')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name][:n]) for name in self.info.graph.inputs}
            labels = np.array(archive['labels'][:n]) if 'labels' in archive else None
        return arrays, labels


def save_split(source, split, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, labels = source.get(split)
    np.savez(directory / (split+'.npz'), **arrays, **({'labels': labels} if labels is not None else {}))
