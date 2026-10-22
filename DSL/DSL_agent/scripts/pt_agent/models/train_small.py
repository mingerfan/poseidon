"""Train the two fixed R8a models on the existing CIFAR-10 training records."""

import argparse
import json
import math
from pathlib import Path
import resource
import signal
import time

RECIPE = "v2-clip1.0"
CLIP_NORM = 1.0
DIVERGENCE_FACTOR = 1.5


def _write(path, record):
    if record.get('error'):
        record = dict(record, epochs=[dict(row, loss=str(row['loss']))
                      if not math.isfinite(row['loss']) else row for row in record['epochs']])
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def augment(batch, rng):
    import numpy as np

    padded = np.pad(batch, ((0, 0), (0, 0), (4, 4), (4, 4)))
    offsets = rng.integers(0, 9, size=(len(batch), 2))
    result = np.stack([padded[i, :, top:top+32, left:left+32]
                       for i, (top, left) in enumerate(offsets)])
    flipped = rng.random(len(batch)) < .5
    result[flipped] = result[flipped, :, :, ::-1]
    return result


def check_divergence(losses: list[float]) -> None:
    """Reject a nonfinite last loss or growth above 1.5 times the first loss."""
    if not math.isfinite(losses[-1]) or losses[-1] > DIVERGENCE_FACTOR*losses[0]:
        raise RuntimeError(f"training diverged at epoch {len(losses)}: loss {losses[-1]} vs first {losses[0]}")


def run_epochs(model, images, labels, epochs, rng, augment_fn=None, report=None):
    import torch
    from torch import nn

    if report is None:
        report = dict(epochs=[], partial_epoch=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        order = rng.permutation(len(images))
        loss_sum = correct = done = 0
        lr = optimizer.param_groups[0]['lr']
        for start in range(0, len(images), 128):
            indices = order[start:start+128]
            batch = images[indices]
            if augment_fn is not None:
                batch = augment_fn(batch, rng)
            targets = torch.from_numpy(labels[indices])
            logits = model(torch.from_numpy(batch))
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            loss_sum += float(loss.detach())*len(indices)
            correct += int((logits.detach().argmax(1) == targets).sum())
            done += len(indices)
            report['partial_epoch'] = dict(epoch=epoch+1, images_done=done, loss=loss_sum/done,
                                           accuracy=100.0*correct/done, seconds=time.monotonic()-epoch_started)
        scheduler.step()
        row = dict(epoch=epoch+1, loss=loss_sum/done, accuracy=100.0*correct/done,
                   seconds=time.monotonic()-epoch_started, learning_rate=lr,
                   next_learning_rate=optimizer.param_groups[0]['lr'])
        report['epochs'].append(row)
        report['partial_epoch'] = None
        print('TRAIN '+report.get('model', type(model).__name__)+' '+json.dumps(row), flush=True)
        check_divergence([entry['loss'] for entry in report['epochs']])
    return report


def train(name, out_dir):
    import numpy as np
    import torch
    from ..data.cifar import read_rows, test_set
    from .zoo import CnnPlain, IMAGENET_NORMALIZATION, Mlp, REPO

    out_dir = Path(out_dir).resolve()
    if out_dir.is_relative_to(REPO):
        raise ValueError('Training weights must be outside the repository')
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path, report_path = out_dir/(name+'.pt'), out_dir/(name+'-train.json')
    if weights_path.exists() or report_path.exists():
        raise FileExistsError('Existing training weights and reports must be preserved')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 4*1024**3
    resource.setrlimit(resource.RLIMIT_AS, (min(cap, soft) if soft != resource.RLIM_INFINITY else cap, hard))
    torch.manual_seed(20261022)
    rng = np.random.default_rng(20261022)
    torch.set_num_threads(2)
    model = CnnPlain() if name == 'cnn_plain' else Mlp()
    epochs = 15 if name == 'cnn_plain' else 20
    report = dict(model=name, complete=False, timeout=False, recipe=RECIPE, clip_norm=CLIP_NORM,
                  seed=20261022, train_images=50000,
                  batch=128, threads=2, epochs_requested=epochs, epochs=[], partial_epoch=None,
                  torch_top1=None, test_images_done=0, correct_test=0, seconds=None)
    _write(report_path, report)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError('R8a training exceeded 60 minutes')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(3600)
    exit_code = 0
    try:
        pixels, labels = read_rows('train', range(50000))
        images = pixels.astype(np.float32)
        del pixels
        images /= 255
        images -= np.asarray(IMAGENET_NORMALIZATION[0], dtype=np.float32).reshape(1, 3, 1, 1)
        images /= np.asarray(IMAGENET_NORMALIZATION[1], dtype=np.float32).reshape(1, 3, 1, 1)
        if name == 'mlp':
            images = images.reshape(len(images), 3072)
        run_epochs(model, images, labels, epochs, rng,
                   augment_fn=augment if name == 'cnn_plain' else None, report=report)
        _write(report_path, report)
        del images, labels
        model.eval()
        images, labels = test_set(normalization=IMAGENET_NORMALIZATION)
        if name == 'mlp':
            images = images.reshape(len(images), 3072)
        with torch.no_grad():
            for start in range(0, len(images), 128):
                batch = torch.from_numpy(images[start:start+128]).float()
                predictions = model(batch).argmax(1).numpy()
                report['correct_test'] += int(np.sum(predictions == labels[start:start+len(predictions)]))
                report['test_images_done'] = start+len(predictions)
        report.update(torch_top1=100.0*report['correct_test']/len(images), complete=True)
    except TimeoutError:
        report['timeout'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        exit_code = 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        temporary = weights_path.with_name(weights_path.name+'.tmp')
        torch.save(model.state_dict(), temporary)
        temporary.replace(weights_path)
        report['seconds'] = time.monotonic()-started
        _write(report_path, report)
    print('TRAIN_REPORT='+str(report_path), flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--model', choices=('cnn_plain', 'mlp'), required=True)
    args = parser.parse_args()
    return train(args.model, args.out_dir)


if __name__ == '__main__':
    raise SystemExit(main())
