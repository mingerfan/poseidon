"""Carry the original10800-second option into an isolated ordered step worker."""
import argparse
from pathlib import Path
import sys

from . import bert_ordered_tuner_step as original
from . import bert_sst2_recorded as recorded


def delegate(argv, metadata_template, qualification):
    saved = recorded.worker_delegate

    def worker(args, template):
        return saved(args, template, step_seconds=10800)

    # Only this new numeric process owns the adapter. The original main driver
    # and other processes retain their own unchanged imported module objects.
    recorded.worker_delegate = worker
    try:
        return original.delegate(argv, metadata_template, qualification)
    finally:
        recorded.worker_delegate = saved


def main(argv=None):
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument('--delegate-worker', action='store_true', required=True)
    cli.add_argument('--step-seconds', type=int, choices=(10800,), required=True)
    cli.add_argument('--metadata-template', type=Path, required=True)
    cli.add_argument('--ordered-qualification', type=Path, required=True)
    args, remaining = cli.parse_known_args(sys.argv[1:] if argv is None else argv)
    return delegate(remaining, args.metadata_template.resolve(), args.ordered_qualification.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
