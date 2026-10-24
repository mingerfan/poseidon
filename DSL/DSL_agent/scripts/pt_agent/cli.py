"""Original model + original inputs. All descriptive information is inferred."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .paths import PLATFORM_DIR, RESULTS_DIR


STAGES = ('probe', 'import', 'info', 'calibrate', 'plan', 'check', 'mock', 'count', 'export', 'compile', 'test')


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def _prepare(args):
    from .data.samples import read, save_split
    from .intake.probe import import_model
    try:
        imported = import_model(args.model, read(args.inputs))
    except Exception as exc:
        from collections import Counter
        entries = getattr(exc, 'entries', [])
        result = dict(complete=False, stage='import', error=f'{type(exc).__name__}: {exc}',
                      unmapped=dict(Counter(entry.get('target', entry['op']) for entry in entries)),
                      diagnostics=entries)
        _write(args.out/'intake.json', result)
        _write(args.out/'REPORT.json', result)
        return 1
    imported.info.source['inputs_path'] = str(args.inputs.resolve())
    imported.info.save(args.out)
    for split in ('calibration', 'heldout'):
        save_split(imported.samples, split, args.out/'samples')
    from .intake.coverage import report, check
    result = dict(complete=True, model_info='model_info.json', coverage=report(imported.info.graph),
                  graph_check=check(imported.info.graph))
    _write(args.out/'intake.json', result)
    return 0


def _prepare_split(args):
    from .data.samples import read, save_split, SampleSplits
    from .models.info import ModelInfo
    info = ModelInfo.load(args.out)
    raw = read(args.inputs)
    if info.source['type'] == 'hf':
        from .intake.hf_frontend import HFInputs, load_local
        source = HFInputs(load_local(info.source['local_path'], attn_implementation='eager'), raw, info)
    else:
        source = SampleSplits(raw, info)
    save_split(source, args.split, args.out/'samples')
    return 0


def _main_stages(args):
    from types import SimpleNamespace
    from .analyzer import calibrate
    from .data.samples import StoredSplits
    from .models.info import ModelInfo
    from .intake.coverage import check
    info = ModelInfo.load(args.out)
    if 'info' in args.stages:
        record = json.loads((args.out/'model_info.json').read_text())
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    checked = check(info.graph)
    _write(args.out/'graph_check.json', checked)
    if any(stage in args.stages for stage in ('check','calibrate','plan','mock','count','export','compile','test')) and not checked['ok']:
        _write(args.out/'REPORT.json', dict(complete=False, stage='check', graph_check=checked))
        return 1
    samples = StoredSplits(args.out/'samples', info)
    if 'calibrate' in args.stages:
        if not checked['ok']:
            _write(args.out/'REPORT.json', dict(complete=False, stage='check', graph_check=checked))
            return 1
        arrays, _ = samples.get('calibration')
        _write(args.out/'calibration.json', calibrate(info.graph, arrays))
    stages = [stage for stage in args.stages if stage not in ('probe', 'import', 'info', 'calibrate')]
    if stages == ['check'] and not (args.out/'program.ptdsl').exists():
        _write(args.out/'REPORT.json', dict(complete=True, stage='check', graph_check=checked))
        return 0
    if not stages:
        _write(args.out/'REPORT.json', dict(complete=True, stage='complete', stages=args.stages,
                                          model_info='model_info.json', graph_check=checked))
        return 0
    from .run import _inside
    command_prefix = [args.intake_python, '-B', '-m', 'pt_agent.cli'] if info.source['type'] in ('hf', 'onnx') else [sys.executable, '-B', '-m', 'pt_agent.cli']

    def prepare_missing(split):
        command = [*command_prefix, '--_worker', 'split', '--model', args.model, '--inputs', str(args.inputs),
                   '--out', str(args.out), '--_split', split, '--_intake-python', args.intake_python]
        subprocess.run(command, check=True, timeout=600)

    # Reuse the existing deterministic stage entry. This U1 intake command has
    # no LLM invocation; a future generation round can supply the frozen planner.
    options = SimpleNamespace(out=args.out, stages=stages, model=info.graph.name,
        model_info=args.out/'model_info.json', prepare_split=prepare_missing,
        models_dir=args.out, planner='rule', deg=args.deg, margin=args.margin,
        refresh='compiler', act_method='cheb', final='test' in stages)
    return _inside(options)


def _nix_worker(args, worker):
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '3600', '--',
               '-m', 'pt_agent.cli', '--_worker', worker, '--model', args.model,
               '--inputs', str(args.inputs), '--out', str(args.out),
               '--stages', ','.join(args.stages), '--_intake-python', args.intake_python,
               '--_deg', str(args.deg), '--_margin', str(args.margin)]
    with (args.out/(worker+'.log')).open('w') as log:
        result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent,
                                stdout=log, stderr=subprocess.STDOUT, timeout=3720)
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--stages', default='probe,import,info')
    parser.add_argument('--_worker', choices=('prepare', 'main', 'split'), help=argparse.SUPPRESS)
    parser.add_argument('--_split', dest='split', help=argparse.SUPPRESS)
    parser.add_argument('--_intake-python', dest='intake_python', default=str(PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'), help=argparse.SUPPRESS)
    parser.add_argument('--_deg', type=int, default=63, dest='deg', help=argparse.SUPPRESS)
    parser.add_argument('--_margin', type=float, default=1.25, dest='margin', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.stages = args.stages.split(',')
    if any(stage not in STAGES for stage in args.stages):
        parser.error('unknown stage')
    if not args.inputs.exists():
        parser.error('original --inputs path does not exist')
    from .models.info import source_kind
    kind = source_kind(args.model)
    if args._worker:
        if args.out is None:
            parser.error('internal worker requires output directory')
        if args._worker == 'prepare' and kind in ('hf', 'onnx') and Path(sys.executable).absolute() != Path(args.intake_python).absolute():
            tail = list(sys.argv[1:] if argv is None else argv)
            os.execv(args.intake_python, [args.intake_python, '-B', '-m', 'pt_agent.cli', *tail])
        if args._worker == 'prepare':
            return _prepare(args)
        if args._worker == 'split':
            return _prepare_split(args)
        return _main_stages(args)
    args.inputs = args.inputs.resolve()
    args.out = args.out or RESULTS_DIR/('u1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    status = _nix_worker(args, 'prepare')
    if status == 0:
        status = _nix_worker(args, 'main')
    if 'info' in args.stages and (args.out/'model_info.json').exists():
        print((args.out/'model_info.json').read_text(),flush=True)
    from .registry import append, git_rev, new_run_id
    os.environ.setdefault('PT_RESULTS', str(RESULTS_DIR))
    report_path = args.out/'REPORT.json'
    record = json.loads(report_path.read_text()) if report_path.exists() else dict(complete=False, stage='intake', exit=status)
    append(dict(run_id=new_run_id('U1', Path(args.model).name, 'intake'), round='U1', model=args.model,
                mode='intake', git_rev=git_rev(), metrics=record, artifacts_dir=str(args.out),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(report_path), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
