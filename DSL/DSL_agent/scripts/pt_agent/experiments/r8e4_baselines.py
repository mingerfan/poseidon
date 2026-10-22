"""Re-evaluate the two retained ResNet20 programs using the held-out split."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ..run import _write


SOURCES = {
    'R6c': 'run-resnet20-rule-20261020T165817/program.ptdsl',
    'R7c-B': 'run-resnet20-agent-20261021T033036/program.ptdsl',
}


def _inside(args):
    from ..analyzer import calibrate
    from ..data.cifar import calibration_set
    from ..dsl.parser import parse
    from ..models.zoo import get
    from ..planners.agent_tools import build_context, validate
    from ..run import _inside as run_inside

    model = get('resnet20', args.models_dir)
    graph = model.import_graph()
    images, _ = calibration_set(normalization=model.normalization)
    calibration = calibrate(graph, {graph.inputs[0]: model.adapt(images)})
    context = build_context(graph, calibration, normalization=model.normalization)
    text = args.source.read_text()
    (args.out/'program.ptdsl').write_text(text)
    report = dict(round='R8e4', model='resnet20', source_path=str(args.source), complete=False,
                  val_split=context.val_split, test_split='test[0:9000]',
                  heldout_validation=validate(parse(text), context))
    _write(args.out/'REPORT.json', report)
    code = run_inside(SimpleNamespace(out=args.out, model='resnet20', models_dir=args.models_dir,
                                     planner='rule', deg=None, margin=None, refresh='compiler',
                                     stages=['test'], final=True))
    report = json.loads((args.out/'REPORT.json').read_text())
    report['round'] = 'R8e4'
    _write(args.out/'REPORT.json', report)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--models-dir', type=Path)
    args = parser.parse_args()
    if args.inside:
        if any(value is None for value in (args.out, args.source, args.models_dir)):
            parser.error('--inside requires explicit out, source and models-dir')
        return _inside(args)
    from ..registry import append, git_rev, new_run_id

    root = Path(os.environ['PT_RESULTS']).resolve()
    out = (args.out or root/('r8e4-baselines-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    out.mkdir(exist_ok=False)
    models = (args.models_dir or root/'models-v2').resolve()
    report = dict(round='R8e4', complete=False, stage='baselines', val_split='test[9000:10000]',
                  test_split='test[0:9000]', api_calls=0, compiles=0, models=[], git_rev=git_rev())
    _write(out/'REPORT.json', report)
    for name, source in SOURCES.items():
        case = out/name
        case.mkdir()
        command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '21600', '--',
                   '-m', 'pt_agent.experiments.r8e4_baselines', '--inside', '--out', str(case),
                   '--source', str(root/source), '--models-dir', str(models)]
        with (case/'stdout.txt').open('w') as stdout, (case/'stderr.txt').open('w') as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr,
                                       cwd=Path(__file__).resolve().parents[2], timeout=21720)
        current = json.loads((case/'REPORT.json').read_text()) if (case/'REPORT.json').exists() else {}
        row = dict(name=name, report_path=str(case/'REPORT.json'), exit_code=completed.returncode,
                   complete=bool(current.get('complete') and completed.returncode == 0),
                   heldout_validation=current.get('heldout_validation'), top1_test=current.get('top1_test'),
                   torch_top1_report=current.get('torch_top1_report'), n_diverged=current.get('n_diverged'))
        report['models'].append(row)
        _write(out/'REPORT.json', report)
    report.update(complete=all(row['complete'] for row in report['models']), stage='complete',
                  run_id=new_run_id('R8e4', 'resnet20', 'baselines'))
    _write(out/'REPORT.json', report)
    append(dict(run_id=report['run_id'], round='R8e4', model='resnet20', mode='baselines',
                git_rev=report['git_rev'], metrics=report, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(out/'REPORT.json'), flush=True)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
