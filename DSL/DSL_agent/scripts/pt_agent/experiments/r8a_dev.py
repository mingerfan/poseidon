"""R8a development-model training, intake and rule-planner evaluation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..models.train_small import RECIPE


def _write(path, record):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _normalization(out):
    import numpy as np
    import torch
    from ..data.cifar import validation_set
    from ..models.zoo import CIFAR_NORMALIZATION, IMAGENET_NORMALIZATION, get

    torch.set_num_threads(2)
    model = get('resnet20_silu', out/'models').module
    result = dict(complete=False, split='train', validation_images=2000, candidates=[], selected=None)
    for name, normalization in [('imagenet', IMAGENET_NORMALIZATION), ('cifar', CIFAR_NORMALIZATION)]:
        images, labels = validation_set(2000, normalization=normalization)
        correct = 0
        started = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(images), 128):
                predictions = model(torch.from_numpy(images[start:start+128]).float()).argmax(1).numpy()
                correct += int(np.sum(predictions == labels[start:start+len(predictions)]))
        row = dict(name=name, normalization=normalization, correct=correct,
                   top1=100.0*correct/len(images), seconds=time.monotonic()-started)
        result['candidates'].append(row)
        _write(out/'normalization.json', result)
        print('NORMALIZATION '+json.dumps(row), flush=True)
    result['selected'] = next((row['name'] for row in result['candidates'] if row['top1'] >= 95.0), None)
    result['complete'] = result['selected'] is not None
    _write(out/'normalization.json', result)
    return 0 if result['complete'] else 1


def _evaluate(name, out, models_dir):
    import copy
    import numpy as np
    import torch
    from ..analyzer import reference_forward
    from ..data.cifar import calibration_set, test_set
    from ..intake.coverage import report as coverage_report
    from ..models.zoo import get

    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)
    path = out/'REFERENCE.json'
    result = dict(model=name, complete=False, stage='intake', frontend='aten', nodes=None, coverage=None,
                  torch_top1=None, ref_diff=None, test_images_done=0, test_correct=0)
    _write(path, result)
    try:
        zm = get(name, models_dir)
        graph = zm.import_graph()
        graph.save(out/'graph')
        result.update(nodes=len(graph.nodes), coverage=coverage_report(graph), normalization=zm.normalization,
                      stage='reference')
        images, _ = calibration_set(normalization=zm.normalization)
        images = zm.adapt(images[:8])
        model64 = copy.deepcopy(zm.module).double().eval()
        with torch.no_grad():
            expected = np.concatenate([model64(torch.from_numpy(images[i:i+1])).numpy()
                                       for i in range(len(images))])
        actual = reference_forward(graph, {graph.inputs[0]: images})[graph.outputs[0]]
        result['ref_diff'] = float(np.max(np.abs(actual-expected)))
        _write(path, result)
        if result['ref_diff'] > 1e-9:
            raise ValueError('reference logits differ from torch float64 by more than 1e-9')
        del model64, images, expected, actual
        result['stage'] = 'torch_accuracy'
        images, labels = test_set(normalization=zm.normalization)
        images = zm.adapt(images)
        with torch.no_grad():
            for start in range(0, len(images), 128):
                logits = zm.module(torch.from_numpy(images[start:start+128]).float()).numpy()
                result['test_correct'] += int(np.sum(logits.argmax(1) == labels[start:start+len(logits)]))
                result['test_images_done'] = start+len(logits)
                _write(path, result)
        result.update(complete=True, stage='complete', torch_top1=100.0*result['test_correct']/len(images))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    _write(path, result)
    print('REFERENCE '+json.dumps(result), flush=True)
    return 0 if result['complete'] else 1


def _outer(args):
    root = Path(os.environ['PT_RESULTS']).resolve()
    out = (args.out or root/('r8a-dev-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out/'REPORT.json'
    if path.exists():
        raise FileExistsError('Existing experiment reports must be preserved')
    models_dir = (args.models_dir or root/'models-v2').resolve()
    names = ('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp')
    result = dict(round='R8a', complete=False, stage='normalization', frontend='aten',
                  train_recipe=RECIPE, models_dir=str(models_dir),
                  normalization=None, training={}, commands=[], models=[dict(model=name, nodes=None,
                  coverage=None, torch_top1=None, ref_diff=None, source_lines=None, agree=None,
                  counts=None, max_depth=None, boots=None, compile_s=None, peak_rss_mib=None,
                  complete=False) for name in names])
    _write(path, result)

    def call(label, command, seconds):
        argv = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds), '--', *command]
        started = time.monotonic()
        with (out/(label+'-stdout.txt')).open('w') as stdout, (out/(label+'-stderr.txt')).open('w') as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=seconds+120)
        result['commands'].append(dict(label=label, command=argv, returncode=completed.returncode,
                                       seconds=time.monotonic()-started))
        _write(path, result)
        if completed.returncode:
            raise RuntimeError(label+' failed; retained stdout/stderr and phase report')

    started = time.monotonic()
    try:
        # A normalization-only phase may already have run for this same output
        # directory before freezing SILU_NORMALIZATION and running offline tests.
        if not (out/'normalization.json').exists():
            call('normalization', ['-m', 'pt_agent.experiments.r8a_dev', '--inside', '--phase',
                                  'normalization', '--out', str(out)], 600)
        selected = json.loads((out/'normalization.json').read_text())
        result['normalization'] = selected
        if not selected['complete'] or selected['selected'] is None:
            raise RuntimeError('Neither SiLU normalization reached 95% on training validation')
        if selected['split'] != 'train' or selected['validation_images'] != 2000:
            raise ValueError('Normalization selection must use the prescribed training validation set')
        for name in ('cnn_plain', 'mlp'):
            result['stage'] = 'train_'+name
            _write(path, result)
            call('train-'+name, ['-m', 'pt_agent.models.train_small', '--out-dir', str(models_dir),
                                '--model', name], 4000)
            result['training'][name] = json.loads((models_dir/(name+'-train.json')).read_text())
            _write(path, result)
        for row in result['models']:
            name = row['model']
            case = out/name
            case.mkdir()
            result['stage'] = 'reference_'+name
            _write(path, result)
            call('reference-'+name, ['-m', 'pt_agent.experiments.r8a_dev', '--inside', '--phase',
                                    'reference', '--model', name, '--models-dir', str(models_dir),
                                    '--out', str(case)], 600)
            reference = json.loads((case/'REFERENCE.json').read_text())
            row.update({key: reference[key] for key in ('nodes', 'coverage', 'torch_top1', 'ref_diff')})
            row['reference_path'] = str(case/'REFERENCE.json')
            rule_out = case/'rule'
            rule_out.mkdir()
            row['rule_report_path'] = str(rule_out/'REPORT.json')
            result['stage'] = 'rule_'+name
            _write(path, result)
            try:
                call('rule-'+name, ['-m', 'pt_agent.run', '--inside', '--out', str(rule_out),
                                   '--model', name, '--models-dir', str(models_dir), '--planner', 'rule',
                                   '--deg', '127', '--margin', '1.5',
                                   '--stages', 'plan,check,mock,count,emit,compile'], 10800)
            finally:
                if (rule_out/'REPORT.json').exists():
                    rule = json.loads((rule_out/'REPORT.json').read_text())
                    row.update({key: rule.get(key) for key in ('source_lines', 'agree', 'counts', 'max_depth')})
                    row.update(boots=(rule.get('plan') or {}).get('boots'),
                               compile_s=(rule.get('compile') or {}).get('seconds'),
                               peak_rss_mib=(rule.get('compile') or {}).get('peak_rss_mib'),
                               complete=bool(rule['complete'] and (rule.get('compile') or {}).get('exit') == 0
                                             and rule.get('plan')), error=rule.get('error'))
                    _write(path, result)
            if not row['complete']:
                raise RuntimeError(name+' did not complete real compilation')
            print('MODEL '+json.dumps(row), flush=True)
        result.update(complete=True, stage='complete')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    if (out/'normalization.json').exists():
        result['normalization'] = json.loads((out/'normalization.json').read_text())
    result['seconds'] = time.monotonic()-started
    _write(path, result)
    from ..registry import append, git_rev, new_run_id

    append(dict(run_id=new_run_id('R8a', 'dev-models', 'rule-baselines'), round='R8a', model='dev-models',
                mode='rule-baselines', git_rev=git_rev(), metrics=result, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(path), flush=True)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--models-dir', type=Path)
    parser.add_argument('--model', choices=('resnet20', 'resnet20_silu', 'cnn_plain', 'mlp'))
    parser.add_argument('--phase', choices=('normalization', 'reference'))
    args = parser.parse_args()
    if args.inside:
        if args.out is None or args.phase is None:
            parser.error('--inside requires --out and --phase')
        if args.phase == 'normalization':
            return _normalization(args.out)
        if args.model is None or args.models_dir is None:
            parser.error('reference requires --model and --models-dir')
        return _evaluate(args.model, args.out, args.models_dir)
    return _outer(args)


if __name__ == '__main__':
    sys.exit(main())
