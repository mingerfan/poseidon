"""Finite U1 import acceptance using already approved local assets and data."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from ..paths import PLATFORM_DIR, RESULTS_DIR, REPO


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def prepare_inputs(args):
    import numpy as np
    from PIL import Image
    from ..data.cifar import read_rows
    # Real training records only. No report-test image is accessed here.
    pixels, labels = read_rows('train', range(223))
    for index, (pixel, label) in enumerate(zip(pixels, labels)):
        image = Image.fromarray(pixel.transpose(1, 2, 0))
        path = args.out/'inputs/images'/str(int(label))/f'{index:04d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    documents = [REPO/'DSL/DSL_agent/docs/pt_agent/specs/U1-model-description.md',
                 REPO/'DSL/DSL_agent/docs/pt_agent/STATUS.md']
    texts = [line for path in documents for line in path.read_text().splitlines()
             if line.strip() and len(line) <= 120]
    if len(texts) < 223:
        raise ValueError('Approved local Markdown documents provide fewer than 223 short nonempty original lines')
    (args.out/'inputs/prompts.txt').write_text('\n'.join(texts[:223])+'\n')
    # User-side ONNX samples: resize, grayscale for MNIST, divide bytes by 255;
    # the ONNX importer itself never normalizes or resizes an input.
    images = [Image.fromarray(pixel.transpose(1, 2, 0)) for pixel in pixels[:8]]
    mnist = np.stack([np.asarray(image.convert('L').resize((28, 28), Image.Resampling.BILINEAR),
                                 dtype=np.float32)[None]/np.float32(255) for image in images])
    squeeze = np.stack([np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR),
                                   dtype=np.float32).transpose(2, 0, 1)/np.float32(255) for image in images])
    np.savez(args.out/'inputs/mnist.npz', Input3=mnist)
    np.savez(args.out/'inputs/squeezenet.npz', data=squeeze)
    write(args.out/'prepare.json', dict(images=223, text_lines=223, image_source='CIFAR train[0:223]',
          onnx_reference_only_samples=8, report_images_read=0))


def hf_acceptance(args):
    import numpy as np
    import torch
    from ..data.samples import read
    from ..intake.hf_frontend import import_local, boundary_module
    from ..intake.graph_exec import forward
    from ..intake.coverage import report, check
    rows = []
    write(args.out/'HF.json', rows)
    for group in ('resnet18', 'bert_tiny', 'qwen_alternative'):
        if args.only and group not in args.only.split(','):
            continue
        paths = list((args.assets/group).glob('*/config.json'))
        if len(paths) != 1:
            raise ValueError(f'Expected one already approved local config for {group}')
        raw = read(args.inputs/'images' if group == 'resnet18' else args.inputs/'prompts.txt')
        imported = import_local(paths[0].parent, raw)
        info = imported.info
        info.save(args.out/group)
        selected = info.split['calibration'][:2]+info.split['heldout'][:2]
        arrays, _ = imported.samples.select(selected)
        wrapper = boundary_module(imported.original)
        max_diff = max_ratio = 0.0
        with torch.no_grad():
            for index in range(len(selected)):
                batch = {name: value[index:index+1] for name, value in arrays.items()}
                actual = forward(info.graph, batch)[info.graph.outputs[0]]
                expected = wrapper(*(torch.from_numpy(value) for value in batch.values())).cpu().numpy()
                diff = np.abs(actual-expected)
                max_diff = max(max_diff, float(diff.max()))
                max_ratio = max(max_ratio, float((diff/(1+np.abs(expected))).max()))
                if not np.all(diff <= 1e-6*(1+np.abs(expected))):
                    raise AssertionError(f'{group}: HF semantic mismatch, max_abs={max_diff}, normalized={max_ratio}')
        if group in ('resnet18', 'bert_tiny') and max_diff != 0:
            raise AssertionError(f'{group}: U1 §12 requires retained zero difference, observed {max_diff}')
        record = json.loads((args.out/group/'model_info.json').read_text())
        row = dict(model=group, samples=len(selected), task=info.task, inputs=info.inputs,
                   sequence=info.sequence, processor=record['input_handling'], coverage=report(info.graph),
                   graph_check=check(info.graph), max_abs_diff=max_diff, max_normalized_diff=max_ratio)
        rows.append(row); write(args.out/'HF.json', rows)
        if group == 'bert_tiny':
            torch.onnx.export(wrapper, tuple(torch.from_numpy(value[:1]) for value in arrays.values()),
                str(args.out/'bert_tiny.onnx'), opset_version=17, input_names=list(arrays),
                output_names=['logits'], do_constant_folding=True)
        print('HF '+group+' '+json.dumps(dict(max_abs_diff=max_diff, nodes=len(info.graph.nodes))), flush=True)
        del imported, wrapper, info, arrays
        import gc
        gc.collect()
    return rows


def onnx_acceptance(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from ..intake.onnx_frontend import import_graph
    from ..intake.graph_exec import forward
    from ..intake.coverage import report, check
    rows = []
    write(args.out/'ONNX.json', rows)
    for group, filename in (('mnist', 'mnist-12.onnx'), ('squeezenet', 'squeezenet1.1-7.onnx')):
        if args.only and group not in args.only.split(','):
            continue
        paths = list((args.assets/group).glob('*/'+filename))
        if len(paths) != 1:
            raise ValueError(f'Expected one already approved local ONNX asset for {group}')
        model = onnx.load(str(paths[0]))
        names = [value.name for value in model.graph.input if value.name not in {w.name for w in model.graph.initializer}]
        with np.load(args.inputs/(group+'.npz'), allow_pickle=False) as archive:
            data = archive[archive.files[0]].copy()
        graph = import_graph(paths[0], {names[0]: data[:1]})
        graph.save(args.out/group)
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(paths[0]), sess_options=options, providers=['CPUExecutionProvider'])
        max_diff = max_ratio = 0.0
        for i in range(len(data)):
            batch = {names[0]: data[i:i+1]}
            expected = session.run(None, batch)[0].astype(np.float64)
            actual = forward(graph, batch)[graph.outputs[0]]
            diff = np.abs(actual-expected)
            max_diff = max(max_diff, float(diff.max()))
            max_ratio = max(max_ratio, float((diff/(1+np.abs(expected))).max()))
            if not np.all(diff <= 1e-4*(1+np.abs(expected))):
                raise AssertionError(f'{group}: ONNX semantic mismatch, max_abs={max_diff}, normalized={max_ratio}')
        rows.append(dict(model=group, samples=len(data), max_abs_diff=max_diff,
                         max_normalized_diff=max_ratio, coverage=report(graph), graph_check=check(graph)))
        write(args.out/'ONNX.json', rows)
        print('ONNX '+group+' '+json.dumps(rows[-1]), flush=True)
    return rows


def sequence_acceptance(args):
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer
    from ..data.samples import read, choose_sequence, encode_texts
    from ..intake.hf_frontend import LOCAL_ONLY
    from ..models.info import ModelInfo
    raw = read(args.inputs/'prompts.txt')
    rows = []
    for name in ('bert_tiny', 'qwen_alternative'):
        info = ModelInfo.load(args.reference/name)
        directory = info.source['local_path']
        tokenizer = AutoTokenizer.from_pretrained(directory, **LOCAL_ONLY)
        config = AutoConfig.from_pretrained(directory, **LOCAL_ONLY)
        texts, _ = raw.select(info.split['calibration']+info.split['heldout'])
        lengths = [len(value) for value in tokenizer(texts, truncation=False, padding=False)['input_ids']]
        assert choose_sequence(lengths, config.max_position_embeddings) == info.sequence
        # L=8 is an isolated tokenizer-contract probe, not a new program or a
        # change to the model's inferred/calibrated length (96 or 80).
        cases = []
        for index in (int(np.argmin(lengths)), int(np.argmax(lengths))):
            text = texts[index]
            plain = tokenizer.encode(text, add_special_tokens=False)
            capacity = 8-tokenizer.num_special_tokens_to_add(pair=False)
            kept = plain[-capacity:] if info.task == 'causal_lm' else plain[:capacity]
            expected = tokenizer.build_inputs_with_special_tokens(kept)
            encoded = encode_texts(tokenizer, [text], {'buckets':[8]}, task=info.task)
            expected_mask = [1]*len(expected)+[0]*(8-len(expected))
            expected += [tokenizer.pad_token_id]*(8-len(expected))
            np.testing.assert_array_equal(encoded['input_ids'][0], expected)
            np.testing.assert_array_equal(encoded['attention_mask'][0], expected_mask)
            cases.append(dict(original_tokens=lengths[index], real_tokens=sum(expected_mask),
                              right_padding=8-sum(expected_mask), truncated=lengths[index]>8))
        rows.append(dict(model=name, length_policy_equal=True, sequence=info.sequence,
                         tokenizer=type(tokenizer).__name__, cases=cases, report_samples_evaluated=0))
    write(args.out/'SEQUENCE.json', rows)
    return rows


def limited(*, trusted_child=False):
    # Ordinary Python remains at 4 GiB. Only the existing trusted trace runner
    # may use its already approved 6 GiB limit in regression subprocesses.
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, (6 if trusted_child else 4)*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))


def inside(args):
    if args.part in ('prepare', 'hf', 'onnx', 'diagnose', 'cross', 'sequence', 'intake-tests') and Path(sys.executable).absolute() != Path(args.intake_python).absolute():
        os.execv(args.intake_python, [args.intake_python, '-B', '-m', 'pt_agent.experiments.u1_model_intake', *sys.argv[1:]])
    result = dict(complete=False, part=args.part)
    try:
        import torch
        torch.set_num_threads(1)
        if args.part == 'prepare':
            prepare_inputs(args)
        elif args.part == 'hf':
            result['models'] = hf_acceptance(args)
        elif args.part == 'onnx':
            result['models'] = onnx_acceptance(args)
        elif args.part == 'diagnose':
            from .u1_attention_diagnostic import run
            result['diagnostic'] = run(args)
        elif args.part == 'legacy':
            from .u1_legacy_equivalence import run
            result['models'] = run(args)
        elif args.part == 'cross':
            from .u1_cross_source import run
            result['models'] = run(args)
        elif args.part == 'generic':
            from .u1_generic_path import run
            result['models'] = run(args)
        elif args.part == 'sequence':
            result['models'] = sequence_acceptance(args)
        elif args.part in ('unit','intake-tests','regression'):
            import unittest
            if args.part == 'regression':
                packaging = {'test_p2_export.py', 'test_toolchain_decoupled.py'}
                tensor = {'test_r8f_tensor_validate.py'}
                files = sorted(Path('pt_agent/tests').glob('test_*.py'))
                selected = [path for path in files if not path.name.startswith('test_u1_') and
                            (path.name in packaging if args.only == 'packaging' else
                             path.name in tensor if args.only == 'tensor' else
                             path.name not in packaging | tensor)]
                suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(
                    'pt_agent.tests.'+path.stem) for path in selected)
                expected = {'main':339, 'packaging':17, 'tensor':17}[args.only]
                if suite.countTestCases() != expected:
                    raise AssertionError(f'Regression inventory differs: {suite.countTestCases()} vs {expected}')
                result['expected'] = expected
            else:
                suite = unittest.defaultTestLoader.discover('pt_agent/tests',pattern=(
                    'test_u1_onnx.py' if args.part == 'intake-tests' else 'test_u1_*.py'))
            outcome = unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
            result.update(tests=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),skipped=len(outcome.skipped),
                          forbidden_imports=[name for name in ('transformers','onnx') if args.part != 'intake-tests' and name in sys.modules])
            if not outcome.wasSuccessful() or result['forbidden_imports']:
                raise AssertionError('U1 unit tests or main-environment import isolation failed')
        result['complete'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        import traceback
        traceback.print_exc()
    write(args.out/'REPORT.json', result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('prepare', 'hf', 'onnx', 'diagnose', 'legacy', 'cross', 'generic', 'sequence', 'unit', 'intake-tests', 'regression'), required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--only')
    parser.add_argument('--models-dir', type=Path, default=RESULTS_DIR/'models-v2')
    parser.add_argument('--assets', type=Path, default=PLATFORM_DIR/'cache/pt-agent-models')
    parser.add_argument('--intake-python', default=str(PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'))
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if args.inside:
        return inside(args)
    args.out = args.out or RESULTS_DIR/('u1-'+args.part+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '3600', '--',
               '-m', 'pt_agent.experiments.u1_model_intake', '--inside', '--part', args.part,
               '--out', str(args.out), '--assets', str(args.assets), '--intake-python', args.intake_python]
    command.extend(('--models-dir',str(args.models_dir)))
    if args.inputs is not None:
        command.extend(('--inputs', str(args.inputs)))
    if args.reference is not None:
        command.extend(('--reference', str(args.reference)))
    if args.only:
        command.extend(('--only', args.only))
    write(args.out/'argv.json', command)
    started = time.monotonic()
    with (args.out/'run.log').open('w') as log:
        from functools import partial
        status = subprocess.run(command, cwd=Path(__file__).resolve().parents[2], stdout=log,
                                stderr=subprocess.STDOUT, preexec_fn=partial(limited, trusted_child=args.part == 'regression'), timeout=3720).returncode
    path = args.out/'REPORT.json'
    record = json.loads(path.read_text()) if path.exists() else dict(complete=False)
    record.update(exit=status, seconds=time.monotonic()-started, llm_calls=0, new_downloads=0)
    write(path, record)
    from ..registry import append, new_run_id, git_rev
    os.environ.setdefault('PT_RESULTS', str(RESULTS_DIR))
    append(dict(run_id=new_run_id('U1', args.part, 'acceptance'), round='U1', model=args.part,
                mode='acceptance', git_rev=git_rev(), metrics=record, artifacts_dir=str(args.out),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(path), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
