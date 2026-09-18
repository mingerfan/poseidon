"""Actual saved native-call artifacts versus independent PyTorch references.

Trusted manual fixtures only. Reuses stock SEAL_HEVM and tc128 parameters;
no Agent execution, API, new backend, bootstrap or package installation.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from probe_native_calls import CASES, SOURCE
from python_compiler_smoke import BUILD, logged
from seal_cpu_golden import PROFILE, KEY_BUILD, execute_artifact, compare, dump
from seal_artifact_gate import require
from native_execution_slots import native_slot


def reference(case):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    require(torch.__version__ == '2.0.1+cpu' and np.__version__ == '1.25.2', 'Pinned packages required')
    x = np.array([[0., 0., 0., 0.], [-.8, .25, .6, -1.],
                  *np.random.default_rng(4201).uniform(-1., 1., (1, 4)), [-1., 1., -1., 1.]], dtype=np.float64)
    y = np.array([[0., 0., 0., 0.], [.2, -.75, .1, .5],
                  *np.random.default_rng(4202).uniform(-1., 1., (1, 4)), [1., -1., 1., -1.]], dtype=np.float64)
    t, u = torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
    if case == 'pair': outputs = [t+.25, t*.5]
    elif case == 'array': outputs = [t+.25, t, t+.75, t+.5]
    elif case == 'identity': outputs = [t]
    elif case == 'nested': outputs = [t*.25+.25]
    elif case == 'repeated': outputs = [2*t+.5]
    elif case == 'two_inputs': outputs = [u-t]
    elif case in ('zero_input', 'plain_return', 'public_argument'): outputs = [t*.5]
    elif case in ('empty_helper', 'zero_dim'): outputs = [t+.25]
    else:
        require(case in ('scalar', 'forward'), 'Unknown oracle')
        outputs = [torch.square(t)+.25]
    return np.stack([x, y], axis=1) if case == 'two_inputs' else x, torch.stack(outputs, dim=1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('probe', type=Path)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--keys', type=Path)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Wrong workspace')
    require(args.probe.resolve().parent == WORK/'results', 'Requires native probe results')
    if not args.inside:
        require(not args.worker and not args.keys, 'Worker requires pinned environment')
        cmd = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'), str(Path(__file__).resolve()), str(args.probe.resolve()), '--inside'])
        return enter_nix(cmd, seconds=1500)
    require(Path(sys.prefix) == VENV and os.environ.get('IN_NIX_SHELL'), 'Pinned environment required')
    if args.worker:
        directory = args.worker.resolve()
        require(directory.is_relative_to(WORK/'results') and args.keys and
                args.keys.resolve().is_relative_to(WORK/'results'), 'Unexpected worker paths')
        binding = json.loads((directory/'binding.json').read_text())
        return execute_artifact(directory, args.keys.resolve(), binding['selectors'],
                                expected_inputs=binding['input_count'])
    require(not args.keys, 'Keys are owned by the experiment')
    import numpy as np
    probe = args.probe.resolve()
    saved = json.loads((probe/'report.json').read_text())
    require([c['case'] for c in saved['cases']] == list(CASES), 'Incomplete compile cohort')
    require(saved['frontend_source_sha256'] == digest(SOURCE) and
            saved['frontend_library_sha256'] == digest(BUILD/'lib/libHecateFrontend.so') and
            saved['profile_sha256'] == digest(PROFILE), 'Producer changed since compile')
    for c in saved['cases']:
        for style in ('direct', 'call'):
            require(c[style].get('trace_exit') == c[style].get('compile_exit') == 0,
                    'Every artifact must pass real trace/compile before execution')
    require(shutil.disk_usage(WORK).free > 3*1024**3, 'Insufficient free disk')
    os.umask(0o077)
    out = Path(tempfile.mkdtemp(prefix='seal-cpu-golden-native-calls-', dir=WORK/'results'))
    print('Native calls CPU evidence:', out, flush=True)
    report = dict(schema=1, status='running', agent_calls=0, backend='upstream_SEAL_HEVM_CPU',
        poseidon_gpu_validated=False, bootstrap_executed=False, cases=[],
        probe=str(probe), probe_sha256=digest(probe/'report.json'),
        source_sha256=digest(SOURCE), runner_sha256=digest(Path(__file__)),
        runtime_sha256=digest(BUILD/'lib/libSEAL_HEVM.so'), profile_sha256=digest(PROFILE),
        key_generator_sha256=digest(KEY_BUILD/'seal_golden_keys'),
        atol=1e-5, rtol=1e-4)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    keys = out/'private-keys'
    dump(out/'report.json', report)
    try:
        with native_slot(WORK/'cache/agent-native-slots'):
            keys.mkdir(mode=0o700)
            require(logged([str(KEY_BUILD/'seal_golden_keys'), str(keys)], out/'parameters.json',
                           seconds=120, env=env) == 0, 'Existing tc128 key helper failed')
            report['parameters'] = json.loads((out/'parameters.json').read_text())
            for c in saved['cases']:
                case = c['case']; inputs, expected = reference(case)
                groups = ([[i, slot] for i in range(expected.shape[1])] for slot in range(4)) if expected.shape[1] > 1 else (
                    [[0, slot] for slot in range(4)],)
                groups = list(groups)
                for style in ('direct', 'call'):
                    source = probe/(case+'-'+style)
                    for index, selectors in enumerate(groups):
                        folder = out/(case+'-'+style+'-'+str(index)); folder.mkdir()
                        for name, field in [('lowered._hecate_golden.hevm', 'hevm_sha256'), ('_hecate_golden.cst', 'cst_sha256')]:
                            require(digest(source/name) == c[style][field], 'Original artifact changed')
                            shutil.copyfile(source/name, folder/name)
                        target = np.stack([expected[:, i, j] for i, j in selectors], axis=1)
                        np.savez(folder/'arrays.npz', inputs=inputs, reference=target)
                        binding = dict(selectors=selectors, input_count=2 if case == 'two_inputs' else 1,
                                       input_names=['zero', 'signed', 'seed4201_4202', 'boundary'])
                        dump(folder/'binding.json', binding)
                        frozen = {p.name:digest(p) for p in folder.iterdir() if p.is_file()}
                        cmd = [str(VENV/'bin/python'), str(Path(__file__).resolve()), str(probe), '--inside',
                               '--worker', str(folder), '--keys', str(keys)]
                        code = logged(cmd, folder/'execution.log', seconds=150, env=env)
                        row = dict(case=case, style=style, group=index, exit_code=code,
                                   binding=binding, frozen_hashes=frozen, status='runtime_failed')
                        report['cases'].append(row)
                        require(all(digest(folder/n) == h for n,h in frozen.items()), 'Frozen input/artifact changed')
                        if code == 0:
                            comparison = compare(np.load(folder/'decrypted.npy', allow_pickle=False), target, 1e-5, 1e-4)
                            row.update(comparison=comparison, execution=json.loads((folder/'execution.json').read_text()),
                                       status='passed' if comparison['passed'] else 'numerical_failed')
                        print(folder.name, row['status'], flush=True)
                        dump(out/'report.json', report)
        report['status'] = 'passed' if all(c['status'] == 'passed' for c in report['cases']) else 'failed'
        numeric = [r['comparison'] for r in report['cases'] if 'comparison' in r]
        report['compared_values'] = sum(r['compared_values'] for r in numeric)
        report['max_absolute_error'] = max(r['max_absolute_error'] for r in numeric) if numeric else None
        report['weighted_mae'] = (sum(r['mae']*r['compared_values'] for r in numeric)/report['compared_values']) if numeric else None
    except Exception as error:
        report.update(status='failed', error=str(error))
        print('Stopped:', error, flush=True)
    finally:
        dump(out/'report.json', report)
        if keys.exists():
            from result_retention import cleanup_run
            try:
                clean = cleanup_run(out, WORK/'results')
                print('Cleaned regenerated key bytes:', clean['bytes'], flush=True)
            except (OSError, ValueError):
                print('Key cleanup deferred; preserve diagnostic evidence', flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
