"""Fixed 8-graph, 3-key paired waterline diagnostic; never calls an Agent API.

Recompile frozen Earth IR through the existing Dacapo/SEAL HEVM path. Retain
every failure; do not adapt the cohort, tolerance, inputs or default profile.
Old payloads provide I/O only, not a claim of generation at a new waterline.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from python_compiler_smoke import BUILD
from seal_cpu_golden import KEY_BUILD, PROFILE, WATERLINE, compare, dump
from seal_artifact_gate import inspect_artifacts, require
from native_execution_slots import native_slot
import candidate_sandbox as sandbox

SCRIPT = Path(__file__).resolve()
RESULTS = WORK/'results'
SCALES = (40, 45)
TRIALS = 3
ROTATIONS = (-3, -2, -1, 1, 2, 3)
# Order and all source-report hashes are fixed before the first execution.
SOURCES = (
    ('square', 'candidate-replay-b1ht8frw', '62c57d2b0ffec563276395eb4288dca65df2d008d440e48f2782ecc96ece205b', 0),
    ('linear', 'agent-deepseek-ku5nb5sf', '2f1a4ac231a3c6e4ac7ec4f4e46c547f951875106e960f913564092c6a52fc13', 0),
    ('mlp2', 'agent-deepseek-brpfj767', '18b0f171d47be1b63ff77237edeebc05f8fa6249eca1076c16e710ef94d473be', 0),
    ('mlp3', 'agent-deepseek-7463lvpk', '0ecc46b73c41394bea404a9f2db503f93b1cba66c8e7a93c3e2758329cc90e70', 2),
    ('fanout', 'agent-deepseek-8me0dlgo', '564eb2e1e148a2f3cd2e3e1485807cc9aaa02d96af1552b9c9332d1acd87dccc', 0),
    ('residual', 'agent-deepseek-lpt6e9gk', '568850c774468ea821cb9449ca69d9c7af5fc82c4628cfcda574c2b16e4f97e4', 1),
    ('wide_mlp8', 'candidate-replay-ixz1my3w', 'c2d3c6ae31beab8e856cab12c3ec6430f9df30da6575a147ab5a8d515393225e', 0),
    ('sum4', 'candidate-replay-t9aglvbt', '4ea525b5baa833cf40df36050a00b01f683a86c5a6e307fc28d3c5e05f1f00e0', 0),
)


def security_parameters(parameters):
    # The experiment uses a common superset of Galois keys, not a different Q/P.
    return {k: v for k, v in parameters.items() if k != 'rotation_steps'}


def checked_source(spec):
    name, folder, report_hash, index = spec
    run = RESULTS/folder
    require(digest(run/'report.json') == report_hash, 'Frozen source report changed')
    report = json.loads((run/'report.json').read_text())
    require(report['backend'] == 'upstream_SEAL_HEVM_CPU' and report['waterline'] == 40,
            'Source backend/waterline changed')
    require(report['tolerance'] == dict(atol=1e-5, rtol=1e-4), 'Source tolerance changed')
    attempt = next(a for a in report['attempts'] if a['index'] == index)
    require(bool(attempt['comparison']['passed']) == (name != 'sum4'),
            'Original pass/failure must remain recorded')
    output = run/f'attempt-{index:02d}'/'output'
    frozen = {str(run/'report.json'): report_hash}
    frozen.update({str(run/n): h for n, h in report['frozen_hashes'].items()})
    frozen.update({str(output/n): h for n, h in attempt['artifact_hashes'].items()})
    for path, value in frozen.items():
        require(digest(Path(path)) == value, 'Archived source/input/artifact mutated')
    payload = run/f'attempt-{index:02d}'/'trace-payload.json'
    require(str(payload) in frozen, 'Missing frozen I/O payload')
    return report, output, payload, frozen


def reference_for(run):
    """Check original fixed weights and independently evaluate row-dot products."""
    import numpy as np
    import torch
    from model_graph import evaluate_reference
    from model_catalog import build_model
    from catalog_graph_migration import expand_legacy
    descriptor = json.loads((run/'model.json').read_text())
    model, shape = build_model(descriptor)
    with np.load(run/'weights.npz', allow_pickle=False) as weights:
        state = model.state_dict()
        require(set(weights.files) == set(state), 'Model state keys differ from frozen weights')
        for name, value in state.items():
            np.testing.assert_array_equal(value.detach().cpu().numpy(), weights[name])
    graph = expand_legacy(descriptor) if descriptor['schema'] == 1 else descriptor
    with np.load(run/'arrays.npz', allow_pickle=False) as data:
        inputs, saved = data['inputs'].copy(), data['reference'].copy()
        logical = data['logical_inputs'].copy()
    require(inputs.shape == (4, 4) and logical.shape == (4, *shape), 'Unexpected frozen input ABI')
    np.testing.assert_array_equal(inputs, logical.reshape(4, 4))
    independent = np.asarray([evaluate_reference(graph, x.tolist()) for x in logical], dtype=np.float64)
    with torch.no_grad():
        torch_reference = np.asarray([model(torch.tensor(x, dtype=torch.float64)).numpy().reshape(-1)
                                      for x in logical])
    # Tight float64 consistency check, independent of CKKS acceptance tolerance.
    np.testing.assert_allclose(saved, independent, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(torch_reference, independent, atol=1e-14, rtol=1e-14)
    return inputs, saved, graph


def native(*args, **kwargs):
    with native_slot(WORK/'cache/agent-native-slots', timeout=120):
        return sandbox.run(*args, **kwargs)


def summarize(report):
    summaries = []
    for scale in SCALES:
        items = [v for t in report['trials'] for v in t['executions'] if v['waterline'] == scale]
        comparisons = [v['comparison'] for v in items if 'comparison' in v]
        count = sum(c['compared_values'] for c in comparisons)
        summaries.append(dict(waterline=scale, planned_program_executions=len(SOURCES)*TRIALS,
            executed=len(comparisons), passed=sum(c['passed'] for c in comparisons),
            execution_failures=sum(v['status'] == 'execution_failed' for v in items),
            compile_failures=sum(c['status'] == 'compile_failed' for c in report['compiled'] if c['waterline'] == scale),
            input_executions=4*len(comparisons), compared_values=count,
            max_absolute_error=max((c['max_absolute_error'] for c in comparisons), default=None),
            mae=sum(c['mae']*c['compared_values'] for c in comparisons)/count if count else None))
    return summaries


def inside(root):
    import numpy as np
    require(os.environ.get('IN_NIX_SHELL') and Path(sys.prefix) == VENV, 'Requires pinned isolation')
    require(WATERLINE == 40, 'Default profile must not change in this experiment')
    report = dict(status='running', purpose='paired_precision_across_graphs',
        waterlines=list(SCALES), planned_key_sets=TRIALS, source_specs=SOURCES,
        common_rotation_steps=list(ROTATIONS), source_request_reused_for_io_only=True,
        threshold=dict(atol=1e-5, rtol=1e-4), production_waterline=40,
        production_profile_changed=False, threshold_changed=False, agent_calls=0,
        llm_generation_validated=False, poseidon_gpu_validated=False, all_semantics_proven=False,
        source_hashes={str(PROFILE): digest(PROFILE)}, sources=[], compiled=[], trials=[])
    paths = [SCRIPT, Path(sandbox.__file__), *[SCRIPT.with_name(n+'.py') for n in (
        'seal_cpu_golden', 'seal_artifact_gate', 'candidate_worker', 'model_catalog', 'model_graph',
        'catalog_graph_migration', 'cipher_abi', 'candidate_contract', 'result_retention')]]
    (root/'producer-snapshots').mkdir()
    for path in paths:
        target = root/'producer-snapshots'/path.name
        shutil.copyfile(path, target)
        report.setdefault('producer_snapshots', {})[str(target.relative_to(root))] = digest(target)
    report['runtime_hashes'] = {str(p): digest(p) for p in (BUILD/'bin/hecate-opt',
        BUILD/'lib/libSEAL_HEVM.so', KEY_BUILD/'seal_golden_keys', KEY_BUILD/'libseal_golden_metadata.so')}
    source_data = {}
    expected_params = None
    for spec in SOURCES:
        name, folder, _, index = spec
        original, old_output, old_payload, frozen = checked_source(spec)
        report['source_hashes'].update(frozen)
        core = security_parameters(original['parameters'])
        if expected_params is None:
            expected_params = core
        require(core == expected_params, 'Different source security profiles')
        inputs, reference, graph = reference_for(RESULTS/folder)
        case = root/name; case.mkdir()
        shutil.copyfile(RESULTS/folder/'arrays.npz', case/'reference-arrays.npz')
        shutil.copyfile(old_payload, case/'payload.json')
        dump(case/'independent-model.json', graph)
        source_data[name] = dict(payload=case/'payload.json', reference=reference)
        report['sources'].append(dict(name=name, run=str(RESULTS/folder), attempt=index,
            originally_passed=original['attempts'][index]['comparison']['passed'],
            source_rotation_steps=original['parameters']['rotation_steps'],
            hashes={str(p.relative_to(root)): digest(p) for p in case.iterdir()}))
        for scale in SCALES:
            out=case/f'compiled-{scale}'; out.mkdir()
            shutil.copyfile(old_output/'candidate_trace.mlir', out/'candidate_trace.mlir')
            for file in old_output.glob('*.cst'):
                require(str(file) in frozen, 'Unfrozen source constant')
                shutil.copyfile(file, out/file.name)
            command=['/hecate-opt', '/out/candidate_trace.mlir', '--eva', '--ckks-config=/profile.json',
                     '--waterline='+str(scale), '--enable-debug-printer', '--mlir-disable-threading',
                     '--verify-each', '-o', '/out/lowered.mlir']
            item=dict(name=name, waterline=scale, directory=str(out), command=command, status='compiling')
            report['compiled'].append(item)
            try:
                require(native(case/'payload.json', out, command, case/f'compile-{scale}.log', seconds=90) == 0,
                        'Compiler process failed')
                gate=inspect_artifacts((out/'lowered._hecate_golden.hevm').read_bytes(),
                                      (out/'_hecate_golden.cst').read_bytes(), rotation_steps=ROTATIONS)
                require(gate['arg_scale'] == [scale] and gate['arg_level'] == [13] and
                        gate['initial_level'] == 13, 'Unexpected input precision/security contract')
                # Result scales/levels and instruction schedules can depend on multiplication.
                # Record their verified values rather than demanding the sum-only pattern.
                np.savez(out/'arrays.npz', inputs=inputs)
                item.update(status='compiled', gate=gate,
                            hashes={p.name: digest(p) for p in out.iterdir() if p.is_file()})
            except Exception as error:
                item.update(status='compile_failed', diagnostic=str(error))
            dump(root/'report.json', report)
    report['security_parameters'] = expected_params
    for trial in range(TRIALS):
        run=Path(tempfile.mkdtemp(prefix='seal-cpu-golden-graph-precision-', dir=RESULTS))
        keys=run/'private-keys'; keys.mkdir(mode=0o700)
        row=dict(index=trial, run=str(run), status='running', executions=[])
        report['trials'].append(row)
        saved=dict(status='running', purpose=report['purpose'], executions=row['executions'], agent_calls=0)
        try:
            with native_slot(WORK/'cache/agent-native-slots', timeout=120):
                with (run/'parameters.json').open('x') as stream, (run/'keygen.log').open('x') as log:
                    code=subprocess.run([str(KEY_BUILD/'seal_golden_keys'), str(keys), *map(str, ROTATIONS)],
                                        stdout=stream, stderr=log, timeout=90).returncode
            require(code == 0, 'Key generation failed')
            params=json.loads((run/'parameters.json').read_text())
            require(security_parameters(params) == expected_params and params['rotation_steps'] == list(ROTATIONS),
                    'Security parameters/common key policy changed')
            saved['parameters']=params
            key_hashes={p.name: digest(p) for p in keys.iterdir()}
            saved['same_key_set_hashes']=key_hashes
            for template in report['compiled']:
                if template['status'] != 'compiled':
                    continue
                name, scale=template['name'], template['waterline']
                out=run/f'{name}-{scale}'; out.mkdir()
                item=dict(name=name, waterline=scale, directory=str(out), status='executing')
                row['executions'].append(item)
                for fname, value in template['hashes'].items():
                    path=Path(template['directory'])/fname
                    require(digest(path) == value, 'Template changed before runtime')
                    shutil.copyfile(path, out/fname)
                require(all(digest(keys/n) == h for n,h in key_hashes.items()), 'Key set changed')
                try:
                    require(native(source_data[name]['payload'], out,
                        [str(VENV/'bin/python'), '/app/candidate_worker.py', 'execute'],
                        run/f'execute-{name}-{scale}.log', seconds=150, keys=keys) == 0, 'SEAL HEVM execution failed')
                    execution=json.loads((out/'execution.json').read_text())
                    require(execution['encrypted_execution'] and not execution['bootstrap_executed'] and
                            execution['input_batches'] == 4, 'Missing real four-group execution')
                    actual=np.load(out/'decrypted.npy', allow_pickle=False)
                    comparison=compare(actual, source_data[name]['reference'], 1e-5, 1e-4)
                    item.update(status='passed' if comparison['passed'] else 'numerical_failure',
                                comparison=comparison, execution=execution,
                                decrypted_sha256=digest(out/'decrypted.npy'),
                                execution_sha256=digest(out/'execution.json'))
                except Exception as error:
                    item.update(status='execution_failed', diagnostic=str(error))
                require(all(digest(keys/n) == h for n,h in key_hashes.items()), 'Runtime mutated keys')
                require(all(digest(out/n) == h for n,h in template['hashes'].items()), 'Runtime mutated artifact/input')
                print('trial', trial, name, scale, item['status'],
                      item.get('comparison', {}).get('max_absolute_error', item.get('diagnostic')), flush=True)
                dump(root/'report.json', report)
            saved['status']='passed' if (len(row['executions']) == len(SOURCES)*len(SCALES) and
                all(x['status'] == 'passed' for x in row['executions'])) else 'failed'
        except Exception as error:
            saved.update(status='failed', diagnostic=str(error))
            raise
        finally:
            row['status']=saved['status']
            dump(run/'report.json', saved)
            if saved.get('parameters'):
                from result_retention import cleanup_run
                cleanup_run(run, RESULTS)
                row['cleanup']=json.loads((run/'key-cleanup-outcome.json').read_text())
            dump(root/'report.json', report)
    for field in ('source_hashes', 'runtime_hashes'):
        require(all(digest(Path(p)) == h for p,h in report[field].items()), 'Evidence/default/binary changed')
    report['summary']=summarize(report)
    report['status']='diagnostic_completed'
    dump(root/'report.json', report)
    print(json.dumps(report['summary'], indent=2), flush=True)
    return 0


def main():
    require(Path.cwd().resolve() == ROOT, 'Requires source root')
    if sys.argv[1:2] != ['--inside']:
        require(not sys.argv[1:], 'Fixed diagnostic has no scope/parameter overrides')
        os.umask(0o077)
        root=Path(tempfile.mkdtemp(prefix='hevm-graph-precision-', dir=RESULTS))
        print('Graph precision diagnostic: '+str(root), flush=True)
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'), str(SCRIPT), '--inside', str(root)])
        return enter_nix(cmd, seconds=1800)
    root=Path(sys.argv[2]).resolve()
    require(root.parent == RESULTS and root.name.startswith('hevm-graph-precision-') and not any(root.iterdir()),
            'Requires fresh diagnostic directory')
    return inside(root)


if __name__ == '__main__':
    raise SystemExit(main())
