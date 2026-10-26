"""Finite S1 comparisons; system Python owns results and the registry."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from ..paths import RESULTS_DIR, REPO, PLATFORM_DIR
from .. import toolchain


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def inventory():
    from ..tests.test_p2_export import cases
    rows = [dict(label='unit-'+name, unit=name) for name in
            ('add', 'chain', 'residual', 'constants', 'no-final-mul')]
    rows += [dict(label=name, model=model, source=str(path)) for name, model, path in cases()[:-1]]
    from .u3_cnn_operators import program_path, COMPLETED
    u2 = json.loads((RESULTS_DIR/'u2-models-20261024T125911245307/COMPARISON.json').read_text())
    rows += [dict(label='u2-'+r['model']+'-'+str(r['slots']), model=r['model'], source=str(program_path(r)))
             for r in u2['variants']]
    for name in ('resnet20_relu', 'squeezenet_silu_avgpool', 'squeezenet_relu_maxpool'):
        model = RESULTS_DIR/COMPLETED[name]/'model'
        rows.append(dict(label='u3a-'+name, info=str(model), source=str(model/'program.ptdsl')))
    return rows


def compare_ir(a, b):
    """Compare ordered SSA links, then ordered-edge DAG isomorphism if order differs."""
    import numpy as np
    def records(f):
        ids = {n.id: i for i, n in enumerate(f.nodes)}
        return [(n.kind, tuple(ids.get(x, x) for x in n.args),
                 (n.type.cipher, n.type.scale, n.type.level), n.attrs) for n in f.nodes]
    first = None
    a_records, b_records = records(a), records(b)
    ordered = a_records == b_records
    for i, (x, y) in enumerate(zip(a.nodes, b.nodes)):
        if a_records[i] != b_records[i]:
            first = dict(operation=i, compiler=a_records[i], managed=b_records[i])
            break
        if x.kind == 'constant' and not np.array_equal(x.data, y.data):
            ordered = False
            first = dict(operation=i, reason='constant payload')
            break
    if len(a.nodes) != len(b.nodes):
        return dict(equal=False, conclusion='不同', first_difference=dict(
                    reason='operation count', compiler=len(a.nodes), managed=len(b.nodes),
                    first_order_difference=first or dict(operation=min(len(a.nodes),len(b.nodes)), reason='trailing operations')))
    boundary = a.inputs == b.inputs and a.result_types == b.result_types and a.attrs == b.attrs
    mapping = {-i-1: -i-1 for i in range(len(a.inputs))}
    pool = defaultdict(list)

    def key(n, args):
        return (n.kind, tuple(args), n.type, json.dumps(n.attrs, sort_keys=True))

    for n in b.nodes:
        pool[key(n, n.args)].append(n)
    for n in a.nodes:
        candidates = pool[key(n, [mapping[v] for v in n.args])]
        found = next((i for i, other in enumerate(candidates)
                      if n.kind != 'constant' or np.array_equal(n.data, other.data)), None)
        if found is None:
            return dict(equal=False, conclusion='不同', first_difference=first or
                        dict(operation=n.id, kind=n.kind, reason='no isomorphic operation'))
        mapping[n.id] = candidates.pop(found).id
    iso = boundary and [mapping[v] for v in a.outputs] == b.outputs
    return dict(equal=iso, conclusion=('相同' if ordered else '同构') if iso else '不同',
                operations=len(a.nodes), boots=sum(n.kind == 'bootstrap' for n in a.nodes),
                first_order_difference=first, boundary_equal=boundary,
                order_note=None if ordered else 'independent operations/constants have different stable order')


def replay(recorded, backend):
    """Render trusted recorded arithmetic through the unchanged compiler-mode emitter."""
    values = {-i-1: backend.input(i) for i in range(len(recorded._inputs))}
    for n in recorded.nodes:
        args = [values[a] for a in n.args]
        if n.kind == 'constant':
            value = backend.const(n.data)
        elif n.kind == 'add':
            value = args[0]+args[1]
        elif n.kind == 'mul':
            value = args[0]*args[1]
        elif n.kind == 'negate':
            value = -args[0]
        elif n.kind == 'rotate':
            value = args[0].rotate(n.attrs['offset'][0])
        else:
            raise ValueError(n.kind)
        values[n.id] = value
    backend.output([values[a.nid] for a in recorded._outputs])


def compiler_reference(row, directory):
    """Recreate the pruned constant references required by S1 §8.3."""
    from ..backends import EmitBackend
    from ..dsl.parser import parse
    from ..lower import lower
    from ..tests.test_s1_manage import record_case
    directory.mkdir()
    if 'unit' in row:
        recorded, _ = record_case(row['unit'])
        emitter = EmitBackend(recorded.slots)
        replay(recorded, emitter)
        del recorded
    else:
        from ..models.zoo import get
        graph = get(row['model'], RESULTS_DIR/'models-v2').import_graph()
        program = parse(Path(row['source']).read_text())
        emitter = EmitBackend(program.header.slots)
        lower(program, graph.weights, emitter)
        del graph
    emitter.write(directory)
    del emitter
    gc.collect()
    result = dict(trace=toolchain.trace(directory, trusted=True))
    write(directory/'REFERENCE.json', result)
    if result['trace']['exit'] != 0:
        raise RuntimeError('RMS reference trace failed')
    result['compiler'] = toolchain.compile(directory, 'dacapo-depth-dp', 'qwen48', trusted=True)
    write(directory/'REFERENCE.json', result)
    if result['compiler']['exit'] != 0:
        raise RuntimeError('RMS reference compile failed')
    result['pruned'] = toolchain.prune(directory, keep=('lowered.earth.mlir', 'lowered._hecate_golden.bundle'))
    write(directory/'REFERENCE.json', result)
    return result


def rms_inside(args):
    import unittest
    from ..tests.test_s1_rms import compiler_suite, ARTIFACT_RESULTS
    toolchain.setup_env()
    labels = {'mlp', 'cnn_plain', 'unit-add', 'unit-chain', 'unit-residual',
              'unit-constants', 'unit-no-final-mul'}
    result = dict(round='S1', part='rms', complete=False, references={}, llm_calls=0, report_images=0)
    for row in inventory():
        if row['label'] in labels:
            directory = args.out/row['label']
            result['references'][row['label']] = compiler_reference(row, directory)
            write(args.out/'REPORT.json', result)
    write(args.out/'REFERENCES.json', {label:str(args.out/label) for label in result['references']})
    outcome = unittest.TextTestRunner(verbosity=2, failfast=True).run(compiler_suite(args.out))
    write(args.out/'RMS_CHECKS.json', ARTIFACT_RESULTS)
    result.update(tests=outcome.testsRun, failures=len(outcome.failures), errors=len(outcome.errors),
                  skips=len(outcome.skipped), complete=outcome.wasSuccessful())
    result['pruned'] = {label:toolchain.prune(args.out/label) for label in result['references']
                        if label != 'cnn_plain' or not result['complete']}
    result['cnn_reference'] = str(args.out/'cnn_plain') if result['complete'] else None
    return result


def case_inside(args):
    from ..backends import EmitBackend, DagBackend
    from ..cost import load_latency
    from ..dsl.parser import parse
    from ..lower import lower
    from ..manage import RecordingBackend, manage, check, emit, read
    from ..manage.greedy import placement_view
    from ..placement import depth_dp
    from ..tests.test_s1_manage import record_case
    from .s0_pipeline_spike import compare_plans
    from .s1_stages import preprocess, stats
    from ..manage import streams
    import shutil
    row = next(r for r in inventory() if r['label'] == args.case)
    result = dict(case=row, complete=False, round='S1', llm_calls=0, report_images=0)
    profile = toolchain.management_profile()
    original, candidate = args.reference or args.out/'compiler', args.out/'dsl'
    if args.reference is None:
        original.mkdir()
    else:
        result['compiler_reference'] = str(original)
        if json.loads((original/'REFERENCE.json').read_text())['compiler']['exit'] != 0:
            raise RuntimeError('Compiler reference is incomplete')
    candidate.mkdir()
    streams.admit(args.out)

    def retain_text(label, path):
        if getattr(args,'proof_text',False):
            actual = streams.resolve(path)
            proof = args.out/'proof-text'
            proof.mkdir(exist_ok=True)
            target = proof/(label+'.mlir'+('.gz' if actual.suffix=='.gz' else ''))
            shutil.copyfile(actual,target)
            result.setdefault('text_outputs',{})[label] = str(target)
    if 'unit' in row:
        recorded, initial = record_case(row['unit'])
        del initial
        emitter = EmitBackend(recorded.slots)
        replay(recorded, emitter)
        direct = depth_dp(placement_view(recorded.nodes,recorded._inputs,[v.nid for v in recorded._outputs]),
            last_boot_level=4,boot_in_level=1,boot_out_level=5,latency_table=load_latency('qwen48'))
    else:
        if row.get('info'):
            from ..models.info import ModelInfo
            graph = ModelInfo.load(Path(row['info'])).graph
        else:
            from ..models.zoo import get
            graph = get(row['model'], RESULTS_DIR/'models-v2').import_graph()
        program = parse(Path(row['source']).read_text())
        emitter = EmitBackend(program.header.slots)
        lower(program, graph.weights, emitter)
        recorded = RecordingBackend(program.header.slots, locations={op.id: op for op in program.ops})
        lower(program, graph.weights, recorded)
        dag = DagBackend(program.header.slots)
        lower(program, graph.weights, dag)
        direct = depth_dp(dag, last_boot_level=4, boot_in_level=1, boot_out_level=5,
                          latency_table=load_latency('qwen48'))
        del graph, dag
    result.update(cost_cuts=direct.cuts,cost_boots=direct.boots)
    if args.reference is None:
        emitter.write(original)
    del emitter
    gc.collect()
    write(args.out/'REPORT.json', result)
    toolchain.setup_env()
    if args.reference is None:
        result['trace'] = toolchain.trace(original, trusted=True)
        write(args.out/'REPORT.json', result)
        if result['trace']['exit'] != 0:
            raise RuntimeError('compiler-mode trace failed')
    if getattr(args,'compressed',False):
        streams.compress_ir(original/'candidate_trace.mlir',force=True)
    if not streams.resolve(original/'candidate_trace.mlir').is_file():
        raise RuntimeError('Stage 0 requires the original compiler trace')
    # Once trace completes its NPZ is consumed. Keep only the trace while the
    # compiler prepares the same shared graph independently of Python tracing.
    result['stage0_pruned'] = toolchain.prune(original,
        keep=('candidate_trace.mlir','lowered.earth.mlir','lowered._hecate_golden.bundle'))
    shared,front,result['frontend_preprocess'] = preprocess(recorded,args.out/'preprocessed',profile,
        compressed=getattr(args,'compressed',False))
    retain_text('preprocessed',args.out/'preprocessed/preprocessed.earth.mlir')
    del recorded
    gc.collect()
    native = read(original/'candidate_trace.mlir',profile)
    result['stage0'] = dict(compare_ir(native,front),compiler_stats=stats(native),managed_stats=stats(front))
    del native,front
    gc.collect()
    write(args.out/'REPORT.json',result)
    if not result['stage0']['equal'] or result['stage0']['conclusion']!='相同':
        raise RuntimeError('Frontend stage 0 differs; stop at first difference')
    result['frontend_pruned'] = toolchain.prune(args.out/'preprocessed')
    managed = manage(shared,profile)
    del shared
    result.update(cuts=managed.cuts,preprocessed_boots=sum(n.kind=='bootstrap' for n in managed.nodes),
                  placement_cuts_equal=managed.cuts==direct.cuts)
    result['placement_boots_equal'] = result['preprocessed_boots']==direct.boots
    check(managed)
    result['managed_checker'] = True
    write(candidate/'management.json',managed.summary())
    if args.reference is None:
        result['compiler'] = toolchain.compile(original,'dacapo-depth-dp','qwen48',trusted=True)
        write(args.out/'REPORT.json',result)
        if result['compiler']['exit']:
            raise RuntimeError('compiler-mode compile failed')
    result['pruned_stages'] = [toolchain.prune(original,
        keep=('lowered.earth.mlir', 'lowered._hecate_golden.bundle'))]
    streams.compress_ir(original/'lowered.earth.mlir',force=getattr(args,'compressed',False))
    # Release each consumed intermediate before the next trusted invocation.
    # The pass sequence and original comparison baseline stay unchanged.
    a = read(original/'lowered.earth.mlir', profile)
    check(a)
    result['compiler_checker'] = True

    def normalize(directory, source):
        command = [str(toolchain.host_paths()['hecate_opt']), str(source),
                   '--ckks-config='+str(toolchain.BOOT_CONFIG_DIR/'compiler-config.json'),
                   '--mlir-disable-threading', '--verify-each',
                   '--pass-pipeline=builtin.module('+toolchain.EARTH_TAIL_PASSES+')',
                   '-o', str(directory/'normalized.earth.mlir')]
        status = toolchain.trusted_run(directory, command, directory/'normalize.log',
            env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
        result[directory.name+'_normalize'] = status
        write(args.out/'REPORT.json', result)
        if status['exit'] != 0:
            raise RuntimeError('Earth normalization failed')

    normalize(original, original/'lowered.earth.mlir')
    repeated = read(original/'normalized.earth.mlir', profile)
    result['baseline_idempotence'] = compare_ir(a, repeated)
    del repeated
    gc.collect()
    if not result['baseline_idempotence']['equal'] or result['baseline_idempotence']['conclusion'] != '相同':
        print('Baseline Earth tail is not order-idempotent; recorded without changing the baseline', flush=True)
    result['pruned_stages'].append(toolchain.prune(original, keep=('lowered._hecate_golden.bundle',)))
    result['emit'] = {}
    emit(managed, candidate/'managed.mlir', metrics=result['emit'],compressed=getattr(args,'compressed',False))
    retain_text('managed',candidate/'managed.mlir')
    del managed
    gc.collect()
    normalize(candidate, candidate/'managed.mlir')
    retain_text('tail',candidate/'normalized.earth.mlir')
    b = read(candidate/'normalized.earth.mlir', profile)
    check(a); check(b)
    result['ir'] = compare_ir(a, b)
    del a, b
    gc.collect()
    write(args.out/'REPORT.json', result)
    if not result['ir']['equal']:
        raise RuntimeError('non-isomorphic managed Earth IR; stop at first difference')
    result['pruned_stages'].append(toolchain.prune(candidate, keep=('managed.mlir',)))
    shutil.copyfile(toolchain.BOOT_CONFIG_DIR/'compiler-config.json', candidate/'boot-compiler-config.json')
    result['managed_compile'] = []
    for i, argv in enumerate(toolchain.managed_compile_argv('dacapo-depth-dp', 'qwen48', candidate)):
        status = toolchain.trusted_run(candidate, toolchain.to_host_argv(argv, candidate),
            candidate/f'compile-{i+1}.log', env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
        result['managed_compile'].append(status)
        write(args.out/'REPORT.json', result)
        if status['exit'] != 0:
            raise RuntimeError('managed conversion/export failed')
        if i == 0:
            result['pruned_stages'].append(toolchain.prune(candidate, keep=('managed.ckks.mlir',)))
    result['plan'] = compare_plans(original/'lowered._hecate_golden.runtime-plan.json',
                                   candidate/'lowered._hecate_golden.runtime-plan.json')
    write(args.out/'REPORT.json', result)
    if not result['plan']['equal']:
        raise RuntimeError('RuntimePlan comparison differs; stop')
    result['pruned'] = {d.name: toolchain.prune(d) for d in (original, candidate)}
    result['complete'] = True
    return result


def inside(args):
    result = dict(complete=False, round='S1', part=args.part, llm_calls=0, report_images=0)
    try:
        if args.part == 'unit':
            import unittest
            from ..tests import test_s1_stages
            test_s1_stages.OUTPUT_DIR = args.out/'native-shared-fanout'
            files = sorted(Path('pt_agent/tests').glob('test_s1_*.py'))
            if args.group=='stream':
                files = [p for p in files if p.name=='test_s1_compat.py']
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
            outcome = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
            result.update(tests=outcome.testsRun, failures=len(outcome.failures), errors=len(outcome.errors),
                          skips=len(outcome.skipped), complete=outcome.wasSuccessful(),group=args.group)
        elif args.part == 'inventory':
            result.update(cases=inventory(), complete=True)
        elif args.part == 'rms':
            result = rms_inside(args)
        elif args.part == 'stages':
            from .s1_stages import diagnose
            result = diagnose(args)
        elif args.part == 'input-proof':
            from .s1_stages import input_proof
            result = input_proof(args)
        elif args.part in ('compat','p2','regression','compression','server'):
            from .s1_acceptance import compatibility, package_case, regression, compression, server_case
            result = dict(compat=compatibility,p2=package_case,regression=regression,compression=compression,
                          server=server_case)[args.part](args)
        else:
            result = case_inside(args)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        existing = args.out/'REPORT.json'
        if existing.exists():
            result.update(json.loads(existing.read_text()))
        result.update(complete=False, error=f'{type(exc).__name__}: {exc}')
        if args.part == 'case':
            result['pruned_after_stop'] = {str(d):toolchain.prune(d) for d in
                (args.reference or args.out/'compiler', args.out/'dsl',args.out/'preprocessed') if d.is_dir()}
        elif args.part in ('input-proof','p2'):
            result['pruned_after_stop'] = {str(d.relative_to(args.out)):toolchain.prune(d)
                for d in args.out.rglob('*') if d.is_dir()}
    write(args.out/'REPORT.json', result)
    return 0 if result['complete'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('unit','inventory','rms','case','stages','input-proof','compat','p2','regression','compression','server'), required=True)
    parser.add_argument('--case')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--group',choices=('main','packaging','tensor','weight','intake','stream'))
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if args.inside:
        return inside(args)
    args.out = args.out or RESULTS_DIR/('s1-'+(args.case or args.part)+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    args.out.mkdir(parents=True, exist_ok=False)
    if args.part=='compat':
        import tarfile
        from .s1_acceptance import BASE
        archive = args.out/'baseline.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git','archive',BASE,'DSL/DSL_agent/scripts/pt_agent'],cwd=REPO,stdout=stream,check=True)
        args.snapshot = args.out/'baseline-tree'
        args.snapshot.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(args.snapshot)
    command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '3600', '--',
               '-m', 'pt_agent.experiments.s1_managed', '--inside', '--part', args.part, '--out', str(args.out)]
    if args.case:
        command += ['--case', args.case]
    if args.reference:
        command += ['--reference', str(args.reference)]
    if args.snapshot:
        command += ['--snapshot',str(args.snapshot)]
    if args.group:
        command += ['--group',args.group]
    write(args.out/'argv.json', command)

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))

    start = time.monotonic()
    print('RUN='+str(args.out), flush=True)
    with (args.out/'run.log').open('w') as stream:
        process = subprocess.run(command, cwd=REPO/'DSL/DSL_agent/scripts', stdout=stream,
                                 stderr=subprocess.STDOUT, preexec_fn=limits, timeout=3720)
    report_path = args.out/'REPORT.json'
    result = json.loads(report_path.read_text()) if report_path.exists() else dict(
        complete=False,round='S1',part=args.part,group=args.group,stage='environment-startup',
        tests=0,llm_calls=0,error='Inner process exited before REPORT.json; see run.log')
    result.update(exit=process.returncode, seconds=time.monotonic()-start)
    write(args.out/'REPORT.json', result)
    from ..registry import append, git_rev, new_run_id
    os.environ.setdefault('PT_RESULTS', str(RESULTS_DIR))
    append(dict(run_id=new_run_id('S1', args.case or args.part, 'managed'), round='S1',
                model=args.case or args.part, mode='managed', git_rev=git_rev(), metrics=result,
                artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(args.out/'REPORT.json'), flush=True)
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
