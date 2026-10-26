"""S1 §9: preserve the native pass prefix while locating management differences."""

import gc
import json
import os
from pathlib import Path

from .. import toolchain
from ..backends import EmitBackend
from ..dsl.parser import parse
from ..lower import lower
from ..manage import RecordingBackend, manage, emit, emit_raw, read
from ..manage.mlir_read import graph_backend, RAW_OPS
from ..paths import RESULTS_DIR
from .s1_managed import compare_ir, inventory, write


def pipeline(stage, profile):
    """optimizer.cpp:578–606; registered spelling from Passes.td:157–171."""
    greedy = ('remove-bootstrap,greedy-bootstrap-placement{waterline='+str(profile['waterline'])+
              ' output-val='+str(profile['output_val'])+' depth-dp=true}')
    if stage != 'a':
        greedy += ',early-modswitch'
    tail = ','+toolchain.EARTH_TAIL_PASSES if stage == 'c' else ''
    return 'builtin.module(func.func('+greedy+')'+tail+')'


def stats(function):
    from collections import Counter
    return dict(operations=len(function.nodes), kinds=dict(Counter(n.kind for n in function.nodes)),
                boots=sum(n.kind == 'bootstrap' for n in function.nodes))


def preprocess(recorded, directory, profile, *, name='golden', compressed=False):
    """§10: run the compiler's save() passes and return its shared input DAG."""
    import shutil
    directory.mkdir()
    metrics = {}
    emit_raw(recorded,directory/'raw.earth.mlir',profile,name=name,metrics=metrics,compressed=compressed)
    shutil.copyfile(toolchain.BOOT_CONFIG_DIR/'compiler-config.json',directory/'boot-compiler-config.json')
    command = toolchain.to_host_argv(toolchain.frontend_preprocess_argv('dacapo-depth-dp','qwen48',directory),directory)
    status = toolchain.trusted_run(directory,command,directory/'preprocess.log',
        env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
    write(directory/'PREPROCESS.json',dict(emit=metrics,status=status,argv=command))
    if status['exit']:
        raise RuntimeError('Frontend preprocessing failed')
    function = read(directory/'preprocessed.earth.mlir',profile,allowed=RAW_OPS)
    return graph_backend(function,recorded.slots), function, dict(emit=metrics,status=status,argv=command)


def minimal_input(shared):
    """§9 fixture: one branch at multiplication depth 3 and one shortcut."""
    recorded = RecordingBackend(1024)
    x = recorded.input(0)
    first = -x
    value = first*x
    value = value*value
    value = value*value
    second = first if shared else -x
    recorded.output([value+second])
    return recorded


def input_proof(args):
    """Minimal §9 fanout: save() CSE shares -x before the management passes."""
    from .s1_managed import replay
    toolchain.setup_env()
    profile = toolchain.management_profile()
    result = dict(round='S1', section=9, mode='frontend-input-proof', complete=False,
                  llm_calls=0, report_images=0, variants={})
    for shared in (False, True):
        label = 'shared-value' if shared else 'repeated-expression'
        directory = args.out/label
        directory.mkdir()
        recorded = minimal_input(shared)
        emitter = EmitBackend(1024)
        replay(recorded, emitter)
        emitter.write(directory)
        row = dict(recorded_negates=sum(n.kind == 'negate' for n in recorded.nodes),
                   phase_comparisons={}, emit={})
        result['variants'][label] = row
        row['trace'] = toolchain.trace(directory, trusted=True)
        if row['trace']['exit']:
            raise RuntimeError('Minimal input proof trace failed')
        source = directory/'candidate_trace.mlir'
        row['frontend_negates'] = sum('"earth.negate"' in line for line in source.open())
        shared_input,preprocessed,row['preprocess'] = preprocess(recorded,directory/'preprocessed',profile)
        native_input = read(source,profile)
        row['phase_comparisons']['0'] = compare_ir(native_input,preprocessed)
        row['managed_input_negates'] = sum(n.kind=='negate' for n in shared_input.nodes)
        if not row['phase_comparisons']['0']['equal'] or row['phase_comparisons']['0']['conclusion']!='相同':
            write(args.out/'REPORT.json',result)
            raise RuntimeError('Minimal stage 0 differs')
        toolchain.prune(directory/'preprocessed')

        def capture(stage, function):
            compiler = directory/('compiler-'+stage)
            candidate = directory/('managed-'+stage)
            compiler.mkdir(); candidate.mkdir()
            argv = [str(toolchain.host_paths()['hecate_opt']), str(source),
                    '--ckks-config='+str(toolchain.BOOT_CONFIG_DIR/'compiler-config.json'),
                    '--mlir-disable-threading', '--verify-each',
                    '--pass-pipeline='+pipeline(stage, profile), '-o', str(compiler/'managed.mlir')]
            status = toolchain.trusted_run(compiler, argv, compiler/'prefix.log',
                env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
            if status['exit']:
                raise RuntimeError('Minimal native '+stage+' failed')
            row['emit'][stage] = {}
            emit(function, candidate/'managed.mlir', metrics=row['emit'][stage])
            for path in (compiler, candidate):
                normalized = [str(toolchain.host_paths()['hecate_opt']), str(path/'managed.mlir'),
                    '--ckks-config='+str(toolchain.BOOT_CONFIG_DIR/'compiler-config.json'),
                    '--mlir-disable-threading', '--verify-each',
                    '--pass-pipeline=builtin.module(canonicalize)',
                    '-o', str(path/'normalized.earth.mlir')]
                status = toolchain.trusted_run(path, normalized, path/'canonicalize.log',
                    env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
                if status['exit']:
                    raise RuntimeError('Minimal canonicalization failed')
            a = read(compiler/'normalized.earth.mlir', profile)
            b = read(candidate/'normalized.earth.mlir', profile)
            row['phase_comparisons'][stage] = dict(compare_ir(a,b),
                                                  compiler_stats=stats(a),managed_stats=stats(b))
            for path in (compiler,candidate):
                # These examples have no public constants; keep a compact IR
                # structure beside the comparison while pruning the MLIR itself.
                write(path/'STRUCTURE.json', [dict(id=n.id,kind=n.kind,args=n.args,
                    scale=n.type.scale,level=n.type.level,attrs=n.attrs) for n in (a if path==compiler else b).nodes])
                toolchain.prune(path)
        manage(shared_input, profile, stage_callback=capture)
        if any(not c['equal'] or c['conclusion']!='相同' for c in row['phase_comparisons'].values()):
            write(args.out/'REPORT.json',result)
            raise RuntimeError('Minimal stage comparison differs')
        row['pruned'] = toolchain.prune(directory)
        write(args.out/'REPORT.json', result)
    result['complete'] = True
    return result


def diagnose(args):
    """Each compiler prefix starts at the same trace: parsing b can reorder uses."""
    toolchain.setup_env()
    row = next(r for r in inventory() if r['label'] == args.case)
    if 'unit' in row or row.get('info'):
        raise ValueError('The §9 full-model diagnosis uses an existing frozen model')
    from ..models.zoo import get
    graph = get(row['model'], RESULTS_DIR/'models-v2').import_graph()
    program = parse(Path(row['source']).read_text())
    profile = toolchain.management_profile()
    result = dict(round='S1', section=9, mode='pass-diagnosis', case=row, complete=False,
                  llm_calls=0, report_images=0, profile=profile, commands=[], comparisons={}, pruned=[])
    original = args.out/'compiler'
    original.mkdir()

    def save():
        write(args.out/'REPORT.json', result)

    def run(source, target, passes, label):
        command = [str(toolchain.host_paths()['hecate_opt']), str(source),
                   '--ckks-config='+str(toolchain.BOOT_CONFIG_DIR/'compiler-config.json'),
                   '--mlir-disable-threading', '--verify-each', '--pass-pipeline='+passes,
                   '-o', str(target)]
        result['commands'].append(dict(label=label, argv=command))
        status = toolchain.trusted_run(target.parent, command, target.parent/(label+'.log'),
            env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
        result['commands'][-1]['status'] = status
        save()
        if status['exit']:
            raise RuntimeError(label+' failed')

    try:
        help_path = args.out/'pass-help.log'
        result['help'] = toolchain.trusted_run(args.out,
            [str(toolchain.host_paths()['hecate_opt']), '--help'], help_path,
            env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
        help_text = help_path.read_text()
        result['registered_options'] = {name:name in help_text for name in
            ('--greedy-bootstrap-placement', '--early-modswitch', '--remove-bootstrap',
             '--depth-dp', '--output-val')}
        save()
        if result['help']['exit'] or not all(result['registered_options'].values()):
            raise RuntimeError('The required registered pass names/options are absent')
        emitter = EmitBackend(program.header.slots)
        lower(program, graph.weights, emitter)
        emitter.write(original)
        del emitter
        gc.collect()
        result['trace'] = toolchain.trace(original, trusted=True)
        save()
        if result['trace']['exit']:
            raise RuntimeError('Compiler reference trace failed')
        result['pruned'].append(toolchain.prune(original,keep=('candidate_trace.mlir',)))
        recorded = RecordingBackend(program.header.slots,locations={op.id:op for op in program.ops})
        lower(program,graph.weights,recorded)
        shared,front,result['preprocess'] = preprocess(recorded,args.out/'preprocessed',profile)
        del recorded,graph
        gc.collect()
        native_input = read(original/'candidate_trace.mlir',profile)
        result['comparisons']['0'] = compare_ir(native_input,front)
        del native_input,front
        gc.collect()
        result['pruned'].append(toolchain.prune(args.out/'preprocessed'))
        save()
        if not result['comparisons']['0']['equal'] or result['comparisons']['0']['conclusion']!='相同':
            raise RuntimeError('Stage 0 differs before management diagnosis')
        result['compiler'] = toolchain.compile(original, 'dacapo-depth-dp', 'qwen48', trusted=True)
        save()
        if result['compiler']['exit']:
            raise RuntimeError('Compiler reference compile failed')
        # Keep the pristine trace for ALL prefixes; intermediate reparsing would
        # discard the use-list history being investigated by §9.
        result['pruned'].append(toolchain.prune(original,
            keep=('candidate_trace.mlir', 'lowered.earth.mlir')))
        golden = read(original/'lowered.earth.mlir', profile)
        result['compiler_stats'] = stats(golden)
        result['pruned'].append(toolchain.prune(original, keep=('candidate_trace.mlir',)))
        references = {}
        for stage in ('c', 'a', 'b'):
            directory = args.out/('compiler-'+stage)
            directory.mkdir()
            references[stage] = directory/'managed.mlir'
            run(original/'candidate_trace.mlir', references[stage], pipeline(stage, profile), 'prefix-'+stage)
            if stage == 'c':
                stage_c = read(references[stage], profile)
                result['comparisons']['c-vs-original'] = compare_ir(golden, stage_c)
                result['pruned'].append(toolchain.prune(directory))
                del golden, stage_c
                gc.collect()
                save()
                if not result['comparisons']['c-vs-original']['equal']:
                    raise RuntimeError('The complete pass prefix differs from original lowered.earth.mlir')
        result['pruned'].append(toolchain.prune(original))

        def capture(stage, function):
            directory = args.out/('managed-'+stage)
            directory.mkdir()
            cpp_path = references[stage]
            run(cpp_path, cpp_path.parent/'normalized.earth.mlir',
                'builtin.module(canonicalize)', 'canonicalize-compiler-'+stage)
            reference = read(cpp_path.parent/'normalized.earth.mlir', profile)
            result['pruned'].append(toolchain.prune(cpp_path.parent))
            result['emit-'+stage] = {}
            emit(function, directory/'managed.mlir', metrics=result['emit-'+stage])
            run(directory/'managed.mlir', directory/'normalized.earth.mlir',
                'builtin.module(canonicalize)', 'canonicalize-managed-'+stage)
            candidate = read(directory/'normalized.earth.mlir', profile)
            comparison = compare_ir(reference, candidate)
            comparison.update(compiler_stats=stats(reference), managed_stats=stats(candidate))
            result['comparisons'][stage] = comparison
            write(directory/'management.json', function.summary())
            result['pruned'].append(toolchain.prune(directory))
            del reference, candidate
            gc.collect()
            save()

        manage(shared, profile, stage_callback=capture)
        result['first_differing_pass'] = next((name for stage,name in
            (('a','GreedyBootstrapPlacement'),('b','EarlyModswitch'))
            if not result['comparisons'][stage]['equal']), None)
        result['complete'] = True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for directory in args.out.iterdir():
            if directory.is_dir():
                result['pruned'].append(toolchain.prune(directory))
        save()
    return result
