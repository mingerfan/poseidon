"""U4 §7.7.1: finite read-only comparison of raw, numbered and frontend DAGs.

Value numbering lives in this diagnostic only until §7.7.1 confirms its gate.
Native trace instrumentation supplies PT-DSL ownership locations only; library
files, arithmetic, pass sequences, the frozen program and its constants stay intact.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import importlib
import inspect
import json
import os
from pathlib import Path
import resource
import runpy
import shutil
import subprocess
import sys
import time

from ..paths import REPO, RESULTS_DIR
from .u4_point_diagnosis import write


def numbered_dag(slots, *, reuse):
    """Keep NumPy-dependent graph code inside Nix; the outer launcher is stdlib."""
    from ..backends.base import _Cipher
    from ..backends.dag import DagBackend

    class NumberedDag(DagBackend):
        """Ordered operands, operation attributes and Backend.const k names."""
        def __init__(self, slots, *, reuse):
            super().__init__(slots)
            self.reuse = reuse
            self.numbering = {}

        def _apply(self, kind, left, right=None, *, step=None):
            operands = tuple(('cipher', value.nid) if isinstance(value, _Cipher) else ('plain', value.name)
                             for value in (left, right) if value is not None)
            key = (kind, operands, step)
            if self.reuse and key in self.numbering:
                return self.numbering[key]
            result = super()._apply(kind, left, right, step=step)
            if self.reuse:
                self.numbering[key] = result
            return result

    return NumberedDag(slots,reuse=reuse)


def trace(args):
    """The existing package trace with diagnostic PT-DSL source attribution."""
    sys.path.insert(0, str(args.package))
    config = json.loads((args.package/'config/model.json').read_text())
    source_lines = json.loads((args.package/'config/management.json').read_text())['source_lines']
    runtime = importlib.import_module(config['name']+'.runtime')
    expr = importlib.import_module('hecate.expr')
    originals = {name:getattr(expr, name) for name in ('getCallerFrame','getProperFrame')}

    def attributed(name):
        original = originals[name]

        def location():
            caller = inspect.currentframe().f_back
            frame = caller
            try:
                while frame is not None:
                    if frame.f_code is runtime.model.forward.__code__:
                        owner = source_lines.get(str(frame.f_lineno))
                        if owner is not None:
                            return (frame, 'ptdsl-op-'+str(owner['id']), owner['lineno'],
                                    frame.f_code.co_name, None, None)
                    frame = frame.f_back
                if name == 'getCallerFrame':
                    frame = caller.f_back
                else:
                    frame = caller
                    while frame is not None and frame.f_code.co_filename == expr.__file__:
                        frame = frame.f_back
                return expr._frameInfo(frame)
            finally:
                del frame, caller

        return location

    for name in originals:
        setattr(expr, name, attributed(name))
    try:
        sys.argv = [str(args.package/'trace_model.py'), '--weights', str(args.weights),
                    '--output-dir', str(args.out/'trace'), '--bytecode']
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        for name, original in originals.items():
            setattr(expr, name, original)


def native(args, result):
    from .. import toolchain
    from ..manage import streams
    toolchain.setup_env()
    from hecate_python_env import VENV
    env = toolchain.package_trace_env()
    env['PYTHONPATH'] = str(REPO/'DSL/DSL_agent/scripts')+os.pathsep+env['PYTHONPATH']
    if 'bytecode_source' in result:
        bytecode = Path(result['bytecode_source'])
    else:
        result['trace'] = toolchain.trusted_run(args.out, [str(VENV/'bin/python'), '-B', '-m',
            'pt_agent.experiments.u4_cse_diagnosis', '--entry', 'trace', '--package', str(args.package),
            '--weights', str(args.weights), '--out', str(args.out)], args.out/'trace.log',env_extra=env)
        write(args.out/'REPORT.json', result)
        if result['trace']['exit']:
            raise RuntimeError('Diagnostic bytecode trace failed')
        bytecode = args.out/'trace/trace_model.mlirbc'
    result['bytecode_bytes'] = bytecode.stat().st_size
    structure = args.out/'preprocessed.structure.mlir'
    command = [str(toolchain.host_paths()['hecate_opt']), str(bytecode),
               '--ckks-config='+str(args.package/'config/compiler-config.json'), '--mlir-disable-threading',
               '--mlir-elide-elementsattrs-if-larger=1', '--mlir-print-op-generic',
               '--mlir-print-debuginfo', '-o', str(structure)]
    result['structure_argv'] = command
    result['structure_export'] = toolchain.trusted_run(args.out, command, args.out/'structure.log',
        env_extra={'LD_LIBRARY_PATH':os.environ['HECATE_PYTHON_LIBRARY_PATH']})
    write(args.out/'REPORT.json', result)
    if result['structure_export']['exit']:
        help_result = subprocess.run([str(toolchain.host_paths()['hecate_opt']),'--help'],
            capture_output=True,text=True)
        (args.out/'hecate-help.txt').write_text(help_result.stdout+help_result.stderr)
        raise RuntimeError('No-pass structural text export failed; see hecate-help.txt')
    result['structure_text_bytes'] = structure.stat().st_size
    structure = streams.compress_ir(structure)
    result.update(structure=str(structure),structure_stored_bytes=structure.stat().st_size,
                  location_instrumentation='Existing P2 source_lines -> diagnostic ptdsl-op NameLoc ownership; files/arithmetic unchanged')
    return result


def placement(dag):
    from ..cost import load_latency
    from ..placement import depth_dp
    return depth_dp(dag,last_boot_level=4,boot_in_level=1,boot_out_level=5,
                    latency_table=load_latency('qwen48'))


def dag_summary(dag, program):
    place = placement(dag)
    rows = defaultdict(Counter)
    for kind, owner in zip(dag.kind, dag.prog_op):
        if kind != 'input':
            rows[owner][kind] += 1
    return dict(counts=dict(Counter(dag.kind)),arithmetic_ops=sum(sum(row.values()) for row in rows.values()),
                boundaries=place.barriers,cuts=place.cuts,boots=place.boots,max_depth=place.max_depth,
                latency_us=place.latency_us,
                by_line=[dict(id=op.id,line=op.lineno,op=op.kind,counts=dict(rows[op.id]),
                              total=sum(rows[op.id].values())) for op in program.ops])


def frontend_summary(function, slots, program):
    from ..manage.mlir_read import graph_backend
    from ..manage.greedy import placement_view
    backend = graph_backend(function, slots)
    dag = placement_view(function.nodes,backend._inputs,function.outputs)
    place = placement(dag)
    types = function.types()
    rows = defaultdict(Counter)
    counts = Counter()
    unknown = Counter()
    for node in function.nodes:
        if node.kind == 'constant':
            counts['constant'] += 1
            continue
        kind = {'rotate':'rot','negate':'neg'}.get(node.kind,node.kind)
        if kind == 'mul':
            kind = 'mul_cc' if all(types[value].cipher for value in node.args) else 'mul_cp'
        counts[kind] += 1
        rows[node.op_id][kind] += 1
        if node.op_id is None:
            unknown[kind] += 1
    return dict(counts=dict(counts),arithmetic_ops=sum(v for k,v in counts.items() if k!='constant'),
                boundaries=place.barriers,cuts=place.cuts,boots=place.boots,max_depth=place.max_depth,
                latency_us=place.latency_us,unattributed=dict(unknown),
                by_line=[dict(id=op.id,line=op.lineno,op=op.kind,counts=dict(rows[op.id]),
                              total=sum(rows[op.id].values())) for op in program.ops])


def diagnose(args, result):
    from ..dsl.parser import parse
    from ..lower import lower
    from ..models.info import ModelInfo
    from ..manage.mlir_read import read, RAW_OPS
    info = ModelInfo.load(args.prepared/'model')
    program = parse(args.program.read_text())
    comparisons = result['graphs'] = {}
    for label, reuse in (('raw',False),('numbered',True)):
        backend = numbered_dag(program.header.slots,reuse=reuse)
        lower(program,info.graph.weights,backend)
        comparisons[label] = dag_summary(backend,program)
        del backend
        gc.collect()
        write(args.out/'REPORT.json',result)
    function = read(Path(result['structure']),{},allowed=RAW_OPS,structure_only=True)
    comparisons['frontend'] = frontend_summary(function,program.header.slots,program)
    del function, info
    gc.collect()
    previous = json.loads(args.reference.read_text())
    log = args.reference.parent/'compile.log'
    import re
    boundaries = re.search(r'Depth-DP bootstrap boundaries: (\d+)',log.read_text())
    comparisons['runtime_plan'] = dict(counts=previous['plan']['ops'],arithmetic_ops=None,
        steps=previous['plan']['steps'],boundaries=int(boundaries[1]) if boundaries else None,
        boots=previous['plan']['boots'],source=str(args.reference))
    ratios = []
    differences = []
    for original, numbered, front in zip(*(comparisons[label]['by_line'] for label in ('raw','numbered','frontend'))):
        delta = original['total']-numbered['total']
        ratios.append(dict(id=original['id'],line=original['line'],op=original['op'],
                          raw=original['total'],numbered=numbered['total'],removed=delta,
                          ratio=original['total']/numbered['total'] if numbered['total'] else None))
        classes = sorted(set(numbered['counts'])|set(front['counts']))
        differing = {kind:dict(numbered=numbered['counts'].get(kind,0),frontend=front['counts'].get(kind,0))
                     for kind in classes if numbered['counts'].get(kind,0)!=front['counts'].get(kind,0)}
        if differing:
            differences.append(dict(id=original['id'],line=original['line'],op=original['op'],classes=differing))
    result['line_deduplication'] = ratios
    result['largest_deduplication'] = sorted(ratios,key=lambda row:row['removed'],reverse=True)[:10]
    result['line_count_differences'] = differences
    result['first_line_count_difference'] = differences[0] if differences else None
    result['cse_sufficient'] = comparisons['numbered']['boots']==comparisons['runtime_plan']['boots']
    result['frontend_matches_runtime'] = comparisons['frontend']['boots']==comparisons['runtime_plan']['boots']
    result['diagnosis_complete'] = True
    result['complete'] = result['cse_sufficient'] and result['frontend_matches_runtime']
    if not result['complete']:
        result['stop_reason'] = '§7.7.1: value numbering alone does not establish compiler bootstrap equality; no correction enabled'
    return result


def liveness(args, result):
    """Identify canonicalizer dead-code effects without rewriting the cost DAG."""
    from ..dsl.parser import parse
    from ..lower import lower
    from ..models.info import ModelInfo
    program = parse(args.program.read_text())
    info = ModelInfo.load(args.prepared/'model')
    backend = numbered_dag(program.header.slots,reuse=True)
    lower(program,info.graph.weights,backend)
    live = set()
    pending = [value.nid for value in backend._outputs]
    while pending:
        nid = pending.pop()
        if nid not in live:
            live.add(nid)
            pending.extend(backend.args[nid])
    counts,by_owner = Counter(),defaultdict(Counter)
    for nid,kind in enumerate(backend.kind):
        if nid in live and kind!='input':
            counts[kind] += 1
            by_owner[backend.prog_op[nid]][kind] += 1
    prior = json.loads(args.liveness_from.read_text())
    front = prior['graphs']['frontend']

    def representation_counts(row):
        # C++ createBinary represents subtraction by add plus explicit negation
        # (or a negated public constant). Compare additive and multiply classes;
        # report the extra native negations separately, without changing a DAG.
        return {kind:row.get('add',0)+row.get('sub',0) if kind=='additive' else row.get(kind,0)
                for kind in ('additive','mul_cc','mul_cp','rot')}

    rows = []
    for op,original,numbered,native_row in zip(program.ops,prior['graphs']['raw']['by_line'],
                                              prior['graphs']['numbered']['by_line'],front['by_line']):
        live_counts = dict(by_owner[op.id])
        rows.append(dict(id=op.id,line=op.lineno,op=op.kind,raw_total=original['total'],
            numbered_total=numbered['total'],live_total=sum(live_counts.values()),counts=live_counts,
            frontend_counts=native_row['counts'],classes_equal=representation_counts(live_counts)==representation_counts(native_row['counts'])))
    result.update(complete=True,observation_only=True,source_diagnosis=str(args.liveness_from),
        output_reachable_counts=dict(counts),output_reachable_total=sum(counts.values()),
        unreachable_arithmetic=prior['graphs']['numbered']['arithmetic_ops']-sum(counts.values()),
        comparable_classes_equal=representation_counts(counts)==representation_counts(front['counts']),
        frontend_explicit_negations=front['counts'].get('neg',0),by_line=rows,
        first_remaining_count_difference=next((row for row in rows if not row['classes_equal']),None),
        cost_model_changed=False,placement_rerun=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside',action='store_true')
    parser.add_argument('--entry',choices=('trace',))
    parser.add_argument('--stage',choices=('native','diagnose','liveness'))
    for key in ('out','package','weights','program','prepared','reference','resume','liveness-from'):
        parser.add_argument('--'+key,type=Path)
    args = parser.parse_args()
    if args.entry:
        trace(args)
        return 0
    if args.inside:
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,6*1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        path = args.out/'REPORT.json'
        result = json.loads(path.read_text())
        try:
            result = {'native':native,'diagnose':diagnose,'liveness':liveness}[args.stage](args,result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result.update(complete=False,error=f'{type(exc).__name__}: {exc}')
            write(args.out/'STOP.json',result)
        write(path,result)
        success = (args.stage=='native' and 'structure' in result or
                   args.stage=='diagnose' and result.get('diagnosis_complete') or
                   args.stage=='liveness' and result.get('complete'))
        return 0 if success else 1
    stamp = datetime.now(timezone.utc)
    run_id = 'u4-cse-diagnosis-'+stamp.strftime('%Y%m%dT%H%M%S%f')
    args.out = args.out or RESULTS_DIR/run_id
    args.out.mkdir(parents=True,exist_ok=False)
    source = json.loads(args.reference.read_text())
    args.package = args.out/Path(source['package']).name
    args.weights = args.out/Path(source['weights']).name
    shutil.copytree(source['package'],args.package)
    shutil.copy2(source['weights'],args.weights)
    result = dict(round='U4',section='7.7.1',complete=False,diagnosis_complete=False,
                  program=str(args.program),reference=str(args.reference),llm_calls=0,report_images=0,ckks_calls=0)
    if args.resume:
        previous = json.loads((args.resume/'REPORT.json').read_text())
        if previous['trace']['exit']:
            raise ValueError('Resume requires a completed diagnostic trace')
        result.update(trace=previous['trace'],bytecode_source=str(args.resume/'trace/trace_model.mlirbc'),
                      resumed_from=str(args.resume/'REPORT.json'),trace_reused=True)
    write(args.out/'REPORT.json',result)
    print('RUN='+str(args.out),flush=True)
    started = time.monotonic()
    exit_code = 0
    for stage in (('liveness',) if args.liveness_from else ('native','diagnose')):
        command = [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds','3600','--','-m',
                   'pt_agent.experiments.u4_cse_diagnosis','--inside','--stage',stage]
        for key in ('out','package','weights','program','prepared','reference'):
            command += ['--'+key,str(getattr(args,key))]
        if args.liveness_from:
            command += ['--liveness-from',str(args.liveness_from)]
        write(args.out/(stage+'-argv.json'),command)
        with (args.out/(stage+'.log')).open('w') as log:
            status = subprocess.run(command,cwd=REPO/'DSL/DSL_agent/scripts',stdout=log,stderr=subprocess.STDOUT)
        if status.returncode:
            exit_code = status.returncode
            break
    result = json.loads((args.out/'REPORT.json').read_text())
    rev = subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=REPO,text=True).strip()
    result.update(run_id=run_id,git_rev=rev,exit=exit_code,seconds=time.monotonic()-started)
    write(args.out/'REPORT.json',result)
    from ..registry import append
    os.environ.setdefault('PT_RESULTS',str(RESULTS_DIR))
    append(dict(run_id=run_id,round='U4',model='bert',mode='cse-diagnosis',git_rev=rev,
                metrics=dict(complete=result['complete'],diagnosis_complete=result.get('diagnosis_complete'),llm_calls=0),
                artifacts_dir=str(args.out),ts=stamp.isoformat()))
    print(json.dumps(dict(report=str(args.out/'REPORT.json'),complete=result['complete'],exit=exit_code),indent=2),flush=True)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
