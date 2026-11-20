"""Decoded ideal arithmetic of the saved, optimized original Orbit graph.

Rescale, upscale, modulus switching and bootstrap preserve decoded ideal
values here.  No ciphertext or native backend is constructed.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import sys
import time
from types import SimpleNamespace

from . import r25_orbit_packed_worker as original
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

MODEL = original.MODEL
MODE = 'orbit-optimized-graph-decoded-ideal-three-probe-math'
SLOTS = 65536
CONSTANT_COUNT = 1547
EXPECTED_NODE_COUNT, EXPECTED_EDGE_COUNT = 12343, 15572
OP_COUNTS = dict(input=1, constant=2712, rotate=1382, mul=2920, add=3022,
    negate=76, rescale=790, upscale=782, modswitch=639, bootstrap=19)
ARITY = dict(original.ARITY, rescale=1, upscale=1, modswitch=1, bootstrap=1)
IDENTITIES = ('rescale', 'upscale', 'modswitch', 'bootstrap')
SOURCE_NAMES = ('graph', 'constants', 'probes', 'probe_source', 'r13_result',
                'r22_result', 'r23_result', 'r24_result', 'r25_result')
SCOPE = 'R23 optimized graph on original train:0,train:1,train:2; decoded ideal identities for scale/modulus/bootstrap operations; original mathematical tolerance'


def _np():
    return original._numeric_modules()


def load_program(path):
    graph = original._read_json(path)
    nodes, edges = graph.get('nodes'), graph.get('edges')
    if not isinstance(nodes, list) or len(nodes) < 2 or not isinstance(edges, list):
        raise ValueError('Graph requires ordered nodes and explicit dependencies')
    ids = [row.get('id') for row in nodes]
    if (ids != ['arg0']+[str(i) for i in range(len(nodes)-1)]
            or graph.get('inputs') != ['arg0'] or graph.get('outputs') != [ids[-1]]):
        raise ValueError('The saved graph must retain original ordered SSA identifiers and single I/O')
    positions = {name:i for i, name in enumerate(ids)}
    incoming = {name:[] for name in ids}
    for edge in edges:
        source, target = edge.get('source'), edge.get('target')
        if source not in positions or target not in positions or positions[source] >= positions[target]:
            raise ValueError('Missing, cyclic or unordered graph dependency')
        incoming[target].append(source)
    counts, constant_ids = Counter(), set()
    for index, node in enumerate(nodes):
        kind, attrs = node.get('op'), node.get('op_descr')
        if (kind not in ARITY or not isinstance(attrs, dict)
                or (kind == 'input') != (index == 0)
                or len(incoming[node['id']]) != ARITY[kind]):
            raise ValueError('Unsupported graph operation, attributes or dependency arity')
        incoming[node['id']].sort(key=positions.__getitem__)
        if kind == 'constant':
            identifier = attrs.get('value')
            if type(identifier) is not int or identifier < 0:
                raise ValueError('Constant ID must be a nonnegative integer')
            constant_ids.add(identifier)
        if kind == 'rotate' and type(attrs.get('offset')) is not int:
            raise ValueError('Rotation offset must be an integer')
        for name in (('upFactor',) if kind == 'upscale' else
                     ('downFactor',) if kind == 'modswitch' else
                     ('targetLevel',) if kind == 'bootstrap' else ()):
            if type(attrs.get(name)) is not int or attrs[name] < 0:
                raise ValueError('Original scale/level attributes must be nonnegative integers')
        counts[kind] += 1
    last = dict(positions)
    for index, node in enumerate(nodes):
        for source in incoming[node['id']]:
            last[source] = index
    last[ids[-1]] = len(nodes)
    live, peak = {'arg0'}, 1
    for index, node in enumerate(nodes[1:], 1):
        if any(source not in live for source in incoming[node['id']]):
            raise ValueError('Last-use plan lost an input')
        live.add(node['id']); peak = max(peak, len(live))
        for source in set(incoming[node['id']]):
            if last[source] == index:
                live.remove(source)
        if last[node['id']] == index:
            live.remove(node['id'])
    if live != {ids[-1]}:
        raise ValueError('Only the original output may remain live')
    return dict(nodes=nodes, incoming=incoming, last_use=last, input=ids[0], output=ids[-1],
        metadata=dict(node_count=len(nodes), edge_count=len(edges), op_counts=dict(counts),
            constant_ids=sorted(constant_ids), static_live_peak=peak,
            repeated_operand_edges=len(edges)-len({(e['source'],e['target']) for e in edges})))


def index_constants(path, slots, expected_count):
    """Read count/length headers only; payload values remain unobserved."""
    path = Path(path)
    if type(slots) is not int or slots < 1 or type(expected_count) is not int or expected_count < 1:
        raise ValueError('Positive slot and constant counts are required')
    entries = []
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        def integer():
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('Truncated constant count/length header')
            return struct.unpack('<q', raw)[0]
        count = integer()
        if count != expected_count:
            raise ValueError('Original constant ID count differs')
        for identifier in range(count):
            length = integer(); offset = stream.tell()
            if length <= 0 or length > (size-offset)//8:
                raise ValueError('Constant payload length is invalid or truncated')
            entries.append(dict(id=identifier, offset=offset, raw_length=length,
                prefix_values=min(length, slots), skipped_values=max(0, length-slots)))
            stream.seek(offset+length*8)
        if stream.tell() != size:
            raise ValueError('Unexpected trailing constant bytes')
    return dict(path=str(path.resolve()), file_bytes=size, slots=slots, count=count,
        entries=entries, payload_values_read=0, cached_constant_arrays=0,
        format='little-endian int64 count; int64 length; float64 payload')


@dataclass(frozen=True)
class LazyConstant:
    identifier: int


class LazyConstants:
    """One file, bounded per-operation reads, and no persistent numeric cache."""
    def __init__(self, index):
        self.index = index
        self.stream = None
        self.reads = self.prefix_values_read = self.expanded_bytes = 0
        self.maximum_prefix_values = 0

    def __enter__(self):
        self.stream = Path(self.index['path']).open('rb')
        if os.fstat(self.stream.fileno()).st_size != self.index['file_bytes']:
            self.stream.close(); self.stream = None
            raise ValueError('Original constants file size changed after indexing')
        return self

    def __exit__(self, *exception):
        self.stream.close(); self.stream = None

    def get(self, identifier):
        np = _np()
        if self.stream is None or type(identifier) is not int or not 0 <= identifier < self.index['count']:
            raise ValueError('Open constant stream and a known ID are required')
        row = self.index['entries'][identifier]
        self.stream.seek(row['offset'])
        self.reads += 1
        prefix = np.fromfile(self.stream, dtype='<f8', count=row['prefix_values'])
        self.prefix_values_read += int(prefix.size)
        self.maximum_prefix_values = max(self.maximum_prefix_values, int(prefix.size))
        if prefix.size != row['prefix_values'] or not np.isfinite(prefix).all():
            raise ValueError('Effective constant prefix is truncated or nonfinite')
        slots = self.index['slots']
        if prefix.size == slots:
            value = prefix
        else:
            value = np.empty(slots, dtype=np.float64)
            for start in range(0, slots, prefix.size):
                end = min(start+prefix.size, slots)
                value[start:end] = prefix[:end-start]
        value.flags.writeable = False
        self.expanded_bytes += value.nbytes
        return value

    def summary(self):
        return dict(materialization_attempts=self.reads, prefix_values_read=self.prefix_values_read,
            maximum_prefix_values=self.maximum_prefix_values, expanded_bytes_over_all_reads=self.expanded_bytes,
            persistent_numeric_cache_bytes=0, per_operation_distinct_constant_limit=2,
            skipped_tail_finite=None, policy='Read effective prefix on numeric consumption; repeated IDs share only the current operation read')


def execute_program(program, constants, packed, result, progress=None):
    np = _np(); slots = packed.size
    original._require_vector(packed, slots)
    pool = {program['input']:packed}
    peak_arrays, peak_bytes = 1, packed.nbytes
    for index, node in enumerate(program['nodes'][1:], 1):
        name, kind = node['id'], node['op']
        result.update(active_node=name, active_op=kind)
        result['op_attempts'] += 1; result['op_attempted_by_kind'][kind] += 1
        dependencies = [pool[source] for source in program['incoming'][name]]
        materialized = {}
        def array(value):
            if isinstance(value, LazyConstant):
                if value.identifier not in materialized:
                    materialized[value.identifier] = constants.get(value.identifier)
                return materialized[value.identifier]
            return value
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if kind == 'constant':
                value = LazyConstant(node['op_descr']['value'])
            elif kind in IDENTITIES:
                value = dependencies[0]
                result['ideal_identity_nodes'][kind] += 1
            elif kind == 'add':
                value = array(dependencies[0])+array(dependencies[1])
            elif kind == 'mul':
                value = array(dependencies[0])*array(dependencies[1])
            elif kind == 'negate':
                value = -array(dependencies[0])
            elif kind == 'rotate':
                value = np.roll(array(dependencies[0]), -node['op_descr']['offset'])
            else:
                raise ValueError('Unsupported decoded operation')
        if not isinstance(value, LazyConstant):
            original._require_vector(value, slots)
        pool[name] = value
        numeric = {id(item):item for item in list(pool.values())+list(materialized.values())
                   if not isinstance(item, LazyConstant)}
        peak_arrays = max(peak_arrays, len(numeric))
        peak_bytes = max(peak_bytes, sum(item.nbytes for item in numeric.values()))
        result['op_completed'] += 1; result['op_completed_by_kind'][kind] += 1
        for source in set(program['incoming'][name]):
            if program['last_use'][source] == index:
                del pool[source]
        if program['last_use'][name] == index:
            del pool[name]
        result.update(maximum_live_numeric_arrays=max(result['maximum_live_numeric_arrays'], peak_arrays),
            maximum_live_numeric_bytes=max(result['maximum_live_numeric_bytes'], peak_bytes))
        del dependencies, materialized, numeric, value
        if progress is not None and index % 256 == 0:
            progress()
    if set(pool) != {program['output']}:
        raise ValueError('Last-use execution must retain only its output')
    output = pool[program['output']]
    if isinstance(output, LazyConstant):
        output = constants.get(output.identifier)
    return original._require_vector(output, slots)


def check_sources(args):
    r25 = original._read_json(args.r25_result)
    w25 = r25.get('worker') or {}
    if (r25.get('round') != 'R25' or r25.get('model') != MODEL or r25.get('status') != 'ok'
            or r25.get('mode') != original.MODE or w25.get('source_checked') is not True
            or r25.get('native_packed_equivalence_verified') is not True
            or w25.get('native_packed_equivalence_verified') is not True
            or any(w25.get(key) != 3 for key in ('packed_forward_attempts','packed_forward_completed',
                'math_check_attempts','math_check_completed'))
            or w25.get('program_metadata',{}).get('constant_ids') != list(range(CONSTANT_COUNT))):
        raise ValueError('Original R25 three-probe arithmetic and all constant IDs must pass')
    paths = (w25.get('source_summary') or {}).get('paths') or {}
    for name in ('constants','probes','probe_source','r13_result','r22_result','r24_result'):
        if not original._same_path((paths.get(name) or {}).get('path'), getattr(args, name)):
            raise ValueError('Explicit source differs from accepted original R25: '+name)
    old_args = SimpleNamespace(**dict(vars(args), graph=Path(paths['graph']['path'])))
    original_summary, r13, r22, r24 = original._check_sources(old_args)
    r23 = original._read_json(args.r23_result)
    valid = (r23.get('phase_results') or {}).get('validate') or {}
    if (r23.get('round') != 'R23' or r23.get('model') != MODEL or r23.get('status') != 'ok'
            or r23.get('mode') != 'orbit-resnet20-native-optimizer-preflight'
            or r23.get('native_optimizer_completed') is not True
            or r23.get('placement_artifact_checked') is not True
            or r23.get('placement_bootstrap_nodes') != 19
            or valid.get('status') != 'ok' or valid.get('graph_check',{}).get('passed') is not True
            or not original._same_path(valid.get('tdag'), args.graph)
            or not original._same_path(r23.get('source_records',{}).get('r22_result'), args.r22_result)
            or Path(args.graph).resolve() != Path(args.r23_result).resolve().parent/'work/validate-worker/tdag.json'):
        raise ValueError('Original R23 optimizer, placement or optimized graph identity differs')
    return dict(paths={name:original._file_record(getattr(args,name)) for name in SOURCE_NAMES},
        original_source=original_summary, reference_source='R24 worker.approximate_probes[].approximate',
        original_source_times_included=False), r13, r24


def initial_result():
    result = dict(round='R33', model=MODEL, mode=MODE, status='running', scope=SCOPE,
        source_checked=False, optimized_decoded_ideal_math_passed=False, probe_rows=[],
        op_attempts=0, op_completed=0, op_attempted_by_kind={k:0 for k in ARITY if k!='input'},
        op_completed_by_kind={k:0 for k in ARITY if k!='input'}, ideal_identity_nodes={k:0 for k in IDENTITIES},
        maximum_live_numeric_arrays=0, maximum_live_numeric_bytes=0,
        graph_load_attempts=0, graph_load_completed=0, constant_index_attempts=0, constant_index_completed=0,
        probe_load_attempts=0, probe_load_completed=0, packed_forward_attempts=0, packed_forward_completed=0,
        math_check_attempts=0, math_check_completed=0, backend_ready=False, security_validated=False,
        accuracy_validated=False, formal_matrix_complete=False, native_bootstrap_executions=0,
        keygen_runs=0, ckks_executions=0, intermediate_decryptions=0, llm_calls=0, api_calls=0,
        new_downloads=0, new_installations=0, torch_model_builds=0, torch_model_forwards=0,
        checkpoint_reads=0, heldout_samples=0, confirm_samples=0, report_samples=0,
        compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None)
    return result


def run_checks(args, recorder, result):
    with recorder.span('source_records'):
        summary, r13, r24 = check_sources(args)
        result.update(source_checked=True, source_summary=summary)
    with recorder.span('graph_load'):
        result['graph_load_attempts'] += 1
        program = load_program(args.graph)
        if (program['metadata']['node_count'] != EXPECTED_NODE_COUNT or program['metadata']['edge_count'] != EXPECTED_EDGE_COUNT
                or program['metadata']['op_counts'] != OP_COUNTS
                or program['metadata']['constant_ids'] != list(range(CONSTANT_COUNT))):
            raise ValueError('The exact original optimized graph inventory differs')
        result['program_metadata'] = program['metadata']; result['graph_load_completed'] += 1
    with recorder.span('constant_index'):
        result['constant_index_attempts'] += 1
        indexed = index_constants(args.constants, SLOTS, CONSTANT_COUNT)
        write_json(args.work_dir/'constant-index.json', indexed)
        result['constant_index_metadata'] = {key:value for key,value in indexed.items() if key!='entries'}
        result['constant_index_completed'] += 1
    with recorder.span('original_saved_training_probes'):
        result['probe_load_attempts'] += 1
        images, labels, source = original._load_probes(args, r13, result)
        references = original._cached_references(r24)
        result['probe_source'] = source; result['probe_load_completed'] += 1
    with LazyConstants(indexed) as constants:
        for index in range(3):
            result.update(active_probe_index=index, active_sample_id=source['sample_ids'][index])
            with recorder.span('optimized_decoded_forward', probe_index=index, sample_id=source['sample_ids'][index]):
                result['packed_forward_attempts'] += 1
                write_json(args.work_dir/'worker-result.json', result)
                packed = original.pack_input(images[index])
                output = execute_program(program, constants, packed, result,
                    progress=lambda:write_json(args.work_dir/'worker-result.json',result))
                decoded = original.decode_output(output)
                result['packed_forward_completed'] += 1
            with recorder.span('original_strict_math', probe_index=index):
                result['math_check_attempts'] += 1
                compared = original._check_math(references[index], decoded)
                result['probe_rows'].append(dict(probe_index=index, sample_id=source['sample_ids'][index],
                    label=int(labels[index]), decoded=decoded.tolist(), math=compared))
                result['math_check_completed'] += 1
                result['constant_reads'] = constants.summary()
                write_json(args.work_dir/'worker-result.json', result)
                if not compared['passed']:
                    raise ValueError('Optimized decoded arithmetic fails the original strict mathematical criterion')
        result['constant_reads'] = constants.summary()
    if (result['op_attempts'] != 3*(EXPECTED_NODE_COUNT-1) or result['op_completed'] != 3*(EXPECTED_NODE_COUNT-1)
            or result['op_completed_by_kind'] != {key:OP_COUNTS.get(key,0)*3 for key in ARITY if key!='input'}):
        raise ValueError('Every original optimized operation must execute once on each of three probes')
    result.update(status='ok', optimized_decoded_ideal_math_passed=True)


def worker(args):
    args.work_dir.mkdir(parents=True, exist_ok=False)
    result = initial_result(); recorder = StageRecorder(args.work_dir/'events.jsonl')
    started = time.perf_counter_ns()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    try:
        with recorder.span('r33_optimized_decoded_ideal_math'):
            run_checks(args, recorder, result)
    except BaseException as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc)[:3000])
    result.update(elapsed_ns=time.perf_counter_ns()-started,
        worker_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(args.work_dir/'worker-result.json', result)
    return result


def execute(args):
    from .. import paths
    out = args.out.resolve()
    if out.exists() or out.parent != paths.RESULTS_DIR.resolve():
        raise ValueError('Use a fresh isolated case directly under the configured results directory')
    if shutil.disk_usage(out.parent).free < 10*1024**3:
        raise RuntimeError('Preserve10GiB free disk before this diagnostic')
    out.mkdir(); shutil.copyfile(__file__, out/'driver.py')
    command = [str(args.numpy_python),'-B','-m','pt_agent.experiments.r33_orbit_optimized_math',
        '--phase','worker','--work-dir',str(out/'worker')]
    for name in SOURCE_NAMES:
        command += ['--'+name.replace('_','-'), str(getattr(args,name).resolve())]
    write_json(out/'command.json', command)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    started = time.perf_counter_ns()
    observation = run_process(command,cwd=Path(__file__).resolve().parents[2],log=out/'worker.log',
        resources=out/'resources.jsonl',timeout=600,mem_gib=.5,fsize_gib=1,env=env)
    path = out/'worker/worker-result.json'; result = json.loads(path.read_text()) if path.is_file() else None
    status = 'ok' if (observation['exit_code']==0 and not observation['timed_out']
        and observation['signal'] is None and not observation['remaining_pids']
        and result and result.get('optimized_decoded_ideal_math_passed') is True) else 'error'
    record = dict(run_id=out.name,round='R33',model=MODEL,mode=MODE,status=status,worker=result,
        process=observation,scope=SCOPE,timing=summarize_events(read_events(out/'worker/events.jsonl')),
        supervised_elapsed_ns=time.perf_counter_ns()-started,artifact_bytes=0)
    record['metrics'] = dict(status=status, optimized_decoded_ideal_math_passed=(result or {}).get('optimized_decoded_ideal_math_passed'),
        packed_forward_completed=(result or {}).get('packed_forward_completed'),
        ideal_bootstrap_nodes_evaluated=(result or {}).get('ideal_identity_nodes',{}).get('bootstrap'),
        real_ckks_executions=0,keygen_runs=0,llm_calls=0,compiled_bootstraps=None,predicted_latency=None,
        CKKS_latency=None,accuracy_acceptance=None)
    for _ in range(20):
        payload = json.dumps(record,ensure_ascii=False,indent=2)+'\n'
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and p!=out/'result.json')+len(payload.encode())
        if (size > 4*1024**3 or shutil.disk_usage(out).free < 10*1024**3) and record['status']!='error':
            record.update(status='error',error='Terminal artifact or free-disk allowance exceeded')
            record['metrics']['status']='error'
            continue
        if size == record['artifact_bytes']:
            (out/'result.json').write_text(payload); break
        record['artifact_bytes'] = size
    else:
        raise RuntimeError('Terminal artifact size did not stabilize')
    from ..registry import git_rev
    row = dict(run_id=out.name,round='R33',model=MODEL,mode=MODE,git_rev=git_rev(),metrics=record['metrics'],
        artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
    with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('run','worker'),required=True)
    for name in SOURCE_NAMES:
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--out',type=Path);parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--numpy-python',type=Path)
    args = parser.parse_args(argv)
    if args.phase=='run' and (args.out is None or args.numpy_python is None):
        parser.error('Outer run requires explicit --out and --numpy-python')
    if args.phase=='worker' and args.work_dir is None:
        parser.error('Pure worker requires explicit --work-dir')
    record = execute(args) if args.phase=='run' else worker(args)
    print(json.dumps(dict(status=record['status'],phase=args.phase)))
    return 0 if record['status']=='ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
