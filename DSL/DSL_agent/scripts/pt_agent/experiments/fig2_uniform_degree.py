"""Bounded per-model Chebyshev31/63 comparison on saved literature CNNs.

The outer owns source checks, supervision and registry. No API, exporter,
compiler, cryptography, confirmation or report path is used.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .r31_lattigo_runtime_plan import physical_memory_observation

ROUND = 'FIG2-UNIFORM-DEGREE'
MODE = 'fixed-domain-cheb31-63-depth-dp-heldout'
MODULE = 'pt_agent.experiments.fig2_uniform_degree_worker'
DEGREES = (31, 63)
SECONDS = 1800
NIX_SECONDS = 1830
OUTER_SECONDS = 1860
GIB = 1024**3
CASE_BYTES = 4*GIB
FREE_BYTES = 10*GIB
RESERVE_BYTES = GIB//1024
MODEL_RUNS = {'resnet20_silu': ('resnet20-silu', 'r17-margin-heldout-20261115', 19),
    'vgg16_silu_avgpool': ('vgg16-silu-avgpool', 'r17a-margin-heldout-20261115', 15),
    'alexnet_silu_avgpool': ('alexnet-silu-avgpool', 'r17a-margin-heldout-20261115', 7)}
ZERO_FIELDS = ('api_calls', 'agent_runs', 'tuner_runs', 'model_intakes', 'original_graph_forwards',
    'training_image_reads', 'emit_runs', 'trace_runs', 'compiler_runs', 'keygen_runs',
    'CKKS_runs', 'confirm_samples', 'report_samples', 'freeze_attempts', 'downloads',
    'installations', 'extra_integrity_checks', 'new_operator_identifier_digests')
PAIRS = ('count', 'plan', 'replay', 'tensor', 'mock', 'math')


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Object source JSON required')
    return value


def load_metadata_template(path):
    """The saved compiler recipe is an argv list, separate from object records.

    The original pure builder parses its recorded identifiers and substitutions;
    no compiler, operator digest computation or content verification runs here.
    """
    from .r29_local_compile_matrix import compiler_argv
    template=json.loads(Path(path).read_text())
    compiler_argv(template,'pars',binary='/metadata-only/hecate-opt',source='/metadata-only/input.mlir',
        profile='/metadata-only/compiler-config.json',output='/metadata-only/lowered.mlir')
    return template


def _same(left, right):
    return isinstance(left, (str, Path)) and Path(left).resolve() == Path(right).resolve()


def _tree(text):
    tree = ast.parse(text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Saved fixed program must contain one program definition')
    return tree


def _acts(tree):
    return sorted((node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'act'), key=lambda x:x.lineno)


def degree_variant(text, degree):
    """Alter only act deg literals; no source program is executed."""
    if type(degree) is not int or degree not in DEGREES:
        raise ValueError('Only the two predeclared degrees are accepted')
    original = _tree(text)
    changed = copy.deepcopy(original)
    for node in _acts(changed):
        attrs = {key.arg: key for key in node.keywords}
        if (ast.literal_eval(attrs['fn'].value) != 'silu'
                or ast.literal_eval(attrs['method'].value) != 'cheb'
                or ast.literal_eval(attrs['deg'].value) != 63):
            raise ValueError('Original SiLU Chebyshev63 program required')
        attrs['deg'].value = ast.copy_location(ast.Constant(degree), attrs['deg'].value)
    if not _acts(changed):
        raise ValueError('The original model must have activations')
    result = ast.unparse(ast.fix_missing_locations(changed))+'\n'
    check_variant(text, result, degree)
    return result


def check_variant(original_text, candidate_text, degree):
    original, candidate = _tree(original_text), _tree(candidate_text)
    left, right = _acts(original), _acts(candidate)
    if len(left) != len(right) or not left:
        raise ValueError('Activation inventory differs')
    for before, after in zip(left, right):
        a = {key.arg:key for key in before.keywords}
        b = {key.arg:key for key in after.keywords}
        if (ast.literal_eval(a['deg'].value) != 63 or ast.literal_eval(b['deg'].value) != degree):
            raise ValueError('Declared degree differs')
        b['deg'].value = copy.deepcopy(a['deg'].value)
    if ast.dump(original, include_attributes=False) != ast.dump(candidate, include_attributes=False):
        raise ValueError('Only the activation degree may differ')
    return dict(passed=True, only_degree_changed=True, activation_count=len(left), degree=degree)


def selected_sources(model):
    short, group, _ = MODEL_RUNS[model]
    rule = paths.RESULTS_DIR/group/model/'result.json'
    saved = read(rule)['worker']['source_preparation']
    info = paths.RESULTS_DIR/('r13-common-'+short+'-20261115')/'worker/model_info/model_info.json'
    record = read(info)
    return dict(rule_result=rule, r14_result=Path(saved['r14_result']), r13_result=Path(saved['r13_result']),
        model_info=info, calibration=Path(saved['calibration']),
        calibration_source=Path(saved['calibration_source']),
        base_program=Path(read(rule)['worker']['artifacts']['program']),
        cached_outputs=Path(read(rule)['worker']['artifacts']['heldout_outputs']),
        records=Path(record['source']['calibration_source']['records']),
        metadata_template=paths.RESULTS_DIR/'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')


def source_gate(model, source):
    """Small JSON/program checks only; NPZ/image/checkpoint payloads stay closed."""
    factory = 'pt_agent.experiments.r13_common_models:'+model
    rule, info = read(source['rule_result']), read(source['model_info'])
    w = rule.get('worker') or {}
    expected = dict(deg=63, margin=1.5, slots=16384, placement='depth-dp', refresh='compiler', act_method='cheb')
    if (rule.get('status') != 'ok' or rule.get('round') not in ('R17','R17a')
            or rule.get('model') != factory or w.get('model') != factory or w.get('status') != 'ok'
            or w.get('rule_options') != expected or w.get('completed_samples') != 1000
            or any(w.get(key) is not True for key in ('source_checked','calibration_reused',
                'math_probes_passed','evaluation_completed','task_passed','domain_passed','all_finite','heldout_acceptance'))):
        raise ValueError('Accepted original fixed63 heldout source required')
    if (info.get('source',{}).get('factory') != factory or w.get('model_source') != info['source']
            or info.get('task') != 'classification' or info.get('metric') != 'classification_top1'
            or info.get('threshold') != {'acc_drop_pp':.5}):
        raise ValueError('Original model, normalization or quality policy differs')
    preparation = w['source_preparation']
    for key in ('r14_result','r13_result','calibration','calibration_source'):
        if not _same(preparation.get(key), source[key]):
            raise ValueError('Cached source path differs: '+key)
    if (not _same(preparation.get('model_info'), Path(source['model_info']).parent)
            or not _same(w['artifacts'].get('program'), source['base_program'])
            or not _same(w['artifacts'].get('heldout_outputs'), source['cached_outputs'])
            or not Path(source['cached_outputs']).resolve().is_relative_to(Path(source['rule_result']).resolve().parent/'worker')):
        raise ValueError('Original program/reference cache path differs')
    calibration, provenance = read(source['calibration']), read(source['calibration_source'])
    if (provenance.get('split') != 'train' or provenance.get('count') != 1000
            or len(provenance.get('indices',[])) != 1000 or len(set(provenance['indices'])) != 1000
            or provenance.get('normalization') != info['source']['normalization']
            or not _same(info['source']['calibration_source']['records'], source['records'])):
        raise ValueError('Cached1000-training calibration source differs')
    graph = read(Path(source['model_info']).parent/'graph/graph.json')
    nodes = [node for node in graph['nodes'] if node[1] == 'act']
    text = Path(source['base_program']).read_text()
    tree = _tree(text)
    decorators = tree.body[0].decorator_list
    if (len(decorators) != 1 or not isinstance(decorators[0],ast.Call)
            or not isinstance(decorators[0].func,ast.Attribute)
            or not isinstance(decorators[0].func.value,ast.Name)
            or decorators[0].func.value.id != 'fhe' or decorators[0].func.attr != 'program'
            or decorators[0].args
            or {item.arg:ast.literal_eval(item.value) for item in decorators[0].keywords}
            != dict(slots=16384,refresh='compiler',placement='depth-dp')):
        raise ValueError('Fixed compiler/depth-DP header differs')
    acts = _acts(tree)
    if len(nodes) != len(acts) or len(acts) != MODEL_RUNS[model][2]:
        raise ValueError('Original activation count differs')
    for node, call in zip(nodes, acts):
        attrs = {kw.arg:ast.literal_eval(kw.value) for kw in call.keywords}
        row = calibration[node[2][0]]
        domain = (float(row['min'])*1.5, float(row['max'])*1.5)
        if attrs != dict(fn='silu', method='cheb', deg=63, domain=domain):
            raise ValueError('Original fixed margin1.5 domain differs')
    sample = w.get('sample_source') or {}
    if (sample.get('indices') != list(range(9000,10000)) or sample.get('count') != 1000
            or sample.get('normalization') != info['source']['normalization']):
        raise ValueError('Original heldout reference split differs')
    for key, path in source.items():
        if key == 'records':
            if not (Path(path)/'test-09000.records').is_file():
                raise FileNotFoundError('Existing heldout shard missing')
        elif not Path(path).is_file():
            raise FileNotFoundError('Missing explicit source: '+key)
    return dict(model=model, factory=factory, paths={k:str(Path(v).resolve()) for k,v in source.items()},
        normalization=info['source']['normalization'], calibration_samples=1000, calibration_reused=True,
        heldout_indices=[9000,9999], heldout_samples=1000, original_training_images_reread=False,
        original_checkpoint=info['source']['checkpoint'], historical_count=w['count_stats'],
        historical_quality={k:w.get(k) for k in ('heldout_acceptance','validation_metrics','max_u','candidate_frozen')})


def validate_worker(record, model):
    if (record.get('round') != ROUND or record.get('mode') != MODE or record.get('model') != model
            or record.get('status') != 'ok' or record.get('source_checked') is not True
            or record.get('metadata_template_checked') is not True
            or record.get('candidate_frozen') is not False
            or any(record.get(key) != 0 for key in ZERO_FIELDS)):
        raise ValueError('Worker identity/source/excluded-action contract differs')
    for key in ('compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance','security_acceptance'):
        if record.get(key) is not None:
            raise ValueError('Unmeasured formal result was populated')
    if any(record.get(name+'_'+boundary) != 1
           for name in ('model_restore','heldout_load','reference_cache_load')
           for boundary in ('attempts','completed')):
        raise ValueError('Source/model/input load counts differ')
    rows = record.get('variants') or []
    if [row.get('degree') for row in rows] != list(DEGREES):
        raise ValueError('Both fixed variants must be completed in order')
    for row in rows:
        if (row.get('status') not in ('accepted','quality_rejected')
                or row.get('completed_samples') != 1000 or row.get('math_passed') is not True
                or row.get('all_finite') is not True or row.get('ast_check',{}).get('only_degree_changed') is not True):
            raise ValueError('Variant evaluation or strict math incomplete')
        for name, expected in (('count',1),('plan',1),('replay',1),('tensor',1000),('mock',3),('math',3)):
            if row.get(name+'_attempts') != expected or row.get(name+'_completed') != expected:
                raise ValueError('Actual variant operation count differs: '+name)
        if row.get('planned_bootstraps') != row.get('plan_replay',{}).get('inner_boots'):
            raise ValueError('Count replay bootstrap count differs')
        metrics = row.get('validation_metrics') or {}
        drop = metrics.get('acc_drop_pp')
        domains = row.get('domain_rows') or []
        if (type(drop) not in (int,float) or not math.isfinite(drop)
                or len(domains) != MODEL_RUNS[model][2]
                or any(type(item.get('max_u')) not in (int,float)
                       or not math.isfinite(item['max_u']) or item.get('count') != 1000 for item in domains)
                or row.get('max_u') != max(item['max_u'] for item in domains)):
            raise ValueError('Finite original quality/coverage evidence is missing')
        task = drop <= .5
        domain = row['max_u'] <= .90
        if row.get('task_passed') is not task or row.get('domain_passed') is not domain:
            raise ValueError('Original0.5pp/max_u0.90 quality calculation differs')
        accepted = task and domain
        if (row.get('heldout_acceptance') is not accepted or (row['status']=='accepted') is not accepted
                or metrics.get('total') != 1000
                or row.get('domain_row_count') != MODEL_RUNS[model][2]):
            raise ValueError('Quality acceptance and original policy differ')
    return True


def worker_command(source, model, work):
    command = [sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(NIX_SECONDS),
        '--python',str(paths.INTAKE_PYTHON),'--','-B','-m',MODULE,'--model',model,'--work-dir',str(work)]
    for key, path in source.items():
        command += ['--'+key.replace('_','-'),str(path)]
    return command


def _terminal(out, result):
    target = out/'result.json'
    for _ in range(12):
        write_json(target,result)
        total = artifact_bytes(out)
        if result.get('artifact_bytes') == total:
            if total+RESERVE_BYTES > CASE_BYTES:
                raise RuntimeError('Terminal case plus completion reserve exceeds4GiB')
            return
        result['artifact_bytes'] = total
    raise RuntimeError('Terminal metadata size did not stabilize')


def register(out, result, registry):
    metrics = {key:result.get(key) for key in ('status','model','worker','guard_trigger','process')}
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('Run registry identity already exists')
        stream.seek(0,os.SEEK_END)
        stream.write(json.dumps(dict(run_id=out.name,round=ROUND,model=result['model'],mode=MODE,
            git_rev=None,metrics=metrics,artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat()),allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def execute(args):
    out = args.out.resolve()
    if out.exists() or out == paths.RESULTS_DIR.resolve() or not out.is_relative_to(paths.RESULTS_DIR.resolve()):
        raise ValueError('New isolated output under the results root required')
    started = time.perf_counter_ns();out.mkdir(parents=True);work=out/'worker';work.mkdir()
    recorder=StageRecorder(out/'events.jsonl');stop=threading.Event();monitor_errors=[]
    result=dict(round=ROUND,mode=MODE,model=args.model,status='running',error=None,worker=None,
        process=None,guard_trigger=None,incoming_signal=None,source_checked=False,
        artifact_bytes=None,artifact_bytes_scope='Persisted case before completion marker; marker separately records final case bytes',
        completion_reserve_bytes=RESERVE_BYTES,
        limits=dict(worker_seconds=SECONDS,Nix_seconds=NIX_SECONDS,outer_seconds=OUTER_SECONDS,
            per_process_AS_gib=12,subtree_RSS_gib=14,minimum_host_available_gib=1.25,
            case_gib=4,minimum_disk_free_gib=10,concurrency=1),**dict.fromkeys(ZERO_FIELDS,0))
    result.update(compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None,security_acceptance=None)
    resource_path=out/'worker-resources.jsonl'
    def check():
        observation=physical_memory_observation(_latest_rss_sample(resource_path),_available_ram_bytes(),
            rss_gib=14,minimum_available_gib=1.25)
        if observation['stop']:
            result['guard_trigger']=dict(kind='physical_memory',**observation)
            raise RuntimeError('; '.join(observation['reasons']))
        if artifact_bytes(out)+RESERVE_BYTES > CASE_BYTES or shutil.disk_usage(out).free < FREE_BYTES:
            result['guard_trigger']=dict(kind='artifact_or_disk',artifact_bytes=artifact_bytes(out),disk_free_bytes=shutil.disk_usage(out).free)
            raise RuntimeError('Finite artifact/disk allowance exhausted')
    def monitor():
        while not stop.wait(.25):
            try:check()
            except BaseException as exc:
                monitor_errors.append(type(exc).__name__+': '+str(exc))
                try:write_json(out/'guard.json',dict(trigger=result['guard_trigger'],error=monitor_errors[-1]))
                finally:os.kill(os.getpid(),signal.SIGTERM)
                return
    def interrupted(signum,frame):
        result['incoming_signal']=signum
        raise InterruptedError('Outer received signal'+str(signum))
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in old:signal.signal(sig,interrupted)
    watcher=threading.Thread(target=monitor,daemon=True)
    lock=None
    try:
        lock=(paths.RESULTS_DIR/'.fig2-uniform-degree.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with recorder.span('uniform_degree_case'):
            with recorder.span('source_gate'):
                check();source=selected_sources(args.model);result['source']=source_gate(args.model,source)
                result['source_checked']=True
                snapshots=[]
                for path in (Path(__file__),Path(__file__).with_name('fig2_uniform_degree_worker.py'),
                             Path(__file__).with_name('r17_margin_validation_worker.py'),
                             Path(__file__).with_name('r15_rule_validation_worker.py')):
                    target=out/(path.name+'.source');shutil.copyfile(path,target)
                    snapshots.append(dict(source=str(path),snapshot=str(target),bytes=target.stat().st_size))
                result['source_snapshots']=snapshots
            watcher.start()
            with recorder.span('uniform_degree_worker'):
                command=worker_command(source,args.model,work);result['argv']=command
                env=dict(os.environ,PYTHONPATH=str(paths.REPO/'DSL/DSL_agent/scripts'),PYTHONDONTWRITEBYTECODE='1')
                for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
                result['process']=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/'worker.log',
                    resources=resource_path,timeout=OUTER_SECONDS,mem_gib=12,fsize_gib=8,env=env)
                path=work/'worker_result.json'
                if path.is_file():result['worker']=read(path)
                observation=result['process']
                if observation.get('signal') is not None:result['incoming_signal']=observation['signal']
                if (observation.get('exit_code')!=0 or observation.get('timed_out') is not False
                        or observation.get('remaining_pids')!=[] or observation.get('signals_sent')!=[]
                        or observation.get('signal') is not None):
                    raise RuntimeError('Worker did not complete cleanly')
                check();validate_worker(result['worker'],args.model)
                result['status']='ok'
    except BaseException as exc:
        result.update(status='resource_stopped' if result['guard_trigger'] else 'interrupted' if result['incoming_signal'] else 'error',
                      error=type(exc).__name__+': '+str(exc)[:2500])
    finally:
        stop.set()
        if watcher.is_alive():watcher.join(timeout=2)
        for sig,handler in old.items():signal.signal(sig,handler)
        if lock is not None:lock.close()
        partial=work/'worker_result.json'
        if partial.is_file():result['worker']=read(partial)
        if monitor_errors:result.update(status='resource_stopped',error=monitor_errors[0])
        result['timing']=summarize_events(read_events(out/'events.jsonl'),time.perf_counter_ns())
        try:_terminal(out,result)
        except BaseException as exc:
            result.update(status='artifact_budget_exceeded',error=type(exc).__name__+': '+str(exc))
            write_json(out/'result.json',result)
        register(out,result,paths.RESULTS_DIR/'registry.jsonl')
        marker=persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        if marker['case_artifact_bytes']>CASE_BYTES or (out/'completion.json').stat().st_size>RESERVE_BYTES:
            raise RuntimeError('Terminal completion exceeded its reserved allowance')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=tuple(MODEL_RUNS),required=True)
    parser.add_argument('--out',type=Path,required=True)
    result=execute(parser.parse_args(argv))
    print(json.dumps(dict(status=result['status'],model=result['model'],error=result['error'])))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
