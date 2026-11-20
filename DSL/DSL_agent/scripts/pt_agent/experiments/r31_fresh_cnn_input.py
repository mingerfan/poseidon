"""Prepare train:0 input and a new frozen fresh-R30 Tensor reference.

The outer process checks the accepted source and owns the registry. The worker
only receives concrete paths and performs one frozen-program Tensor forward.
Native execution, report access and feedback to the Agent are absent.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time

from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json

ROUND = 'R31-FRESH-INPUT'
MODE = 'fresh-frozen-cnn-native-reference-preparation'
MODEL = 'resnet20_silu'
MODULE = 'pt_agent.experiments.r31_fresh_cnn_input'
OVERALL_SECONDS = 600
WORKER_SECONDS = 480
NIX_SECONDS = 540
MEMORY_GIB = 4
CASE_BYTES = 4*1024**3
FREE_BYTES = 10*1024**3
TERMINAL_RESERVE_BYTES = 64*1024
CRITERION = '1e-12+1e-12*abs(reference)'
ZERO_FIELDS = ('llm_calls', 'agent_runs', 'tuner_runs', 'count_runs', 'emit_runs', 'trace_runs',
    'compiler_runs', 'keygen_runs', 'encryption_runs', 'native_bootstrap_runs', 'decryption_runs',
    'original_model_forwards', 'heldout_samples', 'confirm_samples', 'report_samples',
    'downloads', 'installations', 'extra_integrity_checks')
COUNTERS = ('model_info_load_attempts', 'model_info_load_completed', 'probe_load_attempts',
    'probe_load_completed', 'packing_attempts', 'packing_completed', 'tensor_attempts', 'tensor_completed')


def read(path):
    return json.loads(Path(path).read_text())


def check_plan(plan, expected_bootstraps):
    """Check SSA and layout-independent source metadata without bundle reads."""
    if (plan.get('format_version') != 1 or plan.get('finalization') != []
            or len(plan.get('external_inputs', [])) != 1 or len(plan.get('final_outputs', [])) != 1):
        raise ValueError('A format1 single-input/output source RuntimePlan is required')
    values = {row['id']: row for row in plan['values']}
    if len(values) != len(plan['values']):
        raise ValueError('RuntimePlan value identifiers must be unique')
    defined = set(plan['external_inputs']); ordinals = set()
    for row in plan['initialization']+plan['execution']:
        ordinal = row['ordinal']
        if (type(ordinal) is not int or ordinal in ordinals or row['output'] in defined
                or row['output'] not in values or any(x not in defined for x in row.get('inputs', []))):
            raise ValueError('RuntimePlan must retain topological SSA instructions')
        ordinals.add(ordinal); defined.add(row['output'])
    if set(values) != defined or any(x not in defined for x in plan['final_outputs']):
        raise ValueError('RuntimePlan contains unproduced values')
    counts = dict(Counter(x['op'] for x in plan['execution']))
    if type(expected_bootstraps) is not int or counts.get('boot', 0) != expected_bootstraps:
        raise ValueError('Compiled bootstrap count differs from the accepted source')
    return dict(value_count=len(values), execution_count=len(plan['execution']),
        encode_count=len(plan['initialization']), op_counts=counts, bootstrap_count=expected_bootstraps,
        external_inputs=plan['external_inputs'], final_outputs=plan['final_outputs'],
        ssa_verified=True, values_renamed=0, instructions_reordered=0,
        bundle_contents_read=False, symbolic_scales_modified=False)


def _metadata_gate(gate, plan_path):
    info = read(gate['model_info']); provenance = read(gate['probe_source'])
    source = info.get('source') or {}
    if (info.get('task') != 'classification' or info.get('metric') != 'classification_top1'
            or info.get('threshold') != {'acc_drop_pp': .5}
            or source.get('factory') != gate['factory'] or source.get('normalization') != gate['normalization']
            or provenance.get('split') != 'train' or provenance.get('indices') != [0,1,2]
            or provenance.get('sample_ids') != ['train:0','train:1','train:2']
            or provenance.get('normalization') != source['normalization']
            or provenance != source.get('calibration_source')):
        raise ValueError('Current intake training provenance, normalization or model policy differs')
    for field, key in (('original_source','source'), ('checkpoint','checkpoint'),
                       ('inference_preprocessing_source','inference_preprocessing_source')):
        if Path(source[field]).resolve() != Path(gate['original_sources'][key]).resolve():
            raise ValueError('Current original factory/checkpoint/preprocessing identity differs')
    if (len(info['inputs']) != 1 or info['inputs'][0]['shape'] != [1,3,32,32]
            or info['inputs'][0]['dtype'] != 'float64'):
        raise ValueError('Current intake must retain the original float64 CIFAR input')
    result = read(Path(gate['source_case'])/'result.json')
    if result.get('round')!='R30' or result.get('status')!='ok' or result.get('model')!=MODEL:
        raise ValueError('Current source result no longer identifies the accepted fresh model')
    compiled = result['phases']['compile']
    if Path(compiled['runtime_plan']).resolve() != Path(plan_path).resolve():
        raise ValueError('RuntimePlan path differs from the accepted fresh compilation')
    inventory = check_plan(read(plan_path), compiled['compiled_bootstraps'])
    return dict(model_info=info, probe_source=provenance, plan=inventory)


def prepare_source(source_case, model=MODEL, layout_proof=None):
    from .r30_frozen_cnn_report import source_gate
    if model != MODEL:
        raise ValueError('This explicit entry prepares the current ResNet20 source')
    gate = source_gate(source_case, model)
    plan = Path(source_case).resolve()/'post-freeze/emit/lowered._hecate_golden.runtime-plan.json'
    metadata = _metadata_gate(gate, plan)
    return dict(gate=gate, runtime_plan=str(plan), source_metadata=metadata,
        layout_proof=None if layout_proof is None else str(Path(layout_proof).resolve()),
        sample_id='train:0', sample_index=0, source_split='train',
        expected_reference_policy='New current frozen-program Tensor output; no previous expected.json values',
        normalization_reapplied=False, feedback_to_agent=False)


def _initial_result():
    value = dict(round=ROUND, mode=MODE, model=MODEL, status='running', error=None,
        reference_prepared=False, compiled_plan_mathematical_equivalence_verified=None,
        old_expected_values_read=False, previous_input_reused=False, normalization_reapplied=False,
        training_samples_consumed=0, training_npz_rows_loaded=0, source_checked=False,
        accuracy_acceptance=None, security_acceptance=None, CKKS_latency=None,
        feedback_to_agent=False, tensor_reference_is_quality_measurement=False)
    value.update({key:0 for key in ZERO_FIELDS+COUNTERS})
    return value


def _numeric_modules():
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    return np


def load_training_probe(probes, provenance):
    import numpy as np
    with np.load(probes, allow_pickle=False) as archive:
        images = archive['input']; indices = archive['indices']
    if (images.dtype != np.float64 or images.shape != (3,3,32,32)
            or indices.tolist() != [0,1,2] or not np.isfinite(images).all()
            or provenance['sample_ids'] != ['train:0','train:1','train:2']):
        raise ValueError('Current saved three training probes changed dtype, shape, identity or finite values')
    return images[0:1].copy(), dict(npz_rows_loaded=3, reference_rows=1, sample_id='train:0',
        label=provenance.get('labels',[None])[0], normalization_reapplied=False)


def output_layout(program, checked, proof=None):
    """Derive the current final flat-linear layout, optionally check old metadata."""
    import numpy as np
    from ..kernels.layout import FlatLayout, layout_record
    from ..lower import decode_outputs
    if len(program.outputs) != 1:
        raise ValueError('The frozen classifier must retain one output')
    name = program.outputs[0]; producer = program.producer(name)
    if (producer is None or producer.kind != 'linear' or checked.layouts.get(name) != 'flat'
            or checked.shapes.get(name) != (1,10) or program.header.slots != 16384):
        raise ValueError('Current output must be the supported flat ten-class linear head')
    layout = FlatLayout(program.header.slots, 10)
    indices = decode_outputs([np.arange(program.header.slots,dtype=np.float64)[None,:]], [layout])[0].reshape(-1)
    if not np.array_equal(indices, indices.astype(np.int64)):
        raise ValueError('Classifier decoder must select integer scalar slots')
    slots = indices.astype(np.int64).tolist(); record = layout_record(layout)
    crosscheck = None
    if proof is not None:
        if proof.get('status') != 'ok' or proof.get('mathematical_equivalence_verified') is not True:
            raise ValueError('Optional historical output-layout proof must be accepted')
        rows = proof.get('samples') or []
        if (not rows or any(row.get('status') != 'ok' or row.get('decode_indices') != slots
                           or row.get('output_layouts') != [record] for row in rows)):
            raise ValueError('Optional historical decoder layout differs from the current derived layout')
        crosscheck = dict(passed=True, samples=len(rows), fields=['decode_indices','output_layouts'],
            historical_expected_values_used=False)
    return dict(indices=slots, output_layouts=[record], current_output_name=name,
        inference_sources=['dsl.typecheck shapes/layouts','lower.linear -> kernels.linear.linear_flat -> FlatLayout(n,O)',
                           'lower.decode_outputs -> unpack_flat'], historical_metadata_crosscheck=crosscheck)


def prepare_reference(program, weights, image, checked, work, recorder, result, proof=None):
    import numpy as np
    from ..lower import pack_inputs, decode_outputs, _pack_layout
    from ..tensor_sim import run_tensor
    if image.dtype != np.float64 or image.shape != (1,3,32,32) or not np.isfinite(image).all():
        raise ValueError('Reference input must be finite original float64[1,3,32,32]')
    layout = output_layout(program, checked, proof)
    result['packing_attempts'] += 1
    with recorder.span('current_frozen_train0_pack'):
        blocks = pack_inputs(program,[image])
        if len(blocks) != 1 or blocks[0].shape != (1,16384) or blocks[0].dtype != np.float64:
            raise ValueError('Current input must pack into one original16384-slot ciphertext')
        pack = next(op for op in program.ops if op.kind=='pack' and op.inputs==(program.params[0].name,))
        if pack.attrs['layout'] != 'mux' or pack.attrs['gap'] != 1:
            raise ValueError('Current source input packing must retain mux/gap1')
        decoded = decode_outputs(blocks,[_pack_layout(pack,program.params[0].shape,16384)])[0]
        if decoded.shape != image.shape or not np.array_equal(decoded,image):
            raise ValueError('Current original input packing did not preserve all image values')
        input_path = Path(work)/'input.bin'
        blocks[0][0].astype('<f8',copy=False).tofile(input_path)
        result['packing_completed'] += 1
    result['tensor_attempts'] += 1
    with recorder.span('new_frozen_program_tensor_train0'):
        tensor = run_tensor(program,weights,image,cache=False,act_mode='program')
        if tensor.dtype != np.float64 or tensor.shape != (1,10) or not np.isfinite(tensor).all():
            raise ValueError('The new frozen-program Tensor reference must be finite float64[1,10]')
        result['tensor_completed'] += 1
    expected = Path(work)/'expected.json'
    write_json(expected,dict(indices=layout['indices'],values=tensor.reshape(-1).tolist(),
        tensor_values=tensor.reshape(-1).tolist(),sample_id='train:0',dtype='float64',shape=[1,10],
        reference='Current independently frozen freshR30 program, one Tensor forward outside Agent',
        criterion=CRITERION,old_expected_values_used=False,security_acceptance=None,
        accuracy_acceptance=None,compiled_plan_mathematical_equivalence_verified=None))
    result.update(reference_prepared=True,training_samples_consumed=1,input=str(input_path),expected=str(expected),
        output_layout=layout,input_packing=dict(layout='mux',gap=1,scalar_multiplier=1,logical_slots=16384,
            exact_input_roundtrip=True,roundtrip_elements=3072),tensor_values=tensor.reshape(-1).tolist())


def worker(args):
    args.work_dir.mkdir(parents=True,exist_ok=False)
    tmp=args.work_dir/'tmp';tmp.mkdir();os.environ['TMPDIR']=str(tmp);tempfile.tempdir=str(tmp)
    recorder=StageRecorder(args.work_dir/'events.jsonl');result=_initial_result();started=time.perf_counter_ns()
    old = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame): raise TimeoutError('Fresh Tensor reference worker wall allowance exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,WORKER_SECONDS)
    def save(): write_json(args.work_dir/'worker-result.json',result)
    try:
        source=read(args.source_record);gate=source['gate']
        for field,actual in [('model_info',args.model_info),('program',args.program),('probes',args.probes),('probe_source',args.probe_source)]:
            if Path(gate[field]).resolve()!=actual.resolve():raise ValueError('Worker source path differs: '+field)
        if Path(source['runtime_plan']).resolve()!=args.runtime_plan.resolve():raise ValueError('Worker RuntimePlan source differs')
        with recorder.span('fresh_source_metadata'):
            checked_source=_metadata_gate(gate,args.runtime_plan)
            result.update(source_checked=True,source=source,plan_inventory=checked_source['plan'])
        _numeric_modules()
        from ..models.info import ModelInfo
        from ..dsl.parser import parse
        from ..dsl.typecheck import check
        from .r14_common_worker import check_program_signature
        result['model_info_load_attempts']+=1;save()
        with recorder.span('current_fresh_model_info'):
            info=ModelInfo.load(args.model_info);program=parse(args.program.read_text())
            checked=check(program,{k:v.shape for k,v in info.graph.weights.items()},graph=info.graph)
            if not checked.ok or check_program_signature(info,program)['passed'] is not True:
                raise ValueError('Current frozen program typecheck or original model signature failed')
            result['model_info_load_completed']+=1
        result['probe_load_attempts']+=1;save()
        with recorder.span('current_saved_training_probe'):
            image,probes=load_training_probe(args.probes,checked_source['probe_source'])
            result.update(probe_metadata=probes,training_npz_rows_loaded=3);result['probe_load_completed']+=1
        proof = None if args.layout_proof is None else read(args.layout_proof)
        prepare_reference(program,info.graph.weights,image,checked,args.work_dir,recorder,result,proof)
        result['status']='ok'
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        result['elapsed_ns']=time.perf_counter_ns()-started
        result['timing']=summarize_events(read_events(args.work_dir/'events.jsonl'))
        save()
    return result


@contextmanager
def budget_guard(out):
    stop=threading.Event();failures=[];pid=os.getpid()
    def watch():
        while not stop.wait(.5):
            try:
                if artifact_bytes(out)>CASE_BYTES-TERMINAL_RESERVE_BYTES or shutil.disk_usage(out).free<FREE_BYTES:
                    raise RuntimeError('Fresh reference preparation disk allowance exhausted')
            except BaseException as exc:
                failures.append(type(exc).__name__+': '+str(exc));os.kill(pid,signal.SIGTERM);return
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try: yield failures
    finally:
        stop.set();thread.join(timeout=2)


def _command(args,source,work):
    gate=source['gate']
    command=[sys.executable,'-B','-m','pt_agent.tools.nixrun','--seconds',str(NIX_SECONDS),
        '--python',str(args.intake_python),'--','-B','-m',MODULE,'worker','--work-dir',str(work),
        '--source-record',str(args.out/'source-record.json'),'--runtime-plan',source['runtime_plan']]
    for key in ('model_info','program','probes','probe_source'):
        command+=['--'+key.replace('_','-'),gate[key]]
    if source['layout_proof'] is not None:command+=['--layout-proof',source['layout_proof']]
    return command


def _success(value):
    return (value.get('round')==ROUND and value.get('mode')==MODE and value.get('model')==MODEL
        and value.get('status')=='ok' and value.get('source_checked') is True and value.get('reference_prepared') is True
        and all(value.get(k)==1 for k in COUNTERS) and value.get('training_samples_consumed')==1
        and value.get('training_npz_rows_loaded')==3 and all(value.get(k)==0 for k in ZERO_FIELDS)
        and value.get('old_expected_values_read') is False and value.get('normalization_reapplied') is False
        and value.get('compiled_plan_mathematical_equivalence_verified') is None)


def check_outputs(value, work):
    """Validate new small output artifacts without reading packed binary values."""
    work=Path(work).resolve();input_path=Path(value['input']);expected_path=Path(value['expected'])
    if (input_path.resolve()!=work/'input.bin' or expected_path.resolve()!=work/'expected.json'
            or input_path.is_symlink() or expected_path.is_symlink() or input_path.stat().st_size!=16384*8):
        raise ValueError('Reference artifacts must belong to this preparation and retain original slot length')
    expected=read(expected_path);indices=expected.get('indices');values=expected.get('values')
    if (indices!=value['output_layout']['indices'] or indices!=list(range(10))
            or expected.get('shape')!=[1,10] or expected.get('dtype')!='float64'
            or expected.get('sample_id')!='train:0' or expected.get('old_expected_values_used') is not False
            or expected.get('criterion')!=CRITERION or not isinstance(values,list) or len(values)!=10
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in values)
            or values!=value.get('tensor_values') or values!=expected.get('tensor_values')):
        raise ValueError('The new expected artifact differs from its actual Tensor output or decoder')
    return dict(passed=True,input_bytes=input_path.stat().st_size,expected_logits=10,
        binary_values_read=False,current_tensor_values_persisted=True)


def snapshot_sources(out):
    """Retain trusted entry/source-gate code and its concrete dependency paths."""
    directory=Path(out)/'source-snapshots';directory.mkdir()
    parent=Path(__file__).parent
    selected=(('entry.py',Path(__file__),'outer and worker entry'),
        ('source-gate.py',parent/'r30_frozen_cnn_report.py','source_gate and official_normalization'),
        ('agent-audit.py',parent/'r30_common_model_e2e.py','audit_agent and accepted phase contract'),
        ('original-factories.py',parent/'r13_common_models.py','original factory source/checkpoint identities'))
    rows=[]
    for name,path,role in selected:
        destination=directory/name;shutil.copyfile(path,destination);stat=path.stat()
        rows.append(dict(source=str(path.resolve()),snapshot=str(destination),role=role,
            size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    write_json(directory/'source-paths.json',dict(files=rows,integrity_checks=0,
        scope='Trusted source snapshots and filesystem metadata only; no credentials, model weights or sample arrays'))
    return rows


def execute(args):
    from .. import paths
    from ..registry import git_rev
    from .r30_frozen_cnn_report import clean
    started=time.perf_counter_ns();out=args.out.resolve();source_case=args.source_case.resolve()
    if (out==paths.RESULTS_DIR.resolve() or not out.is_relative_to(paths.RESULTS_DIR.resolve())
            or out.is_relative_to(source_case) or source_case.is_relative_to(out)):
        raise ValueError('Preparation must use a new separate child of PT_AGENT_RESULTS_ROOT')
    out.mkdir(parents=True,exist_ok=False);args.out=out
    recorder=StageRecorder(out/'events.jsonl');result=dict(round=ROUND,mode=MODE,model=args.model,
        status='running',error=None,source_case=str(source_case),processes=[],worker=None,
        source_case_modified=False,security_acceptance=None,CKKS_latency=None,accuracy_acceptance=None)
    result.update({k:0 for k in ZERO_FIELDS});revision=git_rev()
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def stop(signum,frame):raise RuntimeError('Fresh reference preparation interrupted: '+str(signum))
    for sig in old:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,OVERALL_SECONDS)
    root_span=recorder.span('fresh_native_reference_preparation')
    root_span.__enter__()
    try:
        with recorder.span('preparation_source_snapshots'):
            result['source_snapshots']=snapshot_sources(out)
        with recorder.span('fresh_accepted_source_preflight'):
            source=prepare_source(source_case,args.model,args.layout_proof)
            write_json(out/'source-record.json',source);result['source']=source
        command=_command(args,source,out/'work')
        env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','NIX_','__NIX_','HECATE_'))
             and k not in ('HECATE','IN_NIX_SHELL','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV')}
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        with budget_guard(out) as failures,recorder.span('bounded_fresh_reference_worker',argv=command):
            observation=run_process(command,cwd=paths.REPO/'DSL/DSL_agent/scripts',log=out/'worker.log',
                resources=out/'resources.jsonl',timeout=min(570,OVERALL_SECONDS-(time.perf_counter_ns()-started)/1e9),
                mem_gib=MEMORY_GIB,fsize_gib=8,env=env)
            result['processes'].append(dict(argv=command,observation=observation))
            if (out/'work/worker-result.json').is_file():result['worker']=read(out/'work/worker-result.json')
            if failures or not clean(observation) or not _success(result.get('worker') or {}):
                raise RuntimeError('New frozen-program input/reference preparation did not complete')
            result['artifact_check']=check_outputs(result['worker'],out/'work')
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        if (out/'work/worker-result.json').is_file():result['worker']=read(out/'work/worker-result.json')
        if result['status']=='error':
            error=RuntimeError(result['error']);root_span.__exit__(type(error),error,None)
        else:root_span.__exit__(None,None,None)
        outer_events=read_events(out/'events.jsonl');inner_events=read_events(out/'work/events.jsonl')
        parent=next((r['event_id'] for r in outer_events if r.get('type')=='begin'
            and r.get('stage_id')=='bounded_fresh_reference_worker'),None)
        for row in inner_events:
            if row.get('parent_id') is None:row['parent_id']=parent
        timing=summarize_events(outer_events+inner_events,time.perf_counter_ns())
        result['timing']=timing;result['recorded_elapsed_seconds']=(time.perf_counter_ns()-started)/1e9
        result['metrics']={key:result.get(key) for key in ('status','model','mode','security_acceptance','accuracy_acceptance','CKKS_latency')}
        result['metrics'].update(worker={key:(result.get('worker') or {}).get(key) for key in
            COUNTERS+ZERO_FIELDS+('reference_prepared','training_samples_consumed')},feedback_to_agent=False)
        result['artifact_bytes']=0
        for _ in range(20):
            write_json(out/'result.json',result);size=artifact_bytes(out)
            if size+TERMINAL_RESERVE_BYTES>CASE_BYTES or shutil.disk_usage(out).free<FREE_BYTES:
                result.update(status='error',error='Fresh reference terminal disk allowance exhausted');result['metrics']['status']='error'
            if size==result['artifact_bytes']:break
            result['artifact_bytes']=size
        else:raise RuntimeError('Fresh reference terminal artifact size did not converge')
        row=dict(run_id=out.name,round=ROUND,model=args.model,mode=MODE,git_rev=revision,
            metrics=result['metrics'],artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],timing=timing,
            required_artifact_bytes=artifact_bytes(out))
        for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    outer=sub.add_parser('prepare');outer.add_argument('--source-case',type=Path,required=True)
    outer.add_argument('--out',type=Path,required=True);outer.add_argument('--model',choices=(MODEL,),default=MODEL)
    outer.add_argument('--intake-python',type=Path);outer.add_argument('--layout-proof',type=Path)
    inner=sub.add_parser('worker')
    for key in ('work-dir','source-record','runtime-plan','model-info','program','probes','probe-source'):
        inner.add_argument('--'+key,type=Path,required=True)
    inner.add_argument('--layout-proof',type=Path)
    args=p.parse_args(argv)
    if args.phase=='prepare':
        if args.intake_python is None:
            from .. import paths
            args.intake_python=paths.INTAKE_PYTHON
        result=execute(args)
    else:result=worker(args)
    print(json.dumps(dict(status=result['status'],error=result.get('error'),reference_prepared=
        (result.get('worker') or result).get('reference_prepared')),allow_nan=False))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
