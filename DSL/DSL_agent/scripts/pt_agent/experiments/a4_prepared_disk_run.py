"""Source-bound fresh123 A4 runner using the existing approved R31 SDK.

Build once, check key-free cache/parameter modes, then execute once. This file
does not require the old interrupted native case to have a successful result.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from .. import paths
from . import a4_prepared_disk_client as client_source
from .a_group_runs import empty_record, append_record
from .a5_a6_rule63_search import guarded_process, sampled_subtree_peak
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json
from .r27_orbit_native_backend import _environment
from .r31_lattigo_binding import NativeBinding, map_precision_plan, dft_static_inventory

MODULE = 'pt_agent.experiments.a4_prepared_disk_run'
ROUND = 'A4-PREPARED-DISK'
GIB = 1024**3
FLAGS = ['-allow-bootstrap','-bootstrap-cache','32','-precision-bits','50',
         '-mod1-inv-degree','15','-log-message-ratio','2',
         '-bootstrap-normalization-log2','3','-auxiliary-prime-bits','61','-c2s-bits','52']


def read(path):
    return json.loads(Path(path).read_text())


def same(a, b):
    return isinstance(a,(str,Path)) and isinstance(b,(str,Path)) and Path(a).resolve()==Path(b).resolve()


def binding():
    return NativeBinding(50,mod1_inv_degree=15,log_message_ratio=2,
        bootstrap_normalization_log2=3,auxiliary_prime_bits=61,c2s_bits=52)


def validate_records(prep, current, math_result, proof, expected, original_plan, mapped_plan):
    """Check mathematical/source contracts using saved records, never old CKKS."""
    inventory = current['source_metadata']['plan']
    worker = math_result.get('worker') or {}
    gate = current['gate']
    if (prep.get('status')!='prepared' or prep.get('variant')!=binding().variant(32)
            or not same(prep.get('source_case'),gate['source_case'])
            or not same(prep.get('source_plan'),current['runtime_plan'])
            or inventory.get('execution_count')!=17863 or inventory.get('bootstrap_count')!=123
            or math_result.get('status')!='ok' or worker.get('status')!='ok'
            or worker.get('source_checked') is not True or worker.get('math_passed') is not True
            or worker.get('mathematical_equivalence_verified') is not True
            or worker.get('replay_attempts')!=1 or worker.get('replay_completed')!=1
            or worker.get('op_completed')!=17863
            or not same(proof.get('source_plan'),current['runtime_plan'])
            or not same(proof.get('current_program'),gate['program'])
            or not same(proof.get('source_case'),gate['source_case'])
            or not same(proof.get('input'),prep.get('input'))
            or not same(proof.get('reference'),prep.get('expected'))
            or proof.get('status')!='strict_math_passed'
            or proof.get('factory',proof.get('source_factory'))!=gate['factory']
            or proof.get('sample_ids')!=['train:0'] or proof.get('operation_count')!=17863
            or proof.get('ideal_bootstrap_count')!=123
            or proof.get('criterion')!='1e-12+1e-12*abs(reference)'
            or expected.get('sample_id')!='train:0' or expected.get('old_expected_values_used') is not False
            or expected.get('indices')!=list(range(10))):
        raise ValueError('Fresh123 original frozen source, strict plan math or train:0 reference differs')
    mapped, report = map_precision_plan(original_plan,binding(),bootstrap_cache=32)
    if mapped!=mapped_plan:
        raise ValueError('Original instruction/value order or explicit C2S52/P61 mapping differs')
    return report


def source_gate(args):
    from . import r31_fresh_cnn_input as fresh
    from .r31_lattigo_runtime_plan import plan_inventory
    prep = read(args.preparation)
    current = fresh.prepare_source(Path(prep['source_case']))
    math_result = read(prep['source_math_result']);proof = read(prep['source_math_proof'])
    completion = read(Path(prep['source_math_result']).parent/'completion.json')
    if (completion.get('status')!='ok' or completion.get('registry_persisted') is not True
            or completion.get('required_artifacts_persisted') is not True):
        raise ValueError('Saved original mathematical proof must be durably closed')
    original_plan = read(prep['source_plan']);mapped = read(prep['bound_plan'])
    expected = read(prep['expected'])
    report = validate_records(prep,current,math_result,proof,expected,original_plan,mapped)
    if not Path(prep['input']).is_file() or not Path(prep['bundle']).is_dir():
        raise ValueError('Retain prepared original training input and full constant bundle')
    manifest = read(Path(prep['bundle'])/'manifest.json')
    inventory = plan_inventory(mapped,manifest,binding=binding())
    prepared = read(args.prepared/'result.json');closed = read(args.prepared/'completion.json')
    if (prepared.get('status')!='ok' or prepared.get('phase')!='prepare'
            or closed.get('status')!='ok' or closed.get('registry_persisted') is not True
            or prepared.get('client',{}).get('variant')!=binding().variant(32)):
        raise ValueError('Reuse the closed, previously approved R31 public-parameter preparation')
    old_client = prepared['client']
    residual = dict(n=old_client['residual_parameters']['n'],
        galois_count=len({op['attrs']['steps']%16384 for op in mapped['execution']
                          if op['op']=='rotate' and op['attrs']['steps']%16384}),
        one_key_bytes=old_client['residual_parameters']['evaluation_key_uint64_payload_bytes'])
    bootstrap = dict(galois_count=len(old_client['bootstrap_galois_elements']),
        one_key_bytes=old_client['bootstrap_parameter_summary']['evaluation_key_uint64_payload_bytes'])
    dft = dft_static_inventory(old_client['bootstrap_parameters'])
    storage = client_source.storage_inventory(mapped,residual,bootstrap,
        dft_bytes=dft['total_rns_coefficient_payload_bytes'])
    module = args.prepared/'work/module'
    mod = (module/'go.mod').read_text()
    if ('require github.com/tuneinsight/lattigo/v6 v6.0.0' not in mod
            or ('replace github.com/tuneinsight/lattigo/v6 => '+prepared['lattigo_source']) not in mod):
        raise ValueError('Keep the exact approved Lattigo6.0.0 isolated replacement')
    for path in (module/'go.sum',args.go_sdk/'bin/go',args.prepared/'work/module-cache'):
        if not path.exists():
            raise ValueError('Required existing offline SDK/cache artifact is absent: '+str(path))
    return dict(preparation=str(args.preparation),source_case=prep['source_case'],
        program=current['gate']['program'],original_plan=prep['source_plan'],plan=prep['bound_plan'],
        bundle=prep['bundle'],input=prep['input'],expected=prep['expected'],
        original_math_proof=prep['source_math_proof'],original_math_result=prep['source_math_result'],
        original_math_completion=str(Path(prep['source_math_result']).parent/'completion.json'),
        mapping=report,inventory=inventory,storage=storage,prepared=str(args.prepared),
        prepared_parameters=old_client,lattigo_source=prepared['lattigo_source'],
        source_sample='train:0',new_model_forwards=0,new_math_replays=0,
        old_native_result_required=False,source_function_or_reference_changed=False)


PUBLIC_PARAMETER_FIELDS = ('variant','precision_bits','c2s_bits','auxiliary_prime_bits','bootstrap_cache_capacity',
                'bootstrap_normalization_log2','bootstrap_input_divisor','residual_parameters',
                'bootstrap_parameter_summary','bootstrap_precision_policy')


def parameter_metadata_check(actual, reference):
    for key in PUBLIC_PARAMETER_FIELDS:
        if actual.get(key)!=reference.get(key):
            raise ValueError('Actual new binary parameter metadata differs: '+key)


def parameter_check(actual, reference):
    parameter_metadata_check(actual, reference)
    if actual.get('status')!='ok' or actual.get('mode')!='parameters':
        raise ValueError('Actual new binary key-free parameters check failed')
    for key in ('keygen_api_attempts','keygen_api_completed','encryption_attempts','bootstrap_attempts',
                'final_decryption_attempts','plaintext_encoding_attempts'):
        if actual.get(key)!=0:
            raise ValueError('Parameter-only mode performed a cryptographic action: '+key)
    return dict(passed=True,actual_public_parameter_constructor=True,keygen=0,ckks=0)


def native_argv(binary, source, out, mode):
    command=[str(binary),'-mode',mode,'-plan',source['plan'],'-bundle',source['bundle'],
        '-input',source['input'],'-expected',source['expected'],*FLAGS,
        '-checkpoint-interval','1000','-output',str(out/(mode+'-client.json')),
        '-events',str(out/(mode+'-go-events.jsonl'))]
    if mode=='execute':
        command+=['-prepared-dir',str(out/'prepared-store')]
    return command


def terminal_category(result, out):
    if result.get('resource_guard') or list(out.glob('*.physical-guard.json')):
        return 'resource_limit'
    if result.get('status')=='ok':
        return 'ok'
    if any((p.get('observation')or{}).get('timed_out') for p in result['processes']):
        return 'timeout'
    return 'driver_error'


def run(args):
    out=args.out.absolute()
    if out.exists() or out.parent.resolve()!=paths.RESULTS_DIR.resolve():
        raise ValueError('Fresh direct configured results child required')
    out.mkdir();work=out/'work';work.mkdir()
    started=time.perf_counter_ns();rec=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,model='resnet20',status='running',error=None,processes=[],
        api_calls=0,new_model_forwards=0,new_math_replays=0,new_source_images=0,
        new_quality_evaluations=0,downloads=0,installations=0,extra_integrity_checks=0,
        security_acceptance=None,formal_prediction=None,ckks_inference_s=None,
        limits=dict(native_seconds=args.native_seconds,address_space_gib=24,rss_gib=14,
                    host_available_gib=1.25,case_gib=40,free_gib=10))
    stopped=threading.Event();old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def incoming(sig,frame):
        result['incoming_signal']=sig
        raise InterruptedError('Own runner stopped by signal '+str(sig))
    for sig in old:signal.signal(sig,incoming)
    def monitor():
        try:
            while not stopped.wait(.5):
                amount=artifact_bytes(out);free=shutil.disk_usage(out).free
                if amount>40*GIB or free<10*GIB:
                    result['resource_guard']=dict(artifact_bytes=amount,free_bytes=free,
                        case_limit_bytes=40*GIB,minimum_free_bytes=10*GIB)
                    try:write_json(out/'resource-guard.json',result['resource_guard'])
                    finally:os.kill(os.getpid(),signal.SIGTERM)
                    return
        except BaseException as exc:
            result['monitor_error']=type(exc).__name__+': '+str(exc)
            if not stopped.is_set():os.kill(os.getpid(),signal.SIGTERM)
    watcher=threading.Thread(target=monitor,daemon=True)
    def phase(name, command, seconds, memory, env, cwd=None):
        row=dict(stage=name,argv=[str(x)for x in command]);result['processes'].append(row)
        with rec.span(name):
            row['observation']=guarded_process(row['argv'],cwd=cwd or work,
                log=out/(name+'.log'),resources=out/(name+'-resources.jsonl'),
                timeout=seconds,mem_gib=memory,fsize_gib=8,env=env)
        obs=row['observation']
        if (obs['exit_code']!=0 or obs['timed_out'] or obs['signal'] is not None
                or obs['remaining_pids'] or obs.get('signals_sent')):
            raise RuntimeError('Bounded own client phase failed: '+name)
    try:
        watcher.start()
        with rec.span('source_and_storage_admission'):
            source=source_gate(args);result['source']=source
            admission=client_source.disk_admission(shutil.disk_usage(out).free,
                source['storage']['disk_projection_bytes'])
            result['disk_admission']=admission
            if not admission['accepted']:
                result['resource_guard']=dict(reason='preparation disk projection plus10GiB reserve',**admission)
                raise RuntimeError('Prepared artifact projection exceeds available disk')
            write_json(out/'source-record.json',source)
        with rec.span('own_source_and_offline_cache_setup'):
            module=work/'module';module.mkdir()
            (module/'main.go').write_text(client_source.build_source())
            for name in ('go.mod','go.sum'):
                shutil.copyfile(args.prepared/'work/module'/name,module/name)
            shutil.copytree(args.prepared/'work/module-cache',work/'module-cache')
            shutil.copytree(args.prepared/'work/build-cache',work/'build-cache')
            for name in ('tmp','gopath','telemetry'):(work/name).mkdir()
            (work/'telemetry/mode').write_text('off\n')
            for path in (Path(__file__),Path(client_source.__file__),Path(client_source.__file__).with_suffix('.go')):
                shutil.copyfile(path,out/path.name)
        env=_environment(work,args.go_sdk);env.update(GOMEMLIMIT='1500MiB',GOTELEMETRY='off')
        binary=work/'a4-client'
        phase('build',[args.go_sdk/'bin/go','build','-p=1','-buildvcs=false','-mod=readonly',
              '-o',binary,'.'],600,4,env,module)
        phase('cache-selftest',native_argv(binary,source,out,'cache-selftest'),30,4,env)
        selftest=read(out/'cache-selftest-client.json')
        if selftest.get('status')!='ok' or selftest.get('cache_selftest_passed')is not True:
            raise ValueError('New binary key-free cache selftest failed')
        phase('parameters',native_argv(binary,source,out,'parameters'),60,4,env)
        result['parameter_check']=parameter_check(read(out/'parameters-client.json'),source['prepared_parameters'])
        env['GOMEMLIMIT']='11GiB'
        phase('execute',native_argv(binary,source,out,'execute'),args.native_seconds,24,env)
        actual=read(out/'execute-client.json')
        parameter_metadata_check(actual,read(out/'parameters-client.json'))
        result['comparison']=client_source.check_client(actual,read(source['plan']),read(source['expected']))
        result.update(status='ok',ckks_inference_s=actual['ckks_inference_s'])
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        stopped.set();watcher.join(timeout=3)
        if watcher.is_alive():result.update(status='error',error='Own disk monitor remains active')
        if result.get('monitor_error'):
            result.update(status='error',error='Own disk monitor failed: '+result['monitor_error'])
        if (out/'execute-client.json').is_file():
            actual=read(out/'execute-client.json')
            result['client_summary']={k:actual.get(k)for k in ('status','error','ckks_inference_s',
                'ckks_inference_completed','no_keygen_or_constant_encoding_during_inference',
                'prepared_key_generation_s','prepared_constant_encoding_s','prepared_store_writes_s',
                'prepared_store_serialized_bytes','prepared_inference_scope')}
        result['timing']=summarize_events(read_events(out/'events.jsonl'))
        result['category']=terminal_category(result,out)
        write_json(out/'result.json',result)
        row=dict(run_id=out.name,round=ROUND,model='resnet20',mode='prepared-disk-native',git_rev=None,
            metrics=dict(status=result['status'],category=result['category'],
                         ckks_inference_s=result['ckks_inference_s'],api_calls=0),
            artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):
                raise ValueError('Preserve a previous A4 registry identity')
            stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        paper=empty_record(run_id=out.name,task='A4',model='resnet20',system='arranger',
            param_group='r31-N16-C2S52-P61-scale50',status=result['category'],
            status_detail=result.get('error') or 'All keys/constants prepared before encryption; bounded disk-backed runtime caches',
            budget_s=args.native_seconds,peak_rss_gib=sampled_subtree_peak(out),
            boots_compiled=123 if result.get('source') else None,
            exec_ops=17863 if result.get('source') else None,ckks_inference_s=result['ckks_inference_s'],
            ckks_scope=(result.get('client_summary')or{}).get('prepared_inference_scope'),
            api_calls=0,source=str(out/'result.json'),finished_utc=datetime.now(timezone.utc).isoformat())
        append_record(args.paper_runs,paper);write_json(out/'paper-record.json',paper)
        if not watcher.is_alive():
            for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--paper-runs',type=Path,required=True)
    p.add_argument('--preparation',type=Path,default=paths.RESULTS_DIR/'fresh-resnet20-c2s52-p61-native-preparation-20261116/preparation.json')
    p.add_argument('--prepared',type=Path,default=paths.RESULTS_DIR/'lattigo-c2s52-p61-prepare-r31-20261116')
    p.add_argument('--go-sdk',type=Path,default=paths.WORKSPACE/'baselines/toolchains/go125/go')
    p.add_argument('--native-seconds',type=int,default=7200)
    args=p.parse_args(argv)
    if not 1<=args.native_seconds<=10800:p.error('Use a finite native wall allowance up to10800s')
    return 0 if run(args)['status']=='ok' else 1


if __name__=='__main__':
    raise SystemExit(main())
