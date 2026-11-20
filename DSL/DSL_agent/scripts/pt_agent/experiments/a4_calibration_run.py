"""Fresh repeated39 preparation, isolated build and one public CKKS calibration.

This independent entry leaves A4's whole-model client and runner unchanged.
No old native duration or whole-model clock enters the calibration table.
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
from . import a4_prepared_disk_client as disk
from . import a4_repeated_calibration as calibration
from .a4_prepared_disk_run import FLAGS, binding, parameter_check, parameter_metadata_check, read, terminal_category
from .a5_a6_rule63_search import guarded_process
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, summarize_events, write_json
from .r27_orbit_native_backend import _environment
from .r31_lattigo_binding import dft_static_inventory
from .r31_lattigo_profile39 import validate_prepared

GIB=1024**3
ROUND='A4-REPEATED39-CALIBRATION'


def native_argv(binary,source,out,mode):
    argv=[str(binary),'-mode',mode,'-plan',source['plan'],'-bundle',source['bundle'],
        '-input',source['input'],'-expected',source['expected'],*FLAGS,
        '-checkpoint-interval','1000','-output',str(out/(mode+'-client.json')),
        '-events',str(out/(mode+'-go-events.jsonl'))]
    if mode=='execute':argv+=['-prepared-dir',str(out/'prepared-store'),
                              '-calibration-metadata',source['metadata']]
    return argv


def setup_source(args,out):
    prepared=read(args.prepared/'result.json');closed=read(args.prepared/'completion.json')
    validate_prepared(prepared,c2s_bits=52)
    if (closed.get('status')!='ok' or closed.get('registry_persisted')is not True
            or closed.get('required_artifacts_persisted')is not True):
        raise ValueError('Reuse the durably closed approved C2S52/P61 preparation')
    module=args.prepared/'work/module';mod=(module/'go.mod').read_text()
    if ('require github.com/tuneinsight/lattigo/v6 v6.0.0'not in mod
            or ('replace github.com/tuneinsight/lattigo/v6 => '+prepared['lattigo_source'])not in mod):
        raise ValueError('Retain the original approved offline R31 Lattigo6.0.0 replacement')
    for path in (module/'go.sum',args.go_sdk/'bin/go',args.prepared/'work/module-cache'):
        if not path.exists():raise ValueError('Required existing offline artifact absent: '+str(path))
    source=calibration.prepare_public(out/'public-fixture',args.base_metadata,args.missing_metadata)
    plan=read(source['plan']);client=prepared['client']
    rotations={op['attrs']['steps']%16384 for op in plan['execution']if op['op']=='rotate'}-{0}
    residual=dict(n=client['residual_parameters']['n'],galois_count=len(rotations),
        one_key_bytes=client['residual_parameters']['evaluation_key_uint64_payload_bytes'])
    bootstrap=dict(galois_count=len(client['bootstrap_galois_elements']),
        one_key_bytes=client['bootstrap_parameter_summary']['evaluation_key_uint64_payload_bytes'])
    dft=dft_static_inventory(client['bootstrap_parameters'])
    source.update(storage=disk.storage_inventory(plan,residual,bootstrap,
        dft_bytes=dft['total_rns_coefficient_payload_bytes']),prepared_parameters=client)
    return source


def run(args):
    out=args.out.absolute()
    if out.exists() or out.parent.resolve()!=paths.RESULTS_DIR.resolve():
        raise ValueError('Fresh direct configured results child required')
    out.mkdir();work=out/'work';work.mkdir()
    started=time.perf_counter_ns();rec=StageRecorder(out/'events.jsonl')
    result=dict(round=ROUND,status='running',error=None,processes=[],api_calls=0,
        new_model_forwards=0,new_model_images=0,new_quality_evaluations=0,
        downloads=0,installations=0,extra_integrity_checks=0,security_acceptance=None,
        old_durations_reused=False,formal_prediction=None,
        limits=dict(native_seconds=args.native_seconds,address_space_gib=24,rss_gib=14,
                    host_available_gib=1.25,case_gib=40,free_gib=10))
    stopped=threading.Event();old={sig:signal.getsignal(sig)for sig in(signal.SIGTERM,signal.SIGINT)}
    def incoming(sig,frame):
        result['incoming_signal']=sig
        raise InterruptedError('Calibration runner stopped by signal '+str(sig))
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
    def phase(name,command,seconds,memory,env,cwd=None):
        row=dict(stage=name,argv=[str(x)for x in command]);result['processes'].append(row)
        with rec.span(name):
            row['observation']=guarded_process(row['argv'],cwd=cwd or work,
                log=out/(name+'.log'),resources=out/(name+'-resources.jsonl'),
                timeout=seconds,mem_gib=memory,fsize_gib=8,env=env)
        obs=row['observation']
        if (obs['exit_code']!=0 or obs['timed_out']or obs['signal']is not None
                or obs['remaining_pids']or obs.get('signals_sent')):
            raise RuntimeError('Bounded calibration phase failed: '+name)
    try:
        watcher.start()
        with rec.span('public_fixture_and_disk_admission'):
            source=setup_source(args,out);result['source']=source
            result['disk_admission']=disk.disk_admission(shutil.disk_usage(out).free,
                source['storage']['disk_projection_bytes'])
            if not result['disk_admission']['accepted']:
                result['resource_guard']=dict(reason='Preparation projection plus10GiB reserve',**result['disk_admission'])
                raise RuntimeError('Prepared calibration artifacts exceed available disk')
            write_json(out/'source-record.json',source)
        with rec.span('own_source_and_offline_cache_setup'):
            module=work/'module';module.mkdir();(module/'main.go').write_text(calibration.build_source())
            for name in('go.mod','go.sum'):shutil.copyfile(args.prepared/'work/module'/name,module/name)
            for name in('module-cache','build-cache'):shutil.copytree(args.prepared/'work'/name,work/name)
            for name in('tmp','gopath','telemetry'):(work/name).mkdir()
            (work/'telemetry/mode').write_text('off\n')
            for p in(Path(__file__),Path(calibration.__file__),Path(calibration.__file__).with_suffix('.go')):
                shutil.copyfile(p,out/p.name)
        env=_environment(work,args.go_sdk);env.update(GOMEMLIMIT='1500MiB',GOTELEMETRY='off')
        binary=work/'a4-calibration-client'
        phase('build',[args.go_sdk/'bin/go','build','-p=1','-buildvcs=false','-mod=readonly','-o',binary,'.'],600,4,env,module)
        phase('cache-selftest',native_argv(binary,source,out,'cache-selftest'),30,4,env)
        selftest=read(out/'cache-selftest-client.json')
        if selftest.get('status')!='ok' or selftest.get('cache_selftest_passed')is not True:
            raise ValueError('New calibration binary key-free cache selftest failed')
        phase('parameters',native_argv(binary,source,out,'parameters'),60,4,env)
        result['parameter_check']=parameter_check(read(out/'parameters-client.json'),source['prepared_parameters'])
        env['GOMEMLIMIT']='11GiB'
        phase('execute',native_argv(binary,source,out,'execute'),args.native_seconds,24,env)
        actual=read(out/'execute-client.json')
        parameter_metadata_check(actual,read(out/'parameters-client.json'))
        result['comparison']=disk.check_client(actual,read(source['plan']),read(source['expected']))
        summary=calibration.summarize(actual,read(source['metadata']))
        write_json(out/'calibration-summary.json',summary)
        if args.projection_plan is not None:
            projection=calibration.project(read(args.projection_plan),summary)
            projection['source_plan']=str(args.projection_plan)
            write_json(out/'operator-projection.json',projection)
            result['operator_projection_status']=projection['status']
        result.update(status='ok',target_count=summary['target_count'],cell_count=summary['cell_count'],
                      ckks_public_fixture_s=actual['ckks_inference_s'])
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        stopped.set()
        if watcher.is_alive():watcher.join(timeout=3)
        if watcher.is_alive():result.update(status='error',error='Calibration monitor remains active')
        if result.get('monitor_error'):result.update(status='error',error='Calibration monitor failed: '+result['monitor_error'])
        result['timing']=summarize_events(read_events(out/'events.jsonl'))
        result['category']=terminal_category(result,out);write_json(out/'result.json',result)
        row=dict(run_id=out.name,round=ROUND,model='public-native39-fixture',mode='prepared-disk-repeated-calibration',
            git_rev=None,metrics=dict(status=result['status'],category=result['category'],
            cell_count=result.get('cell_count'),target_count=result.get('target_count'),api_calls=0),
            artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
        with(paths.RESULTS_DIR/'registry.jsonl').open('a+')as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
            if any(json.loads(line).get('run_id')==out.name for line in stream if line.strip()):raise ValueError('Preserve previous calibration identity')
            stream.seek(0,os.SEEK_END);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
            timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        if not watcher.is_alive():
            for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=paths.RESULTS_DIR/'lattigo-c2s52-p61-prepare-r31-20261116')
    p.add_argument('--base-metadata',type=Path,default=paths.RESULTS_DIR/'independent-c2s52-p61-profile39-subset-prepare-20261116/profile-metadata.json')
    p.add_argument('--missing-metadata',type=Path,default=paths.RESULTS_DIR/'independent-native-three-cells-prepare-20261116/profile-metadata.json')
    p.add_argument('--projection-plan',type=Path,default=paths.RESULTS_DIR/'fresh-resnet20-c2s52-p61-native-preparation-20261116/c2s52-bound-runtime-plan.json')
    p.add_argument('--go-sdk',type=Path,default=paths.WORKSPACE/'baselines/toolchains/go125/go')
    p.add_argument('--native-seconds',type=int,default=3600)
    args=p.parse_args(argv)
    if not 1<=args.native_seconds<=7200:p.error('Finite calibration allowance1..7200s required')
    return 0 if run(args)['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
