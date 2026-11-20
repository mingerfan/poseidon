"""Finite original N17 train:0 execution using an independently checked reference.

Preparation is read-only. The only child is the accepted prepared native binary;
its original scale51 plan and physical32 decoder are retained.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from .. import paths
from . import dacapo_native_binding as binding
from . import dacapo_native_input as reference
from .dacapo_native_plan_math import check_restored_bundle
from .completion_metrics import artifact_bytes, persist_case_completion
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample
from .r31_lattigo_runtime_plan import physical_memory_observation

ROUND='DACAPO-NATIVE-EXECUTE'
MODEL=reference.MODEL
MODE='original-n17-saved-train0-physical32-native'
GIB=1024**3
SECONDS=10800
AS_GIB=24
RSS_GIB=14
AVAILABLE_GIB=1.25
CASE_BYTES=4*GIB
FREE_BYTES=10*GIB
PREPARE_VARIANT='dacapo-original-n17-q55-51x16-p61x3-scale51-logicalslots16-native-default-bootstrap'
ZERO_FIELDS=('api_calls','llm_calls','model_forwards','source_image_reads','checkpoint_reads',
    'compiler_runs','downloads','installations','integrity_checks','heldout_samples','confirm_samples','report_samples')
NATIVE_COUNTS=('keygen_api_attempts','keygen_api_completed','plaintext_encoding_attempts',
    'plaintext_encoding_completed','encryption_attempts','encryption_completed','compute_attempts',
    'compute_completed','bootstrap_attempts','bootstrap_completed','final_decryption_attempts',
    'final_decryption_completed','intermediate_decryptions')


def _read(path):return json.loads(Path(path).read_text())


def native_arguments(source, input_path, expected_path):
    """The fixed original binding; no caller-supplied runtime flags are executed."""
    return ['-mode','execute','-plan',source['plan'],'-bundle',source['bundle'],
        '-input',str(input_path),'-expected',str(expected_path),
        '-scalar-site-proof',source['scalar_source_proof'],'-source-math-proof',source['source_math_proof'],
        '-allow-bootstrap','-precision-bits','51','-bootstrap-cache','1','-check-input-ssa',
        '-trace-key-requests','-checkpoint-interval','1000']


def check_n17_parameters(client):
    residual=client.get('residual_parameters') or {};bootstrap=client.get('bootstrap_parameter_summary') or {}
    for record,level,q_count,p_count in ((residual,16,17,3),(bootstrap,31,32,4)):
        if (record.get('n')!=131072 or record.get('logical_log_slots')!=16
                or record.get('max_level')!=level or record.get('log_default_scale')!=51
                or len(record.get('q',[]))!=q_count or len(record.get('p',[]))!=p_count
                or any(type(x) is not int or x<=0 for x in record.get('q',[])+record.get('p',[]))):
            raise ValueError('Observed original N17 public parameter dimensions differ')
    if bootstrap['q'][:17]!=residual['q'] or bootstrap['p'][:3]!=residual['p']:
        raise ValueError('Observed N17 bootstrap parameters must extend the residual Q/P prefix')
    return dict(residual_parameters=residual,bootstrap_parameter_summary=bootstrap)


def source_gate(prepared, input_prepare):
    prepared=Path(prepared).resolve();input_prepare=Path(input_prepare).resolve()
    prior=_read(prepared/'result.json');inputs=_read(input_prepare/'result.json')
    client=prior.get('parameters') or {}
    if (prior.get('status')!='ok' or prior.get('variant')!=PREPARE_VARIANT
            or prior.get('tests_passed')!=11 or prior.get('actual_model_ckks_runs')!=0
            or any(prior.get(k)!=0 for k in ('extra_integrity_checks','downloads','installations','llm_calls'))
            or prior.get('cache_selftest',{}).get('cache_selftest_passed') is not True
            or prior.get('checkpoint_selftest',{}).get('checkpoint_selftest_passed') is not True):
        raise ValueError('Use the accepted eleven-test zero-key N17 final-decoder preparation')
    steps=prior.get('steps') or []
    if ([row.get('stage') for row in steps]!=['build','n17_tests','cache_selftest','checkpoint_selftest','parameters']
            or any(row.get('process',{}).get('exit_code')!=0 or row['process'].get('timed_out')
                or row['process'].get('signal') or row['process'].get('remaining_pids') for row in steps)):
        raise ValueError('Every finite native preparation process must have completed cleanly')
    capability=reference.check_native_decoder(client)
    if (client.get('variant')!=binding.VARIANT or client.get('precision_bits')!=51
            or client.get('bootstrap_cache_capacity')!=1
            or client.get('requested_checkpoint_interval')!=1000
            or client.get('bootstrap_input_divisor')!=0 or client.get('bootstrap_normalization_log2')!=0
            or any(client.get(k)!=0 for k in ('keygen_api_completed','encryption_completed','compute_completed',
                'bootstrap_attempts','bootstrap_completed','final_decryption_completed','intermediate_decryptions'))):
        raise ValueError('Original N17 precision, bootstrap and zero-key capability metadata differ')
    parameters=check_n17_parameters(client)
    binary=Path(prior['ready_binary']).resolve();source_go=Path(prior['source_go']).resolve()
    if (binary!=prepared/'work/dacapo-client' or source_go!=prepared/'work/module/main.go'
            or not binary.is_file() or not os.access(binary,os.X_OK) or not source_go.is_file()
            or _read(prepared/'parameters.json')!=client):
        raise ValueError('Use the observed prepared binary and its actual compiled source snapshot')
    if (inputs.get('round')!=reference.ROUND or inputs.get('model')!=MODEL or inputs.get('mode')!=reference.MODE
            or inputs.get('status')!='ok' or inputs.get('reference_prepared') is not True
            or inputs.get('source_plan_modified') is not False
            or inputs.get('saved_input_copy_attempts')!=1 or inputs.get('saved_input_copy_completed')!=1
            or any(inputs.get(k)!=0 for k in reference.ZERO_FIELDS)):
        raise ValueError('Use an accepted saved original train:0 input/reference preparation')
    saved=inputs['source'];source=reference.source_gate(saved['math_result'],saved['plan'])
    if source!=saved:raise ValueError('Recomputed original mathematical source differs from input preparation')
    plan=Path(source['plan']).resolve();bundle=Path(source['bundle']).resolve()
    if bundle!=plan.with_name(plan.name.removesuffix('.runtime-plan.json')+'.bundle'):
        raise ValueError('The original plan canonical runtime bundle is required')
    manifest=_read(bundle/'manifest.json')
    archive=Path(_read(source['math_result'])['selected_paths']['archive_manifest']).resolve()
    restored=check_restored_bundle(dict(manifest=manifest),bundle,archive)
    inp=Path(inputs['input']).resolve();expected=Path(inputs['expected']).resolve()
    if (inp!=input_prepare/'input.bin' or expected!=input_prepare/'expected.json'
            or inp.stat().st_size!=reference.INPUT_BYTES or _read(expected)!=reference.make_expected(source)):
        raise ValueError('Saved train:0 bytes metadata or independently prepared physical32 reference differs')
    argv=native_arguments(source,inp,expected)
    if inputs.get('native_arguments')!=argv:raise ValueError('Native argument whitelist differs from the fixed original binding')
    inventory=binding.plan_inventory(_read(plan),manifest,
        source_proof=_read(source['scalar_source_proof']),source_math=_read(source['source_math_proof']),plan_path=plan)
    binding.require_execution_ready(inventory)
    if (inventory.get('variant')!=binding.VARIANT or inventory.get('execution_count')!=12661
            or inventory.get('bootstrap_count')!=19 or inventory.get('inline_scalar_count')!=122
            or inventory.get('original_plan_modified') is not False):
        raise ValueError('The complete original N17 inventory differs')
    return dict(prepared=str(prepared),input_prepare=str(input_prepare),binary=str(binary),
        compiled_source=str(source_go),capability=capability,source=source,inventory=inventory,
        input=str(inp),expected=str(expected),native_arguments=argv,restored_bundle=restored,
        archive_manifest=str(archive),native_parameter_metadata=parameters)


def child_environment():
    env=os.environ.copy()
    env.update(GOMAXPROCS='1',GOMEMLIMIT='11GiB',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    return env


def check_budget(out):
    if artifact_bytes(out)+65536>CASE_BYTES:raise RuntimeError('N17 case disk allowance exhausted')
    if shutil.disk_usage(out).free<FREE_BYTES:raise RuntimeError('N17 free disk reserve exhausted')


def _notify_guard(out,why):
    try:write_json(Path(out)/'guard-observation.json',why)
    finally:os.kill(os.getpid(),signal.SIGTERM)


def check_client(client, inventory, expected, parameter_metadata):
    if (client.get('status')!='ok' or client.get('mode')!='execute' or client.get('variant')!=binding.VARIANT
            or client.get('encrypted_plan_completed') is not True
            or client.get('source_math_and_scalar_sites_verified') is not True
            or client.get('public_fixture_only') is not False or client.get('get_log_enabled') is not False
            or client.get('intermediate_decryptions')!=0
            or client.get('compute_attempts')!=inventory['execution_count']
            or client.get('compute_completed')!=inventory['execution_count']
            or client.get('bootstrap_attempts')!=inventory['bootstrap_count']
            or client.get('bootstrap_completed')!=inventory['bootstrap_count']
            or any(client.get(k)!=1 for k in ('encryption_attempts','encryption_completed',
                'final_decryption_attempts','final_decryption_completed'))):
        raise ValueError('Native original-plan execution did not complete exactly once')
    if check_n17_parameters(client)!=parameter_metadata:
        raise ValueError('Actual execution parameters differ from the accepted zero-key constructor')
    rows=client.get('input_ssa_observations') or []
    first=inventory['initialization_count']
    boots=client.get('bootstrap_observations') or []
    if (client.get('input_ssa_check_enabled') is not True
            or client.get('public_key_request_trace_enabled') is not True
            or len(rows)!=inventory['execution_count']
            or [r.get('ordinal') for r in rows]!=list(range(first,first+inventory['execution_count']))
            or any(r.get('metadata_preserved') is not True for r in rows)
            or dict(Counter(r.get('op') for r in rows))!=inventory['op_counts']
            or client.get('native_bootstrap_api_attempts')!=inventory['bootstrap_count']
            or client.get('native_bootstrap_api_completed')!=inventory['bootstrap_count']
            or len(boots)!=inventory['bootstrap_count']
            or any(r.get('native_completed') is not True or r.get('input_ssa_preserved') is not True
                or r.get('input_value_divisor')!=1 for r in boots)):
        raise ValueError('Complete original input-SSA and native-bootstrap preservation observations are required')
    contract=client.get('final_decoder_contract') or {}
    if (client.get('output_multiplier')!=32 or contract.get('decoder')!=reference.DECODER
            or contract.get('source_plan')!=expected['source_plan']
            or contract.get('source_math_proof')!=expected['source_math_proof']
            or contract.get('source_factory')!=binding.SOURCE_FACTORY or contract.get('sample_id')!='train:0'
            or contract.get('public_fixture') is not False):
        raise ValueError('Native final decoder must retain the original same-plan physical32 contract')
    raw=client.get('final_selected_raw_values');scaled=client.get('final_selected_scaled_values')
    if (not isinstance(raw,list) or not isinstance(scaled,list) or len(raw)!=10 or len(scaled)!=10
            or client.get('final_selected_values')!=scaled
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in raw+scaled)
            or any(s!=r*32 for r,s in zip(raw,scaled))):
        raise ValueError('Physical32 final decoder evidence differs')
    comparison=client.get('comparison') or {}
    if (comparison.get('passed') is not True or comparison.get('elements')!=10
            or comparison.get('absolute_tolerance')!=1e-5 or comparison.get('relative_tolerance')!=1e-4
            or any(abs(a-e)>1e-5+1e-4*abs(e) for a,e in zip(scaled,expected['values']))):
        raise ValueError('Original-unit final CKKS comparison failed')


def _terminal(out,result):
    for _ in range(20):
        write_json(out/'result.json',result);size=artifact_bytes(out)
        if size+65536>CASE_BYTES:
            result.update(status='error',error='Terminal N17 artifacts exceed case allowance')
            result['metrics']['status']='error'
        if result['artifact_bytes']==size:return
        result['artifact_bytes']=size
    raise RuntimeError('Terminal N17 metadata size did not converge')


def execute(args):
    out=Path(args.out).resolve();prepared=Path(args.prepared).resolve();inputs=Path(args.input_prepare).resolve()
    if (out.parent!=paths.RESULTS_DIR.resolve() or out.exists() or out.is_symlink()
            or any(out==p or out in p.parents or p in out.parents for p in (prepared,inputs))):
        raise ValueError('A fresh isolated case under the configured results root is required')
    started=time.perf_counter_ns();deadline=time.monotonic()+SECONDS;out.mkdir()
    recorder=StageRecorder(out/'events.jsonl');stop=threading.Event();guard=[];watcher=None
    result=dict(round=ROUND,model=MODEL,mode=MODE,variant=binding.VARIANT,status='running',error=None,
        native_attempts=0,native_completed=0,client=None,process=None,termination=None,artifact_bytes=0,
        security_acceptance=None,accuracy_acceptance=None,CKKS_latency=None,compiled_bootstraps=None,
        original_plan_modified=False,extra_integrity_checks=0,formal_matrix_complete=False,
        scope='One saved original train:0; sole final physical32 decoder; no task accuracy or security certificate',
        memory_policy=dict(address_space_gib=AS_GIB,subtree_rss_gib=RSS_GIB,minimum_available_ram_gib=AVAILABLE_GIB,
            go_soft_memory_limit='11GiB',single_thread=True,wall_seconds=SECONDS,case_bytes=CASE_BYTES,
            free_bytes=FREE_BYTES,checkpoint_interval=1000,checkpoint_unit='existing_save_request'))
    result.update(dict.fromkeys(ZERO_FIELDS,0))
    old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupt(sig,frame):
        if not guard:result['termination']=dict(cause='incoming_signal',signal=sig)
        raise InterruptedError('Incoming signal '+str(sig))
    for sig in old:signal.signal(sig,interrupt)
    def monitor():
        while not stop.wait(.2):
            try:
                observed=physical_memory_observation(_latest_rss_sample(out/'native-resources.jsonl'),
                    _available_ram_bytes(),rss_gib=RSS_GIB,minimum_available_gib=AVAILABLE_GIB)
                if observed['stop']:
                    why=dict(cause='physical_memory_guard',observation=observed)
                elif time.monotonic()>=deadline:why=dict(cause='wall_allowance')
                else:
                    check_budget(out);continue
            except BaseException as exc:why=dict(cause='artifact_guard',error=type(exc).__name__+': '+str(exc))
            guard.append(why);result['termination']=why
            _notify_guard(out,why)
            return
    try:
        with recorder.span('original_n17_complete_execution'):
            with recorder.span('source_and_capability_gate'):
                selected=source_gate(prepared,inputs);result['selected']=selected
                write_json(out/'source-record.json',selected);check_budget(out)
                for source,name in ((Path(__file__),'execution-source.py'),
                    (Path(binding.__file__),'binding-source.py'),(Path(reference.__file__),'reference-preparation-source.py'),
                    (Path(selected['compiled_source']),'actual-compiled-client.go')):shutil.copyfile(source,out/name)
            with recorder.span('native_single_original_plan'):
                observation=physical_memory_observation(None,_available_ram_bytes(),rss_gib=RSS_GIB,
                    minimum_available_gib=AVAILABLE_GIB)
                if observation['stop']:
                    result['termination']=dict(cause='physical_memory_admission',observation=observation)
                    raise RuntimeError('Available RAM reserve prevents native launch')
                argv=[selected['binary']]+selected['native_arguments']+['-output',str(out/'client.json'),
                    '-events',str(out/'go-events.jsonl')]
                result['command']=argv;result['native_attempts']=1
                watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
                try:
                    result['process']=run_process(argv,cwd=out,log=out/'native.log',resources=out/'native-resources.jsonl',
                        timeout=max(.01,deadline-time.monotonic()),mem_gib=AS_GIB,fsize_gib=8,env=child_environment())
                finally:
                    stop.set();watcher.join(timeout=3)
                proc=result['process']
                if guard:raise RuntimeError('Native execution stopped by '+guard[0]['cause'])
                if proc.get('signal'):
                    result['termination']=dict(cause='incoming_signal',signal=proc['signal'])
                elif proc.get('timed_out'):result['termination']=dict(cause='wall_allowance')
                elif proc.get('exit_code') not in (0,None):result['termination']=dict(cause='child_exit',exit_code=proc['exit_code'])
                if (proc.get('exit_code')!=0 or proc.get('timed_out') or proc.get('signal')
                        or proc.get('remaining_pids')):raise RuntimeError('Native child failed or left descendants')
                result['client']=_read(out/'client.json')
                check_client(result['client'],selected['inventory'],_read(selected['expected']),
                    selected['native_parameter_metadata'])
                result['native_completed']=1;check_budget(out)
        result['status']='ok'
    except BaseException as exc:result.update(status='error',error=type(exc).__name__+': '+str(exc))
    finally:
        stop.set()
        if watcher is not None:watcher.join(timeout=3)
        for sig in old:signal.signal(sig,signal.SIG_IGN)
        try:
            if (out/'client.json').is_file():
                try:result['client']=_read(out/'client.json')
                except (ValueError,OSError) as exc:result['client_read_error']=type(exc).__name__+': '+str(exc)
            result['timing']=summarize_events(read_events(out/'events.jsonl'))
            result['recorded_pre_persistence_seconds']=(time.perf_counter_ns()-started)/1e9
            client=result.get('client') or {}
            result['metrics']={k:result.get(k) for k in ('status','native_attempts','native_completed','termination',
                'variant','security_acceptance','accuracy_acceptance','CKKS_latency','compiled_bootstraps')+ZERO_FIELDS}
            result['metrics']['native_counts']={k:client.get(k) for k in NATIVE_COUNTS}
            result['metrics']['comparison']=client.get('comparison')
            result['metrics']['memory_policy']=result['memory_policy'];_terminal(out,result)
            from ..registry import git_rev
            row=dict(run_id=out.name,round=ROUND,model=MODEL,mode=MODE,git_rev=git_rev(),metrics=result['metrics'],
                artifacts_dir=str(out),ts=datetime.now(timezone.utc).isoformat())
            with (paths.RESULTS_DIR/'registry.jsonl').open('a') as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            persist_case_completion(out,round_label=ROUND,started_ns=started,status=result['status'],
                timing=result['timing'],required_artifact_bytes=artifact_bytes(out))
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','prepared','input-prepare'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv);result=execute(args)
    print(json.dumps(dict(status=result['status'],native_attempts=result['native_attempts'],
        native_completed=result['native_completed'],error=result['error'])))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
