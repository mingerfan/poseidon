"""One isolated bootstrap diagnostic using native HEIR CNN parameters.

This probe never invokes the complete generated CNN or its preprocessing.
It uses one saved training input and preserves the native sparse-key path.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from .. import paths
from .heir_cnn_parameters import go_source, source_literals
from .heir_cnn_preflight import finalize_size
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json


def bootstrap_source(residual, bootstrap):
    source = go_source(residual, bootstrap, profile_bootstrap_keys=True)
    source = source.replace(' "encoding/json"', ' "encoding/json"\n "math"\n "strings"\n "strconv"\n "github.com/tuneinsight/lattigo/v6/core/rlwe"\n "github.com/tuneinsight/lattigo/v6/ring"')
    source = source.replace('runtime.GOMAXPROCS(1);debug.SetMemoryLimit(384<<20)',
                            'runtime.GOMAXPROCS(1);debug.SetMemoryLimit(11<<30)')
    marker = ' result["status"]="ok"'
    if source.count(marker) != 1:
        raise ValueError('The parameter-only source completion point changed')
    extra = r'''
 stages:=map[string]float64{};result["stage_seconds"]=stages
 result["parameter_only_key_requirements"]=result["bootstrap_key_metadata"];delete(result,"bootstrap_key_metadata")
 result["single_bootstrap_primitive_only"]=true;result["full_model_executed"]=false
 result["residual_keypair_attempts"]=0;result["residual_keypair_completed"]=0
 result["bootstrap_keys_attempts"]=0;result["bootstrap_keys_completed"]=0
 result["evaluator_attempts"]=0;result["evaluator_completed"]=0
 result["encode_attempts"]=0;result["encodes"]=0
 result["encrypt_attempts"]=0;result["bootstrap_attempts"]=0;result["decrypt_attempts"]=0
 result["active_stage"]="read_input";write()
 var values []float64
 raw,e:=os.ReadFile(os.Args[2]);if e!=nil{result["status"]="error";result["error"]=e.Error();return}
 if e=json.Unmarshal(raw,&values);e!=nil{result["status"]="error";result["error"]=e.Error();return}
 if len(values)!=16384{result["status"]="error";result["error"]="logical packing requires exactly 16384 entries";return}
 for _,v:=range values{if math.IsNaN(v)||math.IsInf(v,0){result["status"]="error";result["error"]="nonfinite source input";return}}
 // This process never exports secret coefficients or serialized key material.
 done:=make(chan struct{});defer close(done)
 go func(){
   ticker:=time.NewTicker(100*time.Millisecond);defer ticker.Stop()
   for{select{case <-done:return;case <-ticker.C:
     raw,e:=os.ReadFile("/proc/self/status");if e!=nil{continue}
     for _,line:=range strings.Split(string(raw),"\n"){
       if strings.HasPrefix(line,"VmRSS:"){
         fields:=strings.Fields(line);if len(fields)<2{continue};kib,e:=strconv.ParseInt(fields[1],10,64)
         if e==nil&&kib>14*1024*1024{fmt.Fprintln(os.Stderr,"physical RSS guard exceeded 14GiB");os.Exit(125)}
       }
     }
     raw,e=os.ReadFile("/proc/meminfo");if e==nil{
       for _,line:=range strings.Split(string(raw),"\n"){
         if strings.HasPrefix(line,"MemAvailable:"){
           fields:=strings.Fields(line);if len(fields)<2{continue};kib,e:=strconv.ParseInt(fields[1],10,64)
           if e==nil&&kib<1024*1024{fmt.Fprintln(os.Stderr,"host MemAvailable guard below 1GiB");os.Exit(125)}
         }
       }
     }
   }}
 }()
 result["active_stage"]="residual_keypair";result["residual_keypair_attempts"]=1;write();start=time.Now()
 kgen:=rlwe.NewKeyGenerator(p);sk,pk:=kgen.GenKeyPairNew()
 stages["residual_keypair"]=time.Since(start).Seconds();result["residual_keypair_completed"]=1;result["keygen_runs"]=1;write()
 result["active_stage"]="bootstrap_evaluation_keys";result["bootstrap_keys_attempts"]=1;write();start=time.Now()
 keys,_,e:=b.GenEvaluationKeys(sk)
 stages["bootstrap_evaluation_keys"]=time.Since(start).Seconds()
 if e!=nil{result["status"]="error";result["error"]=e.Error();return}
 result["bootstrap_keys_completed"]=1;write();runtime.GC();debug.FreeOSMemory()
 result["active_stage"]="bootstrap_evaluator_DFT";result["evaluator_attempts"]=1;result["dft_matrices_constructed"]=nil;write();start=time.Now()
 evaluator,e:=bootstrapping.NewEvaluator(b,keys)
 stages["bootstrap_evaluator_DFT"]=time.Since(start).Seconds()
 if e!=nil{result["status"]="error";result["error"]=e.Error();return}
 result["evaluator_completed"]=1;result["dft_matrices_constructed"]=2;write();runtime.GC();debug.FreeOSMemory()
 result["active_stage"]="encode";result["encode_attempts"]=1;write();start=time.Now()
 encoder:=ckks.NewEncoder(p);pt:=ckks.NewPlaintext(p,3)
 pt.LogDimensions=ring.Dimensions{Rows:0,Cols:14}
 if e=encoder.Encode(values,pt);e!=nil{result["status"]="error";result["error"]=e.Error();return}
 stages["encode"]=time.Since(start).Seconds();result["encodes"]=1;result["encoded_level"]=pt.Level();result["encoded_scale"]=pt.Scale.Float64();write()
 result["active_stage"]="encrypt";result["encrypt_attempts"]=1;write();start=time.Now()
 encryptor:=rlwe.NewEncryptor(p,pk);ct,e:=encryptor.EncryptNew(pt)
 stages["encrypt"]=time.Since(start).Seconds()
 if e!=nil{result["status"]="error";result["error"]=e.Error();return}
 result["encryptions"]=1;result["input_log_slots"]=ct.LogSlots();result["input_level"]=ct.Level();write()
 result["active_stage"]="bootstrap";result["bootstrap_attempts"]=1;write();start=time.Now()
 out,e:=evaluator.Bootstrap(ct)
 stages["bootstrap"]=time.Since(start).Seconds()
 if e!=nil{result["status"]="error";result["error"]=e.Error();return}
 result["bootstraps"]=1;result["output_log_slots"]=out.LogSlots();result["output_level"]=out.Level();write()
 result["active_stage"]="decrypt_and_check";result["decrypt_attempts"]=1;write();start=time.Now()
 decryptor:=rlwe.NewDecryptor(p,sk);observed:=make([]float64,len(values))
 if e=encoder.Decode(decryptor.DecryptNew(out),observed);e!=nil{result["status"]="error";result["error"]=e.Error();return}
 result["decryptions"]=1
 maxerr,maxratio:=0.0,0.0;passed,finite:=true,true
 for i,want:=range values{
   got:=observed[i];if math.IsNaN(got)||math.IsInf(got,0){finite=false;passed=false;continue}
   err:=math.Abs(got-want);tol:=1e-5+1e-4*math.Abs(want)
   if err>maxerr{maxerr=err};if err/tol>maxratio{maxratio=err/tol};if err>tol{passed=false}
 }
 stages["decrypt_and_check"]=time.Since(start).Seconds()
 result["finite"]=finite;result["CKKS_tolerance_passed"]=passed
 result["max_absolute_error"]=maxerr;result["max_tolerance_ratio"]=maxratio;result["elements_checked"]=len(values)
 result["criterion"]="1e-5 + 1e-4*abs(reference)";result["active_stage"]="complete"
 if !passed{result["status"]="error";result["error"]="native bootstrap exceeds the unchanged CKKS mathematical policy";return}
 result["status"]="ok"
'''
    return source.replace(marker, extra)


def main(argv=None, *, source_builder=None, module_adapter=None,
         backend_metadata=None, lifetime_regression=False):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'probes', 'probe-source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--memory-gib', type=float, default=24, help='Go address-space allowance; independent physical RSS guard remains 14GiB')
    parser.add_argument('--seconds', type=int, default=7200)
    parser.add_argument('--prepare-only', action='store_true', help='Prepare one real training input and compile the diagnostic; do not run keys or CKKS')
    parser.add_argument('--go', type=Path, default=paths.WORKSPACE / 'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/go')
    parser.add_argument('--cache-source', type=Path, default=paths.RESULTS_DIR / 'heir-single-ckks-r21-20261115/work')
    parser.add_argument('--input-python', type=Path, default=paths.WORKSPACE / 'baselines/orbit-venv/bin/python')
    args = parser.parse_args(argv)
    residual, bootstrap = source_literals(args.input)
    if residual['logN'] != 16 or bootstrap['logN'] != 16:
        parser.error('This independent primitive probe requires the actual same-N16 literals')
    out = args.out.absolute(); out.mkdir(parents=True, exist_ok=False)
    for name in ('module', 'tmp', 'telemetry'):
        (out / name).mkdir()
    (out / 'telemetry/mode').write_text('off\n')
    record = dict(round='HEIR-CNN-BOOTSTRAP', model='native-resnet20-parameter-primitive',
                  mode='single-native-bootstrap-engineering-probe', status='running', error=None,
                  limits=dict(address_space_gib=args.memory_gib, physical_rss_gib=14,
                              minimum_host_available_gib=1, GOMEMLIMIT='11GiB',
                              wall_seconds=args.seconds),
                  engineering_input_level=3, engineering_log_dimensions=[0,14],
                  source_sample_id='train:0', source_probe_rows_read=None,
                  planned_encoded_images=1, input_packing=dict(period=4096, repetitions=4, slots=16384),
                  sample_selection_for_quality=False,
                  source_mlir=str(args.input.absolute()), source_probes=str(args.probes.absolute()),
                  source_probe_record=str(args.probe_source.absolute()),
                  model_forward_attempts=0, full_model_executed=False, security_validated=False,
                  llm_calls=0, downloads=0, installations=0, extra_integrity_checks=0,
                  attempts=dict(input=0, build=0, execute=0), completed=dict(input=0, build=0, execute=0),
                  processes=[], native_result=None, compiled_bootstraps=None,
                  predicted_latency=None, CKKS_latency=None, accuracy_acceptance=None)
    if backend_metadata:
        record.update(backend_metadata)
    if lifetime_regression:
        record['attempts']['regression'] = 0
        record['completed']['regression'] = 0
    recorder = StageRecorder(out / 'events.jsonl')
    env = {key: value for key, value in os.environ.items() if key not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV') and not key.startswith(('LD_', 'NIX_', '__NIX_', 'GO'))}
    env.update(GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off',
               GOPATH=str(out / 'gopath'), GOMODCACHE=str(out / 'module-cache'), GOCACHE=str(out / 'build-cache'),
               GOTMPDIR=str(out / 'tmp'), TMPDIR=str(out / 'tmp'), TEST_TELEMETRY_DIR=str(out / 'telemetry'),
               CGO_ENABLED='0', GOMAXPROCS='1', GOMEMLIMIT='11GiB', PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    def run(stage, command, seconds, memory):
        if shutil.disk_usage(out).free < 10 * 1024**3:
            raise RuntimeError('Preserve 10GiB free-disk reserve before the next phase')
        record['attempts'][stage] += 1
        with recorder.span(stage, argv=command):
            obs = run_process(command, cwd=out / 'module', log=out / (stage + '.log'),
                              resources=out / (stage + '-resources.jsonl'), timeout=seconds, mem_gib=memory, env=env)
        record['processes'].append(dict(stage=stage, argv=command, observation=obs))
        if stage == 'execute' and obs['exit_code'] == 125:
            record['termination_origin'] = 'native resource guard; inspect execute.log'
        write_json(out / 'result.json', record)
        if obs['exit_code'] or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:
            raise RuntimeError(stage + ' failed; partial native result and resources retained')
        record['completed'][stage] += 1
    try:
        with recorder.span('single-native-bootstrap-primitive'):
            source = json.loads(args.probe_source.read_text())
            if source.get('split') != 'train' or source.get('indices') != [0, 1, 2] or source.get('sample_ids') != ['train:0', 'train:1', 'train:2']:
                raise ValueError('Only the unchanged saved R13 training probes are accepted')
            record['training_probe_source'] = source
            prepare = r'''import json,sys,numpy as np
from pathlib import Path
with np.load(sys.argv[1],allow_pickle=False) as archive:
    images=archive['input'];indices=archive['indices']
    if images.dtype!=np.float64 or images.shape!=(3,3,32,32) or indices.tolist()!=[0,1,2] or not np.isfinite(images).all():
        raise ValueError('Original R13 training probes must be finite float64[3,3,32,32]')
    packed=np.zeros(16384,dtype=np.float64)
    for offset in (0,4096,8192,12288):packed[offset:offset+3072]=images[0].reshape(-1)
Path(sys.argv[2]).write_text(json.dumps(packed.tolist())+'\n')
'''
            (out / 'prepare_input.py').write_text(prepare)
            run('input', [str(args.input_python), '-B', str(out / 'prepare_input.py'), str(args.probes.absolute()), str(out / 'packed-input.json')], 60, 1)
            record['source_probe_rows_read'] = 3
            (out / 'module/main.go').write_text((source_builder or bootstrap_source)(residual, bootstrap))
            with recorder.span('private-offline-cached-module'):
                for name in ('go.mod', 'go.sum'):
                    shutil.copyfile(args.cache_source / 'module' / name, out / 'module' / name)
                for name in ('module-cache', 'build-cache'):
                    shutil.copytree(args.cache_source / name, out / name)
                if module_adapter is not None:
                    record['isolated_module_adaptation'] = module_adapter(out)
            run('build', [str(args.go), 'build', '-p=1', '-mod=readonly', '-buildvcs=false', '-o', str(out / 'bootstrap-probe'), '.'], 300, 2)
            if lifetime_regression:
                run('regression', [str(out / 'bootstrap-probe'), '--dft-regression', str(out / 'dft-regression.json')], 180, 2)
                record['dft_lifetime_regression'] = json.loads((out / 'dft-regression.json').read_text())
                if record['dft_lifetime_regression'].get('status') != 'ok':
                    raise RuntimeError('DFT lifetime regression failed; original tolerance retained')
            if args.prepare_only:
                record['status'] = 'prepared'
            else:
                run('execute', [str(out / 'bootstrap-probe'), str(out / 'native-result.json'), str(out / 'packed-input.json')], args.seconds, args.memory_gib)
                record['native_result'] = json.loads((out / 'native-result.json').read_text())
                if record['native_result'].get('status') != 'ok' or record['native_result'].get('CKKS_tolerance_passed') is not True:
                    raise RuntimeError('Native primitive did not pass the unchanged CKKS check')
                record['status'] = 'ok'
    except BaseException as error:
        record.update(status='error', error=type(error).__name__ + ': ' + str(error))
    finally:
        if (out / 'native-result.json').is_file():
            try:
                record['native_result'] = json.loads((out / 'native-result.json').read_text())
            except (OSError, ValueError) as error:
                record['native_record_error'] = type(error).__name__ + ': ' + str(error)
                record['status'] = 'error'
                if record.get('error') is None:
                    record['error'] = 'Native checkpoint could not be read; raw partial record retained'
        native = record.get('native_result')
        record['observed_native_counts'] = {
            key: native.get(key) if native is not None else
            0 if record['attempts']['execute'] == 0 else None
            for key in ('keygen_runs', 'residual_keypair_attempts', 'residual_keypair_completed',
                        'bootstrap_keys_attempts', 'bootstrap_keys_completed',
                        'evaluator_attempts', 'evaluator_completed', 'dft_matrices_constructed',
                        'encode_attempts', 'encodes', 'encrypt_attempts', 'encryptions', 'bootstrap_attempts', 'bootstraps',
                        'decrypt_attempts', 'decryptions')}
        write_json(out / 'timing.json', summarize_events(read_events(out / 'events.jsonl')))
        record['ts'] = datetime.now(timezone.utc).isoformat()
        finalize_size(out, record)
        entry = dict(run_id=out.name, round=record['round'], model=record['model'], mode=record['mode'],
                     git_rev='working-tree', metrics={key: record[key] for key in ('status', 'compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance')},
                     artifacts_dir=str(out), ts=record['ts'])
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
            if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
                raise ValueError('Existing result identity')
            stream.seek(0, 2); stream.write(json.dumps(entry) + '\n'); stream.flush(); fcntl.flock(stream, fcntl.LOCK_UN)
    print(json.dumps({key: record[key] for key in ('status', 'error', 'attempts', 'completed')}))
    return 0 if record['status'] in ('ok', 'prepared') else 1


if __name__ == '__main__':
    sys.exit(main())
