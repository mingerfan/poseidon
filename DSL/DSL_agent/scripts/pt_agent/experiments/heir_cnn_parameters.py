"""Construct the precise native HEIR/Lattigo parameter literals without keys."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

from .. import paths
from .r10_paper_metrics import StageRecorder,read_events,run_process,summarize_events,write_json
from .heir_cnn_preflight import finalize_size,native_artifact_metadata


def source_literals(path):
    text=Path(path).read_text()
    records=native_artifact_metadata(path)['parameters']
    if len(records)!=1:raise ValueError('One original native residual literal is required')
    values=re.findall(r'btParamsLiteral = <([^>]*)>',text)
    if len(values)!=1:raise ValueError('One original native bootstrap literal is required')
    match=re.fullmatch(r'logN = (\d+)',values[0])
    if not match:raise ValueError('Additional native bootstrap options require explicit preservation')
    return records[0],dict(logN=int(match.group(1)))


def go_source(residual,bootstrap,*,candidate_logn=None,profile_bootstrap_keys=False):
    vector=lambda values:','.join(str(i) for i in values)
    source=r'''package main
import (
 "encoding/json"
 "fmt"
 "os"
 "runtime"
 "runtime/debug"
 "syscall"
 "time"
 "github.com/tuneinsight/lattigo/v6/schemes/ckks"
 "github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
)
func summary(p ckks.Parameters) map[string]any {
 polyBytes:=int64(p.N())*int64(len(p.Q())+len(p.P()))*8
 return map[string]any{"logN":p.LogN(),"N":p.N(),"max_slots":p.MaxSlots(),
 "Q":p.Q(),"P":p.P(),"Q_bits":p.LogQi(),"P_bits":p.LogPi(),
 "logQ":p.LogQ(),"logP":p.LogP(),"logQP":p.LogQP(),"log_default_scale":p.LogDefaultScale(),
 "ring_type":fmt.Sprint(p.RingType()),"Xs":p.Xs(),"Xs_type":fmt.Sprintf("%T",p.Xs()),
 "Xs_hamming_weight":p.XsHammingWeight(),"Xe":p.Xe(),"Xe_type":fmt.Sprintf("%T",p.Xe()),
 "one_QP_rns_polynomial_payload_bytes":polyBytes,"two_component_QP_element_payload_bytes":2*polyBytes,
 "two_component_Q_only_ciphertext_payload_bytes":int64(p.N())*int64(len(p.Q()))*16,
 "storage_estimate_scope":"RNS coefficient payload only; excludes evaluation keys, DFT tables and runtime objects"}
}
func main(){
 runtime.GOMAXPROCS(1);debug.SetMemoryLimit(384<<20)
 result:=map[string]any{"status":"running","keygen_runs":0,"dft_matrices_constructed":0,
 "encryptions":0,"bootstraps":0,"decryptions":0,"model_forwards":0,"security_validated":false,
 "residual_constructor_attempts":0,"residual_constructor_completed":0,
 "bootstrap_constructor_attempts":0,"bootstrap_constructor_completed":0}
 write:=func(){f,e:=os.Create(os.Args[1]);if e!=nil{panic(e)};defer f.Close();enc:=json.NewEncoder(f);enc.SetIndent("","  ");if e=enc.Encode(result);e!=nil{panic(e)}}
 defer func(){var usage syscall.Rusage;syscall.Getrusage(syscall.RUSAGE_SELF,&usage);result["ru_maxrss_kib"]=usage.Maxrss;write()}()
 result["residual_constructor_attempts"]=1;write();start:=time.Now()
 p,e:=ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN:RESIDUAL_LOGN,Q:[]uint64{RESIDUAL_Q},P:[]uint64{RESIDUAL_P},LogDefaultScale:RESIDUAL_SCALE})
 result["residual_constructor_seconds"]=time.Since(start).Seconds()
 if e!=nil{result["status"]="error";result["error"]=e.Error();return}
 result["residual_constructor_completed"]=1;result["residual"]=summary(p);write()
 logN:=BOOTSTRAP_LOGN;literal:=bootstrapping.ParametersLiteral{LogN:&logN}
 result["bootstrap_constructor_attempts"]=1;write();start=time.Now()
 b,e:=bootstrapping.NewParametersFromLiteral(p,literal)
 result["bootstrap_constructor_seconds"]=time.Since(start).Seconds()
 if e!=nil{result["status"]="error";result["error"]=e.Error();return}
 result["bootstrap_constructor_completed"]=1;result["bootstrap"]=summary(b.BootstrappingParameters)
 result["bootstrap_residual"]=summary(b.ResidualParameters);result["ephemeral_secret_weight"]=b.EphemeralSecretWeight
 result["C2S_depth"]=b.DepthCoeffsToSlots();result["EvalMod_depth"]=b.DepthEvalMod();result["S2C_depth"]=b.DepthSlotsToCoeffs();result["bootstrap_depth"]=b.Depth()
 result["bootstrap_public_parameters"]=b;result["status"]="ok"
}
'''
    if profile_bootstrap_keys:
        source=source.replace('result["bootstrap_public_parameters"]=b;result["status"]="ok"',r'''
 result["bootstrap_public_parameters"]=b
 required:=append(b.GaloisElements(b.BootstrappingParameters),b.BootstrappingParameters.GaloisElementForComplexConjugation())
 unique:=map[uint64]bool{};for _,g:=range required{unique[g]=true}
 qp:=b.BootstrappingParameters
 qcount,pcount:=len(qp.Q()),len(qp.P());digits:=(qcount+pcount-1)/pcount
 perKey:=int64(2)*int64(digits)*int64(qcount+pcount)*int64(qp.N())*8
 result["bootstrap_key_metadata"]=map[string]any{
 "required_galois_elements":required,"unique_galois_elements":len(unique),
 "RNS_decomposition_digits":digits,"one_uncompressed_evaluation_key_coefficient_bytes":perKey,
 "galois_plus_relinearization_key_coefficient_bytes":perKey*int64(len(unique)+1),
 "scope":"Parameter getters and coefficient-storage formula only; excludes encapsulation keys, DFT tables and working buffers",
 "actual_keys_generated":0,"actual_DFT_tables_constructed":0}
 result["status"]="ok"''')
    if candidate_logn is not None:
        source=source.replace('LogN:RESIDUAL_LOGN,Q:[]uint64{RESIDUAL_Q},P:[]uint64{RESIDUAL_P}',
                              'LogN:RESIDUAL_LOGN,LogQ:[]int{RESIDUAL_Q},LogP:[]int{RESIDUAL_P}')
        # Lattigo's LogQi/LogPi policy rounds log2 of primes close to powers of
        # two; integer bit_length would add a bit for primes just above 2**b.
        q=[round(math.log2(v)) for v in residual['Q']];p=[round(math.log2(v)) for v in residual['P']]
        logn=boot_logn=candidate_logn
    else:q,p,logn,boot_logn=residual['Q'],residual['P'],residual['logN'],bootstrap['logN']
    return source.replace('RESIDUAL_LOGN',str(logn)).replace('RESIDUAL_Q',vector(q)).replace('RESIDUAL_P',vector(p)).replace('RESIDUAL_SCALE',str(residual['log_default_scale'])).replace('BOOTSTRAP_LOGN',str(boot_logn))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--go',type=Path,default=paths.WORKSPACE/'baselines/heir-r20-go-backend-20261115/toolchain/go/bin/go')
    p.add_argument('--cache-source',type=Path,default=paths.RESULTS_DIR/'heir-single-ckks-r21-20261115/work')
    p.add_argument('--candidate-logn',type=int,choices=(16,17),help='Explicit candidate using the saved native bit-width policy and Lattigo prime generation; not an HEIR-emitted literal')
    p.add_argument('--profile-bootstrap-keys',action='store_true',help='Read public Galois-element requirements and storage formulas; no keys or DFT matrices')
    a=p.parse_args(argv)
    residual,bootstrap=source_literals(a.input)
    out=a.out.absolute();out.mkdir(parents=True,exist_ok=False)
    for name in ('module','tmp','telemetry'):(out/name).mkdir()
    (out/'telemetry/mode').write_text('off\n')
    (out/'module/main.go').write_text(go_source(residual,bootstrap,candidate_logn=a.candidate_logn,
                                              profile_bootstrap_keys=a.profile_bootstrap_keys))
    compiler_record=a.input.parent/'result.json'
    compiler_source=json.loads(compiler_record.read_text()) if compiler_record.is_file() else {}
    result=dict(round='HEIR-CNN-PARAMS',mode='native-heir-parameter-construction',
                model=compiler_source.get('model','native-parameter-config'),
                status='running',error=None,source_mlir=str(a.input.absolute()),source_residual_literal=residual,
                source_bootstrap_literal=bootstrap,processes=[],attempts=dict(build=0,construct=0),completed=dict(build=0,construct=0),
                scope='Actual residual and complete bootstrap parameter constructors only; no key or DFT generation',
                actual_parameters=None,security_validated=False,formal_matrix_complete=False,
                keygen_runs=0,real_ckks_attempts=0,real_ckks_completed=0,llm_calls=0,model_forwards=0,
                downloads=0,installations=0,extra_integrity_checks=0,native_go_cached_module_mechanism=True,
                telemetry_directory=str(out/'telemetry'),telemetry_mode='off',artifact_bytes=None,
                compiled_bootstraps=None,predicted_latency=None,CKKS_latency=None,accuracy_acceptance=None)
    result['bootstrap_key_metadata_requested']=a.profile_bootstrap_keys
    result['candidate_logn']=a.candidate_logn
    result['candidate_modulus_bit_policy']=(dict(LogQ=[round(math.log2(v)) for v in residual['Q']],
                                                LogP=[round(math.log2(v)) for v in residual['P']])
                                            if a.candidate_logn is not None else None)
    result['native_emitted_literal_reused_exactly']=a.candidate_logn is None
    result['parameter_request_origin']=('native HEIR emitted residual and bootstrap literals' if a.candidate_logn is None else
                                       'explicit candidate LogN; native saved modulus bit-width policy; Lattigo prime constructor; HEIR compile not yet passed')
    result['secret_sampler_source_audit']=dict(keys_constructed=False,actual_main_secret_hamming_weight=None,
        residual_future_sampler='actual residual parameters Xs, via native gen_key_pair',
        same_N_bootstrap_main_secret='extends the original residual secret into Q/P; no bootstrap H192 sampler call',
        different_N_bootstrap_main_secret='Lattigo GenSecretKeyNew using bootstrap parameters Xs',
        encapsulation_secret='GenSecretKeyWithHammingWeightNew(EphemeralSecretWeight)',
        source='Lattigo6.1.0 circuits/ckks/bootstrapping/keys.go:68-115,130-140; original HEIR configure interface')
    events=StageRecorder(out/'events.jsonl')
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV') and not k.startswith(('LD_','NIX_','__NIX_','GO'))}
    env.update(GOTOOLCHAIN='local',GOPROXY='off',GOSUMDB='off',GOPATH=str(out/'gopath'),
               GOMODCACHE=str(out/'module-cache'),GOCACHE=str(out/'build-cache'),GOTMPDIR=str(out/'tmp'),TMPDIR=str(out/'tmp'),
               TEST_TELEMETRY_DIR=str(out/'telemetry'),CGO_ENABLED='0',GOMEMLIMIT='384MiB',GOMAXPROCS='1')
    def run(stage,argv,seconds):
        if shutil.disk_usage(out).free<10*1024**3:raise RuntimeError('Preserve 10GiB free disk reserve')
        result['attempts'][stage]+=1
        with events.span(stage,argv=argv):
            obs=run_process(argv,cwd=out/'module',log=out/(stage+'.log'),resources=out/(stage+'-resources.jsonl'),
                            timeout=seconds,mem_gib=2,env=env)
        result['processes'].append(dict(stage=stage,argv=argv,observation=obs))
        write_json(out/'result.json',result)
        if obs['exit_code']!=0 or obs['timed_out'] or obs['signal'] is not None or obs['remaining_pids']:
            raise RuntimeError(stage+' failed; partial evidence retained')
        result['completed'][stage]+=1
    try:
        with events.span('native-parameter-construction'):
            with events.span('private-cached-module-copy'):
                for name in ('go.mod','go.sum'):shutil.copyfile(a.cache_source/'module'/name,out/'module'/name)
                shutil.copytree(a.cache_source/'module-cache',out/'module-cache')
                shutil.copytree(a.cache_source/'build-cache',out/'build-cache')
            run('build',[str(a.go),'build','-p=1','-mod=readonly','-buildvcs=false','-o',str(out/'constructor'),'.'],300)
            run('construct',[str(out/'constructor'),str(out/'parameters.json')],60)
            result['actual_parameters']=json.loads((out/'parameters.json').read_text())
            if result['actual_parameters'].get('status')!='ok':raise RuntimeError('Native parameter constructor returned an error')
            if result['actual_parameters'].get('ru_maxrss_kib',0)>=512*1024:raise RuntimeError('Constructor exceeded the 0.5GiB physical-memory goal')
            result['status']='ok'
    except BaseException as exc:
        result['status']='error';result['error']=type(exc).__name__+': '+str(exc)
    finally:
        write_json(out/'timing.json',summarize_events(read_events(out/'events.jsonl')))
        result['ts']=datetime.now(timezone.utc).isoformat();finalize_size(out,result)
        entry=dict(run_id=out.name,round=result['round'],model=result['model'],mode=result['mode'],git_rev='working-tree',
                   metrics={k:result[k] for k in ('status','security_validated','compiled_bootstraps','predicted_latency','CKKS_latency','accuracy_acceptance')},
                   artifacts_dir=str(out),ts=result['ts'])
        with (paths.RESULTS_DIR/'registry.jsonl').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            if any(json.loads(l).get('run_id')==out.name for l in f if l.strip()):raise ValueError('Existing result identity')
            f.seek(0,2);f.write(json.dumps(entry)+'\n');f.flush();fcntl.flock(f,fcntl.LOCK_UN)
    print(json.dumps({k:result[k] for k in ('status','error','artifact_bytes')}))
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':sys.exit(main())
