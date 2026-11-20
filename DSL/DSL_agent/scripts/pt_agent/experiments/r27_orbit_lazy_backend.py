"""Isolated Lattigo key-storage adaptation and finite native bootstrap probe.

prepare copies and builds only. regression uses a small deterministic unit
fixture. bootstrap-native uses the original N=131072 Orbit parameters and one
saved training input; callers schedule that 14-GiB phase separately.
"""

import argparse
from datetime import datetime, timezone
import difflib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import time

from pt_agent import paths
from pt_agent.experiments.r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from pt_agent.experiments.r27_orbit_native_backend import _environment, _tree_bytes

GIB = 1024 ** 3
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-native-lazy-galois-key-storage'

PATCHES = (
    ('core/rlwe/keys.go', 'type MemEvaluationKeySet struct {\n',
     'type MemEvaluationKeySet struct {\n'
     '\t// R27 isolated runtime storage callbacks; nil preserves the original map behavior.\n'
     '\tGetGaloisKeyFunc func(uint64) (*GaloisKey, error)\n'
     '\tGetGaloisKeysListFunc func() []uint64\n'),
    ('core/rlwe/keys.go', 'func (evk MemEvaluationKeySet) GetGaloisKey(galEl uint64) (gk *GaloisKey, err error) {\n',
     'func (evk MemEvaluationKeySet) GetGaloisKey(galEl uint64) (gk *GaloisKey, err error) {\n'
     '\tif evk.GetGaloisKeyFunc != nil { return evk.GetGaloisKeyFunc(galEl) }\n'),
    ('core/rlwe/keys.go', 'func (evk MemEvaluationKeySet) GetGaloisKeysList() (galEls []uint64) {\n',
     'func (evk MemEvaluationKeySet) GetGaloisKeysList() (galEls []uint64) {\n'
     '\tif evk.GetGaloisKeysListFunc != nil { return evk.GetGaloisKeysListFunc() }\n'),
    ('circuits/ckks/dft/dft.go', '// NewMatrixFromLiteral generates the factorized DFT/IDFT matrices for the homomorphic encoding/decoding.\n',
     '// R27 isolated observer records matrix construction only; nil leaves original behavior.\n'
     'var R27MatrixObserver func(MatrixLiteral, bool)\n\n'
     '// NewMatrixFromLiteral generates the factorized DFT/IDFT matrices for the homomorphic encoding/decoding.\n'),
    ('circuits/ckks/dft/dft.go', 'func NewMatrixFromLiteral(params ckks.Parameters, d MatrixLiteral, encoder *ckks.Encoder) (Matrix, error) {\n',
     'func NewMatrixFromLiteral(params ckks.Parameters, d MatrixLiteral, encoder *ckks.Encoder) (Matrix, error) {\n'
     '\tif observer := R27MatrixObserver; observer != nil { observer(d, true); defer observer(d, false) }\n'),
    ('circuits/ckks/dft/dft.go', '\t"math/big"\n', '\t"math/big"\n\t"runtime"\n'),
    ('circuits/ckks/dft/dft.go', '\tpVecDFT := d.GenMatrices(params.LogN(), params.EncodingPrecision())\n',
     '\tpVecDFT := d.GenMatrices(params.LogN(), params.EncodingPrecision())\n'
     '\t// R27 isolated lifetime management: FFT scratch has no future use.\n'
     '\truntime.GC()\n'),
    ('circuits/ckks/dft/dft.go', '\t\t\tmatrices = append(matrices, mat)\n\t\t\tidx++\n',
     '\t\t\tmatrices = append(matrices, mat)\n'
     '\t\t\t// Encoding finished; remaining factors retain their own live references.\n'
     '\t\t\tpVecDFT[idx] = nil\n\t\t\truntime.GC()\n\t\t\tidx++\n'),
)

GO_CLIENT = r'''package main

import (
    "encoding/binary"
    "encoding/json"
    "flag"
    "fmt"
    "math"
    "math/big"
    "os"
    "runtime"
    "sort"
    "sync"
    "time"

    "github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
    "github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
    ltcommon "github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
    "github.com/tuneinsight/lattigo/v6/core/rlwe"
    "github.com/tuneinsight/lattigo/v6/ring"
    "github.com/tuneinsight/lattigo/v6/schemes/ckks"
    "github.com/tuneinsight/lattigo/v6/utils"
    "github.com/tuneinsight/lattigo/v6/utils/bignum"
)

var origin=time.Now()
var outputPath,eventPath string
var record map[string]any
var eventLock sync.Mutex
var sequence int
var constructorRunning bool
var constructorKeygenNS int64

func save() {
    data,err:=json.MarshalIndent(record,"","  ");if err!=nil{panic(err)}
    temporary:=outputPath+".tmp"
    if err=os.WriteFile(temporary,append(data,'\n'),0600);err!=nil{panic(err)}
    if err=os.Rename(temporary,outputPath);err!=nil{panic(err)}
}
func emit(value map[string]any) {
    eventLock.Lock();defer eventLock.Unlock()
    value["pid"]=os.Getpid();value["elapsed_ns"]=time.Since(origin).Nanoseconds()
    value["clock_domain"]="Go process monotonic elapsed; do not merge absolute Python timestamps"
    file,err:=os.OpenFile(eventPath,os.O_CREATE|os.O_APPEND|os.O_WRONLY,0600);if err!=nil{panic(err)}
    defer file.Close();if err=json.NewEncoder(file).Encode(value);err!=nil{panic(err)}
}
func stage(name string, metadata map[string]any, fn func()error)(err error){
    sequence++;id:=sequence;start:=time.Now()
    record["active_stage"]=name
    emit(map[string]any{"event_id":id,"stage":name,"type":"begin","metadata":metadata});save()
    defer func(){
        if value:=recover();value!=nil{err=fmt.Errorf("panic: %v",value)}
        status:="ok";if err!=nil{status="error";record["failed_stage"]=name}
        emit(map[string]any{"event_id":id,"stage":name,"type":"end","status":status,"duration_ns":time.Since(start).Nanoseconds(),"error":fmt.Sprint(err)})
        save()
    }()
    return fn()
}
func counter(name string){record[name]=record[name].(int)+1}
func keygen(name string,fn func())error{
    counter("keygen_api_attempts")
    start:=time.Now()
    err:=stage(name,nil,func()error{fn();return nil})
    if constructorRunning{constructorKeygenNS+=time.Since(start).Nanoseconds()}
    if err==nil{counter("keygen_api_completed")}
    return err
}

type lazyKeys struct {
    mu sync.Mutex
    params ckks.Parameters
    sk *rlwe.SecretKey
    generator *rlwe.KeyGenerator
    allowed map[uint64]bool
    list []uint64
    current *rlwe.GaloisKey
    name string
    gets,generations,hits,evictions int
}
func newLazy(name string,p ckks.Parameters,sk *rlwe.SecretKey,list []uint64)*lazyKeys{
    allowed:=map[uint64]bool{};for _,g:=range list{allowed[g]=true}
    list=nil;for g:=range allowed{list=append(list,g)};sort.Slice(list,func(i,j int)bool{return list[i]<list[j]})
    return &lazyKeys{params:p,sk:sk,generator:rlwe.NewKeyGenerator(p),allowed:allowed,list:list,name:name}
}
func(l *lazyKeys) get(g uint64)(*rlwe.GaloisKey,error){
    l.mu.Lock();defer l.mu.Unlock();l.gets++
    if !l.allowed[g]{return nil,fmt.Errorf("unlisted Galois element %d",g)}
    if l.current!=nil&&l.current.GaloisElement==g{l.hits++;return l.current,nil}
    if l.current!=nil{l.current=nil;l.evictions++;runtime.GC()}
    var generated *rlwe.GaloisKey
    err:=keygen(l.name+".galois",func(){generated=l.generator.GenGaloisKeyNew(g,l.sk)})
    if err!=nil{return nil,err}
    l.current=generated;l.generations++
    record[l.name+"_cache"]=l.summaryLocked();save()
    return generated,nil
}
func(l *lazyKeys) keys()[]uint64{l.mu.Lock();defer l.mu.Unlock();return append([]uint64{},l.list...)}
func(l *lazyKeys) summaryLocked()map[string]any{return map[string]any{"get_calls":l.gets,"generated":l.generations,"hits":l.hits,
    "evictions":l.evictions,"capacity":1,"resident_keys":btoi(l.current!=nil),"allowed":len(l.list),"serialized_keys":0}}
func(l *lazyKeys) summary()map[string]any{l.mu.Lock();defer l.mu.Unlock();return l.summaryLocked()}
func(l *lazyKeys) attach(mem *rlwe.MemEvaluationKeySet){mem.GetGaloisKeyFunc=l.get;mem.GetGaloisKeysListFunc=l.keys}
func btoi(value bool)int{if value{return 1};return 0}

func compare(want,got []float64) (map[string]any,error){
    if len(want)!=len(got){return nil,fmt.Errorf("output length mismatch")}
    maximum,ratio:=0.0,0.0;passed:=true
    for i,x:=range want{
        if math.IsNaN(got[i])||math.IsInf(got[i],0)||math.IsNaN(x)||math.IsInf(x,0){return nil,fmt.Errorf("nonfinite result at %d",i)}
        delta:=math.Abs(got[i]-x);bound:=1e-5+1e-4*math.Abs(x)
        maximum=math.Max(maximum,delta);ratio=math.Max(ratio,delta/bound);passed=passed&&delta<=bound
    }
    return map[string]any{"passed":passed,"finite":true,"elements":len(want),"max_absolute_error":maximum,
        "max_error_ratio":ratio,"absolute_tolerance":1e-5,"relative_tolerance":1e-4},nil
}
func finalDecode(name string,p ckks.Parameters,sk *rlwe.SecretKey,ct *rlwe.Ciphertext,want []float64)error{
    counter("final_decryption_attempts")
    return stage(name,nil,func()error{
        pt:=rlwe.NewDecryptor(p,sk).DecryptNew(ct);counter("final_decryption_completed")
        got:=make([]float64,len(want));if err:=ckks.NewEncoder(p).Decode(pt,got);err!=nil{return err}
        comparison,err:=compare(want,got);if err!=nil{return err};record[name+"_comparison"]=comparison
        record[name+"_first_values"]=got[:min(10,len(got))]
        if comparison["passed"]!=true{return fmt.Errorf("original CKKS tolerance failed")};return nil
    })
}

func regression()error{
    p,err:=ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN:10,LogQ:[]int{55,51,51},LogP:[]int{61},LogDefaultScale:51})
    if err!=nil{return err};record["parameters"]=p;record["fixture_scope"]="small-N deterministic evaluation-key regression only; original model parameters preserved separately"
    kg:=rlwe.NewKeyGenerator(p);var sk *rlwe.SecretKey;var pk *rlwe.PublicKey;var rlk *rlwe.RelinearizationKey
    if err=keygen("regression.secret_key",func(){sk=kg.GenSecretKeyNew()});err!=nil{return err}
    if err=keygen("regression.public_key",func(){pk=kg.GenPublicKeyNew(sk)});err!=nil{return err}
    if err=keygen("regression.relinearization_key",func(){rlk=kg.GenRelinearizationKeyNew(sk)});err!=nil{return err}
    rotations:=[]int{1,-1,2};gals:=make([]uint64,len(rotations));for i,r:=range rotations{gals[i]=p.GaloisElement(r)}
    eager:=rlwe.NewMemEvaluationKeySet(rlk)
    for _,g:=range gals{var key *rlwe.GaloisKey;if err=keygen("regression.eager_galois",func(){key=kg.GenGaloisKeyNew(g,sk)});err!=nil{return err};eager.GaloisKeys[g]=key}
    lazy:=newLazy("regression_lazy",p,sk,gals);mem:=rlwe.NewMemEvaluationKeySet(rlk);lazy.attach(mem)
    values:=make([]float64,p.MaxSlots());for i:=range values{values[i]=(float64(i%17)-8)/32}
    want:=make([]float64,len(values));for i:=range want{v:=values[(i+3)%len(values)];want[i]=v*v}
    pt:=ckks.NewPlaintext(p,p.MaxLevel());if err=ckks.NewEncoder(p).Encode(values,pt);err!=nil{return err}
    var ct *rlwe.Ciphertext
    counter("encryption_attempts")
    if err=stage("regression.encrypt",nil,func()(e error){ct,e=rlwe.NewEncryptor(p,pk).EncryptNew(pt);if e==nil{counter("encryption_completed")};return});err!=nil{return err}
    for _,entry:=range []struct{name string;keyset *rlwe.MemEvaluationKeySet}{{"eager",eager},{"lazy",mem}}{
        var result *rlwe.Ciphertext
        if err=stage("regression."+entry.name+".encrypted_chain",nil,func()error{
            eval:=ckks.NewEvaluator(p,entry.keyset);result=ct.CopyNew()
            for _,r:=range []int{1,-1,2,1}{if err:=eval.Rotate(result,r,result);err!=nil{return err}}
            if err:=eval.MulRelin(result,result,result);err!=nil{return err};return eval.Rescale(result,result)
        });err!=nil{return err}
        if err=finalDecode("regression."+entry.name+".final_decrypt",p,sk,result,want);err!=nil{return err}
    }
    record["regression_lazy_cache"]=lazy.summary()
    if lazy.generations!=4||lazy.evictions!=3||len(eager.GaloisKeys)!=3{return fmt.Errorf("cache regression counts differ")}
    // The default nil-callback map path and callback path both executed real keys.
    if err=stage("regression.dft_lifetime",nil,func()error{return dftLifetimeRegression(p)});err!=nil{return err}
    record["default_map_regression_passed"]=true;record["lazy_key_regression_passed"]=true
    return nil
}

// The reference body is copied from the untouched original DFT constructor at
// prepare time. Decode plaintext QP coefficients numerically after a forced GC;
// no secret key or ciphertext decryption enters this matrix-lifetime check.
func dftLifetimeRegression(p ckks.Parameters)error{
    maximum,ratio:=0.0,0.0;elements:=0
    for _,kind:=range []dft.Type{dft.HomomorphicEncode,dft.HomomorphicDecode}{
        literal:=dft.MatrixLiteral{Type:kind,LogSlots:p.LogMaxSlots(),LevelQ:p.MaxLevel(),LevelP:p.PCount()-1,
            Levels:[]int{1,1,1},Format:dft.RepackImagAsReal,LogBSGSRatio:1}
        reference,err:=originalDFTReference(p,literal,ckks.NewEncoder(p));if err!=nil{return err}
        actual,err:=dft.NewMatrixFromLiteral(p,literal,ckks.NewEncoder(p));if err!=nil{return err}
        runtime.GC()
        if len(reference.Matrices)!=len(actual.Matrices){return fmt.Errorf("DFT factor count differs")}
        for i,a:=range actual.Matrices{
            b:=reference.Matrices[i]
            if len(a.Vec)!=len(b.Vec)||a.LevelQ!=b.LevelQ||a.LevelP!=b.LevelP||a.N1!=b.N1{return fmt.Errorf("DFT metadata differs")}
            encoder:=ckks.NewEncoder(p)
            for diagonal,poly:=range a.Vec{
                expected,ok:=b.Vec[diagonal];if !ok{return fmt.Errorf("DFT diagonal missing")}
                decode:=func(value ring.Poly,scale rlwe.Scale)([]complex128,error){
                    pt:=ckks.NewPlaintext(p,a.LevelQ);pt.Scale=scale
                    p.RingQ().AtLevel(a.LevelQ).IMForm(value,pt.Value)
                    slots:=make([]complex128,p.MaxSlots());err:=encoder.Decode(pt,slots);return slots,err
                }
                x,err:=decode(poly.Q,a.Scale);if err!=nil{return err};y,err:=decode(expected.Q,b.Scale);if err!=nil{return err}
                for j:=range x{
                    for _,pair:=range [][2]float64{{real(x[j]),real(y[j])},{imag(x[j]),imag(y[j])}}{
                        if math.IsNaN(pair[0])||math.IsInf(pair[0],0)||math.IsNaN(pair[1])||math.IsInf(pair[1],0){return fmt.Errorf("nonfinite DFT matrix value")}
                        delta:=math.Abs(pair[0]-pair[1]);bound:=1e-12+1e-12*math.Abs(pair[1])
                        maximum=math.Max(maximum,delta);ratio=math.Max(ratio,delta/bound);elements++
                        if delta>bound{return fmt.Errorf("DFT plaintext lifetime mathematical comparison failed")}
                    }
                }
            }
        }
    }
    record["dft_lifetime_regression"]=map[string]any{"passed":true,"elements":elements,"max_absolute_error":maximum,
        "max_error_ratio":ratio,"absolute_tolerance":1e-12,"relative_tolerance":1e-12,
        "scope":"two small-N DFT constructors, after forced GC, plaintext coefficients decoded numerically; no integrity check"}
    return nil
}

func readInput(path string)([]float64,error){
    data,err:=os.ReadFile(path);if err!=nil{return nil,err};if len(data)!=65536*8{return nil,fmt.Errorf("native input byte count mismatch")}
    values:=make([]float64,65536)
    for i:=range values{values[i]=math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]));if math.IsNaN(values[i])||math.IsInf(values[i],0){return nil,fmt.Errorf("nonfinite original input")}}
    return values,nil
}

func nativeBootstrap(input string)error{
    logQ:=append([]int{55},make([]int,16)...);for i:=1;i<=16;i++{logQ[i]=51}
    var p ckks.Parameters;var bp bootstrapping.Parameters;var err error
    if err=stage("native.parameter_constructors",nil,func()error{
        p,err=ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN:17,LogQ:logQ,LogP:[]int{61,61,61},LogDefaultScale:51});if err!=nil{return err}
        bp,err=bootstrapping.NewParametersFromLiteral(p,bootstrapping.ParametersLiteral{LogN:utils.Pointy(17),LogP:[]int{61,61,61,61}})
        return err
    });err!=nil{return err}
    record["parameters"]=p;record["bootstrap_parameters"]=bp
    record["original_native_parameter_group"]=true
    values,err:=readInput(input);if err!=nil{return err}
    record["input_elements"]=len(values);record["input_scope"]="R13 train:0, original normalized RGB, flatten/32, zero-pad to4096, tile16"
    kg:=rlwe.NewKeyGenerator(p);var sk *rlwe.SecretKey;var pk *rlwe.PublicKey
    if err=keygen("native.secret_key",func(){sk=kg.GenSecretKeyNew()});err!=nil{return err}
    if err=keygen("native.public_key",func(){pk=kg.GenPublicKeyNew(sk)});err!=nil{return err}
    bigp:=bp.BootstrappingParameters;kgbig:=rlwe.NewKeyGenerator(bigp);var extended *rlwe.SecretKey
    if err=stage("native.extend_same_secret",nil,func()error{
        extended=rlwe.NewSecretKey(bigp);q,aux:=bigp.RingQ(),bigp.RingP();buffer:=q.NewPoly()
        rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(q,q,sk.Value.Q,buffer,extended.Value.Q)
        rlwe.ExtendBasisSmallNormAndCenterNTTMontgomery(q,aux,sk.Value.Q,buffer,extended.Value.P);return nil
    });err!=nil{return err}
    sparse,err:=rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{LogN:bigp.LogN(),Q:bigp.Q()[:1],P:bigp.P()[:1]});if err!=nil{return err}
    kgsparse:=rlwe.NewKeyGenerator(sparse);var sparseSecret *rlwe.SecretKey
    if err=keygen("native.ephemeral_secret",func(){sparseSecret=kgsparse.GenSecretKeyWithHammingWeightNew(bp.EphemeralSecretWeight)});err!=nil{return err}
    var denseToSparse,sparseToDense *rlwe.EvaluationKey;var rlk *rlwe.RelinearizationKey
    if err=keygen("native.dense_to_sparse",func(){denseToSparse=kgsparse.GenEvaluationKeyNew(extended,sparseSecret)});err!=nil{return err}
    if err=keygen("native.sparse_to_dense",func(){sparseToDense=kgbig.GenEvaluationKeyNew(sparseSecret,extended)});err!=nil{return err}
    if err=keygen("native.bootstrap_relinearization_key",func(){rlk=kgbig.GenRelinearizationKeyNew(extended)});err!=nil{return err}
    lazy:=newLazy("native_bootstrap_lazy",bigp,extended,bp.GaloisElements(bigp))
    mem:=rlwe.NewMemEvaluationKeySet(rlk);lazy.attach(mem)
    keys:=&bootstrapping.EvaluationKeys{EvkDenseToSparse:denseToSparse,EvkSparseToDense:sparseToDense,MemEvaluationKeySet:mem}
    var evaluator *bootstrapping.Evaluator
    var dftStart time.Time;var dftID int;var dftNS int64
    dft.R27MatrixObserver=func(literal dft.MatrixLiteral,begin bool){
        if begin{dftStart=time.Now();sequence++;dftID=sequence;emit(map[string]any{"event_id":dftID,"stage":"native.dft_matrix","type":"begin","literal":literal})
        }else{duration:=time.Since(dftStart).Nanoseconds();dftNS+=duration;emit(map[string]any{"event_id":dftID,"stage":"native.dft_matrix","type":"end","duration_ns":duration});counter("dft_matrices_returned");save()}
    }
    defer func(){dft.R27MatrixObserver=nil}()
    constructorRunning=true;constructorStart:=time.Now()
    counter("constructor_attempts")
    err=stage("native.bootstrap_evaluator_constructor",nil,func()(e error){evaluator,e=bootstrapping.NewEvaluator(bp,keys);if e==nil{counter("constructor_completed")};return})
    constructorRunning=false
    constructorNS:=time.Since(constructorStart).Nanoseconds()
    record["constructor_attribution"]=map[string]int64{"inclusive_ns":constructorNS,"dft_matrix_ns":dftNS,
        "real_key_generation_ns":constructorKeygenNS,"remaining_ns":constructorNS-dftNS-constructorKeygenNS}
    record["constructor_cache_summary"]=lazy.summary();save()
    if err!=nil{return err}
    record["constructor_checkkeys_real_generations"]=lazy.generations
    if lazy.generations!=54{return fmt.Errorf("original checkKeys did not generate all 54 real Galois keys")}
    runtime.GC()
    var ciphertext *rlwe.Ciphertext
    counter("encryption_attempts")
    if err=stage("native.encode_encrypt",nil,func()error{
        pt:=ckks.NewPlaintext(p,3);pt.Scale=p.DefaultScale();if e:=ckks.NewEncoder(p).Encode(values,pt);e!=nil{return e}
        var e error;ciphertext,e=rlwe.NewEncryptor(p,pk).EncryptNew(pt);if e==nil{counter("encryption_completed")};return e
    });err!=nil{return err}
    record["bootstrap_input_level"]=ciphertext.Level();record["bootstrap_input_scale"]=ciphertext.Scale.Log2()
    counter("bootstrap_attempts")
    if err=stage("native.bootstrap",map[string]any{"cold_lazy_key_generation_included":true},func()(e error){
        ciphertext,e=evaluator.Bootstrap(ciphertext);if e==nil{counter("bootstrap_completed")};return
    });err!=nil{return err}
    record["bootstrap_output_level"]=ciphertext.Level();record["bootstrap_output_scale"]=ciphertext.Scale.Log2()
    record["native_bootstrap_lazy_cache"]=lazy.summary()
    if err=finalDecode("native.final_decrypt",p,sk,ciphertext,values);err!=nil{return err}
    record["native_bootstrap_passed"]=true
    return nil
}

func main(){
    mode:=flag.String("mode","regression","regression or bootstrap-native")
    output:=flag.String("output","","terminal JSON")
    events:=flag.String("events","","Go event JSONL")
    input:=flag.String("input","","saved packed input, native mode only")
    flag.Parse();outputPath,eventPath=*output,*events
    record=map[string]any{"round":"R27-lazy","mode":*mode,"status":"running","started_unix_ns":origin.UnixNano(),
        "active_stage":nil,"failed_stage":nil,"error":nil,"get_log_enabled":false,"intermediate_decryptions":0,
        "constructor_attempts":0,"constructor_completed":0,"dft_matrices_returned":0,
        "keygen_api_attempts":0,"keygen_api_completed":0,"encryption_attempts":0,"encryption_completed":0,
        "final_decryption_attempts":0,"final_decryption_completed":0,"bootstrap_attempts":0,"bootstrap_completed":0,
        "default_map_regression_passed":false,"lazy_key_regression_passed":false,"native_bootstrap_passed":false,
        "resident_native_keys_payload_obstruction_bytes":int64(41943040000),"security_acceptance":nil,"accuracy_acceptance":nil,
        "full_model_executed":false,"warm_bootstrap_latency_ns":nil,"extra_integrity_checks":0}
    exit:=0
    defer func(){
        if value:=recover();value!=nil{record["status"]="error";record["error"]=fmt.Sprint(value);exit=1}
        record["elapsed_ns"]=time.Since(origin).Nanoseconds();var m runtime.MemStats;runtime.ReadMemStats(&m)
        record["go_memstats"]=map[string]uint64{"heap_alloc":m.HeapAlloc,"heap_sys":m.HeapSys,"sys":m.Sys,"total_alloc":m.TotalAlloc}
        save();if exit!=0{os.Exit(exit)}
    }()
    if outputPath==""||eventPath==""{panic("explicit result/event paths required")}
    var err error
    switch *mode{case "regression":err=regression();case "bootstrap-native":err=nativeBootstrap(*input);default:err=fmt.Errorf("unknown mode")}
    if err!=nil{record["status"]="error";record["error"]=err.Error();exit=1}else{record["status"]="ok";record["active_stage"]=nil}
}
'''

INPUT_EXPORT = r'''import json,sys
from pathlib import Path
import numpy as np
source,destination,metadata=map(Path,sys.argv[1:])
source_metadata=json.loads(source.with_name('training_probe_source.json').read_text())
if (source_metadata.get('split')!='train' or source_metadata.get('indices')!=[0,1,2]
        or source_metadata.get('sample_ids')!=['train:0','train:1','train:2']
        or source_metadata.get('normalization')!={'mean':[.485,.456,.406],'std':[.229,.224,.225]}):
    raise RuntimeError('Original saved training source differs')
with np.load(source,allow_pickle=False) as saved:
    if set(saved.files)!={'input','labels','indices'}:raise RuntimeError('Unexpected saved R13 probes')
    images=saved['input'];indices=saved['indices'];labels=saved['labels']
    if (images.dtype!=np.dtype('float64') or images.shape!=(3,3,32,32) or indices.tolist()!=[0,1,2]
            or labels.tolist()!=source_metadata['labels']):raise RuntimeError('Original probe identity differs')
    image=np.array(images[0],copy=True)
    if not np.isfinite(image).all():raise RuntimeError('Nonfinite original probe')
    vector=np.zeros(4096,dtype='<f8');vector[:3072]=image.reshape(-1)/32
    np.tile(vector,16).tofile(destination)
    metadata.write_text(json.dumps(dict(source=str(source),sample_id='train:0',label=int(labels[0]),
       source_shape=[3,3,32,32],source_dtype='float64',output_elements=65536,packing='flatten C-order /32, pad4096, tile16',
       numpy=np.__version__,normalization_already_applied=True,source_record=source_metadata),indent=2)+'\n')
'''


def apply_isolated_patch(source, target, patch_dir):
    """Copy an existing source tree, then change only the listed storage/observer sites."""
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', 'fhe_binary', '*.a', '*.so', '__pycache__'))
    patch_dir.mkdir()
    changed = {}
    for relative, before, after in PATCHES:
        path = target / relative
        text = path.read_text()
        if text.count(before) != 1:
            raise RuntimeError('Isolated source anchor differs: ' + relative)
        changed.setdefault(relative, text)
        path.write_text(text.replace(before, after, 1))
    entries = []
    for relative, before in changed.items():
        after = (target / relative).read_text()
        name = relative.replace('/', '__')
        (patch_dir / (name + '.before')).write_text(before)
        difference = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                fromfile='original/' + relative, tofile='isolated/' + relative))
        (patch_dir / (name + '.diff')).write_text(difference)
        entries.append(dict(file=relative, replacements=sum(1 for row in PATCHES if row[0] == relative),
                            difference=str(patch_dir / (name + '.diff'))))
    return entries


def _original_dft_reference(source):
    text = (source / 'circuits/ckks/dft/dft.go').read_text()
    signature = 'func NewMatrixFromLiteral(params ckks.Parameters, d MatrixLiteral, encoder *ckks.Encoder) (Matrix, error) {'
    start = text.index(signature)
    end = text.index('\n}\n', start) + 3
    body = text[start:end]
    body = body.replace(signature, 'func originalDFTReference(params ckks.Parameters, d dft.MatrixLiteral, encoder *ckks.Encoder) (dft.Matrix, error) {', 1)
    return body.replace('RepackImagAsReal', 'dft.RepackImagAsReal').replace('Matrix{', 'dft.Matrix{')


def _latest_rss_sample(path):
    """Read only a bounded log tail; interrupted last JSONL writes are ignored."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            end = stream.tell()
            stream.seek(max(0, end - 65536))
            rows = stream.read().decode(errors='replace').splitlines()
        for line in reversed(rows):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get('processes'), list):
                return dict(sample_ts_ns=row.get('ts_ns'), subtree_rss_bytes=sum(
                    item.get('vmrss_kib', 0) * 1024 for item in row['processes']),
                    processes=[dict(pid=item['pid'], rss_kib=item.get('vmrss_kib')) for item in row['processes']])
    except FileNotFoundError:
        pass
    return None


def _available_ram_bytes():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def execute(args):
    out = args.out.resolve()
    if out.exists() or out.is_symlink():
        raise RuntimeError('A new isolated R27 lazy directory is required')
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    seconds = 1200 if args.phase != 'bootstrap-native' else 7200
    started = time.perf_counter_ns()
    deadline = time.monotonic() + seconds
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(round='R27-lazy', model=MODEL, mode=MODE, phase=args.phase, status='running', error=None,
                  failed_stage=None, git_rev=None, git_branch=None, processes=[], attempted={}, completed={},
                  extra_integrity_checks=0, downloads=0, installations=0, llm_calls=0,
                  original_libraries_modified=False, artifact_bytes=0, resource_monitor_error=None,
                  original_resident_key_payload_bytes=41943040000, native_bootstrap_passed=False,
                  compiled_bootstraps=None, predicted_latency=None, CKKS_latency=None,
                  accuracy_acceptance=None, security_acceptance=None, formal_matrix_complete=False)
    result['memory_policy'] = dict(native_address_space_gib=24, native_subtree_rss_gib=14,
                                 native_go_soft_memory_limit='11GiB', minimum_available_ram_gib=1,
                                 physical_memory_expansion=False, sample_source='R10 child process-tree /proc RSS samples')
    stop = threading.Event()
    active_resource_file = [None]
    def checkpoint():
        if result['resource_monitor_error']:
            raise RuntimeError(result['resource_monitor_error'])
        if time.monotonic() >= deadline:
            raise TimeoutError('R27 lazy phase wall allowance exhausted')
        if _tree_bytes(out) > 4 * GIB or shutil.disk_usage(out).free < 10 * GIB:
            raise RuntimeError('R27 lazy disk allowance exhausted')
    def monitor():
        next_artifact_sample = 0
        while not stop.wait(.1):
            try:
                if time.monotonic() >= next_artifact_sample:
                    with (out / 'artifact-samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(clock_ns=time.perf_counter_ns(), artifact_bytes=_tree_bytes(out), disk_free_bytes=shutil.disk_usage(out).free)) + '\n')
                    checkpoint()
                    next_artifact_sample = time.monotonic() + 1
                if args.phase == 'bootstrap-native' and active_resource_file[0] is not None:
                    observed = _latest_rss_sample(active_resource_file[0])
                    available = _available_ram_bytes()
                    if (observed and observed['subtree_rss_bytes'] > 14 * GIB) or available < GIB:
                        result['memory_guard_observation'] = dict(observed or {}, available_ram_bytes=available,
                            reason='child subtree RSS >14GiB or available host RAM <1GiB')
                        with (out / 'memory-guard-events.jsonl').open('a') as stream:
                            stream.write(json.dumps(result['memory_guard_observation']) + '\n')
                        raise RuntimeError('Native physical RSS/RAM guard stopped the child tree')
            except BaseException as exc:
                result['resource_monitor_error'] = f'{type(exc).__name__}: {exc}'
                os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    def phase(name, argv, timeout, mem=2, cwd=None, child_env=None):
        checkpoint()
        result['failed_stage'] = name
        result['attempted'][name] = result['attempted'].get(name, 0) + 1
        row = dict(stage=name, argv=[str(x) for x in argv], cwd=str(cwd or work), observation=None)
        result['processes'].append(row)
        write_json(out / 'progress.json', result)
        with recorder.span(name, command=row['argv']):
            active_resource_file[0] = out / (name + '-resources.jsonl')
            try:
                observation = run_process(row['argv'], cwd=cwd or work, log=out / (name + '.log'),
                                          resources=active_resource_file[0],
                                          timeout=min(timeout, max(.01, deadline - time.monotonic())),
                                          mem_gib=mem, fsize_gib=8, env=child_env or env)
            finally:
                active_resource_file[0] = None
            row['observation'] = observation
            code = observation['exit_code']
            row['native_termination_signal'] = -code if isinstance(code, int) and code < 0 else None
            write_json(out / 'progress.json', result)
            if code != 0 or observation['timed_out'] or observation['signal'] is not None or observation['remaining_pids']:
                raise RuntimeError('R27 lazy child failed: ' + name)
            result['completed'][name] = result['completed'].get(name, 0) + 1
        checkpoint()
    def interrupt(sig, _):
        raise RuntimeError('R27 lazy outer signal ' + str(sig))
    prior_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        for name in ('tmp', 'telemetry', 'gopath'):
            (work / name).mkdir()
        (work / 'telemetry/mode').write_text('off\n')
        env = _environment(work, args.go_sdk)
        env['GOMEMLIMIT'] = '1500MiB' if args.phase != 'bootstrap-native' else '11GiB'
        shutil.copyfile(Path(__file__), out / 'driver.py')
        watcher.start()
        phase('git-revision', ['git', 'rev-parse', 'HEAD'], 15, cwd=paths.REPO)
        result['git_rev'] = (out / 'git-revision.log').read_text().strip()
        phase('git-branch', ['git', 'branch', '--show-current'], 15, cwd=paths.REPO)
        result['git_branch'] = (out / 'git-branch.log').read_text().strip()
        if args.phase == 'prepare':
            with recorder.span('isolated_copy_and_storage_patch'):
                result['patches'] = apply_isolated_patch(args.lattigo_root, work / 'lattigo', out / 'patches')
                module = work / 'module'
                module.mkdir()
                (module / 'main.go').write_text(GO_CLIENT + '\n' + _original_dft_reference(args.lattigo_root))
                (module / 'go.mod').write_text('module pt-agent-r27-lazy\n\ngo 1.25.0\n\nrequire github.com/tuneinsight/lattigo/v6 v6.0.0\n\nreplace github.com/tuneinsight/lattigo/v6 => ' + str(work / 'lattigo') + '\n')
                shutil.copyfile(args.lattigo_root / 'go.sum', module / 'go.sum')
                shutil.copytree(args.module_cache, work / 'module-cache')
                shutil.copytree(args.build_cache, work / 'build-cache')
            phase('build', [args.go_sdk / 'bin/go', 'build', '-p=1', '-buildvcs=false', '-mod=mod', '-o', work / 'r27-lazy-client', '.'], 900, mem=2, cwd=module)
            result['ready_binary'] = str(work / 'r27-lazy-client')
            result['keygen_api_attempts'] = 0
            result['keygen_api_completed'] = 0
        else:
            prior = json.loads((args.prepared / 'result.json').read_text())
            if prior.get('status') != 'ok' or prior.get('phase') != 'prepare' or prior.get('mode') != MODE:
                raise RuntimeError('A successful isolated prepare result is required')
            binary = Path(prior['ready_binary'])
            if not binary.is_file():
                raise RuntimeError('Prepared client missing')
            result['prepared_result'] = str(args.prepared / 'result.json')
            result['patches'] = prior['patches']
            arguments = [binary, '-mode', args.phase, '-output', out / 'client.json', '-events', out / 'go-events.jsonl']
            if args.phase == 'bootstrap-native':
                export = work / 'export_input.py'
                export.write_text(INPUT_EXPORT)
                phase('saved-training-input-pack', [args.python, '-B', export, args.probes, work / 'train0.bin', out / 'input-source.json'], 60, mem=2)
                arguments += ['-input', work / 'train0.bin']
                result['source_probe'] = str(args.probes)
            phase(args.phase, arguments, 300 if args.phase == 'regression' else 7000,
                  mem=2 if args.phase == 'regression' else 24)
            client = json.loads((out / 'client.json').read_text())
            result['client'] = client
            if client.get('status') != 'ok' or client.get('intermediate_decryptions') != 0 or client.get('get_log_enabled') is not False:
                raise RuntimeError('R27 lazy client contract failed')
            if args.phase == 'regression' and (client.get('lazy_key_regression_passed') is not True or client.get('default_map_regression_passed') is not True):
                raise RuntimeError('R27 lazy regression incomplete')
            if args.phase == 'bootstrap-native' and client.get('native_bootstrap_passed') is not True:
                raise RuntimeError('R27 native bootstrap incomplete')
            result['native_bootstrap_passed'] = client['native_bootstrap_passed']
        result.update(status='ok', failed_stage=None)
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'[:4000])
    finally:
        for sig in prior_handlers:
            signal.signal(sig, signal.SIG_IGN)
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=3)
        if (out / 'client.json').is_file() and 'client' not in result:
            result['client'] = json.loads((out / 'client.json').read_text())
        result['client_terminal_record_observed'] = result.get('client', {}).get('status') in ('ok', 'error')
        result['client_partial_record'] = bool(result.get('client')) and not result['client_terminal_record_observed']
        result['elapsed_ns'] = time.perf_counter_ns() - started
        result['elapsed_scope'] = 'entry through process cleanup; terminal serialization/registry excluded'
        result['timing'] = summarize_events(read_events(out / 'events.jsonl'))
        result['metrics'] = {key: result.get(key) for key in ('status', 'phase', 'attempted', 'completed', 'native_bootstrap_passed',
                             'original_resident_key_payload_bytes', 'extra_integrity_checks', 'downloads', 'installations', 'llm_calls',
                             'compiled_bootstraps', 'predicted_latency', 'CKKS_latency', 'accuracy_acceptance', 'security_acceptance', 'elapsed_ns')}
        result['metrics']['client_counts'] = {key: result.get('client', {}).get(key) for key in (
            'keygen_api_attempts', 'keygen_api_completed', 'encryption_attempts', 'encryption_completed',
            'final_decryption_attempts', 'final_decryption_completed', 'bootstrap_attempts', 'bootstrap_completed')}
        if args.phase == 'prepare':
            result['metrics']['client_counts'] = dict.fromkeys(result['metrics']['client_counts'], 0)
        result['metrics']['client_counts_scope'] = ('completed process terminal counters' if result['client_terminal_record_observed']
                else 'no client executed' if args.phase == 'prepare'
                else 'saved partial counters are observed lower bounds; in-flight completion unknown')
        result['metrics']['client_terminal_record_observed'] = result['client_terminal_record_observed']
        result['metrics']['memory_policy'] = result['memory_policy']
        result['metrics']['memory_guard_observation'] = result.get('memory_guard_observation')
        result['metrics']['intermediate_decryptions'] = result.get('client', {}).get('intermediate_decryptions', 0 if args.phase == 'prepare' else None)
        write_json(out / 'progress.json', result)
        for _ in range(12):
            write_json(out / 'result.json', result)
            size = _tree_bytes(out)
            if size == result['artifact_bytes']:
                break
            result['artifact_bytes'] = size
        row = dict(run_id=out.name, round=result['round'], model=MODEL, mode=MODE, git_rev=result['git_rev'],
                   metrics=result['metrics'], artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
        with (paths.RESULTS_DIR / 'registry.jsonl').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'regression', 'bootstrap-native'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--go-sdk', type=Path, default=paths.ORBIT_ROOT.parent / 'toolchains/go125/go')
    parser.add_argument('--lattigo-root', type=Path, default=paths.ORBIT_ROOT / 'backend/lattigo')
    parser.add_argument('--module-cache', type=Path, default=Path(tempfile.gettempdir()) / 'orbit-backend-20261115/gomodcache')
    parser.add_argument('--build-cache', type=Path, default=Path(tempfile.gettempdir()) / 'orbit-backend-20261115/gocache')
    parser.add_argument('--probes', type=Path, default=paths.RESULTS_DIR / 'r13-common-resnet20-silu-20261115/worker/training_probes.npz')
    parser.add_argument('--python', type=Path, default=paths.ORBIT_ROOT.parent / 'orbit-venv/bin/python')
    args = parser.parse_args(argv)
    if args.phase != 'prepare' and args.prepared is None:
        parser.error('--prepared is required for an execution phase')
    result = execute(args)
    print(json.dumps(dict(status=result['status'], error=result['error'], output=str(args.out), metrics=result['metrics']), ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
