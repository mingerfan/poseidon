"""Isolated Orbit native parameter export; this entry point never generates keys.

The copied upstream package supplies the MLIR reader and its rotation selection.
Only a replacement main is added in the new run directory.  Payload estimates
use the allocation dimensions in RLWE.NewGadgetCiphertext, without allocating
an evaluation key or materializing an encoded DFT matrix.
"""

import argparse
from datetime import datetime, timezone
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
from pt_agent.experiments.r10_paper_metrics import (
    StageRecorder, read_events, run_process, summarize_events, write_json,
)

ROUND = 'R27'
MODEL = 'pt_agent.experiments.r13_common_models:resnet20_silu'
MODE = 'orbit-native-parameters-and-key-payload-estimate'
GIB = 1024 ** 3

GO_MAIN = r'''package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "math/bits"
    "os"
    "runtime"
    "sort"
    "time"

    "github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
    "github.com/tuneinsight/lattigo/v6/core/rlwe"
    "github.com/tuneinsight/lattigo/v6/schemes/ckks"
    "github.com/tuneinsight/lattigo/v6/utils"
)

var processStart = time.Now()
var eventsPath string

func event(stage, kind string, err error) {
    row := map[string]any{"stage":stage,"type":kind,"elapsed_ns":time.Since(processStart).Nanoseconds(),
        "clock_domain":"Go process monotonic relative to processStart","pid":os.Getpid()}
    if err != nil { row["error"] = err.Error() }
    file, openErr := os.OpenFile(eventsPath,os.O_CREATE|os.O_APPEND|os.O_WRONLY,0600)
    if openErr != nil { panic(openErr) }
    defer file.Close()
    if encodeErr := json.NewEncoder(file).Encode(row); encodeErr != nil { panic(encodeErr) }
}

func stage(name string, fn func() error) (err error) {
    event(name,"begin",nil)
    defer func() {
        if recovered := recover(); recovered != nil { err=fmt.Errorf("panic: %v",recovered) }
        event(name,"end",err)
    }()
    return fn()
}

func unique(values []uint64) []uint64 {
    seen:=map[uint64]bool{}
    for _,value:=range values {seen[value]=true}
    answer:=make([]uint64,0,len(seen))
    for value:=range seen {answer=append(answer,value)}
    sort.Slice(answer,func(i,j int)bool{return answer[i]<answer[j]})
    return answer
}

// This is precisely the uint64 coefficient payload of the default degree-one,
// uncompressed evaluation key. Metadata, slice headers and allocator overhead
// are deliberately excluded and explicitly reported by the caller.
func evaluationKeyPayload(p *rlwe.Parameters) map[string]any {
    lq,lp:=p.QCount()-1,p.PCount()-1
    rows:=p.BaseRNSDecompositionVectorSize(lq,lp)
    base:=p.BaseTwoDecompositionVectorSize(lq,lp,0)
    columns:=0
    for i:=0;i<rows;i++ {columns+=base[i]}
    payload:=int64(columns)*2*int64(p.N())*int64(p.QCount()+p.PCount())*8
    return map[string]any{"degree":1,"compressed":false,"base_two_decomposition":0,
        "level_q":lq,"level_p":lp,"rns_rows":rows,"base_two_columns_per_row":base[:rows],
        "vector_qp_count":columns,"uint64_payload_bytes":payload}
}

func payload(p *rlwe.Parameters) int64 {
    return evaluationKeyPayload(p)["uint64_payload_bytes"].(int64)
}

func parameterSummary(p ckks.Parameters) map[string]any {
    qb,pb:=make([]int,p.QCount()),make([]int,p.PCount())
    for i,q:=range p.Q(){qb[i]=bits.Len64(q)}
    for i,q:=range p.P(){pb[i]=bits.Len64(q)}
    return map[string]any{"parameters":p,"n":p.N(),"log_n":p.LogN(),"slots":1<<p.LogMaxSlots(),
        "max_level":p.MaxLevel(),"log_default_scale":p.LogDefaultScale(),"q":p.Q(),"p":p.P(),
        "q_bits":qb,"p_bits":pb,"log_q":p.LogQ(),"log_p":p.LogP(),"log_qp":p.LogQP(),
        "secret_distribution":p.Xs(),"secret_hamming_weight":p.XsHammingWeight(),
        "error_distribution":p.Xe(),"evaluation_key":evaluationKeyPayload(p.GetRLWEParameters())}
}

func execute(mlir string, result map[string]any) error {
    // These literals are copied from the existing Orbit createContext for
    // n=65536,maxLevel=16,bootstrapMinLevel=3,bootstrapMaxLevel=16.
    logQ:=append([]int{55},make([]int,16)...)
    for i:=1;i<=16;i++ {logQ[i]=51}
    residualLiteral:=ckks.ParametersLiteral{LogN:17,LogQ:logQ,LogP:[]int{61,61,61},LogDefaultScale:51}
    bootstrapLiteral:=bootstrapping.ParametersLiteral{LogN:utils.Pointy(17),LogP:[]int{61,61,61,61}}
    result["residual_literal"]=residualLiteral
    result["bootstrap_literal"]=bootstrapLiteral
    var p ckks.Parameters
    var bp bootstrapping.Parameters
    if err:=stage("residual_parameters",func()(err error){p,err=ckks.NewParametersFromLiteral(residualLiteral);return});err!=nil{return err}
    if err:=stage("bootstrap_parameters",func()(err error){bp,err=bootstrapping.NewParametersFromLiteral(p,bootstrapLiteral);return});err!=nil{return err}
    result["residual"]=parameterSummary(p)
    result["bootstrapping"]=parameterSummary(bp.BootstrappingParameters)
    result["bootstrap_parameters"]=bp
    result["bootstrap_depth"]=map[string]int{"coeffs_to_slots":bp.DepthCoeffsToSlots(),"eval_mod":bp.DepthEvalMod(),
        "slots_to_coeffs":bp.DepthSlotsToCoeffs(),"total":bp.Depth(),"log_slots":bp.LogMaxSlots()}
    f:=NewLattigoFHE(65536,"",mlir,"","","","",MLIR,16,3,16,"",false,true)
    var operations []string
    var inputs []Term
    var output int
    var rotations []int
    if err:=stage("original_mlir_reader",func()(err error){
        var expected string
        expected,operations,inputs,output,err=f.ReadFile(mlir)
        if err!=nil{return}
        if expected!="" || len(inputs)!=1{return fmt.Errorf("unexpected input or expected-output section")}
        rotations=f.findUniqueRotsPow2(operations)
        bootRows:=[]map[string]any{}
        opCounts:=map[string]int{}
        for _,line:=range operations {
            id,term:=f.parseMLIROperation(line)
            if term==nil{return fmt.Errorf("cannot parse original operation: %s",line)}
            opCounts[fmt.Sprint(term.Op)]++
            if term.Op==BOOTSTRAP {
                bootRows=append(bootRows,map[string]any{"id":id,"children":term.Children,"remaining_output_level":term.Level,
                    "mlir_target_level":term.Metadata.TargetLevel,"log2_scale":term.Scale.Log2()})
            }
        }
        for _,row:=range bootRows {
            children:=row["children"].([]int)
            if len(children)!=1{return fmt.Errorf("bootstrap arity mismatch")}
            predecessor:=f.terms[children[0]]
            if predecessor==nil{return fmt.Errorf("bootstrap predecessor missing")}
            row["remaining_input_level"]=predecessor.Level
        }
        result["model_program"]=map[string]any{"mlir":mlir,"operations":len(operations),"op_counts_enum":opCounts,
            "input_count":len(inputs),"input_remaining_level":inputs[0].Level,"input_scale":inputs[0].Scale.Log2(),
            "output_id":output,"bootstrap_count":len(bootRows),"bootstrap_nodes":bootRows,
            "level_convention":"remaining_level = bootstrapMaxLevel - MLIR_consumed_level"}
        if len(bootRows)!=19{return fmt.Errorf("expected 19 native bootstrap nodes, observed %d",len(bootRows))}
        return nil
    });err!=nil{return err}
    if err:=stage("galois_and_key_shapes",func()error{
        nativeGal:=make([]uint64,len(rotations))
        for i,r:=range rotations{nativeGal[i]=p.GaloisElement(r)}
        bootGal:=bp.GaloisElements(bp.BootstrappingParameters)
        bootRequested:=append(append([]uint64{},bootGal...),bp.BootstrappingParameters.GaloisElementForComplexConjugation())
        nativeUnique,bootUnique:=unique(nativeGal),unique(bootRequested)
        result["galois"]=map[string]any{"model_rotations_requested":rotations,"model_galois_requested":nativeGal,
            "model_galois_unique":nativeUnique,"bootstrap_galois_api":bootGal,
            "bootstrap_generation_requested":bootRequested,"bootstrap_retained_unique":bootUnique,
            "coeffs_to_slots":bp.CoeffsToSlotsParameters.GaloisElements(bp.BootstrappingParameters),
            "slots_to_coeffs":bp.SlotsToCoeffsParameters.GaloisElements(bp.BootstrappingParameters),
            "conjugation_already_present_in_api":len(bootRequested)!=len(bootUnique)}
        residualKey,bootKey:=payload(p.GetRLWEParameters()),payload(bp.BootstrappingParameters.GetRLWEParameters())
        // Encapsulation uses Q[:1], P[:1], same N, default decomposition.
        sparse,err:=rlwe.NewParametersFromLiteral(rlwe.ParametersLiteral{LogN:bp.BootstrappingParameters.LogN(),
            Q:bp.BootstrappingParameters.Q()[:1],P:bp.BootstrappingParameters.P()[:1]})
        if err!=nil{return err}
        denseToSparse,sparseToDense:=int64(0),int64(0)
        if bp.EphemeralSecretWeight!=0 {denseToSparse=payload(&sparse);sparseToDense=bootKey}
        sameN:=p.N()==bp.BootstrappingParameters.N()
        if !sameN{return fmt.Errorf("unexpected ring-switch keys required")}
        residualSecret:=int64(p.N())*int64(p.QCount()+p.PCount())*8
        bootSecret:=int64(bp.BootstrappingParameters.N())*int64(bp.BootstrappingParameters.QCount()+bp.BootstrappingParameters.PCount())*8
        residualRetained:=residualKey*(1+int64(len(nativeUnique)))
        bootstrapRetained:=bootKey*(1+int64(len(bootUnique)))+denseToSparse+sparseToDense
        retained:=residualRetained+bootstrapRetained+3*residualSecret
        generation:=residualKey*(1+int64(len(nativeGal)))+bootKey*(1+int64(len(bootRequested)))+denseToSparse+sparseToDense+3*residualSecret+bootSecret
        constants:=int64(1547)*65536*8 // conservative existing all-vector equivalent; actual R25 is recorded separately
        result["memory_estimate"]=map[string]any{
            "method":"uint64 coefficient allocation shape; no key allocation, keygen, DFT matrix encoding or CKKS",
            "residual_single_evaluation_key_bytes":residualKey,"bootstrap_single_evaluation_key_bytes":bootKey,
            "residual_secret_key_payload_bytes":residualSecret,"residual_public_key_payload_bytes":2*residualSecret,
            "bootstrap_extended_secret_payload_bytes":bootSecret,"dense_to_sparse_key_bytes":denseToSparse,"sparse_to_dense_key_bytes":sparseToDense,
            "residual_evaluation_keys_retained_bytes":residualRetained,"bootstrap_evaluation_keys_retained_bytes":bootstrapRetained,
            "retained_keys_and_residual_sk_pk_payload_bytes":retained,"requested_key_generation_payload_with_extended_secret_bytes":generation,
            "constants_all_full_vector_upper_bound_bytes":constants,"retained_payload_plus_constant_upper_bound_bytes":retained+constants,
            "physical_memory_bytes_observed_by_outer":nil,"fits_16gib_by_key_payload_alone":retained<=16*(1<<30),
            "key_payload_exceeds_16gib":retained>16*(1<<30),"full_execution_peak_bytes":nil,
            "unmeasured":[]string{"Go slice/map and allocator overhead","ring and key-generator scratch","encoded DFT matrices",
                "bootstrap evaluator scratch","ciphertext SSA and hoisted rotation cache","GC overlapping key allocations"},
            "same_ring_no_ring_switch_keys":sameN,"bootstrap_secret_source":"same residual secret extended to boot Q/P; declared bootstrap Xs alone does not certify generated key distribution",
            "security_acceptance":nil}
        return nil
    });err!=nil{return err}
    var usage runtime.MemStats
    runtime.ReadMemStats(&usage)
    result["go_memstats"]=map[string]uint64{"heap_alloc":usage.HeapAlloc,"heap_sys":usage.HeapSys,"total_alloc":usage.TotalAlloc,"sys":usage.Sys}
    return nil
}

func main(){
    mlir:=flag.String("mlir","","original R23 optimized MLIR")
    output:=flag.String("output","","parameter JSON output")
    eventFile:=flag.String("events","","relative Go clock JSONL")
    flag.Parse()
    eventsPath=*eventFile
    result:=map[string]any{"status":"running","mode":"parameters","started_unix_ns":processStart.UnixNano(),
        "keygen_runs":0,"ckks_runs":0,"encryptions":0,"decryptions":0,"get_log_enabled":false,
        "encoded_dft_matrices_created":0,"evaluation_keys_allocated":0,"extra_integrity_checks":0}
    code:=0
    defer func(){
        if recovered:=recover();recovered!=nil{result["error"]=fmt.Sprint(recovered);result["status"]="error";code=1}
        result["elapsed_ns"]=time.Since(processStart).Nanoseconds()
        data,err:=json.MarshalIndent(result,"","  ")
        if err==nil {err=os.WriteFile(*output,append(data,'\n'),0600)}
        if err!=nil{fmt.Fprintln(os.Stderr,err);code=1}
        if code!=0{os.Exit(code)}
    }()
    if *mlir=="" || *output=="" || eventsPath=="" {panic("explicit paths required")}
    if err:=execute(*mlir,result);err!=nil{result["status"]="error";result["error"]=err.Error();code=1;return}
    result["status"]="ok"
}
'''


def _tree_bytes(root):
    amount = 0
    for parent, _, names in os.walk(root):
        for name in names:
            try:
                amount += (Path(parent) / name).stat().st_size
            except FileNotFoundError:
                pass
    return amount


def _environment(work, sdk):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('GO', 'LD_', 'NIX_', '__NIX_', 'HECATE_'))
           and key not in ('HECATE', 'IN_NIX_SHELL', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')}
    env.update(GOROOT=str(sdk), GOOS='linux', GOARCH='arm64', GO111MODULE='on',
               GOPATH=str(work / 'gopath'), GOMODCACHE=str(work / 'module-cache'),
               GOCACHE=str(work / 'build-cache'), GOTMPDIR=str(work / 'tmp'), TMPDIR=str(work / 'tmp'),
               TEST_TELEMETRY_DIR=str(work / 'telemetry'), GOENV='off', GOWORK='off',
               GOTOOLCHAIN='local', GOPROXY='off', GOSUMDB='off', GOAUTH='off', GOVCS='*:off',
               GOPRIVATE='', GONOPROXY='', GONOSUMDB='', CGO_ENABLED='0', GOMAXPROCS='1',
               GOMEMLIMIT='3GiB', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def execute(args):
    out = args.out.resolve()
    if out.exists() or out.is_symlink():
        raise RuntimeError('R27 requires a new output directory')
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    started = time.perf_counter_ns()
    deadline = time.monotonic() + 1800
    recorder = StageRecorder(out / 'events.jsonl')
    result = dict(round=ROUND, model=MODEL, mode=MODE, status='running', error=None, failed_stage=None,
                  git_rev=None, git_branch=None, process_records=[], retry_count=0,
                  attempted=dict(source_copy=0, build=0, parameter_export=0),
                  completed=dict(source_copy=0, build=0, parameter_export=0),
                  keygen_runs=0, ckks_runs=0, encryptions=0, decryptions=0,
                  model_forwards=0, checkpoint_reads=0, probe_reads=0, compiled_constant_reads=0,
                  llm_calls=0, downloads=0, installations=0, extra_integrity_checks=0,
                  get_log_enabled=False, parameter_export_passed=False, native_backend_ready=False,
                  native_go_build_mechanism=True, native_verification_scope='existing offline Go module/build mechanisms only',
                  compiled_bootstraps=None, predicted_latency=None, ckks_latency=None,
                  accuracy_acceptance=None, security_acceptance=None,
                  full_execution_peak_bytes=None, artifact_bytes=0, resource_monitor_error=None,
                  source_record=dict(lattigo_root=str(args.lattigo_root), original_fhe=str(args.lattigo_root / 'lowering/fhe'),
                                     optimized_mlir=str(args.optimized_mlir), source_r23_result=str(args.r23_result),
                                     source_r25_result=str(args.r25_result), source_r26_result=str(args.r26_result),
                                     module_cache=str(args.module_cache), build_cache=str(args.build_cache),
                                     go_sdk=str(args.go_sdk)))
    stop = threading.Event()
    def checkpoint():
        if result['resource_monitor_error']:
            raise RuntimeError(result['resource_monitor_error'])
        if _tree_bytes(out) > 4 * GIB:
            raise RuntimeError('R27 case exceeds 4 GiB')
        if shutil.disk_usage(out).free < 10 * GIB:
            raise RuntimeError('R27 disk reserve below 10 GiB')
        if time.monotonic() >= deadline:
            raise TimeoutError('R27 1800-second batch allowance exhausted')
    def monitor():
        while not stop.wait(1):
            try:
                size, free = _tree_bytes(out), shutil.disk_usage(out).free
                with (out / 'artifact-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(clock_ns=time.perf_counter_ns(), artifact_bytes=size, disk_free_bytes=free)) + '\n')
                checkpoint()
            except BaseException as exc:
                result['resource_monitor_error'] = f'{type(exc).__name__}: {exc}'
                os.kill(os.getpid(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=monitor, daemon=True)
    def phase(name, argv, seconds, category=None, cwd=None):
        checkpoint()
        result['failed_stage'] = name
        if category:
            result['attempted'][category] += 1
        row = dict(stage=name, argv=[str(value) for value in argv], cwd=str(cwd or work), observation=None,
                   native_termination_signal=None)
        result['process_records'].append(row)
        write_json(out / 'progress.json', result)
        with recorder.span(name, command=row['argv']):
            observation = run_process(row['argv'], cwd=cwd or work, log=out / (name + '.log'),
                                      resources=out / (name + '-resources.jsonl'),
                                      timeout=min(seconds, max(.01, deadline - time.monotonic())),
                                      mem_gib=4, fsize_gib=8, env=env)
            row['observation'] = observation
            code = observation['exit_code']
            row['native_termination_signal'] = -code if isinstance(code, int) and code < 0 else None
            write_json(out / 'progress.json', result)
            if code != 0 or observation['timed_out'] or observation['signal'] is not None or observation['remaining_pids']:
                raise RuntimeError('R27 phase failed: ' + name)
            if category:
                result['completed'][category] += 1
        checkpoint()
    def interrupted(signum, _frame):
        raise RuntimeError('R27 outer signal ' + str(signum))
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL, 1800)
    try:
        for path in (args.go_sdk / 'bin/go', args.optimized_mlir, args.r23_result, args.r25_result, args.r26_result,
                     args.lattigo_root / 'go.mod', args.lattigo_root / 'go.sum'):
            if not path.is_file():
                raise RuntimeError('R27 missing required existing file: ' + str(path))
        for path in (args.module_cache, args.build_cache):
            if not path.is_dir():
                raise RuntimeError('R27 missing existing offline cache: ' + str(path))
        for directory in ('gopath', 'tmp', 'telemetry'):
            (work / directory).mkdir()
        (work / 'telemetry/mode').write_text('off\n')
        env = _environment(work, args.go_sdk)
        watcher.start()
        with recorder.span('source_preflight'):
            records = {key: json.loads(path.read_text()) for key, path in (
                ('r23', args.r23_result), ('r25', args.r25_result), ('r26', args.r26_result))}
            for key, record in records.items():
                if record.get('status') != 'ok' or record.get('model') != MODEL:
                    raise RuntimeError('R27 prior source is not the successful original model: ' + key)
            result['prior_evidence'] = dict(
                r23_bootstrap_count=19, r25_scope=records['r25'].get('metrics', {}).get('native_packed_equivalence_scope'),
                r26_outlier=dict(sample='test:9596', label=5, original_prediction=5, approximate_prediction=2,
                                 max_abs_u=1.0468313185754259, max_logit_difference=54541441.03884921,
                                 source=str(args.r26_result), source_scope='previous R26 paired diagnostic; no new model evaluation'),
                security_acceptance=None, native_parameter_equivalence_to_other_backends=None)
            result['sdk_version_text'] = (args.go_sdk / 'VERSION').read_text()
            result['original_module_text'] = (args.lattigo_root / 'go.mod').read_text()
            result['source_mlir_metadata'] = dict(path=str(args.optimized_mlir), bytes=args.optimized_mlir.stat().st_size)
            result['physical_memory_bytes'] = next(int(row.split()[1]) * 1024 for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemTotal:'))
        result['attempted']['source_copy'] += 1
        with recorder.span('isolated_source_copy'):
            module = work / 'module'
            module.mkdir()
            copied = []
            for source in sorted((args.lattigo_root / 'lowering/fhe').glob('*.go')):
                if source.name == 'main.go' or source.name.endswith('_test.go'):
                    continue
                shutil.copyfile(source, module / source.name)
                copied.append(dict(source=str(source), destination=str(module / source.name), bytes=source.stat().st_size))
            if not copied:
                raise RuntimeError('R27 original FHE source package empty')
            (module / 'main.go').write_text(GO_MAIN)
            (module / 'go.mod').write_text('module pt-agent-r27-orbit\n\ngo 1.25.0\n\nrequire github.com/tuneinsight/lattigo/v6 v6.0.0\n\nreplace github.com/tuneinsight/lattigo/v6 => ' + str(args.lattigo_root) + '\n')
            shutil.copyfile(args.lattigo_root / 'go.sum', module / 'go.sum')
            shutil.copytree(args.module_cache, work / 'module-cache')
            shutil.copytree(args.build_cache, work / 'build-cache')
            result['copied_sources'] = copied
            result['source_copy_contains_original_main'] = False
        result['completed']['source_copy'] += 1
        phase('git-revision', ['git', 'rev-parse', 'HEAD'], 15, cwd=paths.REPO)
        result['git_rev'] = (out / 'git-revision.log').read_text().strip()
        phase('git-branch', ['git', 'branch', '--show-current'], 15, cwd=paths.REPO)
        result['git_branch'] = (out / 'git-branch.log').read_text().strip()
        binary = work / 'r27-orbit-parameters'
        phase('build', [args.go_sdk / 'bin/go', 'build', '-p=1', '-buildvcs=false', '-mod=mod', '-o', binary, '.'],
              900, 'build', cwd=module)
        phase('parameter-export', [binary, '-mlir', args.optimized_mlir, '-output', out / 'parameters.json',
                                   '-events', out / 'go-events.jsonl'], 600, 'parameter_export')
        with recorder.span('parameter_observation'):
            parameters = json.loads((out / 'parameters.json').read_text())
            if parameters.get('status') != 'ok':
                raise RuntimeError('R27 parameter exporter did not return ok')
            for key in ('keygen_runs', 'ckks_runs', 'encryptions', 'decryptions', 'encoded_dft_matrices_created', 'evaluation_keys_allocated'):
                if parameters.get(key) != 0:
                    raise RuntimeError('R27 forbidden action observed: ' + key)
            result['parameters'] = parameters
            estimate = parameters['memory_estimate']
            estimate['physical_memory_bytes_observed_by_outer'] = result['physical_memory_bytes']
            result['memory_estimate'] = estimate
            result['parameter_export_passed'] = True
            result['full_keygen_feasible_by_payload'] = estimate['retained_keys_and_residual_sk_pk_payload_bytes'] < result['physical_memory_bytes']
            result['next_action'] = ('resource design required before full keygen' if not result['full_keygen_feasible_by_payload']
                                     else 'keygen peak and evaluator scratch still require a separately recorded limited attempt')
        result.update(status='ok', failed_stage=None)
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}'[:4000])
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=3)
        result['elapsed_ns'] = time.perf_counter_ns() - started
        result['timing'] = summarize_events(read_events(out / 'events.jsonl'))
        result['artifacts'] = dict(events=str(out / 'events.jsonl'), go_events=str(out / 'go-events.jsonl'),
                                   parameters=str(out / 'parameters.json'), work=str(work), result=str(out / 'result.json'))
        result['metrics'] = {key: result.get(key) for key in (
            'status', 'parameter_export_passed', 'attempted', 'completed', 'memory_estimate', 'full_keygen_feasible_by_payload',
            'keygen_runs', 'ckks_runs', 'encryptions', 'decryptions', 'get_log_enabled', 'llm_calls', 'downloads',
            'installations', 'extra_integrity_checks', 'compiled_bootstraps', 'predicted_latency', 'ckks_latency',
            'accuracy_acceptance', 'security_acceptance', 'elapsed_ns')}
        write_json(out / 'progress.json', result)
        for _ in range(12):
            write_json(out / 'result.json', result)
            size = _tree_bytes(out)
            if result['artifact_bytes'] == size:
                break
            result['artifact_bytes'] = size
        if result['artifact_bytes'] > 4 * GIB and result['status'] == 'ok':
            result.update(status='error', error='R27 terminal artifact allowance exceeded')
            result['metrics']['status'] = 'error'
            for _ in range(12):
                write_json(out / 'result.json', result)
                size = _tree_bytes(out)
                if result['artifact_bytes'] == size:
                    break
                result['artifact_bytes'] = size
        row = dict(run_id=out.name, round=ROUND, model=MODEL, mode=MODE, git_rev=result['git_rev'],
                   metrics=result['metrics'], artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
        registry = paths.RESULTS_DIR / 'registry.jsonl'
        with registry.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            fcntl.flock(stream, fcntl.LOCK_UN)
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('parameters',))
    parser.add_argument('--out', type=Path, default=paths.RESULTS_DIR / 'orbit-native-parameters-r27-20261116')
    parser.add_argument('--lattigo-root', type=Path, default=paths.ORBIT_ROOT / 'backend/lattigo')
    parser.add_argument('--go-sdk', type=Path, default=paths.ORBIT_ROOT.parent / 'toolchains/go125/go')
    parser.add_argument('--module-cache', type=Path, default=Path(tempfile.gettempdir()) / 'orbit-backend-20261115/gomodcache')
    parser.add_argument('--build-cache', type=Path, default=Path(tempfile.gettempdir()) / 'orbit-backend-20261115/gocache')
    parser.add_argument('--r23-result', type=Path, default=paths.RESULTS_DIR / 'orbit-resnet20-optimizer-r23-20261115/result.json')
    parser.add_argument('--r25-result', type=Path, default=paths.RESULTS_DIR / 'orbit-resnet20-packed-r25-20261115/result.json')
    parser.add_argument('--r26-result', type=Path, default=paths.RESULTS_DIR / 'orbit-resnet20-heldout-r26-20261115/result.json')
    parser.add_argument('--optimized-mlir', type=Path, default=paths.RESULTS_DIR / 'orbit-resnet20-optimizer-r23-20261115/work/optimized.mlir')
    result = execute(parser.parse_args(argv))
    print(json.dumps({key: result.get(key) for key in ('status', 'error', 'artifacts', 'memory_estimate', 'next_action')}, ensure_ascii=False))
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
