"""Build an own main overlay and statically consume an accepted Orbit MLIR.

The official parser and graph scheduler remain unchanged. This is a backend
consumption check, separate from the optimizer and the shared interpreter build.
No model arithmetic, external constant payload, context or keys are accessed.
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
from . import orbit_backend_compile_only as shared
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import (durable_json, physical_observation, serial_lock,
    interruption_status, receive_signal, observe_termination)
from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

ROUND = 'ORBIT-BACKEND-PARSER'
MODE = 'official-earth-parser-and-static-SSA-consumption-only'
GIB = 1024**3
TOTAL_SECONDS = 1500
PHASE_SECONDS = dict(build=600, selftest=120, parse=300)
CASE_BYTES = 8*GIB
RESERVE_BYTES = 1024**2
ZERO_FIELDS = ('context_constructions', 'parameter_constructions', 'keygen_runs',
    'encryptions', 'decryptions', 'bootstraps', 'model_arithmetic_runs',
    'constant_payload_reads', 'sample_reads', 'llm_calls', 'downloads', 'installations')

# Original package-private parser/types are available because only main.go is
# overlaid during build. No original source file is changed on disk.
CLIENT = r'''package main
import (
 "encoding/json"
 "flag"
 "fmt"
 "math"
 "os"
 "path/filepath"
 "runtime"
 "strings"
 "time"
)
func parseGraph(path string, slots, maxLevel int) (result map[string]any, err error) {
 defer func(){if v:=recover();v!=nil{err=fmt.Errorf("official parser or scheduler panic: %v",v)}}()
 raw,err:=os.ReadFile(path);if err!=nil{return nil,err}
 returns:=0
 for _,line:=range strings.Split(string(raw),"\n"){
  line=strings.TrimSpace(line)
  if strings.HasPrefix(line,"\"func.return\"")||strings.HasPrefix(line,"return "){returns++}
 }
 if returns!=1{return nil,fmt.Errorf("exactly one explicit source return required")}
 // A metadata holder only. All context/parameter/key pointer fields stay nil.
 holder:=&LattigoFHE{terms:make(map[int]*Term),n:slots,bootstrapMaxLevel:maxLevel,fileType:MLIR}
 expected,operations,inputs,output,err:=holder.ReadFile(path)
 if err!=nil{return nil,err};if expected!=""{return nil,fmt.Errorf("Earth MLIR expected")}
 if len(inputs)!=1||len(operations)==0{return nil,fmt.Errorf("one original input and nonempty graph required")}
 for i,input:=range inputs{
  value:=input;s:=value.Scale.Float64()
  if !value.Secret||math.IsNaN(s)||math.IsInf(s,0)||s<=0{return nil,fmt.Errorf("invalid input type metadata")}
  holder.terms[-1-i]=&value
 }
 kinds:=map[op]string{CONST:"constant",ADD:"add",MUL:"mul",ROT:"rotate",MODSWITCH:"modswitch",NEGATE:"negate",BOOTSTRAP:"bootstrap",RESCALE:"rescale",UPSCALE:"upscale"}
 counts:=map[string]int{"input":len(inputs)}
 seen:=make(map[int]bool);edgeCount,operandEdges,bootCount:=0,0,0
 nodes:=make([]map[string]any,0,len(operations))
 for _,line:=range operations{
  id,term:=holder.parseMLIROperation(line)
  if term==nil||id<0||id>=len(operations)||seen[id]{return nil,fmt.Errorf("missing/duplicate/out-of-range original SSA ID %d",id)}
  kind,ok:=kinds[term.Op];if !ok{return nil,fmt.Errorf("unsupported original Earth operation at%d",id)}
  arity:=1;if term.Op==CONST{arity=0}else if term.Op==ADD||term.Op==MUL{arity=2}
  if len(term.Children)!=arity{return nil,fmt.Errorf("invalid original arity at%d",id)}
  children:=make(map[int]bool)
  for _,child:=range term.Children{
   if child<0{if _,ok:=holder.terms[child];!ok{return nil,fmt.Errorf("unbound original input%d",child)}}else if child>=id||!seen[child]{return nil,fmt.Errorf("original SSA child%d not defined before%d",child,id)}
   children[child]=true
  }
  scale:=term.Scale.Float64()
  if scale<=0||math.IsNaN(scale)||math.IsInf(scale,0)||math.IsNaN(term.Metadata.RMSVar)||math.IsInf(term.Metadata.RMSVar,0){return nil,fmt.Errorf("invalid original scalar type metadata at%d",id)}
  seen[id]=true;counts[kind]++;edgeCount+=len(children);operandEdges+=len(term.Children)
  if term.Op==BOOTSTRAP{bootCount++}
  nodes=append(nodes,map[string]any{"id":id,"op":kind,"inputs":term.Children,"secret":term.Secret,"level":term.Level,"log2_scale":math.Log2(scale),"constant_id":term.Metadata.Value,"bootstrap_target":term.Metadata.TargetLevel})
 }
 for id:=range len(operations){if !seen[id]{return nil,fmt.Errorf("noncontiguous original ID%d",id)}}
 if _,ok:=holder.terms[output];!ok{return nil,fmt.Errorf("original return is unbound")}
 order,depth,back:=holder.getOptimalRunOrder(len(operations))
 if len(order)!=len(operations)||len(depth)!=len(operations)||len(back)!=len(operations){return nil,fmt.Errorf("incomplete original static order")}
 pool,err:=holder.getMaxWorkingPoolSize(len(operations),order);if err!=nil{return nil,err}
 result=map[string]any{"status":"ok","mlir_path":path,"mlir_bytes":len(raw),"mlir_text_reads":2,
  "parsed_operations":len(operations),"parsed_nodes":len(operations)+len(inputs),"input_count":len(inputs),"output_id":output,
  "edge_count":edgeCount,"operand_edges_with_duplicates":operandEdges,"op_counts":counts,"static_bootstrap_nodes":bootCount,
  "topological_order_count":len(order),"static_max_working_pool":pool,"nodes":nodes,
  "official_ReadFile_calls":1,"official_parseMLIROperation_calls":len(operations),"official_getOptimalRunOrder_calls":1,"official_getMaxWorkingPoolSize_calls":1,
  "context_constructions":0,"parameter_constructions":0,"keygen_runs":0,"encryptions":0,"decryptions":0,"bootstraps":0,"model_arithmetic_runs":0,"constant_payload_reads":0,"sample_reads":0}
 return result,nil
}
func selftest() error {
 directory,err:=os.MkdirTemp("","orbit-parser-fixture-");if err!=nil{return err};defer os.RemoveAll(directory)
 header:="^bb0(%arg0: tensor<1x!earth.ci<40 * 0>> loc(unknown)):\n"
 constant:="%0 = \"earth.constant\"() <{value = 0 : i64}> : () -> tensor<1x!earth.pl<40 * 0>>\n"
 add:="%1 = \"earth.add\"(%arg0, %0) : (tensor<1x!earth.ci<40 * 0>>, tensor<1x!earth.pl<40 * 0>>) -> tensor<1x!earth.ci<40 * 0>>\n"
 boot:="%2 = \"earth.bootstrap\"(%1) <{targetLevel = 0 : i64}> : (tensor<1x!earth.ci<40 * 0>>) -> tensor<1x!earth.ci<40 * 0>>\n"
 tail:="\"func.return\"(%2) : (tensor<1x!earth.ci<40 * 0>>) -> ()\n"
 valid:=header+constant+add+boot+tail
 tests:=[]struct{text string;pass bool}{{valid,true},{header+constant+constant+add+boot+tail,false},{strings.Replace(valid,"earth.add","earth.unknown",1),false},{strings.Replace(valid,"%arg0, %0","%3, %0",1),false},{strings.Replace(valid,"\"func.return\"(%2)","\"func.return\"(%99)",1),false},{header+constant+add+boot,false}}
 for i,test:=range tests{path:=filepath.Join(directory,fmt.Sprintf("case-%d.mlir",i));if err=os.WriteFile(path,[]byte(test.text),0600);err!=nil{return err};row,e:=parseGraph(path,16,16);if (e==nil)!=test.pass{return fmt.Errorf("selftest%d acceptance mismatch: %v",i,e)};if test.pass&&(row["parsed_nodes"]!=4||row["static_bootstrap_nodes"]!=1){return fmt.Errorf("selftest counts differ")}}
 return nil
}
func main(){
 runtime.GOMAXPROCS(1)
 input:=flag.String("mlir","","accepted optimized Earth MLIR")
 output:=flag.String("out","","own durable JSON result")
 slots:=flag.Int("slots",65536,"original parser rotation metadata modulus")
 level:=flag.Int("bootstrap-max-level",16,"original type/debt metadata conversion")
 test:=flag.Bool("self-test",false,"six public metadata-only parser/scheduler cases")
 flag.Parse();if *output==""||*slots<=0||*level<0{fmt.Fprintln(os.Stderr,"explicit output and valid metadata required");os.Exit(2)}
 started:=time.Now();result:=map[string]any{"status":"running"}
 var err error
 if *test{err=selftest();result=map[string]any{"status":"ok","selftests":6,"model_parses":0}}else{result,err=parseGraph(*input,*slots,*level)}
 if result==nil{result=make(map[string]any)}
 if err!=nil{result["status"]="error";result["error"]=err.Error()}
 result["elapsed_seconds"]=time.Since(started).Seconds()
 for _,name:=range []string{"context_constructions","parameter_constructions","keygen_runs","encryptions","decryptions","bootstraps","model_arithmetic_runs","constant_payload_reads","sample_reads"}{result[name]=0}
 result["scope"]="Official Earth parser and static graph scheduling; shared public package scalar/CPU/hash initialization retained"
 raw,e:=json.MarshalIndent(result,"","  ");if e!=nil{fmt.Fprintln(os.Stderr,e);os.Exit(1)}
 path:=*output+".tmp";if e=os.WriteFile(path,append(raw,'\n'),0600);e==nil{e=os.Rename(path,*output)}
 if e!=nil{fmt.Fprintln(os.Stderr,e);os.Exit(1)};if err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(1)}
}
'''


def read(path):
    return shared.read(path)


def initialization_gate(audit, module, sdk, cache):
    record = read(audit)
    review = record.get('explicit_review') or {}
    if (record.get('status') != 'passed_source_no_model_context_keys_or_payload_initializers'
            or record.get('unresolved_imports') != []
            or any(review.get(key) != 0 for key in ('model_context_initializers', 'keygen_initializers',
                'model_arithmetic_initializers', 'external_weight_or_constant_payload_reads'))):
        raise ValueError('A complete source initialization audit with no model/context/key/payload entry is required')
    roots = dict(module=module.resolve(), sdk=sdk.resolve(), cache=cache.resolve())
    previous = record.get('roots') or {}
    if any(Path(previous.get(key, '')).resolve() != roots[key] for key in ('module', 'sdk')):
        raise ValueError('Initialization audit SDK/module identity differs')
    rows = record.get('audited_files') or []
    if not rows:raise ValueError('Audited import source inventory missing')
    for row in rows:
        relative = Path(row['relative'])
        if row.get('scope') not in roots or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Audited source path escaped its source root')
        path = roots[row['scope']]/relative
        if not path.is_file() or path.stat().st_size != row['bytes'] or path.stat().st_mtime_ns != row['mtime_ns']:
            raise ValueError('Audited source metadata changed or is unavailable: '+str(path))
    return dict(path=str(Path(audit).resolve()), status=record['status'], source_files=len(rows),
        cache_source_rebased=Path(previous.get('cache', '')).resolve() != roots['cache'],
        metadata_scope='Source path/size/mtime and existing copy lineage; no byte/digest integrity verification',
        shared_public_scalar_initializers=True, crypto_hash_registration=True)


def source_gate(args):
    source = shared.source_gate(args.source_case)
    result = read(args.shared_build_case/'result.json')
    if (result.get('status') != 'ok' or result.get('runtime_interpreter_compiled') is not True
            or result.get('go_build_attempts') != 1 or result.get('go_build_completed') != 1
            or result.get('source') != source):
        raise ValueError('The completed original shared interpreter build must use this exact accepted model package')
    clean(result.get('process') or {})
    binary = result.get('binary') or {}
    if (Path(binary.get('path', '')).resolve() != (args.shared_build_case/'orbit-fhe').resolve()
            or not (args.shared_build_case/'orbit-fhe').is_file()
            or (args.shared_build_case/'orbit-fhe').stat().st_size != binary.get('bytes')
            or not binary.get('bytes')):
        raise ValueError('Completed original shared interpreter binary metadata required')
    module = args.orbit_root/'backend/lattigo'
    if Path(result.get('package', {}).get('interpreter_package', '')).resolve() != (module/'lowering/fhe').resolve():
        raise ValueError('Original backend module source identity differs')
    cache = args.shared_build_case/'work/module-cache'
    if not cache.is_dir():raise ValueError('Completed shared build isolated module cache required')
    audit = initialization_gate(args.initialization_audit, module, args.go_sdk, cache)
    validated = read(args.source_case/'result.json')['phase_results']['validate']['graph_check']
    if (validated.get('passed') is not True or validated.get('acyclic') is not True
            or validated.get('inputs') != ['arg0'] or len(validated.get('outputs', [])) != 1
            or validated.get('placement_bootstrap_nodes') != source['placement_bootstrap_nodes']):
        raise ValueError('Accepted single-input TDAG graph metadata required')
    expected = {key: validated[key] for key in ('node_count', 'edge_count', 'op_counts', 'inputs', 'outputs', 'placement_bootstrap_nodes')}
    expected['output_id'] = int(expected['outputs'][0])
    return dict(optimized_source=source, shared_build_case=str(args.shared_build_case.resolve()),
        initialization=audit, expected_graph=expected, module=str(module.resolve()), module_cache=str(cache.resolve()),
        scope='Optimized Earth MLIR official-parser consumption; no model-specific native code generation')


def clean(observation):
    if (observation.get('exit_code') != 0 or observation.get('timed_out') is not False
            or observation.get('signal') is not None or observation.get('signals_sent') != []
            or observation.get('remaining_pids') != []):
        raise ValueError('Original Go subprocess must close cleanly with no remaining process')


def build_command(sdk, out):
    return [str(sdk/'bin/go'), 'build', '-p=1', '-buildvcs=false', '-mod=readonly',
            '-overlay', str(out/'overlay.json'), '-o', str(out/'orbit-parser'), './lowering/fhe']


def parser_command(out, checked, *, selftest=False):
    command = [str(out/'orbit-parser'), '--out', str(out/('selftest.json' if selftest else 'parsed-model.json')),
        '--slots', str(checked['optimized_source']['optimizer_params']['max_slot']),
        '--bootstrap-max-level', str(checked['optimized_source']['optimizer_params']['bts_ub'])]
    return command+(['--self-test'] if selftest else ['--mlir', checked['optimized_source']['optimized_mlir']['path']])


def check_parsed(record, checked):
    source = checked['optimized_source'];expected = checked['expected_graph'];count = expected['node_count']-len(expected['inputs'])
    if (record.get('status') != 'ok' or Path(record.get('mlir_path', '')).resolve() != Path(source['optimized_mlir']['path']).resolve()
            or record.get('mlir_bytes') != source['optimized_mlir']['bytes']
            or record.get('mlir_text_reads') != 2
            or record.get('parsed_nodes') != expected['node_count'] or record.get('parsed_operations') != count
            or record.get('input_count') != 1 or record.get('output_id') != expected['output_id']
            or record.get('edge_count') != expected['edge_count'] or record.get('op_counts') != expected['op_counts']
            or record.get('static_bootstrap_nodes') != expected['placement_bootstrap_nodes']
            or record.get('topological_order_count') != count or len(record.get('nodes', [])) != count
            or record.get('official_ReadFile_calls') != 1 or record.get('official_parseMLIROperation_calls') != count
            or record.get('official_getOptimalRunOrder_calls') != 1 or record.get('official_getMaxWorkingPoolSize_calls') != 1
            or any(record.get(key) != 0 for key in ZERO_FIELDS[:9])):
        raise ValueError('Complete original parser/static-graph contract differs')
    ids = [node.get('id') for node in record['nodes']]
    if ids != list(range(count)):
        raise ValueError('Every original operation must retain contiguous ordered SSA identity')
    return dict(passed=True, parsed_nodes=expected['node_count'], static_bootstrap_nodes=expected['placement_bootstrap_nodes'],
                model_specific_native_codegen=None, model_arithmetic_executed=False)


def prepare(args, out, checked):
    work = out/'work';work.mkdir()
    for name in ('gopath', 'tmp', 'telemetry'):(work/name).mkdir()
    (work/'telemetry/mode').write_text('off\n')
    shutil.copytree(checked['module_cache'], work/'module-cache', symlinks=False)
    source = out/'source';source.mkdir();shutil.copyfile(__file__, source/Path(__file__).name)
    (source/'parser-main.go').write_text(CLIENT)
    durable_json(out/'overlay.json', dict(Replace={str(Path(checked['module'])/'lowering/fhe/main.go'): str(source/'parser-main.go')}))
    for name in ('parser.go', 'scheme.go', 'exeorder.go'):
        shutil.copyfile(Path(checked['module'])/'lowering/fhe'/name, source/name)
    shutil.copyfile(args.initialization_audit, source/'initialization-audit.json')
    return work


def append_registry(out, result, registry):
    metrics = {key: result[key] for key in ('status', 'attempted', 'completed', 'backend_parser_consumption_verified',
        'compiler_graph_completed', 'model_specific_native_codegen', 'formal_latency', 'accuracy_acceptance', 'security_acceptance')}
    metrics.update({key: 0 for key in ZERO_FIELDS})
    with Path(registry).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX);stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise ValueError('Existing result identity must be preserved')
        row = dict(run_id=out.name, round=ROUND, model=shared.MODEL, mode=MODE, git_rev=None,
            metrics=metrics, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
        stream.seek(0, os.SEEK_END);stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def terminal(out, result):
    result['completion_reserved_bytes'] = RESERVE_BYTES
    for _ in range(20):
        durable_json(out/'result.json', result);size = artifact_bytes(out)
        if size > CASE_BYTES-RESERVE_BYTES:
            result.update(status='resource_stopped', error='Case bytes including completion reserve exhausted')
        if result.get('artifact_bytes') == size:
            durable_json(out/'result.json', result)
            if artifact_bytes(out) == size:return
        result['artifact_bytes'] = size
    raise RuntimeError('Terminal result size did not converge')


def execute(args):
    out = args.out.resolve();root = paths.RESULTS_DIR.resolve()
    if out.exists() or out == root or not out.is_relative_to(root):raise ValueError('Fresh isolated results case required')
    with serial_lock(root):
        out.mkdir();recorder = StageRecorder(out/'events.jsonl');started = time.perf_counter_ns();deadline = time.monotonic()+TOTAL_SECONDS
        result = dict(round=ROUND, model=shared.MODEL, mode=MODE, status='running', error=None,
            attempted={phase: 0 for phase in PHASE_SECONDS}, completed={phase: 0 for phase in PHASE_SECONDS}, phases={},
            backend_parser_consumption_verified=None, compiler_graph_completed=None, model_specific_native_codegen=None,
            formal_latency=None, accuracy_acceptance=None, security_acceptance=None,
            guard_trigger=None, incoming_signal=None, timeout_signal=None, wall_timeout_observed=False,
            limits=dict(AS_gib=12, subtree_RSS_gib=14, minimum_available_gib=1.25, build_seconds=600,
                parse_seconds=300, selftest_seconds=120, total_seconds=1500, case_gib=8, file_gib=8,
                minimum_free_disk_gib=10, concurrency=1), **{key: 0 for key in ZERO_FIELDS})
        active = [None];stop = threading.Event();failures = []
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        def check():
            if failures:raise RuntimeError(failures[0])
            if time.monotonic() >= deadline:result['wall_timeout_observed'] = True;raise TimeoutError('Finite parser job expired')
            observed = physical_observation(_latest_rss_sample(active[0]) if active[0] else None,
                _available_ram_bytes(), shutil.disk_usage(out).free)
            if observed['stop']:
                result['guard_trigger'] = observed;durable_json(out/'resource-guard.json', observed)
                raise RuntimeError('; '.join(observed['reasons']))
            if artifact_bytes(out) > CASE_BYTES-RESERVE_BYTES:
                result['guard_trigger'] = dict(kind='artifact', allowance_bytes=CASE_BYTES)
                raise RuntimeError('Finite case allowance exhausted')
        def monitor():
            while not stop.wait(.25):
                try:check()
                except BaseException as exc:
                    failures.append(type(exc).__name__+': '+str(exc));result['monitor_stop_reason'] = failures[-1]
                    try:durable_json(out/'monitor-stop.json', dict(reason=failures[-1], guard=result.get('guard_trigger')))
                    finally:
                        try:os.kill(os.getpid(), signal.SIGTERM)
                        except OSError as error:result['monitor_signal_error'] = str(error)
                    return
        watcher = threading.Thread(target=monitor, daemon=True)
        for sig in old:signal.signal(sig, lambda signum, frame: receive_signal(result, signum))
        signal.setitimer(signal.ITIMER_REAL, TOTAL_SECONDS)
        try:
            with recorder.span('complete-backend-parser-job'):
                check();watcher.start()
                with recorder.span('source-and-initialization-admission'):
                    checked = source_gate(args);result['source'] = checked;result['compiler_graph_completed'] = True
                    work = prepare(args, out, checked)
                    commands = dict(build=build_command(args.go_sdk, out), selftest=parser_command(out, checked, selftest=True), parse=parser_command(out, checked))
                    result['commands'] = commands;durable_json(out/'source.json', checked)
                result['status'] = 'prepared';durable_json(out/'result.json', result)
                if args.run:
                    env = shared.native_environment(work, args.go_sdk)
                    for phase, command in commands.items():
                        check();result['attempted'][phase] = 1
                        row = dict(argv=command, observation=None, status='running');result['phases'][phase] = row
                        durable_json(out/'result.json', result)
                        with recorder.span(phase, argv=command):
                            active[0] = out/(phase+'-resources.jsonl')
                            try:
                                observation = run_process(command, cwd=Path(checked['module']), log=out/(phase+'.log'), resources=active[0],
                                    timeout=min(PHASE_SECONDS[phase], max(.01, deadline-time.monotonic())), mem_gib=12, fsize_gib=8, env=env)
                                row['observation'] = observation;observe_termination(result, observation)
                            finally:active[0] = None
                            clean(observation);check()
                            if phase == 'build':
                                if not (out/'orbit-parser').is_file():raise ValueError('Compiled parser client missing')
                            elif phase == 'selftest':
                                tests = read(out/'selftest.json')
                                if (tests.get('status') != 'ok' or tests.get('selftests') != 6 or tests.get('model_parses') != 0
                                        or any(tests.get(key) != 0 for key in ZERO_FIELDS[:9])):
                                    raise ValueError('Original parser six public selftests must pass with zero model/context/key actions')
                                result['selftest'] = tests
                            else:
                                result['parsed'] = read(out/'parsed-model.json');row['audit'] = check_parsed(result['parsed'], checked)
                        row['status'] = 'ok';result['completed'][phase] = 1;durable_json(out/'result.json', result)
                    result.update(status='ok', backend_parser_consumption_verified=True)
        except BaseException as exc:
            result.update(status=interruption_status(result), error=type(exc).__name__+': '+str(exc)[:4000])
            if (out/'parsed-model.json').is_file():result['parsed'] = read(out/'parsed-model.json')
            for row in result['phases'].values():
                if row['status'] == 'running':row.update(status=result['status'], error=result['error'])
        finally:
            stop.set()
            if watcher.is_alive():watcher.join(timeout=3)
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in old.items():signal.signal(sig, handler)
            if failures:result.update(status=interruption_status(result), error=failures[0])
            result['timing'] = summarize_events(read_events(out/'events.jsonl'), time.perf_counter_ns())
            terminal(out, result);append_registry(out, result, args.registry or root/'registry.jsonl')
            marker = persist_case_completion(out, round_label=ROUND, started_ns=started, status=result['status'],
                timing=result['timing'], required_artifact_bytes=artifact_bytes(out))
            if marker['case_artifact_bytes'] > CASE_BYTES or (out/'completion.json').stat().st_size > RESERVE_BYTES:
                raise RuntimeError('Completion metadata exceeded its reserved allowance')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'source-case', 'shared-build-case', 'orbit-root', 'go-sdk', 'initialization-audit'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--registry', type=Path);parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):setattr(args, name, value.expanduser().resolve())
    result = execute(args)
    print(json.dumps({key: result[key] for key in ('status', 'error', 'compiler_graph_completed', 'backend_parser_consumption_verified')}))
    return int(result['status'] not in ('ok', 'prepared'))


if __name__ == '__main__':raise SystemExit(main())
