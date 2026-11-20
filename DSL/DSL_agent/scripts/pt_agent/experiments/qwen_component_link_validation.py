"""Constructor-free admission of27 actual Qwen component artifacts.

Plans/manifests and file metadata only. A future native helper loads each plan
through the original generic library and validates the Program. It never runs
that Program or constructs parameters, contexts, keys or ciphertexts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import time

from .. import paths
from . import qwen_component_client_build as builder
from .completion_metrics import artifact_bytes, persist_case_completion
from .heir_initializer_probe import durable_json, serial_lock
from .native_component_matrix import array_items
from .r10_paper_metrics import StageRecorder, run_process, read_events, summarize_events
from .r27_orbit_lazy_backend import _available_ram_bytes, _latest_rss_sample

ROUND = 'QWEN-COMPONENT-LINK-VALIDATION'
MODE = 'constructor-free-original27-artifact-program-and-typed-link-admission'
MODULE = 'pt_agent.experiments.qwen_component_link_validation'
GIB = 1024**3
TOTAL_SECONDS, CASE_BYTES, RESERVE = 1800, 8*GIB, 1024**2
PHASE_SECONDS = dict(build=600, consume=900)
TYPE = dict(kind='ciphertext', context='qwen-bootstrap48', level=5, scale_log2=43,
            components=2, ntt=True, place=dict(kind='host', rank=0))
ZERO = builder.ZERO
CLIENT = r'''package main
import (
 "bytes"; "encoding/json"; "flag"; "fmt"; "go/ast"; "go/parser"; "go/printer"; "go/token"; "os"; "runtime"; "strings"
 q "pt-agent-qwen-component-inspector/componentlib"
)
type Input struct {Program q.Program `json:"program"`; LibrarySource string `json:"library_source"`}
func canon(n ast.Node) string {var b bytes.Buffer;printer.Fprint(&b,token.NewFileSet(),n);return strings.Join(strings.Fields(b.String()),"")}
func proof(path string)(map[string]any,error){
 f,e:=parser.ParseFile(token.NewFileSet(),path,nil,0);if e!=nil{return nil,e}
 functions:=map[string]*ast.FuncDecl{};layers:=false;constants:=map[string]string{}
 for _,d:=range f.Decls{switch n:=d.(type){case *ast.FuncDecl:if n.Name.Name=="init"{return nil,fmt.Errorf("library initializer changed")};functions[n.Name.Name]=n
 case *ast.GenDecl:for _,s:=range n.Specs{if v,ok:=s.(*ast.ValueSpec);ok&&n.Tok==token.CONST&&len(v.Names)==1&&len(v.Values)==1{constants[v.Names[0].Name]=canon(v.Values[0])};if t,ok:=s.(*ast.TypeSpec);ok&&t.Name.Name=="Program"{z,ok:=t.Type.(*ast.StructType);if !ok{return nil,fmt.Errorf("Program shape")};for _,v:=range z.Fields.List{if len(v.Names)==1&&v.Names[0].Name=="Layers"{layers=canon(v.Type)=="[24]ComponentRef"}}}}}}
 if constants["Slots"]!="16384"||constants["ContextID"]!=`"qwen-bootstrap48"`||constants["BootstrapProfile"]!=`"cpu-bootstrap48-level1-to5"`{return nil,fmt.Errorf("original raw43 constants differ")}
 borrow,ok:=functions["BorrowSharedContext"];if !ok||!strings.Contains(canon(borrow.Body),"p.N()!=65536")||!strings.Contains(canon(borrow.Body),"p.DefaultScale().Log2()!=43"){return nil,fmt.Errorf("raw43 borrowed parameter contract differs")}
 forward,ok:=functions["forward"];if !ok||!layers{return nil,fmt.Errorf("original24 layer declaration absent")}
 run,ok:=functions["RunTwoTokens"];if !ok{return nil,fmt.Errorf("original concrete caller absent")}
 count:=func(n ast.Node,name string)int{v:=0;ast.Inspect(n,func(n ast.Node)bool{if c,ok:=n.(*ast.CallExpr);ok&&canon(c.Fun)==name{v++};return true});return v}
 rangeOK:=false;ast.Inspect(forward.Body,func(n ast.Node)bool{if r,ok:=n.(*ast.RangeStmt);ok&&canon(r.X)=="p.Layers"&&count(r.Body,"s.RunComponent")==1{rangeOK=true};return true})
 if !rangeOK||count(forward.Body,"s.RunComponent")!=2||count(run.Body,"s.forward")!=2||count(run.Body,"s.RunComponent")!=2||count(run.Body,"s.BootstrapBoundary")!=3{return nil,fmt.Errorf("actual52-call/3-boundary source structure differs")}
 boundary,ok:=functions["BootstrapBoundary"];if !ok||count(boundary.Body,"input.value.CopyNew")!=1||count(boundary.Body,"s.evaluator.DropLevel")!=1||count(boundary.Body,"s.nativeBoot")!=1||!strings.Contains(canon(boundary.Body),"copied.Level()-1"){return nil,fmt.Errorf("real copied level1 native boundary path differs")}
 required:=map[string]bool{"original:=initial":false,"firstInputs[19]=original[0]":false,"second:=original":false,"second[p.PrefixLength],second[32],second[33]=mixed,attention,selector":false,"next[32],next[33]=current[32],current[33]":false}
 for _,body:=range []ast.Node{run.Body,forward.Body}{ast.Inspect(body,func(n ast.Node)bool{if a,ok:=n.(*ast.AssignStmt);ok{if _,yes:=required[canon(a)];yes{required[canon(a)]=true}};return true})}
 for k,v:=range required{if !v{return nil,fmt.Errorf("actual alias wiring differs: %s",k)}}
 return map[string]any{"component_calls_encoded":52,"boundary_boots_encoded":3,"layer_count":24,"forward_calls_encoded":2,"checked_aliases":required,"component_calls_executed":0,"boundary_boots_executed":0,"slots":16384,"context":"qwen-bootstrap48","nominal_scale_log2":43,"ring_degree":65536},nil
}
func main(){
 manifest:=flag.String("manifest","","source-bound manifest");output:=flag.String("out","","metadata result");flag.Parse()
 result:=map[string]any{"status":"error","error":nil,"program_validations":0,"component_load_attempts":0,"component_loads_completed":0,"payload_reads":0,"context_constructions":0,"parameter_constructions":0,"keygen_runs":0,"RunComponent_calls":0,"RunTwoTokens_calls":0,"encryptions":0,"decryptions":0,"bootstraps":0}
 code:=1
 defer func(){f,e:=os.Create(*output);if e!=nil{fmt.Fprintln(os.Stderr,e);os.Exit(2)};e=json.NewEncoder(f).Encode(result);f.Close();if e!=nil{fmt.Fprintln(os.Stderr,e);os.Exit(2)};os.Exit(code)}()
 var in Input;file,e:=os.Open(*manifest);if e!=nil{result["error"]=e.Error();return};e=json.NewDecoder(file).Decode(&in);file.Close();if e!=nil{result["error"]=e.Error();return}
 source,e:=proof(in.LibrarySource);if e!=nil{result["error"]=e.Error();return};result["actual_library_source_AST"]=source
 if e=in.Program.Validate();e!=nil{result["error"]=e.Error();return};result["program_validations"]=1
 refs:=append([]q.ComponentRef{},in.Program.Layers[:]...);refs=append(refs,in.Program.Head,in.Program.First,in.Program.Terminal)
 rows:=[]map[string]any{}
 for i,ref:=range refs{result["component_load_attempts"]=i+1;c,err:=q.LoadComponent(ref);if err!=nil{result["error"]=err.Error();result["components"]=rows;return};if c==nil{result["error"]="nil original component";return};c=nil;runtime.GC();rows=append(rows,map[string]any{"role":ref.Role,"index":ref.Index,"plan":ref.PlanPath,"bundle":ref.BundlePath,"source_case":ref.SourceCase});result["component_loads_completed"]=i+1;result["components"]=rows}
 result["status"]="ok";result["program_and_original27_artifacts_consumed"]=true;result["whole_generation_executed"]=false;code=0
}
'''


def read(path):
    return json.loads(Path(path).read_text())


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _typed(value):
    keys = tuple(TYPE)
    if (not isinstance(value, dict) or any(key not in value for key in keys)
            or any(type(value[key]) is not type(TYPE[key]) for key in keys)
            or value['kind'] != 'ciphertext' or value['context'] != TYPE['context']
            or value['scale_log2'] != 43 or value['components'] != 2 or value['ntt'] is not True
            or value['place'] != TYPE['place'] or not 1 <= value['level'] <= 5):
        raise ValueError('Retain actual raw43/context/NTT/host ciphertext descriptor')
    return {key: value[key] for key in keys}


def _physical(sample, available, own_rss=None):
    if own_rss is None:
        lines = Path('/proc/self/status').read_text().splitlines()
        own_rss = next(int(line.split()[1])*1024 for line in lines if line.startswith('VmRSS:'))
    child = 0 if sample is None else sample['subtree_rss_bytes']
    observed = builder.physical(dict(subtree_rss_bytes=child+own_rss), available)
    observed.update(supervisor_RSS_bytes=own_rss, child_sample_RSS_bytes=child)
    return observed


def component_metadata(path, role, index, expected_boots):
    """Small I/O view; existing component gate and native loader check all SSA."""
    from .qwen_layer_native_probe import _field
    path = Path(path).resolve()
    inputs, outputs = list(array_items(path, 'external_inputs')), list(array_items(path, 'final_outputs'))
    arity = (34, 32) if role == 'layer' else (34, 19) if role == 'head' else (20, 22) if role == 'first' else (19, 19)
    if (len(inputs), len(outputs)) != arity or len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise ValueError('Each complete component ABI must have unique actual outputs')
    selected = set(inputs+outputs); descriptions = {}
    for value in array_items(path, 'values'):
        if value['id'] in selected:
            if value['id'] in descriptions:
                raise ValueError('Native descriptor is repeated')
            descriptions[value['id']] = _typed(value)
    if set(descriptions) != selected or any(descriptions[i] != TYPE for i in inputs):
        raise ValueError('Original component inputs require actual raw43 level5 descriptors')
    if role in ('layer', 'head') and any(descriptions[i] != TYPE for i in outputs):
        raise ValueError('Original prefill output boot normalization differs')
    counts = Counter(item['op'] for item in array_items(path, 'execution'))
    if counts['boot'] != expected_boots:
        raise ValueError('Actual native bootstrap inventory differs from accepted artifact')
    bundle = path.with_name(path.name.removesuffix('.runtime-plan.json')+'.bundle')
    return dict(role=role, index=index, plan=str(path), bundle=str(bundle),
        inputs=[dict(local_id=i, descriptor=descriptions[i]) for i in inputs],
        outputs=[dict(local_id=i, descriptor=descriptions[i]) for i in outputs],
        native_boots=counts['boot'], native_operation_count=sum(counts.values()), native_op_counts=dict(counts),
        target=_field(path, 'target'), format_version=_field(path, 'format_version'))


def link_manifest(components, initial_state, source_case):
    """Compact namespaced global SSA: actual local IDs are not expanded twice."""
    from .qwen_generation_kernels import _state
    state = _state(initial_state)
    names = [*('layer'+str(i) for i in range(24)), 'head', 'first', 'terminal']
    if set(components) != set(names):
        raise ValueError('Require exactly24 original layers, head, first and terminal')
    canonical = [Path(components[name]['plan']).resolve() for name in names]
    if len(set(canonical)) != 27:
        raise ValueError('All27 original component plans must have distinct canonical paths')
    for name in names:
        role = 'layer' if name.startswith('layer') else name
        index = int(name[5:]) if role == 'layer' else 0
        if components[name]['role'] != role or components[name]['index'] != index:
            raise ValueError('Retain original24-layer order and exact head/kernel roles')
    if any(row.get('external_boundary_bootstraps_in_component', 0) != 0 for row in components.values()):
        raise ValueError('Only original caller may perform3 connector boots; reject double counting')
    targets = [json.dumps(components[name]['target'], sort_keys=True, allow_nan=False) for name in names]
    if len(set(targets)) != 1:
        raise ValueError('All27 original native target/parameter metadata must agree')
    if any(row['format_version'] != 1 for row in components.values()):
        raise ValueError('Use original format1 native component artifacts')
    initial = [dict(id='original:'+str(i), descriptor=dict(TYPE)) for i in range(34)]
    calls = []; connectors = []; ledger = {v['id']: v for v in initial}

    def invoke(name, incoming):
        row = components[name]; ordinal = len(calls); prefix = 'call'+str(ordinal)+':'+name+':'
        if len(incoming) != len(row['inputs']):
            raise ValueError('Actual caller input arity differs')
        for value, wanted in zip(incoming, row['inputs']):
            if ledger.get(value['id']) is not value or _typed(value['descriptor']) != _typed(wanted['descriptor']):
                raise ValueError('A real producer type is incompatible with its component consumer')
        outgoing = []
        for local in row['outputs']:
            value = dict(id=prefix+local['local_id'], descriptor=_typed(local['descriptor']))
            if value['id'] in ledger:
                raise ValueError('Global output identity redefined')
            ledger[value['id']] = value; outgoing.append(value)
        calls.append(dict(ordinal=ordinal, component=name, local_SSA_namespace=prefix,
            native_plan=row['plan'], inputs=[v['id'] for v in incoming], outputs=[v['id'] for v in outgoing],
            native_boots=row['native_boots']))
        return outgoing

    def forward(values):
        current = list(values)
        for i in range(24):
            output = invoke('layer'+str(i), current)
            current = [*output, *current[32:34]]
        return invoke('head', current)

    logits0 = forward(initial)
    chosen0 = invoke('first', [*logits0, initial[0]])
    boundary = []
    for position, name in enumerate(('mixed', 'attention', 'selector'), 19):
        original = chosen0[position]; descriptor = _typed(original['descriptor'])
        value = dict(id='connector:'+name, descriptor=dict(TYPE))
        ledger[value['id']] = value; boundary.append(value)
        connectors.append(dict(op='original.BootstrapBoundary', input=original['id'], output=value['id'],
            input_descriptor=descriptor, drop_levels_before_native_boot=descriptor['level']-1,
            native_boot_input_level=1, output_descriptor=dict(TYPE)))
    second = list(initial)
    second[state['prefix_length']], second[32], second[33] = boundary
    logits1 = forward(second)
    chosen1 = invoke('terminal', logits1)
    final = [*chosen0[:19], *chosen1]
    if len(calls) != 52 or len(final) != 38 or len({v['id'] for v in final}) != 38:
        raise ValueError('Complete original52-call/38-output protocol differs')
    refs = {name: dict(Role=components[name]['role'], Index=components[name]['index'],
        PlanPath=components[name]['plan'], BundlePath=components[name]['bundle'], SourceCase=str(source_case)) for name in names}
    boots = sum(row['native_boots'] for row in calls)+3
    return dict(format='qwen-original27-global-link-v1', variant='artifact-bound52-call-caller3-connectors-raw43',
        source_case=str(source_case), initial_state=state, initial=initial, calls=calls, connectors=connectors,
        second_forward_initial=[v['id'] for v in second], final_outputs=final,
        components=components, native_boots_encoded=boots,
        explicit_boundary_requests=2*(24*32+19)+3,
        native_boot_scope='Two actual prefill component inventories plus both kernels plus3 original caller boots; no native execution occurred',
        global_SSA_scope='All actual local SSA IDs receive the invocation namespace; complete local closure is independently checked. No repeated full-op JSON expansion.',
        declared_native_contract=dict(slots=16384, context=TYPE['context'], nominal_scale_log2=43,
            ring_degree=65536, input_level=5, native_boot_in_level=1, native_boot_out_level=5,
            same_context_required=True, same_physical_parameter_objects_required=True,
            actual_parameter_objects_verified=False),
        full_generation_executed=False, actual_ciphertext_metadata_verified=False,
        program=dict(Layers=[refs['layer'+str(i)] for i in range(24)], Head=refs['head'],
            First=refs['first'], Terminal=refs['terminal'], SourceCase=str(source_case), PrefixLength=state['prefix_length']))


def library_gate(case):
    case = Path(case).resolve(); result = read(case/'result.json'); complete = read(case/'completion.json')
    allowed = {(builder.ROUND, builder.MODE), ('QWEN-COMPONENT-LIBRARY-CPU-OVERLAP',
        'offline-generic-library-only-during-identified-bert-tune3')}
    if ((result.get('round'), result.get('mode')) not in allowed or result.get('status') != 'ok'
            or result.get('generic_library_compile_accepted') is not True or result.get('generic_library_build_passed') is not True
            or any(type(result.get(k)) is not int or result[k] != 1 for k in ('attempted_builds','completed_builds'))
            or complete.get('status') != 'ok' or complete.get('registry_persisted') is not True
            or complete.get('required_artifacts_persisted') is not True or any(result.get(k) != 0 for k in ZERO)
            or any(result.get(k) is not False for k in ('linked_native_program_verified', 'full_qwen_compile_verified', 'whole_generation_verified'))):
        raise ValueError('Require the closed actual constructor-free generic library build')
    builder.clean(result['observation'])
    source = result['source']; archive = Path(result['archive']['path']).resolve()
    runtime = case/'work/module/runtime.go'
    if (archive != case/'work/qwen-generation.a' or not archive.is_file() or archive.is_symlink()
            or not runtime.is_file() or runtime.is_symlink()
            or source.get('go_version') != '1.25.13' or source.get('lattigo_version') != '6.0.0'
            or source.get('parameters_selected') is not False or source.get('actual_component_artifacts_admitted') != 0):
        raise ValueError('Retain the accepted actual archive and its original compiled source snapshot')
    return dict(case=str(case), source=source, actual_library_source=str(runtime), archive=result['archive'],
        timing=result.get('timing'), completion=str(case/'completion.json'))


def source_gate(args):
    from . import qwen_remaining_prefill_compile as remaining
    from . import qwen_generation_kernel_compile as kernel
    checked = _json(remaining.admission(args.source_case, args.probe_case))
    master = read(args.remaining_case/'result.json'); done = read(args.remaining_case/'completion.json')
    if (master.get('round') != remaining.ROUND or master.get('mode') != remaining.MODE or master.get('status') != 'ok'
            or master.get('source') != checked or master.get('full_prefill_verified') is not True
            or master.get('accepted_components') != 25 or done.get('status') != 'ok'
            or done.get('registry_persisted') is not True or done.get('required_artifacts_persisted') is not True
            or set(master.get('components', {})) != {'layer0', *remaining.COMPONENTS}
            or master['components']['layer0'].get('runtime_plan') != checked['first_layer_runtime_plan']):
        raise ValueError('Require the complete same-source25-component prefill continuation')
    rows = {}; path = checked['first_layer_runtime_plan']
    rows['layer0'] = component_metadata(path, 'layer', 0, checked['first_layer_expected_bootstraps'])
    for name in remaining.COMPONENTS:
        record = _json(remaining.accept_component(Path(master['components'][name]['case']), name, checked))
        if record != master['components'][name]:
            raise ValueError('Saved actual prefill component acceptance differs')
        role = record['component']['role']; index = record['component']['index'] if role == 'layer' else 0
        rows[name] = component_metadata(record['runtime_plan'], role, index, record['expected_compiled_bootstraps'])
        gc.collect()
    kchecked = _json(kernel.admission(args.source_case))
    kmaster = read(args.kernel_case/'result.json'); kdone = read(args.kernel_case/'completion.json')
    if (kmaster.get('round') != kernel.ROUND or kmaster.get('mode') != kernel.MODE or kmaster.get('status') != 'ok'
            or kmaster.get('source') != kchecked or kdone.get('status') != 'ok'
            or kdone.get('registry_persisted') is not True or kdone.get('required_artifacts_persisted') is not True
            or set(kmaster.get('components', {})) != {'first', 'terminal'}
            or kmaster.get('accepted_components') != 2 or kmaster.get('two_kernel_native_components_verified') is not True):
        raise ValueError('Require both complete same-C2 actual generation kernel artifacts')
    for kind in ('first', 'terminal'):
        record = _json(kernel.accept_component(Path(kmaster['components'][kind]['case']), kind, kchecked))
        if record != kmaster['components'][kind]:
            raise ValueError('Saved kernel acceptance differs')
        spec = kernel.spec(kind)
        if spec['boundary_bootstraps'] != 0 or spec['future_boundary_bootstraps'] != (3 if kind == 'first' else 0):
            raise ValueError('Original caller owns the only3 connector bootstrap operations')
        native = record['native']; rows[kind] = component_metadata(record['runtime_plan'], kind, 0, native['compiled_bootstraps'])
        rows[kind]['external_boundary_bootstraps_in_component'] = spec['boundary_bootstraps']
    if checked['frozen_source'] != kchecked['frozen_source']:
        raise ValueError('Prefill and kernel artifacts must share the same accepted original frozen program')
    manifest = link_manifest(rows, kchecked['initial_state'], args.source_case)
    library = library_gate(args.library_case)
    return dict(source_case=str(args.source_case), frozen_source=checked['frozen_source'],
        prefill_source=dict(probe_case=str(args.probe_case), remaining_case=str(args.remaining_case)),
        kernel_source=dict(case=str(args.kernel_case), binding=kchecked['binding'], lineage=kchecked['lineage']),
        library=library, manifest=manifest, payload_reads=0)


def initialization_gate(path, library):
    audit = read(path); source = library['source']
    if (audit.get('format') != 'qwen-component-import-initialization-source-audit-v1'
            or audit.get('status') != 'passed_source_no_model_context_keys_or_payload_initializers'
            or Path(audit['roots']['lattigo']).resolve() != Path(source['lattigo_source']).resolve()
            or Path(audit['roots']['sdk']).resolve() != Path(source['sdk']).resolve()
            or Path((audit.get('actual_library_initializer') or {}).get('path', '')).resolve() != Path(library['actual_library_source']).resolve()
            or audit.get('lattigo_version') != '6.0.0'
            or any((audit.get('explicit_review') or {}).get(k) != 0 for k in
                ('model_context_initializers', 'parameter_constructor_initializers', 'keygen_initializers', 'external_payload_reads'))
            or not audit.get('audited_files') or audit.get('unresolved_imports') != []):
        raise ValueError('Same-version source-reviewed import initializers are required before actual consumption')
    text = Path(library['actual_library_source']).read_text()
    imports = sorted(re.findall(r'"([^"]+)"', re.search(r'(?ms)^import\s*\((.*?)\)', text).group(1))) if re.search(r'(?ms)^import\s*\((.*?)\)', text) else []
    if (re.search(r'(?m)^func\s+init\s*\(', text)
            or any(line != 'var safeLeaf=regexp.MustCompile(`^[A-Za-z0-9_-]+$`)' for line in re.findall(r'(?m)^var\s+[^\n]+', text))
            or audit.get('actual_library_imports') != imports):
        raise ValueError('Retain the source-reviewed actual library import/global initializer semantics')
    for row in audit['audited_files']:
        if not Path(row['path']).is_file():
            raise ValueError('Reviewed initializer source path is unavailable')
    return dict(path=str(Path(path).resolve()), status=audit['status'], source_files=len(audit['audited_files']),
        scope='Reviewed public scalar/CPU/registration startup allowed; no model/parameter/context/key/payload constructors')


def prepare(out, checked):
    work = out/'work'; module = work/'module'; module.mkdir(parents=True)
    for name in ('tmp', 'telemetry', 'gopath'):
        (work/name).mkdir()
    (work/'telemetry/mode').write_text('off\n')
    library = checked['library']; previous = Path(library['case'])/'work/module'
    (module/'go.mod').write_text('module pt-agent-qwen-component-inspector\n'+(previous/'go.mod').read_text().split('\n', 1)[1])
    shutil.copyfile(previous/'go.sum', module/'go.sum')
    (module/'componentlib').mkdir(); shutil.copyfile(library['actual_library_source'], module/'componentlib/runtime.go')
    (module/'main.go').write_text(CLIENT)
    manifest = dict(checked['manifest'], library_source=str(module/'componentlib/runtime.go'))
    durable_json(out/'global-link-manifest.json', manifest)
    snapshots = out/'source'; snapshots.mkdir(); shutil.copyfile(__file__, snapshots/Path(__file__).name)
    return [str(Path(library['source']['sdk'])/'bin/go'), 'build', '-p=1', '-mod=readonly', '-buildvcs=false',
            '-o', str(out/'qwen-component-inspector'), '.']


def _close(args, result, started):
    result['timing'] = summarize_events(read_events(args.out/'events.jsonl'))
    try:
        if (time.perf_counter_ns()-started)/1e9 >= TOTAL_SECONDS:
            raise TimeoutError('Complete admission allowance exhausted during terminal persistence')
        if shutil.disk_usage(args.out).free < 10*GIB:
            raise RuntimeError('Terminal disk reserve exhausted')
        observed = _physical(None, _available_ram_bytes())
        if observed['stop']:
            result['resource_guard'] = observed
            raise RuntimeError(str(observed))
    except BaseException as exc:
        result.update(status='error', terminal_error=type(exc).__name__+': '+str(exc))
    for _ in range(20):
        size = artifact_bytes(args.out)
        if size+RESERVE > CASE_BYTES:
            result.update(status='error', error='Terminal artifact allowance including completion reserve exhausted')
        result['constructor_free_program_and_artifacts_accepted'] = result['status'] == 'ok' and result['completed']['consume'] == 1
        result.update(linked_native_program_executed=False, full_generation_executed=False,
            linked_native_program_verified=False, full_qwen_compile_verified=False, whole_generation_verified=False,
            actual_ciphertext_metadata_verified=False, security_acceptance=None, numerical_acceptance=None, CKKS_latency=None)
        durable_json(args.out/'result.json', result)
        if result.get('artifact_bytes') == artifact_bytes(args.out):
            break
        result['artifact_bytes'] = artifact_bytes(args.out)
    else:
        raise RuntimeError('Terminal metadata fixed point failed')
    row = dict(run_id=args.out.name, round=ROUND, model='Qwen2-0.5B', mode=MODE, git_rev=None,
        metrics={key: result[key] for key in ('status', 'attempted', 'completed',
            'constructor_free_program_and_artifacts_accepted', 'linked_native_program_executed', 'full_generation_executed')},
        artifacts_dir=str(args.out), ts=datetime.now(timezone.utc).isoformat())
    with (args.registry or paths.RESULTS_DIR/'registry.jsonl').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX); stream.seek(0)
        if any(json.loads(line).get('run_id') == args.out.name for line in stream if line.strip()):
            raise ValueError('Retain original unique registry identity')
        stream.seek(0, os.SEEK_END); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    marker = persist_case_completion(args.out, round_label=ROUND, started_ns=started,
        status=result['status'], timing=result['timing'], required_artifact_bytes=artifact_bytes(args.out))
    if marker['case_artifact_bytes'] > CASE_BYTES:
        raise RuntimeError('Completion exceeded the reserved case allowance')


def check_consumed(consumed, manifest):
    if (consumed.get('status') != 'ok' or type(consumed.get('program_validations')) is not int or consumed['program_validations'] != 1
            or any(type(consumed.get(k)) is not int or consumed[k] != 27 for k in ('component_load_attempts','component_loads_completed'))
            or any(consumed.get(k) != 0 for k in ('payload_reads','context_constructions','parameter_constructions',
                'keygen_runs','RunComponent_calls','RunTwoTokens_calls','encryptions','decryptions','bootstraps'))
            or consumed.get('actual_library_source_AST', {}).get('component_calls_encoded') != 52
            or consumed.get('actual_library_source_AST', {}).get('boundary_boots_encoded') != 3
            or consumed.get('actual_library_source_AST', {}).get('slots') != 16384
            or consumed.get('actual_library_source_AST', {}).get('nominal_scale_log2') != 43
            or consumed.get('actual_library_source_AST', {}).get('context') != TYPE['context']
            or consumed.get('actual_library_source_AST', {}).get('ring_degree') != 65536
            or consumed.get('program_and_original27_artifacts_consumed') is not True
            or consumed.get('whole_generation_executed') is not False):
        raise ValueError('Original native metadata consumption or actual caller source proof is incomplete')
    program = manifest['program']; refs = [*program['Layers'], program['Head'], program['First'], program['Terminal']]
    expected = [dict(role=r['Role'], index=r['Index'], plan=r['PlanPath'], bundle=r['BundlePath'], source_case=r['SourceCase']) for r in refs]
    if consumed.get('components') != expected:
        raise ValueError('Original native loader did not consume exactly the same27 source-bound artifacts')


def execute(args):
    root = paths.RESULTS_DIR.resolve(); args.out = args.out.resolve()
    if args.out.exists() or args.out.is_symlink() or args.out.parent != root:
        raise ValueError('Use one fresh independent result case')
    started = time.perf_counter_ns(); deadline = time.monotonic()+TOTAL_SECONDS
    with serial_lock(root):
        args.out.mkdir(); recorder = StageRecorder(args.out/'events.jsonl'); stop = threading.Event(); active = [None]; failures = []
        result = dict(round=ROUND, mode=MODE, model='Qwen2-0.5B', status='running', error=None,
            attempted=dict(build=0, consume=0), completed=dict(build=0, consume=0), processes=[],
            constructor_free_program_and_artifacts_accepted=False, limits=dict(total_seconds=1800, phases=PHASE_SECONDS,
                AS_gib=4, RSS_gib=4, minimum_available_gib=2, minimum_free_disk_gib=10, case_gib=8, file_gib=8,
                concurrency=1, retries=0, monitor_seconds=.25), **dict.fromkeys(ZERO, 0))
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        def incoming(sig, frame):
            result['termination'] = dict(signal=sig, cause=failures[-1] if failures else 'incoming_signal')
            raise InterruptedError('Finite metadata-only component admission interrupted')
        for sig in old:
            signal.signal(sig, incoming)
        def check():
            if failures:
                raise RuntimeError(failures[-1])
            if time.monotonic() >= deadline:
                raise TimeoutError('Complete1800-second admission allowance exhausted')
            if artifact_bytes(args.out)+RESERVE > CASE_BYTES or shutil.disk_usage(args.out).free < 10*GIB:
                raise RuntimeError('Preserve8GiB case and10GiB disk reserve')
            observation = _physical(_latest_rss_sample(active[0]) if active[0] else None, _available_ram_bytes())
            if observation['stop']:
                result['resource_guard'] = observation; raise RuntimeError(str(observation))
        def monitor():
            while not stop.wait(.25):
                try:
                    check()
                except BaseException as exc:
                    failures.append(type(exc).__name__+': '+str(exc))
                    try:
                        durable_json(args.out/'guard-failure.json', dict(error=failures[-1]))
                    finally:
                        os.kill(os.getpid(), signal.SIGTERM)
                    return
        watcher = threading.Thread(target=monitor, daemon=True)
        try:
            signal.setitimer(signal.ITIMER_REAL, TOTAL_SECONDS); watcher.start(); check()
            with recorder.span('source-and-global-typed-link-admission'):
                checked = source_gate(args); result['source'] = checked; command = prepare(args.out, checked)
            result['status'] = 'prepared'; durable_json(args.out/'result.json', result)
            if args.run:
                result['initialization_source_audit'] = initialization_gate(args.initialization_audit, checked['library'])
                for name in ('module-cache', 'build-cache'):
                    check(); shutil.copytree(Path(checked['library']['case'])/'work'/name, args.out/'work'/name); check()
                commands = dict(build=command, consume=[str(args.out/'qwen-component-inspector'),
                    '--manifest', str(args.out/'global-link-manifest.json'), '--out', str(args.out/'native-consumption.json')])
                for phase, command in commands.items():
                    check(); result['attempted'][phase] += 1; durable_json(args.out/'result.json', result)
                    active[0] = args.out/(phase+'-resources.jsonl')
                    with recorder.span('constructor-free-'+phase, argv=command):
                        observed = run_process(command, cwd=args.out/'work/module', log=args.out/(phase+'.log'), resources=active[0],
                            timeout=min(PHASE_SECONDS[phase], max(.01, deadline-time.monotonic())), mem_gib=4, fsize_gib=8,
                            env=builder.environment(args.out))
                    active[0] = None; result['processes'].append(dict(phase=phase, argv=command, observation=observed))
                    durable_json(args.out/'result.json', result); builder.clean(observed)
                    if phase == 'consume':
                        consumed = read(args.out/'native-consumption.json'); result['native_consumption'] = consumed
                        check_consumed(consumed, checked['manifest'])
                    result['completed'][phase] += 1; check(); durable_json(args.out/'result.json', result)
                result['status'] = 'ok'
        except BaseException as exc:
            result.update(status='error', error=type(exc).__name__+': '+str(exc))
        finally:
            active[0] = None; stop.set()
            if watcher.is_alive():
                watcher.join(timeout=3)
            signal.setitimer(signal.ITIMER_REAL, 0)
            if failures:
                result.update(status='error', error=failures[-1])
            try:
                _close(args, result, started)
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
        return result


def parser():
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('source-case', 'probe-case', 'remaining-case', 'kernel-case', 'library-case', 'initialization-audit', 'out'):
        command.add_argument('--'+name, type=Path, required=True)
    command.add_argument('--registry', type=Path); command.add_argument('--run', action='store_true')
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    result = execute(args); print(json.dumps({k: result[k] for k in ('status','error','constructor_free_program_and_artifacts_accepted')}))
    return int(result['status'] not in ('ok', 'prepared'))


if __name__ == '__main__':
    raise SystemExit(main())
