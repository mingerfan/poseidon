"""S1 default-mode compatibility, standalone packages and bounded regression groups."""

from dataclasses import asdict, replace
import gc
import importlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import unittest

from .. import toolchain
from ..paths import RESULTS_DIR, WORK_ROOT, DACAPO_ROOT, HECATE_BUILD, PLATFORM_DIR
from .s1_managed import inventory, write


BASE = 'b7662b94'


def server_case(args):
    """§11.4 fallback: export both modes; acceptance awaits server artifacts."""
    import shlex
    import subprocess
    from ..dsl.parser import parse
    from ..export import export_package
    from ..models.info import ModelInfo
    from ..run import _server_job
    row = next(r for r in inventory() if r['label']==args.case)
    graph = ModelInfo.load(Path(row['info'])).graph
    source = Path(row['source']).read_text()
    program = parse(source)
    exported = args.out/'export'
    package = export_package(source,program,graph.weights,exported)
    weights = exported/'weights'/(program.name+'.npz')
    root = args.out/'server_jobs'/row['label']
    commands,dry_runs = {},{}
    for mode in ('compiler','managed'):
        script = _server_job(root/mode,package,weights,mode=mode)
        subprocess.run(['bash','-n',str(script)],check=True)
        commands[mode] = str(script.relative_to(root))
        output = args.out/('dry-run-'+mode)
        status = subprocess.run([sys.executable,'-B',str(package/'compile_model.py'),
            '--mode',mode,'--dry-run','--hecate-opt',str(toolchain.host_paths()['hecate_opt']),
            '--input',str(args.out/'trace'/('managed.mlir' if mode=='managed' else 'trace_model.mlir')),
            '--output-dir',str(output)],capture_output=True,text=True,check=True)
        dry_runs[mode] = json.loads(status.stdout)
    entry = root/'run.sh'
    entry.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        's1_job_root="$(cd -- "$(dirname -- "$0")" && pwd)"\n'+
        ''.join('bash "$s1_job_root"/'+shlex.quote(commands[mode])+'\n' for mode in ('compiler','managed')))
    entry.chmod(0o755)
    subprocess.run(['bash','-n',str(entry)],check=True)
    return dict(complete=True,round='S1',part='server',case=row,server_job=str(entry),
                commands=commands,compile_dry_runs=dry_runs,package=str(package),weights=str(weights),
                server_executed=False,local_completed=False,acceptance_complete=False,
                pending_comparisons=['stage0','managed-ir','runtime-plan'],
                planned_compiler_invocations=1,planned_managed_invocations=2,planned_preprocess_invocations=1,
                llm_calls=0,report_images=0)


def compression(args):
    """§11.4: plain/stdin/temporary-file pipelines with exact text comparison."""
    from types import SimpleNamespace
    from .s1_managed import case_inside
    from ..tests.test_s1_streams import compression_suite
    from ..manage import streams
    from unittest.mock import patch
    result = dict(round='S1',part='compression',complete=False,models={},llm_calls=0,report_images=0)
    for model in ('mlp','cnn_plain'):
        result['models'][model] = {}
        for mode in ('plain','gzip','temporary-file'):
            directory = args.out/model/mode
            directory.mkdir(parents=True)
            child = SimpleNamespace(out=directory,case=model,reference=None,compressed=mode!='plain',proof_text=True)
            try:
                limit = 0 if mode=='temporary-file' else 1024**3
                with patch.object(streams,'TEMP_INPUT_LIMIT',limit):
                    report = case_inside(child)
            except Exception as exc:
                report = dict(complete=False,error=f'{type(exc).__name__}: {exc}')
                write(directory/'REPORT.json',report)
                raise
            write(directory/'REPORT.json',report)
            result['models'][model][mode] = str(directory/'REPORT.json')
            write(args.out/'REPORT.json',result)
    outcome = unittest.TextTestRunner(verbosity=2,failfast=True).run(compression_suite(args.out))
    result.update(tests=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),
                  skips=len(outcome.skipped),complete=outcome.wasSuccessful())
    if not result['complete']:
        return result
    result['pruned_text'] = []
    for path in args.out.glob('*/*/proof-text/*.mlir*'):
        result['pruned_text'].append(str(path))
        path.unlink()
    return result


def emit_comparison(old_emit, old_lower, old_program, emit, lower, program, weights):
    """Pair constant calls as they are generated, avoiding two retained archives.

    Both emitters use their original constant naming/deduplication. Payloads are
    compared before each emitter releases them; no program arithmetic changes.
    """
    import numpy as np
    pending = queue.Queue(maxsize=1)
    finished = object()
    cancelled = threading.Event()
    old_result = {}
    acknowledgements = []
    compared = 0

    class Previous(old_emit):
        _keep_const_data = False

        def const(self, value):
            if cancelled.is_set():
                raise RuntimeError('Constant comparison cancelled')
            ready = threading.Event()
            acknowledgements.append(ready)
            pending.put((value,ready),timeout=120)
            if not ready.wait(120) or cancelled.is_set():
                raise RuntimeError('Constant comparison cancelled')
            return super().const(value)

    class Current(emit):
        _keep_const_data = False

        def const(self,value):
            nonlocal compared
            item = pending.get(timeout=120)
            if not isinstance(item,tuple):
                raise AssertionError('Constant call sequence differs')
            previous,ready = item
            try:
                if not np.array_equal(np.asarray(previous),np.asarray(value)):
                    raise AssertionError('Constant payload differs at call '+str(compared))
                compared += 1
            finally:
                ready.set()
            return super().const(value)

    def previous():
        try:
            backend = Previous(program.header.slots)
            old_lower(old_program,weights,backend)
            old_result['render'] = backend.render()
            pending.put(finished,timeout=120)
        except Exception as exc:
            old_result['error'] = exc
            try:
                pending.put(exc,timeout=1)
            except queue.Full:
                pass

    worker = threading.Thread(target=previous,daemon=True)
    worker.start()
    try:
        backend = Current(program.header.slots)
        lower(program,weights,backend)
        text,constants = backend.render()
        end = pending.get(timeout=120)
        worker.join(5)
        if 'error' in old_result:
            raise old_result['error']
        if end is not finished or worker.is_alive():
            raise AssertionError('Constant call sequence differs at completion')
        old_text,old_constants = old_result['render']
        if old_text!=text or list(old_constants)!=list(constants):
            raise AssertionError('Compiler Emit text or constant order differs')
        return dict(emit_text_equal=True,constants_equal=True,constant_order_equal=True,
                    constants=len(constants),constant_calls_compared=compared)
    finally:
        cancelled.set()
        for ready in acknowledgements:
            ready.set()
        worker.join(5)


def compatibility(args):
    from .u2_multi_layout import baseline_package
    from ..backends import EmitBackend
    from ..cost import estimate
    from ..dsl.parser import parse
    from ..dsl.printer import print_program
    from ..lower import lower
    from ..models.info import ModelInfo
    from ..models.zoo import get
    from ..planners import agent_tools, confirm
    from ..export.package import compile_template
    from ..tests.test_p2_export import cases, FROZEN
    for name,value in dict(POSEIDON_WORK_ROOT=WORK_ROOT,PT_AGENT_RESULTS_ROOT=RESULTS_DIR,
                           PT_DACAPO_ROOT=DACAPO_ROOT,PT_HECATE_BUILD=HECATE_BUILD).items():
        os.environ.setdefault(name,str(value))
    old = baseline_package(args.snapshot)
    old_parse = old('dsl.parser').parse
    rows = [dict(label=n,model=m,source=str(p)) for n,m,p in cases()]
    rows += [r for r in inventory() if r['label'].startswith('u2-')]
    u3 = json.loads((RESULTS_DIR/'u3a-complete-20261024T161354727454/REPORT.json').read_text())
    rows += [dict(label='u3a-'+r['model'],info=str(Path(r['acceptance']).parent),
                  source=str(Path(r['acceptance']).parent/'program.ptdsl')) for r in u3['models']]
    result = dict(complete=False,round='S1',part='compat',baseline_rev=BASE,programs=[],models=[],llm_calls=0,report_images=0)
    for row in rows:
        graph = ModelInfo.load(Path(row['info'])).graph if row.get('info') else get(row['model'],RESULTS_DIR/'models-v2').import_graph()
        text = Path(row['source']).read_text()
        a,b = old_parse(text),parse(text)
        comparison = emit_comparison(old('backends.emit').EmitBackend,old('lower').lower,a,
                                     EmitBackend,lower,b,graph.weights)
        if old('dsl.printer').print_program(a)!=print_program(b):
            raise AssertionError(row['label']+': default PT-DSL printing changed')
        mode = 'pars' if b.header.refresh=='agent' else 'dacapo-depth-dp'
        if old('toolchain').compile_argv(mode,'qwen48',Path('.'))!=toolchain.compile_argv(mode,'qwen48',Path('.')):
            raise AssertionError(row['label']+': compiler command changed')
        if old('export.package').compile_template(mode)!=compile_template(mode):
            raise AssertionError(row['label']+': package compiler command changed')
        result['programs'].append(dict(row,**comparison,default_print_equal=True,compiler_command_equal=True))
        write(args.out/'REPORT.json',result)
        print('COMPAT '+row['label']+' unchanged',flush=True)
        del graph,a,b
        gc.collect()
    for name in ('resnet20','resnet20_silu','cnn_plain','mlp'):
        model = get(name,RESULTS_DIR/'models-v2')
        graph = model.import_graph()
        source = (FROZEN/'models'/name/'program.ptdsl').read_text()
        calibration = json.loads((FROZEN/'models'/name/'calibration.json').read_text())
        a,b = old_parse(source),parse(source)
        cost_a = old('cost').estimate(a,graph.weights)
        cost_b = estimate(b,graph.weights)
        if cost_a!=cost_b or estimate(replace(b,header=replace(b.header,manage='dsl')),graph.weights)!=cost_b:
            raise AssertionError(name+': estimate changed')
        before = old('planners.agent_tools').build_context(graph,calibration,n_val=20,normalization=model.normalization)
        after = agent_tools.build_context(graph,calibration,n_val=20,normalization=model.normalization)
        validations = {}
        for engine in ('mock','tensor'):
            x = old('planners.agent_tools').validate(a,before,engine=engine,cache=False)
            y = agent_tools.validate(b,after,engine=engine,cache=False)
            if x!=y:
                raise AssertionError(name+': '+engine+' validate changed')
            validations[engine] = y
            print('COMPAT '+name+' '+engine+' validate unchanged',flush=True)
        previous_ctx = old('planners.confirm').build_confirm_context(before)
        current_ctx = confirm.build_confirm_context(after)
        for ctx in (previous_ctx,current_ctx):
            ctx.val_images,ctx.val_labels,ctx.val_exact_logits = ctx.val_images[:20],ctx.val_labels[:20],ctx.val_exact_logits[:20]
            if ctx.val_inputs is not None:
                ctx.val_inputs = {k:v[:20] for k,v in ctx.val_inputs.items()}
        def selected(module,context):
            candidate = module.Candidate(source,'frozen',0,cost_b['latency_pred_us'],cost_b['boots_pred'])
            value = module.confirm_select([candidate],context,max_evals=1)
            value['selected'] = asdict(value['selected']) if value['selected'] is not None else None
            return value
        previous_confirm = selected(old('planners.confirm'),previous_ctx)
        current_confirm = selected(confirm,current_ctx)
        if previous_confirm!=current_confirm:
            raise AssertionError(name+': confirmation changed')
        result['models'].append(dict(model=name,heldout_images=20,confirmation_images=20,
            estimate_equal=True,validate_equal=dict(mock=True,tensor=True),confirm_equal=True,
            estimate=cost_b,validations=validations,confirmation=current_confirm))
        write(args.out/'REPORT.json',result)
        del model,graph,before,after,previous_ctx,current_ctx
        old('tensor_sim').clear_caches()
        importlib.import_module('pt_agent.tensor_sim').clear_caches()
        gc.collect()
    result['complete'] = True
    return result


def package_case(args):
    from ..dsl.parser import parse
    from ..models.zoo import get
    from ..export import export_package
    from ..run import _compile_export
    from ..manage import read
    from .s1_managed import compare_ir
    from .s0_pipeline_spike import compare_plans
    row = next(r for r in inventory() if r['label']==args.case)
    if row.get('unit') or row.get('info'):
        raise ValueError('Use the specified frozen/zoo programs for the four package checks')
    graph = get(row['model'],RESULTS_DIR/'models-v2').import_graph()
    text = Path(row['source']).read_text()
    program = parse(text)
    result = dict(complete=False,round='S1',part='p2',case=row,modes={},llm_calls=0,report_images=0)
    toolchain.setup_env()
    for mode in ('compiler','managed'):
        directory = args.out/mode
        package = export_package(text,program,graph.weights,directory)
        weights = directory/'weights'/(program.name+'.npz')
        record = dict(package=str(package),weights=str(weights),mode=mode)
        if mode=='compiler':
            result['modes'][mode] = _compile_export(record)
            write(args.out/'REPORT.json',result)
            if result['modes'][mode]['compile'] is None or result['modes'][mode]['compile']['exit']:
                raise RuntimeError('P2 compiler mode failed')
            result['compiler_pruned'] = toolchain.prune(directory/'compiled',
                keep=('lowered._hecate_'+program.name+'.bundle',))
        else:
            # Split trace/compile here so stage 0 is checked and consumed first.
            from hecate_python_env import VENV
            trace = directory/'trace'
            traced = toolchain.trusted_run(directory,[str(VENV/'bin/python'),'-B',str(package/'trace_model.py'),
                '--mode','managed','--hecate-opt',str(toolchain.host_paths()['hecate_opt']),
                '--weights',str(weights),'--output-dir',str(trace)],directory/'trace.log',
                env_extra=toolchain.package_trace_env())
            result['modes'][mode] = dict(trace=traced)
            write(args.out/'REPORT.json',result)
            if traced['exit']:
                raise RuntimeError('P2 managed trace failed')
            profile = toolchain.management_profile()
            native = read(args.out/'compiler/trace/trace_model.mlir',profile)
            shared = read(trace/'preprocessed.earth.mlir',profile)
            result['stage0'] = compare_ir(native,shared)
            del native,shared
            gc.collect()
            write(args.out/'REPORT.json',result)
            if not result['stage0']['equal'] or result['stage0']['conclusion']!='相同':
                raise RuntimeError('P2 stage 0 differs')
            result['compiler_trace_pruned'] = toolchain.prune(args.out/'compiler/trace')
            result['managed_trace_pruned'] = toolchain.prune(trace,keep=('managed.mlir',))
            compiled = toolchain.trusted_run(directory,[str(VENV/'bin/python'),'-B',str(package/'compile_model.py'),
                '--mode','managed','--hecate-opt',str(toolchain.host_paths()['hecate_opt']),
                '--input',str(trace/'managed.mlir'),'--output-dir',str(directory/'compiled')],
                directory/'compile.log',env_extra=dict(LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
            result['modes'][mode]['compile'] = compiled
            write(args.out/'REPORT.json',result)
            if compiled['exit']:
                raise RuntimeError('P2 managed compile failed')
    result['plan'] = compare_plans(args.out/'compiler/compiled'/('lowered._hecate_'+program.name+'.runtime-plan.json'),
                                  args.out/'managed/compiled'/('lowered._hecate_'+program.name+'.runtime-plan.json'))
    write(args.out/'REPORT.json',result)
    if not result['plan']['equal']:
        raise RuntimeError('P2 RuntimePlan differs')
    result['pruned'] = {mode:{part:toolchain.prune(args.out/mode/part) for part in ('trace','compiled')}
                         for mode in ('compiler','managed')}
    result['complete'] = True
    return result


def regression(args):
    if args.group=='intake':
        python = PLATFORM_DIR/'venvs/intake-2.0.1-cpu/bin/python'
        if Path(sys.executable).absolute()!=python.absolute():
            os.execv(str(python),[str(python),'-B','-m','pt_agent.experiments.s1_managed',*sys.argv[1:]])
    files = sorted(Path('pt_agent/tests').glob('test_*.py'))
    package = {'test_p2_export.py','test_toolchain_decoupled.py'}
    tensor = {'test_r8f_tensor_validate.py'}
    weight = {'test_u3b2_weight_usage.py'}
    if args.group=='intake':
        files = [p for p in files if p.name=='test_u1_onnx.py']
    elif args.group in ('packaging','tensor','weight'):
        selected = dict(packaging=package,tensor=tensor,weight=weight)[args.group]
        files = [p for p in files if p.name in selected]
    elif args.group=='main':
        files = [p for p in files if p.name not in package|tensor|weight and not p.name.startswith('test_s1_')]
    else:
        raise ValueError('Choose one bounded regression group')
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('pt_agent.tests.'+p.stem) for p in files)
    expected = dict(main=435,packaging=17,tensor=17,weight=17,intake=4)
    if suite.countTestCases()!=expected[args.group]:
        raise RuntimeError('Existing regression inventory differs')
    result = unittest.TextTestRunner(verbosity=2,failfast=True).run(suite)
    return dict(complete=result.wasSuccessful(),round='S1',part='regression',group=args.group,
                tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),llm_calls=0)
