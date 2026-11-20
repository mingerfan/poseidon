"""Bounded compilation of DaCapo's unchanged CNN/Poly sources.

This entry exports a compiler artifact and an explicit, provisional native Boot
contract. It does not construct a CKKS context, run a model for quality, or use
the R29 Chebyshev program. The legacy HEAAN execution library is not required.
"""

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import signal
import sys
import threading
import time

from .r10_paper_metrics import StageRecorder, read_events, run_process, summarize_events, write_json
from .completion_metrics import artifact_bytes, persist_case_completion
from .r30_common_model_e2e import (DIAGNOSTIC_ELIDE_OPTION, diagnostic_compile_argv,
                                 compiler_output_variant, compile_artifact_metadata)


ROUND = 'DACAPO-ORIGINAL'
MODEL = 'dacapo-original-resnet20-silu'
MODE = 'original-source-native-profile-compile'
CONTEXT = 'dacapo-original-n17-r51-export'
SPEC_ID = 'dacapo-original-cpu-export-v1'
BOOT_ID = 'dacapo-original-l3-to16-s51-export'
POLY_FILES = ('treeStr.txt', 'coeffStr.txt', 'sgn151527.txt', 'tree15.txt', 'tree27.txt')
MODEL_SOURCES = {
    'resnet20': dict(result_model=MODEL, entry='ResNet', factory='resnet20_silu',
        checkpoint='resnet20.silu.model', r13_run='r13-common-resnet20-silu-20261115',
        default_case_gib=16, default_planning_gib=6),
    'vgg16': dict(result_model='dacapo-original-vgg16-silu-avgpool', entry='VGG16',
        factory='vgg16_silu_avgpool', checkpoint='vgg16_silu_avgpool_model',
        r13_run='r13-common-vgg16-silu-avgpool-20261115',
        default_case_gib=32, default_planning_gib=24),
    'alexnet': dict(result_model='dacapo-original-alexnet-silu-avgpool', entry='AlexNet',
        factory='alexnet_silu_avgpool', checkpoint='alexNet_silu_avgpool_model',
        r13_run='r13-common-alexnet-silu-avgpool-20261115',
        default_case_gib=32, default_planning_gib=24),
}


def model_source(model='resnet20'):
    if model not in MODEL_SOURCES:
        raise ValueError('Unknown original DaCapo model')
    return dict(MODEL_SOURCES[model])


def model_paths(upstream, model='resnet20'):
    """Infer official paths without importing a model or opening a checkpoint."""
    record = model_source(model)
    upstream = Path(upstream)
    return dict(script=upstream / 'examples/benchmarks' / (record['entry'] + '.py'),
        checkpoint=upstream / 'examples/data' / record['checkpoint'],
        model_source=upstream / 'python/poly/poly/models' / (record['entry'] + '.py'),
        trace_relative=Path('traced') / (record['entry'] + '.mlir'),
        factory='pt_agent.experiments.r13_common_models:' + record['factory'])

FORMAL_FIELDS = ('predicted_latency_seconds', 'actual_ckks_latency_seconds',
                 'accuracy_acceptance', 'security_acceptance')


def _read(path):
    return json.loads(Path(path).read_text())


def _metadata(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_compiler_profile(profile):
    expected = dict(runtime='HEAAN-HEVM', polynomialDegree=131072,
                    rescalingFactor=51, levelLowerBound=1, levelUpperBound=29,
                    bootstrapLevelLowerBound=3, bootstrapLevelUpperBound=16)
    if not isinstance(profile, dict) or any(profile.get(k) != v for k, v in expected.items()):
        raise ValueError('The original HEAAN CPU compiler profile has unexpected fields')
    if not isinstance(profile.get('latencyTable'), dict) or not profile['latencyTable']:
        raise ValueError('The original compiler latency table is missing')
    return expected


def make_export_contract(imported, compiler_profile, *, waterline=40):
    """Return a valid V2 export schema; concrete primes/context stay unverified.

    The old importer hard-codes Boot output scale40. The saved E4c native IR
    instead contains level3/scale51 -> level16/scale51. This provisional schema
    states that precise observed contract and is checked against the new plan.
    The initial scale remains the explicitly selected waterline (official40).
    """
    validate_compiler_profile(compiler_profile)
    if type(waterline) is not int or not 0 < waterline <= 51:
        raise ValueError('Waterline must be an integer from1 through51')
    if (imported.get('spec_format_version') != 2 or imported.get('spec_id') != 'dacapo-heaan-cpu-v1'
            or imported.get('target_id') != 'dacapo-heaan-cpu'
            or imported.get('context', {}).get('poly_degree') != 131072
            or imported.get('context', {}).get('rns_moduli_log2') != [51] * 30
            or imported.get('levels') != dict(lower_bound=1, upper_bound=29)):
        raise ValueError('Expected the existing imported HEAAN CPU V2 profile')
    old = imported.get('boot_profiles')
    required = dict(profile_id='dacapo-heaan-cpu-boot-v1', implementation='native',
                    input_level_min=3, input_level_max=16, input_components=2,
                    output_level=16, output_scale_log2=40, output_components=2)
    if (not isinstance(old, list) or len(old) != 1
            or any(old[0].get(k) != v for k, v in required.items())
            or old[0].get('host_requirements') != dict(needs_secret_key=False, needs_host_compute=False)):
        raise ValueError('The imported native Boot profile differs from its source schema')
    if set(old[0]) != set(required) | {'latency_us_by_input_level', 'noise_by_input_level', 'host_requirements'}:
        raise ValueError('The V2 native Boot profile fields differ from the reader schema')
    for field in ('latency_us_by_input_level', 'noise_by_input_level'):
        values = old[0][field]
        if values is not None and (not isinstance(values, list) or len(values) != 30
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError('Invalid imported native Boot cost metadata')
    derived = copy.deepcopy(imported)
    derived.update(spec_id=SPEC_ID, target_id='dacapo-original-cpu-export')
    derived['context'].update(context_id=CONTEXT, default_scale_log2=waterline)
    derived['boot_profiles'][0].update(profile_id=BOOT_ID, output_scale_log2=51)
    # Imported cost tables describe the original external HEAAN measurements.
    # Keep them as source metadata; this driver never reports them as local time.
    notes = dict(boot_profile_shape_checked=True, full_runtime_verifier_executed=False,
                 native_execution_ready=False, concrete_primes_verified=False,
                 imported_default_scale_log2=imported['context'].get('default_scale_log2'),
                 initial_waterline_log2=waterline, compiler_rescaling_factor_log2=51,
                 imported_boot_output_scale_log2=40, provisional_boot_output_scale_log2=51,
                 boot_scale_evidence='E4c ResNet optimized/compile.log:2; ckks poly<2*51*3> -> poly<2*51*16>',
                 export_only=True, imported_profile_unchanged=True, operator_spec_digest_verified=False,
                 imported_provenance_digest_is_source_metadata_only=True,
                 cost_tables_source='official HEAAN CPU profile; external measurements, not local calibration',
                 official_readme_example_waterline=40, native_lattigo_binding_pending=True,
                 compiler_level_upper_bound=29, source_lowering_initial_remaining_level=16,
                 existing_r27_candidate='N17/max16/scale51; verify actual exported value envelope before reuse')
    return derived, notes


def validate_source_record(record, *, upstream, checkpoint, model='resnet20'):
    selected = model_paths(upstream, model)
    factory = selected['factory']
    source = (record.get('worker') or {}).get('model_source') or {}
    if (record.get('round') != 'R13' or record.get('status') != 'ok' or record.get('model') != factory
            or source.get('factory') != factory
            or Path(source.get('checkpoint', '')).resolve() != Path(checkpoint).resolve()
            or Path(source.get('original_source', '')).resolve() != selected['model_source'].resolve()):
        raise ValueError('Saved original ModelInfo source differs from the official selected checkpoint or definition')
    return dict(factory=factory, checkpoint=str(Path(checkpoint).resolve()),
                original_model_source=source['original_source'], prior_times_included=False,
                training_preprocessing_verified=source.get('training_preprocessing_verified'))


def compiler_argv(template, *, binary, source, profile, output, waterline=40,
                  diagnostic_elide_final_ir=False):
    """Use a previously recorded required identifier without computing a digest."""
    if not isinstance(template, list) or not all(isinstance(v, str) for v in template):
        raise ValueError('Compiler metadata template must be a list of strings')
    identifiers = [v for v in template if v.startswith('--runtime-plan-operator-spec-sha256=')]
    if len(identifiers) != 1 or not re.fullmatch(
            r'--runtime-plan-operator-spec-sha256=sha256:[0-9a-f]{64}', identifiers[0]):
        raise ValueError('Expected one existing syntactically valid native metadata identifier')
    if template.count('--runtime-plan-format=json') != 1 or template.count('--runtime-plan-bundle-format=files') != 1:
        raise ValueError('Expected the recorded JSON/files export template')
    command = [str(binary), str(source), '--dacapo', '--boot-placement=optimized',
            '--ckks-config=' + str(profile), '--waterline=' + str(waterline),
            '--output-val=10', '--threshold=0.5', '--dump-pass-pipeline',
            '--mlir-disable-threading', '--verify-each', '--mlir-timing',
            '--runtime-plan-format=json', '--runtime-plan-bundle-format=files',
            '--runtime-plan-id=1', '--runtime-plan-target-id=dacapo-original-cpu-export',
            '--runtime-plan-capability-version=1', '--runtime-plan-operator-spec-id=' + SPEC_ID,
            '--runtime-plan-operator-spec-version=1', identifiers[0],
            '--runtime-plan-context-id=' + CONTEXT, '--runtime-plan-device-count=0',
            '--runtime-plan-ntt=true', '--runtime-plan-boot-implementation=native',
            '--runtime-plan-boot-profile=' + BOOT_ID,
            '--runtime-plan-inline-payload-max-bytes=4096', '--runtime-plan-report-io',
            '-o', str(output)]
    return diagnostic_compile_argv(command, elide_final_ir=diagnostic_elide_final_ir)


def observe_plan(plan, contract):
    """Inspect real format1 output, retaining counts when a contract differs."""
    if plan.get('format_version') != 1:
        raise ValueError('Expected a format1 RuntimePlan')
    values = {str(v['id']): v for v in plan.get('values', [])}
    if len(values) != len(plan.get('values', [])) or not values:
        raise ValueError('Empty or duplicate RuntimePlan values')
    profile = contract['boot_profiles'][0]
    rows, counts, mismatches = [], {}, []
    for ordinal, op in enumerate(plan.get('execution', [])):
        instruction_kind = op.get('kind')
        if instruction_kind not in ('compute', 'encode', 'allocate', 'transfer', 'release', 'fence'):
            raise ValueError('Unexpected format1 execution instruction')
        kind = op.get('op') if instruction_kind == 'compute' else instruction_kind
        if not isinstance(kind, str) or not kind:
            raise ValueError('Compute operation name is missing')
        counts[kind] = counts.get(kind, 0) + 1
        if instruction_kind != 'compute' or kind != 'boot':
            continue
        inputs = op.get('inputs')
        output = op.get('output')
        if not isinstance(inputs, list) or len(inputs) != 1 or str(inputs[0]) not in values or str(output) not in values:
            raise ValueError('Malformed Boot value references')
        a, b, attrs = values[str(inputs[0])], values[str(output)], op.get('attrs') or {}
        passed = (profile['input_level_min'] <= a['level'] <= profile['input_level_max']
                  and a['components'] == profile['input_components']
                  and b['level'] == attrs.get('target_level') == profile['output_level']
                  and b['scale_log2'] == attrs.get('target_scale_log2') == profile['output_scale_log2']
                  and b['components'] == attrs.get('target_components') == profile['output_components']
                  and attrs.get('operator_profile') == profile['profile_id']
                  and attrs.get('implementation') == 'native'
                  and a.get('context') == b.get('context') == CONTEXT)
        row = dict(instruction_ordinal=ordinal, input=inputs[0], output=output,
                   input_level=a['level'], input_scale_log2=a['scale_log2'],
                   output_level=b['level'], output_scale_log2=b['scale_log2'],
                   input_components=a['components'], output_components=b['components'],
                   attrs=attrs, contract_compatible=passed)
        rows.append(row)
        if not passed:
            mismatches.append(ordinal)
    return dict(execution_instructions=len(plan.get('execution', [])), value_count=len(values),
                op_counts=counts, bootstrap_count=len(rows), bootstrap_positions=rows,
                value_levels=sorted({v['level'] for v in values.values()}),
                value_scales_log2=sorted({v['scale_log2'] for v in values.values()}),
                value_components=sorted({v['components'] for v in values.values()}),
                all_value_levels_within_r27_max16=all(type(v['level']) is int and 0 <= v['level'] <= 16 for v in values.values()),
                export_contract_compatible=bool(rows) and not mismatches,
                contract_mismatch_ordinals=mismatches, native_execution_ready=False,
                concrete_prime_binding_verified=False, original_plan_modified=False)


def _clean_process(observation):
    return (observation.get('exit_code') == 0 and not observation.get('timed_out')
            and observation.get('signal') is None and not observation.get('signals_sent')
            and not observation.get('remaining_pids'))


@contextmanager
def _budget_guard(out, *, case_gib, minimum_free_gib):
    """Interrupt the supervisor; its existing cleanup then reaps the child tree."""
    stop, violations, samples = threading.Event(), [], Path(out) / 'artifact-resources.jsonl'
    def check():
        total, free = artifact_bytes(out), shutil.disk_usage(out).free
        row = dict(ts_ns=time.perf_counter_ns(), artifact_bytes=total, disk_free_bytes=free)
        with samples.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
            stream.flush()
        if total > int(case_gib * 1024**3) or free < int(minimum_free_gib * 1024**3):
            raise RuntimeError('Case artifact or remaining-disk allowance exhausted')
        return row
    check()
    # run_process installs its own temporary handlers. Its restore point returns
    # to this handler, covering the small main-thread gaps between child phases.
    def interrupt(signum, frame):
        raise RuntimeError('Case supervisor interrupted by resource monitor or SIGTERM')
    old_handler = signal.signal(signal.SIGTERM, interrupt)
    def monitor():
        while not stop.wait(.5):
            try:
                check()
            except BaseException as exc:
                violations.append(str(exc))
                os.kill(os.getpid(), signal.SIGTERM)
                return
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield violations
    finally:
        stop.set()
        thread.join()
        try:
            check()
        finally:
            signal.signal(signal.SIGTERM, old_handler)


def trace_worker(args):
    """Execute only the trusted original source, with its real __file__."""
    from ..models.dacapo_builders import _torchvision_imports
    work = Path(args.worker_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / 'traced').mkdir(exist_ok=True)
    old_cwd, old_argv, old_path = Path.cwd(), list(sys.argv), list(sys.path)
    old_env = {k: os.environ.get(k) for k in ('HECATE', 'HECATE_BUILD', 'PATH')}
    result = dict(status='running', trace_attempts=1, trace_completed=0,
                  file_gib=getattr(args,'file_gib',8),
                  expected_checkpoint_reads_in_original_function=1, checkpoint_reads_observed=None,
                  source_unmodified=True, quality_model_forwards=0, raw_cifar_reads=0,
                  original_script=str(Path(args.source).resolve()))
    events = StageRecorder(work / 'events.jsonl')
    hc = None
    started = time.perf_counter_ns()
    old_alarm_handler = signal.getsignal(signal.SIGALRM)
    old_alarm = signal.getitimer(signal.ITIMER_REAL)
    def trace_expired(signum, frame):
        raise TimeoutError('Original source trace allowance exhausted')
    try:
        signal.signal(signal.SIGALRM, trace_expired)
        signal.setitimer(signal.ITIMER_REAL, args.worker_seconds)
        os.environ.update(HECATE=str(Path(args.frontend).resolve()),
                          HECATE_BUILD=str(Path(args.compiler).resolve().parents[1]))
        sys.path[:0] = [str(Path(args.frontend).resolve() / 'python/hecate'),
                       str(Path(args.poly_root).resolve()), str(Path(args.einops).resolve())]
        os.chdir(work)
        sys.argv = [str(Path(args.source).resolve())]
        with events.span('original.trace.runpy'):
            with _torchvision_imports():
                runpy.run_path(str(Path(args.source).resolve()), run_name='__main__')
        hc = sys.modules.get('hecate')
        path = work / 'traced' / (Path(args.source).stem + '.mlir')
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError('Original hc.save did not produce ' + str(path.relative_to(work)))
        result.update(status='ok', trace_completed=1, mlir=_metadata(path))
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm_handler)
        if old_alarm[0]:
            signal.setitimer(signal.ITIMER_REAL, *old_alarm)
        hc = hc or sys.modules.get('hecate')
        if hc is not None and getattr(hc, 'ctxt', None):
            try:
                hc.lt.finalize(hc.ctxt)
                result['frontend_finalized'] = True
            except BaseException as exc:
                result.update(status='error', finalize_error=f'{type(exc).__name__}: {exc}')
        sys.argv, sys.path = old_argv, old_path
        os.chdir(old_cwd)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        result['elapsed_ns'] = time.perf_counter_ns() - started
        write_json(work / 'worker_result.json', result)
    return 0 if result['status'] == 'ok' else 1


def compile_worker(args):
    """Run the existing Nix-linked compiler inside its existing pure environment."""
    work = Path(args.worker_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', compile_attempts=0, compile_completed=0,
                  file_gib=getattr(args,'file_gib',8),
                  compiler_output_variant=compiler_output_variant(args),
                  diagnostic_elide_final_ir=getattr(args,'diagnostic_elide_final_ir',False))
    recorder = StageRecorder(work / 'events.jsonl')
    began = time.perf_counter_ns()
    try:
        command = _read(args.command_file)
        if (not isinstance(command, list) or not all(isinstance(a, str) for a in command)
                or command[0] != str(Path(args.compiler).resolve())
                or '--dacapo' not in command or '--boot-placement=optimized' not in command
                or '--runtime-plan-boot-profile=' + BOOT_ID not in command):
            raise ValueError('Unexpected original compiler command')
        chosen=getattr(args,'diagnostic_elide_final_ir',False)
        if command.count(DIAGNOSTIC_ELIDE_OPTION)!=(1 if chosen else 0):
            raise ValueError('The saved compiler command and diagnostic output choice differ')
        if chosen:
            original=[arg for arg in command if arg!=DIAGNOSTIC_ELIDE_OPTION]
            if diagnostic_compile_argv(original,elide_final_ir=True)!=command:
                raise ValueError('Only the reviewed diagnostic printer option may adapt the saved command')
        result['compile_attempts'] = 1
        write_json(work / 'worker_result.json', result)
        with recorder.span('original.compiler.native'):
            observation = run_process(command, cwd=work, log=work / 'native.log',
                resources=work / 'native-resources.jsonl', timeout=args.worker_seconds,
                mem_gib=args.memory_gib,fsize_gib=getattr(args,'file_gib',8),
                env=dict(os.environ, LD_LIBRARY_PATH=os.environ['HECATE_PYTHON_LIBRARY_PATH']))
            result['process'] = observation
            if not _clean_process(observation):
                raise RuntimeError('Original native compiler failed or left a child process')
            result['compile_completed'] = 1
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        result['elapsed_ns'] = time.perf_counter_ns() - began
        write_json(work / 'worker_result.json', result)
    return 0 if result['status'] == 'ok' else 1


def _nix_worker(args, phase, work, *, command_file=None):
    seconds = args.trace_seconds if phase == 'trace' else args.compile_seconds
    inner = ['-m', 'pt_agent.experiments.dacapo_original_compile', '--worker-' + phase,
             '--model', getattr(args, 'model', 'resnet20'),
             '--worker-dir', str(work), '--worker-seconds', str(seconds),
             '--compiler', str(args.compiler), '--memory-gib', str(args.memory_gib)]
    if getattr(args,'file_gib',8)!=8:inner += ['--file-gib',str(args.file_gib)]
    if phase == 'trace':
        inner += ['--source', str(args.source), '--frontend', str(args.frontend),
                  '--poly-root', str(args.poly_root), '--einops', str(args.einops)]
    else:
        inner += ['--command-file', str(command_file)]
        if getattr(args,'diagnostic_elide_final_ir',False):inner += ['--diagnostic-elide-final-ir']
    return [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', str(seconds + 60),
            '--python', str(args.intake_python), '--', *inner]


def _register(registry, result, out, revision):
    row = dict(run_id=out.name, round=ROUND, model=result['model'], mode=MODE, git_rev=revision,
               metrics={k: result.get(k) for k in ('status', 'trace_attempts', 'trace_completed',
                  'compile_attempts', 'compile_completed', 'native_compile_attempts', 'native_compile_completed', 'bootstrap_count',
                         'export_contract_compatible', 'native_execution_ready', *FORMAL_FIELDS)},
               artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat())
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        if any(json.loads(line).get('run_id') == out.name for line in stream if line.strip()):
            raise RuntimeError('Run id already exists in registry')
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    from .. import paths
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter_ns()
    recorder = StageRecorder(out / 'events.jsonl')
    selected_model = getattr(args, 'model', 'resnet20')
    selected = model_source(selected_model)
    result = dict(round=ROUND, model=selected['result_model'], mode=MODE, status='running',
                  original_model_selector=selected_model,
                  trace_attempts=0, trace_completed=0, compile_attempts=0, compile_completed=0,
                  native_compile_attempts=0, native_compile_completed=0,
                  bootstrap_count=None, export_contract_compatible=None, native_execution_ready=False,
                  original_source_variant=True, r29_cheb63_variant=False,
                  api_calls=0, raw_cifar_reads=0, confirm_samples=0, report_samples=0,
                  keygen_runs=0, real_ckks_attempts=0, real_ckks_completed=0,
                  downloads=0, installations=0, extra_integrity_checks=0,
                  diagnostic_elide_final_ir=getattr(args,'diagnostic_elide_final_ir',False),
                  compiler_output_variant=compiler_output_variant(args),
                  limits=dict(file_gib=getattr(args,'file_gib',8),memory_gib=getattr(args,'memory_gib',None),
                      case_gib=args.case_gib,minimum_free_disk_gib=10),
                  **{k: None for k in FORMAL_FIELDS})
    work, revision = out / 'work', 'unknown'
    work.mkdir()
    try:
        with recorder.span('source.preflight'):
            compiler_profile = _read(args.compiler_profile)
            imported = _read(args.imported_spec)
            contract, notes = make_export_contract(imported, compiler_profile, waterline=args.waterline)
            template = _read(args.metadata_template)
            for path in (args.source, args.checkpoint, args.compiler, args.frontend / 'python/hecate/hecate/expr.py',
                         args.compiler.parent.parent / 'lib/libHecateFrontend.so', args.poly_root / 'poly/Poly.py',
                         args.poly_root / 'poly/Func.py', args.poly_root / 'poly/MPCB.py',
                         args.einops / 'einops/__init__.py'):
                if not Path(path).is_file():
                    raise FileNotFoundError(str(path))
            expected = model_paths(args.upstream, selected_model)
            if Path(args.source).resolve() != expected['script'].resolve():
                raise ValueError('Original source must be the selected upstream benchmark')
            if Path(args.checkpoint).resolve() != expected['checkpoint'].resolve():
                raise ValueError('Original source checkpoint differs from its relative getModel path')
            if Path(args.poly_root).resolve() != Path(args.upstream / 'python/poly').resolve():
                raise ValueError('Original polynomial root must come from the same upstream checkout')
            original_model = validate_source_record(_read(args.source_r13_result),
                                                    upstream=args.upstream, checkpoint=args.checkpoint, model=selected_model)
            polynomial_sources = [args.poly_root / 'poly/data' / name for name in POLY_FILES]
            if any(not p.is_file() for p in polynomial_sources):
                raise FileNotFoundError('An original Poly import-time coefficient file is missing')
            result['sources'] = dict(script=_metadata(args.source), checkpoint=_metadata(args.checkpoint),
                saved_r13_result=_metadata(args.source_r13_result), original_model=original_model,
                compiler_profile=_metadata(args.compiler_profile), imported_spec=_metadata(args.imported_spec),
                native_frontend=_metadata(args.compiler.parent.parent / 'lib/libHecateFrontend.so'),
                polynomial_files=[_metadata(p) for p in polynomial_sources],
                native_constructor_pending='Existing R27 is N17/max16/scale51. Official profile upper29, but Common.cpp selects init_level16; inspect actual exported levels before judging reuse. Concrete prime binding and execution remain pending.',
                parameter_provenance='Original HEAAN compiler profile; concrete Q/P and native security unverified')
            write_json(out / 'export-contract.json', contract)
            write_json(out / 'export-contract-notes.json', notes)
            write_json(out / 'compiler-config.json', compiler_profile)
            snapshots = out / 'source'
            snapshots.mkdir()
            shutil.copyfile(__file__, snapshots / Path(__file__).name)
            for path in (args.source, args.poly_root / 'poly/MPCB.py', args.poly_root / 'poly/Poly.py', args.poly_root / 'poly/Func.py'):
                shutil.copyfile(path, snapshots / path.name)
            if shutil.disk_usage(out).free < (args.planning_gib + 10) * 1024**3:
                raise RuntimeError('Insufficient disk for the declared planning allowance and10GiB reserve')
            result['source_checked'] = True
        git_obs = run_process(['git', 'rev-parse', 'HEAD'], cwd=paths.REPO, log=out / 'git.log',
                              resources=out / 'git-resources.jsonl', timeout=15, mem_gib=.5)
        if _clean_process(git_obs):
            revision = (out / 'git.log').read_text().strip()
        trace_work = work / 'trace'
        trace_work.mkdir()
        trace_argv = _nix_worker(args, 'trace', trace_work)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with _budget_guard(out, case_gib=args.case_gib, minimum_free_gib=10) as violations:
            with recorder.span('original.trace'):
                result['trace_attempts'] = 1
                write_json(out / 'result.json', result)
                observation = run_process(trace_argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts',
                    log=out / 'trace.log', resources=out / 'trace-resources.jsonl',
                    timeout=args.trace_seconds + 120, mem_gib=args.memory_gib,
                    fsize_gib=getattr(args,'file_gib',8),env=env)
                result['trace_process'] = observation
                result['trace_argv'] = trace_argv
                record = trace_work / 'worker_result.json'
                result['trace_worker'] = _read(record) if record.is_file() else None
                if violations or not _clean_process(observation) or not result['trace_worker'] or result['trace_worker'].get('status') != 'ok':
                    raise RuntimeError('Original source trace failed or left a child process')
                result['trace_completed'] = 1
            compile_work = work / 'compile'
            compile_work.mkdir()
            command = compiler_argv(template, binary=args.compiler,
                source=trace_work / expected['trace_relative'], profile=out / 'compiler-config.json',
                output=compile_work / 'lowered.mlir', waterline=args.waterline,
                diagnostic_elide_final_ir=getattr(args,'diagnostic_elide_final_ir',False))
            write_json(out / 'compile-argv.json', command)
            compile_argv = _nix_worker(args, 'compile', compile_work, command_file=out / 'compile-argv.json')
            result['compiler_metadata_identifier_scope'] = 'Inherited syntactic identifier from the recorded template; no new digest calculation or binding validation'
            with recorder.span('original.compile.optimized'):
                result['compile_attempts'] = 1
                write_json(out / 'result.json', result)
                observation = run_process(compile_argv, cwd=paths.REPO / 'DSL/DSL_agent/scripts', log=out / 'compile.log',
                    resources=out / 'compile-resources.jsonl', timeout=args.compile_seconds + 120,
                    mem_gib=args.memory_gib,fsize_gib=getattr(args,'file_gib',8),env=env)
                result['compile_process'] = observation
                record = compile_work / 'worker_result.json'
                result['compile_worker'] = _read(record) if record.is_file() else None
                result['native_compile_attempts'] = ((result['compile_worker'] or {}).get('compile_attempts')
                    if result['compile_worker'] is not None else None)
                result['native_compile_completed'] = ((result['compile_worker'] or {}).get('compile_completed')
                    if result['compile_worker'] is not None else None)
                if (violations or not _clean_process(observation) or not result['compile_worker']
                        or result['compile_worker'].get('status') != 'ok'):
                    raise RuntimeError('Native DaCapo optimized compilation failed')
                result['compile_completed'] = 1
            with recorder.span('original.export.inspect'):
                plans = list(compile_work.glob('lowered.*.runtime-plan.json'))
                if len(plans) != 1:
                    raise RuntimeError('Expected exactly one exported original RuntimePlan')
                result['plan'] = _metadata(plans[0])
                compiled_plan = _read(plans[0])
                result['compiler_output'] = compile_artifact_metadata(compile_work, plans[0], compiled_plan,
                    elided=getattr(args,'diagnostic_elide_final_ir',False))
                observation = observe_plan(compiled_plan, contract)
                result.update(observation)
                write_json(out / 'plan-observation.json', observation)
                if not observation['export_contract_compatible']:
                    raise RuntimeError('Compilation completed; exported Boot metadata differs from the provisional contract')
        result['status'] = 'ok'
    except BaseException as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        timing = summarize_events(read_events(out / 'events.jsonl'), time.perf_counter_ns())
        result['timing'] = timing
        result['timing_scopes'] = dict(outer='Exclusive serial outer phase spans; complete endpoint includes required persistence and registry',
            worker='Worker event files and native child timers are nested observations; do not add them to outer inclusive time')
        result['artifact_bytes_before_result'] = artifact_bytes(out)
        result['recorded_elapsed_seconds'] = (time.perf_counter_ns() - started) / 1e9
        write_json(out / 'result.json', result)
        _register(args.registry, result, out, revision)
        completion = persist_case_completion(out, round_label=ROUND, started_ns=started,
            status=result['status'], timing=timing, required_artifact_bytes=artifact_bytes(out))
        if completion['case_artifact_bytes'] > args.case_gib * 1024**3:
            raise RuntimeError('Terminal artifact allowance exhausted; raw result and completion retained')
    return 0 if result['status'] == 'ok' else 1


def main(argv=None):
    from .. import paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--model', choices=tuple(MODEL_SOURCES), default='resnet20')
    parser.add_argument('--worker-trace', action='store_true')
    parser.add_argument('--worker-compile', action='store_true')
    parser.add_argument('--worker-dir', type=Path)
    parser.add_argument('--worker-seconds', type=int, default=1800)
    parser.add_argument('--command-file', type=Path)
    parser.add_argument('--upstream', type=Path, default=paths.DACAPO_UPSTREAM)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--compiler', type=Path, default=paths.HECATE_BUILD / 'bin/hecate-opt')
    parser.add_argument('--frontend', type=Path, default=paths.DACAPO_ROOT)
    parser.add_argument('--poly-root', type=Path)
    parser.add_argument('--einops', type=Path, default=paths.PLATFORM_DIR / 'deps/poly-python/einops-0.6.1')
    parser.add_argument('--intake-python', type=Path, default=paths.INTAKE_PYTHON)
    parser.add_argument('--compiler-profile', type=Path)
    parser.add_argument('--imported-spec', type=Path, default=paths.REPO / 'third_party/ckks-runtime/docs/operator-spec/v2/profiles/dacapo-heaan-cpu.v1.json')
    parser.add_argument('--metadata-template', type=Path, default=paths.RESULTS_DIR / 'E5d-agent-sessions-20261106/bert/acceptance/pars/export/Bert_PTAgent_DSL/config/compile_argv.json')
    parser.add_argument('--source-r13-result', type=Path)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--waterline', type=int, default=40)
    parser.add_argument('--memory-gib', type=float, default=12)
    parser.add_argument('--file-gib',type=float,default=8,
        help='Finite per-file allowance in GiB; default8, maximum16; case and free-disk guards remain active')
    parser.add_argument('--case-gib', type=float)
    parser.add_argument('--planning-gib', type=float)
    parser.add_argument('--trace-seconds', type=int, default=1800)
    parser.add_argument('--compile-seconds', type=int, default=7200)
    parser.add_argument('--diagnostic-elide-final-ir',action='store_true',
        help='Print large non-splat attributes as diagnostic placeholders; preserve the full executable RuntimePlan and bundle')
    args = parser.parse_args(argv)
    if not 0<args.file_gib<=16:parser.error('Use a finite positive per-file allowance at most16GiB')
    if args.worker_trace and args.worker_compile:
        parser.error('Select exactly one worker phase')
    if args.worker_compile:
        if args.worker_dir is None or args.command_file is None or not 0 < args.worker_seconds <= 7200:
            parser.error('Compile worker needs explicit directory/command and at most7200 seconds')
        return compile_worker(args)
    if args.worker_trace:
        if any(getattr(args, k) is None for k in ('worker_dir', 'source', 'poly_root')) or not 0 < args.worker_seconds <= 1800:
            parser.error('Trace worker needs explicit --worker-dir, --source and --poly-root')
        return trace_worker(args)
    if args.out is None:
        parser.error('--out is required')
    selected = model_source(args.model)
    args.case_gib = args.case_gib if args.case_gib is not None else selected['default_case_gib']
    args.planning_gib = args.planning_gib if args.planning_gib is not None else selected['default_planning_gib']
    if (not 0 < args.memory_gib <= 12 or not 0 < args.case_gib <= 32
            or not 0 < args.trace_seconds <= 1800 or not 0 < args.compile_seconds <= 7200
            or not 0 < args.planning_gib <= args.case_gib):
        parser.error('Resource allowances exceed the bounded original-baseline preparation')
    selected_paths = model_paths(args.upstream, args.model)
    args.source = args.source or selected_paths['script']
    args.checkpoint = args.checkpoint or selected_paths['checkpoint']
    args.source_r13_result = args.source_r13_result or paths.RESULTS_DIR / selected['r13_run'] / 'result.json'
    args.poly_root = args.poly_root or args.upstream / 'python/poly'
    args.compiler_profile = args.compiler_profile or args.upstream / 'profiled_HEAAN_CPU.json'
    args.registry = args.registry or paths.RESULTS_DIR / 'registry.jsonl'
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
