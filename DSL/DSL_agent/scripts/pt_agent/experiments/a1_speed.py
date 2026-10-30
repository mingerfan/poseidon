"""A1 §6.10 serial, offline cost equivalence and speed measurements.

The system-Python parent owns the finite inventory and registry. Each old/new
worker receives explicit paths and runs in the existing bounded Nix launcher.
No provider, compiler, report-set or model-generation entry point is used.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
from itertools import zip_longest
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def inventory(results):
    """Reuse the saved finite cohorts, retaining every historical program."""
    results = Path(results)
    models = dict(bert=results/'A1-bert-prepare-20261027T123333-7967/model',
                  qwen=results/'A1-qwen-prepare-20261027T132722-12c2/model')
    rows = []

    def add(label, group, **fields):
        rows.append(dict(id=f'case-{len(rows):03d}', label=label, group=group, **fields))

    for model, directory in models.items():
        for baseline in ('R0', 'R1'):
            add(model+'-'+baseline, 'baselines', model=model, info=str(directory),
                source=str(directory.parent/baseline/'program.ptdsl'), normalized=True)
    replay = read(results/'A1-bert-replay-20261027T160313-a92e/REPLAY.json')
    if len(replay['rows']) != 26:
        raise ValueError('The retained r8-v7 cohort must contain 26 programs')
    for row in replay['rows']:
        add('v7-call-'+str(row['call']).zfill(3), 'v7', model='bert', info=str(models['bert']),
            source=row['source'], normalized=True, historical_stage=row['new_stage'])
    sessions = [('A1-bert-session-20261027T165801-fd6d', 0, 8),
                ('A1-bert-session-20261028T033706-5ca7', 8, 6)]
    for session, offset, count in sessions:
        for index in range(1, count+1):
            add('v8-call-'+str(offset+index).zfill(3), 'v8', model='bert',
                info=str(models['bert']), source=str(results/session/'sources'/f'iteration-{index}.ptdsl'),
                normalized=True)
    stopped = read(results/'A1-transformers-s68-stop-20261028T060310-e9da/REPORT.json')
    for row in stopped['development_tunes']:
        add('v8-tuned-'+str(row['model_call']).zfill(3), 'v8-tuned', model='bert',
            info=str(models['bert']), source=row['source'], normalized=True)
    retained = read(results/'u4-cost78-inventory-20261026T155556805705/MANIFEST.json')
    collection = [row for row in retained['cases'] if any(g['group']=='s1' for g in row['groups'])]
    if len(collection) != 20:
        raise ValueError('Collection M must retain its 20 existing cases')
    for row in collection:
        fields = {key:row[key] for key in ('model', 'source', 'info', 'unit') if key in row}
        if 'model' in fields and 'info' not in fields:
            fields['models_dir'] = str(results/'models-v2')
        add('M-'+row['label'], 'M', normalized=False, **fields)
    for row in rows:
        for key in ('source', 'info', 'models_dir'):
            if key in row and not Path(row[key]).exists():
                raise FileNotFoundError(row[key])
    return dict(round='A1', section='6.10', complete=True, cases=rows,
                counts={group:sum(row['group']==group for row in rows)
                        for group in ('baselines', 'v7', 'v8', 'v8-tuned', 'M')},
                frozen_models_in_M=['resnet20','resnet20_silu','cnn_plain','mlp'],
                llm_calls=0, report_images=0, compile_calls=0)


def select_source(source_root):
    """Load one implementation, retaining only the current location provider.

    A copied package has a different physical parent, so paths.py alone is
    imported from the checkout before selecting its implementation directory.
    All computational modules are then imported from the selected snapshot.
    """
    import pt_agent
    import pt_agent.paths
    root = Path(source_root).resolve()
    if not (root/'cost.py').is_file():
        raise ValueError('source-root must contain the pt_agent package')
    if root.parts[-4:] != ('DSL','DSL_agent','scripts','pt_agent'):
        raise ValueError('A retained implementation requires a checkout/DSL/DSL_agent/scripts/pt_agent '
                         'directory layout for its unchanged relative resource lookups')
    loaded = [name for name in sys.modules if name.startswith('pt_agent.')
              and name not in ('pt_agent.paths',)]
    if loaded:
        raise RuntimeError('Computational modules already loaded before source selection: '+str(loaded))
    pt_agent.__path__ = [str(root)]


def memory():
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:')):
            key, value, _ = line.split()
            fields[key.rstrip(':')+'_bytes'] = int(value)*1024
    fields['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    return fields


QWEN_PEAK_LIMIT = int(3.8*1024**3)


def process_memory(pid):
    """Read one known worker, without summing its launcher or other processes."""
    result = {}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'VmSize:')):
            key, value, _ = line.split()
            result[key.rstrip(':')+'_bytes'] = int(value)*1024
    return result


def process_identity(pid):
    # /proc/<pid>/stat's second field may contain spaces and parentheses.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def dag_records(backend):
    """Ordered values and edges, including node/constant naming; no digests."""
    yield dict(record='boundary', inputs=[value.nid for value in backend._inputs],
               outputs=[value.nid for value in backend._outputs],
               constants=[plain.name for plain in backend._constants.values()])
    for index, (kind, args) in enumerate(zip(backend.kind, backend.args)):
        row = dict(record='node', id=index, kind=kind, args=args,
                   owner=backend.prog_op[index], depth=backend._depth[index],
                   total_depth=backend._total_depth[index])
        if hasattr(backend, 'operands'):
            row.update(operands=backend.operands[index], step=backend.steps[index])
        yield row


def dump_dag(path, backend):
    with gzip.open(path, 'wt', compresslevel=1) as stream:
        for row in dag_records(backend):
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')


@contextmanager
def cost_observer(directory):
    from unittest.mock import patch
    from pt_agent import cost
    original_lower, original_placement = cost.lower, cost.depth_dp
    result = dict(observer_seconds=0.0, stages={})

    def record(phase, backend):
        began = time.monotonic()
        path = directory/(phase+'.jsonl.gz')
        dump_dag(path, backend)
        result['observer_seconds'] += time.monotonic()-began
        result['stages'][phase] = dict(path=str(path), nodes=len(backend.kind), **memory())

    def lower(*args, **kwargs):
        began = time.monotonic()
        answer = original_lower(*args, **kwargs)
        result['lower_seconds'] = time.monotonic()-began
        record('raw-dag', args[2])
        return answer

    def placement(backend, *args, **kwargs):
        record('final-dag', backend)
        began = time.monotonic()
        answer = original_placement(backend, *args, **kwargs)
        result['placement_seconds'] = time.monotonic()-began
        return answer

    with patch.object(cost, 'lower', lower), patch.object(cost, 'depth_dp', placement):
        yield result


def graph_for(row):
    if row.get('info'):
        from pt_agent.models.info import ModelInfo
        info = ModelInfo.load(row['info'])
        return info.graph, info
    from pt_agent.models.zoo import get
    return get(row['model'], Path(row['models_dir'])).import_graph(), None


def unit_estimate(row, directory):
    """The five M kernel fixtures have no PT-DSL source; replay their saved definition."""
    from pt_agent.backends import DagBackend
    from pt_agent.cost import load_latency
    from pt_agent.placement import depth_dp
    from pt_agent.tests.test_s1_manage import record_case
    from pt_agent.experiments.s1_managed import replay
    recorded, _ = record_case(row['unit'])
    backend = DagBackend(recorded.slots)
    backend.op_context = 12
    replay(recorded, backend)
    for phase in ('raw-dag', 'final-dag'):
        dump_dag(directory/(phase+'.jsonl.gz'), backend)
    placement = depth_dp(backend, last_boot_level=4, boot_in_level=1, boot_out_level=5,
                         latency_table=load_latency('qwen48'))
    from dataclasses import asdict
    return dict(kind='M-direct-DAG-fixture', stats=backend.stats(), placement=asdict(placement))


def measure(row, directory, *, direct_estimate=False):
    from dataclasses import asdict
    from pt_agent.dsl.parser import parse
    from pt_agent.dsl.typecheck import check
    from pt_agent.cost import estimate
    result = dict(complete=True, stage='parse', cost=None)
    if 'unit' in row:
        began = time.monotonic()
        result.update(stage='complete', cost=unit_estimate(row, directory),
                      estimate_seconds=time.monotonic()-began)
        return result
    graph, info = graph_for(row)
    text = Path(row['source']).read_text()
    try:
        program = parse(text)
    except Exception as exc:
        result['rejection'] = dict(type=type(exc).__name__, message=str(exc))
        return result
    if row['group'] in ('baselines', 'v7', 'v8', 'v8-tuned'):
        from pt_agent.dsl.interface import check_interface
        interface = check_interface(program, info)
        if direct_estimate:
            result['interface_diagnostic'] = interface
        if not interface['ok'] and not direct_estimate:
            result.update(stage='interface', rejection=interface)
            return result
    checked = check(program, {name:value.shape for name,value in graph.weights.items()},
                    **(dict(graph=graph, prompt_version='r8-v8')
                       if row['group'] in ('baselines', 'v7', 'v8', 'v8-tuned') and not direct_estimate else {}))
    if not checked.ok:
        result.update(stage='typecheck', rejection=[asdict(value) for value in checked.diags])
        return result
    began = time.monotonic()
    with cost_observer(directory) as observed:
        try:
            measured = estimate(program, graph.weights, frontend_normalize=row['normalized'])
        except Exception as exc:
            result.update(stage='estimate', rejection=dict(type=type(exc).__name__, message=str(exc)))
        else:
            result.update(stage='complete', cost=measured)
    result.update(estimate_seconds=time.monotonic()-began-observed['observer_seconds'],
                  observed=observed)
    return result


class EvaluationClock:
    """Experiment-only budget: count cost evaluations without changing decisions."""
    def __init__(self, budget_s, evaluations=None):
        self.budget_s, self.evaluations = budget_s, evaluations
        self.calls = 0
        self.started = None
        self.check_lines = set()
        self.checks = 0
        self.budget_breaks = 0
        self.estimator_calls = 0

    def monotonic(self):
        value = (self.budget_s+1 if self.calls >= self.evaluations else 0.0
                 ) if self.evaluations is not None else time.monotonic()
        if self.started is None:
            self.started = value
        frame = sys._getframe(1)
        if frame.f_lineno in self.check_lines and frame.f_code.co_name == 'tune':
            self.checks += 1
            if value-self.started >= self.budget_s:
                self.budget_breaks += 1
        return value

    def cost(self, estimator):
        def counted(*args, **kwargs):
            self.calls += 1
            return estimator(*args, **kwargs)
        return counted


@contextmanager
def tuner_observer(evaluations=None, *, candidates=None, directory=None, memory_limit=None):
    """Observe calls and bounded candidate fragments without changing the tuner."""
    import inspect
    from unittest.mock import patch
    from pt_agent.planners import transformer_tuner
    clock = EvaluationClock(2400, evaluations)
    lines, start = inspect.getsourcelines(transformer_tuner.tune)
    clock.check_lines = {start+i for i,line in enumerate(lines)
                         if 'if time.monotonic()-started>=budget_s:break' in line}
    original = transformer_tuner.cached_estimate
    original_validate = transformer_tuner.validate
    observer = CandidateObserver(clock, candidates, directory, memory_limit)

    def cached(program, weights, profile, estimator, **kwargs):
        from pt_agent.tensor_sim import cache_info
        phase, frame = tune_phase(program)
        observer.begin_cost(phase, program, frame)
        clock.calls += 1
        began = time.monotonic()
        misses = cache_info()['estimate']['misses']
        try:
            # Estimator identity participates in the existing cache key. Pass
            # it through unchanged; cache hits must remain hits in this probe.
            return original(program, weights, profile, estimator, **kwargs)
        finally:
            clock.estimator_calls += (cache_info()['estimate']['misses']-misses
                                      if kwargs.get('cache',True) else 1)
            observer.component('cost', phase, time.monotonic()-began)

    def validated(program, *args, **kwargs):
        phase, _ = tune_phase(program)
        return observed_validation(original_validate, observer, program, phase, *args, **kwargs)

    with patch.object(transformer_tuner, 'time', SimpleNamespace(monotonic=clock.monotonic)), \
         patch.object(transformer_tuner, 'cached_estimate', cached), \
         patch.object(transformer_tuner, 'validate', validated):
        try:
            yield observer
        finally:
            observer.finish_candidate()
            observer.save()


def tune_phase(program):
    """Identify the frozen tuner's existing starting/alignment/candidate calls."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_globals.get('__name__') == 'pt_agent.planners.transformer_tuner':
            if frame.f_code.co_name == 'finish':
                return 'finish', frame
            if frame.f_code.co_name == 'tune':
                values = frame.f_locals
                if values.get('candidate') is program:
                    return 'candidate', frame
                if values.get('aligned') is program:
                    return 'alignment', frame
                if values.get('current') is program:
                    return 'initial', frame
                raise RuntimeError('Unknown Transformer tuner cost/validation phase')
        frame = frame.f_back
    raise RuntimeError('Cost/validation observer called outside Transformer tuner')


class CandidateLimitReached(Exception):
    def __init__(self, state):
        self.state = state
        super().__init__('The registered candidate fragment is complete')


class CandidateMemoryExceeded(Exception):
    pass


def observed_validation(validate, observer, program, phase, *args, **kwargs):
    """Retain diagnostics without touching the returned feedback or decision."""
    from pt_agent.domain_criteria import capture_violations
    observer.check_memory()
    result, error = None, None
    coarse = bool(args[1:] and args[1] is not None)
    with capture_violations() as diagnostics:
        began = time.monotonic()
        try:
            result = validate(program, *args, **kwargs)
            return result
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            seconds = time.monotonic()-began
            observer.record_validation(program, phase, result, error, diagnostics,
                                       seconds=seconds, engine=kwargs.get('engine'), coarse=coarse)
            observer.component('validation', phase, seconds,
                               engine=kwargs.get('engine'), coarse=coarse)


def tuner_state(frame):
    from copy import deepcopy
    from pt_agent.dsl.printer import print_program
    values = frame.f_locals
    return deepcopy(dict(source=print_program(values['current']), changes=values['changes'],
        cost_before=values['before'], cost_after=values['current_cost'],
        coarse_evals=values['coarse'], full_evals=values['full'], history=values['history'],
        feasible=values['current_result']['feasible']))


class CandidateObserver:
    """Count candidate calls even when cached; exclude starting/alignment calls."""
    def __init__(self, clock, limit=None, directory=None, memory_limit=None):
        self.clock, self.limit, self.directory, self.memory_limit = clock, limit, directory, memory_limit
        self.rows, self.stages, self.active = [], [], None
        self.monitoring_peak = 0
        self.validations, self.last_validation = [], None
        self.serialization_seconds = 0.0

    def check_memory(self):
        measured = memory()
        self.monitoring_peak = max(self.monitoring_peak, measured['VmHWM_bytes'])
        if self.memory_limit is not None and measured['VmHWM_bytes'] > self.memory_limit:
            self.save()
            raise CandidateMemoryExceeded('Qwen single-process VmHWM exceeds 3.8 GiB: '+str(measured))
        return measured

    def begin_cost(self, phase, program, frame):
        self.check_memory()
        if phase != 'candidate':
            return
        self.finish_candidate()
        if self.limit is not None and len(self.rows) >= self.limit:
            raise CandidateLimitReached(tuner_state(frame))
        self.active = dict(candidate=len(self.rows)+1, cost_calls=0, validation_calls=0,
                           wall_seconds=0.0, components=[],
                           wall_scope='cost call plus required validation calls',
                           memory_scope='single-process cumulative VmHWM at each component', **memory())

    def component(self, kind, phase, seconds, **fields):
        measured = memory()
        record = dict(kind=kind, phase=phase, seconds=seconds, **fields, **measured)
        self.stages.append(record)
        if phase == 'candidate' and self.active is not None:
            self.active[kind+'_calls'] += 1
            self.active['wall_seconds'] += seconds
            self.active['components'].append(record)
            for key in ('VmRSS_bytes', 'VmHWM_bytes', 'peak_rss_bytes'):
                self.active[key] = max(self.active.get(key,0), measured.get(key,0))
        self.save()
        self.check_memory()

    def finish_candidate(self):
        if self.active is not None:
            self.rows.append(self.active)
            self.active = None
            self.save()

    def record_validation(self, program, phase, result, error, diagnostics, **measurement):
        """Write the exact candidate and full per-sample chain before any stop."""
        from pt_agent.dsl.printer import print_program
        from pt_agent.planners.transformer_session import json_safe
        began = time.monotonic()
        number = len(self.validations)+1
        candidate = self.active['candidate'] if phase=='candidate' and self.active is not None else None
        name = f'{number:04d}-{phase}'+(f'-{candidate:03d}' if candidate is not None else '')
        directory = Path(self.directory)/'validations'/name if self.directory is not None else None
        record = dict(validation=number, phase=phase, candidate=candidate, **measurement,
                      returned=result is not None, error=error, diagnostics=diagnostics,
                      result_summary={key:result.get(key) for key in
                        ('feasible','diverged','analytic_domain_violation','max_u_calibrated','max_u_analytic')}
                        if result is not None else None)
        source = print_program(program)
        if directory is not None:
            directory.mkdir(parents=True,exist_ok=False)
            (directory/'program.ptdsl').write_text(source)
            record.update(program=str(directory/'program.ptdsl'), report=str(directory/'REPORT.json'))
            write(directory/'REPORT.json',json_safe(record))
        else:
            record['program_text'] = source
        record = json_safe(record)
        self.last_validation = record
        self.validations.append({key:record.get(key) for key in
            ('validation','phase','candidate','program','report','seconds','engine','coarse','returned','error')})
        self.serialization_seconds += time.monotonic()-began
        save_started = time.monotonic()
        self.save()
        self.serialization_seconds += time.monotonic()-save_started

    def save(self):
        if self.directory is not None:
            write(Path(self.directory)/'EVALUATIONS.json', dict(candidates=self.rows,
                active_candidate=self.active, stages=self.stages,
                cost_calls=self.clock.calls, estimator_cache_misses=self.clock.estimator_calls,
                candidate_limit=self.limit, memory_limit=self.memory_limit,
                count_scope='candidate cost calls, including cache hits; starting/alignment calls excluded',
                validations=self.validations, last_validation=self.last_validation,
                diagnostic_serialization_seconds=self.serialization_seconds,
                max_observed_VmHWM_bytes=self.monitoring_peak))


def tune_context(row):
    from pt_agent.models.info import ModelInfo
    from pt_agent.planners.agent_tools import build_context
    from pt_agent.data.samples import StoredSplits
    if row.get('model') not in ('bert', 'qwen') or not row.get('info'):
        raise ValueError('A1 speed tune requires a retained Transformer program')
    info = ModelInfo.load(row['info'])
    samples = StoredSplits(Path(row['info'])/'samples', info)
    context = build_context(info, read(Path(row['info'])/'calibration.json'), samples=samples)
    context.prompt_version = 'r8-v8'
    return context


def measure_tune(row, directory, evaluations=None, candidates=None):
    from dataclasses import asdict
    from pt_agent.dsl.parser import parse
    from pt_agent.planners.tuner import tune
    from pt_agent.planners.transformer_session import json_safe
    context = tune_context(row)
    program = parse(Path(row['source']).read_text())
    began = time.monotonic()
    with tuner_observer(evaluations, candidates=candidates, directory=directory,
                        memory_limit=QWEN_PEAK_LIMIT if candidates is not None else None) as observed:
        # A fragment ends before the eleventh candidate call. Its diagnostic
        # clock does not let differing CPU speed truncate either implementation.
        if candidates is not None:
            observed.clock.evaluations = float('inf')
        try:
            tuned = tune(program, context, budget_s=2400, max_sweeps=1,
                         engine_profile='tensor', cache=evaluations is None)
        except CandidateLimitReached as stop:
            result = stop.state
            result['seconds'] = 0.0
            fragment_complete = True
        else:
            result = {key:value for key,value in asdict(tuned).items() if key != 'program'}
            fragment_complete = candidates is None or len(observed.rows)+(observed.active is not None)==candidates
        observed.check_memory()
    wall = time.monotonic()-began
    (directory/'program.ptdsl').write_text(result['source'])
    return json_safe(dict(complete=fragment_complete, stage='tune', tune=result, tune_wall_seconds=wall,
        budget=dict(kind='candidate-evaluations' if candidates is not None else 'evaluations' if evaluations is not None else 'wall',
                    seconds=2400, evaluations=evaluations, candidates=candidates, cache=evaluations is None),
        cost_evaluations=observed.clock.calls, estimator_cache_misses=observed.clock.estimator_calls,
        candidate_evaluations=observed.rows, budget_checks=observed.clock.checks,
        validations=observed.validations, diagnostic_serialization_seconds=observed.serialization_seconds,
        budget_breaks=observed.clock.budget_breaks,
        complete_sweep=candidates is None and observed.clock.checks>0 and observed.clock.budget_breaks==0,
        fragment=candidates is not None,
        development_only=True, frozen=False))


class AlignmentCaptured(Exception):
    pass


def recheck_alignment(row, directory):
    """Observe exactly the unchanged domain alignment and stop before scanning."""
    from unittest.mock import patch
    from pt_agent.dsl.parser import parse
    from pt_agent.dsl.printer import print_program
    from pt_agent.planners import transformer_tuner
    from pt_agent.planners.transformer_session import json_safe
    from pt_agent.domain_criteria import capture_violations, require_analytic_coverage
    context = tune_context(row)
    program = parse(Path(row['source']).read_text())
    original = transformer_tuner.validate
    captured = {}

    def validation(candidate, *args, **kwargs):
        phase, _ = tune_phase(candidate)
        if phase != 'alignment':
            return original(candidate, *args, **kwargs)
        with capture_violations() as diagnostics:
            result = original(candidate, *args, **kwargs)
        captured.update(validation=result, diagnostics=diagnostics,
                        aligned_source=print_program(candidate))
        (directory/'aligned.ptdsl').write_text(captured['aligned_source'])
        write(directory/'ALIGNMENT.json', json_safe(captured))
        require_analytic_coverage(result)
        raise AlignmentCaptured()

    with patch.object(transformer_tuner, 'validate', validation):
        try:
            transformer_tuner.tune(program, context, budget_s=2400, max_sweeps=1,
                                   engine_profile='tensor')
        except AlignmentCaptured:
            pass
        else:
            raise RuntimeError('The registered alignment validation was not reached')
    return json_safe(dict(complete=True, stage='alignment', **captured,
                         candidate_scan_started=False, llm_calls=0, report_images=0))


def first_difference(left, right, path='$'):
    if type(left) is not type(right):
        return dict(path=path, before=left, after=right, reason='type')
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return dict(path=path, before=sorted(left), after=sorted(right), reason='keys')
        for key in left:
            found = first_difference(left[key], right[key], path+'.'+key)
            if found:
                return found
    elif isinstance(left, list):
        if len(left) != len(right):
            return dict(path=path, before=len(left), after=len(right), reason='length')
        for index, (x, y) in enumerate(zip(left, right)):
            found = first_difference(x, y, path+f'[{index}]')
            if found:
                return found
    elif left != right:
        return dict(path=path, before=left, after=right, reason='value')
    return None


def compare_dag(before, after):
    if not before.exists() or not after.exists():
        return dict(equal=not before.exists() and not after.exists(), rows=0,
                    reason='both absent for rejected program' if not before.exists() and not after.exists()
                    else 'only one DAG exists')
    with gzip.open(before, 'rt') as a, gzip.open(after, 'rt') as b:
        count = 0
        for count, (x, y) in enumerate(zip_longest(a, b), 1):
            if x is None or y is None:
                return dict(equal=False, rows=count, difference=dict(row=count, reason='length'))
            difference = first_difference(json.loads(x), json.loads(y))
            if difference:
                return dict(equal=False, rows=count, difference=dict(row=count, **difference))
    return dict(equal=True, rows=count)


def compare_case(directory):
    before, after = (read(directory/side/'REPORT.json') for side in ('before', 'after'))
    keys = ('stage', 'cost', 'rejection', 'interface_diagnostic')
    difference = first_difference({key:before.get(key) for key in keys},
                                  {key:after.get(key) for key in keys})
    dags = {phase:compare_dag(directory/'before'/(phase+'.jsonl.gz'),
                             directory/'after'/(phase+'.jsonl.gz'))
            for phase in ('raw-dag', 'final-dag')}
    complete = before['complete'] and after['complete'] and difference is None and all(v['equal'] for v in dags.values())
    return dict(complete=complete, cost_and_rejection_equal=difference is None,
                first_difference=difference, dags=dags,
                before={key:before.get(key) for key in ('stage','estimate_seconds','memory','observed')},
                after={key:after.get(key) for key in ('stage','estimate_seconds','memory','observed')})


def compare_tune(directory, evaluations, candidates=None):
    before, after = (read(directory/side/'REPORT.json') for side in ('before','after'))
    # A time budget deliberately permits more moves after the speedup; only the
    # explicitly requested evaluation-count experiment requires exact equality.
    required = evaluations is not None or candidates is not None
    difference = first_difference(before.get('tune'), after.get('tune')) if required else None
    candidate_counts = (len(before.get('candidate_evaluations',[])), len(after.get('candidate_evaluations',[])))
    count_ok = candidates is None or candidate_counts == (candidates,candidates)
    return dict(complete=before['complete'] and after['complete'] and difference is None and count_ok,
                equivalence_required=required, candidate_counts=candidate_counts,
                candidate_count_matches=count_ok,
                first_difference=difference,
                before={key:before.get(key) for key in ('tune_wall_seconds','cost_evaluations',
                    'estimator_cache_misses','candidate_evaluations',
                    'budget_checks','budget_breaks','complete_sweep','memory','tune')},
                after={key:after.get(key) for key in ('tune_wall_seconds','cost_evaluations',
                    'estimator_cache_misses','candidate_evaluations',
                    'budget_checks','budget_breaks','complete_sweep','memory','tune')})


def worker(args):
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 6*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    write(args.out/'WORKER.json', dict(pid=os.getpid(), start_ticks=process_identity(os.getpid())))
    select_source(args.source_root)
    row = next(row for row in read(args.manifest)['cases'] if row['id']==args.case)
    began = time.monotonic()
    try:
        result = (measure_tune(row, args.out, args.evaluations, args.candidate_evaluations) if args.part == 'tune'
                  else recheck_alignment(row, args.out) if args.part == 'alignment'
                  else measure(row, args.out, direct_estimate=args.direct_estimate))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = dict(complete=False, error=dict(type=type(exc).__name__, message=str(exc)))
        for name in ('ALIGNMENT', 'EVALUATIONS'):
            if (args.out/(name+'.json')).exists():
                result[name.lower()] = read(args.out/(name+'.json'))
    origins = {name:str(Path(module.__file__).resolve())
               for name,module in sys.modules.items()
               if name.startswith('pt_agent.') and getattr(module,'__file__',None)}
    mixed = {name:path for name,path in origins.items() if name!='pt_agent.paths'
             and not Path(path).is_relative_to(args.source_root.resolve())}
    if mixed:
        result.update(complete=False, mixed_implementation_modules=mixed)
    result.update(case=row, source_root=str(args.source_root), seconds=time.monotonic()-began,
                  implementation_modules=origins, paths_only_from_checkout=True,
                  memory=memory(), llm_calls=0, report_images=0, compile_calls=0)
    write(args.out/'REPORT.json', result)
    return 0 if result['complete'] else 1


def run_worker(command, target, *, cwd, memory_limit=None):
    """Bounded launch plus optional Qwen single-worker 3.8-GiB peak monitoring."""
    with (target/'run.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        exceeded = None
        while True:
            try:
                code = process.wait(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                pass
            marker = target/'WORKER.json'
            if memory_limit is None or exceeded is not None or not marker.exists():
                continue
            try:
                identity = read(marker)
                pid = identity['pid']
                if process_identity(pid) != identity['start_ticks']:
                    continue
                measured = process_memory(pid)
            except (FileNotFoundError, ProcessLookupError, json.JSONDecodeError):
                continue
            if measured.get('VmHWM_bytes',0) > memory_limit:
                exceeded = dict(complete=False, error='Qwen single-process VmHWM exceeds 3.8 GiB',
                                worker=identity, memory=measured, limit_bytes=memory_limit)
                write(target/'MEMORY_STOP.json', exceeded)
                try:
                    os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if exceeded is not None:
            if (target/'REPORT.json').exists():
                exceeded['worker_report'] = read(target/'REPORT.json')
            if (target/'EVALUATIONS.json').exists():
                exceeded['evaluations'] = read(target/'EVALUATIONS.json')
            write(target/'REPORT.json', exceeded)
            return code or 1
        return code


def run_cases(args):
    from pt_agent.paths import REPO
    manifest = read(args.manifest)
    selected = [row for row in manifest['cases'] if not args.case or row['id'] in args.case.split(',')]
    if args.group:
        selected = [row for row in selected if row['group'] in args.group.split(',')]
    if not selected:
        raise ValueError('No inventory entries selected')
    if args.candidate_evaluations is not None and any(row.get('model')!='qwen' for row in selected):
        raise ValueError('The ten-candidate fragment is registered only for Qwen')
    if args.part=='tune' and args.evaluations is None and args.candidate_evaluations is None:
        if any(row['label']!='v8-call-013' for row in selected):
            raise ValueError('The full 2400-second measurement is registered only for BERT call 13')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    script = Path(__file__).resolve()
    rows = []
    for row in selected:
        directory = args.out/row['id']
        if (directory/'COMPARISON.json').exists():
            saved = read(directory/'COMPARISON.json')
            if not args.resume or not saved['complete']:
                raise RuntimeError('Refusing to rerun retained result '+str(directory))
            rows.append(dict(saved, case=row))
            continue
        sides = (('after',args.after),) if args.part=='alignment' else (('before',args.before),('after',args.after))
        for side, source in sides:
            target = directory/side
            if (target/'REPORT.json').exists():
                if not args.resume or not read(target/'REPORT.json')['complete']:
                    raise RuntimeError('Refusing to overwrite retained worker '+str(target))
                continue
            target.mkdir(parents=True, exist_ok=False)
            free = shutil.disk_usage(target).free
            if free < 10*1024**3:
                stopped = dict(complete=False, error='disk admission below 10 GiB',
                               disk_free_bytes=free, required_bytes=10*1024**3,
                               case=row, side=side, worker_started=False)
                write(target/'STOP.json', stopped)
                write(target/'REPORT.json', stopped)
                result = dict(complete=False, cases=rows, stopped=stopped)
                write(args.out/'REPORT.json', result)
                return result
            command = [sys.executable, '-B', '-m', 'pt_agent.tools.nixrun', '--seconds', '3600', '--',
                       str(script), '--inside', '--part', args.part, '--manifest', str(args.manifest),
                       '--case', row['id'], '--source-root', str(source), '--out', str(target)]
            if args.evaluations is not None:
                command += ['--evaluations', str(args.evaluations)]
            if args.candidate_evaluations is not None:
                command += ['--candidate-evaluations', str(args.candidate_evaluations)]
            if args.direct_estimate:
                command += ['--direct-estimate']
            write(target/'argv.json', command)
            print('COST '+row['label']+' '+side, flush=True)
            code = run_worker(command,target,cwd=REPO/'DSL/DSL_agent/scripts',
                              memory_limit=QWEN_PEAK_LIMIT if args.candidate_evaluations is not None else None)
            if code:
                result = dict(complete=False, cases=rows, stopped=dict(case=row, side=side,
                              exit=code, report=str(target/'REPORT.json')))
                write(args.out/'REPORT.json', result)
                return result
        compared = (compare_tune(directory, args.evaluations, args.candidate_evaluations) if args.part == 'tune'
                    else read(directory/'after/REPORT.json') if args.part == 'alignment' else compare_case(directory))
        write(directory/'COMPARISON.json', compared)
        rows.append(dict(compared, case=row))
        result = dict(complete=len(rows)==len(selected) and all(r['complete'] for r in rows), cases=rows,
                      selected=len(selected), finished=len(rows), llm_calls=0, report_images=0, compile_calls=0)
        write(args.out/'REPORT.json', result)
        if not compared['complete']:
            return result
    # A resume may contain only already-complete comparisons, so no worker loop
    # assigned a new summary. Rebuild it from retained rows without recomputing.
    result = dict(complete=len(rows)==len(selected) and all(r['complete'] for r in rows), cases=rows,
                  selected=len(selected), finished=len(rows), llm_calls=0, report_images=0, compile_calls=0)
    write(args.out/'REPORT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('inventory','compare','tune','alignment'), required=True)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--direct-estimate', action='store_true',
                        help='Retain interface diagnostics without gating the static estimate contract')
    parser.add_argument('--case')
    parser.add_argument('--group')
    parser.add_argument('--evaluations', type=int)
    parser.add_argument('--candidate-evaluations', type=int, choices=(10,))
    for name in ('out','manifest','before','after','source-root','results'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.evaluations is not None and args.evaluations < 2:
        parser.error('An evaluation budget must allow the initial and alignment costs')
    if args.evaluations is not None and args.candidate_evaluations is not None:
        parser.error('Choose either cost-call or candidate-call counting')
    if args.inside:
        return worker(args)
    from pt_agent.paths import RESULTS_DIR
    from pt_agent.registry import append, git_rev, new_run_id
    run_id = new_run_id('A1', 'offline', 'speed-'+args.part)
    args.out = args.out or RESULTS_DIR/run_id
    os.environ['PT_RESULTS'] = str(RESULTS_DIR)
    if args.part == 'inventory':
        args.out.mkdir(parents=True, exist_ok=False)
        result = inventory(args.results or RESULTS_DIR)
        write(args.out/'MANIFEST.json', result)
    else:
        if args.after is None or args.manifest is None or args.part!='alignment' and args.before is None:
            parser.error('A measurement requires --after and --manifest; comparisons also require --before')
        result = run_cases(args)
    write(args.out/'REPORT.json', result)
    append(dict(run_id=run_id, round='A1', model='offline', mode='speed-'+args.part,
                git_rev=git_rev(), metrics=result, artifacts_dir=str(args.out),
                ts=datetime.now(timezone.utc).isoformat()))
    print('REPORT='+str(args.out/'REPORT.json'), flush=True)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    # Direct-file workers intentionally avoid importing the current package's
    # experiment module before selecting the old/new computational package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
