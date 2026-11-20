"""S6: replace provider replies with one deterministic Rule63 seed."""
import argparse
import ast
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import time

from . import s5_tuner_off as off
from . import r30_common_model_e2e as cnn
from . import bert_fresh_cold_3h as bert
from .r10_paper_metrics import write_json

MODULE = 'pt_agent.experiments.s6_no_llm'
TUNER_SECONDS = 2400
TUNE_RULE = 'v8-6.25'


def rule_seed(ctx):
    from ..planners.rule import plan
    text = plan(ctx.graph, ctx.calibration, deg=63, margin=1.5, slots=16384,
                placement='depth-dp', refresh='compiler', act_method='cheb')
    if getattr(ctx.graph, 'intake_stats', {}).get('transformer_normalized'):
        from .a1_bert_fixed_current import rule63_source
        text, _ = rule63_source(text)
    return text


class RuleClient:
    """One local replay reply, no provider construction, credentials or ledger."""
    def __init__(self, context, directory, recorder=None):
        self.context, self.directory, self.recorder = context, Path(directory), recorder
        self.calls, self.replay = 0, None

    def chat(self, messages, *, purpose):
        from contextlib import nullcontext
        from .e9b_cnn_sessions import ReplayClient
        from ..dsl.parser import parse
        from .a1_bert_fixed_current import recipe_metadata
        if self.replay is None:
            ctx = self.context()
            if ctx is None:
                raise ValueError('The original model context must precede the local seed')
            span = self.recorder.span('offline_rule63_seed') if self.recorder else nullcontext()
            with span:
                text = rule_seed(ctx)
                self.directory.mkdir(parents=True, exist_ok=True)
                (self.directory / 'initial-rule63.ptdsl').write_text(text)
                write_json(self.directory / 'initial-rule63-recipe.json', recipe_metadata(parse(text)))
                write_json(self.directory / 'offline-client.json', dict(provider_calls=0,
                    local_reply_count=1, source='planners.rule.plan', deg=63, margin=1.5,
                    rule_header='compiler/depth-dp', agent_placement='program-plan', max_iters=1,
                    tuner=True, tuner_budget_s=TUNER_SECONDS, tune_rule=TUNE_RULE, ledger_rows_added=0))
                self.replay = ReplayClient([json.dumps(dict(program=text))])
        return self.replay.chat(messages, purpose=purpose)


def single_seed_state(original, *, tune_rule=None):
    def create(ctx, *args, **kwargs):
        settings = dict(kwargs, max_iters=1, tuner=True, tuner_budget_s=TUNER_SECONDS)
        if tune_rule is not None:
            settings['tune_rule'] = tune_rule
        return original(ctx, *args, **settings)
    return create


def quality_observer(original, directory, source_text):
    """Record original validation results without another model evaluation."""
    def observe(program, ctx, *args, **kwargs):
        result = original(program, ctx, *args, **kwargs)
        target = directory()
        if target is not None:
            fields = {k: result.get(k) for k in ('top1', 'exact_top1', 'agree', 'total',
                'acc_drop_pp', 'diverged', 'max_u_calibrated', 'max_u_analytic')}
            fields = {k: None if isinstance(v, float) and not math.isfinite(v) else v
                      for k, v in fields.items()}
            row = dict(program=source_text(program), split=ctx.val_split,
                       engine=kwargs.get('engine'), metrics=fields)
            with (Path(target) / 'observed-quality.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        return result
    return observe


@contextmanager
def offline_scope(directory=None):
    from .. import autonomous_budget
    from ..planners import agent, agent_tools, transformer_tuner
    from ..dsl.printer import print_program
    old_client, old_build = autonomous_budget.RenewingLLMClient, agent_tools.build_context
    old_state = agent.new_agent_state
    old_validation, old_tuner_validation = agent_tools.validate, transformer_tuner.validate
    holder, audit_directory = [None], [directory]
    def build(*args, **kwargs):
        holder[0] = old_build(*args, **kwargs)
        return holder[0]
    def client(*args, **kwargs):
        target = directory if directory is not None else Path(kwargs['artifact_dir']).parent
        audit_directory[0] = target
        return RuleClient(lambda: holder[0], target, kwargs.get('recorder'))
    autonomous_budget.RenewingLLMClient = client
    agent_tools.build_context = build
    agent.new_agent_state = single_seed_state(old_state, tune_rule=TUNE_RULE)
    agent_tools.validate = quality_observer(old_validation, lambda: audit_directory[0], print_program)
    transformer_tuner.validate = quality_observer(old_tuner_validation, lambda: audit_directory[0], print_program)
    try:
        yield
    finally:
        autonomous_budget.RenewingLLMClient, agent_tools.build_context = old_client, old_build
        agent.new_agent_state = old_state
        agent_tools.validate, transformer_tuner.validate = old_validation, old_tuner_validation


def offline_audit_function(original):
    """Retain every original quality/math check; distinguish local replies."""
    tree = ast.parse(inspect.getsource(original))
    changed = []
    class Accounting(ast.NodeTransformer):
        def visit_Compare(self, node):
            if (ast.unparse(node.left) == "session.get('calls')" and len(node.comparators) == 1
                    and ast.unparse(node.comparators[0]) == "record.get('actual_api_calls')"):
                node.comparators[0] = ast.parse("record.get('offline_reply_calls')", mode='eval').body
                changed.append(True)
            return self.generic_visit(node)
    Accounting().visit(tree)
    if len(changed) != 1:
        raise ValueError('Exactly the original local/provider accounting comparison must change')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, original.__code__.co_filename, 'exec'), namespace)
    adapted = namespace[original.__name__]
    def audit(record, **kwargs):
        if (record.get('actual_api_calls') != 0 or record.get('offline_reply_calls') != 1
                or not (Path(kwargs['agent_dir']) / 'offline-client.json').is_file()):
            raise ValueError('S6 requires one local rule reply and zero real provider calls')
        result = adapted(record, **kwargs)
        result.update(offline_reply_calls=1, provider_calls=0, client='offline-rule63')
        return result
    return audit


def cnn_worker(argv):
    from . import r28_common_agent as r28
    original = r28.worker
    def worker(args):
        with offline_scope(args.work_dir):
            status = original(args)
        p = args.work_dir / 'worker-result.json'
        value = off.read(p)
        value.update(offline_reply_calls=1 if (args.work_dir / 'offline-client.json').is_file() else 0,
                     provider_calls=0, variant='rule63-seed+tuner+program-plan')
        write_json(p, value)
        return status
    r28.worker = worker
    try:
        return cnn.main(argv)
    finally:
        r28.worker = original


def cnn_command(args, phase, work, *, remaining_seconds):
    argv, seconds = cnn.phase_command(args, phase, work, remaining_seconds=remaining_seconds)
    argv = off.cached_entry(argv)
    i = argv.index('pt_agent.experiments.r30_common_model_e2e')
    argv[i:i+1] = [MODULE, '--cnn-worker']
    argv += ['--tune-rule', TUNE_RULE]
    return argv, seconds


def bert_policy():
    from . import bert_serial_continuation as serial
    serial.STATE_POLICY = dict(serial.STATE_POLICY, max_iters=1, tuner=True,
                              tuner_budget_s=TUNER_SECONDS, tune_rule=TUNE_RULE)


def bert_worker(argv, template, snapshot):
    from .r29_local_compile_matrix import native_metadata_commands
    bert_policy()
    with native_metadata_commands(off.read(template)):
        from .. import analyzer
        from . import bert_sst2_worker as worker
        from . import bert_sst2_recorded as recorded
        old_calibrate, old_recorded = analyzer.calibrate, worker.RecordedRenewingClient
        old_rule = worker.TUNE_RULE
        worker.TUNE_RULE = TUNE_RULE
        analyzer.calibrate = off.inherited_calibration(snapshot, vars(analyzer))
        worker.RecordedRenewingClient = lambda client, *args, **kwargs: client
        try:
            session_dir = Path(argv[argv.index('--session-dir')+1])
            with offline_scope(session_dir):
                return recorded.worker_delegate(argv, template, step_seconds=10800)
        finally:
            analyzer.calibrate, worker.RecordedRenewingClient = old_calibrate, old_recorded
            worker.TUNE_RULE = old_rule


def bert_command(args, phase, work, remaining):
    argv = bert.command(args, phase, work, remaining)
    if phase == 'checkpoint':
        return argv
    argv = off.cached_entry(argv)
    for target, flag in [('pt_agent.experiments.bert_sst2_recorded', '--bert-worker'),
                         (bert.MODULE, '--bert-native')]:
        if target in argv:
            i = argv.index(target)
            argv[i:i+1] = [MODULE, flag]
    return argv + ['--calibration-snapshot', str(args.calibration_snapshot), '--tune-rule', TUNE_RULE]


def bert_outer(argv, snapshot):
    bert_policy()
    namespace = off.clone_module(bert, dict(command=bert_command))
    original = namespace['parser']
    def parser():
        p = original()
        p.add_argument('--calibration-snapshot', type=Path, default=snapshot)
        return p
    namespace['parser'] = parser
    return namespace['main'](argv)


def report_worker_command(argv):
    result = off.cached_entry(argv)
    target = 'pt_agent.experiments.r30_frozen_cnn_report'
    i = result.index(target)
    result[i:i+1] = [MODULE, '--cnn-report-worker']
    return result


def claim_report_access(report, out, gate):
    from .. import paths
    prior = paths.RESULTS_DIR / 'frozen-cnn-report-access' / (Path(gate['source_case']).name+'.json')
    if not prior.is_file():
        return report.claim_access(out, gate)
    marker = off.read(prior)
    failed_dir = Path(marker['out'])
    failed = off.read(failed_dir / 'result.json')
    worker = failed.get('worker') or {}
    if (marker.get('source_case') != gate['source_case'] or marker.get('model') != gate['model']
            or marker.get('report_started') is not False or marker.get('report_complete') is not False
            or marker.get('report_samples') != 0 or failed.get('status') != 'error'
            or worker.get('error') != 'ValueError: Fresh model policy, completed session or actual call accounting differs'
            or worker.get('report_samples_read') != 0 or worker.get('report_samples_completed') != 0
            or failed.get('report_samples_read') != 0 or failed.get('report_samples_completed') != 0
            or (failed_dir / 'report-rows.jsonl').exists() or Path(out).resolve() == failed_dir.resolve()):
        raise ValueError('Only the untouched S6 report adapter gate failure may continue once')
    target = prior.with_name(prior.stem+'-s6-entry-continuation.json')
    report.durable_json(target, dict(marker, out=str(Path(out).resolve()), status='prepared',
        previous_unstarted_access=str(prior), entry_failure=str(failed_dir / 'result.json')), exclusive=True)
    return target


def cnn_report(argv, *, worker=False):
    from . import r30_frozen_cnn_report as report
    from .a5_a6_rule63_search import guarded_process
    old = cnn.audit_agent
    cnn.audit_agent = offline_audit_function(old)
    try:
        if worker:
            return report.main(argv)
        namespace = off.clone_module(report)
        namespace['run_process'] = lambda argv, **kwargs: guarded_process(
            report_worker_command(argv), **off.inherited_file_limit(kwargs))
        namespace['claim_access'] = lambda out, gate: claim_report_access(report, out, gate)
        return namespace['main'](argv)
    finally:
        cnn.audit_agent = old


def defer_report(argv):
    """Queue untouched report access behind the S6b cold controls."""
    from .. import paths
    pending = paths.RESULTS_DIR / 's6-s6b-ordered-no-llm-20261118'
    if not pending.is_dir() or (pending / 'result.json').is_file():
        return False
    target = Path(argv[argv.index('--out')+1])
    target.mkdir(exist_ok=False)
    write_json(target / 'result.json', dict(status='not_run',
        reason='Deferred until S6/S6b cold controls close, as required by S6b section6 and compile priority',
        source_case=argv[argv.index('--source-case')+1], report_samples_read=0,
        report_samples_completed=0, report_samples=0, model_reads=0, sample_reads=0,
        API_calls=0, compiler_attempts=0, report_access_claimed=False,
        metadata_only=True, deferred_entry=True, actual_report_attempts=0))
    return True


def optional_admission(argv, *, now=None):
    """A conditional S6b BERT entry requires its full allocation before cutoff."""
    from .completion_metrics import persist_case_completion, artifact_bytes
    from .r10_paper_metrics import summarize_events
    from .s2_heir_serial import record_registry
    remaining = max(0, int(off.s4.DEADLINE-(time.time() if now is None else now)))
    if remaining >= 10800:
        return False
    started = time.perf_counter_ns()
    out = Path(argv[argv.index('--out')+1])
    out.mkdir(exist_ok=False)
    value = dict(status='not_run', error='Optional S6b entry not dispatched: full three-hour allocation unavailable before publication cutoff',
        reason='Conditional S6b BERT admission leaves the remaining window for required independent reports',
        requested_budget_s=10800, remaining_before_cutoff_s=remaining, actual_allocated_scientific_s=0,
        metadata_only=True, scientific_attempts=0, model_reads=0, sample_reads=0, api_calls=0,
        attempted={}, completed={}, phases={}, processes=[], compile_accepted=False,
        compiled_bootstraps=None, frozen=False, generation_e2e_s=None, report_samples=0,
        mode='optional-scheduling-admission-only', tune_rule=TUNE_RULE,
        full_three_hour_availability_test_run=False)
    write_json(out / 'result.json', value)
    record_registry(out, 'S6B-OPTIONAL-ADMISSION', 'bert-tiny', value['mode'], value['status'],
        dict(metadata_only=True, scientific_attempts=0, API_calls=0))
    persist_case_completion(out, round_label='S6B-OPTIONAL-ADMISSION', started_ns=started,
        status=value['status'], timing=summarize_events([]), required_artifact_bytes=artifact_bytes(out))
    return True


def main(argv=None):
    global TUNE_RULE
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in ('cnn', 'cnn-worker', 'cnn-report', 'cnn-report-worker', 'bert', 'bert-worker', 'bert-native', 'bert-report'):
        p.add_argument('--'+flag, action='store_true')
    p.add_argument('--calibration-snapshot', type=Path)
    p.add_argument('--tune-rule', choices=('v8-6.25', 'v8-6.23'))
    p.add_argument('--s6-continue', action='store_true')
    p.add_argument('--final-report', action='store_true')
    args, rest = p.parse_known_args(argv)
    TUNE_RULE = args.tune_rule or 'v8-6.25'
    if args.tune_rule is None:
        candidates = []
        if '--session-dir' in rest:
            candidates.append(Path(rest[rest.index('--session-dir')+1]) / 'SESSION.json')
        if '--source-case' in rest:
            source = Path(rest[rest.index('--source-case')+1])
            candidates += [source / 'session/SESSION.json', source / 'work/agent/SESSION.json']
        for state in candidates:
            if state.is_file():
                TUNE_RULE = off.read(state)['tune_rule']
                if TUNE_RULE not in ('v8-6.25', 'v8-6.23'):
                    raise ValueError('Retain an explicitly supported existing S6 rule')
                break
    if args.cnn_worker:
        return cnn_worker(rest)
    if args.cnn:
        from .a5_a6_rule63_search import guarded_process
        return off.clone_module(cnn, dict(phase_command=cnn_command, run_process=guarded_process,
            audit_agent=offline_audit_function(cnn.audit_agent),
            agent_variant=lambda args: 'rule63-seed+tuner+program-plan'))['main'](rest)
    if args.cnn_report:
        if not args.final_report and defer_report(rest):
            return 0
        return cnn_report(rest)
    if args.cnn_report_worker:
        return cnn_report(rest, worker=True)
    if args.bert_worker:
        q = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        q.add_argument('--delegate-worker', action='store_true')
        q.add_argument('--step-seconds', type=int, default=10800)
        q.add_argument('--metadata-template', type=Path, required=True)
        selected, inner = q.parse_known_args(rest)
        return bert_worker(inner, selected.metadata_template, args.calibration_snapshot)
    if args.bert_native:
        bert_policy()
        return bert.main(rest)
    if args.bert:
        from .. import paths
        gate = paths.RESULTS_DIR / 's6-sequence-handoff-20261118/hold-before-bert.json'
        if gate.is_file() and not args.s6_continue:
            data = off.read(gate)
            if data.get('status') == 'hold_original_BERT_entry':
                write_json(gate.with_name('BERT-entry-waiting.json'), dict(status='waiting_before_source_entry',
                    model_reads=0, sample_reads=0, API_calls=0, cold_attempts=0))
                while time.time() < off.s4.DEADLINE:
                    time.sleep(2)
                raise TimeoutError('Submission cutoff before the held BERT entry')
        if TUNE_RULE == 'v8-6.23' and optional_admission(rest):
            return 0
        return bert_outer(rest, args.calibration_snapshot)
    if args.bert_report:
        if not args.final_report and defer_report(rest):
            return 0
        bert_policy()
        old_policy, old_worker = off.bert_policy, off.MODULE
        off.bert_policy, off.MODULE = bert_policy, MODULE
        try:
            q = argparse.ArgumentParser()
            for key in ('out', 'source-case', 'metadata-template'):
                q.add_argument('--'+key, type=Path, required=True)
            selected = q.parse_args(rest)
            selected.calibration_snapshot = args.calibration_snapshot
            return off.bert_report(selected)
        finally:
            off.bert_policy, off.MODULE = old_policy, old_worker
    p.error('Select an S6 operation')


if __name__ == '__main__':
    raise SystemExit(main())
