"""Retained reply13 warm branch with a new component-program-plan policy.

Only after a closed Grouped resource stop. Original full-model exact/heldout/C2
remain the quality path. No Provider is constructed: one retained evaluation,
then one original confirmation step. No compiler runs inside this entry.
"""
import ast
import copy
from contextlib import nullcontext
import fcntl
import json
import inspect
import os
from pathlib import Path
import shutil
from types import ModuleType,SimpleNamespace

from . import qwen_guarded_replay as guarded
from . import qwen_agent_grouped_record as grouped
from . import qwen_agent_cost_lifetime as lifetime
from . import qwen_component_cost as component

MODULE='pt_agent.experiments.qwen_retained_component_replay'
ROUND='autonomous-qwen-retained13-component-program-plan'
MODE='warm-retained13-full-model-component-program-plan-no-new-provider'
POLICY=component.POLICY
GIB=1024**3
COMPONENT_STEP_SECONDS=9000
CLOCK_SCOPE='Warm retained reply13 component-program-plan evaluation and original full quality/C2; prior failed full-graph and original network clocks excluded'
COMPLETED_FIELDS=('completed_cost_authority','completed_cost_admission',
    'logical_constant_inventory_gib','logical_constant_inventory_scope')
INVENTORY_SCOPE='Logical dense full-component constant inventory eligibility only; physical RSS/address-space, arithmetic, profile and quality policies unchanged'


class ComponentCostExecutionError(RuntimeError):
    """A failed measurement must stop before quality selection or confirmation."""


def _same(value):return guarded._same(value)


def _completed_enabled(args):
    case=getattr(args,'completed_cost_from',None)
    inventory=getattr(args,'logical_constant_inventory_gib',None)
    if case is None and inventory is None:
        if getattr(args,'completed_cost_admission',None) is not None:
            raise ValueError('Completed cost proof requires its explicit source and inventory policy')
        return False
    if case is None or type(inventory) is not int or inventory!=32:
        raise ValueError('Pair --completed-cost-from with --logical-constant-inventory-gib 32')
    return True


def _completed_base_source(source):
    return {key:value for key,value in source.items() if key not in COMPLETED_FIELDS}


def _retained_source_text(args):
    return (Path(args.partial_from)/'session/sources/iteration-13.ptdsl').read_text()


def _completed_load(args,source):
    from . import qwen_completed_cost_cache as cached
    canonical=Path(args.calibration).parent/cached.PROOF
    supplied=getattr(args,'completed_cost_admission',None)
    if supplied is None or Path(supplied).resolve()!=canonical.resolve():
        raise ValueError('Require explicit canonical outer completed-cost admission')
    return cached.load_worker(Path(args.completed_cost_from),canonical,
        _completed_base_source(source),_retained_source_text(args))


def _inventory_functions(functions):
    """Private original evaluate globals; only logical const_bytes changes."""
    from types import FunctionType
    from . import qwen_agent_resources as resources
    callback=functions['evaluate']
    evaluator=inspect.getclosurevars(callback).nonlocals.get('original_evaluate')
    if (not isinstance(evaluator,FunctionType)
            or evaluator.__globals__.get('__name__')!='pt_agent.planners.agent_tools'):
        raise ValueError('Require the original bridge evaluator before lifetime adaptation')
    profiles=copy.deepcopy(evaluator.__globals__['PROFILES'])
    if profiles['qwen48']['limits']['const_bytes']!=4*GIB:
        raise ValueError('Original qwen48 logical inventory policy changed')
    profiles['qwen48']['limits']['const_bytes']=32*GIB
    module=ModuleType(evaluator.__module__);module.__dict__.update(evaluator.__globals__)
    tools=guarded.bridge._clone_namespace(module,dict(PROFILES=profiles))
    agent=ModuleType(functions['__name__']);agent.__dict__.update(functions)
    return guarded.bridge._clone_namespace(agent,dict(
        evaluate=resources._replace_closure(callback,'original_evaluate',tools['evaluate'])))


def _new(path,value):
    path=Path(path)
    with path.open('x')as out:
        json.dump(value,out,allow_nan=False,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())


def check_write_paths(args):
    guarded.check_write_paths(args)
    source=Path(args.grouped_from).resolve()
    for key in ('out','work_dir','session','codegen','calibration'):
        value=getattr(args,key,None)
        if value is not None:
            target=Path(value).resolve()
            if source==target or source.is_relative_to(target)or target.is_relative_to(source):
                raise ValueError('Retain the original Grouped case outside all new write paths')
    if _completed_enabled(args):
        completed=Path(args.completed_cost_from).resolve()
        for key in ('out','work_dir','session','codegen','calibration'):
            value=getattr(args,key,None)
            if value is not None:
                target=Path(value).resolve()
                if completed==target or completed.is_relative_to(target) or target.is_relative_to(completed):
                    raise ValueError('Retain the completed cost authority outside new write paths')


def _closed_grouped(args,base_source,*,check_processes):
    """Only explicit saved paths; process checks belong to outer admission."""
    from ..planners.transformer_session import load_state
    case=Path(args.grouped_from).resolve();record=guarded.read(case/'result.json')
    done=guarded.read(case/'completion.json');state=load_state(case/'session/SESSION.json')
    if (record.get('round')!=grouped.ROUND or record.get('mode')!=grouped.MODE
            or record.get('model')!=guarded.original.MODEL
            or record.get('status')!='error' or record.get('candidate_frozen')is not False
            or record.get('termination',{}).get('cause')!='own_budget_guard'
            or done.get('round')!=grouped.ROUND or done.get('status')!='error'
            or done.get('required_artifacts_persisted')is not True or done.get('registry_persisted')is not True
            or record.get('api_calls')!=0 or record.get('report_samples')!=0):
        raise ValueError('Require actual closed Grouped own-resource failure with no freeze/API/report')
    if (state.get('phase')!='evaluate' or state.get('calls')!=12 or state.get('iteration')!=13
            or state.get('max_iters')!=15 or state.get('failed')is not None or state.get('pending')is not None
            or state.get('tuner')is not False or state.get('prompt_version')!='r8-v11'
            or state.get('placement')!='program-plan' or state.get('confirm_rule')!='c2'
            or state.get('presentation_variant')!=guarded.bounded.PRESENTATION_ID
            or state.get('method_guidance_version')!=guarded.bounded.METHOD_GUIDANCE_VERSION
            or state.get('prompt_variant_id')!=guarded.bridge.PROMPT_ID
            or state.get('algorithm_variant')!=guarded.bridge.VARIANT
            or state.get('recovery',{}).get('inherited_attempts')!=9
            or state.get('pool') or state.get('best')is not None or state.get('confirmation')is not None
            or [r.get('iteration')for r in state.get('history',[])]!=list(range(1,13))):
        raise ValueError('Grouped must retain the uncommitted reply13 checkpoint')
    if any((case/'session'/name).exists()for name in ('AGENT.json','FROZEN.json','program.ptdsl')):
        raise ValueError('Original Grouped C2/frozen artifacts must remain absent')
    argv=guarded.read(case/'command-1.json')
    if not isinstance(argv,list)or argv.count(grouped.MODULE)!=1:raise ValueError('Saved actual Grouped command required')
    old=grouped.parser().parse_args(argv[argv.index(grouped.MODULE)+1:]);old.out=None
    expected=grouped.source_gate(old)
    if _same(expected)!=_same(record.get('source')):raise ValueError('All original Grouped source fields must match')
    for key,value in base_source.items():
        if key=='resource_policy':continue
        if key not in expected or _same(expected[key])!=_same(value):
            raise ValueError('Retain the exact original guarded source: '+key)
    if (old.partial_from.resolve()!=args.partial_from.resolve()
            or old.resume_from.resolve()!=args.resume_from.resolve()):
        raise ValueError('The paid reply and original twelve-attempt history must share sources')
    original_source=Path(args.partial_from)/'session/sources/iteration-13.ptdsl'
    local_source=case/'session/sources/iteration-13.ptdsl'
    if not local_source.is_file()or ast.dump(ast.parse(local_source.read_text()),include_attributes=False)!=ast.dump(ast.parse(original_source.read_text()),include_attributes=False):
        raise ValueError('Retain the same uncommitted original reply13 program')
    last=None
    with (case/'guard-samples.jsonl').open()as stream:
        for line in stream:
            if line.strip():last=json.loads(line)
    if (last is None or not (last.get('subtree_rss_bytes',0)>16*GIB
            or last.get('available_bytes',2*GIB)<1.25*GIB)):
        raise ValueError('Saved memory evidence must show a real resource stop, not predicted failure or wall alone')
    births={}
    for path in sorted(case.glob('*resources.jsonl')):
        with path.open()as stream:
            for line in stream:
                sample=json.loads(line)
                for item in [*sample.get('processes',[]),sample.get('outer_process')]:
                    if item is not None:births[int(item['pid'])]=int(item['starttime'])
    if not births:raise ValueError('Actual process-birth cleanup evidence required')
    if check_processes:
        from .r10_paper_metrics import _process
        for pid,birth in births.items():
            now=_process(pid)
            if now and now['starttime']==birth and now['state']!='Z':raise ValueError('Grouped process identity remains active')
        registry=case.parent/'registry.jsonl'
        with registry.open()as stream:
            fcntl.flock(stream.fileno(),fcntl.LOCK_SH)
            rows=[json.loads(line)for line in stream if line.strip()and json.loads(line).get('run_id')==case.name]
        fields={'run_id','round','model','mode','git_rev','metrics','artifacts_dir','ts'}
        if (len(rows)!=1 or set(rows[0])!=fields or rows[0]['round']!=grouped.ROUND
                or rows[0]['model']!=guarded.original.MODEL
                or rows[0]['mode']!=grouped.MODE or rows[0]['metrics'].get('status')!='error'
                or Path(rows[0]['artifacts_dir']).resolve()!=case):raise ValueError('Unique original eight-field closed registry row required')
    return dict(case=str(case),round=record['round'],mode=record['mode'],source=_same(expected),
        status='error',cause='own_budget_guard',actual_memory_stop=True,checkpoint_calls=12,
        checkpoint_iteration=13,process_births={str(k):v for k,v in births.items()},
        all_process_identities_closed=True,registry_checked_by_outer=True,
        strict_math_or_full_cost_completion_claimed=False)


def source_gate(args):
    check_write_paths(args)
    source=guarded.source_gate(args)  # All original source checks precede new policy fields.
    root=Path(args.calibration).parent;proof=root/'component-source-admission.json'
    is_inner=getattr(args,'worker_phase',None)is not None
    if is_inner:
        if args.source_admission is None or args.source_admission.resolve()!=proof.resolve():
            raise ValueError('Inner worker requires its explicit canonical outer admission proof')
        saved=guarded.read(proof);checked=_closed_grouped(args,source,check_processes=False)
        if saved!=dict(module=MODULE,policy=POLICY,checked=checked):raise ValueError('Outer/inner closed-source proof differs')
    else:
        checked=_closed_grouped(args,source,check_processes=True)
        _new(proof,dict(module=MODULE,policy=POLICY,checked=checked))
    source.update(component_cost_policy=POLICY,component_policy_source=str(Path(__file__).resolve()),
        component_cost_source=str(Path(component.__file__).resolve()),prior_grouped=checked,
        retained_candidate=13,new_provider_calls_allowed=0,maximum_live_iteration=13,
        inherited_search_attempts=12,inherited_committed_iterations=12,
        retained_reply13_HTTP_already_occurred=True,historical_paid_HTTP_count=None,
        historical_fee=None,pool_scope='only newly measured retained13 under component-program-plan',
        global_optimality=False,full_model_quality_and_C2_unchanged=True,
        timing_scope='Warm retained13 re-evaluation with fresh original full-model references and component placement; all prior attempt clocks separate')
    if _completed_enabled(args):
        from . import qwen_completed_cost_cache as cached
        canonical=root/cached.PROOF
        if not is_inner:
            supplied=getattr(args,'completed_cost_admission',None)
            if supplied is not None and Path(supplied).resolve()!=canonical.resolve():
                raise ValueError('Outer completed-cost proof must use the canonical new case path')
            cached.persist(root,cached.prepare_outer(Path(args.completed_cost_from),source))
            # Explicit worker path is also reused by future read-only frozen gates.
            args.completed_cost_admission=canonical
        loaded=_completed_load(args,source)
        source.update(completed_cost_authority=loaded['provenance'],completed_cost_admission=str(canonical),
            logical_constant_inventory_gib=32,logical_constant_inventory_scope=INVENTORY_SCOPE)
    out=getattr(args,'out',None)
    if out is not None and out.is_dir():
        for path in (Path(__file__),Path(component.__file__)):
            shutil.copy2(path,out/'variant-source'/path.name)
        if _completed_enabled(args):
            shutil.copy2(cached.__file__,out/'variant-source'/Path(cached.__file__).name)
    return source


def seed_partial(args):
    from ..planners.transformer_session import load_state,save_state
    check_write_paths(args);guarded.seed_partial(args)
    marker=args.session/'component-retained13.json'
    if marker.exists():return
    state=load_state(args.session/'SESSION.json')
    if state['pool']or state['best']is not None or state['calls']!=12 or state['iteration']!=13:
        raise ValueError('No foreign-cost candidates may enter the retained13 pool')
    previous=dict(max_iters=state['max_iters'],records=state['records'],llm_min=state['llm_min'])
    _new(args.session/'inherited-cost-policy.json',dict(policy='whole-graph-original-program-plan',
        scope='Historical only, excluded from this branch ranking',old_state=guarded.bounded.original.read_json(Path(args.partial_from)/'session/SESSION.json')))
    state.update(max_iters=13,pool=[],records={},llm_min=None,stale=0,first_feasible=None,
        component_cost_policy=POLICY,new_provider_calls_allowed=0)
    save_state(args.session/'SESSION.json',state)
    replay_marker=args.session/'guarded-replay.json'
    guarded.original.write_json(replay_marker,dict(guarded.read(replay_marker),
        active_component_cost_policy=POLICY,active_max_iters=13,active_new_provider_calls_allowed=0,
        inherited_budget_scope='Original maximum15/remaining14,15 metadata is historical; this branch permits retained13 only'))
    _new(marker,dict(policy=POLICY,retained_iteration=13,inherited_attempts=12,
        previous_policy_summary=dict(max_iters=previous['max_iters'],record_count=len(previous['records']),llm_min=previous['llm_min']),
        live_max_iters=13,pool_scope=[13],new_provider_calls_allowed=0,report_access=0))
    archive=args.calibration.parent/'prior-grouped-evidence';archive.mkdir()
    for name in ('result.json','completion.json','guard-failure.json'):
        shutil.copy2(Path(args.grouped_from)/name,archive/name)
    shutil.copy2(Path(args.grouped_from)/'session/SESSION.json',archive/'SESSION.json')


def session_outcome(directory):
    from ..planners.transformer_session import load_state
    private=guarded.bridge._clone_namespace(guarded.bounded,dict(MAX_ATTEMPTS=13))
    value=private['session_outcome'](directory)
    if value['status']=='not_started':return value
    state=load_state(Path(directory)/'SESSION.json')
    if (state.get('component_cost_policy')!=POLICY or state.get('new_provider_calls_allowed')!=0
            or any(row['iteration']!=13 for row in state['pool'])
            or any(key[0]!=13 or record[0].get('placement_model')!='component-program-plan'
                   for key,record in state['records'].items())):
        raise ValueError('Retain only candidate13 costs under the new explicit policy')
    if value['done']:
        marker=guarded.read(Path(directory)/'component-c2-access.json')
        if marker.get('status')!='completed' or marker.get('calls')!=13:raise ValueError('Original C2 transaction must be durably complete')
    value.update(recovery_policy=POLICY,new_provider_calls=0,warm_retained_candidate=13)
    return value


def agent_functions(directory,diagnostics,*,completed_args=None,completed_source=None):
    sequence=0
    def estimate(program,weights,**kwargs):
        nonlocal sequence
        if completed_args is not None:
            from ..dsl.ir import canonical
            from ..dsl.parser import parse
            if kwargs.get('profile','qwen48')!='qwen48':raise ValueError('Retain original qwen48 profile')
            if canonical(program)!=canonical(parse(_retained_source_text(completed_args))):
                raise ValueError('Current evaluated program differs from the paid retained13 source')
            span=(diagnostics.recorder.span('completed_component_cost_cache_hit',cache_hit=True)
                  if diagnostics.recorder is not None else nullcontext())
            with span:
                loaded=_completed_load(completed_args,completed_source)
            guarded.original.write_json(Path(directory)/'completed-cost-reuse.json',loaded['provenance'])
            return copy.deepcopy(loaded['cost'])
        sequence+=1
        from ..backends.base import BackendError
        try:
            return component.estimate(program,weights,directory=Path(directory)/('component-cost-'+str(sequence)),
                diagnostics=diagnostics,**kwargs)
        except BackendError as exc:
            raise ComponentCostExecutionError('Component measurement failed: '+str(exc)) from exc
    costs=dict.fromkeys(('estimate','estimate_normalized','estimate_fast','estimate_normalized_fast'),estimate)
    base=guarded.bridge._agent_functions()
    if completed_args is not None:
        if not _completed_enabled(completed_args):raise ValueError('Explicit completed-cost branch required')
        base=_inventory_functions(base)
    functions=lifetime.adapt_agent_functions(base,
        diagnostics=diagnostics,costs=costs,scope=nullcontext)
    functions['feedback_message']=guarded.bounded.FeedbackPresenter(Path(directory)/'full-feedback',functions['feedback_message'])
    return functions


def _calibrate(args,recorder,result):
    seed_partial(args)
    for name,target in (('calibration.json',args.calibration),('unit-width-bridge.json',guarded.bridge._bridge_path(args)),
            ('generation-input.json',args.calibration.parent/'generation-input.json')):
        shutil.copy2(Path(args.partial_from)/name,target)
    result.update(calibration_completed=True,bridge_calibration_completed=True,calibration_samples=23,
        calibration_cache=dict(source=str(args.partial_from),fresh_calibration_forwards=0),
        fresh_calibration_forwards=0,calibration_forward_attempts=0,calibration_forward_completed=0,
        stable_softmax_attempts=0,stable_softmax_completed=0)


def _agent_phase(args,recorder,result):
    from ..planners.transformer_session import load_state,save_state,write_session_result
    seed_partial(args);state_path=args.session/'SESSION.json';state=load_state(state_path)
    outcome=session_outcome(args.session)
    if state['phase']=='done':
        result.update(session=outcome,status=outcome['status'],candidate_frozen=outcome['frozen']);return
    if args.worker_phase=='prompt':
        result.update(prompt_completed=True,api_required=False,session=outcome,
            retained_reply13_required=True,new_provider_instances=0,new_provider_calls=0);return
    diagnostics=lifetime.Diagnostics(args.work_dir/'evaluation-stage-events.jsonl')
    functions=(agent_functions(args.session,diagnostics,completed_args=args,completed_source=result['source'])
        if _completed_enabled(args) else agent_functions(args.session,diagnostics))
    if state['phase']=='evaluate':
        if state['calls']!=12 or state['iteration']!=13:raise ValueError('Only local retained13 evaluation is permitted')
        private=guarded.bridge._clone_namespace(guarded,dict(
            bounded=SimpleNamespace(_agent_functions=lambda directory:functions,session_outcome=session_outcome)))
        private['replay_once'](args,recorder,result)
        return
    if state['phase']!='confirm' or state['calls']!=13 or state['pending']is not None:
        raise ValueError('No tune/evaluate/provider phase after retained13')
    claim=args.session/'component-c2-access.json'
    _new(claim,dict(status='started',policy=POLICY,calls=13,report_samples=0))
    working=copy.deepcopy(state)
    try:
        ctx=guarded.bridge._context(args,result)
        with recorder.span('retained13_original_full_C2'):
            functions['agent_step'](ctx,working,trace_path=args.session/'agent_trace.jsonl',client=None)
        if working['phase']!='done' or working['calls']!=13:raise ValueError('Original C2 must close without any Provider step')
        write_session_result(functions['agent_result'](working),args.session,
            prompt_version='r8-v11',tune_policy=working['tune_policy'])
        save_state(state_path,working)
        guarded.original.write_json(claim,dict(status='completed',policy=POLICY,calls=13,report_samples=0))
    except BaseException:
        save_state(args.work_dir/'partial-component-C2-state.json',working)
        raise
    outcome=session_outcome(args.session)
    result.update(api_calls=0,new_provider_calls=0,new_provider_instances=0,session=outcome,
        candidate_frozen=outcome['frozen'],status=outcome['status'],
        step=dict(complete=True,step='confirm',next_step='done',iteration=working['iteration'],session_calls=13))


def phase_command(args,phase,work):
    functions=guarded.bridge._namespace();functions['MODULE']=MODULE
    functions['PHASE_SECONDS']=dict(functions['PHASE_SECONDS'],step=COMPONENT_STEP_SECONDS)
    command=functions['phase_command'](args,phase,work)+['--resume-from',str(args.resume_from),
        '--partial-from',str(args.partial_from),'--grouped-from',str(args.grouped_from),
        '--source-admission',str(args.calibration.parent/'component-source-admission.json'),
        '--rss-gib',str(args.rss_gib),'--minimum-available-gib',str(args.minimum_available_gib),'--seconds',str(args.seconds)]
    if _completed_enabled(args):
        from .qwen_completed_cost_cache import PROOF
        command+=['--completed-cost-from',str(args.completed_cost_from),'--logical-constant-inventory-gib','32',
            '--completed-cost-admission',str(args.calibration.parent/PROOF)]
    return command


def parser():
    value=guarded.parser();value.add_argument('--grouped-from',type=Path,required=True)
    value.add_argument('--source-admission',type=Path)
    value.add_argument('--completed-cost-from',type=Path)
    value.add_argument('--logical-constant-inventory-gib',type=int)
    value.add_argument('--completed-cost-admission',type=Path)
    return value


def _namespace():
    bounded=guarded.bridge._clone_namespace(guarded.bounded,dict(session_outcome=session_outcome))
    private_bounded=ModuleType(guarded.bounded.__name__);private_bounded.__dict__.update(bounded)
    private=guarded.bridge._clone_namespace(guarded,dict(__file__=__file__,__doc__=__doc__,
        MODULE=MODULE,ROUND=ROUND,MODE=MODE,CLOCK_SCOPE=CLOCK_SCOPE,bounded=private_bounded,
        parser=parser,source_gate=source_gate,check_write_paths=check_write_paths,
        _calibrate=_calibrate,_agent_phase=_agent_phase,phase_command=phase_command))
    functions=private['_namespace']()
    # The outer phase supervisor and the nixrun command must share the actual
    # finite component allowance. The original total-case guard still clips it.
    functions['PHASE_SECONDS']=dict(functions['PHASE_SECONDS'],step=COMPONENT_STEP_SECONDS)
    return functions


def main(argv=None):return _namespace()['main'](argv)


if __name__=='__main__':raise SystemExit(main())
