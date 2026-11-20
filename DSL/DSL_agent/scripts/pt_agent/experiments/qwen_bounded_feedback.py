"""Explicit recovery of the failed Qwen Agent with bounded public presentation.

The inherited terminal case and sender are untouched. Full diagnostics and
chronological messages remain local. Only the outbound presentation is projected;
the original evaluator, fitting, domains, C2 and numerical policies are reused.
"""
import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import types

from . import qwen_bridge_agent as bridge
from . import qwen_agent_resources as resource_policy

original = bridge.original
MODULE = 'pt_agent.experiments.qwen_bounded_feedback'
ROUND = 'autonomous-qwen-bounded-feedback-recovery'
MODE = 'original-full-decoder-bounded-feedback-recovery'
PRESENTATION_ID = 'qwen-bounded-history-syntax-methods-v2'
METHOD_GUIDANCE_VERSION = 'supported-methods-fit-diagnostics-v1'
INHERITED_ATTEMPTS, MAX_ATTEMPTS = 9, 15
FEEDBACK_BYTES, REQUEST_BYTES = 96*1024, 768*1024
OLD_CLOCK_SCOPE='Cold continuation from saved original import; historical import/reference clocks excluded'
CLOCK_SCOPE=('Named recovery with cached23-prefix calibration and inherited8provider replies/1local failure; '
    'only new attempts10..15 and this case work enter its clock; original cold clock excluded')
KEY_FIELDS = ('code','line','node','node_id','op_id','call_lines','role','fn','fit','method',
    'degree','deg0','iters','domain','target','k','rho','reason','exception_type',
    'iterations','max_error','abs_E')
SYNTAX_GUIDANCE = '''
Private syntax presentation qwen-bounded-history-syntax-v1:
Keep all supplied model nodes, weights, shapes, approximation domains and policies.
A macro assigns an operator result and returns that tensor name. For example,
the following four-channel example illustrates syntax only:
def signed_example(x):
    y = shuffle(x, index=(2,3,0,1), sign=(-1,-1,1,1))
    return y
Do not return an operator call directly. Put literal tuples inside the operator
attributes (or pass them as macro parameters); standalone tuple assignments,
tuple multiplication, range, comprehensions and ellipses are outside the grammar.
Copy the original shuffle tuples exactly: query-width896 and key-width128 each
require index and sign of that same width, with integer signs -1 or +1. Check
both lengths and all closing parentheses before returning the complete program.
The available <=63 ladder is not an additional parser limit on positive Remez
degrees. All declared numerical choices still pass the unchanged original fitting
and validation checks. A nonconverged fit remains a rejection even at small error.
'''.strip()
METHOD_GUIDANCE = '''
Supported-method presentation supported-methods-fit-diagnostics-v1:
The inherited iteration8 diagnostics retain32 rsqrt Newton initializer failures
and11 SiLU Remez failures. These are original numerical rejections. Newton first
fits its initial polynomial with Remez; extra Newton iterations run only after
that fit succeeds. Increasing iters cannot repair an initializer fit failure.
The existing rsqrt attributes support fit="relative" and lower explicit deg0
values, with up to7 Newton iterations in this named variant. The existing SiLU
method="cheb" uses Chebyshev interpolation and avoids the Remez exchange step.
These are optional choices available to the candidate. Their fitting, mathematical,
finite, domain and quality checks stay unchanged. A small last fit error alone
does not accept an unconverged exchange.
Positive calibrated rsqrt domains use the original v11 logarithmic coverage
coordinate u=2*(ln(x)-ln(lo))/(ln(hi)-ln(lo))-1 and retain max|u|<=0.90.
The supplied minimum domain is a containment requirement. A declared positive
domain may widen from calibration to leave the required coverage room. Preserve
the original calibration source and minimum coverage, and derive no domain from
heldout, confirmation or report results. The public softmax shift, inv_scale,
steps and registered analytical inverse requirements remain fixed.
'''.strip()


def recovery_guidance():
    return SYNTAX_GUIDANCE+'\n\n'+METHOD_GUIDANCE


class PresentationError(ValueError):
    pass


def _json(value):
    return json.dumps(value,ensure_ascii=True,separators=(',',':'),allow_nan=False)


def _bytes(value):
    return len(_json(value).encode('utf-8'))


def wire_size_upper_bound(messages):
    """Exact public settings plus the loader's512-character ASCII key bound.

    The placeholder is an accounting upper bound, never a credential. The
    unchanged loader accepts ASCII33..126 and excludes JSON-escaped characters.
    No environment or credential file is read here.
    """
    from ..llm import Config
    from ..planners.transformer_session import output_tokens
    body=Config(model='deepseek-flash',reasoning_effort='high',
        max_tokens=output_tokens('r8-v11'),timeout_seconds=600).settings()
    body['messages']=messages
    return _bytes(dict(api_key='x'*512,body=body,timeout=600,provider='deepseek',loopback_proxy_port=0))


def _write_new(path, value):
    """No artifact replacement: a repeated presentation identity is an error."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        stream.write(_json(value)+'\n');stream.flush();os.fsync(stream.fileno())


def _short(value, limit=480):
    if not isinstance(value,str) or len(value)<=limit:return value
    marker=' [local full diagnostic retained] '
    keep=(limit-len(marker))//2
    return value[:keep]+marker+value[-keep:]


def compact_feedback(feedback):
    """Preserve every error and its scalar diagnostic; summarize history only."""
    result=copy.deepcopy(feedback)
    if isinstance(result.get('error'),str):result['error']=_short(result['error'])
    if 'errors' in result:
        if not isinstance(result['errors'],list):raise PresentationError('Error records must be a list')
        errors=[]
        for source in feedback['errors']:
            if not isinstance(source,dict):raise PresentationError('Structured error object required')
            row={key:copy.deepcopy(source[key]) for key in KEY_FIELDS if key in source}
            for key in ('message','line_text'):
                if key in source:row[key]=_short(source[key])
            history=source.get('history')
            if history is not None:
                if not isinstance(history,list):raise PresentationError('Diagnostic history must be a list')
                # Retain an already bounded presentation unchanged on later sends.
                row.update(history_count=len(history),history_head=copy.deepcopy(history[:2]),
                    history_tail=copy.deepcopy(history[-2:] if len(history)>2 else []))
            else:
                for key in ('history_count','history_head','history_tail'):
                    if key in source:row[key]=copy.deepcopy(source[key])
            errors.append(row)
        result['errors']=errors
    return result


class FeedbackPresenter:
    def __init__(self, directory, formatter):
        self.directory,self.formatter=Path(directory),formatter

    def __call__(self, feedback, iteration, best, *, tuned=None):
        full=dict(iteration=iteration,feedback=feedback,best_feasible_so_far=best,tuned=tuned)
        _write_new(self.directory/f'iteration-{iteration:04d}.full.json',full)
        shown=self.formatter(compact_feedback(feedback),iteration,best,tuned=tuned)
        if len(shown.encode('utf-8'))>FEEDBACK_BYTES:
            raise PresentationError('All key diagnostics exceed the finite feedback presentation allowance')
        return shown


def _feedback_message(message):
    if message.get('role')!='user':return None
    try:value=json.loads(message['content'])
    except (ValueError,TypeError):return None
    return value if isinstance(value,dict) and isinstance(value.get('feedback'),dict) else None


def project_messages(messages, *, pinned_errors, budget=REQUEST_BYTES):
    """Retain immutable initial messages, pinned numerical errors and recent pairs."""
    if (type(budget) is not int or not 0<budget<=REQUEST_BYTES or len(messages)<2
            or messages[0].get('role')!='system' or messages[1].get('role')!='user'
            or (len(messages)-2)%2):
        raise PresentationError('Finite complete-pair conversation required')
    view=copy.deepcopy(messages);compacted=[];seen={};pairs=[]
    for index in range(2,len(view),2):
        if view[index].get('role')!='assistant' or view[index+1].get('role')!='user':
            raise PresentationError('Chronological assistant/user pairs required')
        parsed=_feedback_message(view[index+1]);iteration=None
        if parsed is not None:
            iteration=parsed.get('iteration');errors=parsed['feedback'].get('errors') or []
            if type(iteration) is not int:raise PresentationError('Explicit feedback iteration required')
            parsed['feedback']=compact_feedback(parsed['feedback'])
            view[index+1]['content']=_json(parsed)
            seen[iteration]=len(errors)
            compacted.append(dict(message_index=index+1,iteration=iteration,error_count=len(errors)))
        pairs.append(dict(indices=[index,index+1],iteration=iteration))
    if any(seen.get(iteration)!=count for iteration,count in pinned_errors.items()):
        raise PresentationError('Pinned inherited numerical diagnostics are incomplete')
    retained=set(range(len(view)));removed=[]
    # Always keep the newest two complete pairs and every pinned failure pair.
    protected={index for pair in pairs[-2:] for index in pair['indices']}
    protected.update(index for pair in pairs if pair['iteration'] in pinned_errors for index in pair['indices'])
    def selected():return [message for index,message in enumerate(view) if index in retained]
    for pair in pairs:
        if wire_size_upper_bound(selected())<=budget:break
        if any(index in protected for index in pair['indices']):continue
        for index in pair['indices']:retained.remove(index)
        removed.append(dict(message_indices=pair['indices'],iteration=pair['iteration'],
            reason='Old complete pair removed from outbound view to meet finite public-message allowance'))
    projected=selected();amount=wire_size_upper_bound(projected)
    if amount>budget:
        raise PresentationError('Required schema/model/recent/pinned diagnostics exceed public-message allowance')
    if projected[:2]!=messages[:2]:raise PresentationError('Immutable initial messages changed')
    return projected,dict(presentation_variant=PRESENTATION_ID,public_messages_JSON_bytes=_bytes(dict(messages=projected)),
        wire_size_upper_bound_bytes=amount,
        allowance_bytes=budget,transport_allowance_unchanged=True,retained_message_indices=sorted(retained),
        removed_pairs=removed,compacted_feedback=compacted,pinned_error_counts=dict(pinned_errors),
        immutable_initial_message_indices=[0,1],full_chronological_messages_retained_locally=True,
        envelope_scope='Same public Config and transport envelope plus512 ASCII key upper bound; credentials unread')


class ProjectedClient:
    """An upstream view adapter; the original provider, loader and sender run last."""
    def __init__(self, client, directory, *, pinned_errors,attempt_offset=INHERITED_ATTEMPTS):
        self.client,self.directory,self.pinned_errors=client,Path(directory),pinned_errors
        self.attempt_offset=attempt_offset

    @property
    def calls(self):return self.client.calls

    def chat(self,messages,*,purpose):
        number=self.attempt_offset+self.calls+1
        if not INHERITED_ATTEMPTS<number<=MAX_ATTEMPTS:
            raise PresentationError('Finite recovery allows cumulative attempts10..15 only')
        full=self.directory/f'attempt-{number:04d}.full-messages.json'
        _write_new(full,dict(purpose=purpose,messages=messages))
        try:
            projected,record=project_messages(messages,pinned_errors=self.pinned_errors)
        except PresentationError as exc:
            _write_new(self.directory/f'attempt-{number:04d}.projection-error.json',dict(error=str(exc),HTTP_sent=False))
            raise
        record.update(purpose=purpose,cumulative_attempt=number,full_messages_artifact=str(full),
            outgoing_message_count=len(projected),HTTP_started=False)
        _write_new(self.directory/f'attempt-{number:04d}.projection.json',record)
        return self.client.chat(projected,purpose=purpose)


def inspect_recovery(directory, *, source=None):
    directory=Path(directory).resolve();record=original.read_json(directory/'result.json')
    state=original.read_json(directory/'session/SESSION.json')
    if (record.get('status')!='error' or record.get('round')!=bridge.ROUND
            or record.get('mode')!=bridge.MODE or state.get('phase')!='done'
            or state.get('failed')!='transport_worker_failed' or state.get('calls')!=INHERITED_ATTEMPTS
            or state.get('iteration')!=INHERITED_ATTEMPTS or state.get('max_iters')!=10
            or state.get('prompt_variant_id')!=bridge.PROMPT_ID or state.get('algorithm_variant')!=bridge.VARIANT
            or state.get('tuner') is not False or state.get('pool') or state.get('best') is not None
            or state.get('pending') is not None or state.get('confirmation',{}).get('selected') is not None):
        raise ValueError('Explicit failed nine-attempt original bridge session required')
    if source is not None:
        for name in ('model_info','original_model','samples','source_rows','previous_linked_math'):
            if not original._same(record['source'][name],source[name]):
                raise ValueError('Recovery must retain original source '+name)
        for name in ('split_indices','algorithm_variant','maximum_square_steps','maximum_newton_iterations'):
            if record['source'].get(name)!=source.get(name):raise ValueError('Original recovery policy differs: '+name)
    calls=directory/'session/calls'
    requests=sorted(calls.glob('call-*.request.json'));responses=sorted(calls.glob('call-*.response.json'))
    error=original.read_json(calls/'call-0009.error.json')
    if (len(requests)!=9 or len(responses)!=8 or error!={'error':'transport_worker_failed','diagnostics':{'stage':'setup'}}
            or (calls/'call-0009.response.json').exists()):
        raise ValueError('Retain eight provider replies and the ninth local setup failure')
    rows=[json.loads(line) for line in (directory/'session/agent_trace.jsonl').read_text().splitlines()]
    failures=[row for row in rows if row.get('iteration')==8]
    if len(failures)!=1 or failures[0]['fb'].get('stage')!='approximation' or len(failures[0]['fb'].get('errors',[]))!=43:
        raise ValueError('All43 inherited approximation failures required')
    calibration=[row['worker'] for row in record.get('phases',[]) if row.get('phase')=='calibration']
    if (len(calibration)!=1 or calibration[0].get('calibration_completed') is not True
            or calibration[0].get('stable_softmax_completed')!=23*24
            or calibration[0].get('bridge_calibration_completed') is not True):
        raise ValueError('Original23-prefix measured bridge cache required')
    return dict(source_case=str(directory),source_state=str(directory/'session/SESSION.json'),
        inherited_phase='done',inherited_failed=state['failed'],inherited_attempts=9,
        inherited_provider_responses=8,inherited_local_setup_failures=1,next_attempt=10,
        maximum_cumulative_attempts=MAX_ATTEMPTS,maximum_new_attempts=MAX_ATTEMPTS-INHERITED_ATTEMPTS,
        pinned_error_counts={8:43},calibration_cache_reused=True,fresh_calibration_forwards=0,
        original_case_modified=False,HTTP_replay=False),state


def seed_recovery(args):
    check_write_paths(args)
    session=Path(args.session);state_path=session/'SESSION.json'
    if state_path.exists():return
    recovery,state=inspect_recovery(args.resume_from)
    session.mkdir(parents=True,exist_ok=True)
    if any(session.iterdir()):raise ValueError('Recovery session must be fresh')
    source=Path(args.resume_from)/'session'
    shutil.copy2(source/'SESSION.json',session/'inherited-SESSION.json')
    shutil.copy2(source/'agent_trace.jsonl',session/'inherited-agent_trace.jsonl')
    for name in ('calls','sources'):shutil.copytree(source/name,session/name)
    state=copy.deepcopy(state)
    state.update(phase='evaluate',failed=None,iteration=10,max_iters=MAX_ATTEMPTS,
        presentation_variant=PRESENTATION_ID,method_guidance_version=METHOD_GUIDANCE_VERSION,recovery=recovery)
    state.pop('confirmation',None)
    state['messages'][0]['content']+='\n\n'+recovery_guidance()
    _write_new(session/'recovery.json',dict(recovery,created_utc=datetime.now(timezone.utc).isoformat(),
        original_terminal_transition=dict(phase='done',failed='transport_worker_failed'),
        new_explicit_transition=dict(phase='evaluate',failed=None,iteration=10,max_iters=15),
        syntax_guidance=SYNTAX_GUIDANCE,method_guidance_version=METHOD_GUIDANCE_VERSION,
        method_guidance=METHOD_GUIDANCE,guidance_scope='Presentation of existing legal methods and retained diagnostic meaning; numerical evaluator and policies unchanged'))
    from ..planners.transformer_session import save_state
    save_state(state_path,state)


def source_gate(args):
    source=bridge.source_gate(args);recovery,_=inspect_recovery(args.resume_from,source=source)
    source.update(presentation_variant=PRESENTATION_ID,recovery=recovery,
        method_guidance_version=METHOD_GUIDANCE_VERSION,
        cost_resource_variant=resource_policy.VARIANT,
        cost_resource_source=str(Path(resource_policy.__file__).resolve()),
        public_message_budget_bytes=REQUEST_BYTES,feedback_budget_bytes=FEEDBACK_BYTES,
        new_identifier_digest_computations=0,shared_sender_modified=False)
    out=getattr(args,'out',None)
    if out is not None and out.is_dir():
        shutil.copy2(__file__,out/'variant-source/qwen_bounded_feedback.py')
        shutil.copy2(resource_policy.__file__,out/'variant-source/qwen_agent_resources.py')
    return source


def _calibrate(args,recorder,result):
    metadata,_=inspect_recovery(args.resume_from)
    for name,target in (('calibration.json',args.calibration),('unit-width-bridge.json',bridge._bridge_path(args)),
                        ('generation-input.json',args.calibration.parent/'generation-input.json')):
        shutil.copy2(Path(args.resume_from)/name,target)
    result.update(calibration_completed=True,bridge_calibration_completed=True,calibration_samples=23,
        calibration_cache=metadata,fresh_calibration_forwards=0,calibration_forward_attempts=0,
        calibration_forward_completed=0,stable_softmax_attempts=0,stable_softmax_completed=0)


def session_outcome(directory):
    """Same selected-source/C2 gate, with the explicit recorded15-attempt budget."""
    directory=Path(directory);path=directory/'SESSION.json'
    if not path.exists():return dict(status='not_started',done=False,frozen=False,next_step='evaluate',calls=0)
    state=original.read_json(path)
    if (state.get('prompt_version')!='r8-v11' or state.get('max_iters')!=MAX_ATTEMPTS
            or state.get('placement')!='program-plan' or state.get('confirm_rule')!='c2'
            or state.get('presentation_variant')!=PRESENTATION_ID or state.get('prompt_variant_id')!=bridge.PROMPT_ID
            or state.get('method_guidance_version')!=METHOD_GUIDANCE_VERSION
            or state.get('algorithm_variant')!=bridge.VARIANT
            or state.get('tuner') is not False or state.get('recovery',{}).get('inherited_attempts')!=9
            or not 9<=state.get('calls',0)<=MAX_ATTEMPTS):raise ValueError('Explicit bounded recovery session policy differs')
    phase=state['phase']
    if phase not in ('evaluate','tune','confirm','done'):raise ValueError('Unknown original Agent phase')
    outcome=dict(status='running',done=phase=='done',frozen=False,next_step=phase,calls=state['calls'],
        iteration=state['iteration'],inherited_attempts=9,new_attempts=state['calls']-9,
        presentation_variant=PRESENTATION_ID)
    if phase!='done':return outcome
    agent=original.read_json(directory/'AGENT.json');selected=agent['confirm']['selected'];frozen_path=directory/'FROZEN.json'
    if state.get('failed') or agent.get('status')=='error':
        return dict(outcome,status='error',error=agent.get('error') or state.get('failed'))
    if selected is None:
        if frozen_path.exists() or agent.get('status')!='confirm_failed':raise ValueError('Unselected terminal frozen artifacts differ')
        return dict(outcome,status='quality_rejected')
    frozen=original.read_json(frozen_path);text=(directory/'program.ptdsl').read_text()
    semantic=lambda source:ast.dump(ast.parse(source),include_attributes=False)
    passed=[row for row in agent['confirm']['evals'] if row.get('passed') is True
        and row.get('iteration')==selected['iteration'] and row.get('origin')==selected['origin']]
    if (agent.get('status')!='feasible' or semantic(text)!=semantic(selected['source'])
            or not original._same(frozen['program'],directory/'program.ptdsl') or len(passed)!=1
            or passed[0].get('tail',{}).get('passed') is not True):
        raise ValueError('Frozen original selected source and passing C2 tail required')
    return dict(outcome,status='ok',frozen=True,selected=selected,
        quality_scope='Internal23cal/20heldout/20confirm; no standard LM corpus or report')


def _agent_functions(directory):
    functions=resource_policy.adapt_agent_functions(bridge._agent_functions())
    functions['feedback_message']=FeedbackPresenter(Path(directory)/'full-feedback',functions['feedback_message'])
    return functions


def _agent_phase(args,recorder,result):
    from ..planners.transformer_session import load_state,save_state,write_session_result,output_tokens
    shutil.copyfile(resource_policy.__file__,args.work_dir/'qwen-agent-resource-policy.py')
    seed_recovery(args);state_path=args.session/'SESSION.json';state=load_state(state_path)
    outcome=session_outcome(args.session)
    if state['phase']=='done':
        result.update(session=outcome,status=outcome['status'],candidate_frozen=outcome['frozen']);return
    pinned={int(key):value for key,value in state['recovery']['pinned_error_counts'].items()}
    if args.worker_phase=='prompt':
        from ..autonomous_budget import audit_public_messages
        projected,record=project_messages(state['messages'],pinned_errors=pinned)
        _write_new(args.work_dir/'public-api-request.json',dict(endpoint='https://api.deepseek.com/chat/completions',
            presentation_variant=PRESENTATION_ID,messages=projected,message_audit=audit_public_messages(projected),
            projection=record,raw_sentences_included=False,raw_weights_included=False,report_samples=0))
        result.update(prompt_completed=True,api_required=True,session=outcome,recovery=state['recovery']);return
    # The public budget gate precedes model reconstruction and provider creation.
    project_messages(state['messages'],pinned_errors=pinned)
    ctx=bridge._context(args,result);functions=_agent_functions(args.session);client=None;before=state['phase']
    if before=='evaluate':
        from ..autonomous_budget import RenewingLLMClient
        from .bert_sst2_worker import RecordedRenewingClient
        recorded=RecordedRenewingClient(RenewingLLMClient(budget_path=args.ledger,allowance_path=args.allowance,
            recorder=recorder,artifact_dir=args.session/'api',model='deepseek-flash',reasoning_effort='high',
            max_tokens=output_tokens('r8-v11'),timeout=600),args.session/'calls',recorder=recorder)
        client=ProjectedClient(recorded,args.session/'presentation',pinned_errors=pinned,attempt_offset=state['calls'])
    try:
        with recorder.span('bounded_agent_'+before):
            functions['agent_step'](ctx,state,trace_path=args.session/'agent_trace.jsonl',client=client)
            if state['phase']=='done':
                write_session_result(functions['agent_result'](state),args.session,prompt_version='r8-v11',tune_policy=state['tune_policy'])
    finally:save_state(state_path,state)
    outcome=session_outcome(args.session)
    result.update(step=dict(complete=True,step=before,next_step=state['phase'],iteration=state['iteration'],session_calls=state['calls']),
        api_calls=client.calls if client else 0,session=outcome,candidate_frozen=outcome['frozen'],recovery=state['recovery'])
    if outcome['done']:result['status']=outcome['status']


def _program(args,result):
    delegated=bridge._clone_namespace(original,{'session_outcome':session_outcome})
    info,program=delegated['_program'](args,result)
    calibration=original.read_json(args.calibration)
    record=bridge.checked_bridge(info,calibration,original.read_json(bridge._bridge_path(args)),args)
    bridge.candidate_gate(program,types.SimpleNamespace(stable_bridge=record))
    return info,program


def phase_command(args,phase,work):
    functions=bridge._namespace();functions['MODULE']=MODULE
    return functions['phase_command'](args,phase,work)+['--resume-from',str(args.resume_from)]


def parser():
    parser=original.parser();parser.add_argument('--resume-from',type=Path,required=True)
    return parser


def check_write_paths(args):
    source=Path(args.resume_from).resolve()
    names=('out','work_dir','session','codegen','calibration')
    for name in names:
        value=getattr(args,name,None)
        if value is None:continue
        target=Path(value).resolve()
        if target==source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError('Recovery writes must use an isolated case outside the inherited source')


def _recovery_clock_label(function):
    """Replace one trusted display literal, preserving its private globals."""
    constants=function.__code__.co_consts
    if constants.count(OLD_CLOCK_SCOPE)!=1:raise ValueError('Original outer clock label source changed')
    code=function.__code__.replace(co_consts=tuple(CLOCK_SCOPE if item==OLD_CLOCK_SCOPE else item for item in constants))
    return types.FunctionType(code,function.__globals__,function.__name__,function.__defaults__,function.__closure__)


def _namespace():
    functions=bridge._namespace()
    existing_execute,existing_worker=_recovery_clock_label(functions['execute']),functions['worker']
    def execute(args):
        check_write_paths(args)
        return existing_execute(args)
    def worker(args):
        check_write_paths(args)
        return existing_worker(args)
    functions.update(__file__=__file__,__doc__=__doc__,MODULE=MODULE,ROUND=ROUND,MODE=MODE,
        source_gate=source_gate,_calibrate=_calibrate,_agent_phase=_agent_phase,_program=_program,
        session_outcome=session_outcome,phase_command=phase_command,parser=parser,execute=execute,worker=worker)
    return functions


def main(argv=None):return _namespace()['main'](argv)


if __name__=='__main__':raise SystemExit(main())
