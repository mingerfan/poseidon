"""Publish S0 source-qualified adapter findings; invoke no compiler or model."""
import argparse
from datetime import datetime,timezone
import html
import json
import os
from pathlib import Path
import signal
import time

from .completion_metrics import artifact_bytes,persist_case_completion
from .r10_paper_metrics import StageRecorder,read_events,summarize_events,write_json
from .a_group_profile_sum_policy import EXCLUDED_OP,POLICY_ID

ROUND='A-GROUP-S0-SAME-CONFIG-FEASIBILITY'
MODE='metadata-only-source-qualified-adapter-note'
MODELS=('resnet20','vgg16','alexnet')
FACTORIES=dict(zip(MODELS,('resnet20_silu','vgg16_silu_avgpool','alexnet_silu_avgpool')))
PARAMETERS=dict(polynomialDegree=65536,rescalingFactor=43,levelLowerBound=1,levelUpperBound=8,
                bootstrapLevelLowerBound=1,bootstrapLevelUpperBound=5)


def read(path):
    path=Path(path)
    if path.is_symlink()or not path.is_file()or path.stat().st_size>4*1024**2:
        raise ValueError('A bounded regular source-evidence JSON is required')
    return json.loads(path.read_text())


def require(value,message):
    if not value:raise ValueError(message)


def project(orbit,heir,*,orbit_path,heir_path):
    """Project explicit saved findings, never reinterpret a gap as a run failure."""
    target=orbit['qwen48_contract']['target'];contract=heir['requested_contract']
    require(all(target.get(k)==v and contract['compiler_profile_fields'].get(k)==v
                for k,v in PARAMETERS.items()),'Both evidence files must describe the exact same qwen48 fields')
    require(target.get('Rule_slots')==contract.get('logical_slots')==16384,'Original logical slots differ')
    require(contract.get('param_group')=='ours-N16-boot5','Same-configuration evidence required')
    require(orbit['current_actions'].get('parser_or_solver_runs')==0
            and orbit['current_actions'].get('new_model_Count_compile_download_patch_API_CKKS')==0,
            'This S0 projection requires the saved zero-execution Orbit source review')
    require(heir['activity'].get('new_native_compiles')==0 and heir['activity'].get('new_model_exports')==0
            and heir['activity'].get('new_math_runs')==0,'This S0 projection requires zero-execution HEIR source review')
    om={r['model']:r for r in orbit['models']};hm={r['model']:r for r in heir['rows']}
    require(len(om)==len(orbit['models'])and len(hm)==len(heir['rows']),'Source models must be unique')
    require(set(MODELS)<=set(om)and set(MODELS)<=set(hm),'Both systems need three original CNN findings')
    rows=[]
    for system,source,index in(('orbit',orbit_path,om),('heir',heir_path,hm)):
        for model in MODELS:
            original=index[model]
            if system=='orbit':
                require(original.get('classification')=='not_run_adapter_required'
                    and original.get('new_compiler_runs')==0 and original.get('same_Rule63_direct_ready')is False,
                    'Unexpected Orbit result requires a new projection policy')
                reasons=[orbit['input_contract']['direct_gap']]
                reasons += [item['fact']for item in orbit['qwen48_contract']['gaps']]
                reasons += [item['fact']for item in orbit['cost_contract']['gaps']]
                arithmetic=None
                count_contract=orbit['bootstrap_count_contract']['required']
                missing=['upscale_single']
            else:
                require(original.get('formal_run_status')=='not_run'
                    and original.get('scope_status')=='adapter_parameter_contract_gap'
                    and original.get('compiler_attempts_in_this_review')==0
                    and original.get('compiler_failure_recorded')is False,
                    'Unexpected HEIR result requires a new projection policy')
                require(original.get('expanded_execution_bootstraps')is None
                    and original.get('predicted_latency_s')is None,'No same-configuration measured result may be inherited')
                reasons=list(original['adaptation_needed'])
                reasons.append(heir['source_interfaces']['profile_prediction_gap'])
                arithmetic=original['original_Rule63_arithmetic_adapter_reusable']
                count_contract='Require one-inference control-flow expansion and ciphertext multiplicity; source text bootstrap sites alone do not establish execution count.'
                missing=list(original['missing_price_ops'])
            rows.append(dict(system=system,model=model,formal_run_status='not_run',diagnostic_status='adapter_gap',
                source_qualification_status=original.get('classification',original.get('scope_status')),
                compiler_attempts=0,compiler_failure_recorded=False,official_system_unsupported_proven=False,
                arithmetic_adapter_reusable=arithmetic,same_Rule63_direct_entry_ready=False,
                exact_qwen48_management_ready=False,fully_expanded_execution_count_available=False,
                same_table_full_prediction_available=False,compile_s=None,expanded_execution_bootstraps=None,
                pred_latency_s=None,known_cost_partial_s=None,
                source_reported_missing_price_ops=missing,
                missing_price_ops=[op for op in missing if op!=EXCLUDED_OP],
                policy_excluded_ops=[op for op in missing if op==EXCLUDED_OP],
                policy_excluded_relin=EXCLUDED_OP in missing,prediction_policy_id=POLICY_ID,
                missing_price_ops_scope='Known source-identified entries; complete same-configuration execution inventory remains unavailable',
                bootstrap_count_contract=count_contract,adapter_gaps=reasons,source_evidence=str(source)))
    extension=[]
    if 'bert_tiny'in hm:
        original=hm['bert_tiny']
        require(original.get('formal_run_status')=='not_run'and original.get('compiler_attempts_in_this_review')==0,
                'BERT extension needs explicit unrun evidence')
        extension.append(dict(system='heir',model='bert_tiny',formal_run_status='not_run',
            diagnostic_status='adapter_gap',compiler_attempts=0,compiler_failure_recorded=False,
            official_system_unsupported_proven=False,adapter_gaps=original['adaptation_needed'],
            source_evidence=str(heir_path),compile_s=None,expanded_execution_bootstraps=None,pred_latency_s=None))
    return dict(schema='s0-same-config-source-feasibility-v1',status='source_qualification_complete',
        param_group='ours-N16-boot5',target=dict(PARAMETERS,logical_slots=16384),rows=rows,extension_rows=extension,
        scope='Current own adapters and source-qualified parameter/input/cost gaps. No same-configuration compiler attempt or official system-wide unsupported result.',
        compiler_attempts=0,new_model_math_Count_compile_API_CKKS=0,runs_jsonl_rows_added=0,
        source_evidence=dict(orbit=str(orbit_path),heir=str(heir_path)),
        concrete_QP_verified=contract.get('concrete_same_QP_verified'),
        compiler_json_has_concrete_Q_primes=contract.get('concrete_Q_primes_in_compiler_json'),
        compiler_json_has_concrete_P_primes=contract.get('concrete_P_primes_in_compiler_json'),
        required_inputs_preserved='Original DSL, ModelInfo graph/weights, RuntimePlan and executable bundle; redundant raw trace/Emit NPZ are not consumed',
        orbit_count_note=orbit['bootstrap_count_contract']['required'],
        heir_parameter_note=heir['semantic_gap']['required'],
        source_reference_only=True,formal_S1_results_available=False)


def source_cases(cases):
    from .a_group_future_closed_archive import closed,producer_births
    from .a_group_cold_closed_archive import clean_cleanup_metadata
    from .. import paths
    require(len(cases)==3 and len(set(cases))==3,'Three distinct explicit A2b source cases required')
    found={}
    for case in cases:
        require(case.parent==paths.RESULTS_DIR.resolve()and not case.is_symlink(),'Source must be a configured direct results child')
        closure,result=closed(case,case.name)
        require(result.get('round')=='COLD-RULE63'and result.get('placement')=='depth'
            and result.get('model')in FACTORIES.values(),'An original cold Rule63 depth case is required')
        require(result.get('monitor_stopped')is True,'Source monitor must be closed')
        clean_cleanup_metadata(result);births=producer_births(case)
        model=next(name for name,factory in FACTORIES.items()if factory==result['model'])
        require(model not in found,'Duplicate source model')
        found[model]=dict(**closure,model=model,scientific_error=result.get('error'),
            saved_producer_births_closed=births,fresh_generation_completed=(result.get('completed')or{}).get('generation')==1,
            scientific_result_unchanged=True,raw_or_constant_payloads_read=0)
    require(set(found)==set(MODELS),'Exactly the original three CNN models are required')
    return [found[name]for name in MODELS]


def render(note):
    def e(text):return html.escape(str(text))
    table=[]
    for row in note['rows']:
        arithmetic=('现有算术可复用，入口待接线'if row['arithmetic_adapter_reusable']else'原程序入口待适配')
        table.append('<tr>'+''.join('<td>'+e(value)+'</td>'for value in(row['system'],row['model'],arithmetic,
            '初始8 / 自举1→5尚未接通','待实际展开','同表映射／缺价待核','未运行'))+'</tr>')
    issues=[]
    for system in('orbit','heir'):
        examples=next(row for row in note['rows']if row['system']==system)
        issues.append('<p><b>'+e(system)+'</b> '+e(' '.join(examples['adapter_gaps']))+'</p>')
    return ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>S0 同配置接入检查</title>'
        '<style>body{font-family:Arial,sans-serif;margin:26px;max-width:1250px;color:#202124;font-size:13px}'
        'h1{font-size:22px}table{border-collapse:collapse;width:100%;font-size:12px}'
        'td,th{border:1px solid #bbc0c5;padding:7px;text-align:left}th{background:#eef2f5}'
        'p{line-height:1.45;overflow-wrap:anywhere}.scope{background:#f2f5f8;padding:12px}</style>'
        '<h1>S0：同一 Rule63 与 qwen48 配置的接入检查</h1>'
        '<p class="scope">目标：N=65536，16384逻辑槽，rescale43，初始可用8级，自举1→5。'
        '本页汇总已保存的源码证据；本轮模型、数学、Count及编译调用均为0。六个正式格均未运行，缺口属于当前接入与参数管理。</p>'
        '<table><thead><tr><th>系统</th><th>模型</th><th>原Rule63</th><th>参数合同</th>'
        '<th>执行自举数</th><th>qwen48预测</th><th>正式状态</th></tr></thead><tbody>'+''.join(table)+
        '</tbody></table>'+''.join(issues)+'<p>预测延迟、自举执行次数和编译时间均留空。'
        '此前不同配置的尝试独立保留。本页未向正式 runs.jsonl 追加编译失败或 S1 结果。</p>'
        '<p>证据：'+e(note['source_evidence']['orbit'])+'；'+e(note['source_evidence']['heir'])+'</p></html>')


def publish(destination,note):
    destination.mkdir(parents=True,exist_ok=True)
    targets=[destination/'s0_same_config_feasibility.json',destination/'s0_same_config_feasibility.html']
    require(not any(os.path.lexists(p)for p in targets),'Preserve existing feasibility publication')
    text=[json.dumps(note,ensure_ascii=False,indent=2)+'\n',render(note)]
    temporary=[]
    for path,value in zip(targets,text):
        stage=path.with_name(path.name+'.tmp-'+str(os.getpid()))
        with stage.open('x')as stream:stream.write(value);stream.flush();os.fsync(stream.fileno())
        temporary.append(stage)
    for stage,path in zip(temporary,targets):os.replace(stage,path)
    descriptor=os.open(str(destination),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return [str(p)for p in targets]


def run(args):
    started=time.perf_counter_ns()
    from .. import paths
    from ..registry import git_rev
    from .cold_rule63_whole_model import append_registry
    require(args.out.parent==paths.RESULTS_DIR.resolve()and not os.path.lexists(args.out),'Fresh configured results child required')
    args.out.mkdir();rec=StageRecorder(args.out/'events.jsonl')
    result=dict(round=ROUND,run_id=args.out.name,model='three-original-CNN-source-feasibility',mode=MODE,
        status='running',error=None,metadata_only=True,compiler_attempts=0,new_models_math_Count_compile_API_CKKS=0,
        formal_paper_rows_added=0,diagnostic_failure_category=None,source_cases=[],outputs=[],
        requested_source_cases=[str(case)for case in args.source_case],
        evidence_sources=dict(orbit=str(args.orbit_evidence),heir=str(args.heir_evidence)))
    old={s:signal.getsignal(s)for s in(signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def interrupted(sig,_frame):raise InterruptedError('S0 metadata publication interrupted by signal'+str(sig))
    try:
        for s in old:signal.signal(s,interrupted)
        with rec.span('closed_A2b_metadata_admission'):result['source_cases']=source_cases(args.source_case)
        with rec.span('read_final_source_evidence'):
            note=project(read(args.orbit_evidence),read(args.heir_evidence),
                orbit_path=args.orbit_evidence,heir_path=args.heir_evidence)
            note.update(created_utc=datetime.now(timezone.utc).isoformat(),source_case_closure=result['source_cases'],
                metadata_case=str(args.out),host_pid_namespace_observed=os.readlink('/proc/self/ns/pid'))
            write_json(args.out/'note.json',note)
        with rec.span('publish_metadata_note'):result['outputs']=publish(args.destination,note)
        result['status']='ok'
    except BaseException as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc)[:2000],diagnostic_failure_category='driver_error')
    finally:
        for s in old:signal.signal(s,signal.SIG_IGN)
        timing=summarize_events(read_events(args.out/'events.jsonl'),termination_observed_ns=time.perf_counter_ns())
        write_json(args.out/'timing.json',timing);write_json(args.out/'result.json',result)
        append_registry(paths.RESULTS_DIR/'registry.jsonl',dict(run_id=args.out.name,round=ROUND,
            model=result['model'],mode=MODE,git_rev=git_rev(),metrics=dict(status=result['status'],metadata_only=True,
            compiler_attempts=0,formal_paper_rows_added=0),artifacts_dir=str(args.out),ts=datetime.now(timezone.utc).isoformat()))
        persist_case_completion(args.out,round_label=ROUND,started_ns=started,status=result['status'],timing=timing,
            required_artifact_bytes=artifact_bytes(args.out))
        for s,handler in old.items():signal.signal(s,handler)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in('orbit-evidence','heir-evidence','out','destination'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-case',type=Path,action='append',required=True)
    args=parser.parse_args(argv)
    for key in('orbit_evidence','heir_evidence','out','destination'):setattr(args,key,getattr(args,key).expanduser().resolve())
    args.source_case=[path.expanduser().resolve()for path in args.source_case]
    result=run(args);print(json.dumps({k:result[k]for k in('status','metadata_only','compiler_attempts','error')}))
    return int(result['status']!='ok')


if __name__=='__main__':raise SystemExit(main())
