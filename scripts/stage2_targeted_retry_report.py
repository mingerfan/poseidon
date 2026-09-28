"""Independent terminal reporting for the r153 retry; never calls an API or executes a candidate."""
import argparse,json,os,sys,time
from pathlib import Path

def summarize():
    from pathlib import Path
    import json,sys,hashlib,collections,datetime
    sys.path.insert(0,'scripts/baseline')
    sys.path.insert(0,'scripts/baseline/benchmarks/tools')
    from benchmark_graph import digest
    from stage2_agent_pilot_plan import sha,check_binding
    from semantic_benchmark_execution import runtime_sources
    from campaign_failure_taxonomy import provider_failure
    from stage2_targeted_retry_plan import classify
    r=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results');q=r/'stage2-targeted-retry-r153'
    parents={}
    def read(path,sealed=True,expected=None):
        path=Path(path);h=sha(path)
        if expected and h!=expected:raise ValueError('Changed parent '+str(path))
        v=json.loads(path.read_text())
        if sealed:check_binding(v)
        parents[str(path)]=h
        return v
    plan=read(q/'plan.json')
    if runtime_sources()!=plan['source_hashes']:raise ValueError('Runtime drift')
    for name,h in plan['proposal_files'].items():
        if sha(Path(name))!=h:raise ValueError('Tool source drift')
    master=read(q/'report.json')
    if master['plan_binding']!=plan['binding']:raise ValueError('Master binding')
    review=read(Path(plan['proposal']),expected=plan['proposal_sha256'])
    auth=read(Path(plan['authorization']),expected=plan['authorization_sha256'])
    seen={};totals=collections.Counter();files_checked=0;shard_states=[]
    terminal={ref['output']:ref for ref in master['rows']}
    for ref in plan['shards']:
        frozen=ref['plan'];out=Path(ref['output']);adir=Path(ref['audit'])
        if ref['output'] not in terminal:
            shard_states.append(dict(output=ref['output'],status='not_independently_audited'))
            continue
        t=terminal[ref['output']]
        a=read(adir/'report.json',expected=t['audit_sha256'])
        sp=read(out/'plan.json')
        if sp!=frozen or a['plan_binding']!=sp['binding'] or a['source_sha256']!=digest(plan['source_hashes']):raise ValueError('Shard identity')
        for f,h in a['parents'].items():
            if sha(Path(f))!=h:raise ValueError('Audit parent changed')
        specs={c['id']:c for c in sp['cases']}
        if len(specs)!=len(sp['cases']) or len(a['rows'])!=len(specs):raise ValueError('Scope count')
        for key in ('generations','http_attempts','first_attempt_passed','successful_after_repair','real_compiled_attempts','real_executed_attempts'):
            totals[key]+=a[key]
        for row in a['rows']:
            ident=row['id']
            if ident in seen or ident not in specs:raise ValueError('Duplicate/unknown task')
            rec=dict(row,group=specs[ident]['group'],runtime_source_sha256=digest(plan['source_hashes']),
                guidance=specs[ident]['request']['generation_guidance']['version'],
                original_status='failed',frozen_evaluation_identity=specs[ident]['evaluation_identity'])
            if row['status'] in ('passed','failed'):
                folder=Path(row['evidence'])
                rep=read(folder/'report.json',False,row['report_sha256'])
                if rep['status']=='provider_failed':
                    rec['terminal_failure_layer']='provider';rec['provider_failure']=provider_failure(rep)
                if row['status']=='passed':
                    aa=read(adir/(ident+'.audit.json'),True,row['audit_sha256'])
                    if aa['case_id']!=ident or aa['request_id']!=specs[ident]['request_id']:raise ValueError('Pass identity')
                    for f,h in aa['files'].items():
                        path=(folder/f).resolve()
                        if not path.is_relative_to(folder.resolve()) or sha(path)!=h:raise ValueError('Pass artifact changed')
                        files_checked+=1
                    c=aa['comparison']
                    if not(c['passed'] and c['atol']==1e-5 and c['rtol']==1e-4 and all(all(x) for x in c['elementwise_pass'])):raise ValueError('Numerical gate')
                    if not any(x.get('compiled') and x.get('executed') and x.get('status')=='passed' for x in rep['attempts']):raise ValueError('Missing real execution')
                    rec['follow_up']='单列本轮成功，保留旧失败；不计入同版本全量成功率'
                else:
                    action,reason,direction=classify([dict(status='failed',report=rep)])
                    rec['diagnosed_category']=action
                    rec['follow_up']=direction+'；本轮授权已使用，不自动再重试'
                    rec['reason']=reason
            else:rec['follow_up']='未获得独立终态；先审计是否已调用，不自动补跑或重置预算'
            seen[ident]=rec
        shard_states.append(dict(output=ref['output'],status=t['status'],statuses=a['statuses']))
    planned={s['id'] for ref in plan['shards'] for s in ref['plan']['cases']}
    missing=planned-set(seen)
    for ident in missing:seen[ident]=dict(id=ident,status='unaudited_or_interrupted',follow_up='先核验现场，不重启或推定未收费')
    if len(seen)!=257:raise ValueError('Final scope')
    if totals['generations']>1028 or totals['http_attempts']>4112:raise ValueError('Budget overrun')
    rows=sorted(seen.values(),key=lambda x:x['id'])
    statuses=dict(collections.Counter(x['status'] for x in rows))
    result=dict(format='poseidon-targeted-retry-result-r154',plan_binding=plan['binding'],parents=parents,
        runtime_source_sha256=digest(plan['source_hashes']),planned=257,statuses=statuses,
        groups={g:dict(collections.Counter(x['status'] for x in rows if x.get('group','unknown')==g)) for g in sorted({x.get('group','unknown') for x in rows})},
        terminal_failure_layers=dict(collections.Counter(x.get('terminal_failure_layer','unknown') for x in rows if x['status']=='failed')),
        totals=dict(totals),verified_pass_artifact_files=files_checked,rows=rows,shards=shard_states,
        queue_finished=master['queue_finished'],queue_failure=master['failure'],seconds=master['seconds'],
        original_failure_review=review['counts'],deferred_pipeline_tasks=33,prior_recovered_not_retried=4,
        no_cross_version_best_result_score=True,no_further_retry_authorized=True,
        limitation='This report rechecks this bounded retry cohort. Historical failures and billing uncertainty remain. Success does not prove arbitrary inputs or all DSL combinations.')
    result['reporter_sha256']=sha(Path(__file__))
    result['binding']=digest(result)
    dest=r/'stage2-targeted-retry-result-r154.json'
    with dest.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:result[k] for k in ('binding','statuses','terminal_failure_layers','totals','seconds','queue_failure')},ensure_ascii=False))
    
    return result
    

def write_summary(result):
    root=Path(__file__).resolve().parents[1]
    from collections import Counter
    s=result['statuses']; t=result['totals']
    terminal=sum(s.get(k,0) for k in ('passed','failed'))
    text=f"""# 第二阶段定向重测结果（r154）

本报告由独立分片审计后生成。只描述本次有界重测，不覆盖旧失败或把不同版本择优合并。
实际开发和执行：ARM Ubuntu /home/lhohy/Code/Poseidon，lhy-agent-dsl。

## 结果

- 本轮计划257项：152静态、94provider、11数值失败的再评测。
- 已审计通过{s.get('passed',0)}，失败{s.get('failed',0)}；其他状态{257-terminal}。
- 首次语义尝试通过{t.get('first_attempt_passed',0)}，修复后通过{t.get('successful_after_repair',0)}。
- 已审计生成{t.get('generations',0)}次，HTTP尝试{t.get('http_attempts',0)}次。
- 真实编译尝试{t.get('real_compiled_attempts',0)}，密态执行尝试{t.get('real_executed_attempts',0)}；
  尝试数含失败及修复，不能当作唯一通过模型数。
- 队列终止原因：{result['queue_failure']}；累计{result['seconds']:.1f}秒。
- 完整状态：{json.dumps(s,ensure_ascii=False)}。
- 未独立审计/中断项不算pass，调用统计不覆盖尚未入账调用，不是账单金额证明。
- 原有33项编译/产物问题本轮不重跑；4项已有恢复成功，不重复调用。
- 每个通过项重新核验了tracing、编译/执行、解密、独立参考、产物哈希及逐项门限；
  本汇总共核验{result['verified_pass_artifact_files']}个通过产物文件。
- 冻结门限保持1e-5 + 1e-4*abs(reference)，没有更改reference或安全参数。
- 本轮授权已经使用；未运行或失败项不得自动另开新批次。

## 失败层及后续方向

终态失败层：{json.dumps(result['terminal_failure_layers'],ensure_ascii=False)}。
完整轨迹仍在各候选report中，不能用最后静态/provider错误掩盖前轮编译或数值失败。

"""
    for row in result['rows']:
        if row['status']=='passed':continue
        text+=f"### {row['id']} — {row['status']}\n\n"
        text+=f"- 最后失败层：{row.get('terminal_failure_layer','未独立确认')}。\n"
        if row.get('provider_failure'):
            text+=f"- Provider安全错误码：{row['provider_failure']['error_code']}。\n"
        for a in row.get('attempts',[]):
            text+=f"- 尝试{a['index']}：{a.get('failure_layer')}；{a.get('diagnostic') or '无附加诊断'}。\n"
        text+=f"- 修改方向：{row.get('follow_up','先复核失败层，不自动重试')}。\n"
        if row.get('evidence'):text+=f"- 原始证据：{row['evidence']}。\n"
        text+="\n"
    text+="""## 原33项暂缓问题

这些任务有真实编译/产物故障轨迹：历史69条Earth乘法类型诊断均在当前固定配置下容量不足。
helper_0112/0118/0119还保留共6份level=1、scale=75的输出产物，不满足scale<60的执行门禁。
下一步核对Earth与HEVM level语义、Mul结果scale、rescale调度及最终输出约束；
再判断能否用保持数学语义的等价求值树执行。不是已证明所有写法都不可执行。
不删多项式项、不降低安全性、不删除产物检查、不用bootstrap模拟通过。
逐项索引和证据在stage2-targeted-retry-r153.md及stage2-targeted-pipeline-diagnosis-r153.json。

## 使用和限制

本轮定向构造使用已有explicit-v4说明；自由/helper保留explicit-v3。
DeepSeek flash是服务商别名，不宣称跨日期模型权重完全相同；本次失败恢复不是严格因果对照。
模型、公开权重、布局和误差门限不变，没有手写替换Agent输出。
第二阶段原r145/r150成绩保留；本轮成功不等于377个语义分区全部成功覆盖。

机器可读完整报告位于：
/home/lhohy/poseidon-work/platforms/aarch64-linux/results/stage2-targeted-retry-result-r154.json
分片独立审计位于同results下stage2-targeted-retry-r153-shard-XX-audit/report.json。
启动前12项离线门禁测试通过，0失败、0skip。
本轮无安装、依赖下载、SDK重建、commit、push或PR；生产源码未修改。
"""
    target=root/'docs/baseline/stage2-targeted-retry-result-r154.md'
    with target.open('x') as f:f.write(text)
    h=root/'docs/baseline/mac-migration-handoff.md';old=h.read_text()
    marker='## r154：第二阶段定向重测已终止并汇总\n\n'
    intro=(marker+f"257项本轮状态：{json.dumps(s,ensure_ascii=False)}。已审计调用{t.get('generations',0)}生成/{t.get('http_attempts',0)}HTTP。\n"
        f"队列结束原因：{result['queue_failure']}。完整报告stage2-targeted-retry-result-r154.md及results同名JSON。\n"
        "33项编译/产物问题未重跑，4项已恢复不重跑；旧分数保留，失败/中断不计pass。\n"
        "本轮有界授权已使用，不再自动重试，不重启旧claim。当前无新安装/commit/push/PR。\n\n"
        "下文r153是启动历史记录，其中“正在执行”不再代表当前状态。\n\n")
    if marker in old:raise ValueError('Handoff already summarized')
    h.write_text(old.split('\n',1)[0]+'\n\n'+intro+old.split('\n',1)[1])
    return target

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    sys.path.insert(0,str(root/'scripts/baseline'))
    sys.path.insert(0,str(root/'scripts/baseline/benchmarks/tools'))
    from stage2_agent_pilot_plan import sha,check_binding
    from semantic_benchmark_execution import runtime_sources
    results=Path('/home/lhohy/poseidon-work/platforms/aarch64-linux/results')
    queue=results/'stage2-targeted-retry-r153'
    plan=json.loads((queue/'plan.json').read_text());check_binding(plan)
    if plan['binding']!='af6f621985a25b1a2df4bd50b11782780bb1188e08625e7603c37cd806130faf':raise ValueError('Exact report cohort')
    if runtime_sources()!=plan['source_hashes']:raise ValueError('Runtime drift')
    for name,h in plan['proposal_files'].items():
        if sha(root/name)!=h:raise ValueError('Tool source drift')
    if args.check:
        print(json.dumps(dict(report_preflight=True,paid_calls=0,source_unchanged=True)));return 0
    began=time.monotonic()
    while not (queue/'report.json').exists():
        if not args.wait:raise ValueError('Queue not terminal; no final report')
        if time.monotonic()-began>=43500:raise TimeoutError('Queue report wait bound')
        time.sleep(5)
    result=summarize()
    path=write_summary(result)
    print(json.dumps(dict(final_report=str(path),binding=result['binding']),ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
