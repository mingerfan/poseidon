"""Read-only terminal audit aggregation for the 50-case r175 Agent repair batch."""
import argparse,collections,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts/baseline"),str(ROOT/"scripts/baseline/benchmarks/tools")]
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_graph import digest
from workspace_paths import RESULTS
from semantic_benchmark_execution import runtime_sources
QUEUE=RESULTS/"stage2-agent-repair-r175"
BINDING=None
def summarize(check=False):
 global BINDING
 parents={}
 def read(p,expected=None,sealed=True):
  p=Path(p)
  if p.is_symlink():raise ValueError("Symlink report")
  if expected and sha(p)!=expected:raise ValueError("Parent changed")
  value=json.loads(p.read_text())
  if sealed:check_binding(value)
  parents[str(p)]=sha(p);return value
 plan=read(QUEUE/"plan.json")
 from stage2_agent_repair_plan_r175 import verify
 verify(plan,QUEUE)
 BINDING=plan["binding"]
 if plan["binding"]!=BINDING or runtime_sources()!=plan["source_hashes"]:raise ValueError("Cohort/source mismatch")
 for n,h in plan["proposal_files"].items():
  if sha(ROOT/n)!=h:raise ValueError("Tool drift")
 if not all(sha(Path(n))==h for n,h in plan["compiler_guard"].items()):raise ValueError("Compiler changed")
 if check:return dict(report_preflight=True,paid_calls=0,binding=BINDING)
 master=read(QUEUE/"report.json")
 if master["plan_binding"]!=BINDING:raise ValueError("Queue identity")
 terminal={r["output"]:r for r in master["rows"]};rows=[];totals=collections.Counter();files_checked=0
 for ref in plan["shards"]:
  spec={s["id"]:s for s in ref["plan"]["cases"]};entry=terminal.get(ref["output"])
  if entry is None:
   for name in spec:rows.append(dict(id=name,status="unaudited_or_not_run",follow_up="先核验调用账本；不自动重启。"))
   continue
  a=read(Path(ref["audit"])/"report.json",entry["audit_sha256"])
  if a["plan_binding"]!=ref["plan"]["binding"] or a["source_sha256"]!=digest(plan["source_hashes"]):raise ValueError("Audit identity")
  for f,h in a["parents"].items():
   if sha(Path(f))!=h:raise ValueError("Audit evidence changed")
  if {r["id"] for r in a["rows"]}!=set(spec):raise ValueError("Audited scope")
  for key in ("generations","http_attempts","first_attempt_passed","successful_after_repair","real_compiled_attempts","real_executed_attempts"):
   totals[key]+=a[key]
  for row in a["rows"]:
   r=dict(row)
   if row["status"]=="passed":
    folder=Path(row["evidence"])
    if folder.is_symlink() or folder.resolve().parent!=RESULTS.resolve():raise ValueError("Evidence path")
    details=read(Path(ref["audit"])/(row["id"]+".audit.json"),row["audit_sha256"])
    if details["request_id"]!=spec[row["id"]]["request_id"]:raise ValueError("Pass request")
    for f,h in details["files"].items():
     p=(folder/f).resolve()
     if not p.is_relative_to(folder.resolve()) or sha(p)!=h:raise ValueError("Pass artifact changed")
     files_checked+=1
    c=details["comparison"]
    if not(c["passed"] and c["atol"]==1e-5 and c["rtol"]==1e-4 and all(all(x) for x in c["elementwise_pass"])):raise ValueError("Numerical acceptance")
    r["follow_up"]="独立审计通过；保留旧失败，不合并为同版本全量成绩。"
   else:r["follow_up"]="保留失败层和全部尝试；本轮不自动追加重试。"
   rows.append(r)
 if len(rows)!=50 or len({r["id"] for r in rows})!=50:raise ValueError("Denominator")
 if totals["generations"]>200 or totals["http_attempts"]>800:raise ValueError("Calls exceed bound")
 statuses=dict(collections.Counter(r["status"] for r in rows))
 result=dict(format="poseidon-agent-repair-result-r176",plan_binding=BINDING,planned=50,statuses=statuses,
  rows=rows,totals=dict(totals),parents=parents,verified_pass_artifact_files=files_checked,
  queue_finished=master["queue_finished"],queue_failure=master["failure"],seconds=master["seconds"],
  source_hashes=plan["source_hashes"],compiler_files_unchanged=93,deferred_compiler_cases=34,
  terminal_failure_layers=dict(collections.Counter(r.get("terminal_failure_layer","unknown") for r in rows if r["status"]=="failed")),
  no_cross_version_best_score=True,automatic_retry=False,reporter_sha256=sha(Path(__file__)))
 result["binding"]=digest(result)
 for path in (RESULTS/"stage2-agent-repair-result-r176.json",ROOT/"docs/baseline/stage2-agent-repair-result-r176.json"):
  with path.open("x") as f:json.dump(result,f,indent=2,ensure_ascii=False)
 text=f"""# 真实 Agent 修复批次结果 r176

本次计划50项，独立审计状态：{json.dumps(statuses,ensure_ascii=False)}。
首次通过{totals['first_attempt_passed']}，修复后通过{totals['successful_after_repair']}。
已审计{totals['generations']}次生成、{totals['http_attempts']}次HTTP；调用数不代表账单金额。
真实编译尝试{totals['real_compiled_attempts']}、密态执行尝试{totals['real_executed_attempts']}，含失败与重复。
最终停止原因：{master['failure']}；运行{master['seconds']:.1f}秒。
未审计或中断不计通过；部分批次未审计时上述调用统计只是已核实部分。

使用DeepSeek deepseek-flash/high/stream与已有explicit-v6说明，旧模型/权重/构造要求/编译配置不变。
本轮是重新生成和最多三轮修复，没有发送人工修复DSL或替换模型响应。
34项编译/产物问题继续暂缓；93个编译器源码及二进制不变。
原历史成绩保持原版本，不以择优成绩冒充同一版本全量成功率。
门限保持1e-5+1e-4*abs(reference)，成功项已核验真实tracing/编译/密态/解密与逐项比较。
本轮有界授权已执行；没有自动追加批次、安装、SDK重建、commit、push或PR。

## 未通过项

"""
 for r in rows:
  if r["status"]=="passed":continue
  text+=f"### {r['id']} — {r['status']}\n\n"
  for a in r.get("attempts",[]):text+=f"- 尝试{a['index']}：{a.get('failure_layer')}；{a.get('diagnostic')}。\n"
  text+=r["follow_up"]+"\n\n"
 path=ROOT/"docs/baseline/stage2-agent-repair-result-r176.md"
 with path.open("x") as f:f.write(text)
 h=ROOT/"docs/baseline/mac-migration-handoff.md";old=h.read_text()
 intro=f"\n## r176：真实Agent修复批次已终态\n\n50项状态{json.dumps(statuses,ensure_ascii=False)}；已审计{totals['generations']}生成/{totals['http_attempts']}HTTP。\n报告stage2-agent-repair-result-r176.md/json；停止原因{master['failure']}。\n34编译案例暂缓，旧失败保留，不自动重试。下文r175为启动历史记录。\n\n"
 h.write_text(old.split("\n",1)[0]+"\n"+intro+old.split("\n",1)[1])
 return {k:result[k] for k in ("binding","statuses","totals","terminal_failure_layers","queue_failure")}
def main():
 p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");args=p.parse_args()
 os.chdir(ROOT);print(json.dumps(summarize(args.check),ensure_ascii=False))
if __name__=="__main__":main()
