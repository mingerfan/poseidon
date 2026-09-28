"""Bind current offline repairs to original failures; never recompute Agent scores."""
import json,hashlib,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from workspace_paths import RESULTS
from benchmark_graph import digest
from semantic_benchmark_execution import runtime_sources
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=RESULTS/"stage2-remaining-r174";parents={}
 def read(p):
  parents[str(p)]=sha(p);return json.loads(p.read_text())
 execution=read(out/"execution-v2/report.json")
 assert execution["completed"] and len(execution["rows"])==50
 before=read(out/"before.json");compat=read(out/"compatibility.json");unit=read(out/"unit.json")
 probe=read(out/"compiler-probe-packed-v2/report.json")
 initial_probe=read(out/"compiler-probe/report.json")
 old=read(ROOT/"docs/baseline/stage2-agent-repair-result-r173.json")
 assert runtime_sources()==execution["source_hashes"]==compat["source_hashes"]==probe["source_hashes"]
 assert all(sha(p)==h for p,h in before["compiler"].items())
 assert all(sha(p)==h for p,h in before["parents"].items())
 oldrows={r["id"]:r for r in old["rows"]};rows=[]
 values=0;groups=0
 for entry in execution["rows"]:
  row=oldrows[entry["id"]];folder=out/"execution-v2"/entry["id"]
  report=read(Path(entry["evidence"])/"report.json")
  assert parents[str(Path(entry["evidence"])/"report.json")]==entry["report_sha256"]
  assert report["agent_calls"]==0 and report["status"]==entry["status"]
  if entry["status"]=="passed":
   audit=read(folder/"audit.json");assert parents[str(folder/"audit.json")]==entry["audit_sha256"]
   values+=audit["comparison"]["compared_values"]
   assert len(audit["comparison"]["actual"])==4
   groups+=len(audit["comparison"]["actual"])
  rows.append(dict(entry,original_diagnostic=row["attempts"][-1]["diagnostic"],
   original_request_id=row["request_id"],original_agent_status="failed"))
 compiler=read(ROOT/"docs/baseline/compiler-blocked-models-r159/index.json")
 compiler_rows=[]
 for r in compiler["rows"]:
  assert sha(ROOT/r["file"])==r["file_sha256"]
  compiler_rows.append(dict(id=r["task_id"],model_id=r["model_id"],model_file=r["file"],model_sha256=r["file_sha256"],
   status="historical_compiler_or_artifact_failure_not_retested",next_action="Compiler capacity/result-scale analysis; compiler changes remain outside authorization."))
 added=oldrows["free_bench_helper_0104"];q=read(Path(added["evidence"])/"request.json")
 modelpath=ROOT/"docs/baseline/compiler-blocked-models-r174/bench_helper_0104.json"
 modelpath.parent.mkdir(exist_ok=True)
 with modelpath.open("x") as f:json.dump(q["model"],f,indent=2)
 logpath=Path(added["evidence"])/"attempt-03/compile.log";parents[str(logpath)]=sha(logpath)
 compiler_rows.append(dict(id=added["id"],model_id=q["model"]["id"],model_file=str(modelpath.relative_to(ROOT)),
  model_sha256=sha(modelpath),status="compiler_capacity_failure",diagnostic=logpath.read_text(),
  capacity_check=dict(consumed_level=12,lhs_log2_scale=90,rescale_factor=60,required=810,limit=780),
  scalar_balanced_probe=initial_probe.get("diagnostic"),exact_model_balanced_probe=probe.get("result",dict(status=probe.get("status"),diagnostic=probe.get("diagnostic")))))
 passed=sum(r["status"]=="passed" for r in rows)
 report=dict(format="poseidon-remaining-repairs-r174",scope=84,original_agent_unresolved=84,
  offline_repair_passed=passed,offline_repair_failed=50-passed,compiler_cases=34,
  new_agent_successes=0,paid_calls=0,reference_input_groups=groups,compared_values=values,
  max_absolute_error=max((r.get("max_absolute_error",0) for r in rows),default=0),
  unit_tests=unit,legacy_requests_exact=compat["old_campaign_exact"],r172_requests_exact=compat["r172_exact"],
  source_hashes=runtime_sources(),compiler_files_unchanged=len(before["compiler"]),
  original_responses_unchanged=True,rows=rows,compiler_rows=compiler_rows,parents=parents,
  limitations=["Deterministic correction witnesses are not Agent successes.",
   "Finite bounded interventions and four input groups are not formal equivalence.",
   "explicit-v6 effectiveness on new Agent generations is untested.",
   "No compiler modification, lowered security, changed models/reference/tolerance or bootstrap."],
  initial_integration_failure="v6 binding list order depended on JSON key order; initial trace rejects preserved in execution; fixed sorting plus roundtrip regression; execution-v2 uses corrected source binding.",
  seconds=execution["seconds"])
 report["binding"]=digest(report)
 dest=ROOT/"docs/baseline/stage2-remaining-repairs-r174.json"
 with dest.open("x") as f:json.dump(report,f,indent=2)
 (out/"final.json").write_text(json.dumps(report,indent=2))
 text=f"""# 剩余失败修复与离线验收 r174

当前84项中，50项类型/定向构造失败已有规则修复样例，离线密态验收 {passed}/50。
这些是确定性转换器和可信构造配方生成的见证，不是新Agent生成；原Agent仍有84项未恢复成绩。
另外34项是编译容量或产物level/scale问题，保持失败/阻塞状态。

## 已确认事实

- 新增可选explicit-v6：逐请求公开常量名称/shape、准确返回数量、允许的旋转步长；
  明确dtype=float、未绑定public_constants、公开数值与Expr单元的区别。
- 针对values、公开数值/序列计数器、continue、fresh unary、public_cells、
  多维item、overlap和*=补充真实操作及返回依赖要求。未放宽AST、类型或贡献检查。
- v1–v5不变；2030原请求及80份r172请求重新构造完全一致。
- 修复样例保留全部原模型、公开常量、布局、构造要求和编译配置，只使用新说明请求ID。
- {passed}项通过真实Hecate tracing、Earth/CKKS、HEVM/CST、SEAL CPU执行、解密及独立审计。
  共{groups}组基础输入、{values}输出值；最大绝对误差{report['max_absolute_error']:.12g}。
  门限仍为abs(actual-reference)<=1e-5+1e-4*abs(reference)。
- 单元/兼容测试{unit['run']-unit['skipped']-unit['failures']-unit['errors']} passed /
  {unit['failures']} failed / {unit['errors']} errors / {unit['skipped']} skipped。
- 93个编译器/运行时源码及二进制哈希保持不变；旧响应未改。
- 新API调用0；无安装、下载、commit、push或PR。ARM Ubuntu lhy-agent-dsl开发，Mac历史checkout未改。

## 编译相关34项

保留r159的33项模型及全部原证据，不重复盲跑。
新增free_bench_helper_0104为输入[2]的ReLU对应多项式：
x*(C27(C15b(C15a(x)))+0.5)，系数完全保持冻结定义。
原失败Earth Mul形状/level一致，但12*60+90=810超过13*60=780。
不能把它解释为两边scale必须相同，也不能据此证明所有等价计算树均不可行。
本轮原模型scalar balanced树首先被AST预算拒绝；改用同布局逐槽并行的balanced树后通过真实tracing，\n仍在编译时得到90*12的左操作数并因810>780拒绝；未进入密态执行。结果在JSON的compiler_rows最后一项；
不删微小系数、不改参数、不以解密重加密替代bootstrap。

33旧模型见compiler-blocked-models-r159/README.md及index.json；
新增模型见compiler-blocked-models-r174/bench_helper_0104.json。
方向仍是计算树可行性和编译器结果scale后置条件分析；按用户要求本轮不修改编译器。

## 推断与尚未验证

50个修复见证说明原任务在现有后端有可执行写法；并不能证明任何原响应的每个拒绝都必然正确，
也不证明新版说明一定提高Agent成功率。Agent需要重新生成并通过同样门禁才能记成功。
34项没有恢复；真实bootstrap、完整Python和Poseidon GPU不在本次范围。

## 实施中保留的失败

初版v6在JSON排序后常量绑定列表顺序不同，沙箱完整性门禁拒绝。
已停止初批，保留execution目录；以排序稳定的元数据和新增往返测试修复后，用execution-v2重新验收。
另一次并行启动离线审计被现有Nix锁拒绝，未调用API，改为串行执行。
初始单元测试曾有注册特征名与测试模块名错误，修正后以当前unit.json为最终结果。\npacked诊断脚本初版换行转义错误，语法拒绝记录保留；修正脚本后真实tracing成功、编译容量仍失败。

## 用法与证据

在现有candidate入口追加 --unified-guidance explicit-v6；默认行为及旧版本保持不变。
组件prepare的options.generation_guidance也接受explicit-v6。
结果根目录：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/stage2-remaining-r174。
prepared.json保存50份原请求及规则样例；execution-v2保存逐项job、日志和独立审计，
完整真实执行证据路径在本报告JSON各row.evidence中。
compatibility.json记录旧请求重构；compiler-probe保留等价树探针。
本轮工具：benchmarks/tools/repair_remaining_r174.py、audit_remaining_r174.py、
probe_remaining_compiler_r174.py、probe_remaining_packed_r174.py、report_remaining_r174.py。固定验收目录拒绝覆盖；不要重复启动旧批次。
下一步是单独批准50项真实Agent有界重测；不发送规则样例或标准DSL。
"""
 (ROOT/"docs/baseline/stage2-remaining-repairs-r174.md").write_text(text)
 print(json.dumps({k:report[k] for k in ("binding","offline_repair_passed","offline_repair_failed","compiler_cases","compared_values","max_absolute_error")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
