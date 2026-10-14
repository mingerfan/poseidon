"""Current-source bounded phase summary. Failed depth controls are not passes."""
import json,hashlib,ast,subprocess
from pathlib import Path
from collections import Counter
from benchmark_runner import DEFAULT,load,dump
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
from workspace_paths import ROOT,WORK
base=WORK/"results";files=[]
def read(name):
 p=base/name;files.append(p);return json.loads(p.read_text())
source=digest(runtime_sources());complete=read("packed-prefix-r27-complete.json")
require(complete["status"]=="passed" and complete["source_sha256"]==source,"Incomplete current phase")
unit=read("packed-prefix-r27-unit.json");preflight=read("packed-prefix-r27-all-constructions.json")
require(unit["success"] and unit["passed"]==250 and not unit["errors"] and not unit["failures"],"Unit failure")
require(preflight["counts"]=={"ready":627} and preflight["source_sha256"]==source,"Construction preflight")
native=read("packed-prefix-r27-native/report.json");regression=read("packed-prefix-r27-regression/report.json")
chunk=read("packed-prefix-r27-chunk-regression/report.json");audit=read("packed-prefix-r27-compiler-evidence.json")
for report,count in ((native,20),(regression,2),(chunk,18)):
 require(report["passed"]==count and report["failed"]==0 and report["source_sha256"]==source,"Encrypted regression")
require(audit["audited"]==audit["numerical_passed"]==40 and audit["numerical_failed"]==0,"Artifact evidence")
compat=read("packed-prefix-r27-compatibility.json")
require(compat["old_requests"]==468 and compat["old_helpers_identical"]==47 and compat["new_helpers"]==0,"Compatibility")
depth=read("baseline-boundaries-r27-depth-final/report.json");require(depth["source_sha256"]==source,"Depth source mismatch")
for row in depth["rows"]:
 require(row["status"]=="repair_budget_exhausted" and len(row["attempts"])==1,"Unexpected depth control")
 a=row["attempts"][0];require(a["failure_layer"]=="compiler" and not a["compiled"] and not a["executed"],"Failure layer changed")
rows,index=load(DEFAULT);frozen={r["model"]["id"]:r for r in rows};directed=[]
for row in native["rows"][:3]:
 folder=Path(row["evidence"]);request=json.loads((folder/"request.json").read_text())
 report=json.loads((folder/"report.json").read_text());p=folder/"attempt-00/output/normalized-source.py"
 ops=len(ast.parse(p.read_text()).body[0].body)-1
 require(ops<=256 and report["rule_baseline_lowering"]["strategy"]=="packed-prefix-v1","Actual unchanged-limit fallback")
 model=frozen[request["model"]["id"]];require(digest(request["model"])==model["model_sha256"],"Changed frozen model")
 directed.append(dict(task_id=row["id"],model_id=model["model"]["id"],model_sha256=model["model_sha256"],
                      split=model["split"],emitted_cipher_operations=ops,
                      native_work=report["attempts"][0]["static_check"]["functions"]["golden"]["expanded_cost"],
                      evidence=str(folder),request_id=request["request_id"]))
summary=dict(schema=1,phase="packed_prefix_r27",status="phase_complete_overall_plan_incomplete",runtime_source_sha256=source,
    checkout=str(ROOT),branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),benchmark_suite=str(DEFAULT.relative_to(ROOT)),
    models=1200,topologies=779,construction_tasks=627,helper_tasks=47,rejection_tasks=48,corpus_changed=False,
    current_source_execution_cases=42,encrypted_passed=40,compiler_failed=2,execution_skipped=0,
    main_batch={k:native[k] for k in ("planned","passed","failed","skipped","compared_values","max_absolute_error","weighted_mae","seconds")},
    directed_budget_repairs=directed,depth_controls=depth["rows"],compiler_artifacts_audited=40,
    unit=unit,construction_preflight_counts=preflight["counts"],construction_preflight_is_not_execution=True,
    old_requests_identical=468,old_helper_tasks_identical=47,
    existing_public_cipher_operation_limit=256,existing_native_work_limit=1024,limits_changed=False,
    tolerance="abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)",reference_changed=False,
    actual_compile=True,actual_seal_cpu=True,actual_decrypt=True,paid_api_calls=0,
    installed=False,downloaded=False,commit=False,push=False,pr=False,x86_retested=False,gpu_retested=False,
    previous_plaintext_scope="r26 full 1200-model dual reference remains historical evidence; this phase reran 16 probes for each of its 20 main cases",
    remaining=["two Max mathematical-model compiler failures","three bootstrap helper families","chunk/helper joint layout",
               "other parameter/resource boundaries","w40 precision diagnosis","final-source full corpus and 627 encrypted construction runs",
               "separately approved paid Agent evaluation"],
    evidence={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
target=ROOT/"docs/baseline/packed-prefix-acceptance.json";require(not target.exists(),"Preserve acceptance");dump(target,summary)
doc=f"""# 定向构造预算修复：按需拆包的规则基线 / r27

已确认：三个原来因公共构造操作数超限而阻塞的任务已真实编译、SEAL执行、解密通过：
construct_121_2、construct_122_1、construct_124_0。
实际公共构造操作数分别为241、241、247，均未超过原256门限。
对应native工作量251、251、260，属于另一套原1024门限，不能混淆两个计数。

## 实现与不变项

统一图的可信规则基线先尝试原逐元素源码，成功时逐字保持。
只有公共构造的准确错误 Expanded ciphertext operation budget exceeded 才尝试packed-prefix-v1。
它在形状已验证时直接逐槽计算算术、幂、多项式和不改变C-order的reshape/flatten；
不同布局、跨元素算子或不适合的广播仍按需拆包。公开数组完全相同的系数可精确视为标量，
不用数值近似阈值。输出只按既有声明的逻辑前缀观察。

全部新源码仍经过原AST、类型、shape、操作数、数值和沙箱检查；失败则保留失败。
不按案例ID选择程序，不更改请求、权重、reference、误差门限、密钥或安全参数。
默认lower()与可行旧源码保持；启动器self-test和benchmark预检共用候选生成入口。
这是受限正确性链路的可运行基线修复，不是Agent程序优化实验。
生成请求与独立reference仍不依赖规则答案成功；不改写真实Provider输出。
chunk布局不使用此fallback，继续走原有实现；实际密态回归另行检查。

## 当前证据

最终源码SHA：{source}。
250单测passed、0失败/错误；3个证据路径依赖单项及含5项的历史源码类skip，共8项不计pass。
全部627构造任务现在静态ready；这不是627次密态执行或Agent成功。
主批20次（3个修复任务、13个通用路径fixture、4个helper回归）全部密态通过；
另有2旧入口与18chunk通过。40次成功执行包含重叠任务，不是40个新模型。
40份产物已审计；主批{native['compared_values']}解密值，max absolute error {native['max_absolute_error']:.16g}，
MAE {native['weighted_mae']:.16g}，满足冻结门限。
主批每例另做16组两套独立明文参考比较；没有把上一轮1200模型检查冒充本轮全量重跑。
468旧请求字节一致、47helper任务及模型/627构造/48拒绝账本不变，冻结模型仍为semantic-v1-virtual-r26。
三个修复模型split：{dict(Counter(d['split'] for d in directed))}；人工使用不作为未接触图Agent评分。

同一最终源码还重新执行了2个失败控制：bench_helper_0040、bench_helper_0044。
两者仍在Earth mul的 ci<90 * 12> / pl<45 * 12> 返回类型推断处失败，未密态执行或解密。
因此最终控制集合为42项：40密态passed、2编译failed、0skipped。失败不算pass。
这是所用规则程序/固定编译配置的观察，不证明该数学图的所有合法DSL都无法执行；
也不能把数学模型实验当作真实HE_Max helper或bootstrap已完成。
原初5例诊断、初版均匀常量数组遗漏、测试计数口径修正日志均保留。

## 使用和继续

Ubuntu /home/lhohy/Code/Poseidon，分支lhy-agent-dsl，修改未提交。
已有入口自动选择受限fallback，无新安装、CLI参数或Provider变更。

    python3 scripts/agent.py --backend local candidate -- --case MODEL.json --self-test \\
      --unified-profile public-v1 --unified-exercise EXERCISE_ID \\
      --compiler-configuration seal-cpu-eva-w45-v1

规则基线的选择及原失败原因记入report.json的rule_baseline_lowering。
原始证据为 {base}/packed-prefix-r27-* 与 baseline-boundaries-r27-depth-final。
[机器验收摘要](packed-prefix-acceptance.json)包含逐项记录和报告哈希。

尚未完成：两项编译失败、bootstrap、chunk/helper联合、参数/资源边界、旧w40精度、
最终源码全量密态/627构造与获单独付费批准后的真实Agent评测。
本轮无安装/下载/付费/commit/push/PR；x86/GPU未重测，Mac副本未动。
"""
(ROOT/"docs/baseline/packed-prefix-lowering.md").write_text(doc)
intro=f"""## 最新接续：三个构造预算阻塞修复 r27（2026-09-22）

已确认：仅ARM Ubuntu /home/lhohy/Code/Poseidon、lhy-agent-dsl，HEAD {summary['head']}，未提交。
请求/模型/DSL契约和256公共操作、1024native工作量门限不变。
可信基线只在原公共操作预算失败时尝试通用逐槽前缀计算、按需拆包；不处理Provider输出。
construct_121_2、construct_122_1、construct_124_0真实密态通过，公共操作数241/241/247。
最终源码40密态执行通过、40产物审计通过；另2个Max数学模型编译失败，未执行/解密，不算pass。
250单测通过/0失败；3单项+含5项旧源码类skip。全627构造静态ready，不等于627密态通过。
主批20例各16组双参考通过；468旧请求和47helper任务保持。
模型1200/779拓扑/627构造/47helper/48拒绝不变；数据freeze仍semantic-v1-virtual-r26。

源码SHA {source}；说明packed-prefix-lowering.md，摘要packed-prefix-acceptance.json。
证据results/packed-prefix-r27-*、baseline-boundaries-r27-depth-final；所有批次结束。
无安装/下载/收费/commit/push/PR，Mac副本未动，x86/GPU未重测。
基于证据：原3定向阻塞已解除；不外推全部图/参数，不称Agent生成成功。
未完成：2编译失败、3类bootstrap helper、chunk/helper联合、其他参数/资源边界、旧w40精度、
最终源码全量密态/627构造，以及另行获批的真实Agent。
下一行动：先评估两项深多项式的合法表达及编译容量边界，或扩展chunk/helper联合；
最终全量运行前以48例代表集重新确认时间/内存/空间预算，所有已有失败证据保留。
下一只读命令：POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/benchmark.py plan。

"""
p=ROOT/"docs/baseline/mac-migration-handoff.md";heading,rest=p.read_text().split("\n",1);p.write_text(heading+"\n\n"+intro+rest.lstrip())
p=ROOT/"docs/baseline/semantic-benchmark-v1.md";heading,rest=p.read_text().split("\n",1)
p.write_text(heading+"\n\n当前r27：[构造预算修复](packed-prefix-lowering.md)解除三个定向任务阻塞；627任务静态ready。\n40次当前源码密态通过，另2项编译失败单列；不是全量密态或Agent成绩。模型/契约/预算不变。\n\n"+rest.lstrip().replace("当前 r26：","历史 r26：",1))
print(json.dumps(dict(status=summary["status"],encrypted_passed=40,compiler_failed=2,construction_ready=627,source_sha256=source)))
