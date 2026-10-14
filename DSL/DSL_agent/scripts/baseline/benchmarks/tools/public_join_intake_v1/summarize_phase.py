"""Produce phase evidence after both frozen qualification tracks close."""
import json,sys,hashlib,subprocess,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE),str(HERE)]
import run as live
import qualification_scope as s
live.no_active()
capture=s.read(HERE/"capture-audit.json");s.bound(capture)
assert capture["status"]=="passed" and capture["uploaded_code_executed_by_audit"]is False
tracks={}
for mode in ("offline","live"):
 p=s.verify(mode);out=Path(p["campaign"]);r=s.read(out/"report.json");s.bound(r)
 a=s.read(out/"audit.json");s.bound(a);n=s.read(out/"numerical-audit.json");s.bound(n)
 terminal=s.read(out/"terminal.json");owner=s.read(out/"owner.json")
 info=live.process_info(owner["pid"])
 assert info is None or str(info["start"])!=owner["start"]
 assert terminal["exit_code"]==0 and terminal["plan_binding"]==p["binding"]
 assert r["status"]=="complete" and r["passed"]==len(p["cases"])
 assert r["failed"]==r["skipped"]==r["not_run"]==0 and r["calls_resolved"]
 assert a["status"]==n["status"]=="passed" and a["report_binding"]==n["report_binding"]==r["binding"]
 assert a["plan_binding"]==n["plan_binding"]==p["binding"]
 assert n["qualification_audit_binding"]==a["binding"] and n["sdk"]==a["sdk"]==r["sdk"]==p["sdk"]
 assert all(n[k]==0 for k in ("new_compiler_calls","new_encrypted_calls","new_provider_calls"))
 assert all(row["package_reloaded"] and row["package_reused"] for row in r["cases"])
 tracks[mode]=dict(status="passed",campaign=str(out),plan_binding=p["binding"],report_binding=r["binding"],
  evidence_audit_binding=a["binding"],numerical_audit_binding=n["binding"],cases=r["passed"],
  failed=0,skipped=0,not_run=0,origin=a["origin"],input_groups=n["input_groups"],
  output_values=n["output_values"],maximum_absolute_error=n["maximum_absolute_error"],
  compiler_calls_confirmed=r["compiler_calls_confirmed"],encrypted_calls_confirmed=r["encrypted_calls_confirmed"],
  generations=r["generations"],http_attempts=r["http_attempts"],calls_resolved=True,
  first_attempt_passed=sum(row["result"]["generations"]==1 for row in r["cases"]) if mode=="live" else None,
  repaired_cases=[row["case"]for row in r["cases"]if row["result"]["generations"]>1] if mode=="live" else [],
  package_reload_and_cache_cases=len(r["cases"]),new_input_and_repeat_cases=sorted(s.REUSE_CASES),
  owner_exited=True,actual_bytes=r["bytes"],seconds=r["seconds"],
  case_results=[{k:row[k]for k in ("case","status","request_id","package_binding")}for row in r["cases"]])
sdk=s.read(HERE/"identity-snapshot.json");before=s.read(HERE/"prechange-identity.json")
assert sdk["backend"]==before["backend"]
allowed=set(("pytorch_public_join_source.py","pytorch_public_join_values.py","pytorch_public_join_normalization.py",
 "pytorch_ordered_contract.py","pytorch_model_intake.py","candidate_bundle.py","pytorch_ordered_intake.py",
 "pytorch_source_package_policy.py","pytorch_spatial_policy.py","pytorch_static_broadcast.py",
 "pytorch_static_tensor_transforms.py","pytorch_public_value_ops.py","pytorch_public_expression_metadata.py",
 "pytorch_public_expression_flow.py","pytorch_static_trace.py","pytorch_public_expression_capture.py","application_capabilities.py"))
for stage in ("generation","compilation","execution","validation"):
 a,b=before[stage]["sources"],sdk[stage]["sources"]
 changed={n for n in set(a)|set(b)if a.get(n)!=b.get(n)}
 assert changed<=allowed,(stage,sorted(changed-allowed))
assert tracks["offline"]["http_attempts"]==0
controls=s.read(HERE/"controls.json");guards=s.read(HERE/"controls-scope.json")
assert controls["status"]==guards["status"]=="passed"
assert all(d[k]==0 for d in (controls,guards)for k in ("failures","errors","skipped","api_calls","compiler_calls","encrypted_calls"))
data=dict(format="poseidon-public-join-phase-acceptance-v1",status="complete",checkout=str(ROOT),
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
 platform="aarch64-linux",source_format="poseidon-trusted-pytorch-v34",default_source="v18",latest_source="v28",
 generation_guidance="explicit-v14",sdk=tracks["offline"].get("sdk",s.read(HERE/"offline-plan.json")["sdk"]),
 source_capture=capture,controls=controls,execution_guards=guards,offline=tracks["offline"],live=tracks["live"],
 numerical_tolerance=dict(atol=1e-5,rtol=1e-4),plaintext_tolerance=dict(atol=1e-12,rtol=1e-12),
 new_contexts=18,compatibility_contexts=3,new_benchmark_topologies_counted=0,
 normalization=dict(public_only="exact data-only fixed Torch signatures",
  mixed="public values materialized using encrypted zero from a checked input anchor",
  zero_before_reduce=True,extra_cipher_operations_bound=True,source_and_effective_state_bound=True),
 scope=["public-only cat/concat/concatenate/stack","mixed public/encrypted joins","static axis and dtype promotion",
  "mask, tensor transform, split, Linear, residual and structured output combinations"],
 source_modules=["pytorch_public_join_source.py","pytorch_public_join_values.py","pytorch_public_join_normalization.py"],
 history=dict(prechange_rejected_probes=str(HERE/"probe-before.json"),qualified_source_probes=str(HERE/"probe-after.json"),
  initial_index_tool_error=str(HERE/"index-audit-api-correction.json"),
  initial_prepare_tool_error=str(HERE/"prepare-initial-wiring.json"),historical_records_rewritten=False),
 confirmed=["isolated source intake and independent reference checks","compiled RuntimePlan/OperatorSpec/bundle checked",
  "real Poseidon CPU encrypted execution and numerical comparison","current provider source preserved byte for byte",
  "package reload and zero-generation/zero-compile cache reuse","three packages with new input and repeat execution"],
 evidence_based_inference=["registered static public/mixed joins can use existing backend interfaces"],
 unverified=["arbitrary Python and dynamic shapes","all combinations or formal equivalence","maximum rank and physical capacity",
  "source function application accuracy","CPU bootstrap integration","CUDA hardware execution","complete ResNet20"],
 next_work=["common source configuration consolidation","larger nonrepeating weights and physical layouts",
  "registered generic operator and model-family combinations"],
 resources=dict(global_limit_bytes=48*1024**3,per_case_and_batch_bytes=4*1024**3,reserve_bytes=4*1024**3,
  actual_global_bytes=live.measure(s.RESULTS),disk_free_bytes=shutil.disk_usage(s.RESULTS).free,
  encrypted_concurrency=1,api_concurrency=1,compiler_jobs=2,link_jobs=1),
 paid_calls=tracks["live"]["http_attempts"],all_http_resolved=True,source_freeze_released=True,
 backend_core_changed=False,backend_identity_unchanged=True,agent_stage_identities_rebound=True,parameters_changed=False,network_changed=False,dependencies_changed=False,
 installations=0,downloads=0,git_actions=[])
assert data["branch"]=="lhy-agent-dsl" and data["resources"]["actual_global_bytes"]<=data["resources"]["global_limit_bytes"]
assert data["resources"]["disk_free_bytes"]>=data["resources"]["reserve_bytes"]
data["binding"]=s.digest(data)
out=ROOT/"docs/baseline/agent-public-join-acceptance-r1480.json"
assert not out.exists();s.write(out,data)
o,l=tracks["offline"],tracks["live"]
md="""# PyTorch v34 拼接接入验收

已接入固定公开量与加密张量的 cat / concat / concatenate / stack，沿用 Poseidon CPU 后端。
来源格式使用 poseidon-trusted-pytorch-v34，生成指导 explicit-v14；默认 v18、latest v28。

| 验收 | 结果 | 实际调用 | 数值范围 |
|---|---|---|---|
| 隔离来源与独立参考 | 21/21 | 编译、密态、API 0 | 336 组 / 2,448 值 |
| 规则候选 | {op}/21 | {oc} 编译 / {oe} 密态，API 0 | {og} 组 / {ov} 值，最大绝对误差 {om:.9g} |
| 当前真实 Agent | {lp}/18 | {lg} 生成 / {lh} HTTP / {lc} 编译 / {le} 密态 | {lig} 组 / {lv} 值，最大绝对误差 {lm:.9g} |

真实 Agent 首次通过 {first} 项，经过自动修复的案例：{repairs}。两轨失败、跳过和未运行均为 0。
全部程序包完成重新加载和缓存复用；mixed_cat_2、mixed_stack_2、combination_5 另完成三组新输入及重复执行。
候选逐字来自 Provider 原始响应。规则程序与真实生成分别计分。

## 已确认

- 拼接序列为 1–8 个静态 Tensor；cat 要求非拼接轴一致，stack 要求相同 shape。
- 覆盖公开 dtype 提升、标量 stack、负轴，以及 mask、拆分、公开变换、Linear、残差和结构化输出组合。
- 公开/加密混合拼接通过 Agent 层显式规范化复用现有计算接口，额外密文运算和状态身份参与绑定。
- 验收包含 tracing、Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、真实密态执行和逐项比较。
- 误差门限保持 1e-5 + 1e-4*abs(reference)，两轨独立数值审计通过，HTTP 全部结算，实际 owner 退出。
- 历史五项输入拒绝和开发工具索引/准备错误记录保留；模型、reference、后端及网络保持。

## 基于证据的推断

本批注册的静态拼接能够通过既有后端接口执行，现阶段无需修改 Poseidon、Runtime 或 Dacapo。

## 尚未验证

任意 Python、动态 shape、最大 rank/物理容量、原模型应用精度、真实 bootstrap 接线、CUDA 实机和完整 ResNet20。
18 个新来源上下文与 3 个旧兼容上下文不增加冻结 Benchmark 的 1,200 模型结构计数。

## 后续

继续整合常用来源配置、较大权重和物理布局，再按通用算子清单验收未见组合。
测试沿用用户总授权；全局 48 GiB、每例及每批 4 GiB、磁盘余量 4 GiB，API 与密态串行、编译 2 / 链接 1。

SDK：{sdk}。阶段报告绑定：{binding}。
[机器可读结果](agent-public-join-acceptance-r1480.json)。
""".format(op=o["cases"],oc=o["compiler_calls_confirmed"],oe=o["encrypted_calls_confirmed"],
 og=o["input_groups"],ov=o["output_values"],om=o["maximum_absolute_error"],
 lp=l["cases"],lg=l["generations"],lh=l["http_attempts"],lc=l["compiler_calls_confirmed"],le=l["encrypted_calls_confirmed"],
 lig=l["input_groups"],lv=l["output_values"],lm=l["maximum_absolute_error"],first=l["first_attempt_passed"],
 repairs=", ".join(l["repaired_cases"])or"无",sdk=data["sdk"],binding=data["binding"])
mp=out.with_suffix(".md");assert not mp.exists();mp.write_text(md)
print(json.dumps({k:data[k]for k in ("status","sdk","paid_calls","binding")}))
