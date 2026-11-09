# E4：现有编译器端到端优化的编译负担（2026-11-05，用户"写 spec"）

目的：给论文 §2 观察 3 补数据，说明现有编译器做端到端优化（DaCapo 的 optimized 自举放置）很慢，规模一大就编不出来。
产出一张"程序规模（Earth op 数）对编译时间"的表和散点数据，论文图由 Claude 另画。

本轮只写驱动脚本，放在结果目录；不改 pt_agent 库、DaCapo 源码、编译器、提示词或判据。API 0。Nix 独占，全部串行。
本轮按用户 2026-11-05 的要求，在可信运行器的同一组限制下（6 GiB、单文件 8 GiB、单进程 3600 s、磁盘剩余 ≥10 GiB）
运行上游 DaCapo 自带的 Python 基准脚本。它们是 DaCapo 作者写的、随仓库提供的第三方源码，本轮没有任何 LLM 生成的代码被执行。

结果目录：`/home/lhohy/pt_agent_results/E4-compile-burden-20261105/`（日期按实际开跑日）。里面放 `run.sh`、`driver.py`、
`launcher.py`、`collect.py` 和各模型子目录。

## 0　口径与已有证据（本轮不重跑）

- 编译器：默认 R121 构建 `toolchain.host_paths()['hecate_opt']`（`$PT_HECATE_BUILD`，标记文件 `pt-boot-window.json` 的 rule 为 `r121`）。
  它是 DaCapo 39c1fb2 加已批准的 13 文件移植补丁，再加 R121 的 GreedyBootstrapPlacement 改动。`--boot-placement` 默认值是
  `optimized`（构建源码 `src/tools/optimizer.cpp:221-228`）。本机没有未改动的上游构建；另建一份需要单独审批，本轮不做。
- 已有对照（collect 时直接读取，不重跑）：
  - M7：A1 冻结的 BERT、Qwen Agent 程序换用 optimized，3600 s 内未完成；depth-dp 下 10.0 s／15.2 s。
    `A1-optimized-20261102/{bert,qwen}/Agent/REPORT.json`。
  - M7b：R8e7 的 mlp、cnn_plain、resnet20_silu、resnet20 换用 optimized，全部在规划器内段错误；depth-dp 下 0.5–7.7 s。
    `A1-optimized-20261102/cnn/<model>/REPORT.json`。
  - M2（团队，188Server，手写 Qwen2.5-0.5B DSL，合成权重）：数值见 §3.3，来源见 `paper-evidence-20261102/EVIDENCE.md` 的 M2 节。

## 1　A 部分：DaCapo 自带的 6 个专家程序

对象：`$PT_DACAPO_UPSTREAM/examples/benchmarks/` 下的 `MLP.py`、`ResNet.py`、`MobileNet.py`、`SqueezeNet.py`、`AlexNet.py`、
`VGG16.py`，原样使用（不复制、不修改）。权重是脚本默认读取的 `examples/data/` 文件（mlp.model、resnet20.silu.model、
mobileNet_silu_model、squeezeNet_silu_avgpool_model、alexNet_silu_avgpool_model、vgg16_silu_avgpool_model），都已在仓库里。
`$PT_DACAPO_ROOT/examples/benchmarks/` 下团队改过的 MLP.py、ResNet.py 本轮不用。

### 1.1　trace（每个模型一次 trusted_run）

顺序：MLP、ResNet、MobileNet、SqueezeNet、AlexNet、VGG16。

外层 `driver.py run-trace <B>` 用 `pt_agent.tools.nixrun --seconds 3900` 进入 Nix，内层调用：

```python
from pt_agent import toolchain
toolchain.setup_env()
from hecate_python_env import VENV
from pt_agent.paths import DACAPO_ROOT, DACAPO_UPSTREAM
env = toolchain.package_trace_env()
env['PYTHONPATH'] = os.pathsep.join([str(SCRIPTS), str(DACAPO_ROOT / 'python/hecate'), str(DACAPO_UPSTREAM / 'python/poly')])
work = OUT / B                       # 先建 work/'traced'
script = DACAPO_UPSTREAM / f'examples/benchmarks/{B}.py'
toolchain.trusted_run(work, [str(VENV / 'bin/python'), '-B', str(OUT / 'launcher.py'), str(script)],
                      work / 'trace.log', env_extra=env)
```

`launcher.py`（结果目录内，约 10 行）：把 `sys.argv` 设为 `[script]`，然后

```python
from pt_agent.models.dacapo_builders import _torchvision_imports
with _torchvision_imports():
    runpy.run_path(script, run_name='__main__')
```

这样写的原因：
- 上游 `python/hecate` 从 `<dacapo>/build` 加载库，上游目录没有 build。`$PT_DACAPO_ROOT/python/hecate` 读 `HECATE_BUILD`
  环境变量，接口与上游相同（多了 `save(..., bytecode=False)` 和取调用帧的方式）。
- 上游 `python/poly` 与 `$PT_DACAPO_ROOT/python/poly` 只差 `Func.py` 里 `HE_Linear` 的可选参数 `nt`，默认值同为 `2**16`。
- 本机 venv 没有 torchvision。脚本和 `poly.models.*` 只在文件头导入 `datasets, transforms`，沿用 U3 的桩
  `_torchvision_imports()`；一旦真的用到，桩会抛出 `TORCHVISION_ERROR`。
- 脚本末尾的 `hc.save("traced", "traced")` 写到工作目录（`work`）下的 `traced/<B>.mlir`；权重按脚本自身路径从上游 `examples/data/` 读取。

记录到 `<B>/TRACE.json`：exit、timeout、seconds、peak_rss_mib、`traced/` 下各文件字节数、§1.4 的计数（Earth op 数，以及脚本里手写的
`earth.bootstrap` 数，`--dacapo` 第一步会删除它们）。

失败处理：
- 单个模型 trace 失败：记录 log_tail，继续下一个模型。
- MLP 和 ResNet 都在 trace 失败，且原因是导入、前端接口或 `TORCHVISION_ERROR`：说明环境问题是系统性的，停止全部运行，写进 STATUS"阻塞"。
- 日志含 `MemoryError`、`std::bad_alloc` 或 `out of memory`：记为"trace 超过 6 GiB"，按 §1.5 导出服务器作业，继续下一个模型。

### 1.2　编译（每个模型两次：先 depth-dp，再 optimized）

已完成 trace 的模型按 Earth op 数从小到大排序后依次编译。每次编译前先建好 `work/P/` 目录（trusted_run 在该目录上检查磁盘）。命令沿用 DaCapo README 的 `hopts dacapo 40 <B> HEAAN CPU`，只加 `--mlir-timing`：

```python
argv = [str(toolchain.host_paths()['hecate_opt']), '--dacapo', f'--boot-placement={P}',
        f'--ckks-config={DACAPO_UPSTREAM}/profiled_HEAAN_CPU.json', '--waterline=40', '--mlir-timing',
        str(work / f'traced/{B}.mlir'), '-o', str(work / P / f'{B}.40.mlir')]
toolchain.trusted_run(work / P, argv, work / P / 'compile.log')       # P in ('depth-dp', 'optimized')
```

- 这条命令与 `toolchain.hecate_command` 不同，没有 `--verify-each`、`--mlir-disable-threading` 和 RuntimePlan 参数，是 DaCapo 自己的用法。
- `profiled_HEAAN_CPU.json` 的 polynomialDegree 为 131072，与脚本里的 `nt = 2**16` 一致；levelUpperBound 29、rescalingFactor 51。
- 后台每 10 s 采样一次命令行含 `work/P/` 的 hecate-opt 进程（pid、etimes、rss、pcpu），写入 `work/P-rss-samples.txt`，
  写法同 `A1-optimized-20261102/run-cnn.sh` 的 `sample`。超时时 `/usr/bin/time` 不留记录，峰值 RSS 取采样最大值，并标为下限。
- **残留进程**：`trusted_run` 超时只终止 `/usr/bin/time`，hecate-opt 会继续运行。每次 trusted_run 返回后，用
  `A1-optimized-20261102/cnn_driver.py` 的 `leftovers(out)` 找出命令行含 `work/P` 的进程，SIGKILL 后等待退出，
  记录 `leftover_processes_killed`。

记录到 `<B>/<P>/COMPILE.json`：
- 结果分类：
  - complete：exit 0；
  - timeout：`timeout` 为真；
  - out_of_memory：日志尾含 `bad_alloc` 或 `out of memory`；
  - crash：其余非零退出，记下退出码。
- 运行数据：seconds、peak_rss_mib（是否为下限）、输出字节数、§1.4 计数的自举数。
- `--mlir-timing` 报告中墙钟时间最长的 10 个 pass（名称和秒数，从日志解析；超时或崩溃时没有报告，记 null；有报告但解析不出时，保存报告原文的前 40 行）。
- log_tail。

DaCapo README 说明，编译过程中出现 `'earth.bootstrap' op failed to infer returned types` 属于正常现象。结果分类只看退出码、timeout 和上面列出的内存字样，不看这类报错。

跳过规则（开跑前定好，中途不改）：
- 第一次 optimized 超时之后，比它大的模型不再跑 optimized，记为"skipped：<模型> 的 optimized 已超时"。
- 崩溃和内存超限不触发跳过。
- depth-dp 对所有已完成 trace 的模型都要跑。

### 1.3　合理性检查

- 已完成编译的 ResNet、MobileNet、SqueezeNet、AlexNet、VGG16 输出里，自举数至少为 1；MLP 不检查。
- 若某个 traced 或输出文件按 §1.4 数出 0 个 Earth op，说明文件格式与预期不同：把前 40 行（每行截到 200 字符）写入
  `<B>/format-sample.txt`，停止并写进 STATUS"阻塞"。

### 1.4　计数规则（driver.py 与 collect.py 共用一个函数）

```python
OP = re.compile(r'^\s*(?:%[^=]*=\s*)?"?(earth|ckks)\.([a-z_0-9]+)"?[\s(<{]')
def count_ops(path):        # 逐行流式读取，不整读进内存
    counts = Counter()
    for line in open(path, errors='replace'):
        m = OP.match(line)
        if m: counts[m.group(1) + '.' + m.group(2)] += 1
    return dict(earth_ops=sum(v for k, v in counts.items() if k.startswith('earth.')),
                boots=counts.get('earth.bootstrap', 0) + counts.get('ckks.bootstrap', 0), by_op=dict(counts))
```

开跑前对下面 4 行做自检，结果不符就停：

| 行 | 期望 |
|---|---|
| `    %0 = "earth.add"(%a, %b) : (tensor<1x!earth.ci<90 * 29>>, tensor<1x!earth.ci<90 * 29>>) -> tensor<1x!earth.ci<90 * 29>>` | earth.add |
| `    %1 = earth.bootstrap %0 {targetLevel = 3 : i64} : tensor<1x!earth.ci<43 * 0>>` | earth.bootstrap |
| `  func.func @ResNet(%arg0: tensor<1x!earth.ci<90 * 29>>) -> tensor<1x!earth.ci<90 * 29>> {` | 无 |
| `    return %3 : tensor<1x!earth.ci<43 * 0>>` | 无 |

### 1.5　服务器作业（只导出，不运行）

trace 或编译因内存超限没有完成的模型，写 `server_jobs/E4-<B>/run.sh`：
- 开头放 `toolchain.server_env_exports()` 的输出；
- trace 和两次编译用与 §1.1、§1.2 相同的命令；
- 每步用 `timeout 14400` 和 `/usr/bin/time -v` 包起来；
- 结尾按 §1.4 计数。

### 1.6　清理（本机磁盘剩余约 14 GB，trusted_run 要求 ≥10 GiB）

- 每个模型两次编译都结束后，删除该模型 `traced/` 和两个输出中超过 50 MiB 的文件，计数和字节数保留在 JSON 里。
  删除记录追加到该模型目录的 `PRUNED.txt`。
- trusted_run 因磁盘剩余不足报错时，停止全部运行并写进 STATUS"阻塞"。

## 2　B 部分：用同一计数口径量出已有 PT-DSL 程序的规模

六个程序的包已在 `A1-optimized-20261102` 中（只 trace，不再编译；编译数据取自 M7/M7b）：

| 模型 | 包 | 权重 |
|---|---|---|
| bert | `bert/Agent/Bert_PTAgent_DSL` | `bert/Agent/REPORT.json` 的 `weights` |
| qwen | `qwen/Agent/Qwen2_PTAgent_DSL` | `qwen/Agent/REPORT.json` 的 `weights` |
| mlp、cnn_plain、resnet20_silu、resnet20 | `cnn/<m>/optimized/<Pkg>_PTAgent_DSL`（即该 REPORT 的 `package`） | 该 REPORT 的 `depth_dp.export.weights` |

步骤（每个程序一次 nixrun）：
1. 把包复制到 `E4-.../ptdsl/<model>/`，只读使用原包。
2. 用 `A1-optimized-20261102/driver.py` 里 `entry('trace', out/'trace')` 的同一调用（`u4_native_continuation --entry trace`，
   `env_extra=trace_environment()`）得到 `trace/trace_model.mlirbc`。
3. 不跑任何 pass，打印成文本：`[hecate_opt, <bc>, f"--ckks-config={toolchain.host_paths()['profile']}", '--mlir-print-op-generic',
   '--mlir-elide-elementsattrs-if-larger=1', '-o', <dir>/trace_model.generic.mlir]`，同样经 trusted_run。
4. 按 §1.4 计数，然后删除 bytecode 和文本两个文件（记入 `PRUNED.txt`）。
5. 结果写 `ptdsl/<model>/SIZE.json`（trace 秒数、RSS、bytecode 字节数、earth_ops）。任一步失败：记录，继续下一个程序。

## 3　C 部分：汇总（collect.py，系统 python3，不进 Nix）

### 3.1　E4.json、E4.md、POINTS.csv

每行一个（程序，放置）组合，字段：
- source：`[本项目 E4]`、`[本项目 M7]`、`[本项目 M7b]`、`[团队 M2]`、`[论文 DaCapo]`
- program、author（`DaCapo expert DSL`、`PT-Agent`、`team hand-written`）
- earth_ops、placement、outcome、compile_s、peak_rss_mib、rss_is_lower_bound、boots、trace_s、notes

数据来源：
- M7 和 M7b 的 compile_s、RSS、outcome、boots 从它们的 REPORT.json 读取（depth-dp 一侧用 REPORT 内记录的 depth-dp 数据）；
  earth_ops 取 B 部分结果。
- POINTS.csv 只含有 earth_ops 的行，供作图。

### 3.2　编译器来源对照

在 E4.md 里列出 `diff -rq` 的结果：
- 比较 `$PT_HECATE_BUILD/../src/lib` 与 `$PT_DACAPO_UPSTREAM/lib`，以及两边的 `tools/`；
- 逐个说明 optimized 放置用到的文件（BypassDetection、CandidateAnalysis、CandidateSelection、DaCapoPlanner、CoverageRecorder
  所在源文件）是否与上游相同。

### 3.3　M2 行（直接写入，来源列注明 EVIDENCE.md M2）

| 程序 | earth_ops | 放置 | outcome | 墙钟 s | 峰值 RSS GiB | 自举 | 备注 |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B 1 block，2 token prefill | 164,731 | optimized | timeout（600 CPU-s，SIGXCPU） | 27.0 | 6.9 | — | 修正前 trace |
| 同上，重试 | 164,731 | optimized | timeout（900 s 墙钟） | 901.8 | 7.0 | — | 修正前 trace；停在 BypassDetection，整图 clone 74,657 次 |
| Qwen2.5-0.5B 1 block（修正后 trace） | 165,169 | depth-dp | complete | 27.0 | 3.59 | 448 | |
| Qwen2.5-0.5B 24 层＋LM head | — | depth-dp | complete | 593 | 59.2 | 10,835 | 前端 451 s／17.8 GiB |

另加两行 `[论文 DaCapo]`：ResNet-20 15.8 s、AlexNet 1,042 s（optimized，DaCapo 论文自报，earth_ops 留空）。

### 3.4　registry

每次 trace、编译和 B 部分 trace 各追加一行 `$PT_RESULTS/registry.jsonl`，写法同 `A1-optimized-20261102/driver.py` 的 `run()`：
- round 填 `E4`；
- mode 填 `dacapo-expert-trace`、`dacapo-expert-depth-dp`、`dacapo-expert-optimized` 或 `ptdsl-size-trace`。

## 4　运行顺序与时间

`run.sh`（setsid nohup，日志 `run.log`）按以下顺序执行：
1. §1.4 自检；
2. 6 个 trace；
3. 按规模排序后编译；
4. B 部分；
5. collect。

每个 nixrun 只做一步。最坏情况约 14 小时：每步上限 3600 s，optimized 在第一次超时后跳过更大的模型。运行期间不启动别的 Nix 作业。
全部结束后再跑一次 `leftovers(OUT)`，确认没有残留进程，并把结果写进 E4.md。

## 5　不要做

- 不改 DaCapo 的脚本、poly、前端或编译器，不改 pt_agent 库；不安装 torchvision，不另建上游编译器。
- 不提高内存、时间或磁盘限制；失败不重试，不为某个模型换参数或换 waterline。
- 不运行服务器作业，不调用 API，不读取任何报告集数据。
- 不提交、不推送（另行审批）。

## 6　完成标准

- A 部分：6 个模型各有 TRACE.json，以及 depth-dp 和 optimized 两份 COMPILE.json；没跑的写明原因（跳过规则、trace 失败或已导出服务器作业）。
- B 部分：6 个 PT-DSL 程序各有 SIZE.json。
- 结果目录里有 E4.json、E4.md、POINTS.csv；registry 行齐全；最后一次残留进程检查为 0。
- STATUS.md 新增 E4 一行，按 AGENTS.md 的格式汇报，"与 spec 不一致之处"逐条列出。
