# R5d：为 pt_agent 自己生成的程序提供可信运行器，完成整网 trace 和编译尝试

用户已于 2026-10-20 批准：pt_agent 的 lowering 产出的程序**可以不进 bwrap 沙箱运行**，但仍在 Nix 环境中执行，并设置以下限制：

- 内存 6 GiB；
- 单个文件 8 GiB；
- 墙钟 1 小时；
- 启动前磁盘剩余空间 ≥10 GiB。

LLM 的输出照旧永不执行。旧流水线和 B 线使用的沙箱及其限制一概不改。本机内存不够时，把同一套命令留作服务器作业。

为什么需要这个运行器：R5c 选出的整网程序（deg=127、margin=1.25，常量 815 MB）在 trace 时，撞上了沙箱 `RLIMIT_FSIZE=16 MiB` 的上限，MLIR 文件停在 16777216 字节。

## 0. 已核实的事实

- 前端 `expr.py` 读取环境变量 `HECATE`，并支持用 `HECATE_BUILD` 覆盖构建目录；它会加载 `$HECATE_BUILD/lib/libHecateFrontend.so`。
- `/usr/bin/time` 存在，可用 `-f "%M %e" -o <文件>` 记录峰值 RSS（KiB）和墙钟秒数。
- 本机磁盘剩余约 26 GB。
- R5c 的整网程序在 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/pt-agent-r5-resnet20-emit-20261020T154644-1731/`，里面有 `candidate_trace.py`（路径写的是沙箱内的 /frontend、/out）和 `constants.npz`。

## 1. 修改与新增

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/toolchain.py                 改：新增 §2 的三个函数；trace/compile 增加 trusted 参数
pt_agent/experiments/r5d_compile.py   新：§3
pt_agent/tests/test_r5d_trusted.py    新：§4
../docs/pt_agent/STATUS.md            改
```

## 2. toolchain.py 的新增部分

```python
TRUSTED_LIMITS = dict(mem_gib=6, fsize_gib=8, wall_seconds=3600, min_free_gib=10)

def host_paths() -> dict:
    """setup_env() 之后，从 poseidon_runtime_paths 读取：
    hecate_opt=BUILD/"bin/hecate-opt", hecate_build=BUILD, hecate=BUILD.parent,
    frontend=DACAPO/"python/hecate/hecate/expr.py", profile=PROFILE, spec=SPEC。"""

def retarget(run_dir: Path) -> None:
    """把 run_dir/candidate_trace.py 里三个由 json.dumps 写出的路径字符串逐一替换为宿主机路径，每一处都必须恰好命中一次：
    "/frontend/hecate/expr.py" → host_paths()["frontend"]
    "/out/constants.npz"       → run_dir/"constants.npz"
    hc.save("/out", "/out")    → hc.save(<run_dir>, <run_dir>)
    原文件先另存为 candidate_trace.sandbox.py。"""

def to_host_argv(argv: list[str], run_dir: Path) -> list[str]:
    """逐项映射：
    "/hecate-opt"                 → hecate_opt
    以 "/out/" 开头的项            → 把前缀换成 run_dir
    "--ckks-config=/profile.json" → "--ckks-config=" + profile
    其他字符串中的 "/out/" 前缀（例如 "-o" 后面那一项）也按同样规则替换。
    映射后，任何一项里只要还有 "/out/"、"/frontend"、"/profile.json"、"/operator-spec.json"，就抛 ValueError。"""

def trusted_run(run_dir: Path, argv: list[str], log: Path, *, env_extra: dict | None = None) -> dict:
    """inside nix。
    1. shutil.disk_usage(run_dir).free < min_free_gib GiB 时抛 RuntimeError。
    2. 执行 ["/usr/bin/time", "-f", "%M %e", "-o", str(log)+".time", *argv]，cwd=run_dir，
       环境变量取当前进程的环境加上 env_extra。
       preexec_fn 中用 resource.setrlimit 设置 RLIMIT_AS = mem_gib GiB、RLIMIT_FSIZE = fsize_gib GiB；
       subprocess.run 的 timeout 取 wall_seconds。stdout 和 stderr 都写入 log。
    3. 返回 dict(exit, seconds, peak_rss_mib, log_tail=日志最后 800 个字符)。超时时 exit=-9，并标记 timeout=True。"""
```

- `trace(run_dir, seconds=600, *, trusted=False)`：trusted=True 时，先调用 `retarget(run_dir)`，再用 `trusted_run` 执行 `[VENV/"bin/python", "-B", str(run_dir/"candidate_trace.py")]`。env_extra 设 `HECATE`、`HECATE_BUILD`，以及 `LD_LIBRARY_PATH=os.environ["HECATE_PYTHON_LIBRARY_PATH"]`。返回值另加 `mlir_bytes`。
- `compile(run_dir, mode, boot=None, seconds=900, *, trusted=False)`：trusted=True 时，argv 取 `to_host_argv(compile_argv(mode, boot, run_dir), run_dir)`，用 `trusted_run` 执行，env_extra 中设 `LD_LIBRARY_PATH`；另外 trace 输入路径中的 `/out/candidate_trace.mlir` 也会经映射改成 run_dir 下的路径。返回值另加 `peak_rss_mib` 和 `plan`（RuntimePlan 是否存在）。
- `trusted=False` 时，行为与 R2b 完全一样。

## 3. experiments/r5d_compile.py

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.experiments.r5d_compile --emit-dir <R5c 的整网程序目录>
```

运行模式同 R5：外层和内层分开，结果目录为 `$PT_RESULTS/r5d-compile-<时间戳>`，registry 写 round="R5d"、model="resnet20"、mode="dacapo-depth-dp-qwen48"。

1. 把 `--emit-dir` 下的 `candidate_trace.py` 和 `constants.npz` 复制到新的 `toolchain.new_run_dir("r5d-resnet20")` 中（不修改原目录），再调用 `trace(..., trusted=True)`。
2. trace 成功后，调用 `compile(run_dir, "dacapo-depth-dp", boot="qwen48", trusted=True)`。
3. 编译成功时调用 `plan_stats`，记录 boots、steps 和 values。
4. 任何一步失败（内存不足、超时、退出码不为 0），都照实记录 exit、timeout、peak_rss_mib 和 log_tail，不重试。同时在结果目录生成 `server_jobs/r5d-resnet20/run.sh`，内容是下面三项：
   - 需要设置的环境变量（POSEIDON_WORK_ROOT、POSEIDON_SDK_MANIFEST，以及作为参数的 run_dir）；
   - 依次调用 `python3 -m pt_agent.experiments.r5d_compile --emit-dir <目录>` 的命令；
   - 一段注释，说明服务器上需要准备相同的 Nix 环境和工具链目录。
5. REPORT.json 包含：trace（exit、seconds、peak_rss_mib、mlir_bytes）、compile（exit、seconds、peak_rss_mib、timeout、log_tail）、plan（boots、steps、values），以及 counts（复制自 R5c 的 emit 记录）。

## 4. 测试（nixrun，不跑整网）

1. 对一份 R2 的 P1 小程序：EmitBackend 写出 → `trace(trusted=True)` → `compile("eva", trusted=True)`，两步的 exit 都是 0，RuntimePlan 存在，peak_rss_mib > 0。
2. `retarget` 后的程序文本中不再含有 "/frontend" 和 "/out/"，另存的 candidate_trace.sandbox.py 存在。对已经 retarget 过的文件再调用一次，抛 ValueError，因为找不到要替换的路径。
3. `to_host_argv(compile_argv("dacapo-depth-dp", "qwen48", d), d)` 中不再出现任何沙箱路径，`-o` 后面那一项以 d 开头。
4. 用 `monkeypatch` 把 `shutil.disk_usage` 改成返回很小的值时，`trusted_run` 抛 RuntimeError。

## 5. 不要做

- 不修改 baseline 的沙箱和 `process_resource_limits`，也不修改旧流水线的任何限制。
- 不执行 LLM 生成的任何内容；trusted 运行器只用于 pt_agent 的 EmitBackend 产出的程序。
- 不在本机执行 CKKS 整网，也不重试失败的编译。

## 6. 完成标准

- `test_r5d_*.py` 全部通过；R1–R5 的测试依然通过。
- `r5d_compile` 跑完，汇报 REPORT.json 的摘要：trace 和 compile 的耗时、峰值内存、MLIR 大小；编译成功时给出 boots 和 steps，失败时给出原因和服务器作业路径。
- STATUS.md 新增 R5d 一行，并删除"阻塞"栏中与 16 MiB 文件限制有关的条目。
