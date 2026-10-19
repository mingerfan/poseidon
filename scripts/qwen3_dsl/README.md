# 完整 Qwen3-0.6B 验证入口

此目录使用完整 28 层 Qwen/Qwen3-0.6B 的预训练权重，固定 revision
`c1899de289a04d12100db370d81485cdf75e47ca`。权重 `model.safetensors` 的
SHA-256 是 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
来源为 [该固定 checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)。
权重、大型 bytecode、常量 bundle 和执行计划保存在服务器工作目录，不进入 Git。

转换入口只依赖 Python 和 NumPy，直接读取 BF16 safetensors，逐项核对 311 个张量、
形状、有限值和 tied embedding/LM 权重。原始张量共 751632384 个参数，包含重复存储的
LM head；去重为 596049920。模型是 hidden 1024、FFN 3072、16 Q heads、8 KV heads、
head dim 128、vocab 151936，使用 Q/K head norm、RoPE 和 causal attention。
客户端先按 token `[9707,11]` 查 embedding，再加密两份 embedding；不是密文 token 查表。
最后位置的完整 logits 分为 149 个输出，另外输出全部 28 层、两位置的 K/V，共 261 个密文。

线性层使用固定 baby-step/giant-step lowering，`B=min(64,input_size)`；上述模型的
全部线性层均为 B64。按 `rotate(s)[i]=x[(i+s)%slots]`，每条矩形对角线写为
`s=g+b`，先缓存 64 个 baby 向量，再将完整物理槽的公开对角线 `d_s` 移位为
`np.roll(d_s,g)`，最后对每个累加组旋转 `g`。所有 `in+out-1` 条对角线及其
权重均保留，没有数值近似或模型裁剪。1024×1024、2048×1024、3072×1024 的
旋转次数分别由 2046/3070/4094 降为 94/110/126；相应转置矩形采用同一实现。
完整槽常量保证带脏数据的输入 padding 也不能污染输出，但增加 trace 的常量
存储量。这只是计算图 lowering；HEFT 算法保持不变。重新生成 trace/plan 后
仍需实际 CKKS 数值验证，FP64 槽检查不能替代完整模型执行。

独立 dense NumPy oracle 覆盖全部七个模型矩阵形状、稠密及短逻辑尾部、脏物理
尾部和小尺寸带 bias 的情形，直接调用正式 `linear`：

```bash
OPENBLAS_NUM_THREADS=2 python test-results/waterline-20261013/dsl-linear-bsgs-proof.py \
  --report test-results/waterline-20261013/dsl-linear-bsgs-production-report.json
```

此命令在仓库根目录运行，只验证槽布局与 dense 矩阵乘的一致性，使用合成 FP64
测试权重。真实 checkpoint 和完整模型的参考与执行仍由下面的流程负责。

`reference.py` 是独立的 dense NumPy FP64 方程参考。非线性 profile 显式提供，校准范围
限于固定两 token 输入；另保存原始 rsqrt/SiLU/softmax 的参考。完整输出固定比较
`atol=0.001, rtol=0.0001`，也检查虚部和所有 padding 槽，不因执行误差改变容差。

```bash
base=/home/xuming/poseidon-waterline-20261013
cd "$base/qwen3_dsl"
OPENBLAS_NUM_THREADS=4 /opt/conda/envs/apollo/bin/python prepare_checkpoint.py \
  --checkpoint "$base/checkpoint" --output "$base/qwen3-fixture" --tokens 9707 11
export HECATE="$base/toolchain" HECATE_BUILD="$base/toolchain"
export PYTHONPATH="$base/toolchain/python" PYTHONDONTWRITEBYTECODE=1
cp profiles/*.json "$base/profiles/"
cp profiles/compiler-profile.json "$base/toolchain/config.json"
OPENBLAS_NUM_THREADS=4 /usr/bin/time -v \
  "$base/toolchain/lib/ld-linux-x86-64.so.2" \
  --library-path "$base/toolchain/lib:/opt/conda/lib:/lib/x86_64-linux-gnu" \
  /opt/conda/bin/python trace_qwen3.py \
  --config "$base/qwen3-fixture/model.json" \
  --approximations "$base/qwen3-fixture/approximations.json" \
  --weights "$base/qwen3-fixture/weights" --slots 8192 --bytecode \
  --output-dir "$base/qwen3-traced-bsgs"
/opt/conda/bin/python compile_plans.py --workspace "$base" \
  --source "$base/qwen3-traced-bsgs/trace_qwen3.mlirbc" \
  --output-name qwen3-plans-bsgs-q50 --devices 1 4
/opt/conda/envs/apollo/bin/python export_runner_fixture.py \
  --signature "$base/qwen3-traced-bsgs/signature.json" \
  --fixture "$base/qwen3-fixture" --spec "$base/profiles/operator-spec.json" \
  --p-moduli-bits 30 30 30 30 30 30 30 30 --output "$base/qwen3-runner-bsgs-q50.json"
```

目录必须尚不存在，脚本不会覆盖旧产物。`calibrate_fixture.py` 可重用已经完整核验的
转换权重生成当前 profile/reference。trace 使用项目原有 Hecate Python frontend。
编译器继续使用 HEFT；latency profile 是估算 fixture，不能用于声称调度最优或模型加速。
编译脚本保存精确命令、状态和 `/usr/bin/time -v` 日志，并仅接受单卡或全部四卡配置。
默认依次编译单卡、全部四卡；单次完整模型编译曾记录峰值 RSS 137225352 KiB，
因此不同时启动两份编译。任一编译子进程失败，脚本最终返回非零。可用
`--devices 4` 单独编译全部卡，并用 `--source` 复用同一完整 BSGS bytecode。
水位计划默认使用 `--prepare-backend auto --lookahead-groups 128`；可显式指定
`--prepare-backend cpu|gpu|auto` 和正整数前瞻组数。Auto 优先选择预算允许的 GPU
编码；运行时仍同时遵守显存、Host、pinned 和 raw 字节预算。旧计划中保存的
窗口和后端不会自动改变，需要重新编译才能采用新的默认窗口。

当前 CKKS 数值配置是 N=16384、50 个 30-bit Q、8 个 30-bit P、默认 scale40。
编译器从该 profile 取 `--waterline=40`，保持 lazy rescale 的触发时机；明文与
Host refresh 输出 scale 为120。runner 按计划描述显式编码输入。逻辑160→40
Rescale 在原生除模前先乘整数校正系数并同步中间 scale，避免先降到约10-bit
再放大舍入误差；不要通过提高 waterline 改变原图的 lazy/Boot 布局来绕过问题。
每个逻辑 level 对应 5 个物理 level，逻辑 level 0..9 映射到物理 level 4..49。
一次逻辑 Rescale 为 120-bit scale 下降额外预留 30-bit 模数空间，运行时依据实际 Q
选择原生除模次数、整数系数校正与剩余 ModSwitch；不只改写 scale 元数据。
默认每算子工作空间上界为 1024 MiB，覆盖 Q50 Rotate 的保守上界。
物理 latency 行 40..49 延用旧表最后一行估算，未作为新硬件测量；逻辑 latency 表
及 HEFT 算法保持不变。此参数更新不代表完整 Qwen 数值验收已经通过。
这个配置使用 `sec_level_type::none`，
不声称满足标准安全级别。Host decrypt/re-encrypt refresh 使用秘密密钥；不是原生 FHE Boot。
这些限制须与实际完整模型运行结果一起披露。N 的调整只改变 CKKS packing/capacity，
不改变模型尺寸、权重、token 或输出范围。最终结果与运行命令见
[Waterline 验证记录](../../test-results/waterline-20261013/README.md)。
