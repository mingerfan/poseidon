# ResNet20 ReLU：原生 no-hoist 流程的 DSL benchmark

将现有 native ReLU ResNet20 的物理执行计划转换为显式 **CKKS DSL**，经真实
`hecate-opt` 编译为 HEVM，再调用原生 Poseidon GPU primitive 执行。这里不是旧的
SiLU 模型，也不是自动规划成56次自举的版本。

## 已完成的验证

同一 image0（真实类别3）、`resnet20_new` 权重、d13 ReLU 多项式：

| 项目 | 保存的全量结果 |
|---|---:|
| DSL / native / 明文预测 | 3 / 3 / 3 |
| DSL 对明文最大 logits 绝对误差 | 0.012577189250295051 |
| native 对明文最大 logits 绝对误差 | 0.007694642024733245 |
| DSL 对 native 加密输出最大差 | 0.006714839973868791 |
| 原生自举位置 / 次数 | 9个block的act1、act2 / 18次 |
| 完整执行 action | 13,420 |
| 实际 GPU RNS exact 检查 | 4,785项全部通过 |

当前是**单样本功能验收**，不是测试集准确率或统计精度等价证明。两次加密使用
独立随机密钥/随机性，最终 logits 不要求逐bit相同；0.1不是本实验的用户验收阈值。
全部原baseline/fold必需门禁通过。原生可选`strict_target_1e_4`日志并非必需验收条件。

输入为27个原生stem布局（每个32768 slots、Q32/scale40），stem输出Q31；19次ReLU，
其中stem不自举，9个残差block各两次自举。Q50/P25、N65536、no-hoist、26GiB pool。
ReLU按原`rescale_many`/x2，Conv按原sequential rescale；plaintext scale使用原生
source、ratio、`exp2(pre)/source`表达式，禁止套用旧51-bit补偿。

实际矩阵路径在日志中是`baseline_S2C_no_hoist`和`baseline_C2S_no_hoist`。
fold实数投影的单个conjugation key storage仍会打印`double_hoist=1`，原native也一样；
这不表示矩阵阶段采用DoubleHoist BSGS。保留原计算和真实日志，不能以删日志代替验证。

`--unsafe-*`沿用原native入口：本实验参数没有生产安全性认证，不把正确性实验作为安全参数证明。

## 目录

交付前还在干净的仓库`dsl`基线上独立构建了本目录，完整CPU loader/21层encoder
和13,420 action元数据门禁通过。重新编译MLIR生成的HEVM SHA与原全量产物一致；
实际常量/27输入的CPU重放误差为1.3343e-14。新构建的700步GPU短测通过273项
RNS exact检查；1800步GPU短测通过644项检查及2次原生no-hoist自举/fold门禁。
`evidence/packaged-*`为这次打包复测，原全量日志保留原文件名；
新构建短测不代表又完成了一次全量GPU测试。

- `source/`：完整显式CKKS MLIR、原生连接图和逐指令lowering manifest。
- `artifacts/`：真实compiler输出HEVM及FNV绑定的f64 scale、native编码规则、rescale分组。
- `fixtures/`：无损gzip常量/27输入、原生合同、解压后的大小与SHA256。
- `runtime/`：单GPU执行入口、动态scale/分组/生命周期调度、CPU单元检查。
- `runtime/native_snapshot/`：测试时原native算术/诊断代码快照。只复用原生API，不修改仓库公共行为。
- `compiler/`：真实EmitHEVM的opt-in native-scales补丁、blb5配置和重新编译脚本。
- `evidence/`：原native和DSL全量/短测日志、逐类别数值对比、编译审计和包装测试记录。

source graph是数学/物理计划描述；HEVM由真实compiler生成，不由Python伪造。
raw HEVM有27,410条指令，包含alloc和lowering展开；恢复native rescale groups后有13,420个action。
自举18次不是18条完整网络指令，也不是56次自动规划结果。

## 无GPU的包检查

```bash
python3 Benchmark/dsl/resnet20_no_hoist/tools/check_package.py
# 加上3组C++ parser/metadata/拒绝坏输入检查；仅需C++20编译器，不启动CUDA设备
python3 Benchmark/dsl/resnet20_no_hoist/tools/check_package.py --cpp
python3 Benchmark/dsl/resnet20_no_hoist/tools/prepare_assets.py
```

常量gzip约4.9MiB，解压488,914,000 bytes；27输入解压7,077,888 bytes。
解压程序验证原始SHA，遇到不匹配已有文件会停止，不覆盖。默认放到本目录被忽略的`build/assets/`。

可选NumPy CPU语义重放，bootstrap按理想恒等，不代表加密精度：

```bash
p=Benchmark/dsl/resnet20_no_hoist
python3 "$p/tools/replay_strict_compiled_hevm.py" \
  "$p/artifacts/strict-native-v1.strict_native_resnet20.hevm" \
  "$p/build/assets/strict-native-v1" "$p/evidence/native_full_20261014.log" \
  "$p/build/compiled-cpu-replay.json"
```

## 构建、元数据检查和GPU复现

要求Linux、C++20、CMake>=3.26.4、CUDA/CUPTI，以及本仓库的离线GPU/RMM依赖。
这是独立opt-in目标，不改变默认CPU/runtime工程。默认CUDA架构70，其他卡可设置
`CUDA_ARCHITECTURES`，默认构建并行度2。

```bash
p=Benchmark/dsl/resnet20_no_hoist
bash "$p/build.sh"
bash "$p/run.sh" metadata
# 以下GPU编号只是示例，脚本会拒绝占用中的卡或不足28GiB空闲的卡。
bash "$p/run.sh" short 2 > "$p/build/short700.log" 2>&1
bash "$p/run.sh" boot-smoke 2 > "$p/build/short1800.log" 2>&1
bash "$p/run.sh" full 2 > "$p/build/full.log" 2>&1
```

先检查每份日志和退出状态再执行下一步。700action在首次自举713之前，正常应显示
`DSL_TRUNCATED ops=700 bootstrap_calls=0`；1800action需完成2次自举和原fold门禁。
full需`ops=13420 bootstrap_calls=18`，所有RNS exact和原bootstrap/fold门禁通过。
不要把metadata模式、CPU700截断或未经过自举的700短测当作完整加密验证。

运行脚本显式禁用`POSEIDON_RESCALE_META_BITS`、`POSEIDON_RESCALE_MODEL_BITS`、
`POSEIDON_DSL_SCALE_SHIFT`，固定pool26GiB及2GiB headroom。实际加密和key setup仅在
`--unsafe-accuracy-only`中启动。程序带observer和RNS下载，不作为online性能基准。

`POSEIDON_TRIDENT_RESNET20_ROOT`可指向相同的既存模型资产根，默认使用仓库
`bench/resnet20_s2c_first/support/data/resnet20`；资产来源及许可边界沿用
`bench/resnet20_s2c_first/support/PROVENANCE.md`，本包不新增模型许可声明。

`build_validated_runtime.sh`仅用于复用原验证树的同版本kernel-object缓存：
设置`POSEIDON_VALIDATED_ROOT`后构建，不可用它混合任意版本的对象与头文件。
正常新checkout使用上述CMake源码构建。

## 重新编译DSL

已有产物可直接运行，不必重新生成。重新编译必须使用支持`poseidon_native_scale:f64`
的真实Hecate编译器；`compiler/EmitHEVM.native-scales.patch`保留了本实验的opt-in改动。
在相符Dacapo版本先`git apply --check`，再应用/构建，不能对不匹配版本盲目覆盖。

```bash
HECATE_OPT=/path/to/hecate-opt bash Benchmark/dsl/resnet20_no_hoist/compiler/compile.sh
```

该脚本重新emit HEVM并逐条核对原manifest。不要使用默认`--dacapo`自动placement删除
原18次bootstrap。新编译产物必须重新审计后才能配合sidecars运行。

## 日志与SHA

保存的原full HEVM SHA256：
`36b9b6f2e66d85b84cd7ca988a04c50a37635689f7f86a8f6710ba384bf221cd`。

保存的原DSL full日志SHA256：
`d49b0c50c6e3a713148b79285027181ae9af36a939451fa1b8fdaee37304fa98`。

`manifest.sha256.json`覆盖包内不可变源/产物/证据；包检查重新读取原日志计算差值，
而不是只相信结果摘要。重新运行因密钥随机性可能有小幅误差变化，应另存新日志。
