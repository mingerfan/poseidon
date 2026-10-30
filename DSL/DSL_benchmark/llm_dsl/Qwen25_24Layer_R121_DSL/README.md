# Qwen2.5 0.5B：24 层 R121 优化方案 DSL

本工程由已发布且完整 CPU CKKS 通过的 **Qwen25_2Layer_R121_DSL** 扩展为 **24 个独立 Decoder Block**。使用相同的算子、非线性近似、QKV 打包、配对 MLP down tile=256 和 CPU 内存后处理。不是复制 12 次两层调用：各层有独立权重与 KV，前 23 层使用配对 MLP 路径，末层保留原末层策略。

**本包已通过小尺寸24层算术检查、真实尺寸编译、原始/优化计划校验及完整24层Mock；尚未执行完整CKKS或GPU测试。** R121 的 74 次自举、45.68 分钟和27.08 GiB RSS只属于两层测试，不能套到本包。24 层实际编译统计见 [MODEL_DATA_ANALYSIS.md](MODEL_DATA_ANALYSIS.md)。

## 模型与接口

| 项目 | 本包配置 |
|---|---|
| Decoder Block | 24，独立权重、独立 KV |
| hidden / intermediate | 896 / 4864 |
| Q heads / KV heads / head dimension | 14 / 2 / 64 |
| 词表 | 151936 |
| 环度 / 每密文槽数 | 65536 / 32768 |
| 目前支持范围 | 最多两个可见 token；位置上限512不代表支持512-token序列 |
| 默认输入 | 两个已知 token 的 embedding 密文 |
| 默认输出 | 末 token 的5个 logits 密文 +24个层 KV 密文，共29个 |

包括每层 RMSNorm、QKV、RoPE、因果 Attention（第二 token 的 Softmax 近似）、O投影、残差、SwiGLU，以及最终 RMSNorm 和完整词表 LM head。首 token 只有一个 key，无用 Q 分支可被编译器裁剪。只请求末 token 的 logits，其他不影响输出的分支也可被裁剪。

权重为明文常量。客户端先查 embedding，再加密输入；输出仍为密文，由客户端解密、采样。默认 `decode-pair` 的两个输入都已经给定，采用已知输入联合计算，不是在加密域内提前生成未知的下一个 token。输出 logits 不再做 Softmax。

- `decode-pair`：2个输入，29个输出；主交付路径。
- `prefill`：两个 token 一起输入，29个输出。
- `prefill-one`：1个输入，29个输出；每层 KV 前256槽有效。
- `decode`：第二个 embedding +24个外部 KV，共25个输入，29个输出。当前只支持一个已缓存 token；跨独立编译图的密文 level/scale 对接尚未验证。

接口 JSON 在 `interfaces/`。两 token KV 前512槽按 `K0,V0,K1,V1` 排列；logits前4块各32768槽，末块20864槽有效。

## 保留的优化

`delivery-config.json` 固定 R121 实际追踪开关。25个继承算术/调度文件与父包逐字节相同，见 `source-binding.json`。

- 宽 Linear 的 BSGS 路径，LM head baby=32；分组 RMSNorm 和 SiLU 近似。
- joint QKV 列分块 tile128，首 query 独立可裁剪，内部 V 布局复用和干净 KV 输出。
- 两个已知 token 的 gate/up/SiLU 配对计算，配对 down 投影 tile256。
- CPU 计划后处理：终端冗余刷新消除、release_ready 调度、64MiB Encode 根缓存。64MiB只限制该缓存，不限制进程总内存。
- 自举由匹配编译器自动插入；CPU参考合约为真实48阶自举，L1→L5、scale43。

配对计算通常利用两块不相交槽区；这不意味着所有层都达到高槽位利用率。24层新增路径已通过真实尺寸编译/Mock；密文误差累积仍需真实CKKS验证。

## 使用

Python3.10+、NumPy，以及目标环境提供的 Hecate 前端、`hecate-opt`、Runtime/Poseidon库。包内不含权重、密钥或二进制库。合成权重生成会写约2.52GB；它不是预训练模型下载器。

```bash
python -B verify_package.py
python -m pip install -r requirements.txt
python -B trace_model.py --describe
python -B validation/check_small_model.py --case basic --output-dir /tmp/qwen24-basic
python -B validation/check_small_model.py --case tiled --output-dir /tmp/qwen24-tiled

# 真实尺寸工作在目标服务器执行，每次使用新目录。
python -B prepare_fixture.py --output-dir /data/qwen24-fixture
python -B trace_model.py --weights /data/qwen24-fixture/weights --output-dir /data/qwen24-traced
python -B compile_model.py --hecate-opt /path/to/hecate-opt --input /data/qwen24-traced/ACTUAL.mlirbc --output-dir /data/qwen24-compiled
python -B optimize_cpu_plan.py --input /data/qwen24-compiled/ACTUAL.runtime-plan.json --output /data/qwen24-plan.json --report /data/qwen24-audit.json
/path/to/verify_runtime_plan /data/qwen24-plan.json frozen/model/package/config/operator-spec.json
```

`ACTUAL`按编译器实际输出替换；常量 bundle 必须跟随计划保存。`trace_model.py --mode prefill|prefill-one|decode` 可选择其他路径；`--dry-run`只列出追踪命令。`compile_model.py --dry-run`只列出编译命令，但要求输入和编译器路径存在。

`prepare_fixture.py`生成默认 decode-pair 的独立 dense 参考与 `runtime-fixture.json`。如选用其他入口，应按对应 `interfaces/` 提供输入与参考，不能直接使用默认 fixture。

`frozen/tools/mock_probe.cpp`、`native_probe.cpp`已将输出与层数约束改成24层/29输出，自举期望次数从实际计划读取。**Mock探针已构建并完成24层运行；native CKKS探针尚未构建执行**。后者仍需要匹配的Runtime API、Poseidon库及native Boot回调。

```bash
/path/to/mock-probe /data/qwen24-plan.json frozen/model/package/config/operator-spec.json /path/to/bundle /data/qwen24-fixture/runtime-fixture.json /data/qwen24-mock.json
OMP_NUM_THREADS=32 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1 OMP_WAIT_POLICY=PASSIVE /path/to/native-probe /data/qwen24-plan.json frozen/model/package/config/operator-spec.json /path/to/bundle /data/qwen24-fixture/runtime-fixture.json /path/to/cpu_bootstrap_reference.json /data/qwen24-ckks.json
```

父版使用实验编译器 SHA256 `12705f29938821d48ae3db3def7767af7a2af6507d40fb0c7e9bfdb98918de68`，含 normalize预算修正；`compiler_reference/`提供补丁和源码绑定，不代表任意仓库 HEAD 都已等价。CPU探针继承选择性内存池回收和 native NAF 接口依赖，需要对应运行库。

GPU可复用DSL源码，按GPU合约重新追踪/编译验证；编译入口支持配套的 `--compiler-config`、`--operator-spec`、`--device-count` 和 `--boot-profile`。`optimize_cpu_plan.py`是单机CPU后处理，不能直接当作GPU调度器。CPU参考精度不等同于GPU验收。

## 验证与来源

`evidence/small-basic.json`、`small-tiled.json`记录实际执行的24层小尺寸算术。每个case检查四条路径、每路径24个Block、独立权重、脏尾部输入、干净输出、外部缓存不变及独立原函数dense参考。tiled case确实覆盖tile256和跨tile的385维MLP中间层。测试未模拟CKKS噪声。

`evidence/parent_two_layer/`严格只属于父版两层结果。`source-binding.json`记录精确继承与修改文件；`manifest.json`校验本包。冻结算子内历史注释保留原出处，以本README的验证范围为准。底层 `frozen/model/package/trace_qwen24.py`保留旧未打包ABI兼容用途，实际交付使用根目录 `trace_model.py`。

## 真实尺寸编译分析已完成

本版24层decode-pair已完成服务器编译与完整Mock，最终计划866次native Boot；全保留逻辑系数1,226.871094GiB，释放图峰值356.500MiB。两者均不含密钥/自举工作区，不能当作CKKS进程RSS。详见[数据与内存主报告](MODEL_DATA_ANALYSIS.md)和[逐层level与自举明细](COMPILED_LEVEL_DETAILS.md)。其他入口未执行真实尺寸编译。
