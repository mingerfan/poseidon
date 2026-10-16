# ResNet18：已验收的 native no-hoist 与 DSL

本包使用 `gs/feat-application` 原版 ResNet18 算术与原 ImageNet 权重，把原物理计划显式转换为 CKKS DSL，经过真实 Hecate 编译器生成 HEVM，再调用 Poseidon GPU primitive。原版与 DSL 均 no-hoist；未替换为 ResNet50 修正实验。

## 已完成的验收

同一 image0，真实标签65，N65536/Q50/P25、26GiB pool、原多项式ReLU、归一化及分类规则：

| 检查 | 结果 |
|---|---|
| 原版 / DSL 分类 | 65 / 65，均正确 |
| 原输入 | 147布局 + 缓存加密ONE，共148输入；1000输出 |
| 自举计划 | 原16个逻辑激活位置、60次pack调用，两侧一致 |
| 真实HEVM | 1709296 raw ops / 854648 actions |
| 每op实际metadata | Q/components/f64 scale bits与编译计划全程一致 |
| 独立加密运行logits最大差 / RMS | 0.005010604173461886 / 0.0017419935604740106 |
| 同密钥、同输入密文全网对照 | 1000个最终密文metadata及全部524288000个RNS words逐位一致 |
| 同密文logits最大差 / RMS | **0 / 0**，两侧仍预测65 |

独立加密产生不同随机噪声；固定密钥及输入密文后，本次完整网络未观察到 DSL 执行差异。新native与旧native独立重复的logits最大差0.013476976225392345。这是单样本功能/流程验收，不是测试集准确率或噪声分布统计。没有采用额外的用户精度阈值。

冻结 HEVM SHA256：`e9f6a906d490766134e810d4f16224f05a9f0c7c7f7ebdf67037595d5c84b4df`。

发布前又独立构建了三个入口；237个后端源文件与实验版本字节一致。模型数据重新生成的trace/payload SHA匹配，真实Hecate重编译的HEVM及两个sidecar逐字节一致，CPU metadata门禁、700步GPU及95990步扩展短测均通过；扩展覆盖8次实际bootstrap以及vector/scalar/raw-mul→relinearize的RNS exact检查。`evidence/packaged-*`保存这次复测；完整GPU及同密文全网证据来自前述冻结源码的原运行，没有冒称新构建又完成了一次全网。

`evidence/`保存原全量、700/扩展短测、真实编译审计、CPU语义门禁、同密文全网日志与结果。4.2GB原native输出RNS记录留实验服务器，未放入Git；capture audit保存大小、SHA及逐word比较结果。复用的原native接口和源码快照在`runtime/`，其共享组件沿用`resnet50_gpu`命名，并不表示本包运行ResNet50。

## 检查与复现

在仓库根目录运行。Python3用于包检查，CPU重放另需NumPy；构建需要Linux、C++20、CMake≥3.26.4、CUDA及本仓库GPU依赖。测试机器为V10032GiB（CUDA架构70）。

```bash
P=DSL/resnet18_no_hoist
python3 "$P/tools/check_package.py"
python3 "$P/tools/prepare_assets.py" --payload
CUDAARCHS=70 bash "$P/tools/build.sh"
python3 "$P/tools/run.py" metadata
python3 "$P/tools/run.py" short --gpu 1
python3 "$P/tools/run.py" full --gpu 1
python3 "$P/tools/run.py" native --gpu 1
python3 "$P/tools/run.py" same-cipher --gpu 1
```

`prepare_assets.py --payload` 从打包的固定权重和image0重建原C++ metadata trace及313MiB无损payload，要求SHA与原测试完全匹配；不会用Python合成HEVM。首次执行需要C++编译器及zlib开发头文件。冻结MLIR、trace、映射、常量均以无损gzip保存；解压及新日志写到被Git忽略的`work/`。现有内容不匹配时停止，避免覆盖。

`short`执行700action及覆盖bootstrap/ReLU的扩展前缀；通过后再full。同密文模式先primitive probe，再同context/keys/147原输入密文+原ONE的native→compiled DSL全网。单卡显存必须空闲且至少28GiB；固定pool26，清除三个禁止scale覆盖变量和DNUM覆盖。此模式会额外产生约4.2GB RNS记录，请预留磁盘空间。

真实编译复核，需要已构建且支持本包native primitive/f64 sidecar/V2 uint32 HEVM的Hecate/LLVM18框架：

```bash
python3 "$P/compiler/compile.py" /path/to/hecate-opt
```

编译会逐字节比较HEVM及两个sidecar与冻结产物。`compiler/source_overlay/`保存实际使用的关键框架源文件（含避免复制整张寄存器映射的修复），需要匹配Hecate源树，不是独立编译器或任意上游版本的一键补丁。

可选，真实编译产物的CPU门禁（理想bootstrap，不代表加密精度）：

```bash
python3 "$P/tools/replay_imagenet_hevm.py" "$P/artifacts/resnet18.hevm" \
  "$P/work/assets/resnet18" "$P/work/cpu-replay.json" --replay \
  --inputs "$P/work/assets/resnet18.inputs.jsonl" \
  --payload "$P/work/assets/resnet18.payload.zlib" --oracle "$P/fixtures/resnet18.oracle.f64" \
  --config-source "$P/runtime/native_snapshot/benchmark/resnet50_gpu/resnet50_config.cpp"
```

`package_manifest.json`绑定发布文件原始字节；`check_package.py`离线复算1000个logits、分类和同密文控制。所有参数只用于本实验正确性对照，不作为生产安全参数认证。
