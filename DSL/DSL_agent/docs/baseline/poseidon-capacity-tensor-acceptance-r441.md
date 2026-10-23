# Poseidon 跨块张量容量验收

2026-10-02，ARM Ubuntu，分支 lhy-agent-dsl。核心 Poseidon CPU 迁移已通过；本批完成非对齐连接、指定轴归约和命名多输出的容量扩展。

## 实现范围

公开形状、轴和权重生成独立布局绑定，候选及reference不能替换绑定。capacity-public-tensors-v4新增C-order索引映射；capacity-chunks-v5规则基线采用旋转、掩码、连续窗口求和与typed helper。末轴stack复用one-hot局部行映射，保持原256公开向量和1024工作预算。内部helper节点名避开所有用户节点。

| 分区 | 新模型 | 代表性范围 |
|---|---:|---|
| concat | 3 | [257]+[263]尾部；二维内轴；四维负轴 |
| stack | 4 | 外轴、内轴、负轴；580输出元素、4输入密文块 |
| sum | 3 | 外轴、多个非连续轴、末轴；keepdims |
| mean | 3 | 同上，独立验证归约除数 |
| 图组合与命名 | 2 | sum/mean→concat，2输入3输出；真实Linear节点与虚拟stack节点重名 |
| 合计 | 15 | 每例16组确定性双参考与生成算术检查 |

中型输入/输出各1024元素、各最多4个物理密文块。上述范围是代表性验收，任意轴组合仍须通过布局、资源、编译和数值检查。公开常量参与的连接及其他跨块算子未由本批声明密态支持。

## 测试结果

| 检查 | passed | failed | blocked / skipped |
|---|---:|---:|---:|
| 静态、参考、组件及兼容回归 r438 | 132 | 0 | 0 |
| 连接/归约与兼容模型 r435 | 17 | 0 | 0 |
| 多输入命名输出组合 r437 | 3 | 0 | 0 |
| 命名修复与受影响回归 r439 | 4 | 0 | 0 |
| 程序包真实重载（含两次公开CLI） | 8 | 0 | 0 |

24次编译密态执行涉及19个不同模型：15个新增、4个兼容回归。96组输入/24,296值；按模型去重76组/21,960值。最大绝对误差1.36997235245e-06。重复的Linear/MLP/stack不增加模型计数。

八次重载32组/11,876值，最大误差8.99221763273e-07。重载重新加密、执行、解密并比较，生成和编译调用均为0；程序包重复使用当前RuntimePlan、OperatorSpec和plaintext bundle。

所有密态数值条件均保持：

~~~text
abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)
~~~

本批候选全部来自规则基线。新真实Agent生成、付费调用均为0。静态检查、算术解释、真实编译和Poseidon CPU密态结果分别记录；不把本批基线通过记为Agent生成成功。

## 绑定与兼容

r435与r437绑定命名修复前590执行/533应用文件。随后只改unified_capacity_constants.py的虚拟名保留以及test_unified_capacity_tensor.py反例；r439重新绑定最终源码，并验收冲突案例、较大stack、旧Linear与真Linear→square→Linear。r440从r439证据导出，公开CLI重载通过。未重复全量17例。

最终执行源码map SHA-256：6a548a73318c57488235748e34160a7192582af2d67eb1798c1f2f923877e682。模型、请求、候选、编译器、依赖、真实参数和产物绑定见配套JSON及各plan.json。修改验证源码后旧包保持原绑定，不改写其manifest冒充当前通过。

v1/v2/v3显式绑定保留；8个旧v3请求和21个无descriptor请求按原版本精确重建。现有x86配置保持，本批实际执行平台是ARM Ubuntu，x86原生执行未验收。冻结1,200模型集未重建。

## 原始结果

- r435：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tensor-numerical-r435
- r437：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tensor-composition-r437
- r438：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tensor-static-r438
- r439：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tensor-collision-r439
- r440：/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tensor-cli-r440

r431源码快照和早期测试失败保持。两个早期静态失败为旧metadata断言及资源错误文本预期；原末轴stack的256向量阻塞通过同模型的tiled映射解除。未删除或简化原模型来获得通过。

## 后续

跨密文Conv/Pool、其他跨块形状/切片、容量真实Agent及定向构造继续推进。12深多项式容量失败、真实bootstrap重复刷新精度、ResNet20物理布局与整网精度分别保持未完成。ratio128有界试验已消耗批准：有符号第二刷新15/16,384值超门限、最大误差2.158906262e-5，随机/边界未运行，未接入Agent。

本批只修改Agent Python、补充模型、测试、开发工具和文档。Dacapo/Runtime核心、Poseidon库、模型reference、安全参数、误差门限及沙箱上限未改；安装、下载、API、commit、push为0。

当前程序包：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tensor-cli-r440/program --execute
~~~
