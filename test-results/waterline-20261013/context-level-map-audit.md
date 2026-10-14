# Context level map：调用方审计与局部开销

本阶段保留原来的按值 getter，不修改 `CrtContext`，不增加 API 私有缓存。这是一项独立 CPU 开销测量和未来可选优化记录，不是本阶段待完成修复，不证明完整模型的主要慢因或预期加速。

## 调用语义与引用边界

[原 getter](../../src/poseidon/crt_context.h) 返回 `const std::unordered_map<uint32_t, parms_id_type>`，因此每次调用复制整张 map。搜索生产源码、测试及仓库内 C++ 诊断调用后，未发现依赖“修改 getter 返回对象”来改变 context 的用法。

- 私有 `parms_id_map_` 仅在 [CrtContext 构造函数](../../src/poseidon/crt_context.cpp) 的三处 `emplace` 填充，未发现构造后的写入入口。当前持有该 context 的 API/evaluator 在方法调用期间保留其所有权；现存调用没有从临时 context 取引用并让它逃逸的模式。
- 大部分调用立即 `.at(level)`，随后复制 `parms_id_type` 或作为参数使用。返回 `const Map&` 可消除这些调用中的整图复制，查找及缺失 level 的异常语义不变。
- [GPU uploader](../../src/poseidon/gpu/gpu_uploader.cpp)、[linear transform](../../src/poseidon/advance/homomorphic_linear_transform.h) 和部分测试使用 `auto map = getter()` 或 `const auto map = getter()`；即使 getter 改成引用，这些局部变量仍是自己的副本，读写语义不变，也不会自动消除这些显式副本。
- [CKKS evaluator](../../src/poseidon/evaluator/evaluator_ckks_base.cpp) 的 `auto &id_level_map = ...parms_id_map()` 当前绑定并延长 const 临时副本的寿命；改为引用后将借用 context 成员。该方法内 context 持续存活，且只调用 `find/end`、读取参数，因而当前用法安全。

如果未来改为引用，调用者必须让 `CrtContext` 存活到引用、iterator 或元素引用的最后使用，不能从临时或即将销毁的 context 保存借用结果；期间也不能移动、赋值或并发修改该对象。当前仓库的只读用法支持此改法，但不构成外部使用方审计。

这不是 ABI 兼容修改：普通非模板成员函数返回类型通常不进入符号名，按值与引用返回的调用约定不同。公开 inline getter 的旧对象/weak 定义不能与新调用方任意混用；相关 shared library、API、runner 及下游消费者须按同一新 header 重建。本阶段不承担这一接口迁移。

## CPU 测量

[小型源码](context-level-map-benchmark.cpp) 链接既有 CPU `libposeidon_shared.so`，构建真实 `N16384/Q50/P8/scale40` context，再取得一份未计时的同内容 map 快照。实际 map 有 **51项、59个bucket**：50个数据 level 和一个 key level（57）。两个私有 `noinline/noclone` getter 仅返回类型不同，并使用相同编译器 memory barrier；动态 xorshift 输入遍历全部50个数据层。

硬件为 AMD Ryzen AI 9 HX 370，固定 CPU0，`OMP_NUM_THREADS=1`、g++16.2.1、`-O3`；具体编译器版本、路径及命令以 [命令记录](context-level-map-commands.json) 为准。每次100万查找，三次交替顺序运行。context构造、一次快照、输入生成和warmup不计入查找时间。计时期间关闭分配计数；另外执行一次100万查找的计数 pass。

| 私有 getter | 100万查找中位数 | 三次范围 | 单独计数 pass 的 `new` 次数 | 累计请求字节 |
| --- | ---: | ---: | ---: | ---: |
| 原按值返回 | 450.414 ms | 444.162–461.346 ms | 52,000,000 | 2,920,000,000 B |
| const引用返回 | 3.162 ms | 3.155–3.166 ms | 0 | 0 B |

所有计时与计数 pass 的累加校验相同。当前标准库中，每次副本产生52次分配，累计请求2920 B；这里的次数和字节是反复申请的总量，**不是同时存活量、RSS峰值或持续增长**。计时仍保留 replacement `new` 的关闭计数检查，不代表完全未插桩的生产 allocator。

[结果及输入hash](context-level-map-result.json)、[原始JSONL日志](context-level-map-benchmark.log)记录实际数据。命令依赖当前工作区已构建的 CPU shared library；迁移工作区须重新提供对应路径及构建，不是空 checkout 即可运行的模型任务。没有运行 GPU、访问服务器或测量完整模型。

完整运行中的 getter 调用总数、其他 Host 工作及调度等待没有由此基准归因。不能将微基准差值或累计分配量直接乘成模型收益，也不能据此把约149分钟前向的主要原因归到这个 getter。当前决定仅保留测量，继续既定模型验收。
