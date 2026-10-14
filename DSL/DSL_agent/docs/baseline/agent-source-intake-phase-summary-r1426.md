# 来源接入阶段结果

51种公开表达与类型转换来源全部完成编译、Poseidon CPU数值验证和程序包复用。26个不同公开请求真实生成，均首次通过；其余25个来源回放对应原样Agent源码，生成来源单独绑定。失败、跳过和未运行均为0。

合计408组输入、2,968个输出值；最大绝对误差2.9944661714533247e-8。26次生成／26次HTTP、51次编译、153次密态调用。该数量按来源写法统计，未声称51个去重数学模型或全DSL／PyTorch覆盖。

latest已选择v27，普通位置参数按--input顺序绑定；嵌套和关键字输入使用--call。27项控制通过，3种已有v27清单完全一致；旧默认v18和显式版本保留，执行与验证身份保持。

静态广播草稿的47个正例、9个拒绝及两类形状查询通过，752组明文输入、10,384个值差异为0；尚未注册、编译或密态运行。下一步接入显式来源契约，再用小批验收。

回放冻结与审计工具的接口／字典顺序问题已修复，原失败日志保留，未重跑生成或后端。后端、参数、数值门限、网络和Git提交保持；bootstrap和CUDA继续待验。

[真实生成审计](agent-normalized-source-live-acceptance-r1420.json) · [原样回放审计](agent-normalized-source-replay-acceptance-r1423.json) · [最新入口](agent-latest-source-selection-acceptance-r1425.json) · [机器可读汇总](agent-source-intake-phase-summary-r1426.json)
