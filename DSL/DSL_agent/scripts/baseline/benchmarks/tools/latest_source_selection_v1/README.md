# 最新来源入口

显式 latest 已选择v27。普通位置参数按--input顺序使用现有调用绑定；--call用于字典、嵌套输入和关键字参数。省略--source-format仍使用v18，显式旧版本和已保存清单保持。

27项现有与新增控制检查通过；resize、ConvTranspose、组合模型的来源清单与旧显式v27完全一致。验收记录：docs/baseline/agent-latest-source-selection-acceptance-r1425.json。该入口调整没有新增API、编译或密态调用。原草稿、补丁和日志保留。
