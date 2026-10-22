# 最新优化版两层Qwen2.5-0.5B DSL

本包严格使用2个独立Decoder Block、2token decode-pair，真实hidden896、FFN4864、14Q/2KV/d64、151936词表。
沿用018全部算术kernel（包括已验证重复mask消除）；保留旧内部模块名qwen24和编译脚本名以兼容工具路径，
实际层数由config/model.json和严格2层的spec/runtime约束，不能把目录名当作24层执行。

推荐入口在父目录trace_combined_model.py，packed接口为2个加密embedding输入、5个logits块+2个KV输出。
基础trace_qwen24.py保留旧prefix接口，不应混用两种输出签名。
原生编译后执行CPU V3按需/有界明文编码变换，显式Release；同48阶真Boot，不传私钥给执行API。
真实CKKS结果以对应reports为准，不能用编译或Mock通过替代。使用合成权重，不代表预训练Qwen校准。
