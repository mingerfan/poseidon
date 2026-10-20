# 24 层 R071：实际编译 level 与自举明细

24 个真实尺寸 Decoder Block、两个 token 顺序 decode。以下均来自本版本实际 CPU RuntimePlan；没有执行 CKKS 或 GPU。所有容量为系数载荷，不是运行 RSS。共享/融合来源单独保留，不重复计入多个子层。

## 1. 优化后所有 Value 按 level 和分量分布

| 类型 / level / 分量 | 数量 | 系数字节 | GiB |
| --- | --- | --- | --- |
| ciphertext:L1:c2 | 70486 | 147819855872 | 137.667969 |
| ciphertext:L2:c2 | 265580 | 835442442240 | 778.066406 |
| ciphertext:L2:c3 | 746 | 3520069632 | 3.278320 |
| ciphertext:L3:c2 | 523738 | 2196716388352 | 2,045.851562 |
| ciphertext:L3:c3 | 579 | 3642753024 | 3.392578 |
| ciphertext:L4:c2 | 95796 | 502246932480 | 467.753906 |
| ciphertext:L4:c3 | 329 | 2587361280 | 2.409668 |
| ciphertext:L5:c2 | 5691 | 35804676096 | 33.345703 |
| ciphertext:L5:c3 | 68 | 641728512 | 0.597656 |
| plaintext:L1:c1 | 296 | 310378496 | 0.289062 |
| plaintext:L2:c1 | 959 | 1508376576 | 1.404785 |
| plaintext:L3:c1 | 6228 | 13061062656 | 12.164062 |
| plaintext:L4:c1 | 18575 | 48693248000 | 45.349121 |
| plaintext:L5:c1 | 2879 | 9056550912 | 8.434570 |

## 2. 原始编译计划所有 Value 按 level 和分量分布

| 类型 / level / 分量 | 数量 | 系数字节 | GiB |
| --- | --- | --- | --- |
| ciphertext:L1:c2 | 70486 | 147819855872 | 137.667969 |
| ciphertext:L2:c2 | 265580 | 835442442240 | 778.066406 |
| ciphertext:L2:c3 | 746 | 3520069632 | 3.278320 |
| ciphertext:L3:c2 | 523738 | 2196716388352 | 2,045.851562 |
| ciphertext:L3:c3 | 579 | 3642753024 | 3.392578 |
| ciphertext:L4:c2 | 95796 | 502246932480 | 467.753906 |
| ciphertext:L4:c3 | 329 | 2587361280 | 2.409668 |
| ciphertext:L5:c2 | 5691 | 35804676096 | 33.345703 |
| ciphertext:L5:c3 | 68 | 641728512 | 0.597656 |
| plaintext:L1:c1 | 82 | 85983232 | 0.080078 |
| plaintext:L2:c1 | 330 | 519045120 | 0.483398 |
| plaintext:L3:c1 | 5655 | 11859394560 | 11.044922 |
| plaintext:L4:c1 | 11101 | 29100605440 | 27.102051 |
| plaintext:L5:c1 | 2531 | 7961837568 | 7.415039 |

## 3. 输入输出实际 level

| 输入/输出 | ValueId | level | scale log2 | 分量 | MiB |
| --- | --- | --- | --- | --- | --- |
| input 0 | 0 | 5 | 43 | 2 | 6.000 |
| input 1 | 1 | 5 | 43 | 2 | 6.000 |
| logits.0:32768 | 966186 | 2 | 43 | 2 | 3.000 |
| logits.32768:65536 | 970312 | 2 | 43 | 2 | 3.000 |
| logits.65536:98304 | 974438 | 2 | 43 | 2 | 3.000 |
| logits.98304:131072 | 978564 | 2 | 43 | 2 | 3.000 |
| logits.131072:151936 | 982690 | 2 | 43 | 2 | 3.000 |
| layer_0.kv_cache | 982691 | 2 | 43 | 2 | 3.000 |
| layer_1.kv_cache | 982692 | 2 | 43 | 2 | 3.000 |
| layer_2.kv_cache | 982693 | 2 | 43 | 2 | 3.000 |
| layer_3.kv_cache | 982694 | 2 | 43 | 2 | 3.000 |
| layer_4.kv_cache | 982695 | 2 | 43 | 2 | 3.000 |
| layer_5.kv_cache | 982696 | 2 | 43 | 2 | 3.000 |
| layer_6.kv_cache | 285715 | 2 | 43 | 2 | 3.000 |
| layer_7.kv_cache | 982697 | 2 | 43 | 2 | 3.000 |
| layer_8.kv_cache | 982698 | 2 | 43 | 2 | 3.000 |
| layer_9.kv_cache | 982699 | 2 | 43 | 2 | 3.000 |
| layer_10.kv_cache | 470874 | 2 | 43 | 2 | 3.000 |
| layer_11.kv_cache | 982700 | 2 | 43 | 2 | 3.000 |
| layer_12.kv_cache | 982701 | 2 | 43 | 2 | 3.000 |
| layer_13.kv_cache | 982702 | 2 | 43 | 2 | 3.000 |
| layer_14.kv_cache | 654423 | 2 | 43 | 2 | 3.000 |
| layer_15.kv_cache | 982703 | 2 | 43 | 2 | 3.000 |
| layer_16.kv_cache | 982704 | 2 | 43 | 2 | 3.000 |
| layer_17.kv_cache | 982705 | 2 | 43 | 2 | 3.000 |
| layer_18.kv_cache | 982706 | 2 | 43 | 2 | 3.000 |
| layer_19.kv_cache | 982707 | 2 | 43 | 2 | 3.000 |
| layer_20.kv_cache | 982708 | 2 | 43 | 2 | 3.000 |
| layer_21.kv_cache | 982709 | 2 | 43 | 2 | 3.000 |
| layer_22.kv_cache | 982710 | 2 | 43 | 2 | 3.000 |
| layer_23.kv_cache | 982711 | 2 | 43 | 2 | 3.000 |

## 4. 实际操作的 level / scale 转换

下表记录每种操作的全部输入和输出类型、level、scale、分量，次数来自最终执行计划。

| 转换 | 次数 |
| --- | --- |
| add_cc:ciphertext:L1:s43:c2,ciphertext:L1:s43:c2->ciphertext:L1:s43:c2 | 24574 |
| add_cc:ciphertext:L2:s43:c2,ciphertext:L2:s43:c2->ciphertext:L2:s43:c2 | 110191 |
| add_cc:ciphertext:L3:s43:c2,ciphertext:L3:s43:c2->ciphertext:L3:s43:c2 | 236147 |
| add_cc:ciphertext:L4:s43:c2,ciphertext:L4:s43:c2->ciphertext:L4:s43:c2 | 34306 |
| add_cp:ciphertext:L1:s43:c2,plaintext:L1:s43:c1->ciphertext:L1:s43:c2 | 599 |
| add_cp:ciphertext:L2:s43:c2,plaintext:L2:s43:c1->ciphertext:L2:s43:c2 | 354 |
| add_cp:ciphertext:L3:s43:c2,plaintext:L3:s43:c1->ciphertext:L3:s43:c2 | 98 |
| add_cp:ciphertext:L4:s43:c2,plaintext:L4:s43:c1->ciphertext:L4:s43:c2 | 95 |
| boot:ciphertext:L1:s43:c2->ciphertext:L5:s43:c2 | 2370 |
| encode:->plaintext:L1:s43:c1 | 296 |
| encode:->plaintext:L2:s43:c1 | 959 |
| encode:->plaintext:L3:s43:c1 | 6228 |
| encode:->plaintext:L4:s43:c1 | 18575 |
| encode:->plaintext:L5:s43:c1 | 2879 |
| mod_switch:ciphertext:L2:s43:c2->ciphertext:L1:s43:c2 | 112 |
| mod_switch:ciphertext:L3:s43:c2->ciphertext:L1:s43:c2 | 17 |
| mod_switch:ciphertext:L3:s43:c2->ciphertext:L2:s43:c2 | 152 |
| mod_switch:ciphertext:L4:s43:c2->ciphertext:L1:s43:c2 | 3 |
| mod_switch:ciphertext:L4:s43:c2->ciphertext:L2:s43:c2 | 19 |
| mod_switch:ciphertext:L4:s43:c2->ciphertext:L3:s43:c2 | 43 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L1:s43:c2 | 177 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L2:s43:c2 | 527 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L3:s43:c2 | 667 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L4:s43:c2 | 1162 |
| mul_cc:ciphertext:L2:s43:c2,ciphertext:L2:s43:c2->ciphertext:L2:s86:c3 | 746 |
| mul_cc:ciphertext:L3:s43:c2,ciphertext:L3:s43:c2->ciphertext:L3:s86:c3 | 579 |
| mul_cc:ciphertext:L4:s43:c2,ciphertext:L4:s43:c2->ciphertext:L4:s86:c3 | 329 |
| mul_cc:ciphertext:L5:s43:c2,ciphertext:L5:s43:c2->ciphertext:L5:s86:c3 | 68 |
| mul_cp:ciphertext:L2:s43:c2,plaintext:L2:s43:c1->ciphertext:L2:s86:c2 | 20209 |
| mul_cp:ciphertext:L3:s43:c2,plaintext:L3:s43:c1->ciphertext:L3:s86:c2 | 27790 |
| mul_cp:ciphertext:L4:s43:c2,plaintext:L4:s43:c1->ciphertext:L4:s86:c2 | 22121 |
| mul_cp:ciphertext:L5:s43:c2,plaintext:L5:s43:c1->ciphertext:L5:s86:c2 | 3245 |
| negate:ciphertext:L2:s43:c2->ciphertext:L2:s43:c2 | 19 |
| negate:ciphertext:L3:s43:c2->ciphertext:L3:s43:c2 | 6 |
| negate:ciphertext:L4:s43:c2->ciphertext:L4:s43:c2 | 23 |
| relinearize:ciphertext:L2:s86:c3->ciphertext:L2:s86:c2 | 746 |
| relinearize:ciphertext:L3:s86:c3->ciphertext:L3:s86:c2 | 579 |
| relinearize:ciphertext:L4:s86:c3->ciphertext:L4:s86:c2 | 329 |
| relinearize:ciphertext:L5:s86:c3->ciphertext:L5:s86:c2 | 68 |
| rescale:ciphertext:L2:s86:c2->ciphertext:L1:s43:c2 | 20955 |
| rescale:ciphertext:L3:s86:c2->ciphertext:L2:s43:c2 | 28369 |
| rescale:ciphertext:L4:s86:c2->ciphertext:L3:s43:c2 | 22450 |
| rescale:ciphertext:L5:s86:c2->ciphertext:L4:s43:c2 | 3313 |
| rotate:ciphertext:L1:s43:c2->ciphertext:L1:s43:c2 | 24049 |
| rotate:ciphertext:L2:s43:c2->ciphertext:L2:s43:c2 | 104994 |
| rotate:ciphertext:L3:s43:c2->ciphertext:L3:s43:c2 | 235958 |
| rotate:ciphertext:L4:s43:c2->ciphertext:L4:s43:c2 | 34447 |
| rotate:ciphertext:L5:s43:c2->ciphertext:L5:s43:c2 | 6 |

## 5. 每个源码阶段产生的 Value

这是累计产生数量及全保留系数容量，不是该阶段的峰值。归属以编译器保存的调试位置为准；融合位置合并显示，unattributed 表示没有足够来源信息，不强行分摊。缓存重编码用相同 Encode 元数据和 payload 关联原来源。

| 来源阶段 | 类型 / level / 分量 | 数量 | 系数 MiB |
| --- | --- | --- | --- |
| external_input | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/gqa_attention | ciphertext:L3:c2 | 22 | 88.000 |
| token_0/layer_00/gqa_attention | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_01/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_01/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_02/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_03/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_03/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_04/gqa_attention | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_05/gqa_attention | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_05/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_06/gqa_attention | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_06/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_06/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_07/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_07/gqa_attention | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_08/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_09/gqa_attention | ciphertext:L1:c2 | 3 | 6.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_09/gqa_attention | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_10/gqa_attention | ciphertext:L1:c2 | 3 | 6.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_10/gqa_attention | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_11/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_11/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_12/gqa_attention | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_12/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_13/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_14/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_15/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_16/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_16/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_17/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_18/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_19/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_19/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_20/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_21/gqa_attention | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_21/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/gqa_attention \| token_0/layer_22/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_00/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_00/gqa_attention | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_00/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_01/gqa_attention | ciphertext:L2:c2 | 9 | 27.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_01/gqa_attention | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_02/gqa_attention | ciphertext:L2:c2 | 11 | 33.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_02/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_03/gqa_attention | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_04/gqa_attention | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_05/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_05/gqa_attention | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_05/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_06/gqa_attention | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_06/gqa_attention | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_06/gqa_attention | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_07/gqa_attention | ciphertext:L1:c2 | 11 | 22.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_07/gqa_attention | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_08/gqa_attention | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_09/gqa_attention | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_09/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_10/gqa_attention | ciphertext:L2:c2 | 11 | 33.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_10/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_11/gqa_attention | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_12/gqa_attention | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_12/gqa_attention | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_12/gqa_attention | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_13/gqa_attention | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_13/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_13/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_14/gqa_attention | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_14/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_15/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_15/gqa_attention | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_15/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_16/gqa_attention | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_17/gqa_attention | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_17/gqa_attention | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_17/gqa_attention | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/gqa_attention \| token_1/layer_18/gqa_attention | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_19/gqa_attention | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_19/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_20/gqa_attention | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_21/gqa_attention | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_22/gqa_attention | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_23/gqa_attention | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_23/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/gqa_attention \| token_1/layer_23/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L3:c2 | 75 | 300.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L4:c2 | 82 | 410.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight | plaintext:L5:c1 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_07/model_list.7.attention.v_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_19/model_list.19.attention.v_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_0/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_14/model_list.14.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.k_weight.weight \| token_1/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L1:c2 | 6 | 12.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L3:c2 | 972 | 3,888.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c2 | 46 | 230.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | plaintext:L2:c1 | 3 | 4.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_05/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 13 | 19.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_05/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 14 | 28.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_05/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_0/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L3:c2 | 29 | 116.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_09/model_list.9.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 15 | 22.500 |
| token_0/layer_00/model_list.0.attention.o_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 8 | 16.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L1:c2 | 77 | 154.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L2:c2 | 220 | 660.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L3:c2 | 27 | 108.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L4:c2 | 195 | 975.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | ciphertext:L5:c2 | 10 | 60.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu | plaintext:L5:c1 | 7 | 21.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L1:c1 | 16 | 16.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_00/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 21 | 31.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/model_list.1.attention.o_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/model_list.1.attention.v_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/swiglu | ciphertext:L2:c2 | 421 | 1,263.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_01/swiglu | ciphertext:L3:c2 | 119 | 476.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L1:c2 | 13 | 26.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L2:c2 | 37 | 111.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/swiglu | ciphertext:L2:c2 | 55 | 165.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/swiglu | ciphertext:L3:c2 | 463 | 1,852.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_02/swiglu | ciphertext:L4:c2 | 22 | 110.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/model_list.3.attention.o_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | ciphertext:L1:c2 | 35 | 70.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | ciphertext:L2:c2 | 210 | 630.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | ciphertext:L2:c3 | 2 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | ciphertext:L3:c2 | 260 | 1,040.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | ciphertext:L4:c2 | 16 | 80.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | ciphertext:L5:c2 | 4 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu | plaintext:L4:c1 | 11 | 27.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu | plaintext:L1:c1 | 12 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu | plaintext:L2:c1 | 9 | 13.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 44 | 176.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/swiglu | ciphertext:L2:c2 | 141 | 423.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/swiglu | ciphertext:L3:c2 | 374 | 1,496.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/swiglu | ciphertext:L4:c2 | 14 | 70.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_04/swiglu | plaintext:L4:c1 | 11 | 27.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/model_list.5.attention.k_weight.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L2:c2 | 46 | 138.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/model_list.5.attention.o_weight.weight | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/swiglu | ciphertext:L3:c2 | 533 | 2,132.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/swiglu | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu | plaintext:L2:c1 | 22 | 33.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/model_list.6.attention.o_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/swiglu | ciphertext:L2:c2 | 89 | 267.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/swiglu | ciphertext:L3:c2 | 426 | 1,704.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/swiglu | ciphertext:L4:c2 | 23 | 115.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_06/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/model_list.7.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L2:c2 | 50 | 150.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | ciphertext:L1:c2 | 9 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | ciphertext:L2:c2 | 30 | 90.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | ciphertext:L3:c2 | 479 | 1,916.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | ciphertext:L4:c2 | 17 | 85.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_07/swiglu | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/model_list.8.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/model_list.8.attention.o_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/swiglu | ciphertext:L1:c2 | 25 | 50.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/swiglu | ciphertext:L2:c2 | 170 | 510.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/swiglu | ciphertext:L3:c2 | 315 | 1,260.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/swiglu | ciphertext:L4:c2 | 14 | 70.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_08/swiglu | plaintext:L4:c1 | 16 | 40.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L1:c2 | 6 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L3:c2 | 48 | 192.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/swiglu | ciphertext:L1:c2 | 77 | 154.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/swiglu | ciphertext:L2:c2 | 116 | 348.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/swiglu | ciphertext:L3:c2 | 324 | 1,296.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/swiglu | ciphertext:L4:c2 | 10 | 50.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_09/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L3:c2 | 38 | 152.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L1:c2 | 3 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_10/swiglu | ciphertext:L3:c2 | 540 | 2,160.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L1:c2 | 18 | 36.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 30 | 90.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L1:c2 | 13 | 26.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L3:c2 | 453 | 1,812.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L4:c2 | 56 | 280.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_11/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L3:c2 | 43 | 172.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/model_list.12.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L1:c2 | 57 | 114.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L2:c2 | 121 | 363.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L2:c3 | 5 | 22.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L3:c2 | 310 | 1,240.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L4:c2 | 33 | 165.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L5:c2 | 7 | 42.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_12/swiglu | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L1:c2 | 9 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L4:c2 | 29 | 145.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.o_weight.weight | plaintext:L5:c1 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/swiglu | ciphertext:L2:c2 | 131 | 393.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/swiglu | ciphertext:L3:c2 | 382 | 1,528.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/swiglu | ciphertext:L4:c2 | 14 | 70.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_13/swiglu | plaintext:L4:c1 | 26 | 65.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L2:c2 | 50 | 150.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_14/swiglu | ciphertext:L2:c2 | 361 | 1,083.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_14/swiglu | ciphertext:L3:c2 | 179 | 716.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 50 | 150.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/swiglu | ciphertext:L1:c2 | 25 | 50.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/swiglu | ciphertext:L2:c2 | 110 | 330.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/swiglu | ciphertext:L3:c2 | 369 | 1,476.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/swiglu | ciphertext:L3:c3 | 2 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/swiglu | ciphertext:L4:c2 | 18 | 90.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_15/swiglu | plaintext:L4:c1 | 15 | 37.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 43 | 172.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/model_list.16.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/model_list.16.attention.v_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L1:c2 | 71 | 142.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L2:c2 | 207 | 621.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L2:c3 | 2 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L3:c2 | 233 | 932.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L3:c3 | 4 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L4:c2 | 10 | 50.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | ciphertext:L5:c2 | 4 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_16/swiglu | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L3:c2 | 23 | 92.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/model_list.17.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | ciphertext:L1:c2 | 30 | 60.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | ciphertext:L2:c2 | 354 | 1,062.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | ciphertext:L3:c2 | 141 | 564.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | plaintext:L3:c1 | 5 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_17/swiglu | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/model_list.18.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/swiglu | ciphertext:L1:c2 | 78 | 156.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/swiglu | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/swiglu | ciphertext:L3:c2 | 443 | 1,772.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/swiglu | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_18/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L2:c2 | 50 | 150.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/swiglu | ciphertext:L3:c2 | 493 | 1,972.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/swiglu | ciphertext:L4:c2 | 34 | 170.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/swiglu | plaintext:L4:c1 | 22 | 55.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_19/swiglu | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/model_list.20.attention.k_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/model_list.20.attention.o_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/swiglu | ciphertext:L1:c2 | 85 | 170.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/swiglu | ciphertext:L2:c2 | 140 | 420.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/swiglu | ciphertext:L3:c2 | 295 | 1,180.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/swiglu | ciphertext:L4:c2 | 13 | 65.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_20/swiglu | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 45 | 180.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/model_list.21.attention.o_weight.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/swiglu | ciphertext:L2:c2 | 169 | 507.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/swiglu | ciphertext:L3:c2 | 347 | 1,388.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_21/swiglu | plaintext:L4:c1 | 24 | 60.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 43 | 172.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/swiglu | ciphertext:L1:c2 | 133 | 266.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/swiglu | ciphertext:L2:c2 | 173 | 519.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/swiglu | ciphertext:L3:c2 | 232 | 928.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/swiglu | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_22/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_0/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L1:c2 | 24 | 48.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L2:c2 | 26 | 78.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L3:c2 | 70 | 280.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/swiglu | ciphertext:L3:c2 | 446 | 1,784.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/swiglu | ciphertext:L4:c2 | 80 | 400.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/swiglu | plaintext:L4:c1 | 15 | 37.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_00/swiglu | plaintext:L5:c1 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L1:c2 | 25 | 50.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 27 | 108.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L2:c2 | 43 | 129.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/swiglu | ciphertext:L2:c2 | 420 | 1,260.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/swiglu | ciphertext:L3:c2 | 109 | 436.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/swiglu | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_01/swiglu | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.o_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.q_weight.weight | ciphertext:L2:c2 | 47 | 141.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.q_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/swiglu | ciphertext:L1:c2 | 7 | 14.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/swiglu | ciphertext:L2:c2 | 202 | 606.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_02/swiglu | ciphertext:L3:c2 | 331 | 1,324.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 48 | 192.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L4:c2 | 44 | 220.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L5:c2 | 6 | 36.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | ciphertext:L1:c2 | 75 | 150.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | ciphertext:L2:c2 | 227 | 681.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | ciphertext:L3:c2 | 203 | 812.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | ciphertext:L3:c3 | 3 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | ciphertext:L4:c2 | 15 | 75.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | ciphertext:L4:c3 | 3 | 22.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | ciphertext:L5:c2 | 6 | 36.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_03/swiglu | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.k_weight.weight | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L2:c2 | 45 | 135.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L2:c2 | 48 | 144.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.q_weight.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/swiglu | ciphertext:L3:c2 | 498 | 1,992.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/swiglu | ciphertext:L4:c2 | 21 | 105.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_04/swiglu | plaintext:L4:c1 | 42 | 105.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/swiglu | ciphertext:L2:c2 | 116 | 348.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/swiglu | ciphertext:L3:c2 | 392 | 1,568.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/swiglu | ciphertext:L4:c2 | 16 | 80.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_05/swiglu | plaintext:L4:c1 | 32 | 80.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.o_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.q_weight.weight | ciphertext:L3:c2 | 48 | 192.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/swiglu | ciphertext:L1:c2 | 40 | 80.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/swiglu | ciphertext:L2:c2 | 118 | 354.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/swiglu | ciphertext:L3:c2 | 363 | 1,452.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/swiglu | ciphertext:L4:c2 | 18 | 90.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_06/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L1:c2 | 6 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L3:c2 | 47 | 188.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L1:c2 | 33 | 66.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L2:c2 | 16 | 48.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | ciphertext:L1:c2 | 116 | 232.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | ciphertext:L2:c2 | 131 | 393.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | ciphertext:L3:c2 | 258 | 1,032.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | ciphertext:L4:c2 | 24 | 120.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | ciphertext:L5:c2 | 6 | 36.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_07/swiglu | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 38 | 152.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L1:c2 | 7 | 14.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L2:c2 | 16 | 48.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | ciphertext:L1:c2 | 55 | 110.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | ciphertext:L3:c2 | 435 | 1,740.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | ciphertext:L4:c2 | 19 | 95.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_08/swiglu | plaintext:L4:c1 | 26 | 65.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L1:c2 | 37 | 74.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L2:c2 | 13 | 39.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L3:c2 | 36 | 144.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/swiglu | ciphertext:L3:c2 | 297 | 1,188.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/swiglu | ciphertext:L4:c2 | 214 | 1,070.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/swiglu | ciphertext:L5:c2 | 9 | 54.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/swiglu | plaintext:L4:c1 | 23 | 57.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_09/swiglu | plaintext:L5:c1 | 8 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L1:c2 | 35 | 70.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 15 | 45.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L2:c2 | 23 | 69.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/swiglu | ciphertext:L1:c2 | 143 | 286.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/swiglu | ciphertext:L2:c2 | 243 | 729.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/swiglu | ciphertext:L3:c2 | 144 | 576.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/swiglu | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_10/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L1:c2 | 36 | 72.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 14 | 42.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L2:c2 | 46 | 138.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/swiglu | ciphertext:L1:c2 | 32 | 64.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/swiglu | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/swiglu | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/swiglu | ciphertext:L4:c2 | 466 | 2,330.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_11/swiglu | ciphertext:L5:c2 | 35 | 210.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L3:c2 | 49 | 196.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L2:c2 | 47 | 141.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | ciphertext:L1:c2 | 49 | 98.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | ciphertext:L2:c2 | 89 | 267.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | ciphertext:L3:c2 | 372 | 1,488.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | ciphertext:L4:c2 | 13 | 65.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_12/swiglu | plaintext:L4:c1 | 28 | 70.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c2 | 25 | 75.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L3:c2 | 49 | 196.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/swiglu | ciphertext:L2:c2 | 122 | 366.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/swiglu | ciphertext:L3:c2 | 288 | 1,152.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/swiglu | ciphertext:L4:c2 | 123 | 615.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_13/swiglu | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L3:c2 | 43 | 172.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/swiglu | ciphertext:L3:c2 | 531 | 2,124.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/swiglu | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_14/swiglu | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L4:c2 | 21 | 105.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.o_weight.weight | plaintext:L5:c1 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.q_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/swiglu | ciphertext:L1:c2 | 15 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/swiglu | ciphertext:L2:c2 | 72 | 216.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/swiglu | ciphertext:L3:c2 | 186 | 744.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/swiglu | ciphertext:L4:c2 | 246 | 1,230.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/swiglu | ciphertext:L5:c2 | 11 | 66.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/swiglu | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_15/swiglu | plaintext:L5:c1 | 8 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 47 | 188.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.q_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/swiglu | ciphertext:L1:c2 | 37 | 74.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/swiglu | ciphertext:L2:c2 | 238 | 714.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/swiglu | ciphertext:L3:c2 | 247 | 988.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_16/swiglu | ciphertext:L4:c2 | 17 | 85.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L1:c2 | 22 | 44.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L3:c2 | 23 | 92.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/swiglu | ciphertext:L3:c2 | 377 | 1,508.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/swiglu | ciphertext:L4:c2 | 136 | 680.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/swiglu | ciphertext:L5:c2 | 5 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/swiglu | plaintext:L4:c1 | 34 | 85.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_17/swiglu | plaintext:L5:c1 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 47 | 188.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/swiglu | ciphertext:L2:c2 | 229 | 687.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/swiglu | ciphertext:L3:c2 | 295 | 1,180.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_18/swiglu | ciphertext:L4:c2 | 16 | 80.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L3:c2 | 44 | 176.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.q_weight.weight | ciphertext:L3:c2 | 48 | 192.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/swiglu | ciphertext:L1:c2 | 88 | 176.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/swiglu | ciphertext:L2:c2 | 184 | 552.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/swiglu | ciphertext:L3:c2 | 253 | 1,012.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_19/swiglu | ciphertext:L4:c2 | 15 | 75.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_20/swiglu | ciphertext:L1:c2 | 48 | 96.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_20/swiglu | ciphertext:L2:c2 | 243 | 729.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_20/swiglu | ciphertext:L3:c2 | 249 | 996.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 48 | 192.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L3:c2 | 44 | 176.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/swiglu | ciphertext:L2:c2 | 217 | 651.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/swiglu | ciphertext:L3:c2 | 306 | 1,224.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_21/swiglu | ciphertext:L4:c2 | 17 | 85.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/model_list.22.attention.q_weight.weight | ciphertext:L3:c2 | 50 | 200.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/swiglu | ciphertext:L2:c2 | 254 | 762.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/swiglu | ciphertext:L3:c2 | 269 | 1,076.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/swiglu | ciphertext:L4:c2 | 9 | 45.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_22/swiglu | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.v_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/swiglu | ciphertext:L2:c2 | 136 | 408.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/swiglu | ciphertext:L3:c2 | 377 | 1,508.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/swiglu | ciphertext:L4:c2 | 13 | 65.000 |
| token_0/layer_00/model_list.0.attention.q_weight.weight \| token_1/layer_23/swiglu | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L3:c2 | 153 | 612.000 |
| token_0/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_00/model_list.0.attention.v_weight.weight \| token_1/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/model_list.0.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.post_Normal.weight | ciphertext:L3:c2 | 108 | 432.000 |
| token_0/layer_00/model_list.0.post_Normal.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L1:c1 | 15 | 15.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L2:c1 | 10 | 15.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu | plaintext:L2:c1 | 16 | 24.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu | plaintext:L3:c1 | 18 | 36.000 |
| token_0/layer_00/model_list.0.post_Normal.weight \| token_1/layer_00/model_list.0.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L4:c2 | 108 | 540.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L1:c1 | 7 | 7.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_00/model_list.0.pre_Normal.weight | plaintext:L5:c1 | 5 | 15.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | plaintext:L1:c1 | 31 | 31.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | plaintext:L2:c1 | 20 | 30.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | plaintext:L3:c1 | 14 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | plaintext:L4:c1 | 41 | 102.500 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_01/model_list.1.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_01/model_list.1.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_01/model_list.1.pre_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_01/model_list.1.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_02/model_list.2.post_Normal.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_02/model_list.2.post_Normal.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_02/model_list.2.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_02/model_list.2.pre_Normal.weight | ciphertext:L2:c2 | 9 | 27.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_02/model_list.2.pre_Normal.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_03/model_list.3.post_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_03/model_list.3.post_Normal.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_03/model_list.3.post_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_03/model_list.3.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_03/model_list.3.pre_Normal.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_04/model_list.4.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_04/model_list.4.pre_Normal.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/model_list.5.post_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/model_list.5.pre_Normal.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/model_list.5.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/model_list.5.pre_Normal.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu | plaintext:L1:c1 | 22 | 22.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu | plaintext:L2:c1 | 28 | 42.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu | plaintext:L3:c1 | 21 | 42.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu \| token_0/layer_07/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 9 | 13.500 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu \| token_0/layer_07/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 9 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu \| token_0/layer_07/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 13 | 32.500 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_05/swiglu \| token_0/layer_07/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L5:c1 | 3 | 9.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_06/model_list.6.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_06/model_list.6.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_06/model_list.6.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_07/model_list.7.post_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_07/model_list.7.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_07/model_list.7.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_08/model_list.8.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_08/model_list.8.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_08/model_list.8.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_09/model_list.9.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_09/model_list.9.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_09/model_list.9.pre_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_09/model_list.9.pre_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_09/model_list.9.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_10/model_list.10.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_10/model_list.10.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_10/model_list.10.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_10/model_list.10.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_11/model_list.11.post_Normal.weight | ciphertext:L1:c2 | 7 | 14.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_11/model_list.11.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_12/model_list.12.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_12/model_list.12.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_12/model_list.12.pre_Normal.weight | ciphertext:L1:c2 | 6 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_12/model_list.12.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_12/model_list.12.pre_Normal.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_13/model_list.13.post_Normal.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_13/model_list.13.post_Normal.weight | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_13/model_list.13.pre_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_14/model_list.14.post_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_14/model_list.14.post_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_14/model_list.14.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_14/model_list.14.pre_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_15/model_list.15.post_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_15/model_list.15.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_16/model_list.16.post_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_16/model_list.16.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_16/model_list.16.pre_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_16/model_list.16.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_17/model_list.17.post_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_17/model_list.17.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_17/model_list.17.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_17/model_list.17.pre_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_18/model_list.18.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_18/model_list.18.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_19/model_list.19.post_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_19/model_list.19.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_19/model_list.19.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_20/model_list.20.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_20/model_list.20.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_20/model_list.20.pre_Normal.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_20/model_list.20.pre_Normal.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_21/model_list.21.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_21/model_list.21.post_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_21/model_list.21.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_21/model_list.21.pre_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_21/model_list.21.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_22/model_list.22.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_22/model_list.22.post_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_22/model_list.22.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_22/model_list.22.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_23/model_list.23.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_0/layer_23/model_list.23.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_00/gqa_attention | plaintext:L1:c1 | 16 | 16.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_00/gqa_attention | plaintext:L2:c1 | 19 | 28.500 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_00/model_list.0.post_Normal.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_00/model_list.0.post_Normal.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_00/model_list.0.post_Normal.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_00/model_list.0.pre_Normal.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_00/model_list.0.pre_Normal.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_01/model_list.1.post_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_01/model_list.1.post_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_01/model_list.1.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_01/model_list.1.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_02/model_list.2.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_02/model_list.2.pre_Normal.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_02/model_list.2.pre_Normal.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_03/model_list.3.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_03/model_list.3.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_03/model_list.3.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_04/model_list.4.post_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_04/model_list.4.pre_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_05/model_list.5.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_05/model_list.5.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_06/model_list.6.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_06/model_list.6.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_06/model_list.6.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_07/model_list.7.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_07/model_list.7.pre_Normal.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_07/model_list.7.pre_Normal.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_08/model_list.8.post_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_08/model_list.8.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_08/model_list.8.pre_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_08/model_list.8.pre_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_09/model_list.9.pre_Normal.weight | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_09/model_list.9.pre_Normal.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_10/model_list.10.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_10/model_list.10.post_Normal.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_10/model_list.10.post_Normal.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_10/model_list.10.pre_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_10/model_list.10.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_11/model_list.11.pre_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_11/model_list.11.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_12/model_list.12.post_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_12/model_list.12.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_12/model_list.12.pre_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_12/model_list.12.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_13/model_list.13.post_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_13/model_list.13.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_13/model_list.13.pre_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_14/model_list.14.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_14/model_list.14.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_14/model_list.14.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_15/model_list.15.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_15/model_list.15.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_15/model_list.15.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_15/model_list.15.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_16/model_list.16.post_Normal.weight | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_16/model_list.16.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_16/model_list.16.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_17/model_list.17.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_17/model_list.17.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_17/model_list.17.pre_Normal.weight | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_18/model_list.18.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_18/model_list.18.pre_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_18/model_list.18.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_19/model_list.19.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_19/model_list.19.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_19/model_list.19.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_20/model_list.20.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_20/model_list.20.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_20/model_list.20.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_21/model_list.21.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_21/model_list.21.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_21/model_list.21.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_22/model_list.22.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_22/model_list.22.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_23/model_list.23.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_23/model_list.23.post_Normal.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_23/model_list.23.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_1/layer_23/model_list.23.pre_Normal.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_shared/layer_shared/final_normal.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_shared/layer_shared/final_normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/model_list.0.pre_Normal.weight \| token_shared/layer_shared/final_normal.weight | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_00/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope | ciphertext:L3:c2 | 14 | 56.000 |
| token_0/layer_00/rope \| token_0/layer_01/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_0/layer_02/rope | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/rope \| token_0/layer_02/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/rope \| token_0/layer_03/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_03/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/rope \| token_0/layer_04/rope | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_00/rope \| token_0/layer_05/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_06/rope | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/rope \| token_0/layer_06/rope | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/rope \| token_0/layer_06/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/rope \| token_0/layer_07/rope | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/rope \| token_0/layer_07/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_08/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_08/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/rope \| token_0/layer_09/rope | ciphertext:L1:c2 | 3 | 6.000 |
| token_0/layer_00/rope \| token_0/layer_09/rope | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/rope \| token_0/layer_10/rope | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/rope \| token_0/layer_10/rope | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/rope \| token_0/layer_11/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_0/layer_11/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/rope \| token_0/layer_12/rope | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/rope \| token_0/layer_12/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/rope \| token_0/layer_13/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_14/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_15/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_16/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_0/layer_17/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_0/layer_18/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_0/layer_19/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_0/layer_20/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_0/layer_21/rope | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/rope \| token_0/layer_21/rope | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/rope \| token_0/layer_22/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_0/layer_23/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_00/rope | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/rope \| token_1/layer_00/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_01/rope | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/rope \| token_1/layer_01/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/rope \| token_1/layer_02/rope | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/rope \| token_1/layer_02/rope | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/rope \| token_1/layer_03/rope | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_00/rope \| token_1/layer_04/rope | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/rope \| token_1/layer_04/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/rope \| token_1/layer_05/rope | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_00/rope \| token_1/layer_05/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_06/rope | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_00/rope \| token_1/layer_07/rope | ciphertext:L1:c2 | 3 | 6.000 |
| token_0/layer_00/rope \| token_1/layer_07/rope | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/rope \| token_1/layer_08/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_09/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_10/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_1/layer_11/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_1/layer_12/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/rope \| token_1/layer_13/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_14/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_15/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_16/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/rope \| token_1/layer_16/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/rope \| token_1/layer_17/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/rope \| token_1/layer_17/rope | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/rope \| token_1/layer_18/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_19/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_20/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_21/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_1/layer_22/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/rope \| token_1/layer_22/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/rope \| token_1/layer_23/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/rope \| token_shared/layer_shared/pack_kv | ciphertext:L1:c2 | 9 | 18.000 |
| token_0/layer_00/rope \| token_shared/layer_shared/pack_kv | ciphertext:L2:c2 | 27 | 81.000 |
| token_0/layer_00/rope \| token_shared/layer_shared/pack_kv | ciphertext:L3:c2 | 52 | 208.000 |
| token_0/layer_00/rope \| token_shared/layer_shared/pack_kv | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_00/rope \| token_shared/layer_shared/pack_kv | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_00/rope \| token_shared/layer_shared/pack_kv | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_00/swiglu | ciphertext:L1:c2 | 1023 | 2,046.000 |
| token_0/layer_00/swiglu | ciphertext:L2:c2 | 4196 | 12,588.000 |
| token_0/layer_00/swiglu | ciphertext:L2:c3 | 27 | 121.500 |
| token_0/layer_00/swiglu | ciphertext:L3:c2 | 8688 | 34,752.000 |
| token_0/layer_00/swiglu | ciphertext:L3:c3 | 20 | 120.000 |
| token_0/layer_00/swiglu | ciphertext:L4:c2 | 3230 | 16,150.000 |
| token_0/layer_00/swiglu | ciphertext:L4:c3 | 11 | 82.500 |
| token_0/layer_00/swiglu | ciphertext:L5:c2 | 168 | 1,008.000 |
| token_0/layer_00/swiglu | ciphertext:L5:c3 | 3 | 27.000 |
| token_0/layer_00/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_00/swiglu | plaintext:L2:c1 | 8 | 12.000 |
| token_0/layer_00/swiglu | plaintext:L3:c1 | 8 | 16.000 |
| token_0/layer_00/swiglu | plaintext:L4:c1 | 233 | 582.500 |
| token_0/layer_00/swiglu | plaintext:L5:c1 | 118 | 354.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | ciphertext:L1:c2 | 7 | 14.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | ciphertext:L2:c2 | 83 | 249.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | ciphertext:L3:c2 | 222 | 888.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_03/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 10 | 15.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_03/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 7 | 14.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_03/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 11 | 27.500 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_03/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L5:c1 | 12 | 36.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/gqa_attention | plaintext:L2:c1 | 32 | 48.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/gqa_attention | plaintext:L3:c1 | 20 | 40.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/gqa_attention | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/gqa_attention | plaintext:L5:c1 | 39 | 117.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 27 | 40.500 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 14 | 28.000 |
| token_0/layer_00/swiglu \| token_0/layer_01/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 31 | 77.500 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu | ciphertext:L4:c2 | 276 | 1,380.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu | ciphertext:L5:c2 | 12 | 72.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu | plaintext:L5:c1 | 24 | 72.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_0/layer_03/swiglu \| token_0/layer_14/swiglu | plaintext:L2:c1 | 8 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_0/layer_03/swiglu \| token_0/layer_14/swiglu | plaintext:L3:c1 | 23 | 46.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_0/layer_03/swiglu \| token_0/layer_14/swiglu | plaintext:L4:c1 | 13 | 32.500 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_0/layer_03/swiglu \| token_0/layer_14/swiglu | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_1/layer_04/swiglu \| token_1/layer_05/swiglu | plaintext:L2:c1 | 6 | 9.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_1/layer_04/swiglu \| token_1/layer_05/swiglu | plaintext:L3:c1 | 6 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_1/layer_04/swiglu \| token_1/layer_05/swiglu | plaintext:L4:c1 | 46 | 115.000 |
| token_0/layer_00/swiglu \| token_0/layer_02/swiglu \| token_1/layer_04/swiglu \| token_1/layer_05/swiglu | plaintext:L5:c1 | 2 | 6.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu | ciphertext:L1:c2 | 54 | 108.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu | ciphertext:L2:c2 | 70 | 210.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu | ciphertext:L4:c2 | 180 | 900.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu \| token_1/layer_04/swiglu | plaintext:L2:c1 | 27 | 40.500 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu \| token_1/layer_04/swiglu | plaintext:L3:c1 | 28 | 56.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu \| token_1/layer_04/swiglu | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu \| token_1/layer_04/swiglu | plaintext:L5:c1 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu \| token_1/layer_05/swiglu \| token_1/layer_09/model_list.9.attention.q_weight.weight | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu \| token_1/layer_05/swiglu \| token_1/layer_09/model_list.9.attention.q_weight.weight | plaintext:L3:c1 | 16 | 32.000 |
| token_0/layer_00/swiglu \| token_0/layer_03/swiglu \| token_1/layer_05/swiglu \| token_1/layer_09/model_list.9.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L2:c1 | 13 | 19.500 |
| token_0/layer_00/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L3:c1 | 7 | 14.000 |
| token_0/layer_00/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L4:c1 | 22 | 55.000 |
| token_0/layer_00/swiglu \| token_0/layer_04/swiglu | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_04/swiglu | ciphertext:L3:c2 | 285 | 1,140.000 |
| token_0/layer_00/swiglu \| token_0/layer_04/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_04/swiglu | plaintext:L4:c1 | 26 | 65.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu | ciphertext:L4:c2 | 280 | 1,400.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu | ciphertext:L5:c2 | 10 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu | plaintext:L5:c1 | 10 | 30.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu | plaintext:L2:c1 | 57 | 85.500 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu | plaintext:L3:c1 | 64 | 128.000 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu | plaintext:L4:c1 | 11 | 27.500 |
| token_0/layer_00/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu | plaintext:L5:c1 | 2 | 6.000 |
| token_0/layer_00/swiglu \| token_0/layer_06/swiglu | ciphertext:L1:c2 | 16 | 32.000 |
| token_0/layer_00/swiglu \| token_0/layer_06/swiglu | ciphertext:L2:c2 | 56 | 168.000 |
| token_0/layer_00/swiglu \| token_0/layer_06/swiglu | ciphertext:L3:c2 | 224 | 896.000 |
| token_0/layer_00/swiglu \| token_0/layer_06/swiglu | ciphertext:L4:c2 | 10 | 50.000 |
| token_0/layer_00/swiglu \| token_0/layer_06/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/swiglu \| token_0/layer_06/swiglu | plaintext:L4:c1 | 7 | 17.500 |
| token_0/layer_00/swiglu \| token_0/layer_07/swiglu | ciphertext:L1:c2 | 199 | 398.000 |
| token_0/layer_00/swiglu \| token_0/layer_07/swiglu | ciphertext:L2:c2 | 85 | 255.000 |
| token_0/layer_00/swiglu \| token_0/layer_07/swiglu | ciphertext:L3:c2 | 26 | 104.000 |
| token_0/layer_00/swiglu \| token_0/layer_07/swiglu | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_0/layer_07/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_08/swiglu | ciphertext:L2:c2 | 287 | 861.000 |
| token_0/layer_00/swiglu \| token_0/layer_08/swiglu | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_00/swiglu \| token_0/layer_08/swiglu | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/swiglu \| token_0/layer_09/swiglu | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/swiglu \| token_0/layer_09/swiglu | ciphertext:L3:c2 | 287 | 1,148.000 |
| token_0/layer_00/swiglu \| token_0/layer_09/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_09/swiglu | plaintext:L4:c1 | 24 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L1:c2 | 42 | 84.000 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L2:c2 | 25 | 75.000 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L3:c2 | 229 | 916.000 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L4:c2 | 11 | 55.000 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_10/swiglu | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/swiglu \| token_0/layer_11/swiglu | ciphertext:L1:c2 | 114 | 228.000 |
| token_0/layer_00/swiglu \| token_0/layer_11/swiglu | ciphertext:L2:c2 | 159 | 477.000 |
| token_0/layer_00/swiglu \| token_0/layer_11/swiglu | ciphertext:L3:c2 | 35 | 140.000 |
| token_0/layer_00/swiglu \| token_0/layer_11/swiglu | plaintext:L2:c1 | 5 | 7.500 |
| token_0/layer_00/swiglu \| token_0/layer_12/swiglu | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/swiglu \| token_0/layer_12/swiglu | ciphertext:L2:c2 | 278 | 834.000 |
| token_0/layer_00/swiglu \| token_0/layer_12/swiglu | ciphertext:L3:c2 | 34 | 136.000 |
| token_0/layer_00/swiglu \| token_0/layer_13/swiglu | ciphertext:L2:c2 | 29 | 87.000 |
| token_0/layer_00/swiglu \| token_0/layer_13/swiglu | ciphertext:L3:c2 | 262 | 1,048.000 |
| token_0/layer_00/swiglu \| token_0/layer_13/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_13/swiglu | plaintext:L4:c1 | 20 | 50.000 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | ciphertext:L1:c2 | 13 | 26.000 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | ciphertext:L2:c2 | 82 | 246.000 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | ciphertext:L3:c2 | 191 | 764.000 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | ciphertext:L4:c2 | 11 | 55.000 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | plaintext:L1:c1 | 3 | 3.000 |
| token_0/layer_00/swiglu \| token_0/layer_14/swiglu | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_00/swiglu \| token_0/layer_15/swiglu | ciphertext:L2:c2 | 279 | 837.000 |
| token_0/layer_00/swiglu \| token_0/layer_15/swiglu | ciphertext:L3:c2 | 32 | 128.000 |
| token_0/layer_00/swiglu \| token_0/layer_15/swiglu | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_0/layer_15/swiglu | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_00/swiglu \| token_0/layer_16/swiglu | ciphertext:L2:c2 | 57 | 171.000 |
| token_0/layer_00/swiglu \| token_0/layer_16/swiglu | ciphertext:L3:c2 | 253 | 1,012.000 |
| token_0/layer_00/swiglu \| token_0/layer_16/swiglu | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_00/swiglu \| token_0/layer_16/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/swiglu \| token_0/layer_17/swiglu | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/swiglu \| token_0/layer_17/swiglu | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_00/swiglu \| token_0/layer_17/swiglu | ciphertext:L3:c2 | 285 | 1,140.000 |
| token_0/layer_00/swiglu \| token_0/layer_17/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_17/swiglu | plaintext:L4:c1 | 26 | 65.000 |
| token_0/layer_00/swiglu \| token_0/layer_18/swiglu | ciphertext:L3:c2 | 289 | 1,156.000 |
| token_0/layer_00/swiglu \| token_0/layer_18/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_18/swiglu | plaintext:L4:c1 | 24 | 60.000 |
| token_0/layer_00/swiglu \| token_0/layer_19/swiglu | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/swiglu \| token_0/layer_19/swiglu | ciphertext:L2:c2 | 42 | 126.000 |
| token_0/layer_00/swiglu \| token_0/layer_19/swiglu | ciphertext:L3:c2 | 110 | 440.000 |
| token_0/layer_00/swiglu \| token_0/layer_19/swiglu | ciphertext:L4:c2 | 154 | 770.000 |
| token_0/layer_00/swiglu \| token_0/layer_19/swiglu | ciphertext:L4:c3 | 2 | 15.000 |
| token_0/layer_00/swiglu \| token_0/layer_19/swiglu | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_00/swiglu \| token_0/layer_20/swiglu | ciphertext:L1:c2 | 7 | 14.000 |
| token_0/layer_00/swiglu \| token_0/layer_20/swiglu | ciphertext:L2:c2 | 116 | 348.000 |
| token_0/layer_00/swiglu \| token_0/layer_20/swiglu | ciphertext:L3:c2 | 190 | 760.000 |
| token_0/layer_00/swiglu \| token_0/layer_21/swiglu | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/swiglu \| token_0/layer_21/swiglu | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_0/layer_21/swiglu | ciphertext:L3:c2 | 271 | 1,084.000 |
| token_0/layer_00/swiglu \| token_0/layer_21/swiglu | ciphertext:L4:c2 | 28 | 140.000 |
| token_0/layer_00/swiglu \| token_0/layer_21/swiglu | plaintext:L4:c1 | 9 | 22.500 |
| token_0/layer_00/swiglu \| token_0/layer_22/swiglu | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/swiglu \| token_0/layer_22/swiglu | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_00/swiglu \| token_0/layer_22/swiglu | ciphertext:L3:c2 | 304 | 1,216.000 |
| token_0/layer_00/swiglu \| token_0/layer_22/swiglu | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_0/layer_22/swiglu | plaintext:L3:c1 | 16 | 32.000 |
| token_0/layer_00/swiglu \| token_0/layer_22/swiglu | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/swiglu | ciphertext:L1:c2 | 30 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/swiglu | ciphertext:L2:c2 | 130 | 390.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/swiglu | ciphertext:L3:c2 | 670 | 2,680.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/swiglu | ciphertext:L4:c2 | 166 | 830.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/swiglu | ciphertext:L5:c2 | 8 | 48.000 |
| token_0/layer_00/swiglu \| token_1/layer_00/swiglu | plaintext:L4:c1 | 33 | 82.500 |
| token_0/layer_00/swiglu \| token_1/layer_00/swiglu | plaintext:L5:c1 | 5 | 15.000 |
| token_0/layer_00/swiglu \| token_1/layer_01/gqa_attention | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/swiglu \| token_1/layer_01/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/swiglu \| token_1/layer_01/swiglu | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/swiglu \| token_1/layer_01/swiglu | ciphertext:L2:c2 | 13 | 39.000 |
| token_0/layer_00/swiglu \| token_1/layer_01/swiglu | ciphertext:L3:c2 | 283 | 1,132.000 |
| token_0/layer_00/swiglu \| token_1/layer_01/swiglu | ciphertext:L4:c2 | 9 | 45.000 |
| token_0/layer_00/swiglu \| token_1/layer_01/swiglu | plaintext:L4:c1 | 20 | 50.000 |
| token_0/layer_00/swiglu \| token_1/layer_02/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_02/swiglu | ciphertext:L1:c2 | 7 | 14.000 |
| token_0/layer_00/swiglu \| token_1/layer_02/swiglu | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/swiglu \| token_1/layer_02/swiglu | ciphertext:L3:c2 | 279 | 1,116.000 |
| token_0/layer_00/swiglu \| token_1/layer_02/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_02/swiglu | plaintext:L4:c1 | 24 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/gqa_attention | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/gqa_attention | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | ciphertext:L1:c2 | 80 | 160.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | ciphertext:L2:c2 | 103 | 309.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | ciphertext:L2:c3 | 4 | 18.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | ciphertext:L3:c2 | 100 | 400.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | ciphertext:L4:c2 | 13 | 65.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | ciphertext:L5:c2 | 8 | 48.000 |
| token_0/layer_00/swiglu \| token_1/layer_03/swiglu | plaintext:L2:c1 | 4 | 6.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | ciphertext:L1:c2 | 97 | 194.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | ciphertext:L3:c2 | 189 | 756.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | ciphertext:L3:c3 | 2 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | ciphertext:L4:c2 | 10 | 50.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/swiglu \| token_1/layer_04/swiglu | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_00/swiglu \| token_1/layer_05/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_05/swiglu | ciphertext:L1:c2 | 22 | 44.000 |
| token_0/layer_00/swiglu \| token_1/layer_05/swiglu | ciphertext:L2:c2 | 18 | 54.000 |
| token_0/layer_00/swiglu \| token_1/layer_05/swiglu | ciphertext:L3:c2 | 254 | 1,016.000 |
| token_0/layer_00/swiglu \| token_1/layer_05/swiglu | ciphertext:L4:c2 | 10 | 50.000 |
| token_0/layer_00/swiglu \| token_1/layer_05/swiglu | plaintext:L4:c1 | 18 | 45.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/swiglu | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/swiglu | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/swiglu | ciphertext:L3:c2 | 285 | 1,140.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_06/swiglu | plaintext:L4:c1 | 24 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_07/gqa_attention | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_00/swiglu \| token_1/layer_07/swiglu | ciphertext:L2:c2 | 280 | 840.000 |
| token_0/layer_00/swiglu \| token_1/layer_07/swiglu | ciphertext:L3:c2 | 32 | 128.000 |
| token_0/layer_00/swiglu \| token_1/layer_07/swiglu | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_08/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_08/swiglu | ciphertext:L1:c2 | 131 | 262.000 |
| token_0/layer_00/swiglu \| token_1/layer_08/swiglu | ciphertext:L2:c2 | 145 | 435.000 |
| token_0/layer_00/swiglu \| token_1/layer_08/swiglu | ciphertext:L3:c2 | 37 | 148.000 |
| token_0/layer_00/swiglu \| token_1/layer_09/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_09/swiglu | ciphertext:L1:c2 | 61 | 122.000 |
| token_0/layer_00/swiglu \| token_1/layer_09/swiglu | ciphertext:L2:c2 | 163 | 489.000 |
| token_0/layer_00/swiglu \| token_1/layer_09/swiglu | ciphertext:L3:c2 | 88 | 352.000 |
| token_0/layer_00/swiglu \| token_1/layer_09/swiglu | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | ciphertext:L1:c2 | 9 | 18.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | ciphertext:L2:c2 | 13 | 39.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | ciphertext:L3:c2 | 188 | 752.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | ciphertext:L4:c2 | 84 | 420.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | ciphertext:L5:c2 | 7 | 42.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_00/swiglu \| token_1/layer_10/swiglu | plaintext:L5:c1 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_11/gqa_attention | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/swiglu \| token_1/layer_11/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_00/swiglu \| token_1/layer_11/swiglu | ciphertext:L2:c2 | 158 | 474.000 |
| token_0/layer_00/swiglu \| token_1/layer_11/swiglu | ciphertext:L3:c2 | 139 | 556.000 |
| token_0/layer_00/swiglu \| token_1/layer_11/swiglu | ciphertext:L4:c2 | 10 | 50.000 |
| token_0/layer_00/swiglu \| token_1/layer_11/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/swiglu \| token_1/layer_11/swiglu | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_00/swiglu \| token_1/layer_12/gqa_attention | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_00/swiglu \| token_1/layer_12/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_00/swiglu \| token_1/layer_12/swiglu | ciphertext:L2:c2 | 257 | 771.000 |
| token_0/layer_00/swiglu \| token_1/layer_12/swiglu | ciphertext:L3:c2 | 55 | 220.000 |
| token_0/layer_00/swiglu \| token_1/layer_12/swiglu | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L1:c2 | 74 | 148.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L2:c2 | 74 | 222.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L2:c3 | 2 | 9.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L3:c2 | 126 | 504.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L3:c3 | 2 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L4:c2 | 18 | 90.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | ciphertext:L5:c2 | 10 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_00/swiglu \| token_1/layer_13/swiglu | plaintext:L4:c1 | 5 | 12.500 |
| token_0/layer_00/swiglu \| token_1/layer_14/gqa_attention | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_00/swiglu \| token_1/layer_14/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_14/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_14/swiglu | ciphertext:L3:c2 | 289 | 1,156.000 |
| token_0/layer_00/swiglu \| token_1/layer_14/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_14/swiglu | plaintext:L4:c1 | 24 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/swiglu | ciphertext:L2:c2 | 9 | 27.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/swiglu | ciphertext:L3:c2 | 220 | 880.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/swiglu | ciphertext:L4:c2 | 69 | 345.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/swiglu | plaintext:L4:c1 | 18 | 45.000 |
| token_0/layer_00/swiglu \| token_1/layer_15/swiglu | plaintext:L5:c1 | 3 | 9.000 |
| token_0/layer_00/swiglu \| token_1/layer_16/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_16/swiglu | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_00/swiglu \| token_1/layer_16/swiglu | ciphertext:L2:c2 | 69 | 207.000 |
| token_0/layer_00/swiglu \| token_1/layer_16/swiglu | ciphertext:L3:c2 | 219 | 876.000 |
| token_0/layer_00/swiglu \| token_1/layer_16/swiglu | ciphertext:L4:c2 | 10 | 50.000 |
| token_0/layer_00/swiglu \| token_1/layer_16/swiglu | plaintext:L4:c1 | 20 | 50.000 |
| token_0/layer_00/swiglu \| token_1/layer_17/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_17/swiglu | ciphertext:L2:c2 | 81 | 243.000 |
| token_0/layer_00/swiglu \| token_1/layer_17/swiglu | ciphertext:L3:c2 | 197 | 788.000 |
| token_0/layer_00/swiglu \| token_1/layer_17/swiglu | ciphertext:L4:c2 | 32 | 160.000 |
| token_0/layer_00/swiglu \| token_1/layer_17/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_00/swiglu \| token_1/layer_17/swiglu | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_17/swiglu | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_00/swiglu \| token_1/layer_18/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_18/swiglu | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_00/swiglu \| token_1/layer_18/swiglu | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/swiglu \| token_1/layer_18/swiglu | ciphertext:L3:c2 | 278 | 1,112.000 |
| token_0/layer_00/swiglu \| token_1/layer_18/swiglu | ciphertext:L4:c2 | 25 | 125.000 |
| token_0/layer_00/swiglu \| token_1/layer_19/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_19/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_19/swiglu | ciphertext:L1:c2 | 8 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_19/swiglu | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_19/swiglu | ciphertext:L3:c2 | 277 | 1,108.000 |
| token_0/layer_00/swiglu \| token_1/layer_19/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_19/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_00/swiglu \| token_1/layer_20/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_20/swiglu | ciphertext:L1:c2 | 9 | 18.000 |
| token_0/layer_00/swiglu \| token_1/layer_20/swiglu | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_00/swiglu \| token_1/layer_20/swiglu | ciphertext:L3:c2 | 301 | 1,204.000 |
| token_0/layer_00/swiglu \| token_1/layer_21/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_00/swiglu \| token_1/layer_21/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_00/swiglu \| token_1/layer_21/swiglu | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_00/swiglu \| token_1/layer_21/swiglu | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_00/swiglu \| token_1/layer_21/swiglu | ciphertext:L3:c2 | 279 | 1,116.000 |
| token_0/layer_00/swiglu \| token_1/layer_21/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_21/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_00/swiglu \| token_1/layer_22/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_22/swiglu | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_00/swiglu \| token_1/layer_22/swiglu | ciphertext:L3:c2 | 279 | 1,116.000 |
| token_0/layer_00/swiglu \| token_1/layer_22/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_00/swiglu \| token_1/layer_22/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_00/swiglu \| token_1/layer_23/gqa_attention | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_00/swiglu \| token_1/layer_23/swiglu | ciphertext:L2:c2 | 170 | 510.000 |
| token_0/layer_00/swiglu \| token_1/layer_23/swiglu | ciphertext:L3:c2 | 75 | 300.000 |
| token_0/layer_00/swiglu \| token_1/layer_23/swiglu | plaintext:L3:c1 | 68 | 136.000 |
| token_0/layer_00/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L1:c1 | 17 | 17.000 |
| token_0/layer_00/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 57 | 85.500 |
| token_0/layer_00/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 50 | 100.000 |
| token_0/layer_00/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_01/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_01/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_01/gqa_attention | ciphertext:L3:c2 | 17 | 68.000 |
| token_0/layer_01/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_01/gqa_attention | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L3:c2 | 154 | 616.000 |
| token_0/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_01/model_list.1.attention.k_weight.weight | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_01/model_list.1.attention.k_weight.weight | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_01/model_list.1.attention.k_weight.weight \| token_1/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 972 | 3,888.000 |
| token_0/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L4:c2 | 33 | 165.000 |
| token_0/layer_01/model_list.1.attention.o_weight.weight | plaintext:L4:c1 | 65 | 162.500 |
| token_0/layer_01/model_list.1.attention.o_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L1:c2 | 25 | 50.000 |
| token_0/layer_01/model_list.1.attention.o_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_01/model_list.1.attention.o_weight.weight \| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L2:c2 | 11 | 33.000 |
| token_0/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_01/model_list.1.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_01/model_list.1.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L3:c2 | 42 | 168.000 |
| token_0/layer_01/model_list.1.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_01/model_list.1.attention.q_weight.weight \| token_1/layer_01/model_list.1.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L3:c2 | 151 | 604.000 |
| token_0/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_01/model_list.1.attention.v_weight.weight | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_01/model_list.1.attention.v_weight.weight | plaintext:L4:c1 | 11 | 27.500 |
| token_0/layer_01/model_list.1.attention.v_weight.weight \| token_1/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_01/model_list.1.post_Normal.weight | ciphertext:L2:c2 | 44 | 132.000 |
| token_0/layer_01/model_list.1.post_Normal.weight | ciphertext:L3:c2 | 60 | 240.000 |
| token_0/layer_01/model_list.1.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_01/model_list.1.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_01/model_list.1.post_Normal.weight \| token_1/layer_01/model_list.1.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_01/model_list.1.pre_Normal.weight | ciphertext:L2:c2 | 102 | 306.000 |
| token_0/layer_01/model_list.1.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_01/model_list.1.pre_Normal.weight \| token_1/layer_01/model_list.1.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_01/rope | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_01/swiglu | ciphertext:L1:c2 | 894 | 1,788.000 |
| token_0/layer_01/swiglu | ciphertext:L2:c2 | 4299 | 12,897.000 |
| token_0/layer_01/swiglu | ciphertext:L2:c3 | 17 | 76.500 |
| token_0/layer_01/swiglu | ciphertext:L3:c2 | 11435 | 45,740.000 |
| token_0/layer_01/swiglu | ciphertext:L3:c3 | 13 | 78.000 |
| token_0/layer_01/swiglu | ciphertext:L4:c2 | 485 | 2,425.000 |
| token_0/layer_01/swiglu | ciphertext:L4:c3 | 4 | 30.000 |
| token_0/layer_01/swiglu | ciphertext:L5:c2 | 42 | 252.000 |
| token_0/layer_01/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_01/swiglu | plaintext:L2:c1 | 5 | 7.500 |
| token_0/layer_01/swiglu | plaintext:L3:c1 | 6 | 12.000 |
| token_0/layer_01/swiglu | plaintext:L4:c1 | 186 | 465.000 |
| token_0/layer_01/swiglu | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_01/swiglu \| token_0/layer_05/swiglu \| token_1/layer_04/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_01/swiglu \| token_0/layer_05/swiglu \| token_1/layer_04/swiglu | plaintext:L3:c1 | 23 | 46.000 |
| token_0/layer_01/swiglu \| token_0/layer_05/swiglu \| token_1/layer_04/swiglu | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_01/swiglu \| token_0/layer_18/swiglu \| token_1/layer_00/swiglu | plaintext:L2:c1 | 6 | 9.000 |
| token_0/layer_01/swiglu \| token_0/layer_18/swiglu \| token_1/layer_00/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_01/swiglu \| token_1/layer_01/swiglu | ciphertext:L1:c2 | 33 | 66.000 |
| token_0/layer_01/swiglu \| token_1/layer_01/swiglu | ciphertext:L2:c2 | 178 | 534.000 |
| token_0/layer_01/swiglu \| token_1/layer_01/swiglu | ciphertext:L3:c2 | 322 | 1,288.000 |
| token_0/layer_01/swiglu \| token_1/layer_01/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_01/swiglu \| token_1/layer_01/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_01/swiglu \| token_1/layer_01/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_01/swiglu \| token_1/layer_01/swiglu | plaintext:L4:c1 | 22 | 55.000 |
| token_0/layer_01/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 17 | 25.500 |
| token_0/layer_01/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 22 | 44.000 |
| token_0/layer_01/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 19 | 47.500 |
| token_0/layer_01/swiglu \| token_shared/layer_shared/lm_head.weight | plaintext:L5:c1 | 8 | 24.000 |
| token_0/layer_02/_decoder_block | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_02/_decoder_block | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_02/_decoder_block | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_02/gqa_attention | ciphertext:L2:c2 | 19 | 57.000 |
| token_0/layer_02/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L2:c2 | 161 | 483.000 |
| token_0/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_02/model_list.2.attention.k_weight.weight \| token_1/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L1:c2 | 3 | 6.000 |
| token_0/layer_02/model_list.2.attention.k_weight.weight \| token_1/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_02/model_list.2.attention.k_weight.weight \| token_1/layer_02/model_list.2.attention.k_weight.weight | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L1:c2 | 658 | 1,316.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L2:c2 | 363 | 1,089.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L2:c3 | 2 | 9.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c2 | 14 | 56.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight \| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c2 | 20 | 80.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight \| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_02/model_list.2.attention.o_weight.weight \| token_1/layer_02/model_list.2.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_02/model_list.2.attention.q_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_02/model_list.2.attention.q_weight.weight \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu \| token_1/layer_02/model_list.2.attention.q_weight.weight \| token_1/layer_04/swiglu | plaintext:L2:c1 | 23 | 34.500 |
| token_0/layer_02/model_list.2.attention.q_weight.weight \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu \| token_1/layer_02/model_list.2.attention.q_weight.weight \| token_1/layer_04/swiglu | plaintext:L3:c1 | 3 | 6.000 |
| token_0/layer_02/model_list.2.attention.q_weight.weight \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu \| token_1/layer_02/model_list.2.attention.q_weight.weight \| token_1/layer_04/swiglu | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_02/model_list.2.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.q_weight.weight | ciphertext:L1:c2 | 18 | 36.000 |
| token_0/layer_02/model_list.2.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.q_weight.weight | ciphertext:L2:c2 | 31 | 93.000 |
| token_0/layer_02/model_list.2.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.q_weight.weight | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_02/model_list.2.attention.q_weight.weight \| token_1/layer_02/model_list.2.attention.q_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L2:c2 | 162 | 486.000 |
| token_0/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_02/model_list.2.attention.v_weight.weight \| token_1/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_02/model_list.2.attention.v_weight.weight \| token_1/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_02/model_list.2.post_Normal.weight | ciphertext:L1:c2 | 32 | 64.000 |
| token_0/layer_02/model_list.2.post_Normal.weight | ciphertext:L2:c2 | 11 | 33.000 |
| token_0/layer_02/model_list.2.post_Normal.weight | ciphertext:L3:c2 | 66 | 264.000 |
| token_0/layer_02/model_list.2.post_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_02/model_list.2.post_Normal.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_02/model_list.2.post_Normal.weight \| token_1/layer_02/model_list.2.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_02/model_list.2.pre_Normal.weight | ciphertext:L2:c2 | 91 | 273.000 |
| token_0/layer_02/model_list.2.pre_Normal.weight | ciphertext:L3:c2 | 17 | 68.000 |
| token_0/layer_02/model_list.2.pre_Normal.weight \| token_1/layer_02/model_list.2.pre_Normal.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_02/rope | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_02/rope | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_02/swiglu | ciphertext:L1:c2 | 1591 | 3,182.000 |
| token_0/layer_02/swiglu | ciphertext:L2:c2 | 4479 | 13,437.000 |
| token_0/layer_02/swiglu | ciphertext:L2:c3 | 20 | 90.000 |
| token_0/layer_02/swiglu | ciphertext:L3:c2 | 7351 | 29,404.000 |
| token_0/layer_02/swiglu | ciphertext:L3:c3 | 15 | 90.000 |
| token_0/layer_02/swiglu | ciphertext:L4:c2 | 3552 | 17,760.000 |
| token_0/layer_02/swiglu | ciphertext:L4:c3 | 10 | 75.000 |
| token_0/layer_02/swiglu | ciphertext:L5:c2 | 191 | 1,146.000 |
| token_0/layer_02/swiglu | ciphertext:L5:c3 | 3 | 27.000 |
| token_0/layer_02/swiglu | plaintext:L1:c1 | 3 | 3.000 |
| token_0/layer_02/swiglu | plaintext:L2:c1 | 4 | 6.000 |
| token_0/layer_02/swiglu | plaintext:L3:c1 | 3 | 6.000 |
| token_0/layer_02/swiglu | plaintext:L4:c1 | 11 | 27.500 |
| token_0/layer_02/swiglu | plaintext:L5:c1 | 240 | 720.000 |
| token_0/layer_02/swiglu \| token_0/layer_03/swiglu | plaintext:L2:c1 | 3 | 4.500 |
| token_0/layer_02/swiglu \| token_0/layer_03/swiglu | plaintext:L3:c1 | 21 | 42.000 |
| token_0/layer_02/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_03/swiglu \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L2:c1 | 5 | 7.500 |
| token_0/layer_02/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_03/swiglu \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L3:c1 | 16 | 32.000 |
| token_0/layer_02/swiglu \| token_0/layer_05/swiglu \| token_1/layer_01/model_list.1.attention.q_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_02/swiglu \| token_0/layer_05/swiglu \| token_1/layer_01/model_list.1.attention.q_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 12 | 24.000 |
| token_0/layer_02/swiglu \| token_0/layer_05/swiglu \| token_1/layer_01/model_list.1.attention.q_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 5 | 12.500 |
| token_0/layer_02/swiglu \| token_0/layer_19/swiglu \| token_1/layer_00/swiglu | plaintext:L2:c1 | 6 | 9.000 |
| token_0/layer_02/swiglu \| token_0/layer_19/swiglu \| token_1/layer_00/swiglu | plaintext:L3:c1 | 11 | 22.000 |
| token_0/layer_02/swiglu \| token_0/layer_19/swiglu \| token_1/layer_00/swiglu | plaintext:L4:c1 | 9 | 22.500 |
| token_0/layer_02/swiglu \| token_1/layer_01/model_list.1.attention.q_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 11 | 16.500 |
| token_0/layer_02/swiglu \| token_1/layer_01/model_list.1.attention.q_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 9 | 18.000 |
| token_0/layer_02/swiglu \| token_1/layer_01/model_list.1.attention.q_weight.weight \| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_02/swiglu \| token_1/layer_02/swiglu | ciphertext:L1:c2 | 25 | 50.000 |
| token_0/layer_02/swiglu \| token_1/layer_02/swiglu | ciphertext:L2:c2 | 93 | 279.000 |
| token_0/layer_02/swiglu \| token_1/layer_02/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_02/swiglu \| token_1/layer_02/swiglu | ciphertext:L3:c2 | 286 | 1,144.000 |
| token_0/layer_02/swiglu \| token_1/layer_02/swiglu | ciphertext:L4:c2 | 15 | 75.000 |
| token_0/layer_02/swiglu \| token_1/layer_02/swiglu | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_03/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_03/_decoder_block | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_03/gqa_attention | ciphertext:L3:c2 | 19 | 76.000 |
| token_0/layer_03/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L3:c2 | 154 | 616.000 |
| token_0/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_03/model_list.3.attention.k_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_03/model_list.3.attention.k_weight.weight \| token_1/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 941 | 3,764.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c2 | 51 | 255.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight | plaintext:L4:c1 | 84 | 210.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 48 | 192.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_03/model_list.3.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_04/swiglu | plaintext:L2:c1 | 4 | 6.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_04/swiglu | plaintext:L3:c1 | 18 | 36.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_04/swiglu \| token_1/layer_05/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_04/swiglu \| token_1/layer_05/swiglu | plaintext:L3:c1 | 4 | 8.000 |
| token_0/layer_03/model_list.3.attention.o_weight.weight \| token_1/layer_04/swiglu \| token_1/layer_05/swiglu | plaintext:L4:c1 | 36 | 90.000 |
| token_0/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_03/model_list.3.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_03/model_list.3.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_03/model_list.3.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L4:c2 | 26 | 130.000 |
| token_0/layer_03/model_list.3.attention.q_weight.weight \| token_1/layer_03/model_list.3.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_0/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L4:c2 | 9 | 45.000 |
| token_0/layer_03/model_list.3.attention.v_weight.weight | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_03/model_list.3.attention.v_weight.weight \| token_1/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_03/model_list.3.post_Normal.weight | ciphertext:L1:c2 | 48 | 96.000 |
| token_0/layer_03/model_list.3.post_Normal.weight | ciphertext:L2:c2 | 28 | 84.000 |
| token_0/layer_03/model_list.3.post_Normal.weight | ciphertext:L3:c2 | 36 | 144.000 |
| token_0/layer_03/model_list.3.post_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_03/model_list.3.post_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_03/model_list.3.post_Normal.weight \| token_1/layer_03/model_list.3.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_03/model_list.3.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_03/model_list.3.pre_Normal.weight | ciphertext:L4:c2 | 90 | 450.000 |
| token_0/layer_03/model_list.3.pre_Normal.weight | ciphertext:L5:c2 | 5 | 30.000 |
| token_0/layer_03/model_list.3.pre_Normal.weight | plaintext:L5:c1 | 6 | 18.000 |
| token_0/layer_03/model_list.3.pre_Normal.weight \| token_1/layer_03/model_list.3.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_03/rope | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_03/rope | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_03/swiglu | ciphertext:L1:c2 | 1505 | 3,010.000 |
| token_0/layer_03/swiglu | ciphertext:L2:c2 | 5600 | 16,800.000 |
| token_0/layer_03/swiglu | ciphertext:L2:c3 | 15 | 67.500 |
| token_0/layer_03/swiglu | ciphertext:L3:c2 | 4398 | 17,592.000 |
| token_0/layer_03/swiglu | ciphertext:L3:c3 | 13 | 78.000 |
| token_0/layer_03/swiglu | ciphertext:L4:c2 | 5310 | 26,550.000 |
| token_0/layer_03/swiglu | ciphertext:L4:c3 | 9 | 67.500 |
| token_0/layer_03/swiglu | ciphertext:L5:c2 | 240 | 1,440.000 |
| token_0/layer_03/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_03/swiglu | plaintext:L1:c1 | 4 | 4.000 |
| token_0/layer_03/swiglu | plaintext:L2:c1 | 7 | 10.500 |
| token_0/layer_03/swiglu | plaintext:L3:c1 | 6 | 12.000 |
| token_0/layer_03/swiglu | plaintext:L4:c1 | 134 | 335.000 |
| token_0/layer_03/swiglu | plaintext:L5:c1 | 151 | 453.000 |
| token_0/layer_03/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_0/layer_16/swiglu \| token_1/layer_00/swiglu \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L2:c1 | 10 | 15.000 |
| token_0/layer_03/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_0/layer_16/swiglu \| token_1/layer_00/swiglu \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L3:c1 | 4 | 8.000 |
| token_0/layer_03/swiglu \| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_0/layer_16/swiglu \| token_1/layer_00/swiglu \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu | plaintext:L2:c1 | 21 | 31.500 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_0/layer_12/swiglu | plaintext:L3:c1 | 21 | 42.000 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu \| token_1/layer_04/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu \| token_1/layer_04/swiglu | plaintext:L3:c1 | 6 | 12.000 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu \| token_1/layer_04/swiglu | plaintext:L4:c1 | 9 | 22.500 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_1/layer_00/swiglu \| token_1/layer_04/swiglu | plaintext:L5:c1 | 3 | 9.000 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_1/layer_04/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_1/layer_04/swiglu | plaintext:L3:c1 | 9 | 18.000 |
| token_0/layer_03/swiglu \| token_0/layer_05/swiglu \| token_1/layer_04/swiglu | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_03/swiglu \| token_0/layer_07/swiglu \| token_1/layer_13/swiglu | plaintext:L2:c1 | 20 | 30.000 |
| token_0/layer_03/swiglu \| token_0/layer_07/swiglu \| token_1/layer_13/swiglu | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_03/swiglu \| token_0/layer_10/swiglu | plaintext:L3:c1 | 22 | 44.000 |
| token_0/layer_03/swiglu \| token_0/layer_12/swiglu \| token_1/layer_00/gqa_attention | plaintext:L1:c1 | 9 | 9.000 |
| token_0/layer_03/swiglu \| token_0/layer_12/swiglu \| token_1/layer_00/gqa_attention | plaintext:L2:c1 | 12 | 18.000 |
| token_0/layer_03/swiglu \| token_1/layer_03/swiglu | ciphertext:L1:c2 | 35 | 70.000 |
| token_0/layer_03/swiglu \| token_1/layer_03/swiglu | ciphertext:L2:c2 | 310 | 930.000 |
| token_0/layer_03/swiglu \| token_1/layer_03/swiglu | ciphertext:L3:c2 | 213 | 852.000 |
| token_0/layer_03/swiglu \| token_1/layer_03/swiglu | ciphertext:L4:c2 | 8 | 40.000 |
| token_0/layer_03/swiglu \| token_1/layer_03/swiglu | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_03/swiglu \| token_1/layer_05/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_03/swiglu \| token_1/layer_05/swiglu | plaintext:L3:c1 | 20 | 40.000 |
| token_0/layer_04/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_04/gqa_attention | ciphertext:L4:c2 | 19 | 95.000 |
| token_0/layer_04/gqa_attention | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L4:c2 | 156 | 780.000 |
| token_0/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L5:c2 | 6 | 36.000 |
| token_0/layer_04/model_list.4.attention.k_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_04/model_list.4.attention.k_weight.weight \| token_1/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_04/model_list.4.attention.k_weight.weight \| token_1/layer_04/model_list.4.attention.k_weight.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 726 | 2,904.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L4:c2 | 299 | 1,495.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L5:c2 | 12 | 72.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L1:c2 | 3 | 6.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L2:c2 | 22 | 66.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 19 | 76.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_04/model_list.4.attention.o_weight.weight \| token_1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L4:c2 | 13 | 65.000 |
| token_0/layer_04/model_list.4.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_04/model_list.4.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L2:c2 | 46 | 138.000 |
| token_0/layer_04/model_list.4.attention.q_weight.weight \| token_1/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L4:c2 | 152 | 760.000 |
| token_0/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L5:c2 | 8 | 48.000 |
| token_0/layer_04/model_list.4.attention.v_weight.weight \| token_1/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_04/model_list.4.post_Normal.weight | ciphertext:L3:c2 | 101 | 404.000 |
| token_0/layer_04/model_list.4.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_04/model_list.4.post_Normal.weight \| token_1/layer_04/model_list.4.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_04/model_list.4.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_04/model_list.4.pre_Normal.weight | ciphertext:L4:c2 | 111 | 555.000 |
| token_0/layer_04/model_list.4.pre_Normal.weight | ciphertext:L5:c2 | 5 | 30.000 |
| token_0/layer_04/model_list.4.pre_Normal.weight \| token_1/layer_04/model_list.4.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_04/rope | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_04/swiglu | ciphertext:L1:c2 | 437 | 874.000 |
| token_0/layer_04/swiglu | ciphertext:L2:c2 | 2880 | 8,640.000 |
| token_0/layer_04/swiglu | ciphertext:L2:c3 | 5 | 22.500 |
| token_0/layer_04/swiglu | ciphertext:L3:c2 | 13051 | 52,204.000 |
| token_0/layer_04/swiglu | ciphertext:L3:c3 | 3 | 18.000 |
| token_0/layer_04/swiglu | ciphertext:L4:c2 | 591 | 2,955.000 |
| token_0/layer_04/swiglu | ciphertext:L4:c3 | 2 | 15.000 |
| token_0/layer_04/swiglu | ciphertext:L5:c2 | 12 | 72.000 |
| token_0/layer_04/swiglu | plaintext:L4:c1 | 632 | 1,580.000 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | ciphertext:L1:c2 | 279 | 558.000 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | ciphertext:L2:c2 | 187 | 561.000 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | ciphertext:L3:c2 | 151 | 604.000 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | ciphertext:L4:c2 | 68 | 340.000 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | ciphertext:L5:c2 | 6 | 36.000 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | plaintext:L2:c1 | 11 | 16.500 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_04/swiglu \| token_1/layer_04/swiglu | plaintext:L5:c1 | 3 | 9.000 |
| token_0/layer_05/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_05/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_05/gqa_attention | ciphertext:L2:c2 | 19 | 57.000 |
| token_0/layer_05/gqa_attention | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L2:c2 | 158 | 474.000 |
| token_0/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_05/model_list.5.attention.k_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_05/model_list.5.attention.k_weight.weight | plaintext:L3:c1 | 4 | 8.000 |
| token_0/layer_05/model_list.5.attention.k_weight.weight \| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_05/model_list.5.attention.k_weight.weight \| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L1:c2 | 19 | 38.000 |
| token_0/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L2:c2 | 956 | 2,868.000 |
| token_0/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L3:c2 | 32 | 128.000 |
| token_0/layer_05/model_list.5.attention.o_weight.weight | plaintext:L3:c1 | 31 | 62.000 |
| token_0/layer_05/model_list.5.attention.o_weight.weight \| token_1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_05/model_list.5.attention.o_weight.weight \| token_1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_05/model_list.5.attention.o_weight.weight \| token_1/layer_05/model_list.5.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L2:c2 | 11 | 33.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_05/model_list.5.attention.q_weight.weight | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L1:c2 | 23 | 46.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L3:c2 | 23 | 92.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_05/model_list.5.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_05/model_list.5.attention.q_weight.weight \| token_1/layer_05/model_list.5.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L2:c2 | 155 | 465.000 |
| token_0/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_05/model_list.5.attention.v_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_05/model_list.5.attention.v_weight.weight | plaintext:L3:c1 | 5 | 10.000 |
| token_0/layer_05/model_list.5.attention.v_weight.weight \| token_1/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_05/model_list.5.attention.v_weight.weight \| token_1/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_05/model_list.5.post_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_05/model_list.5.post_Normal.weight | ciphertext:L2:c2 | 99 | 297.000 |
| token_0/layer_05/model_list.5.post_Normal.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_05/model_list.5.post_Normal.weight | plaintext:L3:c1 | 5 | 10.000 |
| token_0/layer_05/model_list.5.post_Normal.weight \| token_1/layer_05/model_list.5.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_05/model_list.5.pre_Normal.weight | ciphertext:L1:c2 | 26 | 52.000 |
| token_0/layer_05/model_list.5.pre_Normal.weight | ciphertext:L2:c2 | 70 | 210.000 |
| token_0/layer_05/model_list.5.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_05/model_list.5.pre_Normal.weight | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_05/model_list.5.pre_Normal.weight \| token_1/layer_05/model_list.5.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_05/rope | ciphertext:L2:c2 | 9 | 27.000 |
| token_0/layer_05/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_05/rope | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_05/swiglu | ciphertext:L1:c2 | 1860 | 3,720.000 |
| token_0/layer_05/swiglu | ciphertext:L2:c2 | 3158 | 9,474.000 |
| token_0/layer_05/swiglu | ciphertext:L2:c3 | 20 | 90.000 |
| token_0/layer_05/swiglu | ciphertext:L3:c2 | 4343 | 17,372.000 |
| token_0/layer_05/swiglu | ciphertext:L3:c3 | 14 | 84.000 |
| token_0/layer_05/swiglu | ciphertext:L4:c2 | 7200 | 36,000.000 |
| token_0/layer_05/swiglu | ciphertext:L4:c3 | 6 | 45.000 |
| token_0/layer_05/swiglu | ciphertext:L5:c2 | 595 | 3,570.000 |
| token_0/layer_05/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_05/swiglu | plaintext:L2:c1 | 15 | 22.500 |
| token_0/layer_05/swiglu | plaintext:L3:c1 | 53 | 106.000 |
| token_0/layer_05/swiglu | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_05/swiglu | plaintext:L5:c1 | 257 | 771.000 |
| token_0/layer_05/swiglu \| token_0/layer_12/swiglu | plaintext:L1:c1 | 22 | 22.000 |
| token_0/layer_05/swiglu \| token_1/layer_05/swiglu | ciphertext:L1:c2 | 20 | 40.000 |
| token_0/layer_05/swiglu \| token_1/layer_05/swiglu | ciphertext:L2:c2 | 174 | 522.000 |
| token_0/layer_05/swiglu \| token_1/layer_05/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_05/swiglu \| token_1/layer_05/swiglu | ciphertext:L3:c2 | 343 | 1,372.000 |
| token_0/layer_05/swiglu \| token_1/layer_05/swiglu | ciphertext:L4:c2 | 13 | 65.000 |
| token_0/layer_05/swiglu \| token_1/layer_05/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_05/swiglu \| token_1/layer_05/swiglu | plaintext:L4:c1 | 23 | 57.500 |
| token_0/layer_06/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_06/gqa_attention | ciphertext:L1:c2 | 11 | 22.000 |
| token_0/layer_06/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_06/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_06/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_0/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_06/model_list.6.attention.k_weight.weight | plaintext:L4:c1 | 15 | 37.500 |
| token_0/layer_06/model_list.6.attention.k_weight.weight \| token_1/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L1:c2 | 23 | 46.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 949 | 3,796.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L4:c2 | 32 | 160.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_06/model_list.6.attention.o_weight.weight | plaintext:L4:c1 | 64 | 160.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight \| token_1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 23 | 92.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight \| token_1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_06/model_list.6.attention.o_weight.weight \| token_1/layer_06/model_list.6.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_06/model_list.6.attention.q_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_06/model_list.6.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.q_weight.weight | ciphertext:L3:c2 | 49 | 196.000 |
| token_0/layer_06/model_list.6.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_06/model_list.6.attention.q_weight.weight \| token_1/layer_06/model_list.6.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L3:c2 | 148 | 592.000 |
| token_0/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_06/model_list.6.attention.v_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_06/model_list.6.attention.v_weight.weight | plaintext:L4:c1 | 17 | 42.500 |
| token_0/layer_06/model_list.6.attention.v_weight.weight \| token_1/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_06/model_list.6.post_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_06/model_list.6.post_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_0/layer_06/model_list.6.post_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_06/model_list.6.post_Normal.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_06/model_list.6.post_Normal.weight \| token_1/layer_06/model_list.6.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_06/model_list.6.pre_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_06/model_list.6.pre_Normal.weight | ciphertext:L3:c2 | 102 | 408.000 |
| token_0/layer_06/model_list.6.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_06/model_list.6.pre_Normal.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_06/model_list.6.pre_Normal.weight \| token_1/layer_06/model_list.6.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_06/rope | ciphertext:L1:c2 | 6 | 12.000 |
| token_0/layer_06/rope | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_06/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_06/swiglu | ciphertext:L1:c2 | 2091 | 4,182.000 |
| token_0/layer_06/swiglu | ciphertext:L2:c2 | 5929 | 17,787.000 |
| token_0/layer_06/swiglu | ciphertext:L2:c3 | 19 | 85.500 |
| token_0/layer_06/swiglu | ciphertext:L3:c2 | 8668 | 34,672.000 |
| token_0/layer_06/swiglu | ciphertext:L3:c3 | 16 | 96.000 |
| token_0/layer_06/swiglu | ciphertext:L4:c2 | 422 | 2,110.000 |
| token_0/layer_06/swiglu | ciphertext:L4:c3 | 5 | 37.500 |
| token_0/layer_06/swiglu | ciphertext:L5:c2 | 99 | 594.000 |
| token_0/layer_06/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_06/swiglu | plaintext:L2:c1 | 13 | 19.500 |
| token_0/layer_06/swiglu | plaintext:L3:c1 | 26 | 52.000 |
| token_0/layer_06/swiglu | plaintext:L4:c1 | 126 | 315.000 |
| token_0/layer_06/swiglu \| token_1/layer_06/swiglu | ciphertext:L1:c2 | 13 | 26.000 |
| token_0/layer_06/swiglu \| token_1/layer_06/swiglu | ciphertext:L2:c2 | 150 | 450.000 |
| token_0/layer_06/swiglu \| token_1/layer_06/swiglu | ciphertext:L3:c2 | 245 | 980.000 |
| token_0/layer_06/swiglu \| token_1/layer_06/swiglu | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_06/swiglu \| token_1/layer_06/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_06/swiglu \| token_1/layer_06/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_07/_decoder_block | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_07/_decoder_block | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_07/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_07/_decoder_block | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_07/gqa_attention | ciphertext:L2:c2 | 7 | 21.000 |
| token_0/layer_07/gqa_attention | ciphertext:L3:c2 | 14 | 56.000 |
| token_0/layer_07/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_07/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L3:c2 | 160 | 640.000 |
| token_0/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_07/model_list.7.attention.k_weight.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_07/model_list.7.attention.k_weight.weight \| token_1/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_07/model_list.7.attention.k_weight.weight \| token_1/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L2:c2 | 937 | 2,811.000 |
| token_0/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_0/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_07/model_list.7.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_07/model_list.7.attention.o_weight.weight \| token_1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L3:c2 | 10 | 40.000 |
| token_0/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_07/model_list.7.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_07/model_list.7.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L2:c2 | 25 | 75.000 |
| token_0/layer_07/model_list.7.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_07/model_list.7.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_07/model_list.7.attention.q_weight.weight \| token_1/layer_07/model_list.7.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L3:c2 | 160 | 640.000 |
| token_0/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_07/model_list.7.attention.v_weight.weight | plaintext:L4:c1 | 5 | 12.500 |
| token_0/layer_07/model_list.7.attention.v_weight.weight \| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_07/model_list.7.attention.v_weight.weight \| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_07/model_list.7.post_Normal.weight | ciphertext:L2:c2 | 89 | 267.000 |
| token_0/layer_07/model_list.7.post_Normal.weight | ciphertext:L3:c2 | 17 | 68.000 |
| token_0/layer_07/model_list.7.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_07/model_list.7.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_07/model_list.7.post_Normal.weight \| token_1/layer_07/model_list.7.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_07/model_list.7.pre_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_0/layer_07/model_list.7.pre_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_07/model_list.7.pre_Normal.weight | plaintext:L4:c1 | 5 | 12.500 |
| token_0/layer_07/model_list.7.pre_Normal.weight \| token_1/layer_07/model_list.7.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_07/model_list.7.pre_Normal.weight \| token_1/layer_07/model_list.7.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_07/rope | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_07/rope | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_07/swiglu | ciphertext:L1:c2 | 1342 | 2,684.000 |
| token_0/layer_07/swiglu | ciphertext:L2:c2 | 5866 | 17,598.000 |
| token_0/layer_07/swiglu | ciphertext:L2:c3 | 19 | 85.500 |
| token_0/layer_07/swiglu | ciphertext:L3:c2 | 9491 | 37,964.000 |
| token_0/layer_07/swiglu | ciphertext:L3:c3 | 13 | 78.000 |
| token_0/layer_07/swiglu | ciphertext:L4:c2 | 428 | 2,140.000 |
| token_0/layer_07/swiglu | ciphertext:L4:c3 | 11 | 82.500 |
| token_0/layer_07/swiglu | ciphertext:L5:c2 | 46 | 276.000 |
| token_0/layer_07/swiglu | ciphertext:L5:c3 | 3 | 27.000 |
| token_0/layer_07/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_07/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_07/swiglu | plaintext:L4:c1 | 262 | 655.000 |
| token_0/layer_07/swiglu | plaintext:L5:c1 | 2 | 6.000 |
| token_0/layer_07/swiglu \| token_1/layer_07/swiglu | ciphertext:L1:c2 | 26 | 52.000 |
| token_0/layer_07/swiglu \| token_1/layer_07/swiglu | ciphertext:L2:c2 | 97 | 291.000 |
| token_0/layer_07/swiglu \| token_1/layer_07/swiglu | ciphertext:L3:c2 | 291 | 1,164.000 |
| token_0/layer_07/swiglu \| token_1/layer_07/swiglu | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_07/swiglu \| token_1/layer_07/swiglu | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_08/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_08/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_08/gqa_attention | ciphertext:L3:c2 | 20 | 80.000 |
| token_0/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c2 | 153 | 612.000 |
| token_0/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_08/model_list.8.attention.k_weight.weight | plaintext:L4:c1 | 11 | 27.500 |
| token_0/layer_08/model_list.8.attention.k_weight.weight \| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 954 | 3,816.000 |
| token_0/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c2 | 42 | 210.000 |
| token_0/layer_08/model_list.8.attention.o_weight.weight | plaintext:L4:c1 | 83 | 207.500 |
| token_0/layer_08/model_list.8.attention.o_weight.weight \| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L3:c2 | 10 | 40.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L1:c2 | 16 | 32.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L3:c2 | 32 | 128.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_08/model_list.8.attention.q_weight.weight \| token_1/layer_08/model_list.8.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L3:c2 | 152 | 608.000 |
| token_0/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_08/model_list.8.attention.v_weight.weight | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_08/model_list.8.attention.v_weight.weight \| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_08/model_list.8.post_Normal.weight | ciphertext:L2:c2 | 15 | 45.000 |
| token_0/layer_08/model_list.8.post_Normal.weight | ciphertext:L3:c2 | 93 | 372.000 |
| token_0/layer_08/model_list.8.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_08/model_list.8.post_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_08/model_list.8.post_Normal.weight \| token_1/layer_08/model_list.8.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_08/model_list.8.pre_Normal.weight | ciphertext:L3:c2 | 96 | 384.000 |
| token_0/layer_08/model_list.8.pre_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_08/model_list.8.pre_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_08/model_list.8.pre_Normal.weight \| token_1/layer_08/model_list.8.pre_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_08/rope | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_08/rope | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_08/swiglu | ciphertext:L1:c2 | 931 | 1,862.000 |
| token_0/layer_08/swiglu | ciphertext:L2:c2 | 4770 | 14,310.000 |
| token_0/layer_08/swiglu | ciphertext:L2:c3 | 11 | 49.500 |
| token_0/layer_08/swiglu | ciphertext:L3:c2 | 10865 | 43,460.000 |
| token_0/layer_08/swiglu | ciphertext:L3:c3 | 8 | 48.000 |
| token_0/layer_08/swiglu | ciphertext:L4:c2 | 421 | 2,105.000 |
| token_0/layer_08/swiglu | ciphertext:L4:c3 | 8 | 60.000 |
| token_0/layer_08/swiglu | ciphertext:L5:c2 | 20 | 120.000 |
| token_0/layer_08/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_08/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_08/swiglu | plaintext:L4:c1 | 628 | 1,570.000 |
| token_0/layer_08/swiglu \| token_1/layer_08/swiglu | ciphertext:L1:c2 | 16 | 32.000 |
| token_0/layer_08/swiglu \| token_1/layer_08/swiglu | ciphertext:L2:c2 | 111 | 333.000 |
| token_0/layer_08/swiglu \| token_1/layer_08/swiglu | ciphertext:L3:c2 | 276 | 1,104.000 |
| token_0/layer_08/swiglu \| token_1/layer_08/swiglu | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_08/swiglu \| token_1/layer_08/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_08/swiglu \| token_1/layer_08/swiglu | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_08/swiglu \| token_1/layer_08/swiglu | plaintext:L4:c1 | 30 | 75.000 |
| token_0/layer_09/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_09/gqa_attention | ciphertext:L1:c2 | 15 | 30.000 |
| token_0/layer_09/gqa_attention | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L1:c2 | 119 | 238.000 |
| token_0/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L2:c2 | 42 | 126.000 |
| token_0/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_09/model_list.9.attention.k_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_09/model_list.9.attention.k_weight.weight \| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_09/model_list.9.attention.k_weight.weight \| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_09/model_list.9.attention.k_weight.weight \| token_1/layer_09/model_list.9.attention.k_weight.weight | plaintext:L5:c1 | 2 | 6.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L1:c2 | 95 | 190.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L2:c2 | 33 | 99.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L2:c3 | 4 | 18.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L3:c2 | 852 | 3,408.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L4:c2 | 45 | 225.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L5:c2 | 7 | 42.000 |
| token_0/layer_09/model_list.9.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_09/model_list.9.attention.o_weight.weight \| token_1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L1:c2 | 25 | 50.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L1:c2 | 10 | 20.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_09/model_list.9.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L3:c2 | 26 | 104.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L4:c2 | 9 | 45.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_09/model_list.9.attention.q_weight.weight \| token_1/layer_09/model_list.9.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L1:c2 | 118 | 236.000 |
| token_0/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L2:c2 | 42 | 126.000 |
| token_0/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_09/model_list.9.attention.v_weight.weight \| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_09/model_list.9.attention.v_weight.weight \| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_09/model_list.9.post_Normal.weight | ciphertext:L3:c2 | 101 | 404.000 |
| token_0/layer_09/model_list.9.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_09/model_list.9.post_Normal.weight \| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_09/model_list.9.pre_Normal.weight | ciphertext:L1:c2 | 18 | 36.000 |
| token_0/layer_09/model_list.9.pre_Normal.weight | ciphertext:L2:c2 | 91 | 273.000 |
| token_0/layer_09/model_list.9.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_09/model_list.9.pre_Normal.weight \| token_1/layer_09/model_list.9.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_09/model_list.9.pre_Normal.weight \| token_1/layer_09/model_list.9.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_09/rope | ciphertext:L1:c2 | 9 | 18.000 |
| token_0/layer_09/rope | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_09/swiglu | ciphertext:L1:c2 | 991 | 1,982.000 |
| token_0/layer_09/swiglu | ciphertext:L2:c2 | 4615 | 13,845.000 |
| token_0/layer_09/swiglu | ciphertext:L2:c3 | 9 | 40.500 |
| token_0/layer_09/swiglu | ciphertext:L3:c2 | 10957 | 43,828.000 |
| token_0/layer_09/swiglu | ciphertext:L3:c3 | 9 | 54.000 |
| token_0/layer_09/swiglu | ciphertext:L4:c2 | 497 | 2,485.000 |
| token_0/layer_09/swiglu | ciphertext:L4:c3 | 4 | 30.000 |
| token_0/layer_09/swiglu | ciphertext:L5:c2 | 34 | 204.000 |
| token_0/layer_09/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_0/layer_09/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_09/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_09/swiglu | plaintext:L4:c1 | 407 | 1,017.500 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | ciphertext:L1:c2 | 53 | 106.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | ciphertext:L2:c2 | 238 | 714.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | ciphertext:L3:c2 | 218 | 872.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | ciphertext:L4:c2 | 30 | 150.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | ciphertext:L5:c2 | 18 | 108.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | plaintext:L3:c1 | 7 | 14.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_09/swiglu \| token_1/layer_09/swiglu | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_10/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_10/gqa_attention | ciphertext:L1:c2 | 15 | 30.000 |
| token_0/layer_10/gqa_attention | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L2:c2 | 144 | 432.000 |
| token_0/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L3:c2 | 17 | 68.000 |
| token_0/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_10/model_list.10.attention.k_weight.weight \| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L1:c2 | 461 | 922.000 |
| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 278 | 834.000 |
| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L3:c2 | 285 | 1,140.000 |
| token_0/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L4:c2 | 11 | 55.000 |
| token_0/layer_10/model_list.10.attention.o_weight.weight | plaintext:L4:c1 | 5 | 12.500 |
| token_0/layer_10/model_list.10.attention.o_weight.weight \| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 22 | 66.000 |
| token_0/layer_10/model_list.10.attention.o_weight.weight \| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_10/model_list.10.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L3:c2 | 49 | 196.000 |
| token_0/layer_10/model_list.10.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_10/model_list.10.attention.q_weight.weight \| token_1/layer_10/model_list.10.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L1:c2 | 7 | 14.000 |
| token_0/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L2:c2 | 142 | 426.000 |
| token_0/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L3:c2 | 16 | 64.000 |
| token_0/layer_10/model_list.10.attention.v_weight.weight \| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_10/model_list.10.attention.v_weight.weight \| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_10/model_list.10.attention.v_weight.weight \| token_1/layer_10/model_list.10.attention.v_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_10/model_list.10.post_Normal.weight | ciphertext:L3:c2 | 94 | 376.000 |
| token_0/layer_10/model_list.10.post_Normal.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_10/model_list.10.post_Normal.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_10/model_list.10.post_Normal.weight \| token_1/layer_10/model_list.10.post_Normal.weight | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_10/model_list.10.pre_Normal.weight | ciphertext:L2:c2 | 46 | 138.000 |
| token_0/layer_10/model_list.10.pre_Normal.weight | ciphertext:L3:c2 | 56 | 224.000 |
| token_0/layer_10/model_list.10.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_10/model_list.10.pre_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_10/model_list.10.pre_Normal.weight \| token_1/layer_10/model_list.10.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_10/rope | ciphertext:L1:c2 | 11 | 22.000 |
| token_0/layer_10/rope | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_10/swiglu | ciphertext:L1:c2 | 1335 | 2,670.000 |
| token_0/layer_10/swiglu | ciphertext:L2:c2 | 5079 | 15,237.000 |
| token_0/layer_10/swiglu | ciphertext:L2:c3 | 12 | 54.000 |
| token_0/layer_10/swiglu | ciphertext:L3:c2 | 10327 | 41,308.000 |
| token_0/layer_10/swiglu | ciphertext:L3:c3 | 10 | 60.000 |
| token_0/layer_10/swiglu | ciphertext:L4:c2 | 423 | 2,115.000 |
| token_0/layer_10/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_10/swiglu | ciphertext:L5:c2 | 26 | 156.000 |
| token_0/layer_10/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_10/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_10/swiglu | plaintext:L4:c1 | 286 | 715.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | ciphertext:L1:c2 | 27 | 54.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | ciphertext:L2:c2 | 57 | 171.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | ciphertext:L3:c2 | 194 | 776.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | ciphertext:L4:c2 | 127 | 635.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | ciphertext:L5:c2 | 7 | 42.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | plaintext:L4:c1 | 10 | 25.000 |
| token_0/layer_10/swiglu \| token_1/layer_10/swiglu | plaintext:L5:c1 | 4 | 12.000 |
| token_0/layer_11/_decoder_block | ciphertext:L1:c2 | 5 | 10.000 |
| token_0/layer_11/gqa_attention | ciphertext:L3:c2 | 18 | 72.000 |
| token_0/layer_11/gqa_attention | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_11/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L3:c2 | 150 | 600.000 |
| token_0/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_0/layer_11/model_list.11.attention.k_weight.weight | plaintext:L4:c1 | 16 | 40.000 |
| token_0/layer_11/model_list.11.attention.k_weight.weight \| token_1/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_11/model_list.11.attention.k_weight.weight \| token_1/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L1:c2 | 92 | 184.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 879 | 2,637.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L3:c2 | 112 | 448.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_11/model_list.11.attention.o_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L1:c2 | 10 | 20.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 13 | 39.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_11/model_list.11.attention.o_weight.weight \| token_1/layer_11/model_list.11.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_11/model_list.11.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_11/model_list.11.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L2:c2 | 21 | 63.000 |
| token_0/layer_11/model_list.11.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L3:c2 | 28 | 112.000 |
| token_0/layer_11/model_list.11.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_11/model_list.11.attention.q_weight.weight \| token_1/layer_11/model_list.11.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L3:c2 | 150 | 600.000 |
| token_0/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_11/model_list.11.attention.v_weight.weight | plaintext:L4:c1 | 15 | 37.500 |
| token_0/layer_11/model_list.11.attention.v_weight.weight \| token_1/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_11/model_list.11.post_Normal.weight | ciphertext:L1:c2 | 85 | 170.000 |
| token_0/layer_11/model_list.11.post_Normal.weight | ciphertext:L2:c2 | 28 | 84.000 |
| token_0/layer_11/model_list.11.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_11/model_list.11.post_Normal.weight \| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L3:c3 | 3 | 18.000 |
| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_11/model_list.11.pre_Normal.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_11/model_list.11.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_11/model_list.11.pre_Normal.weight \| token_1/layer_11/model_list.11.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_11/rope | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_11/swiglu | ciphertext:L1:c2 | 509 | 1,018.000 |
| token_0/layer_11/swiglu | ciphertext:L2:c2 | 4008 | 12,024.000 |
| token_0/layer_11/swiglu | ciphertext:L2:c3 | 10 | 45.000 |
| token_0/layer_11/swiglu | ciphertext:L3:c2 | 7903 | 31,612.000 |
| token_0/layer_11/swiglu | ciphertext:L3:c3 | 8 | 48.000 |
| token_0/layer_11/swiglu | ciphertext:L4:c2 | 4577 | 22,885.000 |
| token_0/layer_11/swiglu | ciphertext:L4:c3 | 6 | 45.000 |
| token_0/layer_11/swiglu | ciphertext:L5:c2 | 171 | 1,026.000 |
| token_0/layer_11/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_11/swiglu | plaintext:L2:c1 | 16 | 24.000 |
| token_0/layer_11/swiglu | plaintext:L4:c1 | 17 | 42.500 |
| token_0/layer_11/swiglu | plaintext:L5:c1 | 280 | 840.000 |
| token_0/layer_11/swiglu \| token_1/layer_11/swiglu | ciphertext:L1:c2 | 20 | 40.000 |
| token_0/layer_11/swiglu \| token_1/layer_11/swiglu | ciphertext:L2:c2 | 58 | 174.000 |
| token_0/layer_11/swiglu \| token_1/layer_11/swiglu | ciphertext:L3:c2 | 457 | 1,828.000 |
| token_0/layer_11/swiglu \| token_1/layer_11/swiglu | ciphertext:L4:c2 | 11 | 55.000 |
| token_0/layer_11/swiglu \| token_1/layer_11/swiglu | plaintext:L4:c1 | 24 | 60.000 |
| token_0/layer_12/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_12/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_12/gqa_attention | ciphertext:L2:c2 | 18 | 54.000 |
| token_0/layer_12/gqa_attention | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L3:c2 | 131 | 524.000 |
| token_0/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L4:c2 | 22 | 110.000 |
| token_0/layer_12/model_list.12.attention.k_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_12/model_list.12.attention.k_weight.weight \| token_1/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_12/model_list.12.attention.k_weight.weight \| token_1/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L3:c2 | 990 | 3,960.000 |
| token_0/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 50 | 250.000 |
| token_0/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_12/model_list.12.attention.o_weight.weight | plaintext:L4:c1 | 87 | 217.500 |
| token_0/layer_12/model_list.12.attention.o_weight.weight | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_12/model_list.12.attention.o_weight.weight \| token_1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_12/model_list.12.attention.o_weight.weight \| token_1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_12/model_list.12.attention.o_weight.weight \| token_1/layer_12/model_list.12.attention.o_weight.weight | plaintext:L4:c1 | 3 | 7.500 |
| token_0/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_12/model_list.12.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L2:c2 | 25 | 75.000 |
| token_0/layer_12/model_list.12.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_12/model_list.12.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_12/model_list.12.attention.q_weight.weight \| token_1/layer_12/model_list.12.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L3:c2 | 138 | 552.000 |
| token_0/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L4:c2 | 17 | 85.000 |
| token_0/layer_12/model_list.12.attention.v_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_12/model_list.12.attention.v_weight.weight \| token_1/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_12/model_list.12.attention.v_weight.weight \| token_1/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_12/model_list.12.post_Normal.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_12/model_list.12.post_Normal.weight | ciphertext:L3:c2 | 108 | 432.000 |
| token_0/layer_12/model_list.12.post_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_12/model_list.12.post_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_12/model_list.12.post_Normal.weight \| token_1/layer_12/model_list.12.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_12/model_list.12.pre_Normal.weight | ciphertext:L1:c2 | 80 | 160.000 |
| token_0/layer_12/model_list.12.pre_Normal.weight | ciphertext:L2:c2 | 16 | 48.000 |
| token_0/layer_12/model_list.12.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_12/model_list.12.pre_Normal.weight | ciphertext:L5:c2 | 12 | 72.000 |
| token_0/layer_12/model_list.12.pre_Normal.weight | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_12/model_list.12.pre_Normal.weight \| token_1/layer_12/model_list.12.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_12/rope | ciphertext:L2:c2 | 8 | 24.000 |
| token_0/layer_12/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_12/swiglu | ciphertext:L1:c2 | 1133 | 2,266.000 |
| token_0/layer_12/swiglu | ciphertext:L2:c2 | 4123 | 12,369.000 |
| token_0/layer_12/swiglu | ciphertext:L2:c3 | 16 | 72.000 |
| token_0/layer_12/swiglu | ciphertext:L3:c2 | 11199 | 44,796.000 |
| token_0/layer_12/swiglu | ciphertext:L3:c3 | 17 | 102.000 |
| token_0/layer_12/swiglu | ciphertext:L4:c2 | 545 | 2,725.000 |
| token_0/layer_12/swiglu | ciphertext:L4:c3 | 12 | 90.000 |
| token_0/layer_12/swiglu | ciphertext:L5:c2 | 79 | 474.000 |
| token_0/layer_12/swiglu | ciphertext:L5:c3 | 3 | 27.000 |
| token_0/layer_12/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_12/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_12/swiglu | plaintext:L3:c1 | 37 | 74.000 |
| token_0/layer_12/swiglu | plaintext:L4:c1 | 374 | 935.000 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | ciphertext:L1:c2 | 39 | 78.000 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | ciphertext:L2:c2 | 99 | 297.000 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | ciphertext:L3:c2 | 373 | 1,492.000 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | ciphertext:L4:c2 | 37 | 185.000 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | plaintext:L3:c1 | 2 | 4.000 |
| token_0/layer_12/swiglu \| token_1/layer_12/swiglu | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_13/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_13/_decoder_block | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_13/_decoder_block | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_13/gqa_attention | ciphertext:L2:c2 | 18 | 54.000 |
| token_0/layer_13/gqa_attention | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L2:c2 | 149 | 447.000 |
| token_0/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L3:c2 | 17 | 68.000 |
| token_0/layer_13/model_list.13.attention.k_weight.weight \| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_13/model_list.13.attention.k_weight.weight \| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L1:c2 | 625 | 1,250.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c2 | 203 | 609.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L3:c2 | 23 | 92.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L4:c2 | 170 | 850.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L5:c2 | 10 | 60.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight | plaintext:L5:c1 | 6 | 18.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight \| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c2 | 21 | 63.000 |
| token_0/layer_13/model_list.13.attention.o_weight.weight \| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L1:c2 | 19 | 38.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_13/model_list.13.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_13/model_list.13.attention.q_weight.weight \| token_1/layer_13/model_list.13.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L2:c2 | 148 | 444.000 |
| token_0/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L3:c2 | 17 | 68.000 |
| token_0/layer_13/model_list.13.attention.v_weight.weight \| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_13/model_list.13.attention.v_weight.weight \| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_13/model_list.13.attention.v_weight.weight \| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_13/model_list.13.post_Normal.weight | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_13/model_list.13.post_Normal.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_13/model_list.13.post_Normal.weight | ciphertext:L4:c2 | 91 | 455.000 |
| token_0/layer_13/model_list.13.post_Normal.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_0/layer_13/model_list.13.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_13/model_list.13.post_Normal.weight | plaintext:L5:c1 | 3 | 9.000 |
| token_0/layer_13/model_list.13.post_Normal.weight \| token_1/layer_13/model_list.13.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_13/model_list.13.pre_Normal.weight | ciphertext:L2:c2 | 100 | 300.000 |
| token_0/layer_13/model_list.13.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_13/model_list.13.pre_Normal.weight \| token_1/layer_13/model_list.13.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_13/rope | ciphertext:L1:c2 | 1 | 2.000 |
| token_0/layer_13/rope | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_13/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_13/swiglu | ciphertext:L1:c2 | 457 | 914.000 |
| token_0/layer_13/swiglu | ciphertext:L2:c2 | 1649 | 4,947.000 |
| token_0/layer_13/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_13/swiglu | ciphertext:L3:c2 | 14328 | 57,312.000 |
| token_0/layer_13/swiglu | ciphertext:L3:c3 | 2 | 12.000 |
| token_0/layer_13/swiglu | ciphertext:L4:c2 | 540 | 2,700.000 |
| token_0/layer_13/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_13/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_13/swiglu | plaintext:L4:c1 | 780 | 1,950.000 |
| token_0/layer_13/swiglu \| token_1/layer_13/swiglu | ciphertext:L1:c2 | 46 | 92.000 |
| token_0/layer_13/swiglu \| token_1/layer_13/swiglu | ciphertext:L2:c2 | 89 | 267.000 |
| token_0/layer_13/swiglu \| token_1/layer_13/swiglu | ciphertext:L3:c2 | 262 | 1,048.000 |
| token_0/layer_13/swiglu \| token_1/layer_13/swiglu | ciphertext:L4:c2 | 140 | 700.000 |
| token_0/layer_13/swiglu \| token_1/layer_13/swiglu | ciphertext:L5:c2 | 6 | 36.000 |
| token_0/layer_13/swiglu \| token_1/layer_13/swiglu | plaintext:L4:c1 | 17 | 42.500 |
| token_0/layer_13/swiglu \| token_1/layer_13/swiglu | plaintext:L5:c1 | 6 | 18.000 |
| token_0/layer_14/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_14/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_14/gqa_attention | ciphertext:L2:c2 | 19 | 57.000 |
| token_0/layer_14/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L2:c2 | 149 | 447.000 |
| token_0/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L3:c2 | 15 | 60.000 |
| token_0/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_14/model_list.14.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_14/model_list.14.attention.k_weight.weight \| token_1/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_14/model_list.14.attention.k_weight.weight \| token_1/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L2:c2 | 937 | 2,811.000 |
| token_0/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_0/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_14/model_list.14.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_14/model_list.14.attention.o_weight.weight \| token_1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_14/model_list.14.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_14/model_list.14.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_14/model_list.14.attention.q_weight.weight \| token_1/layer_14/model_list.14.attention.q_weight.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L2:c2 | 149 | 447.000 |
| token_0/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L3:c2 | 16 | 64.000 |
| token_0/layer_14/model_list.14.attention.v_weight.weight \| token_1/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_14/model_list.14.post_Normal.weight | ciphertext:L1:c2 | 6 | 12.000 |
| token_0/layer_14/model_list.14.post_Normal.weight | ciphertext:L2:c2 | 93 | 279.000 |
| token_0/layer_14/model_list.14.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_14/model_list.14.post_Normal.weight \| token_1/layer_14/model_list.14.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_14/model_list.14.pre_Normal.weight | ciphertext:L2:c2 | 97 | 291.000 |
| token_0/layer_14/model_list.14.pre_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_14/model_list.14.pre_Normal.weight \| token_1/layer_14/model_list.14.pre_Normal.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_14/rope | ciphertext:L2:c2 | 11 | 33.000 |
| token_0/layer_14/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_14/swiglu | ciphertext:L1:c2 | 660 | 1,320.000 |
| token_0/layer_14/swiglu | ciphertext:L2:c2 | 3342 | 10,026.000 |
| token_0/layer_14/swiglu | ciphertext:L2:c3 | 17 | 76.500 |
| token_0/layer_14/swiglu | ciphertext:L3:c2 | 8974 | 35,896.000 |
| token_0/layer_14/swiglu | ciphertext:L3:c3 | 17 | 102.000 |
| token_0/layer_14/swiglu | ciphertext:L4:c2 | 3902 | 19,510.000 |
| token_0/layer_14/swiglu | ciphertext:L4:c3 | 7 | 52.500 |
| token_0/layer_14/swiglu | ciphertext:L5:c2 | 197 | 1,182.000 |
| token_0/layer_14/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_0/layer_14/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_0/layer_14/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_14/swiglu | plaintext:L4:c1 | 132 | 330.000 |
| token_0/layer_14/swiglu | plaintext:L5:c1 | 137 | 411.000 |
| token_0/layer_14/swiglu \| token_1/layer_14/swiglu | ciphertext:L1:c2 | 82 | 164.000 |
| token_0/layer_14/swiglu \| token_1/layer_14/swiglu | ciphertext:L2:c2 | 258 | 774.000 |
| token_0/layer_14/swiglu \| token_1/layer_14/swiglu | ciphertext:L3:c2 | 74 | 296.000 |
| token_0/layer_14/swiglu \| token_1/layer_14/swiglu | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_14/swiglu \| token_1/layer_14/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_14/swiglu \| token_1/layer_14/swiglu | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_15/_decoder_block | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_15/_decoder_block | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_15/gqa_attention | ciphertext:L2:c2 | 19 | 57.000 |
| token_0/layer_15/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L2:c2 | 148 | 444.000 |
| token_0/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L3:c2 | 18 | 72.000 |
| token_0/layer_15/model_list.15.attention.k_weight.weight \| token_1/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 937 | 2,811.000 |
| token_0/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_0/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_15/model_list.15.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_15/model_list.15.attention.o_weight.weight \| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 25 | 75.000 |
| token_0/layer_15/model_list.15.attention.o_weight.weight \| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L3:c2 | 21 | 84.000 |
| token_0/layer_15/model_list.15.attention.o_weight.weight \| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_15/model_list.15.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L3:c2 | 46 | 184.000 |
| token_0/layer_15/model_list.15.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_15/model_list.15.attention.q_weight.weight \| token_1/layer_15/model_list.15.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L2:c2 | 148 | 444.000 |
| token_0/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L3:c2 | 15 | 60.000 |
| token_0/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_15/model_list.15.attention.v_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_15/model_list.15.attention.v_weight.weight \| token_1/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_15/model_list.15.post_Normal.weight | ciphertext:L1:c2 | 26 | 52.000 |
| token_0/layer_15/model_list.15.post_Normal.weight | ciphertext:L2:c2 | 74 | 222.000 |
| token_0/layer_15/model_list.15.post_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_15/model_list.15.post_Normal.weight \| token_1/layer_15/model_list.15.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_15/model_list.15.pre_Normal.weight | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_15/model_list.15.pre_Normal.weight | ciphertext:L3:c2 | 90 | 360.000 |
| token_0/layer_15/model_list.15.pre_Normal.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_15/model_list.15.pre_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_15/model_list.15.pre_Normal.weight \| token_1/layer_15/model_list.15.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_15/rope | ciphertext:L2:c2 | 9 | 27.000 |
| token_0/layer_15/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_15/swiglu | ciphertext:L1:c2 | 1353 | 2,706.000 |
| token_0/layer_15/swiglu | ciphertext:L2:c2 | 4697 | 14,091.000 |
| token_0/layer_15/swiglu | ciphertext:L2:c3 | 19 | 85.500 |
| token_0/layer_15/swiglu | ciphertext:L3:c2 | 10559 | 42,236.000 |
| token_0/layer_15/swiglu | ciphertext:L3:c3 | 11 | 66.000 |
| token_0/layer_15/swiglu | ciphertext:L4:c2 | 418 | 2,090.000 |
| token_0/layer_15/swiglu | ciphertext:L4:c3 | 3 | 22.500 |
| token_0/layer_15/swiglu | ciphertext:L5:c2 | 32 | 192.000 |
| token_0/layer_15/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_15/swiglu | plaintext:L4:c1 | 298 | 745.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L1:c2 | 9 | 18.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L2:c2 | 344 | 1,032.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L3:c2 | 250 | 1,000.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L4:c2 | 79 | 395.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_15/swiglu \| token_1/layer_15/swiglu | plaintext:L5:c1 | 18 | 54.000 |
| token_0/layer_16/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_16/_decoder_block | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_16/gqa_attention | ciphertext:L3:c2 | 19 | 76.000 |
| token_0/layer_16/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L3:c2 | 154 | 616.000 |
| token_0/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_16/model_list.16.attention.k_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_16/model_list.16.attention.k_weight.weight \| token_1/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 1004 | 4,016.000 |
| token_0/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c2 | 39 | 195.000 |
| token_0/layer_16/model_list.16.attention.o_weight.weight | plaintext:L4:c1 | 90 | 225.000 |
| token_0/layer_16/model_list.16.attention.o_weight.weight \| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L1:c2 | 10 | 20.000 |
| token_0/layer_16/model_list.16.attention.o_weight.weight \| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_16/model_list.16.attention.o_weight.weight \| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_16/model_list.16.attention.q_weight.weight | ciphertext:L3:c2 | 13 | 52.000 |
| token_0/layer_16/model_list.16.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.q_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_16/model_list.16.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.q_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_16/model_list.16.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_16/model_list.16.attention.q_weight.weight \| token_1/layer_16/model_list.16.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L3:c2 | 150 | 600.000 |
| token_0/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_0/layer_16/model_list.16.attention.v_weight.weight | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_16/model_list.16.attention.v_weight.weight \| token_1/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_16/model_list.16.post_Normal.weight | ciphertext:L2:c2 | 106 | 318.000 |
| token_0/layer_16/model_list.16.post_Normal.weight | ciphertext:L3:c2 | 8 | 32.000 |
| token_0/layer_16/model_list.16.post_Normal.weight \| token_1/layer_16/model_list.16.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_16/model_list.16.pre_Normal.weight | ciphertext:L2:c2 | 80 | 240.000 |
| token_0/layer_16/model_list.16.pre_Normal.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_16/model_list.16.pre_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_16/model_list.16.pre_Normal.weight \| token_1/layer_16/model_list.16.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_16/rope | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_16/swiglu | ciphertext:L1:c2 | 2248 | 4,496.000 |
| token_0/layer_16/swiglu | ciphertext:L2:c2 | 2989 | 8,967.000 |
| token_0/layer_16/swiglu | ciphertext:L2:c3 | 15 | 67.500 |
| token_0/layer_16/swiglu | ciphertext:L3:c2 | 11203 | 44,812.000 |
| token_0/layer_16/swiglu | ciphertext:L3:c3 | 8 | 48.000 |
| token_0/layer_16/swiglu | ciphertext:L4:c2 | 508 | 2,540.000 |
| token_0/layer_16/swiglu | ciphertext:L4:c3 | 6 | 45.000 |
| token_0/layer_16/swiglu | ciphertext:L5:c2 | 74 | 444.000 |
| token_0/layer_16/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_16/swiglu | plaintext:L2:c1 | 34 | 51.000 |
| token_0/layer_16/swiglu | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_16/swiglu | plaintext:L4:c1 | 446 | 1,115.000 |
| token_0/layer_16/swiglu | plaintext:L5:c1 | 1 | 3.000 |
| token_0/layer_16/swiglu \| token_1/layer_16/swiglu | ciphertext:L1:c2 | 57 | 114.000 |
| token_0/layer_16/swiglu \| token_1/layer_16/swiglu | ciphertext:L2:c2 | 150 | 450.000 |
| token_0/layer_16/swiglu \| token_1/layer_16/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_16/swiglu \| token_1/layer_16/swiglu | ciphertext:L3:c2 | 179 | 716.000 |
| token_0/layer_16/swiglu \| token_1/layer_16/swiglu | ciphertext:L4:c2 | 33 | 165.000 |
| token_0/layer_16/swiglu \| token_1/layer_16/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_16/swiglu \| token_1/layer_16/swiglu | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_17/_decoder_block | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_17/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_17/_decoder_block | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_17/gqa_attention | ciphertext:L2:c2 | 17 | 51.000 |
| token_0/layer_17/gqa_attention | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L2:c2 | 149 | 447.000 |
| token_0/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L3:c2 | 16 | 64.000 |
| token_0/layer_17/model_list.17.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_17/model_list.17.attention.k_weight.weight \| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L2:c2 | 919 | 2,757.000 |
| token_0/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L3:c2 | 161 | 644.000 |
| token_0/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_17/model_list.17.attention.o_weight.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_17/model_list.17.attention.o_weight.weight \| token_1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L2:c2 | 23 | 69.000 |
| token_0/layer_17/model_list.17.attention.o_weight.weight \| token_1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L3:c2 | 27 | 108.000 |
| token_0/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L2:c2 | 12 | 36.000 |
| token_0/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_17/model_list.17.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_17/model_list.17.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L2:c2 | 45 | 135.000 |
| token_0/layer_17/model_list.17.attention.q_weight.weight \| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L2:c2 | 148 | 444.000 |
| token_0/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L3:c2 | 16 | 64.000 |
| token_0/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_17/model_list.17.attention.v_weight.weight \| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L1:c2 | 4 | 8.000 |
| token_0/layer_17/model_list.17.post_Normal.weight | ciphertext:L2:c2 | 106 | 318.000 |
| token_0/layer_17/model_list.17.post_Normal.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_17/model_list.17.post_Normal.weight \| token_1/layer_17/model_list.17.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_17/model_list.17.pre_Normal.weight | ciphertext:L2:c2 | 106 | 318.000 |
| token_0/layer_17/model_list.17.pre_Normal.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_17/model_list.17.pre_Normal.weight \| token_1/layer_17/model_list.17.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_17/rope | ciphertext:L2:c2 | 10 | 30.000 |
| token_0/layer_17/rope | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_17/swiglu | ciphertext:L1:c2 | 1458 | 2,916.000 |
| token_0/layer_17/swiglu | ciphertext:L2:c2 | 3943 | 11,829.000 |
| token_0/layer_17/swiglu | ciphertext:L2:c3 | 19 | 85.500 |
| token_0/layer_17/swiglu | ciphertext:L3:c2 | 11181 | 44,724.000 |
| token_0/layer_17/swiglu | ciphertext:L3:c3 | 15 | 90.000 |
| token_0/layer_17/swiglu | ciphertext:L4:c2 | 455 | 2,275.000 |
| token_0/layer_17/swiglu | ciphertext:L4:c3 | 8 | 60.000 |
| token_0/layer_17/swiglu | ciphertext:L5:c2 | 33 | 198.000 |
| token_0/layer_17/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_0/layer_17/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_0/layer_17/swiglu | plaintext:L3:c1 | 16 | 32.000 |
| token_0/layer_17/swiglu | plaintext:L4:c1 | 477 | 1,192.500 |
| token_0/layer_17/swiglu \| token_1/layer_17/swiglu | ciphertext:L1:c2 | 48 | 96.000 |
| token_0/layer_17/swiglu \| token_1/layer_17/swiglu | ciphertext:L2:c2 | 284 | 852.000 |
| token_0/layer_17/swiglu \| token_1/layer_17/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_17/swiglu \| token_1/layer_17/swiglu | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_17/swiglu \| token_1/layer_17/swiglu | ciphertext:L4:c2 | 198 | 990.000 |
| token_0/layer_17/swiglu \| token_1/layer_17/swiglu | ciphertext:L5:c2 | 8 | 48.000 |
| token_0/layer_17/swiglu \| token_1/layer_17/swiglu | plaintext:L5:c1 | 8 | 24.000 |
| token_0/layer_18/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_18/gqa_attention | ciphertext:L3:c2 | 16 | 64.000 |
| token_0/layer_18/gqa_attention | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_18/gqa_attention | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L3:c2 | 153 | 612.000 |
| token_0/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_18/model_list.18.attention.k_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_18/model_list.18.attention.k_weight.weight \| token_1/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 985 | 3,940.000 |
| token_0/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L4:c2 | 46 | 230.000 |
| token_0/layer_18/model_list.18.attention.o_weight.weight | plaintext:L4:c1 | 14 | 35.000 |
| token_0/layer_18/model_list.18.attention.o_weight.weight \| token_1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_18/model_list.18.attention.q_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_18/model_list.18.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_18/model_list.18.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_18/model_list.18.attention.q_weight.weight \| token_1/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L4:c2 | 25 | 125.000 |
| token_0/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L3:c2 | 154 | 616.000 |
| token_0/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_18/model_list.18.attention.v_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_18/model_list.18.attention.v_weight.weight \| token_1/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_18/model_list.18.post_Normal.weight | ciphertext:L3:c2 | 102 | 408.000 |
| token_0/layer_18/model_list.18.post_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_18/model_list.18.post_Normal.weight \| token_1/layer_18/model_list.18.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_18/model_list.18.pre_Normal.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_0/layer_18/model_list.18.pre_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_0/layer_18/model_list.18.pre_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_18/model_list.18.pre_Normal.weight \| token_1/layer_18/model_list.18.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_18/rope | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_18/swiglu | ciphertext:L1:c2 | 1701 | 3,402.000 |
| token_0/layer_18/swiglu | ciphertext:L2:c2 | 4836 | 14,508.000 |
| token_0/layer_18/swiglu | ciphertext:L2:c3 | 16 | 72.000 |
| token_0/layer_18/swiglu | ciphertext:L3:c2 | 10150 | 40,600.000 |
| token_0/layer_18/swiglu | ciphertext:L3:c3 | 12 | 72.000 |
| token_0/layer_18/swiglu | ciphertext:L4:c2 | 446 | 2,230.000 |
| token_0/layer_18/swiglu | ciphertext:L4:c3 | 8 | 60.000 |
| token_0/layer_18/swiglu | ciphertext:L5:c2 | 37 | 222.000 |
| token_0/layer_18/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_0/layer_18/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_18/swiglu | plaintext:L4:c1 | 314 | 785.000 |
| token_0/layer_18/swiglu \| token_1/layer_18/swiglu | ciphertext:L1:c2 | 33 | 66.000 |
| token_0/layer_18/swiglu \| token_1/layer_18/swiglu | ciphertext:L2:c2 | 195 | 585.000 |
| token_0/layer_18/swiglu \| token_1/layer_18/swiglu | ciphertext:L3:c2 | 164 | 656.000 |
| token_0/layer_18/swiglu \| token_1/layer_18/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_18/swiglu \| token_1/layer_18/swiglu | ciphertext:L4:c2 | 25 | 125.000 |
| token_0/layer_18/swiglu \| token_1/layer_18/swiglu | ciphertext:L4:c3 | 2 | 15.000 |
| token_0/layer_18/swiglu \| token_1/layer_18/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_19/_decoder_block | ciphertext:L2:c2 | 6 | 18.000 |
| token_0/layer_19/gqa_attention | ciphertext:L3:c2 | 20 | 80.000 |
| token_0/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L3:c2 | 151 | 604.000 |
| token_0/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L4:c2 | 9 | 45.000 |
| token_0/layer_19/model_list.19.attention.k_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_19/model_list.19.attention.k_weight.weight \| token_1/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L2:c2 | 366 | 1,098.000 |
| token_0/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L3:c2 | 603 | 2,412.000 |
| token_0/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L4:c2 | 60 | 300.000 |
| token_0/layer_19/model_list.19.attention.o_weight.weight | plaintext:L4:c1 | 18 | 45.000 |
| token_0/layer_19/model_list.19.attention.o_weight.weight \| token_1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_19/model_list.19.attention.q_weight.weight | ciphertext:L3:c2 | 13 | 52.000 |
| token_0/layer_19/model_list.19.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.q_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_19/model_list.19.attention.q_weight.weight \| token_1/layer_19/model_list.19.attention.q_weight.weight | ciphertext:L3:c2 | 27 | 108.000 |
| token_0/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L3:c2 | 153 | 612.000 |
| token_0/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_19/model_list.19.attention.v_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_0/layer_19/model_list.19.attention.v_weight.weight \| token_1/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_19/model_list.19.post_Normal.weight | ciphertext:L2:c2 | 106 | 318.000 |
| token_0/layer_19/model_list.19.post_Normal.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_19/model_list.19.post_Normal.weight \| token_1/layer_19/model_list.19.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_19/model_list.19.pre_Normal.weight | ciphertext:L3:c2 | 98 | 392.000 |
| token_0/layer_19/model_list.19.pre_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_19/model_list.19.pre_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_19/model_list.19.pre_Normal.weight \| token_1/layer_19/model_list.19.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_19/rope | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_19/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_19/rope | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_19/swiglu | ciphertext:L1:c2 | 1755 | 3,510.000 |
| token_0/layer_19/swiglu | ciphertext:L2:c2 | 5205 | 15,615.000 |
| token_0/layer_19/swiglu | ciphertext:L2:c3 | 19 | 85.500 |
| token_0/layer_19/swiglu | ciphertext:L3:c2 | 5989 | 23,956.000 |
| token_0/layer_19/swiglu | ciphertext:L3:c3 | 12 | 72.000 |
| token_0/layer_19/swiglu | ciphertext:L4:c2 | 3993 | 19,965.000 |
| token_0/layer_19/swiglu | ciphertext:L4:c3 | 7 | 52.500 |
| token_0/layer_19/swiglu | ciphertext:L5:c2 | 191 | 1,146.000 |
| token_0/layer_19/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_19/swiglu | plaintext:L4:c1 | 90 | 225.000 |
| token_0/layer_19/swiglu | plaintext:L5:c1 | 155 | 465.000 |
| token_0/layer_19/swiglu \| token_1/layer_19/swiglu | ciphertext:L1:c2 | 35 | 70.000 |
| token_0/layer_19/swiglu \| token_1/layer_19/swiglu | ciphertext:L2:c2 | 193 | 579.000 |
| token_0/layer_19/swiglu \| token_1/layer_19/swiglu | ciphertext:L3:c2 | 184 | 736.000 |
| token_0/layer_19/swiglu \| token_1/layer_19/swiglu | ciphertext:L3:c3 | 1 | 6.000 |
| token_0/layer_19/swiglu \| token_1/layer_19/swiglu | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_19/swiglu \| token_1/layer_19/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_20/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_20/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_20/gqa_attention | ciphertext:L3:c2 | 18 | 72.000 |
| token_0/layer_20/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_20/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L3:c2 | 60 | 240.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L4:c2 | 91 | 455.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight | plaintext:L5:c1 | 4 | 12.000 |
| token_0/layer_20/model_list.20.attention.k_weight.weight \| token_1/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L2:c2 | 23 | 69.000 |
| token_0/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 933 | 3,732.000 |
| token_0/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L4:c2 | 41 | 205.000 |
| token_0/layer_20/model_list.20.attention.o_weight.weight | plaintext:L4:c1 | 82 | 205.000 |
| token_0/layer_20/model_list.20.attention.o_weight.weight \| token_1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_0/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_0/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L3:c3 | 2 | 12.000 |
| token_0/layer_20/model_list.20.attention.q_weight.weight \| token_1/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L2:c2 | 25 | 75.000 |
| token_0/layer_20/model_list.20.attention.q_weight.weight \| token_1/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L3:c2 | 26 | 104.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L3:c2 | 58 | 232.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L4:c2 | 91 | 455.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight | plaintext:L1:c1 | 1 | 1.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight | plaintext:L5:c1 | 4 | 12.000 |
| token_0/layer_20/model_list.20.attention.v_weight.weight \| token_1/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_20/model_list.20.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_20/model_list.20.post_Normal.weight | ciphertext:L3:c2 | 96 | 384.000 |
| token_0/layer_20/model_list.20.post_Normal.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_20/model_list.20.post_Normal.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_0/layer_20/model_list.20.post_Normal.weight \| token_1/layer_20/model_list.20.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_20/model_list.20.pre_Normal.weight | ciphertext:L2:c2 | 101 | 303.000 |
| token_0/layer_20/model_list.20.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_0/layer_20/model_list.20.pre_Normal.weight \| token_1/layer_20/model_list.20.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_20/model_list.20.pre_Normal.weight \| token_1/layer_20/model_list.20.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_20/rope | ciphertext:L3:c2 | 10 | 40.000 |
| token_0/layer_20/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_20/rope | plaintext:L4:c1 | 2 | 5.000 |
| token_0/layer_20/swiglu | ciphertext:L1:c2 | 967 | 1,934.000 |
| token_0/layer_20/swiglu | ciphertext:L2:c2 | 4720 | 14,160.000 |
| token_0/layer_20/swiglu | ciphertext:L2:c3 | 6 | 27.000 |
| token_0/layer_20/swiglu | ciphertext:L3:c2 | 10972 | 43,888.000 |
| token_0/layer_20/swiglu | ciphertext:L3:c3 | 4 | 24.000 |
| token_0/layer_20/swiglu | ciphertext:L4:c2 | 517 | 2,585.000 |
| token_0/layer_20/swiglu | ciphertext:L4:c3 | 4 | 30.000 |
| token_0/layer_20/swiglu | ciphertext:L5:c2 | 36 | 216.000 |
| token_0/layer_20/swiglu | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_20/swiglu | plaintext:L4:c1 | 286 | 715.000 |
| token_0/layer_20/swiglu \| token_1/layer_20/swiglu | ciphertext:L1:c2 | 67 | 134.000 |
| token_0/layer_20/swiglu \| token_1/layer_20/swiglu | ciphertext:L2:c2 | 320 | 960.000 |
| token_0/layer_20/swiglu \| token_1/layer_20/swiglu | ciphertext:L3:c2 | 156 | 624.000 |
| token_0/layer_20/swiglu \| token_1/layer_20/swiglu | ciphertext:L4:c2 | 12 | 60.000 |
| token_0/layer_20/swiglu \| token_1/layer_20/swiglu | ciphertext:L5:c2 | 3 | 18.000 |
| token_0/layer_21/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_0/layer_21/gqa_attention | ciphertext:L1:c2 | 17 | 34.000 |
| token_0/layer_21/gqa_attention | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L1:c2 | 53 | 106.000 |
| token_0/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L2:c2 | 107 | 321.000 |
| token_0/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L3:c2 | 13 | 52.000 |
| token_0/layer_21/model_list.21.attention.k_weight.weight \| token_1/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_21/model_list.21.attention.k_weight.weight \| token_1/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L1:c2 | 193 | 386.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L2:c2 | 136 | 408.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L2:c3 | 6 | 27.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 620 | 2,480.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c3 | 5 | 30.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c2 | 34 | 170.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L5:c2 | 15 | 90.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_21/model_list.21.attention.o_weight.weight | plaintext:L4:c1 | 45 | 112.500 |
| token_0/layer_21/model_list.21.attention.o_weight.weight \| token_1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_0/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_21/model_list.21.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_21/model_list.21.attention.q_weight.weight \| token_1/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L3:c2 | 27 | 108.000 |
| token_0/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L1:c2 | 53 | 106.000 |
| token_0/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L2:c2 | 108 | 324.000 |
| token_0/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_21/model_list.21.attention.v_weight.weight \| token_1/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L2:c2 | 3 | 9.000 |
| token_0/layer_21/model_list.21.attention.v_weight.weight \| token_1/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_0/layer_21/model_list.21.post_Normal.weight | ciphertext:L2:c2 | 47 | 141.000 |
| token_0/layer_21/model_list.21.post_Normal.weight | ciphertext:L3:c2 | 65 | 260.000 |
| token_0/layer_21/model_list.21.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_21/model_list.21.post_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_0/layer_21/model_list.21.post_Normal.weight \| token_1/layer_21/model_list.21.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_21/model_list.21.pre_Normal.weight | ciphertext:L2:c2 | 99 | 297.000 |
| token_0/layer_21/model_list.21.pre_Normal.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_0/layer_21/model_list.21.pre_Normal.weight \| token_1/layer_21/model_list.21.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_21/model_list.21.pre_Normal.weight \| token_1/layer_21/model_list.21.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_21/rope | ciphertext:L1:c2 | 10 | 20.000 |
| token_0/layer_21/rope | ciphertext:L2:c2 | 2 | 6.000 |
| token_0/layer_21/swiglu | ciphertext:L1:c2 | 1124 | 2,248.000 |
| token_0/layer_21/swiglu | ciphertext:L2:c2 | 5385 | 16,155.000 |
| token_0/layer_21/swiglu | ciphertext:L2:c3 | 9 | 40.500 |
| token_0/layer_21/swiglu | ciphertext:L3:c2 | 9460 | 37,840.000 |
| token_0/layer_21/swiglu | ciphertext:L3:c3 | 10 | 60.000 |
| token_0/layer_21/swiglu | ciphertext:L4:c2 | 972 | 4,860.000 |
| token_0/layer_21/swiglu | ciphertext:L4:c3 | 6 | 45.000 |
| token_0/layer_21/swiglu | ciphertext:L5:c2 | 65 | 390.000 |
| token_0/layer_21/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_0/layer_21/swiglu | plaintext:L1:c1 | 3 | 3.000 |
| token_0/layer_21/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_21/swiglu | plaintext:L4:c1 | 448 | 1,120.000 |
| token_0/layer_21/swiglu | plaintext:L5:c1 | 25 | 75.000 |
| token_0/layer_21/swiglu \| token_1/layer_21/swiglu | ciphertext:L1:c2 | 40 | 80.000 |
| token_0/layer_21/swiglu \| token_1/layer_21/swiglu | ciphertext:L2:c2 | 202 | 606.000 |
| token_0/layer_21/swiglu \| token_1/layer_21/swiglu | ciphertext:L3:c2 | 169 | 676.000 |
| token_0/layer_21/swiglu \| token_1/layer_21/swiglu | ciphertext:L4:c2 | 9 | 45.000 |
| token_0/layer_21/swiglu \| token_1/layer_21/swiglu | ciphertext:L5:c2 | 2 | 12.000 |
| token_0/layer_22/_decoder_block | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_22/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_22/gqa_attention | ciphertext:L3:c2 | 19 | 76.000 |
| token_0/layer_22/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L3:c2 | 154 | 616.000 |
| token_0/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_22/model_list.22.attention.k_weight.weight \| token_1/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L2:c2 | 40 | 120.000 |
| token_0/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 956 | 3,824.000 |
| token_0/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L4:c2 | 41 | 205.000 |
| token_0/layer_22/model_list.22.attention.o_weight.weight \| token_1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_22/model_list.22.attention.o_weight.weight \| token_1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 26 | 104.000 |
| token_0/layer_22/model_list.22.attention.q_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_22/model_list.22.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_22/model_list.22.attention.q_weight.weight \| token_1/layer_22/model_list.22.attention.q_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_22/model_list.22.attention.q_weight.weight \| token_1/layer_22/model_list.22.attention.q_weight.weight | ciphertext:L3:c2 | 27 | 108.000 |
| token_0/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L3:c2 | 153 | 612.000 |
| token_0/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_22/model_list.22.attention.v_weight.weight \| token_1/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_22/model_list.22.post_Normal.weight | ciphertext:L2:c2 | 36 | 108.000 |
| token_0/layer_22/model_list.22.post_Normal.weight | ciphertext:L3:c2 | 71 | 284.000 |
| token_0/layer_22/model_list.22.post_Normal.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_22/model_list.22.post_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_0/layer_22/model_list.22.post_Normal.weight \| token_1/layer_22/model_list.22.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_22/model_list.22.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_22/model_list.22.pre_Normal.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_0/layer_22/model_list.22.pre_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_0/layer_22/model_list.22.pre_Normal.weight \| token_1/layer_22/model_list.22.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_22/rope | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_22/swiglu | ciphertext:L1:c2 | 1061 | 2,122.000 |
| token_0/layer_22/swiglu | ciphertext:L2:c2 | 4942 | 14,826.000 |
| token_0/layer_22/swiglu | ciphertext:L2:c3 | 15 | 67.500 |
| token_0/layer_22/swiglu | ciphertext:L3:c2 | 10688 | 42,752.000 |
| token_0/layer_22/swiglu | ciphertext:L3:c3 | 15 | 90.000 |
| token_0/layer_22/swiglu | ciphertext:L4:c2 | 543 | 2,715.000 |
| token_0/layer_22/swiglu | ciphertext:L4:c3 | 9 | 67.500 |
| token_0/layer_22/swiglu | ciphertext:L5:c2 | 48 | 288.000 |
| token_0/layer_22/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_0/layer_22/swiglu | plaintext:L1:c1 | 3 | 3.000 |
| token_0/layer_22/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_22/swiglu | plaintext:L3:c1 | 1 | 2.000 |
| token_0/layer_22/swiglu | plaintext:L4:c1 | 49 | 122.500 |
| token_0/layer_22/swiglu \| token_1/layer_22/swiglu | ciphertext:L1:c2 | 74 | 148.000 |
| token_0/layer_22/swiglu \| token_1/layer_22/swiglu | ciphertext:L2:c2 | 463 | 1,389.000 |
| token_0/layer_22/swiglu \| token_1/layer_22/swiglu | ciphertext:L3:c2 | 164 | 656.000 |
| token_0/layer_22/swiglu \| token_1/layer_22/swiglu | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_22/swiglu \| token_1/layer_22/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_22/swiglu \| token_1/layer_22/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_0/layer_22/swiglu \| token_1/layer_22/swiglu | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L2:c2 | 5 | 15.000 |
| token_0/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L3:c2 | 155 | 620.000 |
| token_0/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_0/layer_23/model_list.23.attention.k_weight.weight \| token_1/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L3:c2 | 2 | 8.000 |
| token_0/layer_23/model_list.23.attention.k_weight.weight \| token_1/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_23/model_list.23.attention.k_weight.weight \| token_1/layer_23/model_list.23.attention.k_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_23/model_list.23.attention.o_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_23/model_list.23.attention.o_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_0/layer_23/model_list.23.attention.o_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_23/model_list.23.attention.o_weight.weight \| token_1/layer_23/model_list.23.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_23/model_list.23.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L2:c2 | 24 | 72.000 |
| token_0/layer_23/model_list.23.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L3:c2 | 25 | 100.000 |
| token_0/layer_23/model_list.23.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_0/layer_23/model_list.23.attention.q_weight.weight \| token_1/layer_23/model_list.23.attention.q_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_0/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_0/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L3:c2 | 154 | 616.000 |
| token_0/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_0/layer_23/model_list.23.attention.v_weight.weight \| token_1/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L3:c2 | 4 | 16.000 |
| token_0/layer_23/model_list.23.post_Normal.weight \| token_1/layer_23/model_list.23.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_23/model_list.23.post_Normal.weight \| token_1/layer_23/model_list.23.post_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_23/model_list.23.pre_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_0/layer_23/model_list.23.pre_Normal.weight | ciphertext:L3:c2 | 112 | 448.000 |
| token_0/layer_23/model_list.23.pre_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_0/layer_23/model_list.23.pre_Normal.weight \| token_1/layer_23/model_list.23.pre_Normal.weight | ciphertext:L3:c2 | 1 | 4.000 |
| token_0/layer_23/rope | ciphertext:L3:c2 | 12 | 48.000 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | ciphertext:L1:c2 | 66 | 132.000 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | ciphertext:L2:c2 | 629 | 1,887.000 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | ciphertext:L2:c3 | 1 | 4.500 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | ciphertext:L3:c2 | 118 | 472.000 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | ciphertext:L4:c2 | 2 | 10.000 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | ciphertext:L5:c2 | 1 | 6.000 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | plaintext:L3:c1 | 24 | 48.000 |
| token_0/layer_23/swiglu \| token_1/layer_23/swiglu | plaintext:L4:c1 | 3 | 7.500 |
| token_1/layer_00/_decoder_block | ciphertext:L1:c2 | 2 | 4.000 |
| token_1/layer_00/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/_decoder_block | ciphertext:L4:c2 | 2 | 10.000 |
| token_1/layer_00/gqa_attention | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_00/gqa_attention | ciphertext:L3:c2 | 148 | 592.000 |
| token_1/layer_00/gqa_attention | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_00/gqa_attention | plaintext:L2:c1 | 2 | 3.000 |
| token_1/layer_00/gqa_attention | plaintext:L5:c1 | 2 | 6.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_01/gqa_attention | ciphertext:L1:c2 | 5 | 10.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_01/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_01/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_02/gqa_attention | ciphertext:L2:c2 | 1 | 3.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_02/gqa_attention | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_03/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_03/gqa_attention | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_04/gqa_attention | ciphertext:L2:c2 | 7 | 21.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_04/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_05/gqa_attention | ciphertext:L2:c2 | 6 | 18.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_05/gqa_attention | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_06/gqa_attention | ciphertext:L2:c2 | 1 | 3.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_06/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_07/gqa_attention | ciphertext:L1:c2 | 3 | 6.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_07/gqa_attention | ciphertext:L2:c2 | 2 | 6.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_07/gqa_attention | ciphertext:L3:c2 | 2 | 8.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_08/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_09/gqa_attention | ciphertext:L1:c2 | 1 | 2.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_09/gqa_attention | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_09/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_10/gqa_attention | ciphertext:L2:c2 | 7 | 21.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_10/gqa_attention | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_11/gqa_attention | ciphertext:L2:c2 | 7 | 21.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_12/gqa_attention | ciphertext:L1:c2 | 2 | 4.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_12/gqa_attention | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_13/gqa_attention | ciphertext:L2:c2 | 1 | 3.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_13/gqa_attention | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_14/gqa_attention | ciphertext:L3:c2 | 8 | 32.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_15/gqa_attention | ciphertext:L2:c2 | 6 | 18.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_15/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_16/gqa_attention | ciphertext:L1:c2 | 6 | 12.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_16/gqa_attention | ciphertext:L3:c2 | 1 | 4.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_17/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_18/gqa_attention | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_18/gqa_attention | ciphertext:L4:c2 | 2 | 10.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_19/gqa_attention | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_19/gqa_attention | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_20/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_21/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_22/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/gqa_attention \| token_1/layer_23/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_1/layer_00/model_list.0.attention.k_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L1:c2 | 49 | 98.000 |
| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L2:c2 | 349 | 1,047.000 |
| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L3:c2 | 602 | 2,408.000 |
| token_1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c2 | 22 | 110.000 |
| token_1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L4:c1 | 15 | 37.500 |
| token_1/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L3:c2 | 939 | 3,756.000 |
| token_1/layer_00/model_list.0.attention.q_weight.weight | ciphertext:L4:c2 | 50 | 250.000 |
| token_1/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_1/layer_00/model_list.0.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_00/model_list.0.post_Normal.weight | ciphertext:L4:c2 | 107 | 535.000 |
| token_1/layer_00/model_list.0.post_Normal.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_1/layer_00/model_list.0.post_Normal.weight | plaintext:L5:c1 | 5 | 15.000 |
| token_1/layer_00/model_list.0.pre_Normal.weight | ciphertext:L4:c2 | 96 | 480.000 |
| token_1/layer_00/model_list.0.pre_Normal.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_1/layer_00/model_list.0.pre_Normal.weight | plaintext:L5:c1 | 4 | 12.000 |
| token_1/layer_00/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_00/rope | ciphertext:L3:c2 | 23 | 92.000 |
| token_1/layer_00/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_00/rope \| token_1/layer_01/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_00/rope \| token_1/layer_01/rope | ciphertext:L3:c2 | 2 | 8.000 |
| token_1/layer_00/rope \| token_1/layer_02/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_00/rope \| token_1/layer_03/rope | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_00/rope \| token_1/layer_04/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_00/rope \| token_1/layer_05/rope | ciphertext:L1:c2 | 2 | 4.000 |
| token_1/layer_00/rope \| token_1/layer_05/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_06/rope | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_00/rope \| token_1/layer_07/rope | ciphertext:L1:c2 | 4 | 8.000 |
| token_1/layer_00/rope \| token_1/layer_08/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_09/rope | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_00/rope \| token_1/layer_10/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_00/rope \| token_1/layer_11/rope | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_00/rope \| token_1/layer_11/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_1/layer_00/rope \| token_1/layer_12/rope | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_00/rope \| token_1/layer_13/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_14/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_00/rope \| token_1/layer_14/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_00/rope \| token_1/layer_15/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_16/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_17/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_18/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_19/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_20/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_00/rope \| token_1/layer_20/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_00/rope \| token_1/layer_21/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_22/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/rope \| token_1/layer_23/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_00/swiglu | ciphertext:L1:c2 | 783 | 1,566.000 |
| token_1/layer_00/swiglu | ciphertext:L2:c2 | 3027 | 9,081.000 |
| token_1/layer_00/swiglu | ciphertext:L2:c3 | 7 | 31.500 |
| token_1/layer_00/swiglu | ciphertext:L3:c2 | 7440 | 29,760.000 |
| token_1/layer_00/swiglu | ciphertext:L3:c3 | 4 | 24.000 |
| token_1/layer_00/swiglu | ciphertext:L4:c2 | 4667 | 23,335.000 |
| token_1/layer_00/swiglu | ciphertext:L4:c3 | 5 | 37.500 |
| token_1/layer_00/swiglu | ciphertext:L5:c2 | 199 | 1,194.000 |
| token_1/layer_00/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_1/layer_00/swiglu | plaintext:L3:c1 | 28 | 56.000 |
| token_1/layer_00/swiglu | plaintext:L4:c1 | 374 | 935.000 |
| token_1/layer_00/swiglu | plaintext:L5:c1 | 152 | 456.000 |
| token_1/layer_01/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_01/gqa_attention | ciphertext:L1:c2 | 38 | 76.000 |
| token_1/layer_01/gqa_attention | ciphertext:L2:c2 | 94 | 282.000 |
| token_1/layer_01/gqa_attention | ciphertext:L3:c2 | 14 | 56.000 |
| token_1/layer_01/gqa_attention | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L2:c2 | 138 | 414.000 |
| token_1/layer_01/model_list.1.attention.k_weight.weight | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L1:c2 | 708 | 1,416.000 |
| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L2:c2 | 227 | 681.000 |
| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 94 | 376.000 |
| token_1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_01/model_list.1.attention.o_weight.weight | plaintext:L2:c1 | 4 | 6.000 |
| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L2:c2 | 435 | 1,305.000 |
| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L3:c2 | 479 | 1,916.000 |
| token_1/layer_01/model_list.1.attention.q_weight.weight | ciphertext:L4:c2 | 60 | 300.000 |
| token_1/layer_01/model_list.1.attention.q_weight.weight | plaintext:L4:c1 | 28 | 70.000 |
| token_1/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L2:c2 | 136 | 408.000 |
| token_1/layer_01/model_list.1.attention.v_weight.weight | ciphertext:L3:c2 | 20 | 80.000 |
| token_1/layer_01/model_list.1.post_Normal.weight | ciphertext:L2:c2 | 99 | 297.000 |
| token_1/layer_01/model_list.1.post_Normal.weight | ciphertext:L3:c2 | 13 | 52.000 |
| token_1/layer_01/model_list.1.pre_Normal.weight | ciphertext:L3:c2 | 108 | 432.000 |
| token_1/layer_01/model_list.1.pre_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_01/model_list.1.pre_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_1/layer_01/rope | ciphertext:L2:c2 | 16 | 48.000 |
| token_1/layer_01/rope | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_01/swiglu | ciphertext:L1:c2 | 717 | 1,434.000 |
| token_1/layer_01/swiglu | ciphertext:L2:c2 | 3910 | 11,730.000 |
| token_1/layer_01/swiglu | ciphertext:L2:c3 | 12 | 54.000 |
| token_1/layer_01/swiglu | ciphertext:L3:c2 | 10959 | 43,836.000 |
| token_1/layer_01/swiglu | ciphertext:L3:c3 | 9 | 54.000 |
| token_1/layer_01/swiglu | ciphertext:L4:c2 | 453 | 2,265.000 |
| token_1/layer_01/swiglu | ciphertext:L4:c3 | 3 | 22.500 |
| token_1/layer_01/swiglu | ciphertext:L5:c2 | 36 | 216.000 |
| token_1/layer_01/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_1/layer_01/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_1/layer_01/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_1/layer_01/swiglu | plaintext:L4:c1 | 814 | 2,035.000 |
| token_1/layer_02/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_02/gqa_attention | ciphertext:L2:c2 | 64 | 192.000 |
| token_1/layer_02/gqa_attention | ciphertext:L3:c2 | 9 | 36.000 |
| token_1/layer_02/gqa_attention | ciphertext:L4:c2 | 67 | 335.000 |
| token_1/layer_02/gqa_attention | ciphertext:L5:c2 | 2 | 12.000 |
| token_1/layer_02/gqa_attention | plaintext:L5:c1 | 8 | 24.000 |
| token_1/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L2:c2 | 144 | 432.000 |
| token_1/layer_02/model_list.2.attention.k_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c2 | 846 | 3,384.000 |
| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L4:c2 | 99 | 495.000 |
| token_1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_1/layer_02/model_list.2.attention.o_weight.weight | plaintext:L4:c1 | 74 | 185.000 |
| token_1/layer_02/model_list.2.attention.o_weight.weight | plaintext:L5:c1 | 6 | 18.000 |
| token_1/layer_02/model_list.2.attention.q_weight.weight | ciphertext:L2:c2 | 916 | 2,748.000 |
| token_1/layer_02/model_list.2.attention.q_weight.weight | ciphertext:L3:c2 | 73 | 292.000 |
| token_1/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L2:c2 | 145 | 435.000 |
| token_1/layer_02/model_list.2.attention.v_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_1/layer_02/model_list.2.post_Normal.weight | ciphertext:L3:c2 | 96 | 384.000 |
| token_1/layer_02/model_list.2.post_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_02/model_list.2.post_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_1/layer_02/model_list.2.pre_Normal.weight | ciphertext:L2:c2 | 35 | 105.000 |
| token_1/layer_02/model_list.2.pre_Normal.weight | ciphertext:L3:c2 | 65 | 260.000 |
| token_1/layer_02/model_list.2.pre_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_02/model_list.2.pre_Normal.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_1/layer_02/rope | ciphertext:L2:c2 | 20 | 60.000 |
| token_1/layer_02/swiglu | ciphertext:L1:c2 | 489 | 978.000 |
| token_1/layer_02/swiglu | ciphertext:L2:c2 | 3161 | 9,483.000 |
| token_1/layer_02/swiglu | ciphertext:L2:c3 | 10 | 45.000 |
| token_1/layer_02/swiglu | ciphertext:L3:c2 | 12014 | 48,056.000 |
| token_1/layer_02/swiglu | ciphertext:L3:c3 | 10 | 60.000 |
| token_1/layer_02/swiglu | ciphertext:L4:c2 | 559 | 2,795.000 |
| token_1/layer_02/swiglu | ciphertext:L4:c3 | 5 | 37.500 |
| token_1/layer_02/swiglu | ciphertext:L5:c2 | 18 | 108.000 |
| token_1/layer_02/swiglu | ciphertext:L5:c3 | 3 | 27.000 |
| token_1/layer_02/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_02/swiglu | plaintext:L4:c1 | 378 | 945.000 |
| token_1/layer_03/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_03/_decoder_block | ciphertext:L3:c2 | 2 | 8.000 |
| token_1/layer_03/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_03/gqa_attention | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_03/gqa_attention | ciphertext:L4:c2 | 138 | 690.000 |
| token_1/layer_03/gqa_attention | ciphertext:L5:c2 | 5 | 30.000 |
| token_1/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L4:c2 | 150 | 750.000 |
| token_1/layer_03/model_list.3.attention.k_weight.weight | ciphertext:L5:c2 | 6 | 36.000 |
| token_1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 946 | 3,784.000 |
| token_1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c2 | 42 | 210.000 |
| token_1/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L4:c2 | 949 | 4,745.000 |
| token_1/layer_03/model_list.3.attention.q_weight.weight | ciphertext:L5:c2 | 40 | 240.000 |
| token_1/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L4:c2 | 149 | 745.000 |
| token_1/layer_03/model_list.3.attention.v_weight.weight | ciphertext:L5:c2 | 7 | 42.000 |
| token_1/layer_03/model_list.3.post_Normal.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_1/layer_03/model_list.3.post_Normal.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_1/layer_03/model_list.3.post_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_03/model_list.3.pre_Normal.weight | ciphertext:L1:c2 | 10 | 20.000 |
| token_1/layer_03/model_list.3.pre_Normal.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_03/model_list.3.pre_Normal.weight | ciphertext:L3:c2 | 79 | 316.000 |
| token_1/layer_03/model_list.3.pre_Normal.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_1/layer_03/model_list.3.pre_Normal.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_1/layer_03/rope | ciphertext:L4:c2 | 19 | 95.000 |
| token_1/layer_03/rope | ciphertext:L5:c2 | 1 | 6.000 |
| token_1/layer_03/swiglu | ciphertext:L1:c2 | 2454 | 4,908.000 |
| token_1/layer_03/swiglu | ciphertext:L2:c2 | 6512 | 19,536.000 |
| token_1/layer_03/swiglu | ciphertext:L2:c3 | 27 | 121.500 |
| token_1/layer_03/swiglu | ciphertext:L3:c2 | 6865 | 27,460.000 |
| token_1/layer_03/swiglu | ciphertext:L3:c3 | 18 | 108.000 |
| token_1/layer_03/swiglu | ciphertext:L4:c2 | 326 | 1,630.000 |
| token_1/layer_03/swiglu | ciphertext:L4:c3 | 8 | 60.000 |
| token_1/layer_03/swiglu | ciphertext:L5:c2 | 74 | 444.000 |
| token_1/layer_03/swiglu | ciphertext:L5:c3 | 3 | 27.000 |
| token_1/layer_03/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_1/layer_03/swiglu | plaintext:L2:c1 | 23 | 34.500 |
| token_1/layer_03/swiglu | plaintext:L3:c1 | 1 | 2.000 |
| token_1/layer_03/swiglu | plaintext:L4:c1 | 222 | 555.000 |
| token_1/layer_04/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_04/_decoder_block | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_04/gqa_attention | ciphertext:L2:c2 | 136 | 408.000 |
| token_1/layer_04/gqa_attention | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_04/gqa_attention | ciphertext:L4:c2 | 2 | 10.000 |
| token_1/layer_04/gqa_attention | plaintext:L3:c1 | 4 | 8.000 |
| token_1/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L2:c2 | 147 | 441.000 |
| token_1/layer_04/model_list.4.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_04/model_list.4.attention.k_weight.weight | plaintext:L3:c1 | 3 | 6.000 |
| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L2:c2 | 642 | 1,926.000 |
| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 369 | 1,476.000 |
| token_1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L4:c2 | 25 | 125.000 |
| token_1/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L2:c2 | 924 | 2,772.000 |
| token_1/layer_04/model_list.4.attention.q_weight.weight | ciphertext:L3:c2 | 32 | 128.000 |
| token_1/layer_04/model_list.4.attention.q_weight.weight | plaintext:L3:c1 | 33 | 66.000 |
| token_1/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L2:c2 | 146 | 438.000 |
| token_1/layer_04/model_list.4.attention.v_weight.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_04/model_list.4.attention.v_weight.weight | plaintext:L3:c1 | 5 | 10.000 |
| token_1/layer_04/model_list.4.post_Normal.weight | ciphertext:L2:c2 | 103 | 309.000 |
| token_1/layer_04/model_list.4.post_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_04/model_list.4.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_04/model_list.4.pre_Normal.weight | ciphertext:L2:c2 | 97 | 291.000 |
| token_1/layer_04/model_list.4.pre_Normal.weight | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_04/model_list.4.pre_Normal.weight | plaintext:L3:c1 | 3 | 6.000 |
| token_1/layer_04/rope | ciphertext:L2:c2 | 19 | 57.000 |
| token_1/layer_04/rope | plaintext:L3:c1 | 1 | 2.000 |
| token_1/layer_04/swiglu | ciphertext:L1:c2 | 926 | 1,852.000 |
| token_1/layer_04/swiglu | ciphertext:L2:c2 | 4693 | 14,079.000 |
| token_1/layer_04/swiglu | ciphertext:L2:c3 | 14 | 63.000 |
| token_1/layer_04/swiglu | ciphertext:L3:c2 | 8483 | 33,932.000 |
| token_1/layer_04/swiglu | ciphertext:L3:c3 | 12 | 72.000 |
| token_1/layer_04/swiglu | ciphertext:L4:c2 | 1941 | 9,705.000 |
| token_1/layer_04/swiglu | ciphertext:L4:c3 | 7 | 52.500 |
| token_1/layer_04/swiglu | ciphertext:L5:c2 | 102 | 612.000 |
| token_1/layer_04/swiglu | ciphertext:L5:c3 | 3 | 27.000 |
| token_1/layer_04/swiglu | plaintext:L2:c1 | 10 | 15.000 |
| token_1/layer_04/swiglu | plaintext:L3:c1 | 1 | 2.000 |
| token_1/layer_04/swiglu | plaintext:L4:c1 | 551 | 1,377.500 |
| token_1/layer_04/swiglu | plaintext:L5:c1 | 54 | 162.000 |
| token_1/layer_05/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_05/gqa_attention | ciphertext:L2:c2 | 91 | 273.000 |
| token_1/layer_05/gqa_attention | ciphertext:L3:c2 | 52 | 208.000 |
| token_1/layer_05/gqa_attention | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L3:c2 | 144 | 576.000 |
| token_1/layer_05/model_list.5.attention.k_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_05/model_list.5.attention.k_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L3:c2 | 971 | 3,884.000 |
| token_1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L4:c2 | 33 | 165.000 |
| token_1/layer_05/model_list.5.attention.o_weight.weight | plaintext:L4:c1 | 33 | 82.500 |
| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L3:c2 | 923 | 3,692.000 |
| token_1/layer_05/model_list.5.attention.q_weight.weight | ciphertext:L4:c2 | 32 | 160.000 |
| token_1/layer_05/model_list.5.attention.q_weight.weight | plaintext:L4:c1 | 64 | 160.000 |
| token_1/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L3:c2 | 145 | 580.000 |
| token_1/layer_05/model_list.5.attention.v_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_05/model_list.5.attention.v_weight.weight | plaintext:L4:c1 | 10 | 25.000 |
| token_1/layer_05/model_list.5.post_Normal.weight | ciphertext:L3:c2 | 107 | 428.000 |
| token_1/layer_05/model_list.5.post_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_05/model_list.5.post_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_05/model_list.5.pre_Normal.weight | ciphertext:L3:c2 | 108 | 432.000 |
| token_1/layer_05/model_list.5.pre_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_05/model_list.5.pre_Normal.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_1/layer_05/rope | ciphertext:L3:c2 | 23 | 92.000 |
| token_1/layer_05/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_05/swiglu | ciphertext:L1:c2 | 1446 | 2,892.000 |
| token_1/layer_05/swiglu | ciphertext:L2:c2 | 4136 | 12,408.000 |
| token_1/layer_05/swiglu | ciphertext:L2:c3 | 11 | 49.500 |
| token_1/layer_05/swiglu | ciphertext:L3:c2 | 10090 | 40,360.000 |
| token_1/layer_05/swiglu | ciphertext:L3:c3 | 10 | 60.000 |
| token_1/layer_05/swiglu | ciphertext:L4:c2 | 405 | 2,025.000 |
| token_1/layer_05/swiglu | ciphertext:L4:c3 | 1 | 7.500 |
| token_1/layer_05/swiglu | ciphertext:L5:c2 | 31 | 186.000 |
| token_1/layer_05/swiglu | plaintext:L1:c1 | 3 | 3.000 |
| token_1/layer_05/swiglu | plaintext:L2:c1 | 4 | 6.000 |
| token_1/layer_05/swiglu | plaintext:L3:c1 | 4 | 8.000 |
| token_1/layer_05/swiglu | plaintext:L4:c1 | 677 | 1,692.500 |
| token_1/layer_06/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_06/gqa_attention | ciphertext:L1:c2 | 11 | 22.000 |
| token_1/layer_06/gqa_attention | ciphertext:L2:c2 | 11 | 33.000 |
| token_1/layer_06/gqa_attention | ciphertext:L3:c2 | 115 | 460.000 |
| token_1/layer_06/gqa_attention | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_06/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L3:c2 | 151 | 604.000 |
| token_1/layer_06/model_list.6.attention.k_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 909 | 3,636.000 |
| token_1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L4:c2 | 39 | 195.000 |
| token_1/layer_06/model_list.6.attention.o_weight.weight | plaintext:L4:c1 | 80 | 200.000 |
| token_1/layer_06/model_list.6.attention.q_weight.weight | ciphertext:L3:c2 | 947 | 3,788.000 |
| token_1/layer_06/model_list.6.attention.q_weight.weight | ciphertext:L4:c2 | 42 | 210.000 |
| token_1/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_1/layer_06/model_list.6.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_06/model_list.6.post_Normal.weight | ciphertext:L3:c2 | 95 | 380.000 |
| token_1/layer_06/model_list.6.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_06/model_list.6.post_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_1/layer_06/model_list.6.pre_Normal.weight | ciphertext:L3:c2 | 107 | 428.000 |
| token_1/layer_06/model_list.6.pre_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_06/model_list.6.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_06/rope | ciphertext:L3:c2 | 21 | 84.000 |
| token_1/layer_06/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_06/swiglu | ciphertext:L1:c2 | 1175 | 2,350.000 |
| token_1/layer_06/swiglu | ciphertext:L2:c2 | 4656 | 13,968.000 |
| token_1/layer_06/swiglu | ciphertext:L2:c3 | 13 | 58.500 |
| token_1/layer_06/swiglu | ciphertext:L3:c2 | 9866 | 39,464.000 |
| token_1/layer_06/swiglu | ciphertext:L3:c3 | 12 | 72.000 |
| token_1/layer_06/swiglu | ciphertext:L4:c2 | 450 | 2,250.000 |
| token_1/layer_06/swiglu | ciphertext:L4:c3 | 7 | 52.500 |
| token_1/layer_06/swiglu | ciphertext:L5:c2 | 53 | 318.000 |
| token_1/layer_06/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_06/swiglu | plaintext:L1:c1 | 4 | 4.000 |
| token_1/layer_06/swiglu | plaintext:L2:c1 | 3 | 4.500 |
| token_1/layer_06/swiglu | plaintext:L4:c1 | 565 | 1,412.500 |
| token_1/layer_07/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_07/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_07/gqa_attention | ciphertext:L1:c2 | 85 | 170.000 |
| token_1/layer_07/gqa_attention | ciphertext:L2:c2 | 34 | 102.000 |
| token_1/layer_07/gqa_attention | ciphertext:L3:c2 | 31 | 124.000 |
| token_1/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L1:c2 | 46 | 92.000 |
| token_1/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L2:c2 | 104 | 312.000 |
| token_1/layer_07/model_list.7.attention.k_weight.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L3:c2 | 946 | 3,784.000 |
| token_1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L4:c2 | 42 | 210.000 |
| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L1:c2 | 146 | 292.000 |
| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L2:c2 | 777 | 2,331.000 |
| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L2:c3 | 1 | 4.500 |
| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L3:c2 | 63 | 252.000 |
| token_1/layer_07/model_list.7.attention.q_weight.weight | ciphertext:L3:c3 | 2 | 12.000 |
| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L1:c2 | 49 | 98.000 |
| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L2:c2 | 100 | 300.000 |
| token_1/layer_07/model_list.7.attention.v_weight.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_07/model_list.7.post_Normal.weight | ciphertext:L2:c2 | 14 | 42.000 |
| token_1/layer_07/model_list.7.post_Normal.weight | ciphertext:L3:c2 | 95 | 380.000 |
| token_1/layer_07/model_list.7.post_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_07/model_list.7.post_Normal.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_1/layer_07/model_list.7.pre_Normal.weight | ciphertext:L2:c2 | 60 | 180.000 |
| token_1/layer_07/model_list.7.pre_Normal.weight | ciphertext:L3:c2 | 41 | 164.000 |
| token_1/layer_07/model_list.7.pre_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_1/layer_07/model_list.7.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_07/rope | ciphertext:L1:c2 | 15 | 30.000 |
| token_1/layer_07/rope | ciphertext:L2:c2 | 5 | 15.000 |
| token_1/layer_07/swiglu | ciphertext:L1:c2 | 871 | 1,742.000 |
| token_1/layer_07/swiglu | ciphertext:L2:c2 | 5200 | 15,600.000 |
| token_1/layer_07/swiglu | ciphertext:L2:c3 | 14 | 63.000 |
| token_1/layer_07/swiglu | ciphertext:L3:c2 | 9811 | 39,244.000 |
| token_1/layer_07/swiglu | ciphertext:L3:c3 | 8 | 48.000 |
| token_1/layer_07/swiglu | ciphertext:L4:c2 | 378 | 1,890.000 |
| token_1/layer_07/swiglu | ciphertext:L4:c3 | 8 | 60.000 |
| token_1/layer_07/swiglu | ciphertext:L5:c2 | 25 | 150.000 |
| token_1/layer_07/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_1/layer_07/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_1/layer_07/swiglu | plaintext:L4:c1 | 331 | 827.500 |
| token_1/layer_08/_decoder_block | ciphertext:L1:c2 | 3 | 6.000 |
| token_1/layer_08/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_08/gqa_attention | ciphertext:L3:c2 | 140 | 560.000 |
| token_1/layer_08/gqa_attention | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_08/gqa_attention | plaintext:L4:c1 | 6 | 15.000 |
| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L1:c2 | 6 | 12.000 |
| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L2:c2 | 35 | 105.000 |
| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c2 | 105 | 420.000 |
| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_1/layer_08/model_list.8.attention.k_weight.weight | ciphertext:L5:c3 | 1 | 9.000 |
| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L1:c2 | 7 | 14.000 |
| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L2:c2 | 11 | 33.000 |
| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 933 | 3,732.000 |
| token_1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c2 | 37 | 185.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L1:c2 | 355 | 710.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L2:c2 | 312 | 936.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L2:c3 | 2 | 9.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L3:c2 | 283 | 1,132.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L3:c3 | 3 | 18.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L4:c2 | 20 | 100.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | ciphertext:L5:c2 | 10 | 60.000 |
| token_1/layer_08/model_list.8.attention.q_weight.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L1:c2 | 10 | 20.000 |
| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L2:c2 | 10 | 30.000 |
| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L2:c3 | 4 | 18.000 |
| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L3:c2 | 120 | 480.000 |
| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_08/model_list.8.attention.v_weight.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_1/layer_08/model_list.8.attention.v_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_08/model_list.8.post_Normal.weight | ciphertext:L1:c2 | 14 | 28.000 |
| token_1/layer_08/model_list.8.post_Normal.weight | ciphertext:L2:c2 | 69 | 207.000 |
| token_1/layer_08/model_list.8.post_Normal.weight | ciphertext:L3:c2 | 23 | 92.000 |
| token_1/layer_08/model_list.8.pre_Normal.weight | ciphertext:L1:c2 | 17 | 34.000 |
| token_1/layer_08/model_list.8.pre_Normal.weight | ciphertext:L2:c2 | 89 | 267.000 |
| token_1/layer_08/model_list.8.pre_Normal.weight | ciphertext:L3:c2 | 7 | 28.000 |
| token_1/layer_08/rope | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_08/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_08/rope | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_08/swiglu | ciphertext:L1:c2 | 1808 | 3,616.000 |
| token_1/layer_08/swiglu | ciphertext:L2:c2 | 4554 | 13,662.000 |
| token_1/layer_08/swiglu | ciphertext:L2:c3 | 13 | 58.500 |
| token_1/layer_08/swiglu | ciphertext:L3:c2 | 9330 | 37,320.000 |
| token_1/layer_08/swiglu | ciphertext:L3:c3 | 6 | 36.000 |
| token_1/layer_08/swiglu | ciphertext:L4:c2 | 470 | 2,350.000 |
| token_1/layer_08/swiglu | ciphertext:L4:c3 | 2 | 15.000 |
| token_1/layer_08/swiglu | ciphertext:L5:c2 | 71 | 426.000 |
| token_1/layer_08/swiglu | plaintext:L2:c1 | 35 | 52.500 |
| token_1/layer_08/swiglu | plaintext:L3:c1 | 25 | 50.000 |
| token_1/layer_08/swiglu | plaintext:L4:c1 | 309 | 772.500 |
| token_1/layer_09/_decoder_block | ciphertext:L1:c2 | 2 | 4.000 |
| token_1/layer_09/_decoder_block | ciphertext:L2:c2 | 2 | 6.000 |
| token_1/layer_09/gqa_attention | ciphertext:L1:c2 | 1 | 2.000 |
| token_1/layer_09/gqa_attention | ciphertext:L2:c2 | 8 | 24.000 |
| token_1/layer_09/gqa_attention | ciphertext:L3:c2 | 129 | 516.000 |
| token_1/layer_09/gqa_attention | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L2:c2 | 52 | 156.000 |
| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L3:c2 | 14 | 56.000 |
| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L4:c2 | 89 | 445.000 |
| token_1/layer_09/model_list.9.attention.k_weight.weight | ciphertext:L5:c2 | 5 | 30.000 |
| token_1/layer_09/model_list.9.attention.k_weight.weight | plaintext:L5:c1 | 6 | 18.000 |
| token_1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L1:c2 | 750 | 1,500.000 |
| token_1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L2:c2 | 236 | 708.000 |
| token_1/layer_09/model_list.9.attention.o_weight.weight | plaintext:L2:c1 | 2 | 3.000 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L1:c2 | 184 | 368.000 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L2:c2 | 299 | 897.000 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L2:c3 | 5 | 22.500 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L3:c2 | 287 | 1,148.000 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L3:c3 | 3 | 18.000 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L4:c2 | 218 | 1,090.000 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L4:c3 | 3 | 22.500 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | ciphertext:L5:c2 | 29 | 174.000 |
| token_1/layer_09/model_list.9.attention.q_weight.weight | plaintext:L5:c1 | 15 | 45.000 |
| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L2:c2 | 52 | 156.000 |
| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L3:c2 | 15 | 60.000 |
| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L4:c2 | 90 | 450.000 |
| token_1/layer_09/model_list.9.attention.v_weight.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_1/layer_09/model_list.9.attention.v_weight.weight | plaintext:L5:c1 | 6 | 18.000 |
| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L1:c2 | 26 | 52.000 |
| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L2:c2 | 14 | 42.000 |
| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L3:c2 | 44 | 176.000 |
| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_1/layer_09/model_list.9.post_Normal.weight | ciphertext:L5:c2 | 12 | 72.000 |
| token_1/layer_09/model_list.9.pre_Normal.weight | ciphertext:L1:c2 | 81 | 162.000 |
| token_1/layer_09/model_list.9.pre_Normal.weight | ciphertext:L2:c2 | 26 | 78.000 |
| token_1/layer_09/rope | ciphertext:L3:c2 | 22 | 88.000 |
| token_1/layer_09/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_09/swiglu | ciphertext:L1:c2 | 1409 | 2,818.000 |
| token_1/layer_09/swiglu | ciphertext:L2:c2 | 7023 | 21,069.000 |
| token_1/layer_09/swiglu | ciphertext:L2:c3 | 14 | 63.000 |
| token_1/layer_09/swiglu | ciphertext:L3:c2 | 3870 | 15,480.000 |
| token_1/layer_09/swiglu | ciphertext:L3:c3 | 8 | 48.000 |
| token_1/layer_09/swiglu | ciphertext:L4:c2 | 3573 | 17,865.000 |
| token_1/layer_09/swiglu | ciphertext:L4:c3 | 6 | 45.000 |
| token_1/layer_09/swiglu | ciphertext:L5:c2 | 432 | 2,592.000 |
| token_1/layer_09/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_09/swiglu | plaintext:L3:c1 | 143 | 286.000 |
| token_1/layer_09/swiglu | plaintext:L4:c1 | 145 | 362.500 |
| token_1/layer_09/swiglu | plaintext:L5:c1 | 126 | 378.000 |
| token_1/layer_10/_decoder_block | ciphertext:L1:c2 | 3 | 6.000 |
| token_1/layer_10/_decoder_block | ciphertext:L3:c2 | 2 | 8.000 |
| token_1/layer_10/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_10/gqa_attention | ciphertext:L2:c2 | 128 | 384.000 |
| token_1/layer_10/gqa_attention | ciphertext:L3:c2 | 12 | 48.000 |
| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L2:c2 | 14 | 42.000 |
| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L3:c2 | 143 | 572.000 |
| token_1/layer_10/model_list.10.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L1:c2 | 145 | 290.000 |
| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 787 | 2,361.000 |
| token_1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L3:c2 | 55 | 220.000 |
| token_1/layer_10/model_list.10.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L2:c2 | 92 | 276.000 |
| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L3:c2 | 905 | 3,620.000 |
| token_1/layer_10/model_list.10.attention.q_weight.weight | ciphertext:L4:c2 | 39 | 195.000 |
| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L2:c2 | 14 | 42.000 |
| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L3:c2 | 142 | 568.000 |
| token_1/layer_10/model_list.10.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_10/model_list.10.post_Normal.weight | ciphertext:L1:c2 | 2 | 4.000 |
| token_1/layer_10/model_list.10.post_Normal.weight | ciphertext:L4:c2 | 94 | 470.000 |
| token_1/layer_10/model_list.10.post_Normal.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_1/layer_10/model_list.10.post_Normal.weight | plaintext:L5:c1 | 4 | 12.000 |
| token_1/layer_10/model_list.10.pre_Normal.weight | ciphertext:L1:c2 | 1 | 2.000 |
| token_1/layer_10/model_list.10.pre_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_1/layer_10/model_list.10.pre_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_10/rope | ciphertext:L2:c2 | 16 | 48.000 |
| token_1/layer_10/rope | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_10/swiglu | ciphertext:L1:c2 | 1049 | 2,098.000 |
| token_1/layer_10/swiglu | ciphertext:L2:c2 | 3927 | 11,781.000 |
| token_1/layer_10/swiglu | ciphertext:L2:c3 | 23 | 103.500 |
| token_1/layer_10/swiglu | ciphertext:L3:c2 | 7403 | 29,612.000 |
| token_1/layer_10/swiglu | ciphertext:L3:c3 | 17 | 102.000 |
| token_1/layer_10/swiglu | ciphertext:L4:c2 | 3563 | 17,815.000 |
| token_1/layer_10/swiglu | ciphertext:L4:c3 | 10 | 75.000 |
| token_1/layer_10/swiglu | ciphertext:L5:c2 | 187 | 1,122.000 |
| token_1/layer_10/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_10/swiglu | plaintext:L1:c1 | 3 | 3.000 |
| token_1/layer_10/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_10/swiglu | plaintext:L4:c1 | 374 | 935.000 |
| token_1/layer_10/swiglu | plaintext:L5:c1 | 135 | 405.000 |
| token_1/layer_11/_decoder_block | ciphertext:L1:c2 | 3 | 6.000 |
| token_1/layer_11/_decoder_block | ciphertext:L2:c2 | 5 | 15.000 |
| token_1/layer_11/gqa_attention | ciphertext:L2:c2 | 136 | 408.000 |
| token_1/layer_11/gqa_attention | ciphertext:L3:c2 | 13 | 52.000 |
| token_1/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L2:c2 | 145 | 435.000 |
| token_1/layer_11/model_list.11.attention.k_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L1:c2 | 479 | 958.000 |
| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 484 | 1,452.000 |
| token_1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L3:c2 | 24 | 96.000 |
| token_1/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L2:c2 | 921 | 2,763.000 |
| token_1/layer_11/model_list.11.attention.q_weight.weight | ciphertext:L3:c2 | 68 | 272.000 |
| token_1/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L2:c2 | 144 | 432.000 |
| token_1/layer_11/model_list.11.attention.v_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L1:c2 | 18 | 36.000 |
| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L2:c2 | 48 | 144.000 |
| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L3:c2 | 48 | 192.000 |
| token_1/layer_11/model_list.11.post_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_11/model_list.11.pre_Normal.weight | ciphertext:L2:c2 | 63 | 189.000 |
| token_1/layer_11/model_list.11.pre_Normal.weight | ciphertext:L3:c2 | 39 | 156.000 |
| token_1/layer_11/model_list.11.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_11/model_list.11.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_11/rope | ciphertext:L2:c2 | 19 | 57.000 |
| token_1/layer_11/rope | ciphertext:L3:c2 | 1 | 4.000 |
| token_1/layer_11/swiglu | ciphertext:L1:c2 | 788 | 1,576.000 |
| token_1/layer_11/swiglu | ciphertext:L2:c2 | 1777 | 5,331.000 |
| token_1/layer_11/swiglu | ciphertext:L2:c3 | 8 | 36.000 |
| token_1/layer_11/swiglu | ciphertext:L3:c2 | 9344 | 37,376.000 |
| token_1/layer_11/swiglu | ciphertext:L3:c3 | 5 | 30.000 |
| token_1/layer_11/swiglu | ciphertext:L4:c2 | 4104 | 20,520.000 |
| token_1/layer_11/swiglu | ciphertext:L4:c3 | 4 | 30.000 |
| token_1/layer_11/swiglu | ciphertext:L5:c2 | 128 | 768.000 |
| token_1/layer_11/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_1/layer_11/swiglu | plaintext:L4:c1 | 636 | 1,590.000 |
| token_1/layer_12/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_12/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_12/gqa_attention | ciphertext:L1:c2 | 43 | 86.000 |
| token_1/layer_12/gqa_attention | ciphertext:L2:c2 | 53 | 159.000 |
| token_1/layer_12/gqa_attention | ciphertext:L3:c2 | 44 | 176.000 |
| token_1/layer_12/gqa_attention | ciphertext:L3:c3 | 1 | 6.000 |
| token_1/layer_12/gqa_attention | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_12/gqa_attention | plaintext:L4:c1 | 3 | 7.500 |
| token_1/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L2:c2 | 144 | 432.000 |
| token_1/layer_12/model_list.12.attention.k_weight.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L3:c2 | 920 | 3,680.000 |
| token_1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 41 | 205.000 |
| token_1/layer_12/model_list.12.attention.o_weight.weight | plaintext:L4:c1 | 27 | 67.500 |
| token_1/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L2:c2 | 916 | 2,748.000 |
| token_1/layer_12/model_list.12.attention.q_weight.weight | ciphertext:L3:c2 | 73 | 292.000 |
| token_1/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L2:c2 | 145 | 435.000 |
| token_1/layer_12/model_list.12.attention.v_weight.weight | ciphertext:L3:c2 | 11 | 44.000 |
| token_1/layer_12/model_list.12.post_Normal.weight | ciphertext:L2:c2 | 71 | 213.000 |
| token_1/layer_12/model_list.12.post_Normal.weight | ciphertext:L3:c2 | 40 | 160.000 |
| token_1/layer_12/model_list.12.post_Normal.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_1/layer_12/model_list.12.pre_Normal.weight | ciphertext:L2:c2 | 109 | 327.000 |
| token_1/layer_12/model_list.12.pre_Normal.weight | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_12/rope | ciphertext:L2:c2 | 18 | 54.000 |
| token_1/layer_12/rope | ciphertext:L3:c2 | 2 | 8.000 |
| token_1/layer_12/swiglu | ciphertext:L1:c2 | 1656 | 3,312.000 |
| token_1/layer_12/swiglu | ciphertext:L2:c2 | 4893 | 14,679.000 |
| token_1/layer_12/swiglu | ciphertext:L2:c3 | 14 | 63.000 |
| token_1/layer_12/swiglu | ciphertext:L3:c2 | 9258 | 37,032.000 |
| token_1/layer_12/swiglu | ciphertext:L3:c3 | 13 | 78.000 |
| token_1/layer_12/swiglu | ciphertext:L4:c2 | 401 | 2,005.000 |
| token_1/layer_12/swiglu | ciphertext:L4:c3 | 4 | 30.000 |
| token_1/layer_12/swiglu | ciphertext:L5:c2 | 56 | 336.000 |
| token_1/layer_12/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_1/layer_12/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_12/swiglu | plaintext:L3:c1 | 40 | 80.000 |
| token_1/layer_12/swiglu | plaintext:L4:c1 | 404 | 1,010.000 |
| token_1/layer_13/_decoder_block | ciphertext:L2:c2 | 2 | 6.000 |
| token_1/layer_13/_decoder_block | ciphertext:L3:c2 | 2 | 8.000 |
| token_1/layer_13/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_13/_decoder_block | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_13/gqa_attention | ciphertext:L2:c2 | 15 | 45.000 |
| token_1/layer_13/gqa_attention | ciphertext:L3:c2 | 125 | 500.000 |
| token_1/layer_13/gqa_attention | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_13/gqa_attention | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L1:c2 | 84 | 168.000 |
| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L3:c2 | 68 | 272.000 |
| token_1/layer_13/model_list.13.attention.k_weight.weight | ciphertext:L4:c2 | 2 | 10.000 |
| token_1/layer_13/model_list.13.attention.k_weight.weight | plaintext:L2:c1 | 4 | 6.000 |
| token_1/layer_13/model_list.13.attention.k_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c2 | 939 | 2,817.000 |
| token_1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L3:c2 | 47 | 188.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L1:c2 | 463 | 926.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L2:c2 | 39 | 117.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L3:c2 | 423 | 1,692.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L4:c2 | 77 | 385.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L4:c3 | 2 | 15.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | ciphertext:L5:c3 | 1 | 9.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | plaintext:L2:c1 | 20 | 30.000 |
| token_1/layer_13/model_list.13.attention.q_weight.weight | plaintext:L4:c1 | 7 | 17.500 |
| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L1:c2 | 22 | 44.000 |
| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L2:c2 | 1 | 3.000 |
| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L4:c2 | 11 | 55.000 |
| token_1/layer_13/model_list.13.attention.v_weight.weight | ciphertext:L5:c2 | 26 | 156.000 |
| token_1/layer_13/model_list.13.attention.v_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_13/model_list.13.post_Normal.weight | ciphertext:L2:c2 | 99 | 297.000 |
| token_1/layer_13/model_list.13.post_Normal.weight | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_13/model_list.13.pre_Normal.weight | ciphertext:L2:c2 | 92 | 276.000 |
| token_1/layer_13/model_list.13.pre_Normal.weight | ciphertext:L3:c2 | 12 | 48.000 |
| token_1/layer_13/model_list.13.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_13/model_list.13.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_13/rope | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_13/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_13/rope | plaintext:L4:c1 | 1 | 2.500 |
| token_1/layer_13/swiglu | ciphertext:L1:c2 | 1347 | 2,694.000 |
| token_1/layer_13/swiglu | ciphertext:L2:c2 | 3136 | 9,408.000 |
| token_1/layer_13/swiglu | ciphertext:L2:c3 | 14 | 63.000 |
| token_1/layer_13/swiglu | ciphertext:L3:c2 | 8058 | 32,232.000 |
| token_1/layer_13/swiglu | ciphertext:L3:c3 | 11 | 66.000 |
| token_1/layer_13/swiglu | ciphertext:L4:c2 | 3364 | 16,820.000 |
| token_1/layer_13/swiglu | ciphertext:L4:c3 | 8 | 60.000 |
| token_1/layer_13/swiglu | ciphertext:L5:c2 | 206 | 1,236.000 |
| token_1/layer_13/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_13/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_1/layer_13/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_1/layer_13/swiglu | plaintext:L3:c1 | 13 | 26.000 |
| token_1/layer_13/swiglu | plaintext:L4:c1 | 446 | 1,115.000 |
| token_1/layer_13/swiglu | plaintext:L5:c1 | 130 | 390.000 |
| token_1/layer_14/_decoder_block | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_14/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_14/gqa_attention | ciphertext:L3:c2 | 133 | 532.000 |
| token_1/layer_14/gqa_attention | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_14/gqa_attention | plaintext:L4:c1 | 14 | 35.000 |
| token_1/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L3:c2 | 152 | 608.000 |
| token_1/layer_14/model_list.14.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_14/model_list.14.attention.k_weight.weight | plaintext:L4:c1 | 10 | 25.000 |
| token_1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c2 | 943 | 3,772.000 |
| token_1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L4:c2 | 44 | 220.000 |
| token_1/layer_14/model_list.14.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L3:c2 | 957 | 3,828.000 |
| token_1/layer_14/model_list.14.attention.q_weight.weight | ciphertext:L4:c2 | 37 | 185.000 |
| token_1/layer_14/model_list.14.attention.q_weight.weight | plaintext:L4:c1 | 84 | 210.000 |
| token_1/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L3:c2 | 148 | 592.000 |
| token_1/layer_14/model_list.14.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_14/model_list.14.attention.v_weight.weight | plaintext:L4:c1 | 16 | 40.000 |
| token_1/layer_14/model_list.14.post_Normal.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_1/layer_14/model_list.14.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_14/model_list.14.pre_Normal.weight | ciphertext:L3:c2 | 95 | 380.000 |
| token_1/layer_14/model_list.14.pre_Normal.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_14/model_list.14.pre_Normal.weight | plaintext:L4:c1 | 10 | 25.000 |
| token_1/layer_14/rope | ciphertext:L3:c2 | 20 | 80.000 |
| token_1/layer_14/swiglu | ciphertext:L1:c2 | 1644 | 3,288.000 |
| token_1/layer_14/swiglu | ciphertext:L2:c2 | 6119 | 18,357.000 |
| token_1/layer_14/swiglu | ciphertext:L2:c3 | 11 | 49.500 |
| token_1/layer_14/swiglu | ciphertext:L3:c2 | 8104 | 32,416.000 |
| token_1/layer_14/swiglu | ciphertext:L3:c3 | 7 | 42.000 |
| token_1/layer_14/swiglu | ciphertext:L4:c2 | 357 | 1,785.000 |
| token_1/layer_14/swiglu | ciphertext:L4:c3 | 3 | 22.500 |
| token_1/layer_14/swiglu | ciphertext:L5:c2 | 54 | 324.000 |
| token_1/layer_14/swiglu | plaintext:L1:c1 | 3 | 3.000 |
| token_1/layer_14/swiglu | plaintext:L2:c1 | 26 | 39.000 |
| token_1/layer_14/swiglu | plaintext:L4:c1 | 364 | 910.000 |
| token_1/layer_15/_decoder_block | ciphertext:L3:c2 | 8 | 32.000 |
| token_1/layer_15/gqa_attention | ciphertext:L2:c2 | 73 | 219.000 |
| token_1/layer_15/gqa_attention | ciphertext:L3:c2 | 66 | 264.000 |
| token_1/layer_15/gqa_attention | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_15/gqa_attention | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L3:c2 | 144 | 576.000 |
| token_1/layer_15/model_list.15.attention.k_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_15/model_list.15.attention.k_weight.weight | plaintext:L4:c1 | 12 | 30.000 |
| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 568 | 1,704.000 |
| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L3:c2 | 114 | 456.000 |
| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L4:c2 | 327 | 1,635.000 |
| token_1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L5:c2 | 15 | 90.000 |
| token_1/layer_15/model_list.15.attention.o_weight.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_1/layer_15/model_list.15.attention.o_weight.weight | plaintext:L5:c1 | 10 | 30.000 |
| token_1/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L3:c2 | 911 | 3,644.000 |
| token_1/layer_15/model_list.15.attention.q_weight.weight | ciphertext:L4:c2 | 39 | 195.000 |
| token_1/layer_15/model_list.15.attention.q_weight.weight | plaintext:L4:c1 | 78 | 195.000 |
| token_1/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L3:c2 | 142 | 568.000 |
| token_1/layer_15/model_list.15.attention.v_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_15/model_list.15.attention.v_weight.weight | plaintext:L4:c1 | 16 | 40.000 |
| token_1/layer_15/model_list.15.post_Normal.weight | ciphertext:L3:c2 | 106 | 424.000 |
| token_1/layer_15/model_list.15.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_15/model_list.15.post_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_1/layer_15/model_list.15.pre_Normal.weight | ciphertext:L3:c2 | 95 | 380.000 |
| token_1/layer_15/model_list.15.pre_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_15/model_list.15.pre_Normal.weight | plaintext:L4:c1 | 8 | 20.000 |
| token_1/layer_15/rope | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_15/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_15/rope | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_15/swiglu | ciphertext:L1:c2 | 275 | 550.000 |
| token_1/layer_15/swiglu | ciphertext:L2:c2 | 2495 | 7,485.000 |
| token_1/layer_15/swiglu | ciphertext:L2:c3 | 8 | 36.000 |
| token_1/layer_15/swiglu | ciphertext:L3:c2 | 6759 | 27,036.000 |
| token_1/layer_15/swiglu | ciphertext:L3:c3 | 5 | 30.000 |
| token_1/layer_15/swiglu | ciphertext:L4:c2 | 6314 | 31,570.000 |
| token_1/layer_15/swiglu | ciphertext:L4:c3 | 3 | 22.500 |
| token_1/layer_15/swiglu | ciphertext:L5:c2 | 270 | 1,620.000 |
| token_1/layer_15/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_1/layer_15/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_1/layer_15/swiglu | plaintext:L2:c1 | 2 | 3.000 |
| token_1/layer_15/swiglu | plaintext:L4:c1 | 316 | 790.000 |
| token_1/layer_15/swiglu | plaintext:L5:c1 | 215 | 645.000 |
| token_1/layer_16/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_16/_decoder_block | ciphertext:L3:c2 | 1 | 4.000 |
| token_1/layer_16/gqa_attention | ciphertext:L1:c2 | 46 | 92.000 |
| token_1/layer_16/gqa_attention | ciphertext:L2:c2 | 27 | 81.000 |
| token_1/layer_16/gqa_attention | ciphertext:L3:c2 | 70 | 280.000 |
| token_1/layer_16/gqa_attention | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L3:c2 | 139 | 556.000 |
| token_1/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L4:c2 | 13 | 65.000 |
| token_1/layer_16/model_list.16.attention.k_weight.weight | ciphertext:L5:c2 | 2 | 12.000 |
| token_1/layer_16/model_list.16.attention.k_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L1:c2 | 172 | 344.000 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L2:c2 | 111 | 333.000 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L2:c3 | 14 | 63.000 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 672 | 2,688.000 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c3 | 9 | 54.000 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c2 | 39 | 195.000 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 22.500 |
| token_1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L5:c2 | 17 | 102.000 |
| token_1/layer_16/model_list.16.attention.q_weight.weight | ciphertext:L3:c2 | 911 | 3,644.000 |
| token_1/layer_16/model_list.16.attention.q_weight.weight | ciphertext:L4:c2 | 55 | 275.000 |
| token_1/layer_16/model_list.16.attention.q_weight.weight | plaintext:L4:c1 | 46 | 115.000 |
| token_1/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L3:c2 | 143 | 572.000 |
| token_1/layer_16/model_list.16.attention.v_weight.weight | ciphertext:L4:c2 | 13 | 65.000 |
| token_1/layer_16/model_list.16.post_Normal.weight | ciphertext:L2:c2 | 103 | 309.000 |
| token_1/layer_16/model_list.16.post_Normal.weight | ciphertext:L3:c2 | 9 | 36.000 |
| token_1/layer_16/model_list.16.pre_Normal.weight | ciphertext:L3:c2 | 109 | 436.000 |
| token_1/layer_16/model_list.16.pre_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_16/model_list.16.pre_Normal.weight | plaintext:L4:c1 | 6 | 15.000 |
| token_1/layer_16/rope | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_16/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_16/swiglu | ciphertext:L1:c2 | 1208 | 2,416.000 |
| token_1/layer_16/swiglu | ciphertext:L2:c2 | 4384 | 13,152.000 |
| token_1/layer_16/swiglu | ciphertext:L2:c3 | 5 | 22.500 |
| token_1/layer_16/swiglu | ciphertext:L3:c2 | 10259 | 41,036.000 |
| token_1/layer_16/swiglu | ciphertext:L3:c3 | 6 | 36.000 |
| token_1/layer_16/swiglu | ciphertext:L4:c2 | 375 | 1,875.000 |
| token_1/layer_16/swiglu | ciphertext:L4:c3 | 2 | 15.000 |
| token_1/layer_16/swiglu | ciphertext:L5:c2 | 24 | 144.000 |
| token_1/layer_16/swiglu | plaintext:L1:c1 | 2 | 2.000 |
| token_1/layer_16/swiglu | plaintext:L2:c1 | 8 | 12.000 |
| token_1/layer_16/swiglu | plaintext:L4:c1 | 504 | 1,260.000 |
| token_1/layer_17/_decoder_block | ciphertext:L2:c2 | 3 | 9.000 |
| token_1/layer_17/_decoder_block | ciphertext:L3:c2 | 3 | 12.000 |
| token_1/layer_17/gqa_attention | ciphertext:L3:c2 | 137 | 548.000 |
| token_1/layer_17/gqa_attention | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_17/gqa_attention | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L1:c2 | 77 | 154.000 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L2:c2 | 47 | 141.000 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L2:c3 | 3 | 13.500 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L3:c3 | 1 | 6.000 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L4:c3 | 1 | 7.500 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L5:c2 | 4 | 24.000 |
| token_1/layer_17/model_list.17.attention.k_weight.weight | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L3:c2 | 943 | 3,772.000 |
| token_1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L4:c2 | 45 | 225.000 |
| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L1:c2 | 282 | 564.000 |
| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L2:c2 | 556 | 1,668.000 |
| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L2:c3 | 1 | 4.500 |
| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L3:c2 | 137 | 548.000 |
| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_17/model_list.17.attention.q_weight.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_1/layer_17/model_list.17.attention.q_weight.weight | plaintext:L4:c1 | 3 | 7.500 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L1:c2 | 86 | 172.000 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L2:c2 | 34 | 102.000 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L2:c3 | 3 | 13.500 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L3:c2 | 20 | 80.000 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L3:c3 | 3 | 18.000 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L4:c3 | 2 | 15.000 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | ciphertext:L5:c2 | 3 | 18.000 |
| token_1/layer_17/model_list.17.attention.v_weight.weight | plaintext:L4:c1 | 1 | 2.500 |
| token_1/layer_17/model_list.17.post_Normal.weight | ciphertext:L2:c2 | 4 | 12.000 |
| token_1/layer_17/model_list.17.post_Normal.weight | ciphertext:L3:c2 | 95 | 380.000 |
| token_1/layer_17/model_list.17.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_17/model_list.17.pre_Normal.weight | ciphertext:L2:c2 | 71 | 213.000 |
| token_1/layer_17/model_list.17.pre_Normal.weight | ciphertext:L3:c2 | 33 | 132.000 |
| token_1/layer_17/model_list.17.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_17/model_list.17.pre_Normal.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_1/layer_17/rope | ciphertext:L3:c2 | 19 | 76.000 |
| token_1/layer_17/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_17/swiglu | ciphertext:L1:c2 | 834 | 1,668.000 |
| token_1/layer_17/swiglu | ciphertext:L2:c2 | 3837 | 11,511.000 |
| token_1/layer_17/swiglu | ciphertext:L2:c3 | 14 | 63.000 |
| token_1/layer_17/swiglu | ciphertext:L3:c2 | 3742 | 14,968.000 |
| token_1/layer_17/swiglu | ciphertext:L3:c3 | 9 | 54.000 |
| token_1/layer_17/swiglu | ciphertext:L4:c2 | 7407 | 37,035.000 |
| token_1/layer_17/swiglu | ciphertext:L4:c3 | 5 | 37.500 |
| token_1/layer_17/swiglu | ciphertext:L5:c2 | 302 | 1,812.000 |
| token_1/layer_17/swiglu | ciphertext:L5:c3 | 1 | 9.000 |
| token_1/layer_17/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_17/swiglu | plaintext:L4:c1 | 205 | 512.500 |
| token_1/layer_17/swiglu | plaintext:L5:c1 | 272 | 816.000 |
| token_1/layer_18/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_18/gqa_attention | ciphertext:L3:c2 | 145 | 580.000 |
| token_1/layer_18/gqa_attention | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L3:c2 | 153 | 612.000 |
| token_1/layer_18/model_list.18.attention.k_weight.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 946 | 3,784.000 |
| token_1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L4:c2 | 42 | 210.000 |
| token_1/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L3:c2 | 970 | 3,880.000 |
| token_1/layer_18/model_list.18.attention.q_weight.weight | ciphertext:L4:c2 | 19 | 95.000 |
| token_1/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L3:c2 | 152 | 608.000 |
| token_1/layer_18/model_list.18.attention.v_weight.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_18/model_list.18.post_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_1/layer_18/model_list.18.post_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_18/model_list.18.pre_Normal.weight | ciphertext:L2:c2 | 61 | 183.000 |
| token_1/layer_18/model_list.18.pre_Normal.weight | ciphertext:L3:c2 | 42 | 168.000 |
| token_1/layer_18/model_list.18.pre_Normal.weight | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_18/rope | ciphertext:L3:c2 | 19 | 76.000 |
| token_1/layer_18/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_18/swiglu | ciphertext:L1:c2 | 1517 | 3,034.000 |
| token_1/layer_18/swiglu | ciphertext:L2:c2 | 5571 | 16,713.000 |
| token_1/layer_18/swiglu | ciphertext:L2:c3 | 16 | 72.000 |
| token_1/layer_18/swiglu | ciphertext:L3:c2 | 8979 | 35,916.000 |
| token_1/layer_18/swiglu | ciphertext:L3:c3 | 12 | 72.000 |
| token_1/layer_18/swiglu | ciphertext:L4:c2 | 354 | 1,770.000 |
| token_1/layer_18/swiglu | ciphertext:L4:c3 | 7 | 52.500 |
| token_1/layer_18/swiglu | ciphertext:L5:c2 | 43 | 258.000 |
| token_1/layer_18/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_18/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_1/layer_18/swiglu | plaintext:L4:c1 | 25 | 62.500 |
| token_1/layer_19/_decoder_block | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_19/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_19/gqa_attention | ciphertext:L3:c2 | 146 | 584.000 |
| token_1/layer_19/gqa_attention | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_1/layer_19/model_list.19.attention.k_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L3:c2 | 950 | 3,800.000 |
| token_1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L4:c2 | 38 | 190.000 |
| token_1/layer_19/model_list.19.attention.q_weight.weight | ciphertext:L3:c2 | 947 | 3,788.000 |
| token_1/layer_19/model_list.19.attention.q_weight.weight | ciphertext:L4:c2 | 42 | 210.000 |
| token_1/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L3:c2 | 150 | 600.000 |
| token_1/layer_19/model_list.19.attention.v_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_19/model_list.19.post_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_1/layer_19/model_list.19.post_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_19/model_list.19.pre_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_1/layer_19/model_list.19.pre_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_19/rope | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_19/rope | ciphertext:L4:c2 | 2 | 10.000 |
| token_1/layer_19/swiglu | ciphertext:L1:c2 | 1424 | 2,848.000 |
| token_1/layer_19/swiglu | ciphertext:L2:c2 | 5619 | 16,857.000 |
| token_1/layer_19/swiglu | ciphertext:L2:c3 | 16 | 72.000 |
| token_1/layer_19/swiglu | ciphertext:L3:c2 | 8878 | 35,512.000 |
| token_1/layer_19/swiglu | ciphertext:L3:c3 | 12 | 72.000 |
| token_1/layer_19/swiglu | ciphertext:L4:c2 | 382 | 1,910.000 |
| token_1/layer_19/swiglu | ciphertext:L4:c3 | 9 | 67.500 |
| token_1/layer_19/swiglu | ciphertext:L5:c2 | 42 | 252.000 |
| token_1/layer_19/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_19/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_1/layer_19/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_19/swiglu | plaintext:L4:c1 | 141 | 352.500 |
| token_1/layer_20/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_20/gqa_attention | ciphertext:L3:c2 | 144 | 576.000 |
| token_1/layer_20/gqa_attention | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L3:c2 | 148 | 592.000 |
| token_1/layer_20/model_list.20.attention.k_weight.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 943 | 3,772.000 |
| token_1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L4:c2 | 45 | 225.000 |
| token_1/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L3:c2 | 940 | 3,760.000 |
| token_1/layer_20/model_list.20.attention.q_weight.weight | ciphertext:L4:c2 | 45 | 225.000 |
| token_1/layer_20/model_list.20.attention.q_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L3:c2 | 150 | 600.000 |
| token_1/layer_20/model_list.20.attention.v_weight.weight | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_20/model_list.20.post_Normal.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_1/layer_20/model_list.20.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_20/model_list.20.pre_Normal.weight | ciphertext:L3:c2 | 97 | 388.000 |
| token_1/layer_20/model_list.20.pre_Normal.weight | ciphertext:L4:c2 | 3 | 15.000 |
| token_1/layer_20/model_list.20.pre_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_20/rope | ciphertext:L3:c2 | 20 | 80.000 |
| token_1/layer_20/swiglu | ciphertext:L1:c2 | 1431 | 2,862.000 |
| token_1/layer_20/swiglu | ciphertext:L2:c2 | 5560 | 16,680.000 |
| token_1/layer_20/swiglu | ciphertext:L2:c3 | 16 | 72.000 |
| token_1/layer_20/swiglu | ciphertext:L3:c2 | 9012 | 36,048.000 |
| token_1/layer_20/swiglu | ciphertext:L3:c3 | 13 | 78.000 |
| token_1/layer_20/swiglu | ciphertext:L4:c2 | 407 | 2,035.000 |
| token_1/layer_20/swiglu | ciphertext:L4:c3 | 9 | 67.500 |
| token_1/layer_20/swiglu | ciphertext:L5:c2 | 42 | 252.000 |
| token_1/layer_20/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_20/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_1/layer_20/swiglu | plaintext:L2:c1 | 1 | 1.500 |
| token_1/layer_20/swiglu | plaintext:L4:c1 | 33 | 82.500 |
| token_1/layer_21/_decoder_block | ciphertext:L3:c2 | 6 | 24.000 |
| token_1/layer_21/gqa_attention | ciphertext:L3:c2 | 143 | 572.000 |
| token_1/layer_21/gqa_attention | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L3:c2 | 151 | 604.000 |
| token_1/layer_21/model_list.21.attention.k_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 946 | 3,784.000 |
| token_1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c2 | 42 | 210.000 |
| token_1/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L3:c2 | 951 | 3,804.000 |
| token_1/layer_21/model_list.21.attention.q_weight.weight | ciphertext:L4:c2 | 38 | 190.000 |
| token_1/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L3:c2 | 148 | 592.000 |
| token_1/layer_21/model_list.21.attention.v_weight.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_1/layer_21/model_list.21.post_Normal.weight | ciphertext:L3:c2 | 99 | 396.000 |
| token_1/layer_21/model_list.21.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_21/model_list.21.pre_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_1/layer_21/model_list.21.pre_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_21/rope | ciphertext:L3:c2 | 19 | 76.000 |
| token_1/layer_21/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_21/swiglu | ciphertext:L1:c2 | 1510 | 3,020.000 |
| token_1/layer_21/swiglu | ciphertext:L2:c2 | 5576 | 16,728.000 |
| token_1/layer_21/swiglu | ciphertext:L2:c3 | 16 | 72.000 |
| token_1/layer_21/swiglu | ciphertext:L3:c2 | 8824 | 35,296.000 |
| token_1/layer_21/swiglu | ciphertext:L3:c3 | 13 | 78.000 |
| token_1/layer_21/swiglu | ciphertext:L4:c2 | 376 | 1,880.000 |
| token_1/layer_21/swiglu | ciphertext:L4:c3 | 9 | 67.500 |
| token_1/layer_21/swiglu | ciphertext:L5:c2 | 43 | 258.000 |
| token_1/layer_21/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_21/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_1/layer_21/swiglu | plaintext:L4:c1 | 157 | 392.500 |
| token_1/layer_22/_decoder_block | ciphertext:L3:c2 | 5 | 20.000 |
| token_1/layer_22/_decoder_block | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_22/gqa_attention | ciphertext:L3:c2 | 143 | 572.000 |
| token_1/layer_22/gqa_attention | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_1/layer_22/model_list.22.attention.k_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 944 | 3,776.000 |
| token_1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L4:c2 | 44 | 220.000 |
| token_1/layer_22/model_list.22.attention.q_weight.weight | ciphertext:L3:c2 | 932 | 3,728.000 |
| token_1/layer_22/model_list.22.attention.q_weight.weight | ciphertext:L4:c2 | 44 | 220.000 |
| token_1/layer_22/model_list.22.attention.q_weight.weight | plaintext:L4:c1 | 13 | 32.500 |
| token_1/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L3:c2 | 149 | 596.000 |
| token_1/layer_22/model_list.22.attention.v_weight.weight | ciphertext:L4:c2 | 7 | 35.000 |
| token_1/layer_22/model_list.22.post_Normal.weight | ciphertext:L3:c2 | 100 | 400.000 |
| token_1/layer_22/model_list.22.post_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_22/model_list.22.pre_Normal.weight | ciphertext:L3:c2 | 96 | 384.000 |
| token_1/layer_22/model_list.22.pre_Normal.weight | ciphertext:L4:c2 | 4 | 20.000 |
| token_1/layer_22/model_list.22.pre_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_22/rope | ciphertext:L3:c2 | 19 | 76.000 |
| token_1/layer_22/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_22/swiglu | ciphertext:L1:c2 | 1481 | 2,962.000 |
| token_1/layer_22/swiglu | ciphertext:L2:c2 | 5538 | 16,614.000 |
| token_1/layer_22/swiglu | ciphertext:L2:c3 | 16 | 72.000 |
| token_1/layer_22/swiglu | ciphertext:L3:c2 | 8756 | 35,024.000 |
| token_1/layer_22/swiglu | ciphertext:L3:c3 | 13 | 78.000 |
| token_1/layer_22/swiglu | ciphertext:L4:c2 | 383 | 1,915.000 |
| token_1/layer_22/swiglu | ciphertext:L4:c3 | 9 | 67.500 |
| token_1/layer_22/swiglu | ciphertext:L5:c2 | 44 | 264.000 |
| token_1/layer_22/swiglu | ciphertext:L5:c3 | 2 | 18.000 |
| token_1/layer_22/swiglu | plaintext:L1:c1 | 1 | 1.000 |
| token_1/layer_22/swiglu | plaintext:L4:c1 | 284 | 710.000 |
| token_1/layer_23/_decoder_block | ciphertext:L2:c2 | 1 | 3.000 |
| token_1/layer_23/_decoder_block | ciphertext:L3:c2 | 4 | 16.000 |
| token_1/layer_23/_decoder_block | plaintext:L3:c1 | 1 | 2.000 |
| token_1/layer_23/gqa_attention | ciphertext:L3:c2 | 139 | 556.000 |
| token_1/layer_23/gqa_attention | ciphertext:L4:c2 | 6 | 30.000 |
| token_1/layer_23/gqa_attention | plaintext:L4:c1 | 5 | 12.500 |
| token_1/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L3:c2 | 147 | 588.000 |
| token_1/layer_23/model_list.23.attention.k_weight.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_23/model_list.23.attention.k_weight.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L3:c2 | 908 | 3,632.000 |
| token_1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L4:c2 | 40 | 200.000 |
| token_1/layer_23/model_list.23.attention.o_weight.weight | plaintext:L4:c1 | 40 | 100.000 |
| token_1/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L3:c2 | 910 | 3,640.000 |
| token_1/layer_23/model_list.23.attention.q_weight.weight | ciphertext:L4:c2 | 39 | 195.000 |
| token_1/layer_23/model_list.23.attention.q_weight.weight | plaintext:L4:c1 | 40 | 100.000 |
| token_1/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L3:c2 | 141 | 564.000 |
| token_1/layer_23/model_list.23.attention.v_weight.weight | ciphertext:L4:c2 | 8 | 40.000 |
| token_1/layer_23/model_list.23.attention.v_weight.weight | plaintext:L4:c1 | 7 | 17.500 |
| token_1/layer_23/model_list.23.post_Normal.weight | ciphertext:L3:c2 | 95 | 380.000 |
| token_1/layer_23/model_list.23.post_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_23/model_list.23.post_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_23/model_list.23.pre_Normal.weight | ciphertext:L3:c2 | 95 | 380.000 |
| token_1/layer_23/model_list.23.pre_Normal.weight | ciphertext:L4:c2 | 5 | 25.000 |
| token_1/layer_23/model_list.23.pre_Normal.weight | plaintext:L4:c1 | 4 | 10.000 |
| token_1/layer_23/rope | ciphertext:L3:c2 | 18 | 72.000 |
| token_1/layer_23/rope | ciphertext:L4:c2 | 1 | 5.000 |
| token_1/layer_23/rope | plaintext:L4:c1 | 1 | 2.500 |
| token_1/layer_23/swiglu | ciphertext:L1:c2 | 1227 | 2,454.000 |
| token_1/layer_23/swiglu | ciphertext:L2:c2 | 7035 | 21,105.000 |
| token_1/layer_23/swiglu | ciphertext:L2:c3 | 8 | 36.000 |
| token_1/layer_23/swiglu | ciphertext:L3:c2 | 6706 | 26,824.000 |
| token_1/layer_23/swiglu | ciphertext:L3:c3 | 7 | 42.000 |
| token_1/layer_23/swiglu | ciphertext:L4:c2 | 202 | 1,010.000 |
| token_1/layer_23/swiglu | ciphertext:L4:c3 | 3 | 22.500 |
| token_1/layer_23/swiglu | ciphertext:L5:c2 | 15 | 90.000 |
| token_1/layer_23/swiglu | plaintext:L3:c1 | 1140 | 2,280.000 |
| token_1/layer_23/swiglu | plaintext:L4:c1 | 184 | 460.000 |
| token_shared/layer_shared/final_normal.weight | ciphertext:L2:c2 | 55 | 165.000 |
| token_shared/layer_shared/final_normal.weight | ciphertext:L3:c2 | 27 | 108.000 |
| token_shared/layer_shared/final_normal.weight | plaintext:L3:c1 | 27 | 54.000 |
| token_shared/layer_shared/interface | ciphertext:L2:c2 | 21 | 63.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L1:c2 | 866 | 1,732.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L2:c2 | 8819 | 26,457.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L2:c3 | 9 | 40.500 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L3:c2 | 3982 | 15,928.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L3:c3 | 5 | 30.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L4:c2 | 2840 | 14,200.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L4:c3 | 4 | 30.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L5:c2 | 145 | 870.000 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L5:c3 | 1 | 9.000 |
| token_shared/layer_shared/lm_head.weight | plaintext:L1:c1 | 56 | 56.000 |
| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 60 | 90.000 |
| token_shared/layer_shared/lm_head.weight | plaintext:L3:c1 | 3864 | 7,728.000 |
| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 2 | 5.000 |
| token_shared/layer_shared/lm_head.weight | plaintext:L5:c1 | 123 | 369.000 |
| token_shared/layer_shared/pack_kv | ciphertext:L1:c2 | 28 | 56.000 |
| token_shared/layer_shared/pack_kv | ciphertext:L2:c2 | 104 | 312.000 |
| token_shared/layer_shared/pack_kv | ciphertext:L3:c2 | 213 | 852.000 |
| token_shared/layer_shared/pack_kv | ciphertext:L4:c2 | 20 | 100.000 |
| token_shared/layer_shared/pack_kv | ciphertext:L5:c2 | 1 | 6.000 |
| token_shared/layer_shared/pack_kv | plaintext:L4:c1 | 3 | 7.500 |

## 6. 每次自举的位置

ordinal 是优化后计划的指令序号，ValueId 可与计划逐项对应。完整相邻生产者和消费者见 JSON。这里只是编译计划中的 native Boot，没有执行这些自举。

| 序号 | ordinal | 输入 Value/L | 输出 Value/L | 源码来源 |
| --- | --- | --- | --- | --- |
| 1 | 85 | 106/L1 | 110/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.o_weight.weight |
| 2 | 142 | 186/L1 | 209/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 3 | 158 | 189/L1 | 210/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 4 | 185 | 192/L1 | 211/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 5 | 214 | 195/L1 | 212/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 6 | 266 | 274/L1 | 289/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 7 | 693 | 109/L1 | 111/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.o_weight.weight |
| 8 | 744 | 199/L1 | 213/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 9 | 760 | 202/L1 | 214/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 10 | 785 | 205/L1 | 215/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 11 | 812 | 208/L1 | 216/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 12 | 863 | 288/L1 | 290/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 13 | 1206 | 2052/L1 | 2058/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 14 | 1258 | 2174/L1 | 2230/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 15 | 3413 | 2021/L1 | 2055/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 16 | 3452 | 2213/L1 | 2233/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 17 | 3796 | 1904/L1 | 2053/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 18 | 4173 | 2036/L1 | 2056/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 19 | 4211 | 2210/L1 | 2232/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 20 | 4294 | 3351/L1 | 3353/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 21 | 4332 | 3411/L1 | 3413/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 22 | 4407 | 3487/L1 | 3496/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 23 | 4418 | 1920/L1 | 2054/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 24 | 4421 | 2051/L1 | 2057/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 25 | 4477 | 2219/L1 | 2234/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 26 | 4497 | 3546/L1 | 3548/L5 | qwen24-stage/token_0/layer_04/model_list.4.attention.o_weight.weight ; qwen24-stage/token_1/layer_04/model_list.4.attention.o_weight.weight |
| 27 | 4581 | 3608/L1 | 4210/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 28 | 4670 | 3633/L1 | 4211/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 29 | 4760 | 3658/L1 | 4212/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 30 | 4850 | 3683/L1 | 4213/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 31 | 4940 | 3708/L1 | 4214/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 32 | 5030 | 3733/L1 | 4215/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 33 | 5120 | 3758/L1 | 4216/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 34 | 5210 | 3783/L1 | 4217/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 35 | 5300 | 3808/L1 | 4218/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 36 | 5390 | 3833/L1 | 4219/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 37 | 5480 | 3858/L1 | 4220/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 38 | 5570 | 3883/L1 | 4221/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 39 | 5660 | 3908/L1 | 4222/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 40 | 5750 | 3933/L1 | 4223/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 41 | 5840 | 3958/L1 | 4224/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 42 | 5930 | 3983/L1 | 4225/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 43 | 6020 | 4008/L1 | 4226/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 44 | 6110 | 4033/L1 | 4227/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 45 | 6200 | 4058/L1 | 4228/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 46 | 6290 | 4083/L1 | 4229/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 47 | 6380 | 4108/L1 | 4230/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 48 | 6470 | 4133/L1 | 4231/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 49 | 6560 | 4158/L1 | 4232/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 50 | 6650 | 4183/L1 | 4233/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 51 | 6741 | 4208/L1 | 4234/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 52 | 6790 | 16412/L1 | 16413/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 53 | 6801 | 16411/L1 | 16416/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 54 | 6889 | 16468/L1 | 16490/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 55 | 6898 | 16471/L1 | 16491/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 56 | 6909 | 16475/L1 | 16492/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 57 | 6938 | 16478/L1 | 16493/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 58 | 6953 | 16487/L1 | 16488/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 59 | 6978 | 16481/L1 | 16494/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 60 | 7007 | 16484/L1 | 16495/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 61 | 7021 | 16564/L1 | 16572/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 62 | 7040 | 16567/L1 | 16573/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 63 | 7088 | 16624/L1 | 16626/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 64 | 18790 | 34651/L1 | 34654/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 65 | 18815 | 34704/L1 | 34706/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 66 | 18838 | 34778/L1 | 34791/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 67 | 18869 | 34782/L1 | 34792/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_00/swiglu |
| 68 | 18912 | 34786/L1 | 34793/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 69 | 30626 | 34046/L1 | 34652/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_00/swiglu |
| 70 | 30649 | 34838/L1 | 34840/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_00/swiglu |
| 71 | 43559 | 41654/L1 | 41655/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 72 | 43641 | 2206/L1 | 2231/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 73 | 45795 | 3335/L1 | 3352/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 74 | 45874 | 3407/L1 | 3412/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 75 | 45914 | 3477/L1 | 3493/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 76 | 45923 | 3480/L1 | 3494/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 77 | 45931 | 3453/L1 | 3490/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 78 | 45950 | 3483/L1 | 3495/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 79 | 45964 | 3457/L1 | 3491/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 80 | 45992 | 3461/L1 | 3492/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 81 | 46008 | 3488/L1 | 3489/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 82 | 46036 | 3543/L1 | 3547/L5 | qwen24-stage/token_0/layer_04/model_list.4.attention.o_weight.weight ; qwen24-stage/token_1/layer_04/model_list.4.attention.o_weight.weight |
| 83 | 46076 | 3583/L1 | 4209/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 84 | 57771 | 15866/L1 | 16414/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 85 | 57791 | 16465/L1 | 16489/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 86 | 57812 | 16554/L1 | 16571/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 87 | 57834 | 16524/L1 | 16568/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 88 | 57865 | 16529/L1 | 16569/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 89 | 57882 | 16534/L1 | 16570/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 90 | 57894 | 16485/L1 | 16486/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 91 | 57901 | 16610/L1 | 16625/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 92 | 69591 | 16408/L1 | 16415/L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 93 | 82515 | 34648/L1 | 34653/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 94 | 82603 | 34701/L1 | 34705/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 95 | 82658 | 34766/L1 | 34787/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 96 | 82674 | 34769/L1 | 34788/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_00/swiglu |
| 97 | 82703 | 34772/L1 | 34789/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 98 | 82734 | 34775/L1 | 34790/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 99 | 82760 | 34828/L1 | 34839/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 100 | 83140 | 41651/L1 | 41657/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 101 | 83149 | 41653/L1 | 41658/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 102 | 83238 | 41754/L1 | 41779/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 103 | 83247 | 41757/L1 | 41780/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 104 | 83258 | 41760/L1 | 41781/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 105 | 83284 | 41763/L1 | 41782/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_00/swiglu |
| 106 | 83299 | 41776/L1 | 41777/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 107 | 83320 | 41766/L1 | 41783/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 108 | 83346 | 41769/L1 | 41784/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 109 | 83360 | 42902/L1 | 42907/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 110 | 83379 | 42905/L1 | 42908/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 111 | 83430 | 42972/L1 | 42974/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 112 | 83774 | 44431/L1 | 44439/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 113 | 83826 | 44519/L1 | 44594/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 114 | 85979 | 44399/L1 | 44436/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 115 | 86018 | 44546/L1 | 44595/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 116 | 86367 | 41636/L1 | 41656/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 117 | 86406 | 41774/L1 | 41775/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 118 | 86749 | 44414/L1 | 44437/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 119 | 86804 | 44584/L1 | 44599/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 120 | 86830 | 44593/L1 | 44600/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 121 | 86902 | 56338/L1 | 56339/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 122 | 86912 | 56336/L1 | 56341/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 123 | 86986 | 56380/L1 | 56382/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 124 | 86998 | 44429/L1 | 44438/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 125 | 87054 | 44552/L1 | 44596/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 126 | 87074 | 56423/L1 | 56427/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 127 | 87188 | 57105/L1 | 57444/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 128 | 87276 | 57119/L1 | 57445/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 129 | 87366 | 57133/L1 | 57446/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 130 | 87456 | 57147/L1 | 57447/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 131 | 87546 | 57161/L1 | 57448/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 132 | 87636 | 57175/L1 | 57449/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 133 | 87726 | 57189/L1 | 57450/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 134 | 87816 | 57203/L1 | 57451/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 135 | 87906 | 57217/L1 | 57452/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 136 | 87996 | 57231/L1 | 57453/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 137 | 88086 | 57245/L1 | 57454/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_00/swiglu |
| 138 | 88176 | 57259/L1 | 57455/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 139 | 88266 | 57273/L1 | 57456/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 140 | 88356 | 57287/L1 | 57457/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 141 | 88446 | 57301/L1 | 57458/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 142 | 88536 | 57315/L1 | 57459/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 143 | 88626 | 57329/L1 | 57460/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 144 | 88716 | 57343/L1 | 57461/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 145 | 88806 | 57357/L1 | 57462/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 146 | 88896 | 57371/L1 | 57463/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 147 | 88986 | 57385/L1 | 57464/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 148 | 89076 | 57399/L1 | 57465/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 149 | 89166 | 57413/L1 | 57466/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 150 | 89256 | 57427/L1 | 57467/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 151 | 89348 | 57442/L1 | 57468/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 152 | 89438 | 64081/L1 | 64083/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 153 | 89474 | 64150/L1 | 64163/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 154 | 89483 | 64153/L1 | 64164/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 155 | 89491 | 64137/L1 | 64159/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 156 | 89510 | 64156/L1 | 64165/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 157 | 89524 | 64140/L1 | 64160/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 158 | 89552 | 64143/L1 | 64161/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 159 | 89568 | 64157/L1 | 64158/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 160 | 89594 | 64250/L1 | 64255/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 161 | 89635 | 64291/L1 | 64293/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 162 | 101331 | 75527/L1 | 76078/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 163 | 101345 | 76238/L1 | 76241/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 164 | 101365 | 76306/L1 | 76323/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 165 | 101390 | 77425/L1 | 77427/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 166 | 101399 | 76309/L1 | 76324/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 167 | 101424 | 76312/L1 | 76325/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 168 | 101441 | 76315/L1 | 76326/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 169 | 113156 | 76069/L1 | 76079/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 170 | 113179 | 77481/L1 | 77483/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 171 | 126089 | 83817/L1 | 83826/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 172 | 126176 | 41751/L1 | 41778/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 173 | 128332 | 42892/L1 | 42906/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 174 | 128424 | 42957/L1 | 42973/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 175 | 128479 | 44290/L1 | 44432/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 176 | 128495 | 44293/L1 | 44433/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 177 | 128525 | 44296/L1 | 44434/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 178 | 128556 | 44299/L1 | 44435/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 179 | 128609 | 44575/L1 | 44598/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 180 | 140316 | 56333/L1 | 56340/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 181 | 140337 | 56376/L1 | 56381/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 182 | 140357 | 56414/L1 | 56424/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 183 | 140385 | 56417/L1 | 56425/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 184 | 140424 | 56421/L1 | 56426/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 185 | 152118 | 56316/L1 | 56337/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 186 | 152141 | 57090/L1 | 57443/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 187 | 165047 | 64047/L1 | 64082/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 188 | 165128 | 64146/L1 | 64162/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 189 | 165186 | 64238/L1 | 64251/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 190 | 165201 | 64241/L1 | 64252/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 191 | 165228 | 64244/L1 | 64253/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 192 | 165257 | 64247/L1 | 64254/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 193 | 165275 | 64277/L1 | 64292/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 194 | 165365 | 74910/L1 | 76074/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 195 | 165450 | 74935/L1 | 76075/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 196 | 165540 | 74960/L1 | 76076/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 197 | 165630 | 74985/L1 | 76077/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 198 | 165676 | 76235/L1 | 76240/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 199 | 165760 | 83902/L1 | 83904/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 200 | 165790 | 83948/L1 | 83956/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 201 | 165799 | 83951/L1 | 83957/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 202 | 165824 | 83954/L1 | 83958/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 203 | 165894 | 95683/L1 | 95686/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 204 | 165935 | 95711/L1 | 95713/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 205 | 166283 | 97152/L1 | 97160/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 206 | 166335 | 97237/L1 | 97323/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 207 | 168492 | 97120/L1 | 97157/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 208 | 168536 | 97263/L1 | 97279/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 209 | 168589 | 74810/L1 | 76070/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 210 | 168674 | 74835/L1 | 76071/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 211 | 168764 | 74860/L1 | 76072/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 212 | 168855 | 74885/L1 | 76073/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 213 | 168899 | 76219/L1 | 76239/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 214 | 168939 | 76320/L1 | 76321/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 215 | 169288 | 97135/L1 | 97158/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 216 | 169326 | 97261/L1 | 97278/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 217 | 169376 | 97322/L1 | 97325/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 218 | 169442 | 103811/L1 | 103812/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 219 | 169450 | 103810/L1 | 103814/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 220 | 169526 | 103884/L1 | 103886/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 221 | 169536 | 97150/L1 | 97159/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 222 | 169592 | 97269/L1 | 97280/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 223 | 169642 | 103970/L1 | 103973/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 224 | 171797 | 105048/L1 | 105050/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 225 | 171881 | 105435/L1 | 105438/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 226 | 171932 | 105529/L1 | 105546/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 227 | 171948 | 105532/L1 | 105547/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 228 | 171977 | 105535/L1 | 105548/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 229 | 172008 | 105538/L1 | 105549/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 230 | 172062 | 106662/L1 | 106666/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 231 | 183758 | 117601/L1 | 118152/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 232 | 183779 | 118205/L1 | 118210/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 233 | 183800 | 118254/L1 | 118259/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 234 | 183820 | 118230/L1 | 118255/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 235 | 183851 | 118233/L1 | 118256/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 236 | 183868 | 118236/L1 | 118257/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 237 | 183880 | 118206/L1 | 118207/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 238 | 195579 | 118143/L1 | 118153/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 239 | 195602 | 129988/L1 | 129991/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 240 | 208505 | 136301/L1 | 136303/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 241 | 208590 | 76303/L1 | 76322/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 242 | 210737 | 77422/L1 | 77426/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 243 | 210815 | 77471/L1 | 77482/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 244 | 210853 | 83211/L1 | 83823/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 245 | 210862 | 83214/L1 | 83824/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 246 | 210872 | 82001/L1 | 83820/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 247 | 210891 | 83217/L1 | 83825/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 248 | 210907 | 82004/L1 | 83821/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 249 | 210937 | 82007/L1 | 83822/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 250 | 210953 | 83818/L1 | 83819/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 251 | 210980 | 83899/L1 | 83903/L5 | qwen24-stage/token_0/layer_01/swiglu ; qwen24-stage/token_1/layer_01/swiglu |
| 252 | 211020 | 83943/L1 | 83955/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 253 | 222712 | 95138/L1 | 95684/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 254 | 222726 | 95696/L1 | 95712/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 255 | 222748 | 97011/L1 | 97153/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 256 | 222772 | 97257/L1 | 97277/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 257 | 222781 | 97014/L1 | 97154/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 258 | 222804 | 97017/L1 | 97155/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 259 | 222821 | 97020/L1 | 97156/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 260 | 234530 | 95680/L1 | 95685/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 261 | 234553 | 97316/L1 | 97324/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 262 | 247458 | 103807/L1 | 103813/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 263 | 247547 | 103880/L1 | 103885/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 264 | 247661 | 103953/L1 | 103971/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 265 | 247680 | 103956/L1 | 103972/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 266 | 247731 | 104948/L1 | 105049/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 267 | 248085 | 105432/L1 | 105437/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 268 | 248169 | 136371/L1 | 136375/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 269 | 248220 | 136433/L1 | 136444/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 270 | 248236 | 136436/L1 | 136445/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 271 | 248266 | 136439/L1 | 136446/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 272 | 248297 | 136442/L1 | 136447/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 273 | 248348 | 142934/L1 | 142938/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_03/swiglu |
| 274 | 248692 | 144406/L1 | 144418/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 275 | 248744 | 144509/L1 | 144559/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 276 | 250897 | 144374/L1 | 144415/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 277 | 250937 | 144545/L1 | 144563/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 278 | 251285 | 105416/L1 | 105436/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 279 | 251325 | 105543/L1 | 105544/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 280 | 251668 | 144389/L1 | 144416/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 281 | 251706 | 144543/L1 | 144562/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_03/swiglu |
| 282 | 251787 | 144629/L1 | 144631/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 283 | 251834 | 145001/L1 | 145015/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 284 | 251843 | 144998/L1 | 145014/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 285 | 251853 | 145006/L1 | 145016/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 286 | 251884 | 145011/L1 | 145017/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 287 | 251915 | 145096/L1 | 145104/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 288 | 251920 | 144404/L1 | 144417/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 289 | 251976 | 144551/L1 | 144564/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 290 | 252024 | 146204/L1 | 146206/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 291 | 254178 | 147322/L1 | 147324/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 292 | 254262 | 147409/L1 | 147414/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 293 | 254364 | 147505/L1 | 147510/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 294 | 254383 | 147508/L1 | 147511/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 295 | 254430 | 159227/L1 | 159230/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_03/swiglu |
| 296 | 266146 | 170664/L1 | 170669/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 297 | 266171 | 170706/L1 | 170710/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 298 | 266195 | 170753/L1 | 170762/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 299 | 266227 | 170756/L1 | 170763/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 300 | 266270 | 170760/L1 | 170764/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 301 | 277980 | 170659/L1 | 170667/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 302 | 278003 | 177237/L1 | 177239/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 303 | 290914 | 183730/L1 | 183732/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 304 | 290994 | 105525/L1 | 105545/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 305 | 293151 | 106663/L1 | 106664/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 306 | 293161 | 106648/L1 | 106665/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 307 | 293250 | 117044/L1 | 118146/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 308 | 293259 | 117047/L1 | 118147/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 309 | 293270 | 117050/L1 | 118148/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 310 | 293299 | 117053/L1 | 118149/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 311 | 293314 | 118144/L1 | 118145/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 312 | 293339 | 117056/L1 | 118150/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 313 | 293369 | 117059/L1 | 118151/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 314 | 293383 | 118199/L1 | 118208/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 315 | 293402 | 118202/L1 | 118209/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 316 | 293449 | 118251/L1 | 118258/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 317 | 305136 | 129978/L1 | 129990/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 318 | 305158 | 135701/L1 | 136302/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 319 | 305180 | 136361/L1 | 136372/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 320 | 305207 | 136364/L1 | 136373/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 321 | 305246 | 136368/L1 | 136374/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 322 | 316955 | 129970/L1 | 129989/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 323 | 316978 | 136429/L1 | 136443/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 324 | 329886 | 142935/L1 | 142936/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 325 | 329893 | 142920/L1 | 142937/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 326 | 329976 | 144259/L1 | 144409/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 327 | 329985 | 144262/L1 | 144410/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 328 | 329993 | 144265/L1 | 144411/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 329 | 330022 | 144268/L1 | 144412/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 330 | 330037 | 144407/L1 | 144408/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_03/swiglu |
| 331 | 330061 | 144271/L1 | 144413/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 332 | 330087 | 144274/L1 | 144414/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 333 | 330101 | 144536/L1 | 144560/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 334 | 330120 | 144539/L1 | 144561/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 335 | 330168 | 144613/L1 | 144630/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 336 | 330521 | 144995/L1 | 145013/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 337 | 330612 | 183817/L1 | 183822/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 338 | 330730 | 183913/L1 | 183918/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 339 | 330749 | 183916/L1 | 183919/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 340 | 330796 | 184287/L1 | 184290/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 341 | 331140 | 185757/L1 | 185768/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 342 | 331192 | 186805/L1 | 186942/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 343 | 333347 | 185725/L1 | 185765/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 344 | 333388 | 186928/L1 | 186945/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 345 | 333738 | 144979/L1 | 145012/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 346 | 333778 | 145101/L1 | 145102/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 347 | 334121 | 185740/L1 | 185766/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 348 | 334159 | 186926/L1 | 186944/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 349 | 334242 | 187027/L1 | 187029/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 350 | 334282 | 187079/L1 | 187083/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_03/swiglu |
| 351 | 334357 | 187145/L1 | 187147/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_03/swiglu |
| 352 | 334369 | 185755/L1 | 185767/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 353 | 334425 | 186934/L1 | 186946/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 354 | 334439 | 187177/L1 | 187179/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_03/swiglu |
| 355 | 334529 | 198915/L1 | 199518/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 356 | 334614 | 198940/L1 | 199519/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 357 | 334704 | 198965/L1 | 199520/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 358 | 334794 | 198990/L1 | 199521/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 359 | 334884 | 199015/L1 | 199522/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 360 | 334974 | 199040/L1 | 199523/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 361 | 335064 | 199065/L1 | 199524/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 362 | 335154 | 199090/L1 | 199525/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 363 | 335244 | 199115/L1 | 199526/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 364 | 335334 | 199140/L1 | 199527/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 365 | 335424 | 199165/L1 | 199528/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 366 | 335514 | 199190/L1 | 199529/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 367 | 335604 | 199215/L1 | 199530/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 368 | 335694 | 199240/L1 | 199531/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 369 | 335784 | 199265/L1 | 199532/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 370 | 335874 | 199290/L1 | 199533/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 371 | 335964 | 199315/L1 | 199534/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 372 | 336054 | 199340/L1 | 199535/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 373 | 336144 | 199365/L1 | 199536/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 374 | 336234 | 199390/L1 | 199537/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 375 | 336324 | 199415/L1 | 199538/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 376 | 336414 | 199440/L1 | 199539/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 377 | 336504 | 199465/L1 | 199540/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 378 | 336594 | 199490/L1 | 199541/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 379 | 336685 | 199515/L1 | 199542/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 380 | 336734 | 200030/L1 | 200031/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 381 | 336743 | 200027/L1 | 200033/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 382 | 336826 | 200094/L1 | 200116/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 383 | 336835 | 200097/L1 | 200117/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 384 | 336843 | 200100/L1 | 200118/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 385 | 336869 | 200103/L1 | 200119/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 386 | 336884 | 200113/L1 | 200114/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 387 | 336905 | 200106/L1 | 200120/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 388 | 336931 | 200109/L1 | 200121/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 389 | 336945 | 200187/L1 | 200192/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 390 | 336964 | 200190/L1 | 200193/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 391 | 337011 | 206079/L1 | 206230/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 392 | 348710 | 218429/L1 | 218432/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 393 | 348731 | 218486/L1 | 218491/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 394 | 348755 | 218534/L1 | 218543/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 395 | 348786 | 218537/L1 | 218544/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 396 | 348828 | 218541/L1 | 218545/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 397 | 360543 | 218424/L1 | 218430/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 398 | 360566 | 218896/L1 | 218905/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 399 | 373476 | 225449/L1 | 225450/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 400 | 373558 | 145092/L1 | 145103/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 401 | 375715 | 146190/L1 | 146205/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_03/swiglu |
| 402 | 375806 | 147222/L1 | 147323/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 403 | 375865 | 147397/L1 | 147410/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 404 | 375881 | 147400/L1 | 147411/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 405 | 375911 | 147403/L1 | 147412/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 406 | 375942 | 147406/L1 | 147413/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 407 | 375993 | 147497/L1 | 147509/L5 | qwen24-stage/token_1/layer_02/swiglu |
| 408 | 387688 | 158671/L1 | 159228/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 409 | 387710 | 170662/L1 | 170668/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 410 | 387730 | 170696/L1 | 170707/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 411 | 387752 | 170699/L1 | 170708/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 412 | 387787 | 170703/L1 | 170709/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 413 | 387796 | 170665/L1 | 170666/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 414 | 399489 | 159213/L1 | 159229/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 415 | 399512 | 170750/L1 | 170761/L5 | qwen24-stage/token_0/layer_04/swiglu |
| 416 | 412419 | 177227/L1 | 177238/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 417 | 412501 | 183130/L1 | 183731/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 418 | 412555 | 183805/L1 | 183818/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 419 | 412571 | 183808/L1 | 183819/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 420 | 412600 | 183811/L1 | 183820/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 421 | 412630 | 183814/L1 | 183821/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 422 | 412681 | 183905/L1 | 183917/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 423 | 413032 | 184273/L1 | 184289/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 424 | 413040 | 225442/L1 | 225452/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 425 | 413129 | 226570/L1 | 226589/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 426 | 413138 | 226573/L1 | 226590/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 427 | 413149 | 226576/L1 | 226591/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 428 | 413179 | 226579/L1 | 226592/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 429 | 413194 | 226586/L1 | 226587/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 430 | 413218 | 226582/L1 | 226593/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 431 | 413247 | 226585/L1 | 226594/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 432 | 413261 | 226675/L1 | 226680/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 433 | 413280 | 226678/L1 | 226681/L5 | qwen24-stage/token_0/layer_04/swiglu ; qwen24-stage/token_1/layer_04/swiglu |
| 434 | 413326 | 226749/L1 | 226754/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 435 | 413696 | 228245/L1 | 228263/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 436 | 413724 | 228254/L1 | 228264/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 437 | 415895 | 228221/L1 | 228259/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 438 | 415916 | 228218/L1 | 228258/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 439 | 415936 | 238916/L1 | 240033/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 440 | 416285 | 184257/L1 | 184288/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 441 | 416323 | 185762/L1 | 185763/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 442 | 416676 | 228233/L1 | 228261/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 443 | 416697 | 228230/L1 | 228260/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 444 | 416732 | 240023/L1 | 240038/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 445 | 416760 | 240032/L1 | 240039/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 446 | 416834 | 240087/L1 | 240091/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 447 | 416875 | 240132/L1 | 240144/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 448 | 416884 | 240135/L1 | 240145/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 449 | 416905 | 240139/L1 | 240146/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 450 | 416931 | 228237/L1 | 228262/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 451 | 416985 | 238922/L1 | 240034/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 452 | 417005 | 240195/L1 | 240197/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_04/swiglu |
| 453 | 417118 | 247302/L1 | 247640/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 454 | 417206 | 247316/L1 | 247641/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 455 | 417296 | 247330/L1 | 247642/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 456 | 417386 | 247344/L1 | 247643/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 457 | 417476 | 247358/L1 | 247644/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 458 | 417566 | 247372/L1 | 247645/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 459 | 417656 | 247386/L1 | 247646/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 460 | 417746 | 247400/L1 | 247647/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 461 | 417836 | 247414/L1 | 247648/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 462 | 417926 | 247428/L1 | 247649/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 463 | 418016 | 247442/L1 | 247650/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 464 | 418106 | 247456/L1 | 247651/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 465 | 418196 | 247470/L1 | 247652/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 466 | 418286 | 247484/L1 | 247653/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 467 | 418376 | 247498/L1 | 247654/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 468 | 418466 | 247512/L1 | 247655/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 469 | 418556 | 247526/L1 | 247656/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 470 | 418646 | 247540/L1 | 247657/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 471 | 418736 | 247554/L1 | 247658/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 472 | 418826 | 247568/L1 | 247659/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 473 | 418916 | 247582/L1 | 247660/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 474 | 419006 | 247596/L1 | 247661/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 475 | 419096 | 247610/L1 | 247662/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 476 | 419186 | 247624/L1 | 247663/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 477 | 419278 | 247638/L1 | 247664/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 478 | 419372 | 247864/L1 | 247866/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 479 | 419413 | 247938/L1 | 247954/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 480 | 419422 | 247941/L1 | 247955/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 481 | 419430 | 247915/L1 | 247947/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 482 | 419449 | 247944/L1 | 247956/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 483 | 419463 | 247918/L1 | 247948/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 484 | 419491 | 247921/L1 | 247949/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 485 | 419507 | 247945/L1 | 247946/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 486 | 419535 | 248014/L1 | 248016/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 487 | 419576 | 248250/L1 | 248259/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 488 | 431276 | 259316/L1 | 259865/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 489 | 431291 | 259899/L1 | 259908/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 490 | 431311 | 260995/L1 | 261006/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 491 | 431336 | 261069/L1 | 261071/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 492 | 431345 | 260998/L1 | 261007/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 493 | 431373 | 261001/L1 | 261008/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 494 | 431390 | 261004/L1 | 261009/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 495 | 443097 | 259858/L1 | 259866/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 496 | 443120 | 261119/L1 | 261123/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 497 | 456024 | 267606/L1 | 267608/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 498 | 456108 | 185625/L1 | 185764/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 499 | 458265 | 186922/L1 | 186943/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_03/swiglu |
| 500 | 458347 | 187011/L1 | 187028/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 501 | 458377 | 187069/L1 | 187080/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 502 | 458386 | 187072/L1 | 187081/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 503 | 458415 | 187075/L1 | 187082/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 504 | 458493 | 187141/L1 | 187146/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 505 | 458533 | 187171/L1 | 187178/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 506 | 470221 | 198348/L1 | 199516/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 507 | 470240 | 200025/L1 | 200032/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 508 | 470263 | 200091/L1 | 200115/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 509 | 470285 | 200079/L1 | 200110/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 510 | 470312 | 200082/L1 | 200111/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 511 | 470330 | 200085/L1 | 200112/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_03/swiglu |
| 512 | 470342 | 200028/L1 | 200029/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 513 | 470349 | 200177/L1 | 200191/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 514 | 482045 | 198890/L1 | 199517/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 515 | 482146 | 204872/L1 | 206080/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 516 | 482230 | 204880/L1 | 206081/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 517 | 482318 | 204888/L1 | 206082/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 518 | 482406 | 204896/L1 | 206083/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 519 | 482494 | 204904/L1 | 206084/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 520 | 482582 | 204912/L1 | 206085/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 521 | 482670 | 204920/L1 | 206086/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 522 | 482758 | 204928/L1 | 206087/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 523 | 482846 | 204936/L1 | 206088/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 524 | 482934 | 204944/L1 | 206089/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 525 | 483022 | 204952/L1 | 206090/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 526 | 483110 | 204960/L1 | 206091/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 527 | 483198 | 204968/L1 | 206092/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 528 | 483286 | 204976/L1 | 206093/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 529 | 483374 | 204984/L1 | 206094/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 530 | 483462 | 204992/L1 | 206095/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 531 | 483550 | 205000/L1 | 206096/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 532 | 483638 | 205008/L1 | 206097/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 533 | 483726 | 205016/L1 | 206098/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 534 | 483814 | 205024/L1 | 206099/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 535 | 483902 | 205032/L1 | 206100/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 536 | 483990 | 205040/L1 | 206101/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 537 | 484078 | 205048/L1 | 206102/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 538 | 484166 | 205056/L1 | 206103/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 539 | 484254 | 205064/L1 | 206104/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 540 | 484342 | 205072/L1 | 206105/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 541 | 484430 | 205080/L1 | 206106/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 542 | 484518 | 205088/L1 | 206107/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 543 | 484606 | 205096/L1 | 206108/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 544 | 484694 | 205104/L1 | 206109/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 545 | 484782 | 205112/L1 | 206110/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 546 | 484870 | 205120/L1 | 206111/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 547 | 484958 | 205128/L1 | 206112/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 548 | 485046 | 205136/L1 | 206113/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 549 | 485134 | 205144/L1 | 206114/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 550 | 485222 | 205152/L1 | 206115/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 551 | 485310 | 205160/L1 | 206116/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 552 | 485398 | 205168/L1 | 206117/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 553 | 485486 | 205176/L1 | 206118/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 554 | 485574 | 205184/L1 | 206119/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 555 | 485662 | 205192/L1 | 206120/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 556 | 485750 | 205200/L1 | 206121/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 557 | 485838 | 205208/L1 | 206122/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 558 | 485926 | 205216/L1 | 206123/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 559 | 486014 | 205224/L1 | 206124/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 560 | 486102 | 205232/L1 | 206125/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 561 | 486190 | 205240/L1 | 206126/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 562 | 486278 | 205248/L1 | 206127/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 563 | 486366 | 205256/L1 | 206128/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 564 | 486454 | 205264/L1 | 206129/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 565 | 486542 | 205272/L1 | 206130/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 566 | 486630 | 205280/L1 | 206131/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 567 | 486718 | 205288/L1 | 206132/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 568 | 486806 | 205296/L1 | 206133/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 569 | 486894 | 205304/L1 | 206134/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 570 | 486982 | 205312/L1 | 206135/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 571 | 487070 | 205320/L1 | 206136/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 572 | 487158 | 205328/L1 | 206137/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 573 | 487246 | 205336/L1 | 206138/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 574 | 487334 | 205344/L1 | 206139/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 575 | 487422 | 205352/L1 | 206140/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 576 | 487510 | 205360/L1 | 206141/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 577 | 487598 | 205368/L1 | 206142/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 578 | 487686 | 205376/L1 | 206143/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 579 | 487774 | 205384/L1 | 206144/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 580 | 487862 | 205392/L1 | 206145/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 581 | 487950 | 205400/L1 | 206146/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 582 | 488038 | 205408/L1 | 206147/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 583 | 488126 | 205416/L1 | 206148/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 584 | 488214 | 205424/L1 | 206149/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 585 | 488302 | 205432/L1 | 206150/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 586 | 488390 | 205440/L1 | 206151/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 587 | 488478 | 205448/L1 | 206152/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 588 | 488566 | 205456/L1 | 206153/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 589 | 488654 | 205464/L1 | 206154/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 590 | 488742 | 205472/L1 | 206155/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 591 | 488830 | 205480/L1 | 206156/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 592 | 488918 | 205488/L1 | 206157/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 593 | 489006 | 205496/L1 | 206158/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 594 | 489094 | 205504/L1 | 206159/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 595 | 489182 | 205512/L1 | 206160/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 596 | 489270 | 205520/L1 | 206161/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 597 | 489358 | 205528/L1 | 206162/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 598 | 489446 | 205536/L1 | 206163/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 599 | 489534 | 205544/L1 | 206164/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 600 | 489622 | 205552/L1 | 206165/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 601 | 489710 | 205560/L1 | 206166/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 602 | 489798 | 205568/L1 | 206167/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 603 | 489886 | 205576/L1 | 206168/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 604 | 489974 | 205584/L1 | 206169/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 605 | 490062 | 205592/L1 | 206170/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 606 | 490150 | 205600/L1 | 206171/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 607 | 490238 | 205608/L1 | 206172/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 608 | 490326 | 205616/L1 | 206173/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 609 | 490414 | 205624/L1 | 206174/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 610 | 490502 | 205632/L1 | 206175/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 611 | 490590 | 205640/L1 | 206176/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 612 | 490678 | 205648/L1 | 206177/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 613 | 490766 | 205656/L1 | 206178/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 614 | 490854 | 205664/L1 | 206179/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 615 | 490942 | 205672/L1 | 206180/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 616 | 491030 | 205680/L1 | 206181/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 617 | 491118 | 205688/L1 | 206182/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 618 | 491206 | 205696/L1 | 206183/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 619 | 491294 | 205704/L1 | 206184/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 620 | 491382 | 205712/L1 | 206185/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 621 | 491470 | 205720/L1 | 206186/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 622 | 491558 | 205728/L1 | 206187/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 623 | 491646 | 205736/L1 | 206188/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 624 | 491734 | 205744/L1 | 206189/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 625 | 491822 | 205752/L1 | 206190/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 626 | 491910 | 205760/L1 | 206191/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 627 | 491998 | 205768/L1 | 206192/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 628 | 492086 | 205776/L1 | 206193/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 629 | 492174 | 205784/L1 | 206194/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 630 | 492262 | 205792/L1 | 206195/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 631 | 492350 | 205800/L1 | 206196/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 632 | 492438 | 205808/L1 | 206197/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 633 | 492526 | 205816/L1 | 206198/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 634 | 492614 | 205824/L1 | 206199/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 635 | 492702 | 205832/L1 | 206200/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 636 | 492790 | 205840/L1 | 206201/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 637 | 492878 | 205848/L1 | 206202/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 638 | 492966 | 205856/L1 | 206203/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 639 | 493054 | 205864/L1 | 206204/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 640 | 493142 | 205872/L1 | 206205/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 641 | 493230 | 205880/L1 | 206206/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 642 | 493318 | 205888/L1 | 206207/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 643 | 493406 | 205896/L1 | 206208/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 644 | 493494 | 205904/L1 | 206209/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 645 | 493582 | 205912/L1 | 206210/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 646 | 493670 | 205920/L1 | 206211/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 647 | 493758 | 205928/L1 | 206212/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 648 | 493846 | 205936/L1 | 206213/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 649 | 493934 | 205944/L1 | 206214/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 650 | 494022 | 205952/L1 | 206215/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 651 | 494110 | 205960/L1 | 206216/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 652 | 494198 | 205968/L1 | 206217/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 653 | 494286 | 205976/L1 | 206218/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 654 | 494374 | 205984/L1 | 206219/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 655 | 494462 | 205992/L1 | 206220/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 656 | 494550 | 206000/L1 | 206221/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 657 | 494638 | 206008/L1 | 206222/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 658 | 494726 | 206016/L1 | 206223/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 659 | 494814 | 206024/L1 | 206224/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 660 | 494902 | 206032/L1 | 206225/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 661 | 494990 | 206040/L1 | 206226/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 662 | 495078 | 206048/L1 | 206227/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 663 | 495166 | 206056/L1 | 206228/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 664 | 495256 | 206065/L1 | 206229/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 665 | 495350 | 218427/L1 | 218431/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 666 | 495401 | 218474/L1 | 218487/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 667 | 495417 | 218477/L1 | 218488/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 668 | 495446 | 218480/L1 | 218489/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 669 | 495477 | 218483/L1 | 218490/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 670 | 495503 | 218531/L1 | 218542/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 671 | 495616 | 218844/L1 | 218901/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 672 | 495702 | 218858/L1 | 218902/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 673 | 495790 | 218872/L1 | 218903/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 674 | 495879 | 218886/L1 | 218904/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 675 | 495974 | 267665/L1 | 267667/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 676 | 496015 | 279387/L1 | 279401/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 677 | 496024 | 279390/L1 | 279402/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 678 | 496035 | 278293/L1 | 279396/L5 | qwen24-stage/token_0/layer_05/swiglu ; qwen24-stage/token_1/layer_05/swiglu |
| 679 | 496054 | 279393/L1 | 279403/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 680 | 496071 | 278296/L1 | 279397/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 681 | 496103 | 278299/L1 | 279398/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 682 | 496119 | 279394/L1 | 279395/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 683 | 496145 | 279447/L1 | 279449/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 684 | 496187 | 279484/L1 | 279491/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 685 | 496541 | 280933/L1 | 280938/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 686 | 496594 | 281019/L1 | 281040/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 687 | 498749 | 280899/L1 | 280935/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 688 | 498785 | 281009/L1 | 281038/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 689 | 498872 | 218787/L1 | 218897/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 690 | 498955 | 218801/L1 | 218898/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 691 | 499043 | 218815/L1 | 218899/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 692 | 499133 | 218830/L1 | 218900/L5 | qwen24-stage/token_0/layer_05/swiglu |
| 693 | 499180 | 225447/L1 | 225448/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 694 | 499530 | 280915/L1 | 280936/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 695 | 499566 | 281013/L1 | 281039/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 696 | 499622 | 287538/L1 | 287540/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 697 | 499688 | 287614/L1 | 287615/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 698 | 499696 | 287613/L1 | 287617/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 699 | 499772 | 287692/L1 | 287697/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 700 | 499782 | 280931/L1 | 280937/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 701 | 499808 | 281027/L1 | 281041/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 702 | 499835 | 281036/L1 | 281042/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 703 | 499890 | 287757/L1 | 287759/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 704 | 502045 | 289155/L1 | 289158/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 705 | 502129 | 289253/L1 | 289261/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 706 | 502180 | 290374/L1 | 290385/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 707 | 502196 | 290377/L1 | 290386/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 708 | 502225 | 290380/L1 | 290387/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_07/swiglu |
| 709 | 502256 | 290383/L1 | 290388/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 710 | 502310 | 290481/L1 | 290490/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 711 | 514005 | 301413/L1 | 301958/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 712 | 514022 | 301991/L1 | 301995/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_07/swiglu |
| 713 | 514043 | 313708/L1 | 313714/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_06/swiglu |
| 714 | 514063 | 312615/L1 | 313709/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 715 | 514092 | 312618/L1 | 313710/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 716 | 514113 | 312621/L1 | 313711/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 717 | 514125 | 301992/L1 | 301993/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 718 | 525825 | 301955/L1 | 301959/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_07/swiglu |
| 719 | 525848 | 313758/L1 | 313762/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 720 | 538758 | 320091/L1 | 320095/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 721 | 538842 | 225440/L1 | 225451/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 722 | 540988 | 226567/L1 | 226588/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 723 | 541072 | 226665/L1 | 226679/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 724 | 541124 | 226726/L1 | 226750/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 725 | 541140 | 226729/L1 | 226751/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 726 | 541167 | 226732/L1 | 226752/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 727 | 541196 | 226735/L1 | 226753/L5 | qwen24-stage/token_0/layer_04/swiglu ; qwen24-stage/token_1/layer_04/swiglu |
| 728 | 541249 | 228209/L1 | 228257/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 729 | 552943 | 239473/L1 | 240036/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 730 | 552960 | 240072/L1 | 240090/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 731 | 552981 | 240128/L1 | 240143/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 732 | 553003 | 240106/L1 | 240140/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 733 | 553033 | 240109/L1 | 240141/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_04/swiglu |
| 734 | 553054 | 240112/L1 | 240142/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 735 | 553066 | 240088/L1 | 240089/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 736 | 564766 | 240015/L1 | 240037/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_04/swiglu |
| 737 | 564789 | 240193/L1 | 240196/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 738 | 577698 | 247288/L1 | 247639/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 739 | 577780 | 247830/L1 | 247865/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 740 | 577831 | 247925/L1 | 247950/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 741 | 577847 | 247928/L1 | 247951/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 742 | 577876 | 247931/L1 | 247952/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 743 | 577907 | 247934/L1 | 247953/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 744 | 577933 | 248011/L1 | 248015/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 745 | 578017 | 248161/L1 | 248255/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 746 | 578106 | 248186/L1 | 248256/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 747 | 578196 | 248211/L1 | 248257/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 748 | 578286 | 248236/L1 | 248258/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 749 | 578338 | 259905/L1 | 259906/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 750 | 578422 | 320157/L1 | 320159/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 751 | 578473 | 326646/L1 | 326657/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 752 | 578489 | 326649/L1 | 326658/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 753 | 578514 | 326652/L1 | 326659/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 754 | 578541 | 326655/L1 | 326660/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 755 | 578593 | 326753/L1 | 326762/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 756 | 578945 | 328228/L1 | 328234/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 757 | 578998 | 328303/L1 | 328355/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 758 | 581152 | 328194/L1 | 328231/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 759 | 581195 | 328341/L1 | 328358/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 760 | 581248 | 248061/L1 | 248251/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 761 | 581333 | 248086/L1 | 248252/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 762 | 581423 | 248111/L1 | 248253/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 763 | 581514 | 248136/L1 | 248254/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 764 | 581563 | 258757/L1 | 259862/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 765 | 581585 | 258755/L1 | 259861/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 766 | 581945 | 328210/L1 | 328232/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 767 | 581988 | 328339/L1 | 328357/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 768 | 582069 | 328745/L1 | 328748/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 769 | 582118 | 328827/L1 | 328842/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 770 | 582127 | 328824/L1 | 328841/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 771 | 582138 | 328830/L1 | 328843/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 772 | 582169 | 328833/L1 | 328844/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 773 | 582200 | 329944/L1 | 329946/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 774 | 582206 | 328226/L1 | 328233/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 775 | 582262 | 328347/L1 | 328359/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 776 | 582310 | 330009/L1 | 330011/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 777 | 584468 | 331107/L1 | 331112/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 778 | 584563 | 331190/L1 | 331209/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 779 | 584572 | 331193/L1 | 331210/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 780 | 584584 | 331196/L1 | 331211/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 781 | 584614 | 331199/L1 | 331212/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 782 | 584629 | 331206/L1 | 331207/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 783 | 584654 | 331202/L1 | 331213/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 784 | 584684 | 331205/L1 | 331214/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 785 | 584698 | 342948/L1 | 342954/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.o_weight.weight |
| 786 | 584717 | 342951/L1 | 342955/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.o_weight.weight |
| 787 | 584765 | 342986/L1 | 342990/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.o_weight.weight |
| 788 | 596460 | 353889/L1 | 354436/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 789 | 596486 | 354481/L1 | 354485/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_09/swiglu |
| 790 | 596509 | 360957/L1 | 360966/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 791 | 596536 | 360960/L1 | 360967/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 792 | 596578 | 360964/L1 | 360968/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 793 | 596587 | 354482/L1 | 354483/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 794 | 608287 | 354431/L1 | 354437/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 795 | 608310 | 361030/L1 | 361032/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 796 | 621221 | 367378/L1 | 367383/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 797 | 621243 | 258765/L1 | 259863/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 798 | 621268 | 258774/L1 | 259864/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 799 | 621309 | 259897/L1 | 259907/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 800 | 623449 | 260992/L1 | 261005/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 801 | 623537 | 261066/L1 | 261070/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 802 | 623569 | 261103/L1 | 261120/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 803 | 623578 | 261106/L1 | 261121/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 804 | 623605 | 261109/L1 | 261122/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_06/swiglu |
| 805 | 623680 | 267006/L1 | 267607/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 806 | 623720 | 267631/L1 | 267666/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 807 | 635415 | 278841/L1 | 279399/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 808 | 635434 | 279444/L1 | 279448/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 809 | 635449 | 279460/L1 | 279487/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 810 | 635471 | 280799/L1 | 280934/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 811 | 635482 | 279463/L1 | 279488/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 812 | 635510 | 279466/L1 | 279489/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 813 | 635531 | 279469/L1 | 279490/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 814 | 635543 | 279485/L1 | 279486/L5 | qwen24-stage/token_0/layer_05/swiglu ; qwen24-stage/token_1/layer_05/swiglu |
| 815 | 647244 | 279383/L1 | 279400/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 816 | 647267 | 281005/L1 | 281037/L5 | qwen24-stage/token_1/layer_05/swiglu |
| 817 | 660172 | 287532/L1 | 287539/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 818 | 660254 | 287610/L1 | 287616/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 819 | 660305 | 287679/L1 | 287693/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 820 | 660321 | 287682/L1 | 287694/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 821 | 660351 | 287685/L1 | 287695/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 822 | 660382 | 287688/L1 | 287696/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 823 | 660436 | 287743/L1 | 287758/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 824 | 660788 | 289055/L1 | 289157/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 825 | 660883 | 367461/L1 | 367480/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_07/swiglu |
| 826 | 660892 | 367464/L1 | 367481/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 827 | 660903 | 367467/L1 | 367482/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 828 | 660933 | 367470/L1 | 367483/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 829 | 660948 | 367477/L1 | 367478/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 830 | 660973 | 367473/L1 | 367484/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_07/swiglu |
| 831 | 661003 | 367476/L1 | 367485/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 832 | 661017 | 367871/L1 | 367877/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 833 | 661036 | 367874/L1 | 367878/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_07/swiglu |
| 834 | 661083 | 367956/L1 | 367964/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 835 | 661427 | 370453/L1 | 370458/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 836 | 661479 | 370555/L1 | 370595/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 837 | 663632 | 370421/L1 | 370455/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 838 | 663672 | 370581/L1 | 370598/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 839 | 664020 | 289039/L1 | 289156/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 840 | 664060 | 289258/L1 | 289259/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 841 | 664403 | 370436/L1 | 370456/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 842 | 664441 | 370579/L1 | 370597/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 843 | 664523 | 370685/L1 | 370690/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 844 | 664593 | 370760/L1 | 370768/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 845 | 664602 | 370763/L1 | 370769/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 846 | 664624 | 370766/L1 | 370770/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 847 | 664646 | 370451/L1 | 370457/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 848 | 664702 | 370587/L1 | 370599/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 849 | 664716 | 382480/L1 | 382483/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 850 | 664755 | 382505/L1 | 382507/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 851 | 666911 | 383577/L1 | 383582/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 852 | 666995 | 383642/L1 | 383644/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.k_weight.weight |
| 853 | 667025 | 383686/L1 | 383694/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.v_weight.weight |
| 854 | 667034 | 383689/L1 | 383695/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.v_weight.weight |
| 855 | 667059 | 383692/L1 | 383696/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.v_weight.weight |
| 856 | 667129 | 390179/L1 | 390181/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 857 | 667170 | 390237/L1 | 390239/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 858 | 678869 | 401151/L1 | 401702/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 859 | 678890 | 401747/L1 | 401749/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 860 | 678908 | 401775/L1 | 401788/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 861 | 678936 | 402156/L1 | 402159/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 862 | 678949 | 401778/L1 | 401789/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 863 | 678978 | 401781/L1 | 401790/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 864 | 678999 | 401784/L1 | 401791/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 865 | 679011 | 401785/L1 | 401786/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 866 | 690711 | 401693/L1 | 401703/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 867 | 690734 | 402218/L1 | 402221/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 868 | 703644 | 409517/L1 | 409543/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 869 | 703724 | 289250/L1 | 289260/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 870 | 705868 | 290370/L1 | 290384/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 871 | 705959 | 290452/L1 | 290484/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 872 | 705968 | 290455/L1 | 290485/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_07/swiglu |
| 873 | 705978 | 290458/L1 | 290486/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 874 | 706006 | 290461/L1 | 290487/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 875 | 706021 | 290482/L1 | 290483/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 876 | 706044 | 290464/L1 | 290488/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 877 | 706072 | 290467/L1 | 290489/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 878 | 706086 | 300868/L1 | 301956/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 879 | 706105 | 300871/L1 | 301957/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 880 | 706153 | 301988/L1 | 301994/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 881 | 717843 | 313163/L1 | 313712/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 882 | 717868 | 313748/L1 | 313761/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 883 | 717890 | 319484/L1 | 320092/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 884 | 717919 | 319487/L1 | 320093/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 885 | 717960 | 319491/L1 | 320094/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 886 | 717969 | 313759/L1 | 313760/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 887 | 729670 | 313705/L1 | 313713/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 888 | 729693 | 320154/L1 | 320158/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 889 | 742596 | 326642/L1 | 326656/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 890 | 742681 | 326723/L1 | 326756/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 891 | 742690 | 326726/L1 | 326757/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 892 | 742698 | 326729/L1 | 326758/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 893 | 742724 | 326732/L1 | 326759/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_08/swiglu |
| 894 | 742739 | 326754/L1 | 326755/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 895 | 742760 | 326735/L1 | 326760/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 896 | 742786 | 326738/L1 | 326761/L5 | qwen24-stage/token_0/layer_08/swiglu |
| 897 | 742800 | 328091/L1 | 328229/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 898 | 742819 | 328094/L1 | 328230/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 899 | 742867 | 328335/L1 | 328356/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 900 | 743217 | 328729/L1 | 328747/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 901 | 743308 | 409743/L1 | 409745/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 902 | 743367 | 409814/L1 | 409832/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 903 | 743383 | 409817/L1 | 409833/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 904 | 743414 | 409820/L1 | 409834/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 905 | 743445 | 409823/L1 | 409835/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 906 | 743471 | 409893/L1 | 409895/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_08/swiglu |
| 907 | 743555 | 410680/L1 | 410786/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 908 | 743644 | 410705/L1 | 410787/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 909 | 743734 | 410730/L1 | 410788/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 910 | 743824 | 410755/L1 | 410789/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 911 | 743891 | 423140/L1 | 423159/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.q_weight.weight ; qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 912 | 743919 | 423149/L1 | 423160/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 913 | 743981 | 409955/L1 | 410757/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 914 | 744066 | 409980/L1 | 410758/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 915 | 744156 | 410005/L1 | 410759/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 916 | 744246 | 410030/L1 | 410760/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 917 | 744336 | 410055/L1 | 410761/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 918 | 744426 | 410080/L1 | 410762/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 919 | 744516 | 410105/L1 | 410763/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_08/swiglu |
| 920 | 744606 | 410130/L1 | 410764/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 921 | 744696 | 410155/L1 | 410765/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 922 | 744786 | 410180/L1 | 410766/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 923 | 744876 | 410205/L1 | 410767/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 924 | 744966 | 410230/L1 | 410768/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 925 | 745056 | 410255/L1 | 410769/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 926 | 745146 | 410280/L1 | 410770/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 927 | 745236 | 410305/L1 | 410771/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 928 | 745326 | 410330/L1 | 410772/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 929 | 745416 | 410355/L1 | 410773/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 930 | 745506 | 410380/L1 | 410774/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 931 | 745596 | 410405/L1 | 410775/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 932 | 745686 | 410430/L1 | 410776/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 933 | 745776 | 410455/L1 | 410777/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_08/swiglu |
| 934 | 745866 | 410480/L1 | 410778/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 935 | 745956 | 410505/L1 | 410779/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 936 | 746046 | 410530/L1 | 410780/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 937 | 746136 | 410555/L1 | 410781/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 938 | 746189 | 423119/L1 | 423155/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 939 | 746212 | 423117/L1 | 423154/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 940 | 746230 | 423207/L1 | 423254/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 941 | 746579 | 328713/L1 | 328746/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 942 | 746618 | 328838/L1 | 328839/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 943 | 746671 | 410580/L1 | 410782/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 944 | 746756 | 410605/L1 | 410783/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 945 | 746846 | 410630/L1 | 410784/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 946 | 746937 | 410655/L1 | 410785/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 947 | 746986 | 423128/L1 | 423157/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 948 | 747008 | 423126/L1 | 423156/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 949 | 747075 | 423251/L1 | 423258/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 950 | 747141 | 423321/L1 | 423322/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 951 | 747149 | 423320/L1 | 423326/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 952 | 747224 | 423378/L1 | 423380/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 953 | 747240 | 423132/L1 | 423158/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 954 | 747290 | 423213/L1 | 423255/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.q_weight.weight ; qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 955 | 747338 | 429856/L1 | 429860/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 956 | 749494 | 430967/L1 | 430976/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 957 | 749576 | 431060/L1 | 431063/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 958 | 749627 | 431123/L1 | 431134/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 959 | 749643 | 431126/L1 | 431135/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 960 | 749673 | 431129/L1 | 431136/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 961 | 749704 | 431132/L1 | 431137/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 962 | 749755 | 431540/L1 | 431549/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 963 | 761451 | 443086/L1 | 443655/L5 | qwen24-stage/token_0/layer_00/model_list.0.pre_Normal.weight ; qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 964 | 761472 | 444141/L1 | 444147/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 965 | 761493 | 444223/L1 | 444235/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 966 | 761515 | 444196/L1 | 444226/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 967 | 761546 | 444199/L1 | 444227/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 968 | 761563 | 444202/L1 | 444228/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 969 | 761575 | 444144/L1 | 444145/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 970 | 761582 | 444304/L1 | 444307/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 971 | 773278 | 443628/L1 | 443656/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 972 | 773380 | 449000/L1 | 450194/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 973 | 773467 | 449008/L1 | 450195/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 974 | 773555 | 449016/L1 | 450196/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 975 | 773643 | 449024/L1 | 450197/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 976 | 773731 | 449032/L1 | 450198/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 977 | 773819 | 449040/L1 | 450199/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 978 | 773907 | 449048/L1 | 450200/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 979 | 773995 | 449056/L1 | 450201/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 980 | 774083 | 449064/L1 | 450202/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 981 | 774171 | 449072/L1 | 450203/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 982 | 774259 | 449080/L1 | 450204/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 983 | 774347 | 449088/L1 | 450205/L5 | qwen24-stage/token_0/layer_09/swiglu ; qwen24-stage/token_1/layer_09/swiglu |
| 984 | 774435 | 449096/L1 | 450206/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 985 | 774523 | 449104/L1 | 450207/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 986 | 774611 | 449112/L1 | 450208/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 987 | 774699 | 449120/L1 | 450209/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 988 | 774787 | 449128/L1 | 450210/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 989 | 774875 | 449136/L1 | 450211/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 990 | 774963 | 449144/L1 | 450212/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 991 | 775051 | 449152/L1 | 450213/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 992 | 775139 | 449160/L1 | 450214/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 993 | 775227 | 449168/L1 | 450215/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 994 | 775315 | 449176/L1 | 450216/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 995 | 775403 | 449184/L1 | 450217/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 996 | 775491 | 449192/L1 | 450218/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 997 | 775579 | 449200/L1 | 450219/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 998 | 775667 | 449208/L1 | 450220/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 999 | 775755 | 449216/L1 | 450221/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1000 | 775843 | 449224/L1 | 450222/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1001 | 775931 | 449232/L1 | 450223/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1002 | 776019 | 449240/L1 | 450224/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1003 | 776107 | 449248/L1 | 450225/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1004 | 776195 | 449256/L1 | 450226/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1005 | 776283 | 449264/L1 | 450227/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1006 | 776371 | 449272/L1 | 450228/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1007 | 776459 | 449280/L1 | 450229/L5 | qwen24-stage/token_0/layer_09/swiglu ; qwen24-stage/token_1/layer_09/swiglu |
| 1008 | 776547 | 449288/L1 | 450230/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1009 | 776635 | 449296/L1 | 450231/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1010 | 776723 | 449304/L1 | 450232/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1011 | 776811 | 449312/L1 | 450233/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1012 | 776899 | 449320/L1 | 450234/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1013 | 776987 | 449328/L1 | 450235/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1014 | 777075 | 449336/L1 | 450236/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1015 | 777163 | 449344/L1 | 450237/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1016 | 777251 | 449352/L1 | 450238/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1017 | 777339 | 449360/L1 | 450239/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1018 | 777427 | 449368/L1 | 450240/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1019 | 777515 | 449376/L1 | 450241/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1020 | 777603 | 449384/L1 | 450242/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1021 | 777691 | 449392/L1 | 450243/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1022 | 777779 | 449400/L1 | 450244/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1023 | 777867 | 449408/L1 | 450245/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1024 | 777955 | 449416/L1 | 450246/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1025 | 778043 | 449424/L1 | 450247/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1026 | 778131 | 449432/L1 | 450248/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1027 | 778219 | 449440/L1 | 450249/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1028 | 778307 | 449448/L1 | 450250/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1029 | 778395 | 449456/L1 | 450251/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1030 | 778483 | 449464/L1 | 450252/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1031 | 778571 | 449472/L1 | 450253/L5 | qwen24-stage/token_0/layer_09/swiglu ; qwen24-stage/token_1/layer_09/swiglu |
| 1032 | 778659 | 449480/L1 | 450254/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1033 | 778747 | 449488/L1 | 450255/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1034 | 778835 | 449496/L1 | 450256/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1035 | 778923 | 449504/L1 | 450257/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1036 | 779011 | 449512/L1 | 450258/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1037 | 779099 | 449520/L1 | 450259/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1038 | 779187 | 449528/L1 | 450260/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1039 | 779275 | 449536/L1 | 450261/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1040 | 779363 | 449544/L1 | 450262/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1041 | 779451 | 449552/L1 | 450263/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1042 | 779539 | 449560/L1 | 450264/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1043 | 779627 | 449568/L1 | 450265/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1044 | 779715 | 449576/L1 | 450266/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1045 | 779803 | 449584/L1 | 450267/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1046 | 779891 | 449592/L1 | 450268/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1047 | 779979 | 449600/L1 | 450269/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1048 | 780067 | 449608/L1 | 450270/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1049 | 780155 | 449616/L1 | 450271/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1050 | 780243 | 449624/L1 | 450272/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1051 | 780331 | 449632/L1 | 450273/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1052 | 780419 | 449640/L1 | 450274/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1053 | 780507 | 449648/L1 | 450275/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1054 | 780595 | 449656/L1 | 450276/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1055 | 780683 | 449664/L1 | 450277/L5 | qwen24-stage/token_0/layer_09/swiglu ; qwen24-stage/token_1/layer_09/swiglu |
| 1056 | 780771 | 449672/L1 | 450278/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1057 | 780859 | 449680/L1 | 450279/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1058 | 780947 | 449688/L1 | 450280/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1059 | 781035 | 449696/L1 | 450281/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1060 | 781123 | 449704/L1 | 450282/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1061 | 781211 | 449712/L1 | 450283/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1062 | 781299 | 449720/L1 | 450284/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1063 | 781387 | 449728/L1 | 450285/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1064 | 781475 | 449736/L1 | 450286/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1065 | 781563 | 449744/L1 | 450287/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1066 | 781651 | 449752/L1 | 450288/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1067 | 781739 | 449760/L1 | 450289/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1068 | 781827 | 449768/L1 | 450290/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1069 | 781915 | 449776/L1 | 450291/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1070 | 782003 | 449784/L1 | 450292/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1071 | 782091 | 449792/L1 | 450293/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1072 | 782179 | 449800/L1 | 450294/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1073 | 782267 | 449808/L1 | 450295/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1074 | 782355 | 449816/L1 | 450296/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1075 | 782443 | 449824/L1 | 450297/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1076 | 782531 | 449832/L1 | 450298/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1077 | 782619 | 449840/L1 | 450299/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1078 | 782707 | 449848/L1 | 450300/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1079 | 782795 | 449856/L1 | 450301/L5 | qwen24-stage/token_0/layer_09/swiglu ; qwen24-stage/token_1/layer_09/swiglu |
| 1080 | 782883 | 449864/L1 | 450302/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1081 | 782971 | 449872/L1 | 450303/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1082 | 783059 | 449880/L1 | 450304/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1083 | 783147 | 449888/L1 | 450305/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1084 | 783235 | 449896/L1 | 450306/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1085 | 783323 | 449904/L1 | 450307/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1086 | 783411 | 449912/L1 | 450308/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1087 | 783499 | 449920/L1 | 450309/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1088 | 783587 | 449928/L1 | 450310/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1089 | 783675 | 449936/L1 | 450311/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1090 | 783763 | 449944/L1 | 450312/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1091 | 783851 | 449952/L1 | 450313/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1092 | 783939 | 449960/L1 | 450314/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1093 | 784027 | 449968/L1 | 450315/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1094 | 784115 | 449976/L1 | 450316/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1095 | 784203 | 449984/L1 | 450317/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1096 | 784291 | 449992/L1 | 450318/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1097 | 784379 | 450000/L1 | 450319/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1098 | 784467 | 450008/L1 | 450320/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1099 | 784555 | 450016/L1 | 450321/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1100 | 784643 | 450024/L1 | 450322/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1101 | 784731 | 450032/L1 | 450323/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1102 | 784819 | 450040/L1 | 450324/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1103 | 784907 | 450048/L1 | 450325/L5 | qwen24-stage/token_0/layer_09/swiglu ; qwen24-stage/token_1/layer_09/swiglu |
| 1104 | 784995 | 450056/L1 | 450326/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1105 | 785083 | 450064/L1 | 450327/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1106 | 785171 | 450072/L1 | 450328/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1107 | 785259 | 450080/L1 | 450329/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1108 | 785347 | 450088/L1 | 450330/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1109 | 785435 | 450096/L1 | 450331/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1110 | 785523 | 450104/L1 | 450332/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1111 | 785611 | 450112/L1 | 450333/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1112 | 785699 | 450120/L1 | 450334/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1113 | 785787 | 450128/L1 | 450335/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1114 | 785875 | 450136/L1 | 450336/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1115 | 785963 | 450144/L1 | 450337/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1116 | 786051 | 450152/L1 | 450338/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1117 | 786139 | 450160/L1 | 450339/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1118 | 786227 | 450168/L1 | 450340/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1119 | 786315 | 450176/L1 | 450341/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1120 | 786404 | 450184/L1 | 450342/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1121 | 786496 | 450192/L1 | 450343/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1122 | 786593 | 328821/L1 | 328840/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1123 | 788739 | 329940/L1 | 329945/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1124 | 788823 | 329995/L1 | 330010/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1125 | 788876 | 330998/L1 | 331108/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1126 | 788892 | 331001/L1 | 331109/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1127 | 788919 | 331004/L1 | 331110/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1128 | 788947 | 331007/L1 | 331111/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1129 | 789000 | 331187/L1 | 331208/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 1130 | 800695 | 342396/L1 | 342952/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.o_weight.weight |
| 1131 | 800713 | 342972/L1 | 342989/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.o_weight.weight |
| 1132 | 800734 | 353347/L1 | 354435/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1133 | 800754 | 349530/L1 | 354432/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1134 | 800783 | 349533/L1 | 354433/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1135 | 800803 | 349536/L1 | 354434/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1136 | 800815 | 342987/L1 | 342988/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.o_weight.weight |
| 1137 | 812516 | 342938/L1 | 342953/L5 | qwen24-stage/token_0/layer_09/model_list.9.attention.o_weight.weight |
| 1138 | 812539 | 354478/L1 | 354484/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1139 | 825448 | 360954/L1 | 360965/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1140 | 825530 | 361020/L1 | 361031/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1141 | 825581 | 366769/L1 | 367379/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1142 | 825597 | 366772/L1 | 367380/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_07/swiglu |
| 1143 | 825622 | 366775/L1 | 367381/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1144 | 825649 | 366778/L1 | 367382/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1145 | 825702 | 367458/L1 | 367479/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1146 | 826052 | 367861/L1 | 367876/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1147 | 826134 | 462676/L1 | 462679/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1148 | 826185 | 462724/L1 | 462735/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1149 | 826201 | 462727/L1 | 462736/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1150 | 826230 | 462730/L1 | 462737/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1151 | 826261 | 462733/L1 | 462738/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1152 | 826287 | 462788/L1 | 462792/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1153 | 826400 | 464075/L1 | 464148/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1154 | 826488 | 464089/L1 | 464149/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1155 | 826578 | 464103/L1 | 464150/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1156 | 826669 | 464117/L1 | 464151/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1157 | 826741 | 470825/L1 | 470847/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1158 | 826823 | 463669/L1 | 464119/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1159 | 826908 | 463683/L1 | 464120/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1160 | 826998 | 463697/L1 | 464121/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1161 | 827088 | 463711/L1 | 464122/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1162 | 827178 | 463725/L1 | 464123/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1163 | 827268 | 463739/L1 | 464124/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1164 | 827358 | 463753/L1 | 464125/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1165 | 827448 | 463767/L1 | 464126/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1166 | 827538 | 463781/L1 | 464127/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1167 | 827628 | 463795/L1 | 464128/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1168 | 827718 | 463809/L1 | 464129/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1169 | 827808 | 463823/L1 | 464130/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1170 | 827898 | 463837/L1 | 464131/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1171 | 827988 | 463851/L1 | 464132/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1172 | 828078 | 463865/L1 | 464133/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1173 | 828168 | 463879/L1 | 464134/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1174 | 828258 | 463893/L1 | 464135/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1175 | 828348 | 463907/L1 | 464136/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1176 | 828438 | 463921/L1 | 464137/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1177 | 828528 | 463935/L1 | 464138/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1178 | 828618 | 463949/L1 | 464139/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1179 | 828708 | 463963/L1 | 464140/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1180 | 828798 | 463977/L1 | 464141/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1181 | 828888 | 463991/L1 | 464142/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1182 | 828979 | 464005/L1 | 464143/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1183 | 829032 | 470816/L1 | 470845/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1184 | 829386 | 367845/L1 | 367875/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1185 | 829423 | 367961/L1 | 367962/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_07/swiglu |
| 1186 | 829505 | 464019/L1 | 464144/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1187 | 829590 | 464033/L1 | 464145/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1188 | 829680 | 464047/L1 | 464146/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1189 | 829771 | 464061/L1 | 464147/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1190 | 829821 | 470819/L1 | 470846/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1191 | 829877 | 470934/L1 | 470936/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1192 | 829943 | 471015/L1 | 471016/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1193 | 829953 | 471014/L1 | 471021/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1194 | 830034 | 471096/L1 | 471098/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1195 | 830061 | 470833/L1 | 470848/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1196 | 830088 | 470842/L1 | 470849/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1197 | 830142 | 471465/L1 | 471468/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1198 | 832297 | 472574/L1 | 472582/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1199 | 832381 | 473699/L1 | 473701/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1200 | 832432 | 473777/L1 | 473788/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_12/swiglu |
| 1201 | 832448 | 473780/L1 | 473789/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1202 | 832473 | 473783/L1 | 473790/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1203 | 832500 | 473786/L1 | 473791/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1204 | 832552 | 473884/L1 | 473887/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1205 | 844247 | 484790/L1 | 485334/L5 | qwen24-stage/token_0/layer_13/model_list.13.attention.o_weight.weight |
| 1206 | 844273 | 497040/L1 | 497045/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1207 | 844296 | 497075/L1 | 497084/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1208 | 844328 | 497078/L1 | 497085/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1209 | 844370 | 497082/L1 | 497086/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_10/swiglu |
| 1210 | 844379 | 497041/L1 | 497042/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_10/swiglu |
| 1211 | 856079 | 485332/L1 | 485335/L5 | qwen24-stage/token_0/layer_13/model_list.13.attention.o_weight.weight |
| 1212 | 856102 | 497136/L1 | 497140/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1213 | 869013 | 503463/L1 | 503465/L5 | qwen24-stage/token_0/layer_13/swiglu |
| 1214 | 869099 | 367942/L1 | 367963/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 1215 | 871244 | 370321/L1 | 370454/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1216 | 871328 | 370575/L1 | 370596/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1217 | 871379 | 370660/L1 | 370686/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1218 | 871395 | 370663/L1 | 370687/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1219 | 871420 | 370666/L1 | 370688/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1220 | 871448 | 370669/L1 | 370689/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1221 | 871501 | 370756/L1 | 370767/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 1222 | 883196 | 381932/L1 | 382481/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 1223 | 883210 | 382491/L1 | 382506/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 1224 | 883232 | 383468/L1 | 383578/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 1225 | 883256 | 383639/L1 | 383643/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.k_weight.weight |
| 1226 | 883268 | 383471/L1 | 383579/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 1227 | 883296 | 383474/L1 | 383580/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 1228 | 883317 | 383477/L1 | 383581/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 1229 | 895025 | 382474/L1 | 382482/L5 | qwen24-stage/token_1/layer_08/model_list.8.attention.q_weight.weight |
| 1230 | 895048 | 383681/L1 | 383693/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_08/model_list.8.attention.v_weight.weight |
| 1231 | 907955 | 390176/L1 | 390180/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1232 | 908034 | 390223/L1 | 390238/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1233 | 908075 | 400603/L1 | 401699/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1234 | 908084 | 400606/L1 | 401700/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1235 | 908095 | 396785/L1 | 401696/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1236 | 908114 | 400609/L1 | 401701/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1237 | 908131 | 396788/L1 | 401697/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_10/swiglu |
| 1238 | 908163 | 396791/L1 | 401698/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1239 | 908179 | 401694/L1 | 401695/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1240 | 908207 | 401744/L1 | 401748/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1241 | 908246 | 401772/L1 | 401787/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1242 | 908601 | 402153/L1 | 402158/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1243 | 908685 | 509956/L1 | 509958/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1244 | 908736 | 510034/L1 | 510045/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1245 | 908752 | 510037/L1 | 510046/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1246 | 908777 | 510040/L1 | 510047/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1247 | 908804 | 510043/L1 | 510048/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1248 | 908855 | 510141/L1 | 510144/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1249 | 909199 | 511580/L1 | 511585/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1250 | 909251 | 511957/L1 | 512025/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1251 | 911404 | 511548/L1 | 511582/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1252 | 911444 | 512012/L1 | 512029/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1253 | 911791 | 402137/L1 | 402157/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1254 | 911830 | 402202/L1 | 402219/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1255 | 912172 | 511563/L1 | 511583/L5 | qwen24-stage/token_0/layer_10/swiglu ; qwen24-stage/token_1/layer_10/swiglu |
| 1256 | 912212 | 512010/L1 | 512028/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1257 | 912293 | 512126/L1 | 512134/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1258 | 912341 | 513235/L1 | 513244/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1259 | 912350 | 513232/L1 | 513243/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1260 | 912361 | 513238/L1 | 513245/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1261 | 912392 | 513241/L1 | 513246/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1262 | 912422 | 513309/L1 | 513311/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_10/swiglu |
| 1263 | 912427 | 511578/L1 | 511584/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1264 | 912482 | 512018/L1 | 512030/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1265 | 912530 | 513379/L1 | 513384/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1266 | 914691 | 514498/L1 | 514499/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_10/swiglu |
| 1267 | 914701 | 514497/L1 | 514501/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1268 | 914790 | 526235/L1 | 526255/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1269 | 914799 | 526238/L1 | 526256/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1270 | 914811 | 526241/L1 | 526257/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1271 | 914837 | 526244/L1 | 526258/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1272 | 914852 | 526251/L1 | 526252/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1273 | 914873 | 526247/L1 | 526259/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1274 | 914903 | 526250/L1 | 526260/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1275 | 914917 | 526309/L1 | 526316/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1276 | 914936 | 526312/L1 | 526317/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1277 | 914984 | 526360/L1 | 526365/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1278 | 926686 | 537958/L1 | 537961/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1279 | 926711 | 544421/L1 | 544423/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1280 | 926735 | 544484/L1 | 544493/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1281 | 926766 | 544487/L1 | 544494/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1282 | 926808 | 544491/L1 | 544495/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1283 | 938520 | 537946/L1 | 537959/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1284 | 938543 | 544562/L1 | 544567/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1285 | 951453 | 550906/L1 | 550907/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1286 | 951502 | 402208/L1 | 402220/L5 | qwen24-stage/token_0/layer_10/swiglu |
| 1287 | 951615 | 408581/L1 | 409518/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1288 | 951703 | 408595/L1 | 409519/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1289 | 951793 | 408609/L1 | 409520/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1290 | 951883 | 408623/L1 | 409521/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1291 | 951973 | 408637/L1 | 409522/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1292 | 952063 | 408651/L1 | 409523/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1293 | 952153 | 408665/L1 | 409524/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1294 | 952243 | 408679/L1 | 409525/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1295 | 952333 | 408693/L1 | 409526/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1296 | 952423 | 408707/L1 | 409527/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1297 | 952513 | 408721/L1 | 409528/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1298 | 952603 | 408735/L1 | 409529/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1299 | 952693 | 408749/L1 | 409530/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1300 | 952783 | 408763/L1 | 409531/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1301 | 952873 | 408777/L1 | 409532/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1302 | 952963 | 408791/L1 | 409533/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1303 | 953053 | 408805/L1 | 409534/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1304 | 953143 | 408819/L1 | 409535/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1305 | 953233 | 408833/L1 | 409536/L5 | qwen24-stage/token_0/layer_08/swiglu ; qwen24-stage/token_1/layer_08/swiglu |
| 1306 | 953323 | 408847/L1 | 409537/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1307 | 953413 | 408861/L1 | 409538/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1308 | 953503 | 408875/L1 | 409539/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1309 | 953593 | 408889/L1 | 409540/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1310 | 953683 | 408903/L1 | 409541/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1311 | 953775 | 408917/L1 | 409542/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1312 | 953868 | 409740/L1 | 409744/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1313 | 953909 | 409804/L1 | 409829/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1314 | 953918 | 409807/L1 | 409830/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1315 | 953926 | 409783/L1 | 409826/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1316 | 953945 | 409810/L1 | 409831/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1317 | 953959 | 409786/L1 | 409827/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1318 | 953987 | 409789/L1 | 409828/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1319 | 954003 | 409824/L1 | 409825/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1320 | 954030 | 409889/L1 | 409894/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1321 | 954070 | 409930/L1 | 410756/L5 | qwen24-stage/token_1/layer_08/swiglu |
| 1322 | 965766 | 422568/L1 | 423152/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1323 | 965788 | 423245/L1 | 423257/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1324 | 965808 | 423310/L1 | 423323/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1325 | 965830 | 423313/L1 | 423324/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1326 | 965865 | 423317/L1 | 423325/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1327 | 965874 | 423252/L1 | 423253/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1328 | 977570 | 423110/L1 | 423153/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1329 | 977593 | 423374/L1 | 423379/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 1330 | 990497 | 429857/L1 | 429858/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1331 | 990505 | 429842/L1 | 429859/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1332 | 990593 | 430852/L1 | 430970/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1333 | 990602 | 430855/L1 | 430971/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_11/swiglu |
| 1334 | 990613 | 430858/L1 | 430972/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1335 | 990639 | 430861/L1 | 430973/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1336 | 990654 | 430968/L1 | 430969/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1337 | 990675 | 430864/L1 | 430974/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1338 | 990701 | 430867/L1 | 430975/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1339 | 990715 | 431054/L1 | 431061/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1340 | 990734 | 431057/L1 | 431062/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1341 | 990784 | 431119/L1 | 431133/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1342 | 991139 | 431543/L1 | 431544/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1343 | 991148 | 550905/L1 | 550909/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1344 | 991231 | 551295/L1 | 551315/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1345 | 991240 | 551298/L1 | 551316/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1346 | 991248 | 551301/L1 | 551317/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1347 | 991274 | 551304/L1 | 551318/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1348 | 991289 | 551311/L1 | 551312/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1349 | 991310 | 551307/L1 | 551319/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1350 | 991336 | 551310/L1 | 551320/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1351 | 991350 | 551416/L1 | 551427/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1352 | 991369 | 551419/L1 | 551428/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1353 | 991416 | 552528/L1 | 552530/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1354 | 991768 | 553992/L1 | 553997/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_14/swiglu |
| 1355 | 991821 | 554089/L1 | 554139/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1356 | 993978 | 553958/L1 | 553994/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1357 | 994022 | 554125/L1 | 554145/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1358 | 994378 | 431509/L1 | 431546/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1359 | 994400 | 431507/L1 | 431545/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_11/swiglu |
| 1360 | 994757 | 553974/L1 | 553995/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1361 | 994799 | 554123/L1 | 554144/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1362 | 994913 | 554237/L1 | 554239/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1363 | 994957 | 565953/L1 | 565962/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1364 | 994966 | 565956/L1 | 565963/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1365 | 994990 | 565959/L1 | 565964/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1366 | 995013 | 553990/L1 | 553996/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1367 | 995069 | 554131/L1 | 554146/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1368 | 995083 | 565989/L1 | 565991/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1369 | 995122 | 566024/L1 | 566031/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1370 | 997276 | 567127/L1 | 567129/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1371 | 997352 | 567183/L1 | 567185/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1372 | 997389 | 573663/L1 | 573676/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_15/swiglu |
| 1373 | 997398 | 573666/L1 | 573677/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1374 | 997406 | 573053/L1 | 573672/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1375 | 997425 | 573669/L1 | 573678/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1376 | 997439 | 573056/L1 | 573673/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1377 | 997467 | 573059/L1 | 573674/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1378 | 997483 | 573670/L1 | 573671/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1379 | 997511 | 573754/L1 | 573756/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1380 | 997552 | 573805/L1 | 573809/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1381 | 1009248 | 584856/L1 | 585400/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1382 | 1009263 | 585426/L1 | 585428/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1383 | 1009283 | 585783/L1 | 585795/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1384 | 1009308 | 585873/L1 | 585879/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1385 | 1009317 | 585786/L1 | 585796/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1386 | 1009345 | 585789/L1 | 585797/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1387 | 1009362 | 585792/L1 | 585798/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1388 | 1021073 | 585398/L1 | 585401/L5 | qwen24-stage/token_0/layer_12/swiglu ; qwen24-stage/token_1/layer_12/swiglu |
| 1389 | 1021096 | 586873/L1 | 586901/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1390 | 1034006 | 593362/L1 | 593364/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1391 | 1034028 | 431517/L1 | 431547/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1392 | 1034053 | 431526/L1 | 431548/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 1393 | 1034146 | 441944/L1 | 443630/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1394 | 1034231 | 441969/L1 | 443631/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1395 | 1034321 | 441994/L1 | 443632/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1396 | 1034411 | 442019/L1 | 443633/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1397 | 1034501 | 442044/L1 | 443634/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1398 | 1034591 | 442069/L1 | 443635/L5 | qwen24-stage/token_0/layer_00/model_list.0.pre_Normal.weight ; qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1399 | 1034681 | 442094/L1 | 443636/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1400 | 1034771 | 442119/L1 | 443637/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1401 | 1034861 | 442144/L1 | 443638/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1402 | 1034951 | 442169/L1 | 443639/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1403 | 1035041 | 442194/L1 | 443640/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1404 | 1035131 | 442219/L1 | 443641/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1405 | 1035221 | 442244/L1 | 443642/L5 | qwen24-stage/token_0/layer_00/model_list.0.pre_Normal.weight ; qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1406 | 1035311 | 442269/L1 | 443643/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1407 | 1035401 | 442294/L1 | 443644/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1408 | 1035491 | 442319/L1 | 443645/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 1409 | 1035581 | 442344/L1 | 443646/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1410 | 1035671 | 442369/L1 | 443647/L5 | qwen24-stage/token_0/layer_00/model_list.0.pre_Normal.weight ; qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1411 | 1035761 | 442394/L1 | 443648/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1412 | 1035851 | 442419/L1 | 443649/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1413 | 1035941 | 442444/L1 | 443650/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1414 | 1036031 | 442469/L1 | 443651/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1415 | 1036121 | 442494/L1 | 443652/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1416 | 1036211 | 442519/L1 | 443653/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1417 | 1036302 | 442544/L1 | 443654/L5 | qwen24-stage/token_1/layer_09/model_list.9.post_Normal.weight |
| 1418 | 1036350 | 444142/L1 | 444143/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1419 | 1036359 | 444138/L1 | 444146/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1420 | 1036446 | 444205/L1 | 444229/L5 | qwen24-stage/token_0/layer_09/swiglu ; qwen24-stage/token_1/layer_09/swiglu |
| 1421 | 1036455 | 444208/L1 | 444230/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1422 | 1036466 | 444211/L1 | 444231/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1423 | 1036492 | 444214/L1 | 444232/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1424 | 1036507 | 444224/L1 | 444225/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1425 | 1036528 | 444217/L1 | 444233/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1426 | 1036554 | 444220/L1 | 444234/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1427 | 1036568 | 444295/L1 | 444305/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1428 | 1036587 | 444298/L1 | 444306/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1429 | 1036637 | 448992/L1 | 450193/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1430 | 1048341 | 462673/L1 | 462678/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1431 | 1048363 | 462720/L1 | 462734/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1432 | 1048383 | 462777/L1 | 462789/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1433 | 1048411 | 462780/L1 | 462790/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1434 | 1048450 | 462784/L1 | 462791/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1435 | 1060150 | 462637/L1 | 462677/L5 | qwen24-stage/token_1/layer_09/swiglu |
| 1436 | 1060173 | 463655/L1 | 464118/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1437 | 1073077 | 470813/L1 | 470844/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_12/swiglu |
| 1438 | 1073163 | 470928/L1 | 470935/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1439 | 1073218 | 471002/L1 | 471017/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1440 | 1073234 | 471005/L1 | 471018/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1441 | 1073263 | 471008/L1 | 471019/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1442 | 1073295 | 471011/L1 | 471020/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1443 | 1073348 | 471092/L1 | 471097/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1444 | 1073697 | 471451/L1 | 471467/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1445 | 1073779 | 593437/L1 | 593446/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1446 | 1073831 | 593518/L1 | 593529/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1447 | 1073846 | 593521/L1 | 593530/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1448 | 1073875 | 593524/L1 | 593531/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1449 | 1073906 | 593527/L1 | 593532/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1450 | 1073925 | 593569/L1 | 593571/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1451 | 1074016 | 606033/L1 | 606140/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1452 | 1074101 | 606058/L1 | 606141/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1453 | 1074191 | 606083/L1 | 606142/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1454 | 1074281 | 606108/L1 | 606143/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1455 | 1074327 | 606775/L1 | 606780/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1456 | 1074380 | 606847/L1 | 606896/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1457 | 1074433 | 605308/L1 | 606111/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1458 | 1074518 | 605333/L1 | 606112/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1459 | 1074608 | 605358/L1 | 606113/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1460 | 1074698 | 605383/L1 | 606114/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1461 | 1074788 | 605408/L1 | 606115/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1462 | 1074878 | 605433/L1 | 606116/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1463 | 1074968 | 605458/L1 | 606117/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1464 | 1075058 | 605483/L1 | 606118/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1465 | 1075148 | 605508/L1 | 606119/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1466 | 1075238 | 605533/L1 | 606120/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1467 | 1075328 | 605558/L1 | 606121/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1468 | 1075418 | 605583/L1 | 606122/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1469 | 1075508 | 605608/L1 | 606123/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1470 | 1075598 | 605633/L1 | 606124/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1471 | 1075688 | 605658/L1 | 606125/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1472 | 1075778 | 605683/L1 | 606126/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1473 | 1075868 | 605708/L1 | 606127/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1474 | 1075958 | 605733/L1 | 606128/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1475 | 1076048 | 605758/L1 | 606129/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1476 | 1076138 | 605783/L1 | 606130/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1477 | 1076228 | 605808/L1 | 606131/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1478 | 1076318 | 605833/L1 | 606132/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1479 | 1076408 | 605858/L1 | 606133/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1480 | 1076498 | 605883/L1 | 606134/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1481 | 1076588 | 605908/L1 | 606135/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1482 | 1076637 | 606743/L1 | 606777/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1483 | 1076681 | 606882/L1 | 606902/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1484 | 1077029 | 471435/L1 | 471466/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1485 | 1077069 | 472579/L1 | 472580/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_12/swiglu |
| 1486 | 1077123 | 605933/L1 | 606136/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1487 | 1077208 | 605958/L1 | 606137/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1488 | 1077298 | 605983/L1 | 606138/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1489 | 1077389 | 606008/L1 | 606139/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1490 | 1077433 | 606758/L1 | 606778/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1491 | 1077472 | 606880/L1 | 606901/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1492 | 1077557 | 606964/L1 | 606966/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1493 | 1077596 | 607001/L1 | 607003/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1494 | 1077671 | 613484/L1 | 613486/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1495 | 1077683 | 606773/L1 | 606779/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1496 | 1077739 | 606888/L1 | 606903/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1497 | 1077759 | 613537/L1 | 613538/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1498 | 1079942 | 614673/L1 | 614676/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1499 | 1080026 | 614740/L1 | 614742/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_16/swiglu |
| 1500 | 1080077 | 615123/L1 | 615135/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1501 | 1080093 | 615126/L1 | 615136/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1502 | 1080118 | 615129/L1 | 615137/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1503 | 1080145 | 615132/L1 | 615138/L5 | qwen24-stage/token_0/layer_13/model_list.13.attention.q_weight.weight ; qwen24-stage/token_1/layer_13/model_list.13.attention.q_weight.weight |
| 1504 | 1080197 | 615856/L1 | 615885/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1505 | 1091893 | 627377/L1 | 627923/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1506 | 1091912 | 627988/L1 | 627999/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1507 | 1091933 | 628063/L1 | 628069/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1508 | 1091960 | 628041/L1 | 628064/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1509 | 1091991 | 628044/L1 | 628065/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1510 | 1092012 | 628047/L1 | 628066/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1511 | 1092024 | 627991/L1 | 627992/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1512 | 1092031 | 628118/L1 | 628120/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1513 | 1103725 | 627919/L1 | 627924/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1514 | 1116652 | 646124/L1 | 646127/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1515 | 1116734 | 472474/L1 | 472581/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1516 | 1118883 | 473696/L1 | 473700/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_12/swiglu |
| 1517 | 1118967 | 473773/L1 | 473787/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1518 | 1119073 | 473867/L1 | 473885/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1519 | 1119092 | 473870/L1 | 473886/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 1520 | 1119139 | 484248/L1 | 485333/L5 | qwen24-stage/token_0/layer_13/model_list.13.attention.o_weight.weight |
| 1521 | 1130844 | 497037/L1 | 497044/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1522 | 1130865 | 497072/L1 | 497083/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1523 | 1130885 | 497119/L1 | 497137/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1524 | 1130909 | 497122/L1 | 497138/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_10/swiglu |
| 1525 | 1130947 | 497126/L1 | 497139/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1526 | 1142658 | 497032/L1 | 497043/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1527 | 1142681 | 502863/L1 | 503464/L5 | qwen24-stage/token_0/layer_13/swiglu |
| 1528 | 1155585 | 509953/L1 | 509957/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1529 | 1155667 | 510030/L1 | 510044/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1530 | 1155769 | 510124/L1 | 510142/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1531 | 1155788 | 510127/L1 | 510143/L5 | qwen24-stage/token_0/layer_10/swiglu ; qwen24-stage/token_1/layer_10/swiglu |
| 1532 | 1155836 | 511448/L1 | 511581/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1533 | 1156187 | 512006/L1 | 512027/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1534 | 1156271 | 646176/L1 | 646178/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1535 | 1156322 | 646242/L1 | 646255/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_17/swiglu |
| 1536 | 1156338 | 646245/L1 | 646256/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1537 | 1156366 | 646248/L1 | 646257/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1538 | 1156397 | 646251/L1 | 646258/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1539 | 1156423 | 646302/L1 | 646304/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1540 | 1156536 | 654016/L1 | 654089/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1541 | 1156624 | 654030/L1 | 654090/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1542 | 1156714 | 654044/L1 | 654091/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1543 | 1156805 | 654058/L1 | 654092/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1544 | 1156877 | 654354/L1 | 654376/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1545 | 1156959 | 653610/L1 | 654060/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1546 | 1157044 | 653624/L1 | 654061/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1547 | 1157134 | 653638/L1 | 654062/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1548 | 1157224 | 653652/L1 | 654063/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1549 | 1157314 | 653666/L1 | 654064/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1550 | 1157404 | 653680/L1 | 654065/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1551 | 1157494 | 653694/L1 | 654066/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1552 | 1157584 | 653708/L1 | 654067/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1553 | 1157674 | 653722/L1 | 654068/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1554 | 1157764 | 653736/L1 | 654069/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1555 | 1157854 | 653750/L1 | 654070/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1556 | 1157944 | 653764/L1 | 654071/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1557 | 1158034 | 653778/L1 | 654072/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1558 | 1158124 | 653792/L1 | 654073/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1559 | 1158214 | 653806/L1 | 654074/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1560 | 1158304 | 653820/L1 | 654075/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1561 | 1158394 | 653834/L1 | 654076/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1562 | 1158484 | 653848/L1 | 654077/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1563 | 1158574 | 653862/L1 | 654078/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1564 | 1158664 | 653876/L1 | 654079/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1565 | 1158754 | 653890/L1 | 654080/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1566 | 1158844 | 653904/L1 | 654081/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1567 | 1158934 | 653918/L1 | 654082/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1568 | 1159024 | 653932/L1 | 654083/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1569 | 1159115 | 653946/L1 | 654084/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1570 | 1159169 | 654345/L1 | 654374/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1571 | 1159523 | 511990/L1 | 512026/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1572 | 1159561 | 512131/L1 | 512132/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1573 | 1159643 | 653960/L1 | 654085/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1574 | 1159728 | 653974/L1 | 654086/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1575 | 1159818 | 653988/L1 | 654087/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1576 | 1159910 | 654002/L1 | 654088/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1577 | 1159962 | 654348/L1 | 654375/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1578 | 1160018 | 654470/L1 | 654475/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1579 | 1160083 | 654522/L1 | 654523/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1580 | 1160101 | 654552/L1 | 654561/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1581 | 1160110 | 654549/L1 | 654560/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1582 | 1160120 | 654555/L1 | 654562/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1583 | 1160150 | 654558/L1 | 654563/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1584 | 1160181 | 654931/L1 | 654934/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1585 | 1160208 | 654362/L1 | 654377/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1586 | 1160234 | 654371/L1 | 654378/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1587 | 1160289 | 655011/L1 | 655019/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1588 | 1162445 | 657147/L1 | 657149/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1589 | 1162529 | 657229/L1 | 657231/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1590 | 1162580 | 657312/L1 | 657326/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1591 | 1162596 | 657315/L1 | 657327/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1592 | 1162626 | 657318/L1 | 657328/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1593 | 1162657 | 657321/L1 | 657329/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1594 | 1162708 | 657400/L1 | 657402/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1595 | 1174404 | 679974/L1 | 680519/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1596 | 1174427 | 680540/L1 | 680546/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1597 | 1174447 | 680595/L1 | 680606/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1598 | 1174471 | 680598/L1 | 680607/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1599 | 1174512 | 680602/L1 | 680608/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_18/swiglu |
| 1600 | 1174521 | 680543/L1 | 680544/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1601 | 1186221 | 680516/L1 | 680520/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1602 | 1186244 | 680652/L1 | 680654/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1603 | 1199154 | 693406/L1 | 693408/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1604 | 1199240 | 512110/L1 | 512133/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1605 | 1201386 | 513229/L1 | 513242/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1606 | 1201470 | 513305/L1 | 513310/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1607 | 1201523 | 513356/L1 | 513380/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1608 | 1201539 | 513359/L1 | 513381/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1609 | 1201566 | 513362/L1 | 513382/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1610 | 1201594 | 513365/L1 | 513383/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_10/swiglu |
| 1611 | 1201647 | 514495/L1 | 514500/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 1612 | 1213342 | 525690/L1 | 526253/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1613 | 1213362 | 526299/L1 | 526315/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1614 | 1213383 | 526346/L1 | 526364/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1615 | 1213403 | 526333/L1 | 526361/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1616 | 1213434 | 526336/L1 | 526362/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1617 | 1213451 | 526339/L1 | 526363/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1618 | 1213463 | 526313/L1 | 526314/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1619 | 1225164 | 526232/L1 | 526254/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1620 | 1225187 | 537956/L1 | 537960/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1621 | 1238094 | 544418/L1 | 544422/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1622 | 1238176 | 544481/L1 | 544492/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1623 | 1238227 | 544543/L1 | 544563/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1624 | 1238243 | 544546/L1 | 544564/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1625 | 1238268 | 544549/L1 | 544565/L5 | qwen24-stage/token_1/layer_11/swiglu |
| 1626 | 1238295 | 544552/L1 | 544566/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_11/swiglu |
| 1627 | 1238347 | 550903/L1 | 550908/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1628 | 1238700 | 551292/L1 | 551314/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1629 | 1238784 | 693488/L1 | 693490/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1630 | 1238835 | 693571/L1 | 693585/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1631 | 1238851 | 693574/L1 | 693586/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1632 | 1238876 | 693577/L1 | 693587/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1633 | 1238903 | 693580/L1 | 693588/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1634 | 1238954 | 693659/L1 | 693661/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1635 | 1239298 | 695418/L1 | 695424/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1636 | 1239350 | 695525/L1 | 695582/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1637 | 1241503 | 695386/L1 | 695421/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1638 | 1241543 | 695562/L1 | 695585/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1639 | 1241891 | 551276/L1 | 551313/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1640 | 1241930 | 551424/L1 | 551425/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1641 | 1242273 | 695401/L1 | 695422/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1642 | 1242311 | 695560/L1 | 695584/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1643 | 1242392 | 696704/L1 | 696706/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1644 | 1242431 | 696761/L1 | 696763/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1645 | 1242502 | 696826/L1 | 696835/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1646 | 1242514 | 695416/L1 | 695423/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1647 | 1242570 | 695568/L1 | 695586/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1648 | 1242589 | 696883/L1 | 696885/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1649 | 1242673 | 696944/L1 | 697546/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1650 | 1242763 | 696969/L1 | 697547/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1651 | 1242853 | 696994/L1 | 697548/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1652 | 1242943 | 697019/L1 | 697549/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1653 | 1243033 | 697044/L1 | 697550/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1654 | 1243123 | 697069/L1 | 697551/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1655 | 1243213 | 697094/L1 | 697552/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1656 | 1243303 | 697119/L1 | 697553/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1657 | 1243393 | 697144/L1 | 697554/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1658 | 1243483 | 697169/L1 | 697555/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1659 | 1243573 | 697194/L1 | 697556/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1660 | 1243663 | 697219/L1 | 697557/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1661 | 1243753 | 697244/L1 | 697558/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1662 | 1243843 | 697269/L1 | 697559/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1663 | 1243933 | 697294/L1 | 697560/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1664 | 1244023 | 697319/L1 | 697561/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1665 | 1244113 | 697344/L1 | 697562/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1666 | 1244203 | 697369/L1 | 697563/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1667 | 1244293 | 697394/L1 | 697564/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1668 | 1244383 | 697419/L1 | 697565/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1669 | 1244473 | 697444/L1 | 697566/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1670 | 1244563 | 697469/L1 | 697567/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1671 | 1244653 | 697494/L1 | 697568/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1672 | 1244743 | 697519/L1 | 697569/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1673 | 1244834 | 697544/L1 | 697570/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 1674 | 1244882 | 709741/L1 | 709742/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1675 | 1244893 | 709740/L1 | 709745/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1676 | 1244982 | 709794/L1 | 709815/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1677 | 1244991 | 709797/L1 | 709816/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1678 | 1245003 | 709800/L1 | 709817/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1679 | 1245032 | 709803/L1 | 709818/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1680 | 1245047 | 709812/L1 | 709813/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1681 | 1245071 | 709806/L1 | 709819/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1682 | 1245100 | 709809/L1 | 709820/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1683 | 1245114 | 709881/L1 | 709889/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1684 | 1245133 | 709884/L1 | 709890/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1685 | 1245181 | 709938/L1 | 709940/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1686 | 1256885 | 727958/L1 | 727961/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1687 | 1256906 | 728010/L1 | 728012/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1688 | 1256930 | 728078/L1 | 728090/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1689 | 1256956 | 728081/L1 | 728091/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1690 | 1256999 | 728085/L1 | 728092/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1691 | 1268713 | 727356/L1 | 727959/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1692 | 1268736 | 728136/L1 | 728138/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 1693 | 1281646 | 734948/L1 | 734949/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1694 | 1281728 | 551406/L1 | 551426/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1695 | 1283883 | 552514/L1 | 552529/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1696 | 1283967 | 553858/L1 | 553993/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1697 | 1284018 | 554110/L1 | 554140/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1698 | 1284034 | 554113/L1 | 554141/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_14/swiglu |
| 1699 | 1284062 | 554116/L1 | 554142/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1700 | 1284093 | 554119/L1 | 554143/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1701 | 1284144 | 554223/L1 | 554238/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 1702 | 1295839 | 565407/L1 | 565960/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1703 | 1295859 | 565983/L1 | 565990/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1704 | 1295874 | 566001/L1 | 566027/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1705 | 1295899 | 567027/L1 | 567128/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1706 | 1295908 | 566004/L1 | 566028/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1707 | 1295936 | 566007/L1 | 566029/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1708 | 1295953 | 566010/L1 | 566030/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1709 | 1295965 | 566025/L1 | 566026/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1710 | 1307660 | 565949/L1 | 565961/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1711 | 1307683 | 567149/L1 | 567184/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1712 | 1320590 | 573659/L1 | 573675/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1713 | 1320673 | 573751/L1 | 573755/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1714 | 1320703 | 573785/L1 | 573806/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1715 | 1320712 | 573788/L1 | 573807/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1716 | 1320737 | 573791/L1 | 573808/L5 | qwen24-stage/token_0/layer_15/swiglu |
| 1717 | 1320815 | 584314/L1 | 585399/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1718 | 1320855 | 585424/L1 | 585427/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1719 | 1321207 | 585780/L1 | 585794/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1720 | 1321216 | 734947/L1 | 734952/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_15/swiglu |
| 1721 | 1321305 | 735048/L1 | 735073/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1722 | 1321314 | 735051/L1 | 735074/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1723 | 1321325 | 735054/L1 | 735075/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1724 | 1321355 | 735057/L1 | 735076/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_15/swiglu |
| 1725 | 1321370 | 735070/L1 | 735071/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1726 | 1321395 | 735060/L1 | 735077/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1727 | 1321425 | 735063/L1 | 735078/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1728 | 1321439 | 736196/L1 | 736201/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1729 | 1321458 | 736199/L1 | 736202/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1730 | 1321504 | 736265/L1 | 736267/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1731 | 1321848 | 737724/L1 | 737732/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1732 | 1321900 | 737808/L1 | 737882/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1733 | 1324053 | 737692/L1 | 737729/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1734 | 1324093 | 737835/L1 | 737883/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1735 | 1324441 | 585764/L1 | 585793/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1736 | 1324480 | 586874/L1 | 586875/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1737 | 1324823 | 737707/L1 | 737730/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1738 | 1324881 | 737872/L1 | 737887/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1739 | 1324907 | 737881/L1 | 737888/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1740 | 1324980 | 749624/L1 | 749625/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1741 | 1324990 | 749622/L1 | 749627/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1742 | 1325061 | 749666/L1 | 749668/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1743 | 1325073 | 737722/L1 | 737731/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 1744 | 1325129 | 737841/L1 | 737884/L5 | qwen24-stage/token_0/layer_15/swiglu ; qwen24-stage/token_1/layer_15/swiglu |
| 1745 | 1325149 | 749709/L1 | 749713/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1746 | 1325261 | 750364/L1 | 750702/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1747 | 1325349 | 750378/L1 | 750703/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1748 | 1325440 | 750392/L1 | 750704/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1749 | 1325530 | 750406/L1 | 750705/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1750 | 1325620 | 750420/L1 | 750706/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1751 | 1325710 | 750434/L1 | 750707/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1752 | 1325800 | 750448/L1 | 750708/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1753 | 1325890 | 750462/L1 | 750709/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1754 | 1325980 | 750476/L1 | 750710/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1755 | 1326070 | 750490/L1 | 750711/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1756 | 1326160 | 750504/L1 | 750712/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1757 | 1326250 | 750518/L1 | 750713/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1758 | 1326340 | 750532/L1 | 750714/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1759 | 1326430 | 750546/L1 | 750715/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1760 | 1326520 | 750560/L1 | 750716/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1761 | 1326610 | 750574/L1 | 750717/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1762 | 1326700 | 750588/L1 | 750718/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1763 | 1326790 | 750602/L1 | 750719/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1764 | 1326880 | 750616/L1 | 750720/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1765 | 1326970 | 750630/L1 | 750721/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1766 | 1327060 | 750644/L1 | 750722/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1767 | 1327150 | 750658/L1 | 750723/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1768 | 1327240 | 750672/L1 | 750724/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1769 | 1327330 | 750686/L1 | 750725/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1770 | 1327423 | 750700/L1 | 750726/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 1771 | 1327516 | 757337/L1 | 757339/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1772 | 1327557 | 757406/L1 | 757419/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1773 | 1327566 | 757409/L1 | 757420/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1774 | 1327574 | 757393/L1 | 757415/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1775 | 1327593 | 757412/L1 | 757421/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1776 | 1327607 | 757396/L1 | 757416/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1777 | 1327635 | 757399/L1 | 757417/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1778 | 1327651 | 757413/L1 | 757414/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1779 | 1327677 | 757502/L1 | 757507/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1780 | 1327718 | 757543/L1 | 757545/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 1781 | 1339413 | 768643/L1 | 769194/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1782 | 1339429 | 769352/L1 | 769355/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1783 | 1339449 | 769420/L1 | 769437/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1784 | 1339469 | 770537/L1 | 770539/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1785 | 1339478 | 769423/L1 | 769438/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1786 | 1339501 | 769426/L1 | 769439/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1787 | 1339522 | 769429/L1 | 769440/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1788 | 1351230 | 769185/L1 | 769195/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1789 | 1351253 | 770593/L1 | 770595/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 1790 | 1364163 | 776929/L1 | 776938/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 1791 | 1364247 | 585870/L1 | 585878/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1792 | 1364327 | 586527/L1 | 586876/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1793 | 1364411 | 586541/L1 | 586877/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1794 | 1364500 | 586555/L1 | 586878/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1795 | 1364589 | 586569/L1 | 586879/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1796 | 1364678 | 586583/L1 | 586880/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1797 | 1364767 | 586597/L1 | 586881/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1798 | 1364856 | 586611/L1 | 586882/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1799 | 1364945 | 586625/L1 | 586883/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1800 | 1365035 | 586639/L1 | 586884/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1801 | 1365125 | 586653/L1 | 586885/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1802 | 1365215 | 586667/L1 | 586886/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1803 | 1365305 | 586681/L1 | 586887/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1804 | 1365395 | 586695/L1 | 586888/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1805 | 1365485 | 586709/L1 | 586889/L5 | qwen24-stage/token_0/layer_12/swiglu ; qwen24-stage/token_1/layer_12/swiglu |
| 1806 | 1365575 | 586723/L1 | 586890/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1807 | 1365665 | 586737/L1 | 586891/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1808 | 1365755 | 586751/L1 | 586892/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1809 | 1365845 | 586765/L1 | 586893/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1810 | 1365935 | 586779/L1 | 586894/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1811 | 1366025 | 586793/L1 | 586895/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1812 | 1366115 | 586807/L1 | 586896/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1813 | 1366205 | 586821/L1 | 586897/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1814 | 1366295 | 586835/L1 | 586898/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1815 | 1366386 | 586849/L1 | 586899/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1816 | 1366480 | 586863/L1 | 586900/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1817 | 1366572 | 592762/L1 | 593363/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1818 | 1366609 | 593428/L1 | 593443/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1819 | 1366618 | 593431/L1 | 593444/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1820 | 1366626 | 593384/L1 | 593440/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_12/swiglu |
| 1821 | 1366645 | 593434/L1 | 593445/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1822 | 1366659 | 593387/L1 | 593441/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1823 | 1366687 | 593390/L1 | 593442/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1824 | 1366703 | 593438/L1 | 593439/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1825 | 1366730 | 593514/L1 | 593528/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1826 | 1366771 | 593566/L1 | 593570/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 1827 | 1378466 | 604741/L1 | 606109/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1828 | 1378479 | 606643/L1 | 606776/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1829 | 1378499 | 606867/L1 | 606897/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1830 | 1378519 | 606948/L1 | 606965/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1831 | 1378528 | 606870/L1 | 606898/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1832 | 1378554 | 606873/L1 | 606899/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1833 | 1378573 | 606876/L1 | 606900/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1834 | 1390280 | 605283/L1 | 606110/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1835 | 1390303 | 606997/L1 | 607002/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1836 | 1403212 | 613480/L1 | 613485/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1837 | 1403304 | 613541/L1 | 613574/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1838 | 1403313 | 613544/L1 | 613575/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1839 | 1403325 | 613547/L1 | 613576/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1840 | 1403351 | 613550/L1 | 613577/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1841 | 1403366 | 613572/L1 | 613573/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1842 | 1403387 | 613553/L1 | 613578/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_16/swiglu |
| 1843 | 1403413 | 613556/L1 | 613579/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1844 | 1403427 | 614570/L1 | 614674/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_16/swiglu |
| 1845 | 1403446 | 614573/L1 | 614675/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1846 | 1403496 | 614737/L1 | 614741/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1847 | 1403849 | 615119/L1 | 615134/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_0/layer_16/swiglu |
| 1848 | 1403942 | 777014/L1 | 777016/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 1849 | 1403972 | 777060/L1 | 777068/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 1850 | 1403981 | 777063/L1 | 777069/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 1851 | 1404009 | 777066/L1 | 777070/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 1852 | 1404087 | 788795/L1 | 788798/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1853 | 1404127 | 788822/L1 | 788824/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1854 | 1404471 | 790255/L1 | 790263/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1855 | 1404523 | 790340/L1 | 790426/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1856 | 1406678 | 790223/L1 | 790260/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1857 | 1406720 | 790366/L1 | 790382/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1858 | 1407069 | 615103/L1 | 615133/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 1859 | 1407110 | 615857/L1 | 615858/L5 | qwen24-stage/token_0/layer_13/model_list.13.attention.v_weight.weight ; qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1860 | 1407453 | 790238/L1 | 790261/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1861 | 1407491 | 790364/L1 | 790381/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1862 | 1407541 | 790425/L1 | 790428/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1863 | 1407607 | 796914/L1 | 796915/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.q_weight.weight |
| 1864 | 1407615 | 796913/L1 | 796917/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.q_weight.weight |
| 1865 | 1407689 | 796987/L1 | 796989/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.k_weight.weight |
| 1866 | 1407699 | 790253/L1 | 790262/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1867 | 1407755 | 790372/L1 | 790383/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1868 | 1407802 | 797072/L1 | 797075/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.v_weight.weight |
| 1869 | 1409955 | 798175/L1 | 798177/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1870 | 1410039 | 798562/L1 | 798565/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1871 | 1410090 | 798656/L1 | 798673/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1872 | 1410106 | 798659/L1 | 798674/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1873 | 1410135 | 798662/L1 | 798675/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1874 | 1410167 | 798665/L1 | 798676/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1875 | 1410219 | 799789/L1 | 799793/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 1876 | 1421915 | 810864/L1 | 811415/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1877 | 1421935 | 811468/L1 | 811473/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1878 | 1421960 | 811517/L1 | 811522/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1879 | 1421980 | 811493/L1 | 811518/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1880 | 1422009 | 811496/L1 | 811519/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1881 | 1422030 | 811499/L1 | 811520/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1882 | 1422042 | 811469/L1 | 811470/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1883 | 1433745 | 811406/L1 | 811416/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1884 | 1433768 | 823251/L1 | 823254/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1885 | 1446678 | 829564/L1 | 829566/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1886 | 1446815 | 615242/L1 | 615860/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1887 | 1446902 | 615267/L1 | 615861/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1888 | 1446992 | 615292/L1 | 615862/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1889 | 1447082 | 615317/L1 | 615863/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1890 | 1447172 | 615342/L1 | 615864/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1891 | 1447262 | 615367/L1 | 615865/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1892 | 1447352 | 615392/L1 | 615866/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1893 | 1447442 | 615417/L1 | 615867/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1894 | 1447532 | 615442/L1 | 615868/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1895 | 1447622 | 615467/L1 | 615869/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1896 | 1447712 | 615492/L1 | 615870/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1897 | 1447802 | 615517/L1 | 615871/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1898 | 1447892 | 615542/L1 | 615872/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1899 | 1447982 | 615567/L1 | 615873/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1900 | 1448072 | 615592/L1 | 615874/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1901 | 1448162 | 615617/L1 | 615875/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1902 | 1448252 | 615642/L1 | 615876/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1903 | 1448342 | 615667/L1 | 615877/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1904 | 1448432 | 615692/L1 | 615878/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1905 | 1448522 | 615717/L1 | 615879/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1906 | 1448612 | 615742/L1 | 615880/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1907 | 1448702 | 615767/L1 | 615881/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1908 | 1448792 | 615792/L1 | 615882/L5 | qwen24-stage/token_0/layer_13/model_list.13.attention.v_weight.weight ; qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1909 | 1448882 | 615817/L1 | 615883/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1910 | 1448973 | 615842/L1 | 615884/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 1911 | 1449021 | 627920/L1 | 627921/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1912 | 1449032 | 626835/L1 | 627922/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1913 | 1449121 | 627970/L1 | 627993/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1914 | 1449130 | 627973/L1 | 627994/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1915 | 1449141 | 627976/L1 | 627995/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1916 | 1449171 | 627979/L1 | 627996/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1917 | 1449186 | 627989/L1 | 627990/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1918 | 1449211 | 627982/L1 | 627997/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1919 | 1449241 | 627985/L1 | 627998/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1920 | 1449255 | 628057/L1 | 628067/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1921 | 1449274 | 628060/L1 | 628068/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1922 | 1449321 | 628112/L1 | 628119/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1923 | 1461024 | 645524/L1 | 646126/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1924 | 1461046 | 646173/L1 | 646177/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1925 | 1461069 | 646231/L1 | 646252/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1926 | 1461093 | 646234/L1 | 646253/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1927 | 1461132 | 646238/L1 | 646254/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1928 | 1472826 | 644322/L1 | 646125/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1929 | 1472849 | 646298/L1 | 646303/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1930 | 1485756 | 653596/L1 | 654059/L5 | qwen24-stage/token_0/layer_00/swiglu ; qwen24-stage/token_1/layer_13/swiglu |
| 1931 | 1485842 | 654342/L1 | 654373/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1932 | 1485897 | 654455/L1 | 654471/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1933 | 1485913 | 654458/L1 | 654472/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1934 | 1485940 | 654461/L1 | 654473/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1935 | 1485969 | 654464/L1 | 654474/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1936 | 1486021 | 654546/L1 | 654559/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1937 | 1486374 | 654927/L1 | 654933/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1938 | 1486458 | 829634/L1 | 829638/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1939 | 1486509 | 829696/L1 | 829707/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1940 | 1486525 | 829699/L1 | 829708/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1941 | 1486550 | 829702/L1 | 829709/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1942 | 1486577 | 829705/L1 | 829710/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 1943 | 1486628 | 836197/L1 | 836201/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1944 | 1486972 | 837669/L1 | 837681/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1945 | 1487024 | 837772/L1 | 837822/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1946 | 1489179 | 837637/L1 | 837678/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1947 | 1489220 | 837808/L1 | 837826/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1948 | 1489569 | 654911/L1 | 654932/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1949 | 1489609 | 655016/L1 | 655017/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1950 | 1489952 | 837652/L1 | 837679/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1951 | 1489990 | 837806/L1 | 837825/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1952 | 1490073 | 837892/L1 | 837894/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1953 | 1490121 | 838263/L1 | 838273/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1954 | 1490130 | 838260/L1 | 838272/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1955 | 1490142 | 838266/L1 | 838274/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1956 | 1490173 | 838269/L1 | 838275/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1957 | 1490204 | 838315/L1 | 838322/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1958 | 1490209 | 837667/L1 | 837680/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1959 | 1490265 | 837814/L1 | 837827/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1960 | 1490315 | 838346/L1 | 838347/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1961 | 1492470 | 839398/L1 | 839399/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1962 | 1492554 | 839439/L1 | 839440/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1963 | 1492605 | 839476/L1 | 839486/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1964 | 1492621 | 839479/L1 | 839487/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1965 | 1492646 | 839482/L1 | 839488/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1966 | 1492673 | 839485/L1 | 839489/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1967 | 1492724 | 839524/L1 | 839525/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 1968 | 1504421 | 850404/L1 | 850947/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1969 | 1504449 | 850958/L1 | 850961/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1970 | 1504474 | 850982/L1 | 850990/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1971 | 1504505 | 850985/L1 | 850991/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1972 | 1504548 | 850989/L1 | 850992/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1973 | 1504557 | 850959/L1 | 850960/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1974 | 1516257 | 850946/L1 | 850948/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1975 | 1516280 | 851014/L1 | 851015/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1976 | 1529191 | 857316/L1 | 857317/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 1977 | 1529271 | 654997/L1 | 655018/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 1978 | 1531417 | 657047/L1 | 657148/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1979 | 1531501 | 657226/L1 | 657230/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1980 | 1531552 | 657299/L1 | 657322/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1981 | 1531568 | 657302/L1 | 657323/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1982 | 1531593 | 657305/L1 | 657324/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1983 | 1531620 | 657308/L1 | 657325/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1984 | 1531673 | 657386/L1 | 657401/L5 | qwen24-stage/token_0/layer_17/swiglu |
| 1985 | 1543370 | 678890/L1 | 680517/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1986 | 1543395 | 680537/L1 | 680545/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1987 | 1543417 | 680585/L1 | 680603/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1988 | 1543448 | 680588/L1 | 680604/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1989 | 1543491 | 680592/L1 | 680605/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1990 | 1543500 | 680541/L1 | 680542/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1991 | 1555201 | 679432/L1 | 680518/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1992 | 1555224 | 680642/L1 | 680653/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1993 | 1568128 | 692806/L1 | 693407/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1994 | 1568210 | 693485/L1 | 693489/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1995 | 1568261 | 693558/L1 | 693581/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1996 | 1568277 | 693561/L1 | 693582/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1997 | 1568302 | 693564/L1 | 693583/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1998 | 1568329 | 693567/L1 | 693584/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 1999 | 1568381 | 693645/L1 | 693660/L5 | qwen24-stage/token_0/layer_18/swiglu |
| 2000 | 1568733 | 695286/L1 | 695420/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2001 | 1568817 | 857357/L1 | 857358/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2002 | 1568868 | 857394/L1 | 857404/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2003 | 1568884 | 857397/L1 | 857405/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2004 | 1568909 | 857400/L1 | 857406/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2005 | 1568936 | 857403/L1 | 857407/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2006 | 1568987 | 857442/L1 | 857443/L5 | qwen24-stage/token_0/layer_18/swiglu ; qwen24-stage/token_1/layer_18/swiglu |
| 2007 | 1569331 | 858855/L1 | 858859/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2008 | 1569383 | 858919/L1 | 858956/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2009 | 1571536 | 858823/L1 | 858856/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2010 | 1571575 | 858942/L1 | 858958/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2011 | 1571923 | 695270/L1 | 695419/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2012 | 1571963 | 695580/L1 | 695581/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2013 | 1572306 | 858838/L1 | 858857/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2014 | 1572344 | 858940/L1 | 858957/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2015 | 1572425 | 859000/L1 | 859001/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2016 | 1572472 | 859025/L1 | 859033/L5 | qwen24-stage/token_0/layer_18/swiglu ; qwen24-stage/token_1/layer_18/swiglu |
| 2017 | 1572481 | 859022/L1 | 859032/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2018 | 1572491 | 859028/L1 | 859034/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2019 | 1572522 | 859031/L1 | 859035/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2020 | 1572553 | 859058/L1 | 859059/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2021 | 1572558 | 858853/L1 | 858858/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2022 | 1572614 | 858948/L1 | 858959/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2023 | 1572662 | 859082/L1 | 859083/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2024 | 1574818 | 860134/L1 | 860135/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2025 | 1574902 | 860175/L1 | 860176/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2026 | 1574953 | 860212/L1 | 860222/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2027 | 1574969 | 860215/L1 | 860223/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2028 | 1574994 | 860218/L1 | 860224/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2029 | 1575023 | 860221/L1 | 860225/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2030 | 1575076 | 860260/L1 | 860261/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2031 | 1586776 | 871140/L1 | 871683/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2032 | 1586804 | 871694/L1 | 871697/L5 | qwen24-stage/token_0/layer_19/swiglu ; qwen24-stage/token_1/layer_19/swiglu |
| 2033 | 1586828 | 871718/L1 | 871726/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2034 | 1586861 | 871721/L1 | 871727/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2035 | 1586903 | 871725/L1 | 871728/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2036 | 1586912 | 871695/L1 | 871696/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2037 | 1598612 | 871682/L1 | 871684/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2038 | 1598635 | 871750/L1 | 871751/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2039 | 1611546 | 878052/L1 | 878053/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2040 | 1611626 | 695556/L1 | 695583/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2041 | 1613769 | 696688/L1 | 696705/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2042 | 1613847 | 696757/L1 | 696762/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2043 | 1613885 | 696816/L1 | 696832/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2044 | 1613894 | 696819/L1 | 696833/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2045 | 1613904 | 696795/L1 | 696829/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2046 | 1613923 | 696822/L1 | 696834/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2047 | 1613939 | 696798/L1 | 696830/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2048 | 1613969 | 696801/L1 | 696831/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2049 | 1613985 | 696827/L1 | 696828/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2050 | 1614012 | 696881/L1 | 696884/L5 | qwen24-stage/token_0/layer_14/swiglu ; qwen24-stage/token_1/layer_14/swiglu |
| 2051 | 1614052 | 696919/L1 | 697545/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2052 | 1625744 | 709196/L1 | 709743/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2053 | 1625765 | 709791/L1 | 709814/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2054 | 1625790 | 709871/L1 | 709888/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2055 | 1625816 | 709846/L1 | 709885/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2056 | 1625846 | 709849/L1 | 709886/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2057 | 1625865 | 709852/L1 | 709887/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2058 | 1625877 | 709810/L1 | 709811/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2059 | 1625884 | 709924/L1 | 709939/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2060 | 1637579 | 709738/L1 | 709744/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2061 | 1650500 | 727956/L1 | 727960/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2062 | 1650591 | 728007/L1 | 728011/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2063 | 1650646 | 728066/L1 | 728086/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2064 | 1650662 | 728069/L1 | 728087/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2065 | 1650691 | 728072/L1 | 728088/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2066 | 1650722 | 728075/L1 | 728089/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2067 | 1650748 | 728126/L1 | 728137/L5 | qwen24-stage/token_0/layer_19/swiglu |
| 2068 | 1651129 | 734945/L1 | 734951/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2069 | 1651213 | 878093/L1 | 878094/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2070 | 1651264 | 878130/L1 | 878140/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2071 | 1651280 | 878133/L1 | 878141/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2072 | 1651311 | 878136/L1 | 878142/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2073 | 1651342 | 878139/L1 | 878143/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2074 | 1651398 | 878179/L1 | 878180/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2075 | 1651742 | 879592/L1 | 879596/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2076 | 1651794 | 879656/L1 | 879693/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2077 | 1653947 | 879560/L1 | 879593/L5 | qwen24-stage/token_0/layer_19/swiglu ; qwen24-stage/token_1/layer_19/swiglu |
| 2078 | 1653987 | 879679/L1 | 879695/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2079 | 1654335 | 734930/L1 | 734950/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2080 | 1654375 | 735068/L1 | 735069/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2081 | 1654718 | 879575/L1 | 879594/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2082 | 1654756 | 879677/L1 | 879694/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2083 | 1654837 | 879737/L1 | 879738/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2084 | 1654884 | 879762/L1 | 879770/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2085 | 1654893 | 879759/L1 | 879769/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2086 | 1654903 | 879765/L1 | 879771/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2087 | 1654934 | 879768/L1 | 879772/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2088 | 1654965 | 879795/L1 | 879796/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2089 | 1654970 | 879590/L1 | 879595/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2090 | 1655026 | 879685/L1 | 879696/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2091 | 1655074 | 879819/L1 | 879820/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2092 | 1657230 | 880871/L1 | 880872/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2093 | 1657314 | 880912/L1 | 880913/L5 | qwen24-stage/token_0/layer_19/swiglu ; qwen24-stage/token_1/layer_19/swiglu |
| 2094 | 1657365 | 880949/L1 | 880959/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2095 | 1657381 | 880952/L1 | 880960/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2096 | 1657406 | 880955/L1 | 880961/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2097 | 1657435 | 880958/L1 | 880962/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2098 | 1657488 | 880997/L1 | 880998/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2099 | 1669187 | 892013/L1 | 892556/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2100 | 1669215 | 892567/L1 | 892570/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2101 | 1669239 | 892591/L1 | 892599/L5 | qwen24-stage/token_0/layer_20/swiglu ; qwen24-stage/token_1/layer_20/swiglu |
| 2102 | 1669271 | 892594/L1 | 892600/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2103 | 1669314 | 892598/L1 | 892601/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2104 | 1669323 | 892568/L1 | 892569/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2105 | 1681023 | 892555/L1 | 892557/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2106 | 1681046 | 892623/L1 | 892624/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2107 | 1693956 | 898925/L1 | 898926/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2108 | 1694036 | 735045/L1 | 735072/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_15/swiglu |
| 2109 | 1696179 | 736186/L1 | 736200/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2110 | 1696263 | 736251/L1 | 736266/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2111 | 1696314 | 737583/L1 | 737725/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2112 | 1696330 | 737586/L1 | 737726/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2113 | 1696356 | 737589/L1 | 737727/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2114 | 1696385 | 737592/L1 | 737728/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2115 | 1696438 | 737864/L1 | 737886/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2116 | 1708140 | 749619/L1 | 749626/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 2117 | 1708163 | 749662/L1 | 749667/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 2118 | 1708187 | 749700/L1 | 749710/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 2119 | 1708218 | 749703/L1 | 749711/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 2120 | 1708261 | 749707/L1 | 749712/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 2121 | 1719971 | 749603/L1 | 749623/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 2122 | 1719994 | 750350/L1 | 750701/L5 | qwen24-stage/token_0/layer_20/swiglu |
| 2123 | 1732898 | 757303/L1 | 757338/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 2124 | 1732978 | 757402/L1 | 757418/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 2125 | 1733030 | 757490/L1 | 757503/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 2126 | 1733045 | 757493/L1 | 757504/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 2127 | 1733073 | 757496/L1 | 757505/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 2128 | 1733104 | 757499/L1 | 757506/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 2129 | 1733123 | 757529/L1 | 757544/L5 | qwen24-stage/token_1/layer_16/model_list.16.attention.o_weight.weight |
| 2130 | 1733214 | 768026/L1 | 769190/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2131 | 1733299 | 768051/L1 | 769191/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2132 | 1733389 | 768076/L1 | 769192/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2133 | 1733479 | 768101/L1 | 769193/L5 | qwen24-stage/token_0/layer_16/swiglu ; qwen24-stage/token_1/layer_16/swiglu |
| 2134 | 1733527 | 769350/L1 | 769354/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2135 | 1733611 | 898966/L1 | 898967/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2136 | 1733662 | 899003/L1 | 899013/L5 | qwen24-stage/token_0/layer_20/swiglu ; qwen24-stage/token_1/layer_20/swiglu |
| 2137 | 1733678 | 899006/L1 | 899014/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2138 | 1733703 | 899009/L1 | 899015/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2139 | 1733730 | 899012/L1 | 899016/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2140 | 1733782 | 899052/L1 | 899053/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2141 | 1734131 | 900473/L1 | 900477/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2142 | 1734183 | 900537/L1 | 900574/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2143 | 1736337 | 900441/L1 | 900474/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2144 | 1736380 | 900560/L1 | 900576/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2145 | 1736434 | 767926/L1 | 769186/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2146 | 1736519 | 767951/L1 | 769187/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2147 | 1736609 | 767976/L1 | 769188/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2148 | 1736700 | 768001/L1 | 769189/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2149 | 1736744 | 769334/L1 | 769353/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2150 | 1736784 | 769434/L1 | 769435/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2151 | 1737132 | 900456/L1 | 900475/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2152 | 1737171 | 900558/L1 | 900575/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2153 | 1737252 | 900618/L1 | 900619/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2154 | 1737301 | 900643/L1 | 900651/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2155 | 1737310 | 900640/L1 | 900650/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2156 | 1737322 | 900646/L1 | 900652/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2157 | 1737353 | 900649/L1 | 900653/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2158 | 1737384 | 900676/L1 | 900677/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2159 | 1737389 | 900471/L1 | 900476/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2160 | 1737445 | 900566/L1 | 900577/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2161 | 1737493 | 900700/L1 | 900701/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2162 | 1739647 | 901752/L1 | 901753/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2163 | 1739731 | 901793/L1 | 901794/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2164 | 1739782 | 901830/L1 | 901840/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2165 | 1739798 | 901833/L1 | 901841/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2166 | 1739823 | 901836/L1 | 901842/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2167 | 1739850 | 901839/L1 | 901843/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2168 | 1739902 | 901878/L1 | 901879/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2169 | 1751598 | 912758/L1 | 913301/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2170 | 1751625 | 913312/L1 | 913315/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2171 | 1751648 | 913336/L1 | 913344/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2172 | 1751680 | 913339/L1 | 913345/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2173 | 1751722 | 913343/L1 | 913346/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2174 | 1751731 | 913313/L1 | 913314/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2175 | 1763431 | 913300/L1 | 913302/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2176 | 1763454 | 913368/L1 | 913369/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2177 | 1776365 | 919670/L1 | 919671/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2178 | 1776445 | 769417/L1 | 769436/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2179 | 1778592 | 770534/L1 | 770538/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2180 | 1778672 | 770583/L1 | 770594/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2181 | 1778712 | 776323/L1 | 776935/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2182 | 1778721 | 776326/L1 | 776936/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2183 | 1778731 | 775113/L1 | 776932/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2184 | 1778750 | 776329/L1 | 776937/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2185 | 1778766 | 775116/L1 | 776933/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2186 | 1778796 | 775119/L1 | 776934/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2187 | 1778812 | 776930/L1 | 776931/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2188 | 1778840 | 777011/L1 | 777015/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2189 | 1778880 | 777055/L1 | 777067/L5 | qwen24-stage/token_0/layer_21/model_list.21.attention.o_weight.weight |
| 2190 | 1790568 | 788250/L1 | 788796/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2191 | 1790584 | 788808/L1 | 788823/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2192 | 1790606 | 790114/L1 | 790256/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2193 | 1790632 | 790360/L1 | 790380/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2194 | 1790643 | 790117/L1 | 790257/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2195 | 1790672 | 790120/L1 | 790258/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2196 | 1790691 | 790123/L1 | 790259/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2197 | 1802398 | 788792/L1 | 788797/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2198 | 1802421 | 790419/L1 | 790427/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2199 | 1815324 | 796910/L1 | 796916/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.q_weight.weight |
| 2200 | 1815410 | 796983/L1 | 796988/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.k_weight.weight |
| 2201 | 1815522 | 797055/L1 | 797073/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.v_weight.weight |
| 2202 | 1815541 | 797058/L1 | 797074/L5 | qwen24-stage/token_1/layer_17/model_list.17.attention.v_weight.weight |
| 2203 | 1815589 | 798075/L1 | 798176/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2204 | 1815942 | 798559/L1 | 798564/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2205 | 1816026 | 919711/L1 | 919712/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2206 | 1816077 | 919748/L1 | 919758/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2207 | 1816093 | 919751/L1 | 919759/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2208 | 1816124 | 919754/L1 | 919760/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2209 | 1816157 | 919757/L1 | 919761/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2210 | 1816208 | 919796/L1 | 919797/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2211 | 1816552 | 921209/L1 | 921213/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2212 | 1816604 | 921273/L1 | 921310/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2213 | 1818757 | 921177/L1 | 921210/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2214 | 1818797 | 921296/L1 | 921312/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2215 | 1819145 | 798543/L1 | 798563/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2216 | 1819185 | 798670/L1 | 798671/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2217 | 1819528 | 921192/L1 | 921211/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2218 | 1819566 | 921294/L1 | 921311/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2219 | 1819647 | 921354/L1 | 921355/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2220 | 1819694 | 921379/L1 | 921387/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2221 | 1819703 | 921376/L1 | 921386/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2222 | 1819713 | 921382/L1 | 921388/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2223 | 1819744 | 921385/L1 | 921389/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2224 | 1819775 | 921412/L1 | 921413/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2225 | 1819780 | 921207/L1 | 921212/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2226 | 1819836 | 921302/L1 | 921313/L5 | qwen24-stage/token_0/layer_21/swiglu ; qwen24-stage/token_1/layer_21/swiglu |
| 2227 | 1819884 | 921436/L1 | 921437/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2228 | 1822038 | 922513/L1 | 922514/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2229 | 1822122 | 922554/L1 | 922555/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2230 | 1822173 | 922591/L1 | 922601/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2231 | 1822189 | 922594/L1 | 922602/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2232 | 1822214 | 922597/L1 | 922603/L5 | qwen24-stage/token_0/layer_21/swiglu ; qwen24-stage/token_1/layer_21/swiglu |
| 2233 | 1822242 | 922600/L1 | 922604/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2234 | 1822297 | 922639/L1 | 922640/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2235 | 1833995 | 933655/L1 | 934198/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2236 | 1834024 | 934209/L1 | 934212/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2237 | 1834047 | 934233/L1 | 934241/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2238 | 1834079 | 934236/L1 | 934242/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2239 | 1834120 | 934240/L1 | 934243/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2240 | 1834129 | 934210/L1 | 934211/L5 | qwen24-stage/token_0/layer_22/swiglu ; qwen24-stage/token_1/layer_22/swiglu |
| 2241 | 1845831 | 934197/L1 | 934199/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2242 | 1845854 | 934265/L1 | 934266/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2243 | 1858764 | 940717/L1 | 940718/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2244 | 1858844 | 798652/L1 | 798672/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2245 | 1861006 | 799790/L1 | 799791/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2246 | 1861016 | 799775/L1 | 799792/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2247 | 1861105 | 810307/L1 | 811409/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2248 | 1861114 | 810310/L1 | 811410/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2249 | 1861125 | 810313/L1 | 811411/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2250 | 1861155 | 810316/L1 | 811412/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2251 | 1861170 | 811407/L1 | 811408/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2252 | 1861195 | 810319/L1 | 811413/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2253 | 1861225 | 810322/L1 | 811414/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2254 | 1861239 | 811462/L1 | 811471/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2255 | 1861258 | 811465/L1 | 811472/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2256 | 1861305 | 811514/L1 | 811521/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2257 | 1873011 | 823241/L1 | 823253/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2258 | 1873030 | 828964/L1 | 829565/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_0/layer_22/swiglu |
| 2259 | 1873050 | 829624/L1 | 829635/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2260 | 1873072 | 829627/L1 | 829636/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2261 | 1873107 | 829631/L1 | 829637/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2262 | 1884808 | 823233/L1 | 823252/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2263 | 1884831 | 829692/L1 | 829706/L5 | qwen24-stage/token_1/layer_17/swiglu |
| 2264 | 1897741 | 836198/L1 | 836199/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2265 | 1897750 | 836183/L1 | 836200/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2266 | 1897836 | 837522/L1 | 837672/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2267 | 1897845 | 837525/L1 | 837673/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2268 | 1897857 | 837528/L1 | 837674/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2269 | 1897887 | 837531/L1 | 837675/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2270 | 1897902 | 837670/L1 | 837671/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2271 | 1897927 | 837534/L1 | 837676/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2272 | 1897956 | 837537/L1 | 837677/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2273 | 1897970 | 837799/L1 | 837823/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2274 | 1897989 | 837802/L1 | 837824/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2275 | 1898037 | 837876/L1 | 837893/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2276 | 1898381 | 838257/L1 | 838271/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2277 | 1898469 | 940758/L1 | 940759/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2278 | 1898525 | 940795/L1 | 940805/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2279 | 1898541 | 940798/L1 | 940806/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2280 | 1898571 | 940801/L1 | 940807/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2281 | 1898604 | 940804/L1 | 940808/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2282 | 1898655 | 940843/L1 | 940844/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2283 | 1898999 | 942256/L1 | 942260/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2284 | 1899051 | 942320/L1 | 942357/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2285 | 1901206 | 942224/L1 | 942257/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2286 | 1901250 | 942343/L1 | 942359/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2287 | 1901596 | 838241/L1 | 838270/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2288 | 1901637 | 838320/L1 | 838321/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2289 | 1901982 | 942239/L1 | 942258/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2290 | 1902024 | 942341/L1 | 942358/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2291 | 1902109 | 942401/L1 | 942402/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2292 | 1902159 | 942426/L1 | 942434/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2293 | 1902168 | 942423/L1 | 942433/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2294 | 1902180 | 942429/L1 | 942435/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2295 | 1902211 | 942432/L1 | 942436/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2296 | 1902241 | 942459/L1 | 942460/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2297 | 1902246 | 942254/L1 | 942259/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2298 | 1902304 | 942349/L1 | 942360/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2299 | 1902354 | 942483/L1 | 942484/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2300 | 1904509 | 943560/L1 | 943561/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2301 | 1904593 | 943601/L1 | 943602/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2302 | 1904644 | 943638/L1 | 943648/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2303 | 1904660 | 943641/L1 | 943649/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2304 | 1904687 | 943644/L1 | 943650/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2305 | 1904716 | 943647/L1 | 943651/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2306 | 1904769 | 943687/L1 | 943688/L5 | qwen24-stage/token_1/layer_22/swiglu |
| 2307 | 1916465 | 954839/L1 | 955382/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2308 | 1916494 | 955393/L1 | 955396/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2309 | 1916518 | 955417/L1 | 955425/L5 | qwen24-stage/token_0/layer_23/swiglu ; qwen24-stage/token_1/layer_23/swiglu |
| 2310 | 1916550 | 955420/L1 | 955426/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2311 | 1916593 | 955424/L1 | 955427/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2312 | 1916602 | 955394/L1 | 955395/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2313 | 1928305 | 955381/L1 | 955383/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2314 | 1928328 | 955449/L1 | 955450/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2315 | 1941239 | 961901/L1 | 961902/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2316 | 1941326 | 961942/L1 | 961943/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2317 | 1941383 | 961979/L1 | 961989/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2318 | 1941399 | 961982/L1 | 961990/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2319 | 1941428 | 961985/L1 | 961991/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2320 | 1941459 | 961988/L1 | 961992/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2321 | 1941485 | 962014/L1 | 962015/L5 | qwen24-stage/token_1/layer_23/swiglu |
| 2322 | 1982869 | 2229/L1 | 2235/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 2323 | 1982888 | 41773/L1 | 41785/L5 | qwen24-stage/token_1/layer_00/swiglu |
| 2324 | 1982914 | 44560/L1 | 44597/L5 | qwen24-stage/token_0/layer_00/swiglu |
| 2325 | 1982933 | 76319/L1 | 76327/L5 | qwen24-stage/token_0/layer_01/swiglu |
| 2326 | 1982957 | 97276/L1 | 97281/L5 | qwen24-stage/token_1/layer_01/swiglu |
| 2327 | 1982976 | 105542/L1 | 105550/L5 | qwen24-stage/token_0/layer_02/swiglu |
| 2328 | 1983000 | 144558/L1 | 144565/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 2329 | 1983019 | 145100/L1 | 145105/L5 | qwen24-stage/token_0/layer_03/swiglu |
| 2330 | 1983043 | 186941/L1 | 186947/L5 | qwen24-stage/token_0/layer_00/model_list.0.attention.q_weight.weight ; qwen24-stage/token_1/layer_03/swiglu |
| 2331 | 1983060 | 185761/L1 | 185769/L5 | qwen24-stage/token_1/layer_03/swiglu |
| 2332 | 1983079 | 228256/L1 | 228265/L5 | qwen24-stage/token_0/layer_04/swiglu ; qwen24-stage/token_1/layer_04/swiglu |
| 2333 | 1983088 | 238931/L1 | 240035/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 2334 | 1983105 | 225446/L1 | 225453/L5 | qwen24-stage/token_1/layer_04/swiglu |
| 2335 | 1983138 | 259860/L1 | 259867/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 2336 | 1983147 | 259904/L1 | 259909/L5 | qwen24-stage/token_0/layer_06/swiglu |
| 2337 | 1983171 | 328354/L1 | 328360/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 2338 | 1983190 | 289257/L1 | 289262/L5 | qwen24-stage/token_0/layer_07/swiglu |
| 2339 | 1983214 | 370594/L1 | 370600/L5 | qwen24-stage/token_0/layer_09/swiglu |
| 2340 | 1983231 | 328837/L1 | 328845/L5 | qwen24-stage/token_1/layer_06/swiglu |
| 2341 | 1983250 | 423151/L1 | 423161/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 2342 | 1983259 | 423222/L1 | 423256/L5 | qwen24-stage/token_1/layer_09/model_list.9.attention.q_weight.weight |
| 2343 | 1983276 | 367960/L1 | 367965/L5 | qwen24-stage/token_1/layer_07/swiglu |
| 2344 | 1983297 | 470213/L1 | 470843/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 2345 | 1983343 | 512024/L1 | 512031/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 2346 | 1983355 | 431542/L1 | 431550/L5 | qwen24-stage/token_0/layer_11/swiglu |
| 2347 | 1983364 | 441919/L1 | 443629/L5 | qwen24-stage/token_0/layer_12/model_list.12.pre_Normal.weight |
| 2348 | 1983388 | 554138/L1 | 554147/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 2349 | 1983407 | 472578/L1 | 472583/L5 | qwen24-stage/token_0/layer_12/swiglu |
| 2350 | 1983431 | 606895/L1 | 606904/L5 | qwen24-stage/token_0/layer_16/swiglu |
| 2351 | 1983448 | 512130/L1 | 512135/L5 | qwen24-stage/token_1/layer_10/swiglu |
| 2352 | 1983470 | 654339/L1 | 654372/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 2353 | 1983494 | 551423/L1 | 551429/L5 | qwen24-stage/token_0/layer_14/swiglu |
| 2354 | 1983518 | 695575/L1 | 695587/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2355 | 1983537 | 585877/L1 | 585880/L5 | qwen24-stage/token_1/layer_12/swiglu |
| 2356 | 1983563 | 737849/L1 | 737885/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2357 | 1983582 | 615217/L1 | 615859/L5 | qwen24-stage/token_1/layer_13/model_list.13.attention.v_weight.weight |
| 2358 | 1983606 | 790379/L1 | 790384/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2359 | 1983625 | 655015/L1 | 655020/L5 | qwen24-stage/token_1/layer_13/swiglu |
| 2360 | 1983649 | 837821/L1 | 837828/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2361 | 1983668 | 695579/L1 | 695588/L5 | qwen24-stage/token_1/layer_14/swiglu |
| 2362 | 1983692 | 858955/L1 | 858960/L5 | qwen24-stage/token_1/layer_18/swiglu |
| 2363 | 1983711 | 735067/L1 | 735079/L5 | qwen24-stage/token_1/layer_15/swiglu |
| 2364 | 1983735 | 879692/L1 | 879697/L5 | qwen24-stage/token_1/layer_19/swiglu |
| 2365 | 1983754 | 769433/L1 | 769441/L5 | qwen24-stage/token_1/layer_16/swiglu |
| 2366 | 1983778 | 900573/L1 | 900578/L5 | qwen24-stage/token_1/layer_20/swiglu |
| 2367 | 1983797 | 798669/L1 | 798677/L5 | qwen24-stage/token_0/layer_21/swiglu |
| 2368 | 1983821 | 921309/L1 | 921314/L5 | qwen24-stage/token_1/layer_21/swiglu |
| 2369 | 1983840 | 838319/L1 | 838323/L5 | qwen24-stage/token_0/layer_22/swiglu |
| 2370 | 1983865 | 942356/L1 | 942361/L5 | qwen24-stage/token_1/layer_22/swiglu |

## 7. 峰值时刻存活分布

优化后图数据峰值：0.567383 GiB；指令 ordinal 1974811。调用者持有的输入密文按整个执行期间驻留计算。

| 类型 / level / 分量 | 存活数 | 系数 MiB |
| --- | --- | --- |
| ciphertext:L2:c2 | 56 | 168.000 |
| ciphertext:L3:c2 | 39 | 156.000 |
| ciphertext:L4:c2 | 1 | 5.000 |
| ciphertext:L5:c2 | 42 | 252.000 |


来源：`4f69c6ee95dc3cae5653800e70e9e857b047cdf6370e968f1e80ed62ac2a62b7`。源码阶段映射核验：`{"passed": true, "value_count": 982710, "labelled_values": 982710, "unlabelled_values": 0, "multiple_labels": 61418, "note": "Fused/shared locations are reported jointly; never duplicated across totals", "reencode_ids_resolved_from_identical_payload": 9238}`。
