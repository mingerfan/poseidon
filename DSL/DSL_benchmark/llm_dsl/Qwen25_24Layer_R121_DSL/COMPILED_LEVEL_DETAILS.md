# R121对应24层：完整level、数据量与自举明细

来自真实宽度24层、两个已知token的实际CPU编译计划，且完整Mock通过。此处native Boot是计划算子；没有执行真实CKKS或GPU。

执行计划 SHA256：`0f963b49633181d96392974d3a9726cd84d674d8daea806913ab197f55c33e7a`。

诊断位置与ValueId对应核验：`{"passed": true, "value_count": 355600, "labelled_values": 355600, "unlabelled_values": 0, "multiple_labels": 6203, "note": "Fused/shared locations are reported jointly; never duplicated across totals", "reencode_ids_resolved_from_identical_payload": 4522}`。

## 1. 优化后明文/密文的level与分量分布

| 类型 / level / 分量 | 数量 | 系数字节 | GiB |
| --- | --- | --- | --- |
| ciphertext:L1:c2 | 72044 | 151087218688 | 140.710938 |
| ciphertext:L2:c2 | 59549 | 187324956672 | 174.459961 |
| ciphertext:L2:c3 | 200 | 943718400 | 0.878906 |
| ciphertext:L3:c2 | 57979 | 243181551616 | 226.480469 |
| ciphertext:L3:c3 | 220 | 1384120320 | 1.289062 |
| ciphertext:L4:c2 | 78406 | 411073249280 | 382.841797 |
| ciphertext:L4:c3 | 220 | 1730150400 | 1.611328 |
| ciphertext:L5:c2 | 27619 | 173763723264 | 161.830078 |
| ciphertext:L5:c3 | 98 | 924844032 | 0.861328 |
| plaintext:L1:c1 | 263 | 275775488 | 0.256836 |
| plaintext:L2:c1 | 31530 | 49592401920 | 46.186523 |
| plaintext:L3:c1 | 549 | 1151336448 | 1.072266 |
| plaintext:L4:c1 | 7656 | 20069744640 | 18.691406 |
| plaintext:L5:c1 | 23791 | 74840014848 | 69.700195 |

## 2. 原始编译计划的level分布

| 类型 / level / 分量 | 数量 | 系数字节 | GiB |
| --- | --- | --- | --- |
| ciphertext:L1:c2 | 72044 | 151087218688 | 140.710938 |
| ciphertext:L2:c2 | 59549 | 187324956672 | 174.459961 |
| ciphertext:L2:c3 | 200 | 943718400 | 0.878906 |
| ciphertext:L3:c2 | 57979 | 243181551616 | 226.480469 |
| ciphertext:L3:c3 | 220 | 1384120320 | 1.289062 |
| ciphertext:L4:c2 | 78406 | 411073249280 | 382.841797 |
| ciphertext:L4:c3 | 220 | 1730150400 | 1.611328 |
| ciphertext:L5:c2 | 27619 | 173763723264 | 161.830078 |
| ciphertext:L5:c3 | 98 | 924844032 | 0.861328 |
| plaintext:L1:c1 | 56 | 58720256 | 0.054688 |
| plaintext:L2:c1 | 30796 | 48437919744 | 45.111328 |
| plaintext:L3:c1 | 24 | 50331648 | 0.046875 |
| plaintext:L4:c1 | 4828 | 12656312320 | 11.787109 |
| plaintext:L5:c1 | 23563 | 74122788864 | 69.032227 |

## 3. 输入输出的实际level

| 数据 | ValueId | level | scale log2 | 分量 | MiB |
| --- | --- | --- | --- | --- | --- |
| input 0 | 0 | 5 | 43 | 2 | 6.000 |
| input 1 | 1 | 5 | 43 | 2 | 6.000 |
| logits.0:32768 | 339073 | 1 | 43 | 2 | 2.000 |
| logits.32768:65536 | 343199 | 1 | 43 | 2 | 2.000 |
| logits.65536:98304 | 347325 | 1 | 43 | 2 | 2.000 |
| logits.98304:131072 | 351451 | 1 | 43 | 2 | 2.000 |
| logits.131072:151936 | 355577 | 1 | 43 | 2 | 2.000 |
| layer_0.kv_cache | 355578 | 2 | 43 | 2 | 3.000 |
| layer_1.kv_cache | 355579 | 2 | 43 | 2 | 3.000 |
| layer_2.kv_cache | 355580 | 2 | 43 | 2 | 3.000 |
| layer_3.kv_cache | 355581 | 2 | 43 | 2 | 3.000 |
| layer_4.kv_cache | 355582 | 2 | 43 | 2 | 3.000 |
| layer_5.kv_cache | 355583 | 2 | 43 | 2 | 3.000 |
| layer_6.kv_cache | 355584 | 2 | 43 | 2 | 3.000 |
| layer_7.kv_cache | 355585 | 2 | 43 | 2 | 3.000 |
| layer_8.kv_cache | 355586 | 2 | 43 | 2 | 3.000 |
| layer_9.kv_cache | 355587 | 2 | 43 | 2 | 3.000 |
| layer_10.kv_cache | 355588 | 2 | 43 | 2 | 3.000 |
| layer_11.kv_cache | 355589 | 2 | 43 | 2 | 3.000 |
| layer_12.kv_cache | 355590 | 2 | 43 | 2 | 3.000 |
| layer_13.kv_cache | 355591 | 2 | 43 | 2 | 3.000 |
| layer_14.kv_cache | 355592 | 2 | 43 | 2 | 3.000 |
| layer_15.kv_cache | 355593 | 2 | 43 | 2 | 3.000 |
| layer_16.kv_cache | 355594 | 2 | 43 | 2 | 3.000 |
| layer_17.kv_cache | 355595 | 2 | 43 | 2 | 3.000 |
| layer_18.kv_cache | 355596 | 2 | 43 | 2 | 3.000 |
| layer_19.kv_cache | 355597 | 2 | 43 | 2 | 3.000 |
| layer_20.kv_cache | 355598 | 2 | 43 | 2 | 3.000 |
| layer_21.kv_cache | 355599 | 2 | 43 | 2 | 3.000 |
| layer_22.kv_cache | 355600 | 2 | 43 | 2 | 3.000 |
| layer_23.kv_cache | 355601 | 2 | 43 | 2 | 3.000 |

## 4. 每种操作的level/scale变化

| 输入→输出类型/level/scale/分量 | 次数 |
| --- | --- |
| add_cc:ciphertext:L1:s43:c2,ciphertext:L1:s43:c2->ciphertext:L1:s43:c2 | 33703 |
| add_cc:ciphertext:L2:s43:c2,ciphertext:L2:s43:c2->ciphertext:L2:s43:c2 | 11322 |
| add_cc:ciphertext:L3:s43:c2,ciphertext:L3:s43:c2->ciphertext:L3:s43:c2 | 26473 |
| add_cc:ciphertext:L4:s43:c2,ciphertext:L4:s43:c2->ciphertext:L4:s43:c2 | 33045 |
| add_cp:ciphertext:L1:s43:c2,plaintext:L1:s43:c1->ciphertext:L1:s43:c2 | 267 |
| add_cp:ciphertext:L2:s43:c2,plaintext:L2:s43:c1->ciphertext:L2:s43:c2 | 152 |
| add_cp:ciphertext:L3:s43:c2,plaintext:L3:s43:c1->ciphertext:L3:s43:c2 | 23 |
| add_cp:ciphertext:L4:s43:c2,plaintext:L4:s43:c1->ciphertext:L4:s43:c2 | 24 |
| boot:ciphertext:L1:s43:c2->ciphertext:L5:s43:c2 | 866 |
| encode:->plaintext:L1:s43:c1 | 263 |
| encode:->plaintext:L2:s43:c1 | 31530 |
| encode:->plaintext:L3:s43:c1 | 549 |
| encode:->plaintext:L4:s43:c1 | 7656 |
| encode:->plaintext:L5:s43:c1 | 23791 |
| mod_switch:ciphertext:L2:s43:c2->ciphertext:L1:s43:c2 | 23 |
| mod_switch:ciphertext:L3:s43:c2->ciphertext:L1:s43:c2 | 24 |
| mod_switch:ciphertext:L3:s43:c2->ciphertext:L2:s43:c2 | 168 |
| mod_switch:ciphertext:L4:s43:c2->ciphertext:L1:s43:c2 | 1 |
| mod_switch:ciphertext:L4:s43:c2->ciphertext:L2:s43:c2 | 73 |
| mod_switch:ciphertext:L4:s43:c2->ciphertext:L3:s43:c2 | 96 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L1:s43:c2 | 120 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L2:s43:c2 | 586 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L3:s43:c2 | 215 |
| mod_switch:ciphertext:L5:s43:c2->ciphertext:L4:s43:c2 | 315 |
| mul_cc:ciphertext:L2:s43:c2,ciphertext:L2:s43:c2->ciphertext:L2:s86:c3 | 200 |
| mul_cc:ciphertext:L3:s43:c2,ciphertext:L3:s43:c2->ciphertext:L3:s86:c3 | 220 |
| mul_cc:ciphertext:L4:s43:c2,ciphertext:L4:s43:c2->ciphertext:L4:s86:c3 | 220 |
| mul_cc:ciphertext:L5:s43:c2,ciphertext:L5:s43:c2->ciphertext:L5:s86:c3 | 98 |
| mul_cp:ciphertext:L2:s43:c2,plaintext:L2:s43:c1->ciphertext:L2:s86:c2 | 33349 |
| mul_cp:ciphertext:L3:s43:c2,plaintext:L3:s43:c1->ciphertext:L3:s86:c2 | 2252 |
| mul_cp:ciphertext:L4:s43:c2,plaintext:L4:s43:c1->ciphertext:L4:s86:c2 | 8739 |
| mul_cp:ciphertext:L5:s43:c2,plaintext:L5:s43:c1->ciphertext:L5:s86:c2 | 25026 |
| negate:ciphertext:L3:s43:c2->ciphertext:L3:s43:c2 | 48 |
| relinearize:ciphertext:L2:s86:c3->ciphertext:L2:s86:c2 | 200 |
| relinearize:ciphertext:L3:s86:c3->ciphertext:L3:s86:c2 | 220 |
| relinearize:ciphertext:L4:s86:c3->ciphertext:L4:s86:c2 | 220 |
| relinearize:ciphertext:L5:s86:c3->ciphertext:L5:s86:c2 | 98 |
| rescale:ciphertext:L2:s86:c2->ciphertext:L1:s43:c2 | 33549 |
| rescale:ciphertext:L3:s86:c2->ciphertext:L2:s43:c2 | 2472 |
| rescale:ciphertext:L4:s86:c2->ciphertext:L3:s43:c2 | 8959 |
| rescale:ciphertext:L5:s86:c2->ciphertext:L4:s43:c2 | 25124 |
| rotate:ciphertext:L1:s43:c2->ciphertext:L1:s43:c2 | 4357 |
| rotate:ciphertext:L2:s43:c2->ciphertext:L2:s43:c2 | 11227 |
| rotate:ciphertext:L3:s43:c2->ciphertext:L3:s43:c2 | 19693 |
| rotate:ciphertext:L4:s43:c2->ciphertext:L4:s43:c2 | 10939 |
| rotate:ciphertext:L5:s43:c2->ciphertext:L5:s43:c2 | 1627 |

## 5. 明文Encode实际level与scale

| level / scale | Encode次数 |
| --- | --- |
| L1:s43 | 263 |
| L2:s43 | 31530 |
| L3:s43 | 549 |
| L4:s43 | 7656 |
| L5:s43 | 23791 |

## 6. 按Block归属的累计数据量

每个Value只计一次。多层融合或无单一Block标签的Value统一进入共享行；这是累计产生的系数载荷，不是该层峰值。

| 归属 | PT数量 | PT GiB | CT数量 | CT GiB |
| --- | --- | --- | --- | --- |
| Block 00 | 3809 | 7.792480 | 11715 | 44.450684 |
| Block 01 | 2549 | 5.856934 | 11089 | 42.827148 |
| Block 02 | 2479 | 6.005859 | 11051 | 44.191406 |
| Block 03 | 2311 | 5.419434 | 11320 | 44.618652 |
| Block 04 | 2321 | 4.813965 | 11319 | 40.771484 |
| Block 05 | 2250 | 4.590332 | 11636 | 42.539062 |
| Block 06 | 2186 | 4.690430 | 11478 | 42.590332 |
| Block 07 | 2465 | 5.604980 | 11173 | 43.029297 |
| Block 08 | 2552 | 6.106445 | 10978 | 43.559570 |
| Block 09 | 2326 | 5.571777 | 11310 | 45.303223 |
| Block 10 | 2312 | 4.898926 | 11328 | 41.436035 |
| Block 11 | 2192 | 4.418457 | 11558 | 41.813477 |
| Block 12 | 2083 | 4.413574 | 11559 | 43.030273 |
| Block 13 | 2378 | 5.343262 | 11258 | 42.567383 |
| Block 14 | 2632 | 6.231445 | 10897 | 43.205566 |
| Block 15 | 2337 | 5.714844 | 11301 | 45.985840 |
| Block 16 | 2298 | 4.993164 | 11340 | 42.174316 |
| Block 17 | 2193 | 4.356934 | 11532 | 41.288574 |
| Block 18 | 2065 | 4.350098 | 11600 | 43.076660 |
| Block 19 | 2304 | 5.123047 | 11331 | 42.587891 |
| Block 20 | 2642 | 6.218750 | 10888 | 42.826172 |
| Block 21 | 2378 | 5.879395 | 11255 | 46.111816 |
| Block 22 | 2632 | 4.175781 | 11047 | 32.019043 |
| Block 23 | 2543 | 3.952148 | 14418 | 43.592773 |
| 共享/接口/多层融合 | 5552 | 9.384766 | 21954 | 65.367188 |

## 7. 所有源码阶段的level和累计载荷

按编译器位置标签归属。融合/共享来源保留联合标签，避免重复计入多层。缓存重新Encode通过相同payload和元数据关联原始来源。

| 源码阶段 | 类型/level/分量 | 数量 | 字节 | GiB |
| --- | --- | --- | --- | --- |
| external_input | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block | plaintext:L2:c1 | 1337 | 2102919168 | 1.958496 |
| token_0-1/layer_00/_decoder_block | plaintext:L1:c1 | 96 | 100663296 | 0.093750 |
| token_0-1/layer_00/_decoder_block | ciphertext:L2:c2 | 2024 | 6366953472 | 5.929688 |
| token_0-1/layer_00/_decoder_block | ciphertext:L1:c2 | 2447 | 5131730944 | 4.779297 |
| token_0-1/layer_00/_decoder_block | ciphertext:L5:c2 | 348 | 2189426688 | 2.039062 |
| token_0-1/layer_00/_decoder_block | plaintext:L5:c1 | 339 | 1066401792 | 0.993164 |
| token_0-1/layer_00/_decoder_block | ciphertext:L4:c2 | 1914 | 10034872320 | 9.345703 |
| token_0-1/layer_00/_decoder_block | ciphertext:L4:c3 | 7 | 55050240 | 0.051270 |
| token_0-1/layer_00/_decoder_block | ciphertext:L2:c3 | 8 | 37748736 | 0.035156 |
| token_0-1/layer_00/_decoder_block | plaintext:L4:c1 | 596 | 1562378240 | 1.455078 |
| token_0-1/layer_00/_decoder_block | ciphertext:L3:c2 | 2532 | 10619977728 | 9.890625 |
| token_0-1/layer_00/_decoder_block | ciphertext:L3:c3 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_00/_decoder_block | plaintext:L3:c1 | 191 | 400556032 | 0.373047 |
| token_0-1/layer_00/_decoder_block | ciphertext:L5:c3 | 3 | 28311552 | 0.026367 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_00/attention | ciphertext:L2:c2 | 14 | 44040192 | 0.041016 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_00/gqa_attention | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_00/gqa_attention | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | ciphertext:L2:c2 | 21 | 66060288 | 0.061523 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | ciphertext:L3:c2 | 16 | 67108864 | 0.062500 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | ciphertext:L4:c2 | 21 | 110100480 | 0.102539 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | plaintext:L5:c1 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | ciphertext:L1:c2 | 13 | 27262976 | 0.025391 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | plaintext:L2:c1 | 9 | 14155776 | 0.013184 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_02/_decoder_block | ciphertext:L3:c2 | 22 | 92274688 | 0.085938 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_02/_decoder_block | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_02/_decoder_block | ciphertext:L1:c2 | 32 | 67108864 | 0.062500 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_02/_decoder_block | ciphertext:L2:c2 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_02/_decoder_block | plaintext:L2:c1 | 15 | 23592960 | 0.021973 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_03/_decoder_block | ciphertext:L1:c2 | 13 | 27262976 | 0.025391 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_03/_decoder_block | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_03/_decoder_block | ciphertext:L2:c2 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_03/_decoder_block | ciphertext:L4:c2 | 34 | 178257920 | 0.166016 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_03/_decoder_block | ciphertext:L5:c2 | 12 | 75497472 | 0.070312 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_03/_decoder_block | plaintext:L5:c1 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_04/_decoder_block | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_04/_decoder_block | ciphertext:L1:c2 | 9 | 18874368 | 0.017578 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_04/_decoder_block | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_04/_decoder_block | ciphertext:L4:c2 | 43 | 225443840 | 0.209961 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_04/_decoder_block | ciphertext:L5:c2 | 17 | 106954752 | 0.099609 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_04/_decoder_block | plaintext:L5:c1 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | ciphertext:L4:c2 | 15 | 78643200 | 0.073242 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | plaintext:L5:c1 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | ciphertext:L5:c2 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | ciphertext:L2:c2 | 14 | 44040192 | 0.041016 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | ciphertext:L1:c2 | 8 | 16777216 | 0.015625 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | plaintext:L2:c1 | 23 | 36175872 | 0.033691 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | ciphertext:L3:c2 | 35 | 146800640 | 0.136719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | ciphertext:L4:c2 | 35 | 183500800 | 0.170898 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | plaintext:L5:c1 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | ciphertext:L1:c2 | 3 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | ciphertext:L3:c2 | 14 | 58720256 | 0.054688 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | ciphertext:L2:c2 | 18 | 56623104 | 0.052734 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | plaintext:L4:c1 | 3 | 7864320 | 0.007324 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | ciphertext:L1:c2 | 14 | 29360128 | 0.027344 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | ciphertext:L5:c2 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | ciphertext:L4:c2 | 40 | 209715200 | 0.195312 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | ciphertext:L3:c2 | 13 | 54525952 | 0.050781 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_08/_decoder_block | ciphertext:L2:c2 | 42 | 132120576 | 0.123047 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_08/_decoder_block | ciphertext:L1:c2 | 28 | 58720256 | 0.054688 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_08/_decoder_block | plaintext:L2:c1 | 16 | 25165824 | 0.023438 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | ciphertext:L1:c2 | 23 | 48234496 | 0.044922 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | ciphertext:L2:c2 | 21 | 66060288 | 0.061523 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | plaintext:L2:c1 | 7 | 11010048 | 0.010254 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | ciphertext:L4:c2 | 20 | 104857600 | 0.097656 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | ciphertext:L3:c2 | 2 | 8388608 | 0.007812 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_10/_decoder_block | ciphertext:L1:c2 | 6 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_10/_decoder_block | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_10/_decoder_block | ciphertext:L2:c2 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_10/_decoder_block | ciphertext:L4:c2 | 40 | 209715200 | 0.195312 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_10/_decoder_block | ciphertext:L5:c2 | 17 | 106954752 | 0.099609 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_10/_decoder_block | plaintext:L5:c1 | 16 | 50331648 | 0.046875 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | ciphertext:L4:c2 | 15 | 78643200 | 0.073242 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | ciphertext:L5:c2 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | ciphertext:L2:c2 | 24 | 75497472 | 0.070312 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | ciphertext:L1:c2 | 11 | 23068672 | 0.021484 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | ciphertext:L3:c2 | 19 | 79691776 | 0.074219 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | ciphertext:L4:c2 | 24 | 125829120 | 0.117188 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | plaintext:L5:c1 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | ciphertext:L2:c2 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | ciphertext:L1:c2 | 17 | 35651584 | 0.033203 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | ciphertext:L3:c2 | 23 | 96468992 | 0.089844 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | ciphertext:L3:c2 | 18 | 75497472 | 0.070312 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | ciphertext:L4:c2 | 42 | 220200960 | 0.205078 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | ciphertext:L5:c2 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | plaintext:L4:c1 | 4 | 10485760 | 0.009766 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | ciphertext:L2:c2 | 12 | 37748736 | 0.035156 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_14/_decoder_block | ciphertext:L2:c2 | 19 | 59768832 | 0.055664 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_14/_decoder_block | ciphertext:L4:c2 | 3 | 15728640 | 0.014648 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_14/_decoder_block | ciphertext:L3:c2 | 10 | 41943040 | 0.039062 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_14/_decoder_block | ciphertext:L1:c2 | 35 | 73400320 | 0.068359 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_14/_decoder_block | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_14/_decoder_block | plaintext:L2:c1 | 18 | 28311552 | 0.026367 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | ciphertext:L1:c2 | 28 | 58720256 | 0.054688 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | ciphertext:L2:c2 | 16 | 50331648 | 0.046875 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | plaintext:L2:c1 | 14 | 22020096 | 0.020508 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | ciphertext:L3:c2 | 13 | 54525952 | 0.050781 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | ciphertext:L5:c2 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | ciphertext:L4:c2 | 8 | 41943040 | 0.039062 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | plaintext:L5:c1 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | ciphertext:L1:c2 | 7 | 14680064 | 0.013672 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | ciphertext:L3:c2 | 11 | 46137344 | 0.042969 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | plaintext:L4:c1 | 4 | 10485760 | 0.009766 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | ciphertext:L4:c2 | 32 | 167772160 | 0.156250 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | plaintext:L5:c1 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | ciphertext:L5:c2 | 15 | 94371840 | 0.087891 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | ciphertext:L5:c2 | 11 | 69206016 | 0.064453 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | plaintext:L5:c1 | 12 | 37748736 | 0.035156 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | ciphertext:L4:c2 | 17 | 89128960 | 0.083008 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | plaintext:L2:c1 | 4 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | ciphertext:L2:c2 | 24 | 75497472 | 0.070312 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | ciphertext:L1:c2 | 14 | 29360128 | 0.027344 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | ciphertext:L3:c2 | 3 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | ciphertext:L5:c2 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | ciphertext:L4:c2 | 13 | 68157440 | 0.063477 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | ciphertext:L2:c2 | 11 | 34603008 | 0.032227 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | ciphertext:L1:c2 | 10 | 20971520 | 0.019531 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | ciphertext:L3:c2 | 41 | 171966464 | 0.160156 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | ciphertext:L3:c2 | 30 | 125829120 | 0.117188 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | ciphertext:L4:c2 | 31 | 162529280 | 0.151367 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | plaintext:L5:c1 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | ciphertext:L5:c2 | 14 | 88080384 | 0.082031 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | plaintext:L4:c1 | 11 | 28835840 | 0.026855 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | ciphertext:L4:c2 | 18 | 94371840 | 0.087891 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | ciphertext:L3:c2 | 7 | 29360128 | 0.027344 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | plaintext:L5:c1 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | plaintext:L2:c1 | 11 | 17301504 | 0.016113 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | ciphertext:L2:c2 | 15 | 47185920 | 0.043945 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | ciphertext:L1:c2 | 27 | 56623104 | 0.052734 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_21/_decoder_block | ciphertext:L4:c2 | 13 | 68157440 | 0.063477 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_21/_decoder_block | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_21/_decoder_block | ciphertext:L3:c2 | 12 | 50331648 | 0.046875 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_21/_decoder_block | plaintext:L2:c1 | 17 | 26738688 | 0.024902 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_21/_decoder_block | ciphertext:L2:c2 | 15 | 47185920 | 0.043945 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_21/_decoder_block | ciphertext:L1:c2 | 28 | 58720256 | 0.054688 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_22/_decoder_block | ciphertext:L1:c2 | 37 | 77594624 | 0.072266 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_22/_decoder_block | ciphertext:L2:c2 | 20 | 62914560 | 0.058594 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_22/_decoder_block | plaintext:L2:c1 | 16 | 25165824 | 0.023438 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_22/_decoder_block | ciphertext:L3:c2 | 11 | 46137344 | 0.042969 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_22/_decoder_block | ciphertext:L4:c2 | 2 | 10485760 | 0.009766 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/_decoder_block | ciphertext:L1:c2 | 4 | 8388608 | 0.007812 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/_decoder_block | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/_decoder_block | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/model_list.23.post_Normal.weight | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/model_list.23.post_Normal.weight | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/model_list.23.post_Normal.weight | ciphertext:L1:c2 | 6 | 12582912 | 0.011719 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/swiglu | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/swiglu | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/swiglu | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/swiglu | ciphertext:L2:c2 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/_decoder_block \| token_shared/layer_shared/final_normal.weight | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/_decoder_block \| token_shared/layer_shared/final_normal.weight | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/_decoder_block \| token_shared/layer_shared/final_normal.weight | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/attention | plaintext:L1:c1 | 49 | 51380224 | 0.047852 |
| token_0-1/layer_00/attention | ciphertext:L3:c2 | 23 | 96468992 | 0.089844 |
| token_0-1/layer_00/attention | plaintext:L3:c1 | 47 | 98566144 | 0.091797 |
| token_0-1/layer_00/attention | ciphertext:L2:c2 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_00/attention | plaintext:L2:c1 | 94 | 147849216 | 0.137695 |
| token_0-1/layer_00/attention | ciphertext:L1:c2 | 4 | 8388608 | 0.007812 |
| token_0-1/layer_00/attention | ciphertext:L5:c2 | 153 | 962592768 | 0.896484 |
| token_0-1/layer_00/attention | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention | ciphertext:L4:c2 | 317 | 1661992960 | 1.547852 |
| token_0-1/layer_00/attention | plaintext:L5:c1 | 152 | 478150656 | 0.445312 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L3:c1 | 47 | 98566144 | 0.091797 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c2 | 54 | 283115520 | 0.263672 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L5:c2 | 29 | 182452224 | 0.169922 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L5:c1 | 17 | 53477376 | 0.049805 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/rope | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/_decoder_block | ciphertext:L3:c2 | 9 | 37748736 | 0.035156 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/_decoder_block | ciphertext:L4:c2 | 5 | 26214400 | 0.024414 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/attention | ciphertext:L2:c2 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/attention | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/attention | ciphertext:L3:c2 | 18 | 75497472 | 0.070312 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/attention | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 4 | 16777216 | 0.015625 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L4:c2 | 57 | 298844160 | 0.278320 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L5:c2 | 18 | 113246208 | 0.105469 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | plaintext:L5:c1 | 14 | 44040192 | 0.041016 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/_decoder_block | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/_decoder_block | ciphertext:L1:c2 | 9 | 18874368 | 0.017578 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/attention | ciphertext:L4:c2 | 12 | 62914560 | 0.058594 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/attention | ciphertext:L3:c2 | 11 | 46137344 | 0.042969 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/attention | ciphertext:L2:c2 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/attention | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/model_list.2.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c2 | 91 | 381681664 | 0.355469 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L4:c2 | 3 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/rope | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/_decoder_block | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/_decoder_block | ciphertext:L1:c2 | 6 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/_decoder_block | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | ciphertext:L2:c2 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | ciphertext:L3:c2 | 2 | 8388608 | 0.007812 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 41 | 171966464 | 0.160156 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | plaintext:L4:c1 | 14 | 36700160 | 0.034180 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c2 | 30 | 157286400 | 0.146484 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L2:c2 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/rope | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/_decoder_block | ciphertext:L4:c2 | 6 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/_decoder_block | ciphertext:L5:c2 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/attention | ciphertext:L1:c2 | 21 | 44040192 | 0.041016 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/attention | ciphertext:L2:c2 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/attention | plaintext:L2:c1 | 4 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L1:c2 | 35 | 73400320 | 0.068359 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L2:c2 | 24 | 75497472 | 0.070312 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L2:c1 | 13 | 20447232 | 0.019043 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L4:c2 | 23 | 120586240 | 0.112305 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/rope | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/_decoder_block | plaintext:L5:c1 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/_decoder_block | ciphertext:L4:c2 | 11 | 57671680 | 0.053711 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/_decoder_block | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/attention | plaintext:L5:c1 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/attention | ciphertext:L4:c2 | 10 | 52428800 | 0.048828 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/attention | ciphertext:L5:c2 | 13 | 81788928 | 0.076172 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/model_list.5.attention.o_weight.weight | plaintext:L2:c1 | 21 | 33030144 | 0.030762 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L2:c2 | 21 | 66060288 | 0.061523 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L1:c2 | 58 | 121634816 | 0.113281 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/rope | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/_decoder_block | ciphertext:L1:c2 | 12 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/_decoder_block | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/attention | ciphertext:L5:c2 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/attention | plaintext:L5:c1 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/attention | ciphertext:L4:c2 | 17 | 89128960 | 0.083008 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L4:c2 | 45 | 235929600 | 0.219727 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/model_list.6.attention.o_weight.weight | plaintext:L5:c1 | 25 | 78643200 | 0.073242 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L5:c2 | 30 | 188743680 | 0.175781 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/rope | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/_decoder_block | ciphertext:L4:c2 | 3 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/_decoder_block | ciphertext:L3:c2 | 11 | 46137344 | 0.042969 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/attention | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/attention | ciphertext:L3:c2 | 30 | 125829120 | 0.117188 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/attention | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L4:c2 | 52 | 272629760 | 0.253906 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/model_list.7.attention.o_weight.weight | plaintext:L5:c1 | 21 | 66060288 | 0.061523 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L5:c2 | 27 | 169869312 | 0.158203 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/_decoder_block | ciphertext:L1:c2 | 7 | 14680064 | 0.013672 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/_decoder_block | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | ciphertext:L3:c2 | 12 | 50331648 | 0.046875 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | plaintext:L4:c1 | 6 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 51 | 213909504 | 0.199219 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/model_list.8.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2621440 | 0.002441 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L2:c2 | 47 | 147849216 | 0.137695 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/_decoder_block | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/_decoder_block | ciphertext:L1:c2 | 11 | 23068672 | 0.021484 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/attention | ciphertext:L2:c2 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/attention | ciphertext:L1:c2 | 7 | 14680064 | 0.013672 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/attention | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/attention | ciphertext:L4:c2 | 19 | 99614720 | 0.092773 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L4:c2 | 32 | 167772160 | 0.156250 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L3:c2 | 57 | 239075328 | 0.222656 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/model_list.9.attention.o_weight.weight | plaintext:L4:c1 | 12 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/rope | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/_decoder_block | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/_decoder_block | plaintext:L5:c1 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/_decoder_block | ciphertext:L4:c2 | 6 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/attention | ciphertext:L2:c2 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/attention | plaintext:L2:c1 | 4 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/attention | ciphertext:L1:c2 | 20 | 41943040 | 0.039062 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L1:c2 | 14 | 29360128 | 0.027344 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/model_list.10.attention.o_weight.weight | plaintext:L2:c1 | 16 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 21 | 66060288 | 0.061523 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L5:c2 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L4:c2 | 40 | 209715200 | 0.195312 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/rope | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/_decoder_block | ciphertext:L4:c2 | 6 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/_decoder_block | ciphertext:L5:c2 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/attention | plaintext:L5:c1 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/attention | ciphertext:L4:c2 | 12 | 62914560 | 0.058594 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/attention | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/attention | ciphertext:L1:c2 | 3 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/attention | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L1:c2 | 57 | 119537664 | 0.111328 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 29 | 91226112 | 0.084961 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/model_list.11.attention.o_weight.weight | plaintext:L2:c1 | 14 | 22020096 | 0.020508 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/rope | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/_decoder_block | ciphertext:L3:c2 | 14 | 58720256 | 0.054688 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/attention | plaintext:L5:c1 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/attention | ciphertext:L4:c2 | 12 | 62914560 | 0.058594 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/attention | ciphertext:L5:c2 | 12 | 75497472 | 0.070312 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L5:c2 | 13 | 81788928 | 0.076172 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 34 | 178257920 | 0.166016 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/model_list.12.attention.o_weight.weight | plaintext:L5:c1 | 25 | 78643200 | 0.073242 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L2:c2 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L1:c2 | 22 | 46137344 | 0.042969 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/model_list.12.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/rope | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/_decoder_block | ciphertext:L1:c2 | 6 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/_decoder_block | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/_decoder_block | ciphertext:L4:c2 | 2 | 10485760 | 0.009766 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/_decoder_block | ciphertext:L3:c2 | 5 | 20971520 | 0.019531 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/attention | ciphertext:L3:c2 | 14 | 58720256 | 0.054688 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/attention | ciphertext:L2:c2 | 19 | 59768832 | 0.055664 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L4:c2 | 39 | 204472320 | 0.190430 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/model_list.13.attention.o_weight.weight | plaintext:L5:c1 | 29 | 91226112 | 0.084961 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L5:c2 | 32 | 201326592 | 0.187500 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/rope | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/_decoder_block | ciphertext:L1:c2 | 8 | 16777216 | 0.015625 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/_decoder_block | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/_decoder_block | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/attention | ciphertext:L4:c2 | 14 | 73400320 | 0.068359 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/attention | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/attention | ciphertext:L2:c2 | 15 | 47185920 | 0.043945 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/attention | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c2 | 7 | 29360128 | 0.027344 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L2:c2 | 56 | 176160768 | 0.164062 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L1:c2 | 37 | 77594624 | 0.072266 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/rope | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/_decoder_block | ciphertext:L1:c2 | 7 | 14680064 | 0.013672 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/_decoder_block | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/attention | ciphertext:L4:c2 | 29 | 152043520 | 0.141602 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/attention | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L3:c2 | 49 | 205520896 | 0.191406 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L4:c2 | 23 | 120586240 | 0.112305 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 17 | 53477376 | 0.049805 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/model_list.15.attention.o_weight.weight | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L1:c2 | 7 | 14680064 | 0.013672 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/rope | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/_decoder_block | ciphertext:L4:c2 | 5 | 26214400 | 0.024414 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/_decoder_block | plaintext:L5:c1 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/_decoder_block | ciphertext:L5:c2 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/attention | plaintext:L2:c1 | 9 | 14155776 | 0.013184 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/attention | ciphertext:L1:c2 | 12 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/attention | ciphertext:L2:c2 | 12 | 37748736 | 0.035156 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L1:c2 | 21 | 44040192 | 0.041016 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L2:c2 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/model_list.16.attention.o_weight.weight | plaintext:L2:c1 | 20 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c2 | 27 | 141557760 | 0.131836 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 19 | 79691776 | 0.074219 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/rope | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/_decoder_block | ciphertext:L4:c2 | 7 | 36700160 | 0.034180 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/_decoder_block | plaintext:L5:c1 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/attention | ciphertext:L4:c2 | 8 | 41943040 | 0.039062 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/attention | plaintext:L5:c1 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/attention | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/attention | ciphertext:L1:c2 | 12 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/attention | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/attention | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L1:c2 | 44 | 92274688 | 0.085938 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/model_list.17.attention.o_weight.weight | plaintext:L2:c1 | 24 | 37748736 | 0.035156 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L2:c2 | 32 | 100663296 | 0.093750 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/rope | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/_decoder_block | ciphertext:L3:c2 | 14 | 58720256 | 0.054688 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/attention | ciphertext:L4:c2 | 14 | 73400320 | 0.068359 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/attention | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/attention | plaintext:L5:c1 | 12 | 37748736 | 0.035156 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L4:c2 | 34 | 178257920 | 0.166016 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/model_list.18.attention.o_weight.weight | plaintext:L5:c1 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L2:c2 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L1:c2 | 35 | 73400320 | 0.068359 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/model_list.18.attention.o_weight.weight | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/rope | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/_decoder_block | ciphertext:L4:c2 | 14 | 73400320 | 0.068359 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | ciphertext:L3:c2 | 8 | 33554432 | 0.031250 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | ciphertext:L2:c2 | 15 | 47185920 | 0.043945 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | ciphertext:L1:c2 | 4 | 8388608 | 0.007812 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | ciphertext:L4:c2 | 2 | 10485760 | 0.009766 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L5:c2 | 26 | 163577856 | 0.152344 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L4:c2 | 42 | 220200960 | 0.205078 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/model_list.19.attention.o_weight.weight | plaintext:L5:c1 | 32 | 100663296 | 0.093750 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/rope | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/_decoder_block | ciphertext:L4:c2 | 13 | 68157440 | 0.063477 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/_decoder_block | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/attention | ciphertext:L3:c2 | 22 | 92274688 | 0.085938 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/attention | plaintext:L4:c1 | 3 | 7864320 | 0.007324 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/attention | ciphertext:L4:c2 | 3 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/attention | ciphertext:L2:c2 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L2:c2 | 18 | 56623104 | 0.052734 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L5:c2 | 15 | 94371840 | 0.087891 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L4:c2 | 8 | 41943040 | 0.039062 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 28 | 117440512 | 0.109375 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L4:c3 | 2 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | plaintext:L5:c1 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/_decoder_block | ciphertext:L1:c2 | 8 | 16777216 | 0.015625 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/_decoder_block | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/_decoder_block | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/attention | ciphertext:L4:c2 | 16 | 83886080 | 0.078125 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/attention | plaintext:L4:c1 | 8 | 20971520 | 0.019531 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/attention | ciphertext:L3:c2 | 12 | 50331648 | 0.046875 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 52 | 218103808 | 0.203125 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c2 | 7 | 36700160 | 0.034180 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L2:c2 | 27 | 84934656 | 0.079102 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L1:c2 | 8 | 16777216 | 0.015625 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2621440 | 0.002441 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/_decoder_block | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/_decoder_block | ciphertext:L1:c2 | 12 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/attention | plaintext:L2:c1 | 8 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/attention | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/attention | ciphertext:L2:c2 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L2:c2 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L1:c2 | 31 | 65011712 | 0.060547 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | plaintext:L2:c1 | 13 | 20447232 | 0.019043 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L4:c2 | 12 | 62914560 | 0.058594 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 33 | 138412032 | 0.128906 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | plaintext:L4:c1 | 8 | 20971520 | 0.019531 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/rope | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/attention | plaintext:L2:c1 | 8 | 12582912 | 0.011719 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/attention | ciphertext:L1:c2 | 12 | 25165824 | 0.023438 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/attention | ciphertext:L2:c2 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L1:c2 | 25 | 52428800 | 0.048828 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/model_list.23.attention.o_weight.weight | plaintext:L2:c1 | 18 | 28311552 | 0.026367 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/rope | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | ciphertext:L3:c2 | 20 | 83886080 | 0.078125 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | ciphertext:L5:c2 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | ciphertext:L4:c2 | 20 | 104857600 | 0.097656 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | ciphertext:L1:c2 | 19 | 39845888 | 0.037109 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | plaintext:L5:c1 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | ciphertext:L2:c2 | 14 | 44040192 | 0.041016 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | plaintext:L2:c1 | 10 | 15728640 | 0.014648 |
| token_0-1/layer_00/gqa_attention | plaintext:L1:c1 | 72 | 75497472 | 0.070312 |
| token_0-1/layer_00/gqa_attention | plaintext:L4:c1 | 1 | 2621440 | 0.002441 |
| token_0-1/layer_00/gqa_attention | plaintext:L2:c1 | 49 | 77070336 | 0.071777 |
| token_0-1/layer_00/gqa_attention | ciphertext:L2:c2 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/gqa_attention | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/gqa_attention | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/gqa_attention | plaintext:L5:c1 | 47 | 147849216 | 0.137695 |
| token_0-1/layer_00/gqa_attention | ciphertext:L5:c2 | 49 | 308281344 | 0.287109 |
| token_0-1/layer_00/gqa_attention | ciphertext:L4:c2 | 95 | 498073600 | 0.463867 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/_decoder_block | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/gqa_attention | ciphertext:L3:c2 | 7 | 29360128 | 0.027344 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/gqa_attention | ciphertext:L4:c2 | 5 | 26214400 | 0.024414 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/gqa_attention | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/gqa_attention | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/gqa_attention | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/gqa_attention | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_02/_decoder_block | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_02/gqa_attention | ciphertext:L4:c2 | 7 | 36700160 | 0.034180 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_02/gqa_attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_02/gqa_attention | ciphertext:L1:c2 | 3 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_02/gqa_attention | ciphertext:L3:c2 | 8 | 33554432 | 0.031250 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_03/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_03/gqa_attention | ciphertext:L4:c2 | 10 | 52428800 | 0.048828 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_03/gqa_attention | ciphertext:L3:c2 | 8 | 33554432 | 0.031250 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_03/gqa_attention | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_04/_decoder_block | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_04/gqa_attention | plaintext:L2:c1 | 8 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_04/gqa_attention | ciphertext:L1:c2 | 7 | 14680064 | 0.013672 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_04/gqa_attention | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/_decoder_block | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/_decoder_block | plaintext:L4:c1 | 23 | 60293120 | 0.056152 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/_decoder_block | plaintext:L2:c1 | 23 | 36175872 | 0.033691 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/_decoder_block | plaintext:L3:c1 | 22 | 46137344 | 0.042969 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/gqa_attention | ciphertext:L4:c2 | 5 | 26214400 | 0.024414 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/gqa_attention | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/gqa_attention | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/gqa_attention | ciphertext:L1:c2 | 9 | 18874368 | 0.017578 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/gqa_attention | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_06/_decoder_block | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_06/gqa_attention | ciphertext:L4:c2 | 13 | 68157440 | 0.063477 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_06/gqa_attention | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_06/gqa_attention | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_07/_decoder_block | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_07/gqa_attention | ciphertext:L2:c2 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_07/gqa_attention | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_07/gqa_attention | ciphertext:L4:c2 | 2 | 10485760 | 0.009766 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_07/gqa_attention | plaintext:L5:c1 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_08/_decoder_block | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_08/gqa_attention | ciphertext:L3:c2 | 9 | 37748736 | 0.035156 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_08/gqa_attention | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_08/gqa_attention | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_08/gqa_attention | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_09/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_09/gqa_attention | ciphertext:L4:c2 | 10 | 52428800 | 0.048828 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_09/gqa_attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_09/gqa_attention | ciphertext:L3:c2 | 7 | 29360128 | 0.027344 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_09/gqa_attention | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_10/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_10/gqa_attention | ciphertext:L1:c2 | 10 | 20971520 | 0.019531 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_10/gqa_attention | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_10/gqa_attention | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_11/_decoder_block | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_11/gqa_attention | ciphertext:L2:c2 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_11/gqa_attention | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_11/gqa_attention | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_12/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_12/gqa_attention | ciphertext:L4:c2 | 12 | 62914560 | 0.058594 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_12/gqa_attention | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_12/gqa_attention | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/gqa_attention | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/gqa_attention | ciphertext:L1:c2 | 4 | 8388608 | 0.007812 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/gqa_attention | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/gqa_attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/gqa_attention | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/_decoder_block | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/gqa_attention | ciphertext:L4:c2 | 8 | 41943040 | 0.039062 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/gqa_attention | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/gqa_attention | ciphertext:L3:c2 | 4 | 16777216 | 0.015625 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/gqa_attention | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/gqa_attention | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/gqa_attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_15/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_15/gqa_attention | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_15/gqa_attention | ciphertext:L3:c2 | 7 | 29360128 | 0.027344 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_15/gqa_attention | plaintext:L4:c1 | 4 | 10485760 | 0.009766 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_15/gqa_attention | ciphertext:L2:c2 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_16/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_16/gqa_attention | ciphertext:L1:c2 | 9 | 18874368 | 0.017578 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_16/gqa_attention | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_16/gqa_attention | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_17/_decoder_block | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_17/gqa_attention | ciphertext:L1:c2 | 10 | 20971520 | 0.019531 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_17/gqa_attention | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_17/gqa_attention | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_18/_decoder_block | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_18/gqa_attention | ciphertext:L5:c2 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_18/gqa_attention | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_18/gqa_attention | ciphertext:L4:c2 | 9 | 47185920 | 0.043945 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_19/_decoder_block | ciphertext:L4:c2 | 1 | 5242880 | 0.004883 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_19/gqa_attention | plaintext:L5:c1 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_19/gqa_attention | ciphertext:L4:c2 | 9 | 47185920 | 0.043945 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_19/gqa_attention | ciphertext:L5:c2 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_20/_decoder_block | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_20/gqa_attention | ciphertext:L3:c2 | 16 | 67108864 | 0.062500 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_20/gqa_attention | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_21/_decoder_block | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_21/gqa_attention | ciphertext:L3:c2 | 9 | 37748736 | 0.035156 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_21/gqa_attention | ciphertext:L4:c2 | 5 | 26214400 | 0.024414 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_21/gqa_attention | plaintext:L4:c1 | 3 | 7864320 | 0.007324 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_21/gqa_attention | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/_decoder_block | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/_decoder_block | plaintext:L2:c1 | 46 | 72351744 | 0.067383 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/gqa_attention | ciphertext:L1:c2 | 12 | 25165824 | 0.023438 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/gqa_attention | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/gqa_attention | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/gqa_attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_23/gqa_attention | ciphertext:L1:c2 | 10 | 20971520 | 0.019531 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_23/gqa_attention | ciphertext:L2:c2 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_23/gqa_attention | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c3 | 2 | 15728640 | 0.014648 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L4:c2 | 1025 | 5373952000 | 5.004883 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L3:c2 | 5 | 20971520 | 0.019531 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L3:c1 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L2:c1 | 48 | 75497472 | 0.070312 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L1:c2 | 4 | 8388608 | 0.007812 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L3:c3 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | plaintext:L5:c1 | 505 | 1588592640 | 1.479492 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | ciphertext:L5:c2 | 492 | 3095396352 | 2.882812 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight \| token_0-1/layer_22/_decoder_block | plaintext:L2:c1 | 46 | 72351744 | 0.067383 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight \| token_0-1/layer_22/_decoder_block | plaintext:L3:c1 | 46 | 96468992 | 0.089844 |
| token_0-1/layer_00/rope | plaintext:L3:c1 | 47 | 98566144 | 0.091797 |
| token_0-1/layer_00/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/rope | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/rope | plaintext:L2:c1 | 47 | 73924608 | 0.068848 |
| token_0-1/layer_00/rope | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_00/rope | ciphertext:L4:c2 | 22 | 115343360 | 0.107422 |
| token_0-1/layer_00/rope | ciphertext:L5:c2 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_00/rope | plaintext:L5:c1 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_00/rope \| token_0-1/layer_01/rope | ciphertext:L3:c2 | 11 | 46137344 | 0.042969 |
| token_0-1/layer_00/rope \| token_0-1/layer_02/rope | ciphertext:L4:c2 | 11 | 57671680 | 0.053711 |
| token_0-1/layer_00/rope \| token_0-1/layer_03/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/rope \| token_0-1/layer_03/rope | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_03/rope | ciphertext:L5:c2 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_00/rope \| token_0-1/layer_03/rope | ciphertext:L4:c2 | 7 | 36700160 | 0.034180 |
| token_0-1/layer_00/rope \| token_0-1/layer_04/rope | ciphertext:L1:c2 | 6 | 12582912 | 0.011719 |
| token_0-1/layer_00/rope \| token_0-1/layer_04/rope | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/rope \| token_0-1/layer_04/rope | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/rope \| token_0-1/layer_05/rope | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_00/rope \| token_0-1/layer_05/rope | plaintext:L5:c1 | 5 | 15728640 | 0.014648 |
| token_0-1/layer_00/rope \| token_0-1/layer_05/rope | ciphertext:L4:c2 | 2 | 10485760 | 0.009766 |
| token_0-1/layer_00/rope \| token_0-1/layer_06/rope | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_00/rope \| token_0-1/layer_06/rope | plaintext:L5:c1 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/rope \| token_0-1/layer_06/rope | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/rope \| token_0-1/layer_07/rope | ciphertext:L3:c2 | 3 | 12582912 | 0.011719 |
| token_0-1/layer_00/rope \| token_0-1/layer_07/rope | ciphertext:L2:c2 | 8 | 25165824 | 0.023438 |
| token_0-1/layer_00/rope \| token_0-1/layer_08/rope | ciphertext:L3:c2 | 6 | 25165824 | 0.023438 |
| token_0-1/layer_00/rope \| token_0-1/layer_08/rope | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/rope \| token_0-1/layer_08/rope | plaintext:L4:c1 | 1 | 2621440 | 0.002441 |
| token_0-1/layer_00/rope \| token_0-1/layer_09/rope | ciphertext:L4:c2 | 10 | 52428800 | 0.048828 |
| token_0-1/layer_00/rope \| token_0-1/layer_09/rope | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_10/rope | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/rope \| token_0-1/layer_10/rope | ciphertext:L2:c2 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/rope \| token_0-1/layer_10/rope | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/rope \| token_0-1/layer_11/rope | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/rope \| token_0-1/layer_11/rope | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/rope \| token_0-1/layer_11/rope | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/rope \| token_0-1/layer_12/rope | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_00/rope \| token_0-1/layer_12/rope | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_00/rope \| token_0-1/layer_12/rope | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/rope \| token_0-1/layer_13/rope | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_00/rope \| token_0-1/layer_13/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/rope \| token_0-1/layer_13/rope | ciphertext:L1:c2 | 3 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_14/rope | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_00/rope \| token_0-1/layer_14/rope | ciphertext:L4:c2 | 7 | 36700160 | 0.034180 |
| token_0-1/layer_00/rope \| token_0-1/layer_14/rope | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_00/rope \| token_0-1/layer_15/rope | ciphertext:L4:c2 | 8 | 41943040 | 0.039062 |
| token_0-1/layer_00/rope \| token_0-1/layer_15/rope | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_15/rope | ciphertext:L3:c2 | 2 | 8388608 | 0.007812 |
| token_0-1/layer_00/rope \| token_0-1/layer_16/rope | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_16/rope | plaintext:L2:c1 | 4 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_16/rope | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/rope \| token_0-1/layer_17/rope | ciphertext:L1:c2 | 6 | 12582912 | 0.011719 |
| token_0-1/layer_00/rope \| token_0-1/layer_17/rope | plaintext:L2:c1 | 2 | 3145728 | 0.002930 |
| token_0-1/layer_00/rope \| token_0-1/layer_17/rope | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/rope \| token_0-1/layer_18/rope | plaintext:L5:c1 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_18/rope | ciphertext:L4:c2 | 8 | 41943040 | 0.039062 |
| token_0-1/layer_00/rope \| token_0-1/layer_18/rope | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_19/rope | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_00/rope \| token_0-1/layer_19/rope | plaintext:L5:c1 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/rope \| token_0-1/layer_19/rope | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_00/rope \| token_0-1/layer_20/rope | ciphertext:L3:c2 | 11 | 46137344 | 0.042969 |
| token_0-1/layer_00/rope \| token_0-1/layer_21/rope | ciphertext:L3:c2 | 6 | 25165824 | 0.023438 |
| token_0-1/layer_00/rope \| token_0-1/layer_21/rope | ciphertext:L4:c2 | 3 | 15728640 | 0.014648 |
| token_0-1/layer_00/rope \| token_0-1/layer_21/rope | plaintext:L4:c1 | 4 | 10485760 | 0.009766 |
| token_0-1/layer_00/rope \| token_0-1/layer_22/rope | plaintext:L2:c1 | 4 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_22/rope | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_00/rope \| token_0-1/layer_22/rope | ciphertext:L2:c2 | 2 | 6291456 | 0.005859 |
| token_0-1/layer_00/rope \| token_0-1/layer_23/rope | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_00/rope \| token_0-1/layer_23/rope | plaintext:L2:c1 | 3 | 4718592 | 0.004395 |
| token_0-1/layer_00/rope \| token_0-1/layer_23/rope | ciphertext:L1:c2 | 5 | 10485760 | 0.009766 |
| token_0-1/layer_01/_decoder_block | ciphertext:L3:c2 | 995 | 4173332480 | 3.886719 |
| token_0-1/layer_01/_decoder_block | ciphertext:L4:c2 | 2730 | 14313062400 | 13.330078 |
| token_0-1/layer_01/_decoder_block | ciphertext:L2:c2 | 1545 | 4860149760 | 4.526367 |
| token_0-1/layer_01/_decoder_block | ciphertext:L1:c2 | 2213 | 4640997376 | 4.322266 |
| token_0-1/layer_01/_decoder_block | ciphertext:L5:c2 | 1058 | 6656360448 | 6.199219 |
| token_0-1/layer_01/_decoder_block | ciphertext:L4:c3 | 4 | 31457280 | 0.029297 |
| token_0-1/layer_01/_decoder_block | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_01/_decoder_block | ciphertext:L3:c3 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_01/_decoder_block | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_01/_decoder_block | plaintext:L2:c1 | 1015 | 1596456960 | 1.486816 |
| token_0-1/layer_01/_decoder_block | plaintext:L5:c1 | 951 | 2991587328 | 2.786133 |
| token_0-1/layer_01/_decoder_block | plaintext:L4:c1 | 243 | 637009920 | 0.593262 |
| token_0-1/layer_01/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_01/attention | ciphertext:L3:c2 | 317 | 1329594368 | 1.238281 |
| token_0-1/layer_01/attention | ciphertext:L4:c2 | 49 | 256901120 | 0.239258 |
| token_0-1/layer_01/attention | ciphertext:L2:c2 | 137 | 430964736 | 0.401367 |
| token_0-1/layer_01/attention | ciphertext:L1:c2 | 102 | 213909504 | 0.199219 |
| token_0-1/layer_01/attention | ciphertext:L5:c2 | 15 | 94371840 | 0.087891 |
| token_0-1/layer_01/attention | ciphertext:L5:c3 | 2 | 18874368 | 0.017578 |
| token_0-1/layer_01/attention | ciphertext:L3:c3 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_01/attention | ciphertext:L2:c3 | 2 | 9437184 | 0.008789 |
| token_0-1/layer_01/attention | ciphertext:L4:c3 | 2 | 15728640 | 0.014648 |
| token_0-1/layer_01/gqa_attention | ciphertext:L3:c2 | 66 | 276824064 | 0.257812 |
| token_0-1/layer_01/gqa_attention | ciphertext:L4:c2 | 55 | 288358400 | 0.268555 |
| token_0-1/layer_01/gqa_attention | ciphertext:L1:c2 | 2 | 4194304 | 0.003906 |
| token_0-1/layer_01/gqa_attention | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_01/gqa_attention | plaintext:L5:c1 | 25 | 78643200 | 0.073242 |
| token_0-1/layer_01/gqa_attention | ciphertext:L5:c2 | 23 | 144703488 | 0.134766 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c2 | 297 | 1245708288 | 1.160156 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L4:c2 | 626 | 3282042880 | 3.056641 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L2:c2 | 381 | 1198522368 | 1.116211 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L1:c2 | 92 | 192937984 | 0.179688 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L5:c2 | 329 | 2069889024 | 1.927734 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L3:c3 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | plaintext:L5:c1 | 312 | 981467136 | 0.914062 |
| token_0-1/layer_01/rope | ciphertext:L3:c2 | 28 | 117440512 | 0.109375 |
| token_0-1/layer_02/_decoder_block | ciphertext:L5:c2 | 1584 | 9965666304 | 9.281250 |
| token_0-1/layer_02/_decoder_block | plaintext:L5:c1 | 1561 | 4910481408 | 4.573242 |
| token_0-1/layer_02/_decoder_block | ciphertext:L4:c2 | 3265 | 17118003200 | 15.942383 |
| token_0-1/layer_02/_decoder_block | ciphertext:L3:c2 | 349 | 1463812096 | 1.363281 |
| token_0-1/layer_02/_decoder_block | plaintext:L4:c1 | 1 | 2621440 | 0.002441 |
| token_0-1/layer_02/_decoder_block | ciphertext:L2:c2 | 1239 | 3897556992 | 3.629883 |
| token_0-1/layer_02/_decoder_block | ciphertext:L1:c2 | 1813 | 3802136576 | 3.541016 |
| token_0-1/layer_02/_decoder_block | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_02/_decoder_block | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_02/_decoder_block | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_02/_decoder_block | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_02/_decoder_block | plaintext:L2:c1 | 833 | 1310195712 | 1.220215 |
| token_0-1/layer_02/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_02/attention | ciphertext:L4:c2 | 93 | 487587840 | 0.454102 |
| token_0-1/layer_02/attention | plaintext:L5:c1 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_02/attention | ciphertext:L5:c2 | 21 | 132120576 | 0.123047 |
| token_0-1/layer_02/attention | ciphertext:L3:c2 | 186 | 780140544 | 0.726562 |
| token_0-1/layer_02/attention | ciphertext:L2:c2 | 251 | 789577728 | 0.735352 |
| token_0-1/layer_02/attention | ciphertext:L1:c2 | 59 | 123731968 | 0.115234 |
| token_0-1/layer_02/attention | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_02/attention | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_02/attention | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_02/attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_02/gqa_attention | ciphertext:L4:c2 | 66 | 346030080 | 0.322266 |
| token_0-1/layer_02/gqa_attention | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_02/gqa_attention | ciphertext:L1:c2 | 3 | 6291456 | 0.005859 |
| token_0-1/layer_02/gqa_attention | ciphertext:L3:c2 | 92 | 385875968 | 0.359375 |
| token_0-1/layer_02/gqa_attention | plaintext:L4:c1 | 4 | 10485760 | 0.009766 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L4:c2 | 276 | 1447034880 | 1.347656 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L5:c2 | 49 | 308281344 | 0.287109 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | plaintext:L4:c1 | 62 | 162529280 | 0.151367 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c2 | 986 | 4135583744 | 3.851562 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L2:c2 | 488 | 1535115264 | 1.429688 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L1:c2 | 172 | 360710144 | 0.335938 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L5:c3 | 2 | 18874368 | 0.017578 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L3:c3 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L2:c3 | 2 | 9437184 | 0.008789 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | ciphertext:L4:c3 | 2 | 15728640 | 0.014648 |
| token_0-1/layer_02/rope | ciphertext:L4:c2 | 24 | 125829120 | 0.117188 |
| token_0-1/layer_02/rope | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_03/_decoder_block | plaintext:L5:c1 | 1180 | 3711959040 | 3.457031 |
| token_0-1/layer_03/_decoder_block | ciphertext:L4:c2 | 2709 | 14202961920 | 13.227539 |
| token_0-1/layer_03/_decoder_block | ciphertext:L5:c2 | 1245 | 7832862720 | 7.294922 |
| token_0-1/layer_03/_decoder_block | ciphertext:L2:c2 | 1592 | 5007998976 | 4.664062 |
| token_0-1/layer_03/_decoder_block | plaintext:L2:c1 | 684 | 1075838976 | 1.001953 |
| token_0-1/layer_03/_decoder_block | ciphertext:L1:c2 | 1693 | 3550478336 | 3.306641 |
| token_0-1/layer_03/_decoder_block | plaintext:L4:c1 | 73 | 191365120 | 0.178223 |
| token_0-1/layer_03/_decoder_block | ciphertext:L3:c2 | 1465 | 6144655360 | 5.722656 |
| token_0-1/layer_03/_decoder_block | ciphertext:L5:c3 | 3 | 28311552 | 0.026367 |
| token_0-1/layer_03/_decoder_block | ciphertext:L3:c3 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_03/_decoder_block | ciphertext:L2:c3 | 3 | 14155776 | 0.013184 |
| token_0-1/layer_03/_decoder_block | ciphertext:L4:c3 | 5 | 39321600 | 0.036621 |
| token_0-1/layer_03/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_03/attention | ciphertext:L5:c2 | 15 | 94371840 | 0.087891 |
| token_0-1/layer_03/attention | ciphertext:L4:c2 | 25 | 131072000 | 0.122070 |
| token_0-1/layer_03/attention | plaintext:L5:c1 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_03/attention | ciphertext:L1:c2 | 287 | 601882624 | 0.560547 |
| token_0-1/layer_03/attention | plaintext:L2:c1 | 138 | 217055232 | 0.202148 |
| token_0-1/layer_03/attention | ciphertext:L2:c2 | 146 | 459276288 | 0.427734 |
| token_0-1/layer_03/attention | ciphertext:L3:c2 | 4 | 16777216 | 0.015625 |
| token_0-1/layer_03/attention | ciphertext:L2:c3 | 3 | 14155776 | 0.013184 |
| token_0-1/layer_03/gqa_attention | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_03/gqa_attention | ciphertext:L5:c2 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_03/gqa_attention | ciphertext:L4:c2 | 78 | 408944640 | 0.380859 |
| token_0-1/layer_03/gqa_attention | ciphertext:L3:c2 | 69 | 289406976 | 0.269531 |
| token_0-1/layer_03/gqa_attention | plaintext:L4:c1 | 26 | 68157440 | 0.063477 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c2 | 853 | 4472176640 | 4.165039 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L5:c2 | 89 | 559939584 | 0.521484 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | plaintext:L4:c1 | 191 | 500695040 | 0.466309 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c2 | 683 | 2864709632 | 2.667969 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L2:c2 | 249 | 783286272 | 0.729492 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L1:c2 | 54 | 113246208 | 0.105469 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L3:c3 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_03/rope | ciphertext:L4:c2 | 17 | 89128960 | 0.083008 |
| token_0-1/layer_03/rope | ciphertext:L3:c2 | 3 | 12582912 | 0.011719 |
| token_0-1/layer_03/rope | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_03/rope | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_03/rope | ciphertext:L5:c2 | 4 | 25165824 | 0.023438 |
| token_0-1/layer_04/_decoder_block | ciphertext:L1:c2 | 1513 | 3172990976 | 2.955078 |
| token_0-1/layer_04/_decoder_block | ciphertext:L2:c2 | 1702 | 5354029056 | 4.986328 |
| token_0-1/layer_04/_decoder_block | plaintext:L2:c1 | 570 | 896532480 | 0.834961 |
| token_0-1/layer_04/_decoder_block | ciphertext:L4:c2 | 2713 | 14223933440 | 13.247070 |
| token_0-1/layer_04/_decoder_block | ciphertext:L5:c2 | 897 | 5643436032 | 5.255859 |
| token_0-1/layer_04/_decoder_block | plaintext:L4:c1 | 312 | 817889280 | 0.761719 |
| token_0-1/layer_04/_decoder_block | ciphertext:L3:c2 | 2276 | 9546235904 | 8.890625 |
| token_0-1/layer_04/_decoder_block | ciphertext:L4:c3 | 8 | 62914560 | 0.058594 |
| token_0-1/layer_04/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_04/_decoder_block | ciphertext:L3:c3 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_04/_decoder_block | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_04/_decoder_block | plaintext:L5:c1 | 758 | 2384461824 | 2.220703 |
| token_0-1/layer_04/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_04/attention | ciphertext:L1:c2 | 317 | 664797184 | 0.619141 |
| token_0-1/layer_04/attention | plaintext:L2:c1 | 158 | 248512512 | 0.231445 |
| token_0-1/layer_04/attention | ciphertext:L2:c2 | 153 | 481296384 | 0.448242 |
| token_0-1/layer_04/gqa_attention | ciphertext:L2:c2 | 46 | 144703488 | 0.134766 |
| token_0-1/layer_04/gqa_attention | plaintext:L2:c1 | 39 | 61341696 | 0.057129 |
| token_0-1/layer_04/gqa_attention | ciphertext:L1:c2 | 87 | 182452224 | 0.169922 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | plaintext:L2:c1 | 475 | 747110400 | 0.695801 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L2:c2 | 480 | 1509949440 | 1.406250 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L1:c2 | 982 | 2059403264 | 1.917969 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c2 | 9 | 37748736 | 0.035156 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L5:c2 | 18 | 113246208 | 0.105469 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L4:c2 | 76 | 398458880 | 0.371094 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_04/rope | ciphertext:L2:c2 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_04/rope | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_0-1/layer_04/rope | plaintext:L2:c1 | 7 | 11010048 | 0.010254 |
| token_0-1/layer_05/_decoder_block | ciphertext:L2:c2 | 1761 | 5539627008 | 5.159180 |
| token_0-1/layer_05/_decoder_block | plaintext:L2:c1 | 651 | 1023934464 | 0.953613 |
| token_0-1/layer_05/_decoder_block | ciphertext:L1:c2 | 1604 | 3363831808 | 3.132812 |
| token_0-1/layer_05/_decoder_block | ciphertext:L3:c2 | 2974 | 12473860096 | 11.617188 |
| token_0-1/layer_05/_decoder_block | plaintext:L3:c1 | 27 | 56623104 | 0.052734 |
| token_0-1/layer_05/_decoder_block | ciphertext:L2:c3 | 12 | 56623104 | 0.052734 |
| token_0-1/layer_05/_decoder_block | ciphertext:L5:c2 | 501 | 3152019456 | 2.935547 |
| token_0-1/layer_05/_decoder_block | plaintext:L5:c1 | 286 | 899678208 | 0.837891 |
| token_0-1/layer_05/_decoder_block | ciphertext:L4:c2 | 2593 | 13594787840 | 12.661133 |
| token_0-1/layer_05/_decoder_block | ciphertext:L4:c3 | 10 | 78643200 | 0.073242 |
| token_0-1/layer_05/_decoder_block | plaintext:L4:c1 | 578 | 1515192320 | 1.411133 |
| token_0-1/layer_05/_decoder_block | ciphertext:L3:c3 | 10 | 62914560 | 0.058594 |
| token_0-1/layer_05/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_05/_decoder_block \| token_0-1/layer_23/swiglu | plaintext:L3:c1 | 44 | 92274688 | 0.085938 |
| token_0-1/layer_05/_decoder_block \| token_0-1/layer_23/swiglu | plaintext:L2:c1 | 22 | 34603008 | 0.032227 |
| token_0-1/layer_05/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_05/attention | ciphertext:L1:c2 | 22 | 46137344 | 0.042969 |
| token_0-1/layer_05/attention | plaintext:L2:c1 | 10 | 15728640 | 0.014648 |
| token_0-1/layer_05/attention | ciphertext:L2:c2 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_05/attention | ciphertext:L5:c2 | 140 | 880803840 | 0.820312 |
| token_0-1/layer_05/attention | ciphertext:L4:c2 | 304 | 1593835520 | 1.484375 |
| token_0-1/layer_05/attention | plaintext:L5:c1 | 142 | 446693376 | 0.416016 |
| token_0-1/layer_05/gqa_attention | ciphertext:L4:c2 | 36 | 188743680 | 0.175781 |
| token_0-1/layer_05/gqa_attention | plaintext:L5:c1 | 18 | 56623104 | 0.052734 |
| token_0-1/layer_05/gqa_attention | ciphertext:L5:c2 | 15 | 94371840 | 0.087891 |
| token_0-1/layer_05/gqa_attention | plaintext:L2:c1 | 28 | 44040192 | 0.041016 |
| token_0-1/layer_05/gqa_attention | ciphertext:L2:c2 | 25 | 78643200 | 0.073242 |
| token_0-1/layer_05/gqa_attention | ciphertext:L1:c2 | 50 | 104857600 | 0.097656 |
| token_0-1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L1:c2 | 938 | 1967128576 | 1.832031 |
| token_0-1/layer_05/model_list.5.attention.o_weight.weight | plaintext:L2:c1 | 464 | 729808896 | 0.679688 |
| token_0-1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L2:c2 | 480 | 1509949440 | 1.406250 |
| token_0-1/layer_05/model_list.5.attention.o_weight.weight | plaintext:L5:c1 | 40 | 125829120 | 0.117188 |
| token_0-1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L5:c2 | 40 | 251658240 | 0.234375 |
| token_0-1/layer_05/model_list.5.attention.o_weight.weight | ciphertext:L4:c2 | 83 | 435159040 | 0.405273 |
| token_0-1/layer_05/rope | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_05/rope | ciphertext:L4:c2 | 17 | 89128960 | 0.083008 |
| token_0-1/layer_05/rope | ciphertext:L5:c2 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_06/_decoder_block | ciphertext:L4:c2 | 1690 | 8860467200 | 8.251953 |
| token_0-1/layer_06/_decoder_block | ciphertext:L2:c2 | 2449 | 7703887872 | 7.174805 |
| token_0-1/layer_06/_decoder_block | ciphertext:L3:c2 | 2451 | 10280239104 | 9.574219 |
| token_0-1/layer_06/_decoder_block | ciphertext:L1:c2 | 2435 | 5106565120 | 4.755859 |
| token_0-1/layer_06/_decoder_block | ciphertext:L5:c2 | 251 | 1579155456 | 1.470703 |
| token_0-1/layer_06/_decoder_block | ciphertext:L4:c3 | 7 | 55050240 | 0.051270 |
| token_0-1/layer_06/_decoder_block | ciphertext:L5:c3 | 2 | 18874368 | 0.017578 |
| token_0-1/layer_06/_decoder_block | ciphertext:L3:c3 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_06/_decoder_block | ciphertext:L2:c3 | 6 | 28311552 | 0.026367 |
| token_0-1/layer_06/_decoder_block | plaintext:L2:c1 | 1026 | 1613758464 | 1.502930 |
| token_0-1/layer_06/_decoder_block | plaintext:L4:c1 | 378 | 990904320 | 0.922852 |
| token_0-1/layer_06/_decoder_block | plaintext:L5:c1 | 66 | 207618048 | 0.193359 |
| token_0-1/layer_06/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_06/attention | ciphertext:L3:c2 | 29 | 121634816 | 0.113281 |
| token_0-1/layer_06/attention | ciphertext:L4:c2 | 303 | 1588592640 | 1.479492 |
| token_0-1/layer_06/attention | plaintext:L3:c1 | 23 | 48234496 | 0.044922 |
| token_0-1/layer_06/attention | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_06/attention | plaintext:L5:c1 | 143 | 449839104 | 0.418945 |
| token_0-1/layer_06/attention | ciphertext:L5:c2 | 145 | 912261120 | 0.849609 |
| token_0-1/layer_06/gqa_attention | plaintext:L5:c1 | 46 | 144703488 | 0.134766 |
| token_0-1/layer_06/gqa_attention | ciphertext:L4:c2 | 81 | 424673280 | 0.395508 |
| token_0-1/layer_06/gqa_attention | ciphertext:L5:c2 | 45 | 283115520 | 0.263672 |
| token_0-1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L2:c2 | 23 | 72351744 | 0.067383 |
| token_0-1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L3:c2 | 2 | 8388608 | 0.007812 |
| token_0-1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L4:c2 | 1032 | 5410652160 | 5.039062 |
| token_0-1/layer_06/model_list.6.attention.o_weight.weight | plaintext:L5:c1 | 496 | 1560281088 | 1.453125 |
| token_0-1/layer_06/model_list.6.attention.o_weight.weight | ciphertext:L5:c2 | 492 | 3095396352 | 2.882812 |
| token_0-1/layer_06/rope | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_06/rope | ciphertext:L4:c2 | 16 | 83886080 | 0.078125 |
| token_0-1/layer_06/rope | ciphertext:L5:c2 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_07/_decoder_block | ciphertext:L1:c2 | 2312 | 4848615424 | 4.515625 |
| token_0-1/layer_07/_decoder_block | ciphertext:L2:c2 | 1486 | 4674551808 | 4.353516 |
| token_0-1/layer_07/_decoder_block | ciphertext:L5:c2 | 928 | 5838471168 | 5.437500 |
| token_0-1/layer_07/_decoder_block | ciphertext:L5:c3 | 5 | 47185920 | 0.043945 |
| token_0-1/layer_07/_decoder_block | ciphertext:L4:c2 | 2530 | 13264486400 | 12.353516 |
| token_0-1/layer_07/_decoder_block | ciphertext:L3:c2 | 1422 | 5964300288 | 5.554688 |
| token_0-1/layer_07/_decoder_block | ciphertext:L3:c3 | 7 | 44040192 | 0.041016 |
| token_0-1/layer_07/_decoder_block | ciphertext:L2:c3 | 7 | 33030144 | 0.030762 |
| token_0-1/layer_07/_decoder_block | ciphertext:L4:c3 | 7 | 55050240 | 0.051270 |
| token_0-1/layer_07/_decoder_block | plaintext:L2:c1 | 1019 | 1602748416 | 1.492676 |
| token_0-1/layer_07/_decoder_block | plaintext:L5:c1 | 797 | 2507145216 | 2.334961 |
| token_0-1/layer_07/_decoder_block | plaintext:L4:c1 | 243 | 637009920 | 0.593262 |
| token_0-1/layer_07/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_07/attention | ciphertext:L1:c2 | 35 | 73400320 | 0.068359 |
| token_0-1/layer_07/attention | ciphertext:L2:c2 | 13 | 40894464 | 0.038086 |
| token_0-1/layer_07/attention | ciphertext:L4:c2 | 41 | 214958080 | 0.200195 |
| token_0-1/layer_07/attention | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_07/attention | ciphertext:L3:c2 | 535 | 2243952640 | 2.089844 |
| token_0-1/layer_07/attention | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_07/attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_07/attention | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_07/gqa_attention | ciphertext:L3:c2 | 8 | 33554432 | 0.031250 |
| token_0-1/layer_07/gqa_attention | ciphertext:L2:c2 | 61 | 191889408 | 0.178711 |
| token_0-1/layer_07/gqa_attention | ciphertext:L4:c2 | 56 | 293601280 | 0.273438 |
| token_0-1/layer_07/gqa_attention | plaintext:L5:c1 | 25 | 78643200 | 0.073242 |
| token_0-1/layer_07/gqa_attention | ciphertext:L5:c2 | 22 | 138412032 | 0.128906 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L1:c2 | 123 | 257949696 | 0.240234 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L2:c2 | 297 | 934281216 | 0.870117 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L3:c2 | 71 | 297795584 | 0.277344 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L5:c2 | 393 | 2472542208 | 2.302734 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L4:c2 | 774 | 4057989120 | 3.779297 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | plaintext:L5:c1 | 378 | 1189085184 | 1.107422 |
| token_0-1/layer_07/rope | ciphertext:L3:c2 | 11 | 46137344 | 0.042969 |
| token_0-1/layer_07/rope | ciphertext:L2:c2 | 17 | 53477376 | 0.049805 |
| token_0-1/layer_08/_decoder_block | ciphertext:L4:c2 | 3227 | 16918773760 | 15.756836 |
| token_0-1/layer_08/_decoder_block | ciphertext:L5:c2 | 1583 | 9959374848 | 9.275391 |
| token_0-1/layer_08/_decoder_block | plaintext:L5:c1 | 1561 | 4910481408 | 4.573242 |
| token_0-1/layer_08/_decoder_block | ciphertext:L2:c2 | 1306 | 4108320768 | 3.826172 |
| token_0-1/layer_08/_decoder_block | ciphertext:L3:c2 | 81 | 339738624 | 0.316406 |
| token_0-1/layer_08/_decoder_block | ciphertext:L1:c2 | 1975 | 4141875200 | 3.857422 |
| token_0-1/layer_08/_decoder_block | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_08/_decoder_block | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_08/_decoder_block | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_08/_decoder_block | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_08/_decoder_block | plaintext:L2:c1 | 912 | 1434451968 | 1.335938 |
| token_0-1/layer_08/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_08/attention | ciphertext:L4:c2 | 279 | 1462763520 | 1.362305 |
| token_0-1/layer_08/attention | plaintext:L5:c1 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_08/attention | ciphertext:L5:c2 | 44 | 276824064 | 0.257812 |
| token_0-1/layer_08/attention | ciphertext:L2:c2 | 122 | 383778816 | 0.357422 |
| token_0-1/layer_08/attention | ciphertext:L3:c2 | 99 | 415236096 | 0.386719 |
| token_0-1/layer_08/attention | ciphertext:L1:c2 | 57 | 119537664 | 0.111328 |
| token_0-1/layer_08/attention | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_08/attention | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_08/attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_08/attention | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_08/attention | plaintext:L4:c1 | 9 | 23592960 | 0.021973 |
| token_0-1/layer_08/gqa_attention | ciphertext:L3:c2 | 59 | 247463936 | 0.230469 |
| token_0-1/layer_08/gqa_attention | plaintext:L4:c1 | 9 | 23592960 | 0.021973 |
| token_0-1/layer_08/gqa_attention | ciphertext:L4:c2 | 10 | 52428800 | 0.048828 |
| token_0-1/layer_08/gqa_attention | ciphertext:L2:c2 | 94 | 295698432 | 0.275391 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | plaintext:L5:c1 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L5:c2 | 39 | 245366784 | 0.228516 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c2 | 157 | 823132160 | 0.766602 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c2 | 1193 | 5003804672 | 4.660156 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | plaintext:L4:c1 | 34 | 89128960 | 0.083008 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L2:c2 | 427 | 1343225856 | 1.250977 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L1:c2 | 182 | 381681664 | 0.355469 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L5:c3 | 2 | 18874368 | 0.017578 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L3:c3 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L2:c3 | 2 | 9437184 | 0.008789 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | ciphertext:L4:c3 | 2 | 15728640 | 0.014648 |
| token_0-1/layer_08/rope | ciphertext:L3:c2 | 17 | 71303168 | 0.066406 |
| token_0-1/layer_08/rope | plaintext:L4:c1 | 7 | 18350080 | 0.017090 |
| token_0-1/layer_08/rope | ciphertext:L4:c2 | 4 | 20971520 | 0.019531 |
| token_0-1/layer_09/_decoder_block | ciphertext:L5:c2 | 1320 | 8304721920 | 7.734375 |
| token_0-1/layer_09/_decoder_block | plaintext:L5:c1 | 1265 | 3979345920 | 3.706055 |
| token_0-1/layer_09/_decoder_block | ciphertext:L4:c2 | 2754 | 14438891520 | 13.447266 |
| token_0-1/layer_09/_decoder_block | ciphertext:L2:c2 | 1578 | 4963958784 | 4.623047 |
| token_0-1/layer_09/_decoder_block | plaintext:L2:c1 | 684 | 1075838976 | 1.001953 |
| token_0-1/layer_09/_decoder_block | ciphertext:L1:c2 | 1671 | 3504340992 | 3.263672 |
| token_0-1/layer_09/_decoder_block | ciphertext:L3:c2 | 1333 | 5591007232 | 5.207031 |
| token_0-1/layer_09/_decoder_block | plaintext:L4:c1 | 34 | 89128960 | 0.083008 |
| token_0-1/layer_09/_decoder_block | ciphertext:L5:c3 | 3 | 28311552 | 0.026367 |
| token_0-1/layer_09/_decoder_block | ciphertext:L3:c3 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_09/_decoder_block | ciphertext:L2:c3 | 3 | 14155776 | 0.013184 |
| token_0-1/layer_09/_decoder_block | ciphertext:L4:c3 | 5 | 39321600 | 0.036621 |
| token_0-1/layer_09/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_09/attention | ciphertext:L4:c2 | 267 | 1399848960 | 1.303711 |
| token_0-1/layer_09/attention | plaintext:L5:c1 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_09/attention | ciphertext:L5:c2 | 53 | 333447168 | 0.310547 |
| token_0-1/layer_09/attention | ciphertext:L1:c2 | 141 | 295698432 | 0.275391 |
| token_0-1/layer_09/attention | plaintext:L2:c1 | 62 | 97517568 | 0.090820 |
| token_0-1/layer_09/attention | ciphertext:L2:c2 | 79 | 248512512 | 0.231445 |
| token_0-1/layer_09/attention | ciphertext:L3:c2 | 10 | 41943040 | 0.039062 |
| token_0-1/layer_09/attention | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_09/attention | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_09/attention | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_09/gqa_attention | ciphertext:L4:c2 | 88 | 461373440 | 0.429688 |
| token_0-1/layer_09/gqa_attention | ciphertext:L5:c2 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_09/gqa_attention | ciphertext:L3:c2 | 75 | 314572800 | 0.292969 |
| token_0-1/layer_09/gqa_attention | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L4:c2 | 699 | 3664773120 | 3.413086 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L5:c2 | 67 | 421527552 | 0.392578 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | plaintext:L4:c1 | 260 | 681574400 | 0.634766 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L3:c2 | 791 | 3317694464 | 3.089844 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L2:c2 | 259 | 814743552 | 0.758789 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L1:c2 | 59 | 123731968 | 0.115234 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_09/rope | ciphertext:L4:c2 | 25 | 131072000 | 0.122070 |
| token_0-1/layer_09/rope | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_10/_decoder_block | ciphertext:L1:c2 | 1516 | 3179282432 | 2.960938 |
| token_0-1/layer_10/_decoder_block | plaintext:L2:c1 | 571 | 898105344 | 0.836426 |
| token_0-1/layer_10/_decoder_block | ciphertext:L2:c2 | 1698 | 5341446144 | 4.974609 |
| token_0-1/layer_10/_decoder_block | ciphertext:L4:c2 | 2599 | 13626245120 | 12.690430 |
| token_0-1/layer_10/_decoder_block | ciphertext:L5:c2 | 944 | 5939134464 | 5.531250 |
| token_0-1/layer_10/_decoder_block | plaintext:L4:c1 | 312 | 817889280 | 0.761719 |
| token_0-1/layer_10/_decoder_block | ciphertext:L3:c2 | 2276 | 9546235904 | 8.890625 |
| token_0-1/layer_10/_decoder_block | ciphertext:L4:c3 | 8 | 62914560 | 0.058594 |
| token_0-1/layer_10/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_10/_decoder_block | ciphertext:L3:c3 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_10/_decoder_block | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_10/_decoder_block | plaintext:L5:c1 | 825 | 2595225600 | 2.416992 |
| token_0-1/layer_10/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_10/attention | plaintext:L2:c1 | 159 | 250085376 | 0.232910 |
| token_0-1/layer_10/attention | ciphertext:L2:c2 | 152 | 478150656 | 0.445312 |
| token_0-1/layer_10/attention | ciphertext:L1:c2 | 317 | 664797184 | 0.619141 |
| token_0-1/layer_10/gqa_attention | plaintext:L2:c1 | 42 | 66060288 | 0.061523 |
| token_0-1/layer_10/gqa_attention | ciphertext:L2:c2 | 44 | 138412032 | 0.128906 |
| token_0-1/layer_10/gqa_attention | ciphertext:L1:c2 | 86 | 180355072 | 0.167969 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c2 | 409 | 1286602752 | 1.198242 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L1:c2 | 847 | 1776287744 | 1.654297 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | plaintext:L2:c1 | 394 | 619708416 | 0.577148 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L3:c2 | 9 | 37748736 | 0.035156 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L5:c2 | 49 | 308281344 | 0.287109 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L4:c2 | 331 | 1735393280 | 1.616211 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_10/rope | ciphertext:L2:c2 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_10/rope | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_0-1/layer_10/rope | plaintext:L2:c1 | 7 | 11010048 | 0.010254 |
| token_0-1/layer_11/_decoder_block | ciphertext:L1:c2 | 1609 | 3374317568 | 3.142578 |
| token_0-1/layer_11/_decoder_block | ciphertext:L2:c2 | 1751 | 5508169728 | 5.129883 |
| token_0-1/layer_11/_decoder_block | plaintext:L2:c1 | 603 | 948436992 | 0.883301 |
| token_0-1/layer_11/_decoder_block | ciphertext:L3:c2 | 2764 | 11593056256 | 10.796875 |
| token_0-1/layer_11/_decoder_block | ciphertext:L2:c3 | 12 | 56623104 | 0.052734 |
| token_0-1/layer_11/_decoder_block | ciphertext:L5:c2 | 565 | 3554672640 | 3.310547 |
| token_0-1/layer_11/_decoder_block | ciphertext:L4:c2 | 2673 | 14014218240 | 13.051758 |
| token_0-1/layer_11/_decoder_block | ciphertext:L4:c3 | 10 | 78643200 | 0.073242 |
| token_0-1/layer_11/_decoder_block | ciphertext:L3:c3 | 10 | 62914560 | 0.058594 |
| token_0-1/layer_11/_decoder_block | plaintext:L4:c1 | 555 | 1454899200 | 1.354980 |
| token_0-1/layer_11/_decoder_block | plaintext:L5:c1 | 319 | 1003487232 | 0.934570 |
| token_0-1/layer_11/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_11/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_11/attention | ciphertext:L1:c2 | 28 | 58720256 | 0.054688 |
| token_0-1/layer_11/attention | plaintext:L2:c1 | 16 | 25165824 | 0.023438 |
| token_0-1/layer_11/attention | ciphertext:L2:c2 | 21 | 66060288 | 0.061523 |
| token_0-1/layer_11/attention | ciphertext:L5:c2 | 138 | 868220928 | 0.808594 |
| token_0-1/layer_11/attention | ciphertext:L4:c2 | 289 | 1515192320 | 1.411133 |
| token_0-1/layer_11/attention | plaintext:L5:c1 | 136 | 427819008 | 0.398438 |
| token_0-1/layer_11/gqa_attention | ciphertext:L1:c2 | 90 | 188743680 | 0.175781 |
| token_0-1/layer_11/gqa_attention | ciphertext:L2:c2 | 42 | 132120576 | 0.123047 |
| token_0-1/layer_11/gqa_attention | plaintext:L2:c1 | 40 | 62914560 | 0.058594 |
| token_0-1/layer_11/model_list.11.attention.o_weight.weight | plaintext:L2:c1 | 515 | 810024960 | 0.754395 |
| token_0-1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L2:c2 | 498 | 1566572544 | 1.458984 |
| token_0-1/layer_11/model_list.11.attention.o_weight.weight | ciphertext:L1:c2 | 1032 | 2164260864 | 2.015625 |
| token_0-1/layer_11/rope | ciphertext:L2:c2 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_11/rope | ciphertext:L1:c2 | 12 | 25165824 | 0.023438 |
| token_0-1/layer_11/rope | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_12/_decoder_block | ciphertext:L4:c2 | 1731 | 9075425280 | 8.452148 |
| token_0-1/layer_12/_decoder_block | ciphertext:L3:c2 | 2771 | 11622416384 | 10.824219 |
| token_0-1/layer_12/_decoder_block | ciphertext:L2:c2 | 2192 | 6895435776 | 6.421875 |
| token_0-1/layer_12/_decoder_block | ciphertext:L1:c2 | 2415 | 5064622080 | 4.716797 |
| token_0-1/layer_12/_decoder_block | ciphertext:L5:c2 | 232 | 1459617792 | 1.359375 |
| token_0-1/layer_12/_decoder_block | ciphertext:L4:c3 | 8 | 62914560 | 0.058594 |
| token_0-1/layer_12/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_12/_decoder_block | ciphertext:L3:c3 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_12/_decoder_block | ciphertext:L2:c3 | 8 | 37748736 | 0.035156 |
| token_0-1/layer_12/_decoder_block | plaintext:L2:c1 | 991 | 1558708224 | 1.451660 |
| token_0-1/layer_12/_decoder_block | plaintext:L4:c1 | 379 | 993525760 | 0.925293 |
| token_0-1/layer_12/_decoder_block | plaintext:L5:c1 | 27 | 84934656 | 0.079102 |
| token_0-1/layer_12/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_12/attention | ciphertext:L3:c2 | 30 | 125829120 | 0.117188 |
| token_0-1/layer_12/attention | ciphertext:L4:c2 | 316 | 1656750080 | 1.542969 |
| token_0-1/layer_12/attention | ciphertext:L5:c2 | 140 | 880803840 | 0.820312 |
| token_0-1/layer_12/attention | plaintext:L5:c1 | 142 | 446693376 | 0.416016 |
| token_0-1/layer_12/gqa_attention | ciphertext:L4:c2 | 86 | 450887680 | 0.419922 |
| token_0-1/layer_12/gqa_attention | plaintext:L5:c1 | 44 | 138412032 | 0.128906 |
| token_0-1/layer_12/gqa_attention | ciphertext:L5:c2 | 42 | 264241152 | 0.246094 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L3:c2 | 20 | 83886080 | 0.078125 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L4:c2 | 972 | 5096079360 | 4.746094 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L5:c2 | 473 | 2975858688 | 2.771484 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | plaintext:L5:c1 | 459 | 1443889152 | 1.344727 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L2:c2 | 45 | 141557760 | 0.131836 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | plaintext:L2:c1 | 33 | 51904512 | 0.048340 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | ciphertext:L1:c2 | 43 | 90177536 | 0.083984 |
| token_0-1/layer_12/rope | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_12/rope | ciphertext:L4:c2 | 13 | 68157440 | 0.063477 |
| token_0-1/layer_12/rope | ciphertext:L5:c2 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_13/_decoder_block | ciphertext:L2:c2 | 1906 | 5995757568 | 5.583984 |
| token_0-1/layer_13/_decoder_block | ciphertext:L3:c2 | 1584 | 6643777536 | 6.187500 |
| token_0-1/layer_13/_decoder_block | ciphertext:L1:c2 | 2368 | 4966055936 | 4.625000 |
| token_0-1/layer_13/_decoder_block | ciphertext:L5:c2 | 766 | 4819255296 | 4.488281 |
| token_0-1/layer_13/_decoder_block | ciphertext:L5:c3 | 3 | 28311552 | 0.026367 |
| token_0-1/layer_13/_decoder_block | ciphertext:L4:c2 | 2207 | 11571036160 | 10.776367 |
| token_0-1/layer_13/_decoder_block | ciphertext:L3:c3 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_13/_decoder_block | ciphertext:L2:c3 | 7 | 33030144 | 0.030762 |
| token_0-1/layer_13/_decoder_block | ciphertext:L4:c3 | 7 | 55050240 | 0.051270 |
| token_0-1/layer_13/_decoder_block | plaintext:L2:c1 | 1025 | 1612185600 | 1.501465 |
| token_0-1/layer_13/_decoder_block | plaintext:L5:c1 | 643 | 2022703104 | 1.883789 |
| token_0-1/layer_13/_decoder_block | plaintext:L4:c1 | 239 | 626524160 | 0.583496 |
| token_0-1/layer_13/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_13/attention | ciphertext:L3:c2 | 322 | 1350565888 | 1.257812 |
| token_0-1/layer_13/attention | ciphertext:L2:c2 | 293 | 921698304 | 0.858398 |
| token_0-1/layer_13/attention | ciphertext:L4:c2 | 13 | 68157440 | 0.063477 |
| token_0-1/layer_13/gqa_attention | ciphertext:L2:c2 | 34 | 106954752 | 0.099609 |
| token_0-1/layer_13/gqa_attention | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_0-1/layer_13/gqa_attention | ciphertext:L1:c2 | 34 | 71303168 | 0.066406 |
| token_0-1/layer_13/gqa_attention | ciphertext:L4:c2 | 52 | 272629760 | 0.253906 |
| token_0-1/layer_13/gqa_attention | ciphertext:L5:c2 | 28 | 176160768 | 0.164062 |
| token_0-1/layer_13/gqa_attention | plaintext:L5:c1 | 23 | 72351744 | 0.067383 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L3:c2 | 57 | 239075328 | 0.222656 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c2 | 67 | 210763776 | 0.196289 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L1:c2 | 59 | 123731968 | 0.115234 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L5:c2 | 464 | 2919235584 | 2.718750 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L4:c2 | 944 | 4949278720 | 4.609375 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | plaintext:L5:c1 | 445 | 1399848960 | 1.303711 |
| token_0-1/layer_13/rope | ciphertext:L2:c2 | 22 | 69206016 | 0.064453 |
| token_0-1/layer_13/rope | ciphertext:L3:c2 | 2 | 8388608 | 0.007812 |
| token_0-1/layer_13/rope | ciphertext:L1:c2 | 4 | 8388608 | 0.007812 |
| token_0-1/layer_14/_decoder_block | ciphertext:L4:c2 | 3260 | 17091788800 | 15.917969 |
| token_0-1/layer_14/_decoder_block | ciphertext:L5:c2 | 1592 | 10015997952 | 9.328125 |
| token_0-1/layer_14/_decoder_block | plaintext:L5:c1 | 1561 | 4910481408 | 4.573242 |
| token_0-1/layer_14/_decoder_block | ciphertext:L2:c2 | 1069 | 3362783232 | 3.131836 |
| token_0-1/layer_14/_decoder_block | ciphertext:L1:c2 | 2094 | 4391436288 | 4.089844 |
| token_0-1/layer_14/_decoder_block | ciphertext:L5:c3 | 2 | 18874368 | 0.017578 |
| token_0-1/layer_14/_decoder_block | ciphertext:L3:c2 | 78 | 327155712 | 0.304688 |
| token_0-1/layer_14/_decoder_block | ciphertext:L4:c3 | 5 | 39321600 | 0.036621 |
| token_0-1/layer_14/_decoder_block | ciphertext:L3:c3 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_14/_decoder_block | ciphertext:L2:c3 | 2 | 9437184 | 0.008789 |
| token_0-1/layer_14/_decoder_block | plaintext:L2:c1 | 985 | 1549271040 | 1.442871 |
| token_0-1/layer_14/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_14/attention | ciphertext:L4:c2 | 258 | 1352663040 | 1.259766 |
| token_0-1/layer_14/attention | plaintext:L5:c1 | 11 | 34603008 | 0.032227 |
| token_0-1/layer_14/attention | ciphertext:L5:c2 | 34 | 213909504 | 0.199219 |
| token_0-1/layer_14/attention | plaintext:L4:c1 | 42 | 110100480 | 0.102539 |
| token_0-1/layer_14/attention | ciphertext:L3:c2 | 167 | 700448768 | 0.652344 |
| token_0-1/layer_14/attention | ciphertext:L2:c2 | 86 | 270532608 | 0.251953 |
| token_0-1/layer_14/attention | ciphertext:L1:c2 | 30 | 62914560 | 0.058594 |
| token_0-1/layer_14/gqa_attention | ciphertext:L1:c2 | 56 | 117440512 | 0.109375 |
| token_0-1/layer_14/gqa_attention | ciphertext:L5:c2 | 14 | 88080384 | 0.082031 |
| token_0-1/layer_14/gqa_attention | ciphertext:L4:c2 | 33 | 173015040 | 0.161133 |
| token_0-1/layer_14/gqa_attention | ciphertext:L3:c2 | 23 | 96468992 | 0.089844 |
| token_0-1/layer_14/gqa_attention | ciphertext:L2:c2 | 43 | 135266304 | 0.125977 |
| token_0-1/layer_14/gqa_attention | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_14/gqa_attention | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_14/gqa_attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L5:c2 | 12 | 75497472 | 0.070312 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L4:c2 | 87 | 456130560 | 0.424805 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L1:c2 | 99 | 207618048 | 0.193359 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | plaintext:L4:c1 | 25 | 65536000 | 0.061035 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c2 | 1195 | 5012193280 | 4.667969 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L2:c2 | 617 | 1940914176 | 1.807617 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L4:c3 | 2 | 15728640 | 0.014648 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_14/rope | ciphertext:L5:c2 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_14/rope | ciphertext:L4:c2 | 14 | 73400320 | 0.068359 |
| token_0-1/layer_14/rope | ciphertext:L3:c2 | 7 | 29360128 | 0.027344 |
| token_0-1/layer_14/rope | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_15/_decoder_block | ciphertext:L4:c2 | 2866 | 15026094080 | 13.994141 |
| token_0-1/layer_15/_decoder_block | ciphertext:L5:c2 | 1387 | 8726249472 | 8.126953 |
| token_0-1/layer_15/_decoder_block | plaintext:L5:c1 | 1344 | 4227858432 | 3.937500 |
| token_0-1/layer_15/_decoder_block | ciphertext:L2:c2 | 1468 | 4617928704 | 4.300781 |
| token_0-1/layer_15/_decoder_block | plaintext:L2:c1 | 667 | 1049100288 | 0.977051 |
| token_0-1/layer_15/_decoder_block | ciphertext:L1:c2 | 1602 | 3359637504 | 3.128906 |
| token_0-1/layer_15/_decoder_block | ciphertext:L3:c2 | 1274 | 5343543296 | 4.976562 |
| token_0-1/layer_15/_decoder_block | ciphertext:L2:c3 | 6 | 28311552 | 0.026367 |
| token_0-1/layer_15/_decoder_block | ciphertext:L4:c3 | 6 | 47185920 | 0.043945 |
| token_0-1/layer_15/_decoder_block | plaintext:L4:c1 | 27 | 70778880 | 0.065918 |
| token_0-1/layer_15/_decoder_block | ciphertext:L5:c3 | 3 | 28311552 | 0.026367 |
| token_0-1/layer_15/_decoder_block | ciphertext:L3:c3 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_15/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_15/attention | ciphertext:L4:c2 | 524 | 2747269120 | 2.558594 |
| token_0-1/layer_15/attention | plaintext:L5:c1 | 10 | 31457280 | 0.029297 |
| token_0-1/layer_15/attention | ciphertext:L5:c2 | 80 | 503316480 | 0.468750 |
| token_0-1/layer_15/attention | ciphertext:L3:c2 | 4 | 16777216 | 0.015625 |
| token_0-1/layer_15/attention | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_15/attention | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_15/attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_15/attention | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_15/gqa_attention | ciphertext:L4:c2 | 27 | 141557760 | 0.131836 |
| token_0-1/layer_15/gqa_attention | plaintext:L4:c1 | 20 | 52428800 | 0.048828 |
| token_0-1/layer_15/gqa_attention | ciphertext:L3:c2 | 59 | 247463936 | 0.230469 |
| token_0-1/layer_15/gqa_attention | ciphertext:L2:c2 | 74 | 232783872 | 0.216797 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | plaintext:L5:c1 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L5:c2 | 56 | 352321536 | 0.328125 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L4:c2 | 572 | 2998927360 | 2.792969 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L3:c2 | 864 | 3623878656 | 3.375000 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | plaintext:L4:c1 | 255 | 668467200 | 0.622559 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c2 | 268 | 843055104 | 0.785156 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L1:c2 | 112 | 234881024 | 0.218750 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L5:c3 | 1 | 9437184 | 0.008789 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_15/rope | ciphertext:L4:c2 | 22 | 115343360 | 0.107422 |
| token_0-1/layer_15/rope | ciphertext:L5:c2 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_15/rope | plaintext:L4:c1 | 4 | 10485760 | 0.009766 |
| token_0-1/layer_15/rope | ciphertext:L3:c2 | 3 | 12582912 | 0.011719 |
| token_0-1/layer_16/_decoder_block | ciphertext:L1:c2 | 1515 | 3177185280 | 2.958984 |
| token_0-1/layer_16/_decoder_block | plaintext:L2:c1 | 567 | 891813888 | 0.830566 |
| token_0-1/layer_16/_decoder_block | ciphertext:L2:c2 | 1703 | 5357174784 | 4.989258 |
| token_0-1/layer_16/_decoder_block | ciphertext:L3:c2 | 2225 | 9332326400 | 8.691406 |
| token_0-1/layer_16/_decoder_block | plaintext:L4:c1 | 274 | 718274560 | 0.668945 |
| token_0-1/layer_16/_decoder_block | ciphertext:L4:c2 | 2543 | 13332643840 | 12.416992 |
| token_0-1/layer_16/_decoder_block | ciphertext:L5:c2 | 991 | 6234832896 | 5.806641 |
| token_0-1/layer_16/_decoder_block | ciphertext:L4:c3 | 8 | 62914560 | 0.058594 |
| token_0-1/layer_16/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_16/_decoder_block | ciphertext:L3:c3 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_16/_decoder_block | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_16/_decoder_block | plaintext:L5:c1 | 906 | 2850029568 | 2.654297 |
| token_0-1/layer_16/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_16/attention | plaintext:L2:c1 | 154 | 242221056 | 0.225586 |
| token_0-1/layer_16/attention | ciphertext:L2:c2 | 152 | 478150656 | 0.445312 |
| token_0-1/layer_16/attention | ciphertext:L1:c2 | 322 | 675282944 | 0.628906 |
| token_0-1/layer_16/gqa_attention | ciphertext:L1:c2 | 86 | 180355072 | 0.167969 |
| token_0-1/layer_16/gqa_attention | plaintext:L2:c1 | 43 | 67633152 | 0.062988 |
| token_0-1/layer_16/gqa_attention | ciphertext:L2:c2 | 43 | 135266304 | 0.125977 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L1:c2 | 686 | 1438646272 | 1.339844 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | plaintext:L2:c1 | 313 | 492306432 | 0.458496 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L2:c2 | 344 | 1082130432 | 1.007812 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c2 | 31 | 130023424 | 0.121094 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L5:c2 | 79 | 497025024 | 0.462891 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c2 | 567 | 2972712960 | 2.768555 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | plaintext:L4:c1 | 34 | 89128960 | 0.083008 |
| token_0-1/layer_16/rope | plaintext:L2:c1 | 5 | 7864320 | 0.007324 |
| token_0-1/layer_16/rope | ciphertext:L1:c2 | 14 | 29360128 | 0.027344 |
| token_0-1/layer_16/rope | ciphertext:L2:c2 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_17/_decoder_block | ciphertext:L1:c2 | 1605 | 3365928960 | 3.134766 |
| token_0-1/layer_17/_decoder_block | plaintext:L2:c1 | 602 | 946864128 | 0.881836 |
| token_0-1/layer_17/_decoder_block | ciphertext:L2:c2 | 1751 | 5508169728 | 5.129883 |
| token_0-1/layer_17/_decoder_block | ciphertext:L3:c2 | 2575 | 10800332800 | 10.058594 |
| token_0-1/layer_17/_decoder_block | ciphertext:L2:c3 | 12 | 56623104 | 0.052734 |
| token_0-1/layer_17/_decoder_block | ciphertext:L5:c2 | 647 | 4070572032 | 3.791016 |
| token_0-1/layer_17/_decoder_block | ciphertext:L4:c2 | 2752 | 14428405760 | 13.437500 |
| token_0-1/layer_17/_decoder_block | ciphertext:L4:c3 | 9 | 70778880 | 0.065918 |
| token_0-1/layer_17/_decoder_block | ciphertext:L3:c3 | 10 | 62914560 | 0.058594 |
| token_0-1/layer_17/_decoder_block | plaintext:L4:c1 | 501 | 1313341440 | 1.223145 |
| token_0-1/layer_17/_decoder_block | plaintext:L5:c1 | 382 | 1201668096 | 1.119141 |
| token_0-1/layer_17/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_17/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_17/attention | plaintext:L2:c1 | 88 | 138412032 | 0.128906 |
| token_0-1/layer_17/attention | ciphertext:L2:c2 | 101 | 317718528 | 0.295898 |
| token_0-1/layer_17/attention | ciphertext:L1:c2 | 170 | 356515840 | 0.332031 |
| token_0-1/layer_17/attention | plaintext:L5:c1 | 66 | 207618048 | 0.193359 |
| token_0-1/layer_17/attention | ciphertext:L5:c2 | 66 | 415236096 | 0.386719 |
| token_0-1/layer_17/attention | ciphertext:L4:c2 | 137 | 718274560 | 0.668945 |
| token_0-1/layer_17/gqa_attention | plaintext:L2:c1 | 43 | 67633152 | 0.062988 |
| token_0-1/layer_17/gqa_attention | ciphertext:L2:c2 | 44 | 138412032 | 0.128906 |
| token_0-1/layer_17/gqa_attention | ciphertext:L1:c2 | 85 | 178257920 | 0.166016 |
| token_0-1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L2:c2 | 496 | 1560281088 | 1.453125 |
| token_0-1/layer_17/model_list.17.attention.o_weight.weight | ciphertext:L1:c2 | 1046 | 2193620992 | 2.042969 |
| token_0-1/layer_17/model_list.17.attention.o_weight.weight | plaintext:L2:c1 | 503 | 791150592 | 0.736816 |
| token_0-1/layer_17/rope | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_17/rope | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_0-1/layer_17/rope | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_18/_decoder_block | plaintext:L4:c1 | 399 | 1045954560 | 0.974121 |
| token_0-1/layer_18/_decoder_block | ciphertext:L4:c2 | 1823 | 9557770240 | 8.901367 |
| token_0-1/layer_18/_decoder_block | ciphertext:L3:c2 | 3044 | 12767461376 | 11.890625 |
| token_0-1/layer_18/_decoder_block | ciphertext:L2:c2 | 2044 | 6429868032 | 5.988281 |
| token_0-1/layer_18/_decoder_block | ciphertext:L1:c2 | 2265 | 4750049280 | 4.423828 |
| token_0-1/layer_18/_decoder_block | ciphertext:L5:c2 | 230 | 1447034880 | 1.347656 |
| token_0-1/layer_18/_decoder_block | ciphertext:L4:c3 | 9 | 70778880 | 0.065918 |
| token_0-1/layer_18/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_18/_decoder_block | ciphertext:L3:c3 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_18/_decoder_block | ciphertext:L2:c3 | 8 | 37748736 | 0.035156 |
| token_0-1/layer_18/_decoder_block | plaintext:L2:c1 | 913 | 1436024832 | 1.337402 |
| token_0-1/layer_18/_decoder_block | plaintext:L5:c1 | 32 | 100663296 | 0.093750 |
| token_0-1/layer_18/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_18/attention | ciphertext:L4:c2 | 309 | 1620049920 | 1.508789 |
| token_0-1/layer_18/attention | ciphertext:L3:c2 | 23 | 96468992 | 0.089844 |
| token_0-1/layer_18/attention | plaintext:L4:c1 | 18 | 47185920 | 0.043945 |
| token_0-1/layer_18/attention | plaintext:L5:c1 | 141 | 443547648 | 0.413086 |
| token_0-1/layer_18/attention | ciphertext:L5:c2 | 146 | 918552576 | 0.855469 |
| token_0-1/layer_18/gqa_attention | ciphertext:L4:c2 | 88 | 461373440 | 0.429688 |
| token_0-1/layer_18/gqa_attention | plaintext:L5:c1 | 43 | 135266304 | 0.125977 |
| token_0-1/layer_18/gqa_attention | ciphertext:L5:c2 | 41 | 257949696 | 0.240234 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L3:c2 | 18 | 75497472 | 0.070312 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | plaintext:L4:c1 | 5 | 13107200 | 0.012207 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L4:c2 | 815 | 4272947200 | 3.979492 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L5:c2 | 402 | 2529165312 | 2.355469 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | plaintext:L5:c1 | 399 | 1255145472 | 1.168945 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L2:c2 | 117 | 368050176 | 0.342773 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | plaintext:L2:c1 | 104 | 163577856 | 0.152344 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | ciphertext:L1:c2 | 186 | 390070272 | 0.363281 |
| token_0-1/layer_18/rope | ciphertext:L4:c2 | 11 | 57671680 | 0.053711 |
| token_0-1/layer_18/rope | ciphertext:L5:c2 | 8 | 50331648 | 0.046875 |
| token_0-1/layer_18/rope | plaintext:L5:c1 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_19/_decoder_block | ciphertext:L3:c2 | 1772 | 7432306688 | 6.921875 |
| token_0-1/layer_19/_decoder_block | plaintext:L4:c1 | 240 | 629145600 | 0.585938 |
| token_0-1/layer_19/_decoder_block | ciphertext:L4:c2 | 2094 | 10978590720 | 10.224609 |
| token_0-1/layer_19/_decoder_block | ciphertext:L2:c2 | 2079 | 6539968512 | 6.090820 |
| token_0-1/layer_19/_decoder_block | ciphertext:L1:c2 | 2391 | 5014290432 | 4.669922 |
| token_0-1/layer_19/_decoder_block | ciphertext:L5:c2 | 634 | 3988783104 | 3.714844 |
| token_0-1/layer_19/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_19/_decoder_block | ciphertext:L3:c3 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_19/_decoder_block | ciphertext:L2:c3 | 8 | 37748736 | 0.035156 |
| token_0-1/layer_19/_decoder_block | ciphertext:L4:c3 | 8 | 62914560 | 0.058594 |
| token_0-1/layer_19/_decoder_block | plaintext:L2:c1 | 1026 | 1613758464 | 1.502930 |
| token_0-1/layer_19/_decoder_block | plaintext:L5:c1 | 495 | 1557135360 | 1.450195 |
| token_0-1/layer_19/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_19/attention | ciphertext:L3:c2 | 127 | 532676608 | 0.496094 |
| token_0-1/layer_19/attention | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_19/attention | ciphertext:L4:c2 | 10 | 52428800 | 0.048828 |
| token_0-1/layer_19/attention | ciphertext:L2:c2 | 370 | 1163919360 | 1.083984 |
| token_0-1/layer_19/attention | ciphertext:L1:c2 | 99 | 207618048 | 0.193359 |
| token_0-1/layer_19/attention | ciphertext:L5:c2 | 12 | 75497472 | 0.070312 |
| token_0-1/layer_19/attention | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_19/attention | ciphertext:L3:c3 | 2 | 12582912 | 0.011719 |
| token_0-1/layer_19/attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_19/attention | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_19/attention | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_19/gqa_attention | ciphertext:L1:c2 | 1 | 2097152 | 0.001953 |
| token_0-1/layer_19/gqa_attention | ciphertext:L5:c2 | 42 | 264241152 | 0.246094 |
| token_0-1/layer_19/gqa_attention | ciphertext:L4:c2 | 86 | 450887680 | 0.419922 |
| token_0-1/layer_19/gqa_attention | plaintext:L5:c1 | 43 | 135266304 | 0.125977 |
| token_0-1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L3:c2 | 23 | 96468992 | 0.089844 |
| token_0-1/layer_19/model_list.19.attention.o_weight.weight | plaintext:L4:c1 | 1 | 2621440 | 0.002441 |
| token_0-1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L4:c2 | 1035 | 5426380800 | 5.053711 |
| token_0-1/layer_19/model_list.19.attention.o_weight.weight | ciphertext:L5:c2 | 497 | 3126853632 | 2.912109 |
| token_0-1/layer_19/model_list.19.attention.o_weight.weight | plaintext:L5:c1 | 489 | 1538260992 | 1.432617 |
| token_0-1/layer_19/rope | ciphertext:L5:c2 | 13 | 81788928 | 0.076172 |
| token_0-1/layer_19/rope | ciphertext:L4:c2 | 11 | 57671680 | 0.053711 |
| token_0-1/layer_19/rope | plaintext:L5:c1 | 4 | 12582912 | 0.011719 |
| token_0-1/layer_20/_decoder_block | ciphertext:L4:c2 | 3342 | 17521704960 | 16.318359 |
| token_0-1/layer_20/_decoder_block | plaintext:L5:c1 | 1526 | 4800380928 | 4.470703 |
| token_0-1/layer_20/_decoder_block | ciphertext:L5:c2 | 1559 | 9808379904 | 9.134766 |
| token_0-1/layer_20/_decoder_block | plaintext:L4:c1 | 14 | 36700160 | 0.034180 |
| token_0-1/layer_20/_decoder_block | ciphertext:L3:c2 | 56 | 234881024 | 0.218750 |
| token_0-1/layer_20/_decoder_block | ciphertext:L2:c2 | 1044 | 3284140032 | 3.058594 |
| token_0-1/layer_20/_decoder_block | plaintext:L2:c1 | 1013 | 1593311232 | 1.483887 |
| token_0-1/layer_20/_decoder_block | ciphertext:L1:c2 | 2094 | 4391436288 | 4.089844 |
| token_0-1/layer_20/_decoder_block | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_20/_decoder_block | ciphertext:L2:c3 | 3 | 14155776 | 0.013184 |
| token_0-1/layer_20/_decoder_block | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_20/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_20/attention | plaintext:L5:c1 | 11 | 34603008 | 0.032227 |
| token_0-1/layer_20/attention | ciphertext:L5:c2 | 27 | 169869312 | 0.158203 |
| token_0-1/layer_20/attention | ciphertext:L4:c2 | 106 | 555745280 | 0.517578 |
| token_0-1/layer_20/attention | ciphertext:L3:c2 | 269 | 1128267776 | 1.050781 |
| token_0-1/layer_20/attention | plaintext:L4:c1 | 32 | 83886080 | 0.078125 |
| token_0-1/layer_20/attention | ciphertext:L2:c2 | 109 | 342884352 | 0.319336 |
| token_0-1/layer_20/attention | ciphertext:L1:c2 | 74 | 155189248 | 0.144531 |
| token_0-1/layer_20/gqa_attention | ciphertext:L3:c2 | 161 | 675282944 | 0.628906 |
| token_0-1/layer_20/gqa_attention | plaintext:L4:c1 | 2 | 5242880 | 0.004883 |
| token_0-1/layer_20/gqa_attention | ciphertext:L4:c2 | 6 | 31457280 | 0.029297 |
| token_0-1/layer_20/gqa_attention | ciphertext:L2:c2 | 3 | 9437184 | 0.008789 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L5:c2 | 59 | 371195904 | 0.345703 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L4:c2 | 136 | 713031680 | 0.664062 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | plaintext:L5:c1 | 23 | 72351744 | 0.067383 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c2 | 869 | 3644850176 | 3.394531 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | plaintext:L4:c1 | 15 | 39321600 | 0.036621 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L2:c2 | 734 | 2308964352 | 2.150391 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L1:c2 | 197 | 413138944 | 0.384766 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L5:c3 | 2 | 18874368 | 0.017578 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L3:c3 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L2:c3 | 3 | 14155776 | 0.013184 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | ciphertext:L4:c3 | 3 | 23592960 | 0.021973 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_20/rope | ciphertext:L3:c2 | 22 | 92274688 | 0.085938 |
| token_0-1/layer_20/rope | plaintext:L4:c1 | 3 | 7864320 | 0.007324 |
| token_0-1/layer_20/rope | ciphertext:L4:c2 | 3 | 15728640 | 0.014648 |
| token_0-1/layer_21/_decoder_block | ciphertext:L5:c2 | 1477 | 9292480512 | 8.654297 |
| token_0-1/layer_21/_decoder_block | ciphertext:L4:c2 | 3118 | 16347299840 | 15.224609 |
| token_0-1/layer_21/_decoder_block | plaintext:L5:c1 | 1422 | 4473225216 | 4.166016 |
| token_0-1/layer_21/_decoder_block | ciphertext:L2:c2 | 1436 | 4517265408 | 4.207031 |
| token_0-1/layer_21/_decoder_block | plaintext:L2:c1 | 640 | 1006632960 | 0.937500 |
| token_0-1/layer_21/_decoder_block | ciphertext:L1:c2 | 1529 | 3206545408 | 2.986328 |
| token_0-1/layer_21/_decoder_block | ciphertext:L3:c2 | 1002 | 4202692608 | 3.914062 |
| token_0-1/layer_21/_decoder_block | plaintext:L4:c1 | 15 | 39321600 | 0.036621 |
| token_0-1/layer_21/_decoder_block | ciphertext:L5:c3 | 3 | 28311552 | 0.026367 |
| token_0-1/layer_21/_decoder_block | ciphertext:L3:c3 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_21/_decoder_block | ciphertext:L2:c3 | 3 | 14155776 | 0.013184 |
| token_0-1/layer_21/_decoder_block | ciphertext:L4:c3 | 5 | 39321600 | 0.036621 |
| token_0-1/layer_21/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_21/attention | ciphertext:L4:c2 | 376 | 1971322880 | 1.835938 |
| token_0-1/layer_21/attention | plaintext:L5:c1 | 11 | 34603008 | 0.032227 |
| token_0-1/layer_21/attention | ciphertext:L5:c2 | 49 | 308281344 | 0.287109 |
| token_0-1/layer_21/attention | plaintext:L4:c1 | 98 | 256901120 | 0.239258 |
| token_0-1/layer_21/attention | ciphertext:L3:c2 | 137 | 574619648 | 0.535156 |
| token_0-1/layer_21/gqa_attention | ciphertext:L4:c2 | 37 | 193986560 | 0.180664 |
| token_0-1/layer_21/gqa_attention | ciphertext:L3:c2 | 62 | 260046848 | 0.242188 |
| token_0-1/layer_21/gqa_attention | plaintext:L4:c1 | 32 | 83886080 | 0.078125 |
| token_0-1/layer_21/gqa_attention | ciphertext:L2:c2 | 22 | 69206016 | 0.064453 |
| token_0-1/layer_21/gqa_attention | ciphertext:L1:c2 | 24 | 50331648 | 0.046875 |
| token_0-1/layer_21/gqa_attention | ciphertext:L3:c3 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_21/gqa_attention | ciphertext:L2:c3 | 1 | 4718592 | 0.004395 |
| token_0-1/layer_21/gqa_attention | ciphertext:L5:c2 | 6 | 37748736 | 0.035156 |
| token_0-1/layer_21/gqa_attention | plaintext:L2:c1 | 1 | 1572864 | 0.001465 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L5:c2 | 49 | 308281344 | 0.287109 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c2 | 516 | 2705326080 | 2.519531 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | plaintext:L5:c1 | 6 | 18874368 | 0.017578 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L3:c2 | 960 | 4026531840 | 3.750000 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | plaintext:L4:c1 | 144 | 377487360 | 0.351562 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L2:c2 | 311 | 978321408 | 0.911133 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L1:c2 | 99 | 207618048 | 0.193359 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | ciphertext:L4:c3 | 2 | 15728640 | 0.014648 |
| token_0-1/layer_21/rope | ciphertext:L3:c2 | 18 | 75497472 | 0.070312 |
| token_0-1/layer_21/rope | ciphertext:L4:c2 | 5 | 26214400 | 0.024414 |
| token_0-1/layer_21/rope | plaintext:L4:c1 | 7 | 18350080 | 0.017090 |
| token_0-1/layer_22/_decoder_block | plaintext:L2:c1 | 1825 | 2870476800 | 2.673340 |
| token_0-1/layer_22/_decoder_block | ciphertext:L1:c2 | 3994 | 8376025088 | 7.800781 |
| token_0-1/layer_22/_decoder_block | ciphertext:L2:c2 | 2648 | 8329887744 | 7.757812 |
| token_0-1/layer_22/_decoder_block | ciphertext:L3:c2 | 1403 | 5884608512 | 5.480469 |
| token_0-1/layer_22/_decoder_block | plaintext:L4:c1 | 163 | 427294720 | 0.397949 |
| token_0-1/layer_22/_decoder_block | ciphertext:L4:c2 | 561 | 2941255680 | 2.739258 |
| token_0-1/layer_22/_decoder_block | ciphertext:L5:c2 | 69 | 434110464 | 0.404297 |
| token_0-1/layer_22/_decoder_block | ciphertext:L4:c3 | 8 | 62914560 | 0.058594 |
| token_0-1/layer_22/_decoder_block | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_0-1/layer_22/_decoder_block | ciphertext:L3:c3 | 9 | 56623104 | 0.052734 |
| token_0-1/layer_22/_decoder_block | ciphertext:L2:c3 | 8 | 37748736 | 0.035156 |
| token_0-1/layer_22/_decoder_block | plaintext:L3:c1 | 3 | 6291456 | 0.005859 |
| token_0-1/layer_22/_decoder_block | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_22/_decoder_block \| token_shared/layer_shared/interface | plaintext:L3:c1 | 46 | 96468992 | 0.089844 |
| token_0-1/layer_22/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_22/attention | ciphertext:L2:c2 | 154 | 484442112 | 0.451172 |
| token_0-1/layer_22/attention | ciphertext:L1:c2 | 319 | 668991488 | 0.623047 |
| token_0-1/layer_22/attention | plaintext:L2:c1 | 155 | 243793920 | 0.227051 |
| token_0-1/layer_22/gqa_attention | ciphertext:L1:c2 | 78 | 163577856 | 0.152344 |
| token_0-1/layer_22/gqa_attention | plaintext:L2:c1 | 37 | 58195968 | 0.054199 |
| token_0-1/layer_22/gqa_attention | ciphertext:L2:c2 | 42 | 132120576 | 0.123047 |
| token_0-1/layer_22/gqa_attention | ciphertext:L3:c2 | 6 | 25165824 | 0.023438 |
| token_0-1/layer_22/gqa_attention | ciphertext:L2:c3 | 2 | 9437184 | 0.008789 |
| token_0-1/layer_22/gqa_attention | ciphertext:L5:c2 | 3 | 18874368 | 0.017578 |
| token_0-1/layer_22/gqa_attention | ciphertext:L4:c2 | 3 | 15728640 | 0.014648 |
| token_0-1/layer_22/gqa_attention | ciphertext:L4:c3 | 1 | 7864320 | 0.007324 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L1:c2 | 526 | 1103101952 | 1.027344 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | plaintext:L2:c1 | 254 | 399507456 | 0.372070 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L2:c2 | 271 | 852492288 | 0.793945 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L2:c3 | 2 | 9437184 | 0.008789 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L5:c2 | 78 | 490733568 | 0.457031 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L4:c2 | 642 | 3365928960 | 3.134766 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c2 | 193 | 809500672 | 0.753906 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | ciphertext:L3:c3 | 1 | 6291456 | 0.005859 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | plaintext:L4:c1 | 140 | 367001600 | 0.341797 |
| token_0-1/layer_22/rope | ciphertext:L1:c2 | 13 | 27262976 | 0.025391 |
| token_0-1/layer_22/rope | ciphertext:L2:c2 | 9 | 28311552 | 0.026367 |
| token_0-1/layer_22/rope | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_23/_decoder_block | ciphertext:L1:c2 | 68 | 142606336 | 0.132812 |
| token_0-1/layer_23/_decoder_block | ciphertext:L2:c2 | 30 | 94371840 | 0.087891 |
| token_0-1/layer_23/_decoder_block | plaintext:L2:c1 | 27 | 42467328 | 0.039551 |
| token_0-1/layer_23/attention | plaintext:L1:c1 | 2 | 2097152 | 0.001953 |
| token_0-1/layer_23/attention | plaintext:L2:c1 | 156 | 245366784 | 0.228516 |
| token_0-1/layer_23/attention | ciphertext:L2:c2 | 151 | 475004928 | 0.442383 |
| token_0-1/layer_23/attention | ciphertext:L1:c2 | 321 | 673185792 | 0.626953 |
| token_0-1/layer_23/gqa_attention | ciphertext:L1:c2 | 76 | 159383552 | 0.148438 |
| token_0-1/layer_23/gqa_attention | plaintext:L2:c1 | 39 | 61341696 | 0.057129 |
| token_0-1/layer_23/gqa_attention | ciphertext:L2:c2 | 36 | 113246208 | 0.105469 |
| token_0-1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L1:c2 | 526 | 1103101952 | 1.027344 |
| token_0-1/layer_23/model_list.23.attention.o_weight.weight | plaintext:L2:c1 | 249 | 391643136 | 0.364746 |
| token_0-1/layer_23/model_list.23.attention.o_weight.weight | ciphertext:L2:c2 | 260 | 817889280 | 0.761719 |
| token_0-1/layer_23/model_list.23.post_Normal.weight | ciphertext:L1:c2 | 54 | 113246208 | 0.105469 |
| token_0-1/layer_23/model_list.23.post_Normal.weight | plaintext:L2:c1 | 27 | 42467328 | 0.039551 |
| token_0-1/layer_23/model_list.23.post_Normal.weight | ciphertext:L2:c2 | 25 | 78643200 | 0.073242 |
| token_0-1/layer_23/model_list.23.post_Normal.weight \| token_shared/layer_shared/final_normal.weight | ciphertext:L2:c2 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_23/rope | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_0-1/layer_23/rope | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_0-1/layer_23/rope | ciphertext:L2:c2 | 7 | 22020096 | 0.020508 |
| token_0-1/layer_23/swiglu | ciphertext:L2:c2 | 3120 | 9814671360 | 9.140625 |
| token_0-1/layer_23/swiglu | ciphertext:L1:c2 | 4161 | 8726249472 | 8.126953 |
| token_0-1/layer_23/swiglu | plaintext:L2:c1 | 1802 | 2834300928 | 2.639648 |
| token_0-1/layer_23/swiglu | ciphertext:L3:c2 | 4650 | 19503513600 | 18.164062 |
| token_0-1/layer_23/swiglu | ciphertext:L2:c3 | 12 | 56623104 | 0.052734 |
| token_0-1/layer_23/swiglu | ciphertext:L5:c2 | 100 | 629145600 | 0.585938 |
| token_0-1/layer_23/swiglu | ciphertext:L4:c2 | 792 | 4152360960 | 3.867188 |
| token_0-1/layer_23/swiglu | ciphertext:L4:c3 | 6 | 47185920 | 0.043945 |
| token_0-1/layer_23/swiglu | ciphertext:L3:c3 | 5 | 31457280 | 0.029297 |
| token_0-1/layer_23/swiglu | plaintext:L4:c1 | 230 | 602931200 | 0.561523 |
| token_0-1/layer_23/swiglu | plaintext:L5:c1 | 1 | 3145728 | 0.002930 |
| token_0-1/layer_23/swiglu | plaintext:L3:c1 | 4 | 8388608 | 0.007812 |
| token_0-1/layer_23/swiglu | ciphertext:L5:c3 | 2 | 18874368 | 0.017578 |
| token_shared/layer_shared/final_normal.weight | ciphertext:L3:c2 | 1 | 4194304 | 0.003906 |
| token_shared/layer_shared/final_normal.weight | plaintext:L2:c1 | 25 | 39321600 | 0.036621 |
| token_shared/layer_shared/final_normal.weight | ciphertext:L1:c2 | 53 | 111149056 | 0.103516 |
| token_shared/layer_shared/final_normal.weight | ciphertext:L2:c2 | 24 | 75497472 | 0.070312 |
| token_shared/layer_shared/interface | ciphertext:L2:c2 | 38 | 119537664 | 0.111328 |
| token_shared/layer_shared/interface | plaintext:L3:c1 | 1 | 2097152 | 0.001953 |
| token_shared/layer_shared/interface | ciphertext:L5:c2 | 9 | 56623104 | 0.052734 |
| token_shared/layer_shared/interface | ciphertext:L4:c2 | 18 | 94371840 | 0.087891 |
| token_shared/layer_shared/interface | plaintext:L5:c1 | 7 | 22020096 | 0.020508 |
| token_shared/layer_shared/interface | ciphertext:L3:c2 | 12 | 50331648 | 0.046875 |
| token_shared/layer_shared/interface | ciphertext:L1:c2 | 15 | 31457280 | 0.029297 |
| token_shared/layer_shared/interface | plaintext:L2:c1 | 6 | 9437184 | 0.008789 |
| token_shared/layer_shared/lm_head.weight | plaintext:L4:c1 | 230 | 602931200 | 0.561523 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L4:c2 | 937 | 4912578560 | 4.575195 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L3:c2 | 2155 | 9038725120 | 8.417969 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L2:c2 | 4993 | 15706619904 | 14.627930 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L1:c2 | 8179 | 17152606208 | 15.974609 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L5:c2 | 229 | 1440743424 | 1.341797 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L4:c3 | 8 | 62914560 | 0.058594 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L5:c3 | 4 | 37748736 | 0.035156 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L3:c3 | 8 | 50331648 | 0.046875 |
| token_shared/layer_shared/lm_head.weight | ciphertext:L2:c3 | 4 | 18874368 | 0.017578 |
| token_shared/layer_shared/lm_head.weight | plaintext:L2:c1 | 3860 | 6071255040 | 5.654297 |
| token_shared/layer_shared/lm_head.weight | plaintext:L5:c1 | 140 | 440401920 | 0.410156 |
| token_shared/layer_shared/pack_kv | ciphertext:L4:c2 | 50 | 262144000 | 0.244141 |
| token_shared/layer_shared/pack_kv | plaintext:L5:c1 | 21 | 66060288 | 0.061523 |
| token_shared/layer_shared/pack_kv | ciphertext:L5:c2 | 16 | 100663296 | 0.093750 |
| token_shared/layer_shared/pack_kv | ciphertext:L2:c2 | 46 | 144703488 | 0.134766 |
| token_shared/layer_shared/pack_kv | ciphertext:L1:c2 | 42 | 88080384 | 0.082031 |
| token_shared/layer_shared/pack_kv | ciphertext:L3:c2 | 42 | 176160768 | 0.164062 |
| token_shared/layer_shared/pack_kv | plaintext:L4:c1 | 8 | 20971520 | 0.019531 |
| token_shared/layer_shared/pack_kv | plaintext:L2:c1 | 19 | 29884416 | 0.027832 |

## 8. 每次自举的位置

ordinal是最终执行计划中的指令序号；相邻生产者/消费者保存在JSON。全部为编译计划，不表示真实自举已执行。

| 序号 | ordinal | 输入Value / level | 输出Value / level | 源码阶段 |
| --- | --- | --- | --- | --- |
| 1 | 17 | 63 / L1 | 69 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 2 | 21 | 66 / L1 | 70 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 3 | 25 | 68 / L1 | 71 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 4 | 111 | 113 / L1 | 114 / L5 | qwen24-stage/token_0-1/layer_00/attention |
| 5 | 230 | 178 / L1 | 179 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 6 | 297 | 210 / L1 | 214 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 7 | 299 | 213 / L1 | 215 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 8 | 700 | 833 / L1 | 872 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 9 | 1617 | 834 / L1 | 873 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 10 | 1620 | 869 / L1 | 874 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 11 | 1622 | 870 / L1 | 875 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 12 | 1624 | 871 / L1 | 876 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 13 | 1675 | 975 / L1 | 2150 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 14 | 1752 | 1022 / L1 | 2151 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 15 | 1799 | 1659 / L1 | 2152 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 16 | 1934 | 2044 / L1 | 2153 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 17 | 1943 | 2149 / L1 | 2155 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 18 | 2040 | 2204 / L1 | 2205 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 19 | 2044 | 2191 / L1 | 2206 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 20 | 2046 | 2195 / L1 | 2207 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 21 | 2050 | 2199 / L1 | 2208 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 22 | 2054 | 2203 / L1 | 2209 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 23 | 4305 | 2146 / L1 | 2154 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 24 | 4350 | 2238 / L1 | 2239 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 25 | 6513 | 3304 / L1 | 3308 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 26 | 6517 | 3307 / L1 | 3309 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 27 | 6603 | 3348 / L1 | 3349 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 28 | 6722 | 3402 / L1 | 3403 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 29 | 6810 | 3446 / L1 | 3447 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 30 | 15174 | 11794 / L1 | 11802 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 31 | 23523 | 11803 / L1 | 11804 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 32 | 23527 | 11801 / L1 | 11805 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 33 | 23598 | 11834 / L1 | 11843 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 34 | 23611 | 11838 / L1 | 11844 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 35 | 23622 | 11842 / L1 | 11845 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 36 | 23639 | 11863 / L1 | 11864 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 37 | 28031 | 13929 / L1 | 13934 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 38 | 28035 | 13931 / L1 | 13935 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 39 | 28039 | 13933 / L1 | 13936 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 40 | 28125 | 13975 / L1 | 13976 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 41 | 28244 | 14029 / L1 | 14030 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 42 | 28311 | 14060 / L1 | 14064 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 43 | 28313 | 14063 / L1 | 14065 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 44 | 28713 | 14678 / L1 | 14717 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 45 | 29630 | 14679 / L1 | 14718 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 46 | 29633 | 14714 / L1 | 14719 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 47 | 29635 | 14715 / L1 | 14720 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 48 | 29637 | 14716 / L1 | 14721 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 49 | 29688 | 14811 / L1 | 15976 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 50 | 29765 | 14855 / L1 | 15977 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 51 | 29812 | 15490 / L1 | 15978 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 52 | 29949 | 15873 / L1 | 15979 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 53 | 29958 | 15975 / L1 | 15981 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 54 | 30055 | 16026 / L1 | 16027 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 55 | 30059 | 16016 / L1 | 16028 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 56 | 30061 | 16019 / L1 | 16029 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 57 | 30065 | 16022 / L1 | 16030 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 58 | 30069 | 16025 / L1 | 16031 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 59 | 32320 | 15973 / L1 | 15980 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 60 | 32365 | 16059 / L1 | 16060 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 61 | 34528 | 17124 / L1 | 17128 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 62 | 34532 | 17127 / L1 | 17129 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 63 | 34618 | 17168 / L1 | 17169 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 64 | 34737 | 17222 / L1 | 17223 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 65 | 34825 | 17266 / L1 | 17267 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 66 | 43189 | 25614 / L1 | 25621 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 67 | 51538 | 25622 / L1 | 25623 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 68 | 51542 | 25620 / L1 | 25624 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 69 | 51613 | 25647 / L1 | 25655 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 70 | 51626 | 25650 / L1 | 25656 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 71 | 51637 | 25654 / L1 | 25657 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 72 | 51654 | 25673 / L1 | 25674 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 73 | 56046 | 27737 / L1 | 27742 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 74 | 56050 | 27739 / L1 | 27743 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 75 | 56054 | 27741 / L1 | 27744 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 76 | 56140 | 27783 / L1 | 27784 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 77 | 56259 | 27837 / L1 | 27838 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 78 | 56326 | 27868 / L1 | 27872 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 79 | 56328 | 27871 / L1 | 27873 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 80 | 56728 | 28486 / L1 | 28525 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 81 | 57645 | 28487 / L1 | 28526 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 82 | 57648 | 28522 / L1 | 28527 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 83 | 57650 | 28523 / L1 | 28528 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 84 | 57652 | 28524 / L1 | 28529 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 85 | 57703 | 28619 / L1 | 29784 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 86 | 57780 | 28663 / L1 | 29785 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 87 | 57827 | 29298 / L1 | 29786 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 88 | 57964 | 29681 / L1 | 29787 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 89 | 57973 | 29783 / L1 | 29789 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 90 | 58070 | 29834 / L1 | 29835 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 91 | 58074 | 29824 / L1 | 29836 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 92 | 58076 | 29827 / L1 | 29837 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 93 | 58080 | 29830 / L1 | 29838 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 94 | 58084 | 29833 / L1 | 29839 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 95 | 60335 | 29781 / L1 | 29788 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 96 | 60380 | 29867 / L1 | 29868 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 97 | 62543 | 30932 / L1 | 30936 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 98 | 62547 | 30935 / L1 | 30937 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 99 | 62633 | 30976 / L1 | 30977 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 100 | 62752 | 31030 / L1 | 31031 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 101 | 62840 | 31074 / L1 | 31075 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 102 | 71204 | 39422 / L1 | 39429 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 103 | 79553 | 39430 / L1 | 39431 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 104 | 79557 | 39428 / L1 | 39432 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 105 | 79628 | 39455 / L1 | 39463 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 106 | 79641 | 39458 / L1 | 39464 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 107 | 79652 | 39462 / L1 | 39465 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 108 | 79669 | 39481 / L1 | 39482 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 109 | 84061 | 41545 / L1 | 41550 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 110 | 84065 | 41547 / L1 | 41551 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 111 | 84069 | 41549 / L1 | 41552 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 112 | 84155 | 41591 / L1 | 41592 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 113 | 84274 | 41645 / L1 | 41646 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 114 | 84341 | 41676 / L1 | 41680 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 115 | 84343 | 41679 / L1 | 41681 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 116 | 84743 | 42294 / L1 | 42333 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 117 | 85660 | 42295 / L1 | 42334 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 118 | 85663 | 42330 / L1 | 42335 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 119 | 85665 | 42331 / L1 | 42336 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 120 | 85667 | 42332 / L1 | 42337 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 121 | 85718 | 42427 / L1 | 43592 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 122 | 85795 | 42471 / L1 | 43593 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 123 | 85842 | 43106 / L1 | 43594 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 124 | 85979 | 43489 / L1 | 43595 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 125 | 85988 | 43591 / L1 | 43597 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 126 | 86085 | 43642 / L1 | 43643 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 127 | 86089 | 43632 / L1 | 43644 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 128 | 86091 | 43635 / L1 | 43645 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 129 | 86095 | 43638 / L1 | 43646 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 130 | 86099 | 43641 / L1 | 43647 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 131 | 88350 | 43589 / L1 | 43596 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 132 | 88395 | 43675 / L1 | 43676 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 133 | 90558 | 44740 / L1 | 44744 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 134 | 90562 | 44743 / L1 | 44745 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 135 | 90648 | 44784 / L1 | 44785 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 136 | 90767 | 44838 / L1 | 44839 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 137 | 90855 | 44882 / L1 | 44883 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 138 | 99219 | 53230 / L1 | 53237 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 139 | 107568 | 53238 / L1 | 53239 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 140 | 107572 | 53236 / L1 | 53240 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 141 | 107643 | 53263 / L1 | 53271 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 142 | 107656 | 53266 / L1 | 53272 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 143 | 107667 | 53270 / L1 | 53273 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 144 | 107684 | 53289 / L1 | 53290 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 145 | 112076 | 55353 / L1 | 55358 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 146 | 112080 | 55355 / L1 | 55359 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 147 | 112084 | 55357 / L1 | 55360 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 148 | 112170 | 55399 / L1 | 55400 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 149 | 112289 | 55453 / L1 | 55454 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 150 | 112356 | 55484 / L1 | 55488 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 151 | 112358 | 55487 / L1 | 55489 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 152 | 112758 | 56102 / L1 | 56141 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 153 | 113675 | 56103 / L1 | 56142 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 154 | 113678 | 56138 / L1 | 56143 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 155 | 113680 | 56139 / L1 | 56144 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 156 | 113682 | 56140 / L1 | 56145 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 157 | 113733 | 56235 / L1 | 57400 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 158 | 113810 | 56279 / L1 | 57401 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 159 | 113857 | 56914 / L1 | 57402 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 160 | 113994 | 57297 / L1 | 57403 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 161 | 114003 | 57399 / L1 | 57405 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 162 | 114100 | 57450 / L1 | 57451 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 163 | 114104 | 57440 / L1 | 57452 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 164 | 114106 | 57443 / L1 | 57453 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 165 | 114110 | 57446 / L1 | 57454 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 166 | 114114 | 57449 / L1 | 57455 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 167 | 116365 | 57397 / L1 | 57404 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 168 | 116410 | 57483 / L1 | 57484 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 169 | 118573 | 58548 / L1 | 58552 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 170 | 118577 | 58551 / L1 | 58553 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 171 | 118663 | 58592 / L1 | 58593 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 172 | 118782 | 58646 / L1 | 58647 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 173 | 118870 | 58690 / L1 | 58691 / L5 | qwen24-stage/token_shared/layer_shared/lm_head.weight |
| 174 | 127234 | 67038 / L1 | 67045 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 175 | 135583 | 67046 / L1 | 67047 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 176 | 135587 | 67044 / L1 | 67048 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 177 | 135658 | 67071 / L1 | 67079 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 178 | 135671 | 67074 / L1 | 67080 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 179 | 135682 | 67078 / L1 | 67081 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 180 | 135699 | 67097 / L1 | 67098 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 181 | 140091 | 69161 / L1 | 69166 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 182 | 140095 | 69163 / L1 | 69167 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 183 | 140099 | 69165 / L1 | 69168 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 184 | 140185 | 69207 / L1 | 69208 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 185 | 140304 | 69261 / L1 | 69262 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 186 | 140371 | 69292 / L1 | 69296 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 187 | 140373 | 69295 / L1 | 69297 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 188 | 140773 | 69910 / L1 | 69949 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 189 | 141690 | 69911 / L1 | 69950 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 190 | 141693 | 69946 / L1 | 69951 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 191 | 141695 | 69947 / L1 | 69952 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 192 | 141697 | 69948 / L1 | 69953 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block |
| 193 | 141748 | 70043 / L1 | 71208 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 194 | 141825 | 70087 / L1 | 71209 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 195 | 141872 | 70722 / L1 | 71210 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 196 | 142009 | 71105 / L1 | 71211 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 197 | 142018 | 71207 / L1 | 71213 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 198 | 142115 | 71258 / L1 | 71259 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 199 | 142119 | 71248 / L1 | 71260 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 200 | 142121 | 71251 / L1 | 71261 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 201 | 142125 | 71254 / L1 | 71262 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 202 | 142129 | 71257 / L1 | 71263 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 203 | 144380 | 71205 / L1 | 71212 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_01/attention |
| 204 | 144425 | 71291 / L1 | 71292 / L5 | qwen24-stage/token_0-1/layer_01/attention |
| 205 | 146588 | 72356 / L1 | 72360 / L5 | qwen24-stage/token_0-1/layer_01/model_list.1.attention.o_weight.weight |
| 206 | 146592 | 72359 / L1 | 72361 / L5 | qwen24-stage/token_0-1/layer_01/model_list.1.attention.o_weight.weight |
| 207 | 146678 | 72400 / L1 | 72401 / L5 | qwen24-stage/token_0-1/layer_01/model_list.1.attention.o_weight.weight |
| 208 | 146797 | 72454 / L1 | 72455 / L5 | qwen24-stage/token_0-1/layer_01/model_list.1.attention.o_weight.weight |
| 209 | 146885 | 72498 / L1 | 72499 / L5 | qwen24-stage/token_0-1/layer_01/model_list.1.attention.o_weight.weight |
| 210 | 155249 | 80846 / L1 | 80853 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 211 | 163598 | 80854 / L1 | 80855 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 212 | 163602 | 80852 / L1 | 80856 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 213 | 163673 | 80879 / L1 | 80887 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 214 | 163686 | 80882 / L1 | 80888 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 215 | 163697 | 80886 / L1 | 80889 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 216 | 163714 | 80905 / L1 | 80906 / L5 | qwen24-stage/token_0-1/layer_01/_decoder_block |
| 217 | 168106 | 82969 / L1 | 82974 / L5 | qwen24-stage/token_0-1/layer_02/attention |
| 218 | 168110 | 82971 / L1 | 82975 / L5 | qwen24-stage/token_0-1/layer_02/attention |
| 219 | 168114 | 82973 / L1 | 82976 / L5 | qwen24-stage/token_0-1/layer_02/attention |
| 220 | 168200 | 83015 / L1 | 83016 / L5 | qwen24-stage/token_0-1/layer_02/attention |
| 221 | 168319 | 83069 / L1 | 83070 / L5 | qwen24-stage/token_0-1/layer_02/attention |
| 222 | 168386 | 83100 / L1 | 83104 / L5 | qwen24-stage/token_0-1/layer_02/attention |
| 223 | 168388 | 83103 / L1 | 83105 / L5 | qwen24-stage/token_0-1/layer_02/attention |
| 224 | 168788 | 83718 / L1 | 83757 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 225 | 169705 | 83719 / L1 | 83758 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 226 | 169708 | 83754 / L1 | 83759 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 227 | 169710 | 83755 / L1 | 83760 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 228 | 169712 | 83756 / L1 | 83761 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 229 | 169763 | 83851 / L1 | 85016 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 230 | 169840 | 83895 / L1 | 85017 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 231 | 169887 | 84530 / L1 | 85018 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 232 | 170024 | 84913 / L1 | 85019 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 233 | 170033 | 85015 / L1 | 85021 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 234 | 170130 | 85066 / L1 | 85067 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 235 | 170134 | 85056 / L1 | 85068 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 236 | 170136 | 85059 / L1 | 85069 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 237 | 170140 | 85062 / L1 | 85070 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 238 | 170144 | 85065 / L1 | 85071 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 239 | 172395 | 85013 / L1 | 85020 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 240 | 172440 | 85099 / L1 | 85100 / L5 | qwen24-stage/token_0-1/layer_02/model_list.2.attention.o_weight.weight |
| 241 | 174603 | 86164 / L1 | 86168 / L5 | qwen24-stage/token_0-1/layer_02/_decoder_block |
| 242 | 174607 | 86167 / L1 | 86169 / L5 | qwen24-stage/token_0-1/layer_02/_decoder_block |
| 243 | 174693 | 86208 / L1 | 86209 / L5 | qwen24-stage/token_0-1/layer_02/_decoder_block |
| 244 | 174812 | 86262 / L1 | 86263 / L5 | qwen24-stage/token_0-1/layer_02/_decoder_block |
| 245 | 174900 | 86306 / L1 | 86307 / L5 | qwen24-stage/token_0-1/layer_02/_decoder_block |
| 246 | 183264 | 94654 / L1 | 94661 / L5 | qwen24-stage/token_0-1/layer_03/attention |
| 247 | 191613 | 94662 / L1 | 94663 / L5 | qwen24-stage/token_0-1/layer_03/attention |
| 248 | 191617 | 94660 / L1 | 94664 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_03/attention |
| 249 | 191688 | 94687 / L1 | 94695 / L5 | qwen24-stage/token_0-1/layer_03/gqa_attention |
| 250 | 191701 | 94690 / L1 | 94696 / L5 | qwen24-stage/token_0-1/layer_03/attention |
| 251 | 191712 | 94694 / L1 | 94697 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_03/attention |
| 252 | 191729 | 94713 / L1 | 94714 / L5 | qwen24-stage/token_0-1/layer_03/rope |
| 253 | 196121 | 96777 / L1 | 96782 / L5 | qwen24-stage/token_0-1/layer_03/model_list.3.attention.o_weight.weight |
| 254 | 196125 | 96779 / L1 | 96783 / L5 | qwen24-stage/token_0-1/layer_03/model_list.3.attention.o_weight.weight |
| 255 | 196129 | 96781 / L1 | 96784 / L5 | qwen24-stage/token_0-1/layer_03/model_list.3.attention.o_weight.weight |
| 256 | 196215 | 96823 / L1 | 96824 / L5 | qwen24-stage/token_0-1/layer_03/model_list.3.attention.o_weight.weight |
| 257 | 196334 | 96877 / L1 | 96878 / L5 | qwen24-stage/token_0-1/layer_03/model_list.3.attention.o_weight.weight |
| 258 | 196401 | 96908 / L1 | 96912 / L5 | qwen24-stage/token_0-1/layer_03/model_list.3.attention.o_weight.weight |
| 259 | 196403 | 96911 / L1 | 96913 / L5 | qwen24-stage/token_0-1/layer_03/model_list.3.attention.o_weight.weight |
| 260 | 196803 | 97526 / L1 | 97565 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 261 | 197720 | 97527 / L1 | 97566 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 262 | 197723 | 97562 / L1 | 97567 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 263 | 197725 | 97563 / L1 | 97568 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 264 | 197727 | 97564 / L1 | 97569 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 265 | 197778 | 97659 / L1 | 98824 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 266 | 197855 | 97703 / L1 | 98825 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 267 | 197902 | 98338 / L1 | 98826 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 268 | 198039 | 98721 / L1 | 98827 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 269 | 198048 | 98823 / L1 | 98829 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 270 | 198145 | 98874 / L1 | 98875 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 271 | 198149 | 98864 / L1 | 98876 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 272 | 198151 | 98867 / L1 | 98877 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 273 | 198155 | 98870 / L1 | 98878 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 274 | 198159 | 98873 / L1 | 98879 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 275 | 200410 | 98821 / L1 | 98828 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 276 | 200455 | 98907 / L1 | 98908 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 277 | 202618 | 99972 / L1 | 99976 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 278 | 202622 | 99975 / L1 | 99977 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 279 | 202708 | 100016 / L1 | 100017 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 280 | 202827 | 100070 / L1 | 100071 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 281 | 202915 | 100114 / L1 | 100115 / L5 | qwen24-stage/token_0-1/layer_03/_decoder_block |
| 282 | 211279 | 108462 / L1 | 108469 / L5 | qwen24-stage/token_0-1/layer_04/model_list.4.attention.o_weight.weight |
| 283 | 219628 | 108470 / L1 | 108471 / L5 | qwen24-stage/token_0-1/layer_04/model_list.4.attention.o_weight.weight |
| 284 | 219632 | 108468 / L1 | 108472 / L5 | qwen24-stage/token_0-1/layer_04/model_list.4.attention.o_weight.weight |
| 285 | 219703 | 108495 / L1 | 108503 / L5 | qwen24-stage/token_0-1/layer_04/model_list.4.attention.o_weight.weight |
| 286 | 219716 | 108498 / L1 | 108504 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_04/model_list.4.attention.o_weight.weight |
| 287 | 219727 | 108502 / L1 | 108505 / L5 | qwen24-stage/token_0-1/layer_04/model_list.4.attention.o_weight.weight |
| 288 | 219744 | 108521 / L1 | 108522 / L5 | qwen24-stage/token_0-1/layer_04/model_list.4.attention.o_weight.weight |
| 289 | 224136 | 110585 / L1 | 110590 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 290 | 224140 | 110587 / L1 | 110591 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 291 | 224144 | 110589 / L1 | 110592 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 292 | 224230 | 110631 / L1 | 110632 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 293 | 224349 | 110685 / L1 | 110686 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 294 | 224416 | 110716 / L1 | 110720 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 295 | 224418 | 110719 / L1 | 110721 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 296 | 224818 | 111334 / L1 | 111373 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 297 | 225735 | 111335 / L1 | 111374 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 298 | 225738 | 111370 / L1 | 111375 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 299 | 225740 | 111371 / L1 | 111376 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 300 | 225742 | 111372 / L1 | 111377 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 301 | 225793 | 111467 / L1 | 112632 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 302 | 225870 | 111511 / L1 | 112633 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 303 | 225917 | 112146 / L1 | 112634 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 304 | 226054 | 112529 / L1 | 112635 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 305 | 226063 | 112631 / L1 | 112637 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 306 | 226160 | 112682 / L1 | 112683 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 307 | 226164 | 112672 / L1 | 112684 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 308 | 226166 | 112675 / L1 | 112685 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 309 | 226170 | 112678 / L1 | 112686 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 310 | 226174 | 112681 / L1 | 112687 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 311 | 228425 | 112629 / L1 | 112636 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 312 | 228470 | 112715 / L1 | 112716 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 313 | 230633 | 113780 / L1 | 113784 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 314 | 230637 | 113783 / L1 | 113785 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 315 | 230723 | 113824 / L1 | 113825 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 316 | 230842 | 113878 / L1 | 113879 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 317 | 230930 | 113922 / L1 | 113923 / L5 | qwen24-stage/token_0-1/layer_04/_decoder_block |
| 318 | 239294 | 122270 / L1 | 122277 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 319 | 247643 | 122278 / L1 | 122279 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 320 | 247647 | 122276 / L1 | 122280 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 321 | 247718 | 122303 / L1 | 122311 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 322 | 247731 | 122306 / L1 | 122312 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 323 | 247742 | 122310 / L1 | 122313 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 324 | 247759 | 122329 / L1 | 122330 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 325 | 252151 | 124393 / L1 | 124398 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 326 | 252155 | 124395 / L1 | 124399 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 327 | 252159 | 124397 / L1 | 124400 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 328 | 252245 | 124439 / L1 | 124440 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 329 | 252364 | 124493 / L1 | 124494 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 330 | 252431 | 124524 / L1 | 124528 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 331 | 252433 | 124527 / L1 | 124529 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 332 | 252833 | 125142 / L1 | 125181 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 333 | 253750 | 125143 / L1 | 125182 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 334 | 253753 | 125178 / L1 | 125183 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 335 | 253755 | 125179 / L1 | 125184 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 336 | 253757 | 125180 / L1 | 125185 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 337 | 253808 | 125275 / L1 | 126440 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 338 | 253885 | 125319 / L1 | 126441 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 339 | 253932 | 125954 / L1 | 126442 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 340 | 254069 | 126337 / L1 | 126443 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 341 | 254078 | 126439 / L1 | 126445 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 342 | 254175 | 126490 / L1 | 126491 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 343 | 254179 | 126480 / L1 | 126492 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 344 | 254181 | 126483 / L1 | 126493 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 345 | 254185 | 126486 / L1 | 126494 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 346 | 254189 | 126489 / L1 | 126495 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 347 | 256440 | 126437 / L1 | 126444 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 348 | 256485 | 126523 / L1 | 126524 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 349 | 258648 | 127588 / L1 | 127592 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 350 | 258652 | 127591 / L1 | 127593 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 351 | 258738 | 127632 / L1 | 127633 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 352 | 258857 | 127686 / L1 | 127687 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 353 | 258945 | 127730 / L1 | 127731 / L5 | qwen24-stage/token_0-1/layer_05/_decoder_block |
| 354 | 267309 | 136078 / L1 | 136085 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 355 | 275658 | 136086 / L1 | 136087 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 356 | 275662 | 136084 / L1 | 136088 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 357 | 275733 | 136111 / L1 | 136119 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 358 | 275746 | 136114 / L1 | 136120 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 359 | 275757 | 136118 / L1 | 136121 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 360 | 275774 | 136137 / L1 | 136138 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 361 | 280166 | 138201 / L1 | 138206 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block ; qwen24-stage/token_0-1/layer_06/_decoder_block |
| 362 | 280170 | 138203 / L1 | 138207 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 363 | 280174 | 138205 / L1 | 138208 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 364 | 280260 | 138247 / L1 | 138248 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 365 | 280379 | 138301 / L1 | 138302 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 366 | 280446 | 138332 / L1 | 138336 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 367 | 280448 | 138335 / L1 | 138337 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 368 | 280848 | 138950 / L1 | 138989 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 369 | 281765 | 138951 / L1 | 138990 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 370 | 281768 | 138986 / L1 | 138991 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 371 | 281770 | 138987 / L1 | 138992 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 372 | 281772 | 138988 / L1 | 138993 / L5 | qwen24-stage/token_0-1/layer_06/_decoder_block |
| 373 | 281823 | 139083 / L1 | 140248 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 374 | 281900 | 139127 / L1 | 140249 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block ; qwen24-stage/token_0-1/layer_07/_decoder_block |
| 375 | 281947 | 139762 / L1 | 140250 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 376 | 282084 | 140145 / L1 | 140251 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 377 | 282093 | 140247 / L1 | 140253 / L5 | qwen24-stage/token_0-1/layer_00/_decoder_block ; qwen24-stage/token_0-1/layer_07/_decoder_block |
| 378 | 282190 | 140298 / L1 | 140299 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 379 | 282194 | 140288 / L1 | 140300 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 380 | 282196 | 140291 / L1 | 140301 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 381 | 282200 | 140294 / L1 | 140302 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 382 | 282204 | 140297 / L1 | 140303 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 383 | 284455 | 140245 / L1 | 140252 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 384 | 284500 | 140331 / L1 | 140332 / L5 | qwen24-stage/token_0-1/layer_07/attention |
| 385 | 286663 | 141396 / L1 | 141400 / L5 | qwen24-stage/token_0-1/layer_07/model_list.7.attention.o_weight.weight |
| 386 | 286667 | 141399 / L1 | 141401 / L5 | qwen24-stage/token_0-1/layer_07/model_list.7.attention.o_weight.weight |
| 387 | 286753 | 141440 / L1 | 141441 / L5 | qwen24-stage/token_0-1/layer_07/model_list.7.attention.o_weight.weight |
| 388 | 286872 | 141494 / L1 | 141495 / L5 | qwen24-stage/token_0-1/layer_07/model_list.7.attention.o_weight.weight |
| 389 | 286960 | 141538 / L1 | 141539 / L5 | qwen24-stage/token_0-1/layer_07/model_list.7.attention.o_weight.weight |
| 390 | 295324 | 149886 / L1 | 149893 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 391 | 303673 | 149894 / L1 | 149895 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 392 | 303677 | 149892 / L1 | 149896 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 393 | 303748 | 149919 / L1 | 149927 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 394 | 303761 | 149922 / L1 | 149928 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 395 | 303772 | 149926 / L1 | 149929 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 396 | 303789 | 149945 / L1 | 149946 / L5 | qwen24-stage/token_0-1/layer_07/_decoder_block |
| 397 | 308181 | 152009 / L1 | 152014 / L5 | qwen24-stage/token_0-1/layer_08/attention |
| 398 | 308185 | 152011 / L1 | 152015 / L5 | qwen24-stage/token_0-1/layer_08/attention |
| 399 | 308189 | 152013 / L1 | 152016 / L5 | qwen24-stage/token_0-1/layer_08/attention |
| 400 | 308275 | 152055 / L1 | 152056 / L5 | qwen24-stage/token_0-1/layer_08/attention |
| 401 | 308394 | 152109 / L1 | 152110 / L5 | qwen24-stage/token_0-1/layer_08/attention |
| 402 | 308461 | 152140 / L1 | 152144 / L5 | qwen24-stage/token_0-1/layer_08/attention |
| 403 | 308463 | 152143 / L1 | 152145 / L5 | qwen24-stage/token_0-1/layer_08/attention |
| 404 | 308863 | 152758 / L1 | 152797 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 405 | 309780 | 152759 / L1 | 152798 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 406 | 309783 | 152794 / L1 | 152799 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 407 | 309785 | 152795 / L1 | 152800 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 408 | 309787 | 152796 / L1 | 152801 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 409 | 309838 | 152891 / L1 | 154056 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 410 | 309915 | 152935 / L1 | 154057 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 411 | 309962 | 153570 / L1 | 154058 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 412 | 310099 | 153953 / L1 | 154059 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 413 | 310108 | 154055 / L1 | 154061 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 414 | 310205 | 154106 / L1 | 154107 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 415 | 310209 | 154096 / L1 | 154108 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 416 | 310211 | 154099 / L1 | 154109 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 417 | 310215 | 154102 / L1 | 154110 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 418 | 310219 | 154105 / L1 | 154111 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 419 | 312470 | 154053 / L1 | 154060 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 420 | 312515 | 154139 / L1 | 154140 / L5 | qwen24-stage/token_0-1/layer_08/model_list.8.attention.o_weight.weight |
| 421 | 314678 | 155204 / L1 | 155208 / L5 | qwen24-stage/token_0-1/layer_08/_decoder_block |
| 422 | 314682 | 155207 / L1 | 155209 / L5 | qwen24-stage/token_0-1/layer_08/_decoder_block |
| 423 | 314768 | 155248 / L1 | 155249 / L5 | qwen24-stage/token_0-1/layer_08/_decoder_block |
| 424 | 314887 | 155302 / L1 | 155303 / L5 | qwen24-stage/token_0-1/layer_08/_decoder_block |
| 425 | 314975 | 155346 / L1 | 155347 / L5 | qwen24-stage/token_0-1/layer_08/_decoder_block |
| 426 | 323339 | 163694 / L1 | 163701 / L5 | qwen24-stage/token_0-1/layer_09/attention |
| 427 | 331688 | 163702 / L1 | 163703 / L5 | qwen24-stage/token_0-1/layer_09/attention |
| 428 | 331692 | 163700 / L1 | 163704 / L5 | qwen24-stage/token_0-1/layer_09/attention |
| 429 | 331763 | 163727 / L1 | 163735 / L5 | qwen24-stage/token_0-1/layer_09/attention |
| 430 | 331776 | 163730 / L1 | 163736 / L5 | qwen24-stage/token_0-1/layer_09/attention |
| 431 | 331787 | 163734 / L1 | 163737 / L5 | qwen24-stage/token_0-1/layer_09/attention |
| 432 | 331804 | 163753 / L1 | 163754 / L5 | qwen24-stage/token_0-1/layer_09/attention |
| 433 | 336196 | 165817 / L1 | 165822 / L5 | qwen24-stage/token_0-1/layer_09/model_list.9.attention.o_weight.weight |
| 434 | 336200 | 165819 / L1 | 165823 / L5 | qwen24-stage/token_0-1/layer_09/model_list.9.attention.o_weight.weight |
| 435 | 336204 | 165821 / L1 | 165824 / L5 | qwen24-stage/token_0-1/layer_09/model_list.9.attention.o_weight.weight |
| 436 | 336290 | 165863 / L1 | 165864 / L5 | qwen24-stage/token_0-1/layer_09/model_list.9.attention.o_weight.weight |
| 437 | 336409 | 165917 / L1 | 165918 / L5 | qwen24-stage/token_0-1/layer_09/model_list.9.attention.o_weight.weight |
| 438 | 336476 | 165948 / L1 | 165952 / L5 | qwen24-stage/token_0-1/layer_09/model_list.9.attention.o_weight.weight |
| 439 | 336478 | 165951 / L1 | 165953 / L5 | qwen24-stage/token_0-1/layer_09/model_list.9.attention.o_weight.weight |
| 440 | 336878 | 166566 / L1 | 166605 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 441 | 337795 | 166567 / L1 | 166606 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 442 | 337798 | 166602 / L1 | 166607 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 443 | 337800 | 166603 / L1 | 166608 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 444 | 337802 | 166604 / L1 | 166609 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 445 | 337853 | 166699 / L1 | 167864 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 446 | 337930 | 166743 / L1 | 167865 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 447 | 337977 | 167378 / L1 | 167866 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 448 | 338114 | 167761 / L1 | 167867 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 449 | 338123 | 167863 / L1 | 167869 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 450 | 338220 | 167914 / L1 | 167915 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 451 | 338224 | 167904 / L1 | 167916 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 452 | 338226 | 167907 / L1 | 167917 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 453 | 338230 | 167910 / L1 | 167918 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 454 | 338234 | 167913 / L1 | 167919 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 455 | 340485 | 167861 / L1 | 167868 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 456 | 340530 | 167947 / L1 | 167948 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 457 | 342693 | 169012 / L1 | 169016 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 458 | 342697 | 169015 / L1 | 169017 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 459 | 342783 | 169056 / L1 | 169057 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 460 | 342902 | 169110 / L1 | 169111 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 461 | 342990 | 169154 / L1 | 169155 / L5 | qwen24-stage/token_0-1/layer_09/_decoder_block |
| 462 | 351354 | 177502 / L1 | 177509 / L5 | qwen24-stage/token_0-1/layer_10/model_list.10.attention.o_weight.weight |
| 463 | 359703 | 177510 / L1 | 177511 / L5 | qwen24-stage/token_0-1/layer_10/model_list.10.attention.o_weight.weight |
| 464 | 359707 | 177508 / L1 | 177512 / L5 | qwen24-stage/token_0-1/layer_10/model_list.10.attention.o_weight.weight |
| 465 | 359778 | 177535 / L1 | 177543 / L5 | qwen24-stage/token_0-1/layer_10/model_list.10.attention.o_weight.weight |
| 466 | 359791 | 177538 / L1 | 177544 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_10/model_list.10.attention.o_weight.weight |
| 467 | 359802 | 177542 / L1 | 177545 / L5 | qwen24-stage/token_0-1/layer_10/model_list.10.attention.o_weight.weight |
| 468 | 359819 | 177561 / L1 | 177562 / L5 | qwen24-stage/token_0-1/layer_10/model_list.10.attention.o_weight.weight |
| 469 | 364211 | 179625 / L1 | 179630 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 470 | 364215 | 179627 / L1 | 179631 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 471 | 364219 | 179629 / L1 | 179632 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 472 | 364305 | 179671 / L1 | 179672 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 473 | 364424 | 179725 / L1 | 179726 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 474 | 364491 | 179756 / L1 | 179760 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 475 | 364493 | 179759 / L1 | 179761 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 476 | 364893 | 180374 / L1 | 180413 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 477 | 365810 | 180375 / L1 | 180414 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 478 | 365813 | 180410 / L1 | 180415 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 479 | 365815 | 180411 / L1 | 180416 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 480 | 365817 | 180412 / L1 | 180417 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 481 | 365868 | 180507 / L1 | 181672 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 482 | 365945 | 180551 / L1 | 181673 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 483 | 365992 | 181186 / L1 | 181674 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 484 | 366129 | 181569 / L1 | 181675 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 485 | 366138 | 181671 / L1 | 181677 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 486 | 366235 | 181722 / L1 | 181723 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 487 | 366239 | 181712 / L1 | 181724 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 488 | 366241 | 181715 / L1 | 181725 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 489 | 366245 | 181718 / L1 | 181726 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 490 | 366249 | 181721 / L1 | 181727 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 491 | 368500 | 181669 / L1 | 181676 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 492 | 368545 | 181755 / L1 | 181756 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 493 | 370708 | 182820 / L1 | 182824 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 494 | 370712 | 182823 / L1 | 182825 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 495 | 370798 | 182864 / L1 | 182865 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 496 | 370917 | 182918 / L1 | 182919 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 497 | 371005 | 182962 / L1 | 182963 / L5 | qwen24-stage/token_0-1/layer_10/_decoder_block |
| 498 | 379369 | 191310 / L1 | 191317 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 499 | 387718 | 191318 / L1 | 191319 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 500 | 387722 | 191316 / L1 | 191320 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 501 | 387793 | 191343 / L1 | 191351 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 502 | 387806 | 191346 / L1 | 191352 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 503 | 387817 | 191350 / L1 | 191353 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 504 | 387834 | 191369 / L1 | 191370 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 505 | 392226 | 193433 / L1 | 193438 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 506 | 392230 | 193435 / L1 | 193439 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 507 | 392234 | 193437 / L1 | 193440 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 508 | 392320 | 193479 / L1 | 193480 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 509 | 392439 | 193533 / L1 | 193534 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 510 | 392506 | 193564 / L1 | 193568 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 511 | 392508 | 193567 / L1 | 193569 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 512 | 392908 | 194182 / L1 | 194221 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 513 | 393825 | 194183 / L1 | 194222 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 514 | 393828 | 194218 / L1 | 194223 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 515 | 393830 | 194219 / L1 | 194224 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 516 | 393832 | 194220 / L1 | 194225 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 517 | 393883 | 194315 / L1 | 195480 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 518 | 393960 | 194359 / L1 | 195481 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 519 | 394007 | 194994 / L1 | 195482 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 520 | 394144 | 195377 / L1 | 195483 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 521 | 394153 | 195479 / L1 | 195485 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 522 | 394250 | 195530 / L1 | 195531 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 523 | 394254 | 195520 / L1 | 195532 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 524 | 394256 | 195523 / L1 | 195533 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 525 | 394260 | 195526 / L1 | 195534 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 526 | 394264 | 195529 / L1 | 195535 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 527 | 396515 | 195477 / L1 | 195484 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 528 | 396560 | 195563 / L1 | 195564 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 529 | 398723 | 196628 / L1 | 196632 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 530 | 398727 | 196631 / L1 | 196633 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 531 | 398813 | 196672 / L1 | 196673 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 532 | 398932 | 196726 / L1 | 196727 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 533 | 399020 | 196770 / L1 | 196771 / L5 | qwen24-stage/token_0-1/layer_11/_decoder_block |
| 534 | 407384 | 205118 / L1 | 205125 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 535 | 415733 | 205126 / L1 | 205127 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 536 | 415737 | 205124 / L1 | 205128 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 537 | 415808 | 205151 / L1 | 205159 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 538 | 415821 | 205154 / L1 | 205160 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 539 | 415832 | 205158 / L1 | 205161 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 540 | 415849 | 205177 / L1 | 205178 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 541 | 420241 | 207241 / L1 | 207246 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 542 | 420245 | 207243 / L1 | 207247 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 543 | 420249 | 207245 / L1 | 207248 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 544 | 420335 | 207287 / L1 | 207288 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 545 | 420454 | 207341 / L1 | 207342 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 546 | 420521 | 207372 / L1 | 207376 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 547 | 420523 | 207375 / L1 | 207377 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 548 | 420923 | 207990 / L1 | 208029 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 549 | 421840 | 207991 / L1 | 208030 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 550 | 421843 | 208026 / L1 | 208031 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 551 | 421845 | 208027 / L1 | 208032 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 552 | 421847 | 208028 / L1 | 208033 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 553 | 421898 | 208123 / L1 | 209288 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 554 | 421975 | 208167 / L1 | 209289 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 555 | 422022 | 208802 / L1 | 209290 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 556 | 422159 | 209185 / L1 | 209291 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 557 | 422168 | 209287 / L1 | 209293 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 558 | 422265 | 209338 / L1 | 209339 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 559 | 422269 | 209328 / L1 | 209340 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 560 | 422271 | 209331 / L1 | 209341 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 561 | 422275 | 209334 / L1 | 209342 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 562 | 422279 | 209337 / L1 | 209343 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 563 | 424530 | 209285 / L1 | 209292 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 564 | 424575 | 209371 / L1 | 209372 / L5 | qwen24-stage/token_0-1/layer_12/_decoder_block |
| 565 | 426738 | 210436 / L1 | 210440 / L5 | qwen24-stage/token_0-1/layer_13/model_list.13.attention.o_weight.weight |
| 566 | 426742 | 210439 / L1 | 210441 / L5 | qwen24-stage/token_0-1/layer_13/model_list.13.attention.o_weight.weight |
| 567 | 426828 | 210480 / L1 | 210481 / L5 | qwen24-stage/token_0-1/layer_13/model_list.13.attention.o_weight.weight |
| 568 | 426947 | 210534 / L1 | 210535 / L5 | qwen24-stage/token_0-1/layer_13/model_list.13.attention.o_weight.weight |
| 569 | 427035 | 210578 / L1 | 210579 / L5 | qwen24-stage/token_0-1/layer_13/model_list.13.attention.o_weight.weight |
| 570 | 435399 | 218926 / L1 | 218933 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 571 | 443748 | 218934 / L1 | 218935 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 572 | 443752 | 218932 / L1 | 218936 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 573 | 443823 | 218959 / L1 | 218967 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 574 | 443836 | 218962 / L1 | 218968 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 575 | 443847 | 218966 / L1 | 218969 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 576 | 443864 | 218985 / L1 | 218986 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 577 | 448256 | 221049 / L1 | 221054 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_13/_decoder_block |
| 578 | 448260 | 221051 / L1 | 221055 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 579 | 448264 | 221053 / L1 | 221056 / L5 | qwen24-stage/token_0-1/layer_13/_decoder_block |
| 580 | 448350 | 221095 / L1 | 221096 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 581 | 448469 | 221149 / L1 | 221150 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 582 | 448536 | 221180 / L1 | 221184 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 583 | 448538 | 221183 / L1 | 221185 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 584 | 448938 | 221798 / L1 | 221837 / L5 | qwen24-stage/token_0-1/layer_14/rope |
| 585 | 449855 | 221799 / L1 | 221838 / L5 | qwen24-stage/token_0-1/layer_00/rope ; qwen24-stage/token_0-1/layer_14/rope |
| 586 | 449858 | 221834 / L1 | 221839 / L5 | qwen24-stage/token_0-1/layer_14/rope |
| 587 | 449860 | 221835 / L1 | 221840 / L5 | qwen24-stage/token_0-1/layer_14/rope |
| 588 | 449862 | 221836 / L1 | 221841 / L5 | qwen24-stage/token_0-1/layer_00/rope ; qwen24-stage/token_0-1/layer_14/rope |
| 589 | 449913 | 221931 / L1 | 223096 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 590 | 449990 | 221975 / L1 | 223097 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 591 | 450037 | 222610 / L1 | 223098 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 592 | 450174 | 222993 / L1 | 223099 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 593 | 450183 | 223095 / L1 | 223101 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 594 | 450280 | 223146 / L1 | 223147 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 595 | 450284 | 223136 / L1 | 223148 / L5 | qwen24-stage/token_0-1/layer_00/gqa_attention ; qwen24-stage/token_0-1/layer_14/gqa_attention |
| 596 | 450286 | 223139 / L1 | 223149 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 597 | 450290 | 223142 / L1 | 223150 / L5 | qwen24-stage/token_0-1/layer_14/model_list.14.attention.o_weight.weight |
| 598 | 450294 | 223145 / L1 | 223151 / L5 | qwen24-stage/token_0-1/layer_14/model_list.14.attention.o_weight.weight |
| 599 | 452545 | 223093 / L1 | 223100 / L5 | qwen24-stage/token_0-1/layer_14/gqa_attention |
| 600 | 452590 | 223179 / L1 | 223180 / L5 | qwen24-stage/token_0-1/layer_14/model_list.14.attention.o_weight.weight |
| 601 | 454753 | 224244 / L1 | 224248 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 602 | 454757 | 224247 / L1 | 224249 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 603 | 454843 | 224288 / L1 | 224289 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 604 | 454962 | 224342 / L1 | 224343 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 605 | 455050 | 224386 / L1 | 224387 / L5 | qwen24-stage/token_0-1/layer_14/_decoder_block |
| 606 | 463414 | 232734 / L1 | 232741 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 607 | 471763 | 232742 / L1 | 232743 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 608 | 471767 | 232740 / L1 | 232744 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 609 | 471838 | 232767 / L1 | 232775 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 610 | 471851 | 232770 / L1 | 232776 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 611 | 471862 | 232774 / L1 | 232777 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 612 | 471879 | 232793 / L1 | 232794 / L5 | qwen24-stage/token_0-1/layer_15/attention |
| 613 | 476271 | 234857 / L1 | 234862 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 614 | 476275 | 234859 / L1 | 234863 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 615 | 476279 | 234861 / L1 | 234864 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 616 | 476365 | 234903 / L1 | 234904 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 617 | 476484 | 234957 / L1 | 234958 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 618 | 476551 | 234988 / L1 | 234992 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 619 | 476553 | 234991 / L1 | 234993 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 620 | 476953 | 235606 / L1 | 235645 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 621 | 477870 | 235607 / L1 | 235646 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 622 | 477873 | 235642 / L1 | 235647 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 623 | 477875 | 235643 / L1 | 235648 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 624 | 477877 | 235644 / L1 | 235649 / L5 | qwen24-stage/token_0-1/layer_15/model_list.15.attention.o_weight.weight |
| 625 | 477928 | 235739 / L1 | 236904 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 626 | 478005 | 235783 / L1 | 236905 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 627 | 478052 | 236418 / L1 | 236906 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 628 | 478189 | 236801 / L1 | 236907 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 629 | 478198 | 236903 / L1 | 236909 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 630 | 478295 | 236954 / L1 | 236955 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 631 | 478299 | 236944 / L1 | 236956 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 632 | 478301 | 236947 / L1 | 236957 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 633 | 478305 | 236950 / L1 | 236958 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 634 | 478309 | 236953 / L1 | 236959 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 635 | 480560 | 236901 / L1 | 236908 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 636 | 480605 | 236987 / L1 | 236988 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 637 | 482768 | 238052 / L1 | 238056 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 638 | 482772 | 238055 / L1 | 238057 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 639 | 482858 | 238096 / L1 | 238097 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 640 | 482977 | 238150 / L1 | 238151 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 641 | 483065 | 238194 / L1 | 238195 / L5 | qwen24-stage/token_0-1/layer_15/_decoder_block |
| 642 | 491429 | 246542 / L1 | 246549 / L5 | qwen24-stage/token_0-1/layer_16/model_list.16.attention.o_weight.weight |
| 643 | 499778 | 246550 / L1 | 246551 / L5 | qwen24-stage/token_0-1/layer_16/model_list.16.attention.o_weight.weight |
| 644 | 499782 | 246548 / L1 | 246552 / L5 | qwen24-stage/token_0-1/layer_16/model_list.16.attention.o_weight.weight |
| 645 | 499853 | 246575 / L1 | 246583 / L5 | qwen24-stage/token_0-1/layer_16/model_list.16.attention.o_weight.weight |
| 646 | 499866 | 246578 / L1 | 246584 / L5 | qwen24-stage/token_0-1/layer_16/model_list.16.attention.o_weight.weight |
| 647 | 499877 | 246582 / L1 | 246585 / L5 | qwen24-stage/token_0-1/layer_16/model_list.16.attention.o_weight.weight |
| 648 | 499894 | 246601 / L1 | 246602 / L5 | qwen24-stage/token_0-1/layer_16/model_list.16.attention.o_weight.weight |
| 649 | 504286 | 248665 / L1 | 248670 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 650 | 504290 | 248667 / L1 | 248671 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 651 | 504294 | 248669 / L1 | 248672 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 652 | 504380 | 248711 / L1 | 248712 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 653 | 504499 | 248765 / L1 | 248766 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 654 | 504566 | 248796 / L1 | 248800 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 655 | 504568 | 248799 / L1 | 248801 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 656 | 504968 | 249414 / L1 | 249453 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 657 | 505885 | 249415 / L1 | 249454 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 658 | 505888 | 249450 / L1 | 249455 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 659 | 505890 | 249451 / L1 | 249456 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 660 | 505892 | 249452 / L1 | 249457 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 661 | 505943 | 249547 / L1 | 250712 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 662 | 506020 | 249591 / L1 | 250713 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 663 | 506067 | 250226 / L1 | 250714 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 664 | 506204 | 250609 / L1 | 250715 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 665 | 506213 | 250711 / L1 | 250717 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 666 | 506310 | 250762 / L1 | 250763 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 667 | 506314 | 250752 / L1 | 250764 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 668 | 506316 | 250755 / L1 | 250765 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 669 | 506320 | 250758 / L1 | 250766 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 670 | 506324 | 250761 / L1 | 250767 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 671 | 508575 | 250709 / L1 | 250716 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 672 | 508620 | 250795 / L1 | 250796 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 673 | 510783 | 251860 / L1 | 251864 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 674 | 510787 | 251863 / L1 | 251865 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 675 | 510873 | 251904 / L1 | 251905 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 676 | 510992 | 251958 / L1 | 251959 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 677 | 511080 | 252002 / L1 | 252003 / L5 | qwen24-stage/token_0-1/layer_16/_decoder_block |
| 678 | 519444 | 260350 / L1 | 260357 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 679 | 527793 | 260358 / L1 | 260359 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 680 | 527797 | 260356 / L1 | 260360 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 681 | 527868 | 260383 / L1 | 260391 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 682 | 527881 | 260386 / L1 | 260392 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 683 | 527892 | 260390 / L1 | 260393 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 684 | 527909 | 260409 / L1 | 260410 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 685 | 532301 | 262473 / L1 | 262478 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 686 | 532305 | 262475 / L1 | 262479 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 687 | 532309 | 262477 / L1 | 262480 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 688 | 532395 | 262519 / L1 | 262520 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 689 | 532514 | 262573 / L1 | 262574 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 690 | 532581 | 262604 / L1 | 262608 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 691 | 532583 | 262607 / L1 | 262609 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 692 | 532983 | 263222 / L1 | 263261 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 693 | 533900 | 263223 / L1 | 263262 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 694 | 533903 | 263258 / L1 | 263263 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 695 | 533905 | 263259 / L1 | 263264 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 696 | 533907 | 263260 / L1 | 263265 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 697 | 533958 | 263355 / L1 | 264520 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 698 | 534035 | 263399 / L1 | 264521 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 699 | 534082 | 264034 / L1 | 264522 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 700 | 534219 | 264417 / L1 | 264523 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 701 | 534228 | 264519 / L1 | 264525 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 702 | 534325 | 264570 / L1 | 264571 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 703 | 534329 | 264560 / L1 | 264572 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 704 | 534331 | 264563 / L1 | 264573 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 705 | 534335 | 264566 / L1 | 264574 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 706 | 534339 | 264569 / L1 | 264575 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 707 | 536590 | 264517 / L1 | 264524 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 708 | 536635 | 264603 / L1 | 264604 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 709 | 538798 | 265668 / L1 | 265672 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 710 | 538802 | 265671 / L1 | 265673 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 711 | 538888 | 265712 / L1 | 265713 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 712 | 539007 | 265766 / L1 | 265767 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 713 | 539095 | 265810 / L1 | 265811 / L5 | qwen24-stage/token_0-1/layer_17/_decoder_block |
| 714 | 547459 | 274158 / L1 | 274165 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 715 | 555808 | 274166 / L1 | 274167 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 716 | 555812 | 274164 / L1 | 274168 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 717 | 555883 | 274191 / L1 | 274199 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 718 | 555896 | 274194 / L1 | 274200 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 719 | 555907 | 274198 / L1 | 274201 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 720 | 555924 | 274217 / L1 | 274218 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 721 | 560316 | 276281 / L1 | 276286 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 722 | 560320 | 276283 / L1 | 276287 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 723 | 560324 | 276285 / L1 | 276288 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 724 | 560410 | 276327 / L1 | 276328 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 725 | 560529 | 276381 / L1 | 276382 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 726 | 560596 | 276412 / L1 | 276416 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 727 | 560598 | 276415 / L1 | 276417 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 728 | 560998 | 277030 / L1 | 277069 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 729 | 561915 | 277031 / L1 | 277070 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 730 | 561918 | 277066 / L1 | 277071 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 731 | 561920 | 277067 / L1 | 277072 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 732 | 561922 | 277068 / L1 | 277073 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 733 | 561973 | 277163 / L1 | 278328 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 734 | 562050 | 277207 / L1 | 278329 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 735 | 562097 | 277842 / L1 | 278330 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 736 | 562234 | 278225 / L1 | 278331 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 737 | 562243 | 278327 / L1 | 278333 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 738 | 562340 | 278378 / L1 | 278379 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 739 | 562344 | 278368 / L1 | 278380 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 740 | 562346 | 278371 / L1 | 278381 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 741 | 562350 | 278374 / L1 | 278382 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 742 | 562354 | 278377 / L1 | 278383 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 743 | 564605 | 278325 / L1 | 278332 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 744 | 564650 | 278411 / L1 | 278412 / L5 | qwen24-stage/token_0-1/layer_18/_decoder_block |
| 745 | 566813 | 279476 / L1 | 279480 / L5 | qwen24-stage/token_0-1/layer_19/attention |
| 746 | 566817 | 279479 / L1 | 279481 / L5 | qwen24-stage/token_0-1/layer_19/attention |
| 747 | 566903 | 279520 / L1 | 279521 / L5 | qwen24-stage/token_0-1/layer_19/attention |
| 748 | 567022 | 279574 / L1 | 279575 / L5 | qwen24-stage/token_0-1/layer_19/attention |
| 749 | 567110 | 279618 / L1 | 279619 / L5 | qwen24-stage/token_0-1/layer_19/attention |
| 750 | 575474 | 287966 / L1 | 287973 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 751 | 583823 | 287974 / L1 | 287975 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 752 | 583827 | 287972 / L1 | 287976 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 753 | 583898 | 287999 / L1 | 288007 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 754 | 583911 | 288002 / L1 | 288008 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 755 | 583922 | 288006 / L1 | 288009 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 756 | 583939 | 288025 / L1 | 288026 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 757 | 588331 | 290089 / L1 | 290094 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 758 | 588335 | 290091 / L1 | 290095 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 759 | 588339 | 290093 / L1 | 290096 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 760 | 588425 | 290135 / L1 | 290136 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 761 | 588544 | 290189 / L1 | 290190 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 762 | 588611 | 290220 / L1 | 290224 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 763 | 588613 | 290223 / L1 | 290225 / L5 | qwen24-stage/token_0-1/layer_19/_decoder_block |
| 764 | 589013 | 290838 / L1 | 290877 / L5 | qwen24-stage/token_0-1/layer_20/attention |
| 765 | 589930 | 290839 / L1 | 290878 / L5 | qwen24-stage/token_0-1/layer_20/attention |
| 766 | 589933 | 290874 / L1 | 290879 / L5 | qwen24-stage/token_0-1/layer_20/attention |
| 767 | 589935 | 290875 / L1 | 290880 / L5 | qwen24-stage/token_0-1/layer_20/attention |
| 768 | 589937 | 290876 / L1 | 290881 / L5 | qwen24-stage/token_0-1/layer_20/attention |
| 769 | 589988 | 290971 / L1 | 292136 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 770 | 590065 | 291015 / L1 | 292137 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 771 | 590112 | 291650 / L1 | 292138 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 772 | 590249 | 292033 / L1 | 292139 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 773 | 590258 | 292135 / L1 | 292141 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 774 | 590355 | 292186 / L1 | 292187 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 775 | 590359 | 292176 / L1 | 292188 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 776 | 590361 | 292179 / L1 | 292189 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 777 | 590365 | 292182 / L1 | 292190 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 778 | 590369 | 292185 / L1 | 292191 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 779 | 592620 | 292133 / L1 | 292140 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 780 | 592665 | 292219 / L1 | 292220 / L5 | qwen24-stage/token_0-1/layer_00/attention ; qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 781 | 594828 | 293284 / L1 | 293288 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 782 | 594832 | 293287 / L1 | 293289 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 783 | 594918 | 293328 / L1 | 293329 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 784 | 595037 | 293382 / L1 | 293383 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 785 | 595125 | 293426 / L1 | 293427 / L5 | qwen24-stage/token_0-1/layer_20/model_list.20.attention.o_weight.weight |
| 786 | 603489 | 301774 / L1 | 301781 / L5 | qwen24-stage/token_0-1/layer_20/_decoder_block |
| 787 | 611838 | 301782 / L1 | 301783 / L5 | qwen24-stage/token_0-1/layer_20/_decoder_block |
| 788 | 611842 | 301780 / L1 | 301784 / L5 | qwen24-stage/token_0-1/layer_20/_decoder_block |
| 789 | 611913 | 301807 / L1 | 301815 / L5 | qwen24-stage/token_0-1/layer_20/_decoder_block |
| 790 | 611926 | 301810 / L1 | 301816 / L5 | qwen24-stage/token_0-1/layer_20/_decoder_block |
| 791 | 611937 | 301814 / L1 | 301817 / L5 | qwen24-stage/token_0-1/layer_20/_decoder_block |
| 792 | 611954 | 301833 / L1 | 301834 / L5 | qwen24-stage/token_0-1/layer_20/_decoder_block |
| 793 | 616346 | 303897 / L1 | 303902 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 794 | 616350 | 303899 / L1 | 303903 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 795 | 616354 | 303901 / L1 | 303904 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 796 | 616440 | 303943 / L1 | 303944 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 797 | 616559 | 303997 / L1 | 303998 / L5 | qwen24-stage/token_0-1/layer_21/gqa_attention |
| 798 | 616626 | 304028 / L1 | 304032 / L5 | qwen24-stage/token_0-1/layer_21/gqa_attention |
| 799 | 616628 | 304031 / L1 | 304033 / L5 | qwen24-stage/token_0-1/layer_21/gqa_attention |
| 800 | 617028 | 304646 / L1 | 304685 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 801 | 617945 | 304647 / L1 | 304686 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 802 | 617948 | 304682 / L1 | 304687 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 803 | 617950 | 304683 / L1 | 304688 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 804 | 617952 | 304684 / L1 | 304689 / L5 | qwen24-stage/token_0-1/layer_21/model_list.21.attention.o_weight.weight |
| 805 | 618003 | 304779 / L1 | 305944 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 806 | 618080 | 304823 / L1 | 305945 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 807 | 618127 | 305458 / L1 | 305946 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 808 | 618264 | 305841 / L1 | 305947 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 809 | 618273 | 305943 / L1 | 305949 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 810 | 618370 | 305994 / L1 | 305995 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 811 | 618374 | 305984 / L1 | 305996 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 812 | 618376 | 305987 / L1 | 305997 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 813 | 618380 | 305990 / L1 | 305998 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 814 | 618384 | 305993 / L1 | 305999 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 815 | 620635 | 305941 / L1 | 305948 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 816 | 620680 | 306027 / L1 | 306028 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 817 | 622843 | 307092 / L1 | 307096 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 818 | 622847 | 307095 / L1 | 307097 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 819 | 622933 | 307136 / L1 | 307137 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 820 | 623052 | 307190 / L1 | 307191 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 821 | 623140 | 307234 / L1 | 307235 / L5 | qwen24-stage/token_0-1/layer_21/_decoder_block |
| 822 | 631504 | 315582 / L1 | 315589 / L5 | qwen24-stage/token_0-1/layer_22/gqa_attention |
| 823 | 639854 | 315590 / L1 | 315591 / L5 | qwen24-stage/token_0-1/layer_22/gqa_attention |
| 824 | 639858 | 315588 / L1 | 315592 / L5 | qwen24-stage/token_0-1/layer_22/gqa_attention |
| 825 | 639932 | 315615 / L1 | 315623 / L5 | qwen24-stage/token_0-1/layer_22/model_list.22.attention.o_weight.weight |
| 826 | 639946 | 315618 / L1 | 315624 / L5 | qwen24-stage/token_0-1/layer_22/model_list.22.attention.o_weight.weight |
| 827 | 639957 | 315622 / L1 | 315625 / L5 | qwen24-stage/token_0-1/layer_22/model_list.22.attention.o_weight.weight |
| 828 | 639974 | 315641 / L1 | 315642 / L5 | qwen24-stage/token_0-1/layer_22/model_list.22.attention.o_weight.weight |
| 829 | 644364 | 317707 / L1 | 317710 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 830 | 644368 | 317709 / L1 | 317711 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 831 | 644457 | 317750 / L1 | 317751 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 832 | 644580 | 317804 / L1 | 317805 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 833 | 644648 | 317835 / L1 | 317839 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 834 | 644650 | 317838 / L1 | 317840 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 835 | 645050 | 318453 / L1 | 318492 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 836 | 645972 | 318454 / L1 | 318493 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 837 | 645988 | 318489 / L1 | 318494 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 838 | 645990 | 318490 / L1 | 318495 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 839 | 645992 | 318491 / L1 | 318496 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 840 | 646043 | 318563 / L1 | 318641 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 841 | 646121 | 318595 / L1 | 318642 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 842 | 646161 | 318605 / L1 | 318643 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 843 | 646300 | 318638 / L1 | 318644 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 844 | 646309 | 318640 / L1 | 318645 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 845 | 646407 | 318690 / L1 | 318691 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 846 | 646411 | 318680 / L1 | 318692 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 847 | 646413 | 318683 / L1 | 318693 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 848 | 646417 | 318686 / L1 | 318694 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 849 | 646421 | 318689 / L1 | 318695 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 850 | 646477 | 318723 / L1 | 318724 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 851 | 648666 | 319813 / L1 | 319814 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 852 | 648750 | 319856 / L1 | 319857 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 853 | 648870 | 319915 / L1 | 319919 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 854 | 648874 | 319918 / L1 | 319920 / L5 | qwen24-stage/token_0-1/layer_22/_decoder_block |
| 855 | 665578 | 324114 / L1 | 328273 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 856 | 665634 | 328272 / L1 | 328274 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 857 | 665636 | 328297 / L1 | 328298 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 858 | 665640 | 328286 / L1 | 328299 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 859 | 665653 | 328289 / L1 | 328300 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 860 | 665657 | 328293 / L1 | 328301 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 861 | 665668 | 328296 / L1 | 328302 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 862 | 665712 | 328330 / L1 | 328331 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 863 | 678640 | 334794 / L1 | 334795 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 864 | 678725 | 334835 / L1 | 334836 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 865 | 678852 | 334891 / L1 | 334895 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |
| 866 | 678856 | 334894 / L1 | 334896 / L5 | qwen24-stage/token_0-1/layer_23/swiglu |

## 9. 各阶段执行时的全图同时存活峰值

每行统计执行该阶段操作后，全图仍存活的系数载荷，包含先前阶段未释放值；不是此阶段独占内存。不同类型峰值可能不同时出现，不能把各自最大值相加当作总峰值。

| 源码阶段 | ordinal | 合计MiB | PT MiB | CT MiB |
| --- | --- | --- | --- | --- |
| token_0-1/layer_00/_decoder_block | 126409 | 302.500 | 39.500 | 263.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_00/attention | 142547 | 138.000 | 64.000 | 74.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_00/gqa_attention | 4338 | 95.000 | 64.000 | 31.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_01/_decoder_block | 154379 | 299.000 | 41.000 | 258.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_02/_decoder_block | 182098 | 302.000 | 41.000 | 261.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_03/_decoder_block | 182345 | 299.000 | 41.000 | 258.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_04/_decoder_block | 210091 | 305.500 | 39.500 | 266.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_05/_decoder_block | 238208 | 308.000 | 41.000 | 267.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_06/_decoder_block | 269144 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_07/_decoder_block | 294470 | 314.000 | 41.000 | 273.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_08/_decoder_block | 322253 | 317.000 | 41.000 | 276.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_09/_decoder_block | 350552 | 323.500 | 39.500 | 284.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_10/_decoder_block | 350200 | 319.500 | 39.500 | 280.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_11/_decoder_block | 378327 | 322.500 | 39.500 | 283.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_12/_decoder_block | 406578 | 330.000 | 41.000 | 289.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_13/_decoder_block | 439654 | 223.500 | 39.500 | 184.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_14/_decoder_block | 462368 | 332.000 | 41.000 | 291.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_15/_decoder_block | 490612 | 341.500 | 39.500 | 302.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_16/_decoder_block | 490313 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_17/_decoder_block | 518438 | 338.000 | 41.000 | 297.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_18/_decoder_block | 546177 | 340.500 | 39.500 | 301.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_19/_decoder_block | 580291 | 238.500 | 39.500 | 199.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_20/_decoder_block | 602435 | 347.000 | 41.000 | 306.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_21/_decoder_block | 630242 | 350.000 | 41.000 | 309.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_22/_decoder_block | 630425 | 347.000 | 41.000 | 306.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/_decoder_block | 659240 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/model_list.23.post_Normal.weight | 92051 | 221.500 | 39.500 | 182.000 |
| token_0-1/layer_00/_decoder_block \| token_0-1/layer_23/swiglu | 108971 | 139.000 | 63.000 | 76.000 |
| token_0-1/layer_00/_decoder_block \| token_shared/layer_shared/final_normal.weight | 688948 | 191.000 | 0.000 | 191.000 |
| token_0-1/layer_00/attention | 128637 | 190.500 | 39.500 | 151.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/model_list.0.attention.o_weight.weight | 131409 | 190.500 | 39.500 | 151.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_00/rope | 129887 | 188.500 | 42.500 | 146.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/_decoder_block | 166038 | 121.000 | 39.000 | 82.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/attention | 144184 | 138.000 | 64.000 | 74.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/model_list.1.attention.o_weight.weight | 150299 | 249.500 | 39.500 | 210.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_01/rope | 145427 | 91.000 | 34.000 | 57.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/_decoder_block | 182337 | 299.000 | 41.000 | 258.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/attention | 165404 | 138.500 | 62.500 | 76.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/model_list.2.attention.o_weight.weight | 170441 | 147.500 | 63.500 | 84.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_02/rope | 168488 | 104.000 | 39.000 | 65.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/_decoder_block | 210077 | 304.500 | 39.500 | 265.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/attention | 182420 | 302.500 | 39.500 | 263.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/model_list.3.attention.o_weight.weight | 191967 | 148.000 | 64.000 | 84.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_03/rope | 191716 | 114.500 | 62.500 | 52.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/_decoder_block | 245437 | 201.500 | 39.500 | 162.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/attention | 210097 | 305.000 | 41.000 | 264.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/model_list.4.attention.o_weight.weight | 210315 | 305.500 | 39.500 | 266.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_04/rope | 204512 | 239.000 | 41.000 | 198.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/_decoder_block | 268956 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/attention | 245889 | 202.500 | 39.500 | 163.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/model_list.5.attention.o_weight.weight | 238058 | 308.500 | 39.500 | 269.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_05/rope | 247064 | 194.500 | 42.500 | 152.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/_decoder_block | 282453 | 149.500 | 63.500 | 86.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/attention | 269404 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/model_list.6.attention.o_weight.weight | 272044 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_06/rope | 270579 | 197.500 | 42.500 | 155.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/_decoder_block | 306761 | 104.500 | 11.500 | 93.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/attention | 284461 | 143.000 | 64.000 | 79.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/model_list.7.attention.o_weight.weight | 295615 | 208.500 | 39.500 | 169.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_07/rope | 286602 | 65.500 | 6.500 | 59.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/_decoder_block | 321427 | 304.500 | 39.500 | 265.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/attention | 309098 | 147.000 | 45.000 | 102.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/model_list.8.attention.o_weight.weight | 311723 | 163.000 | 64.000 | 99.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_08/rope | 308980 | 146.000 | 45.000 | 101.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/_decoder_block | 350187 | 317.000 | 41.000 | 276.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/attention | 321962 | 311.000 | 41.000 | 270.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/model_list.9.attention.o_weight.weight | 332231 | 163.000 | 63.000 | 100.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_09/rope | 333347 | 164.000 | 63.000 | 101.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/_decoder_block | 386184 | 217.500 | 39.500 | 178.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/attention | 350216 | 319.500 | 39.500 | 280.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/model_list.10.attention.o_weight.weight | 350404 | 320.000 | 41.000 | 279.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_10/rope | 350964 | 260.000 | 41.000 | 219.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/_decoder_block | 409731 | 220.500 | 39.500 | 181.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/attention | 374426 | 280.500 | 39.500 | 241.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/model_list.11.attention.o_weight.weight | 378125 | 323.500 | 39.500 | 284.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_11/rope | 374420 | 276.500 | 39.500 | 237.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/_decoder_block | 424899 | 141.000 | 54.000 | 87.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/attention | 410095 | 220.500 | 39.500 | 181.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/model_list.12.attention.o_weight.weight | 406559 | 332.500 | 39.500 | 293.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_12/rope | 411270 | 212.500 | 42.500 | 170.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/_decoder_block | 447890 | 105.000 | 9.000 | 96.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/attention | 425558 | 117.000 | 34.000 | 83.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/model_list.13.attention.o_weight.weight | 436250 | 223.500 | 39.500 | 184.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_13/rope | 426447 | 86.000 | 9.000 | 77.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/_decoder_block | 462560 | 332.000 | 41.000 | 291.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/attention | 450057 | 181.500 | 63.500 | 118.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/model_list.14.attention.o_weight.weight | 450074 | 171.500 | 63.500 | 108.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_14/rope | 450045 | 176.500 | 63.500 | 113.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/_decoder_block | 490307 | 334.500 | 39.500 | 295.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/attention | 472436 | 179.000 | 63.000 | 116.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/model_list.15.attention.o_weight.weight | 478067 | 185.000 | 64.000 | 121.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_15/rope | 475515 | 125.000 | 9.000 | 116.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/_decoder_block | 526819 | 232.500 | 39.500 | 193.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/attention | 490393 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/model_list.16.attention.o_weight.weight | 490505 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_16/rope | 488853 | 314.000 | 41.000 | 273.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/_decoder_block | 550334 | 235.500 | 39.500 | 196.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/attention | 518120 | 338.500 | 39.500 | 299.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/model_list.17.attention.o_weight.weight | 518424 | 338.500 | 39.500 | 299.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_17/rope | 511446 | 257.000 | 41.000 | 216.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/_decoder_block | 566079 | 123.500 | 21.500 | 102.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/attention | 550748 | 235.500 | 39.500 | 196.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/model_list.18.attention.o_weight.weight | 546655 | 345.500 | 39.500 | 306.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_18/rope | 551960 | 234.500 | 39.500 | 195.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/_decoder_block | 589088 | 177.000 | 45.000 | 132.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/attention | 570009 | 288.500 | 39.500 | 249.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/model_list.19.attention.o_weight.weight | 577081 | 238.500 | 39.500 | 199.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_19/rope | 575478 | 220.500 | 42.500 | 178.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/_decoder_block | 612491 | 194.000 | 63.000 | 131.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/attention | 590927 | 192.500 | 63.500 | 129.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/model_list.20.attention.o_weight.weight | 597509 | 285.500 | 39.500 | 246.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_20/rope | 591062 | 184.500 | 62.500 | 122.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/_decoder_block | 630414 | 349.500 | 39.500 | 310.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/attention | 612629 | 192.000 | 63.000 | 129.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/model_list.21.attention.o_weight.weight | 618142 | 200.000 | 64.000 | 136.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_21/rope | 613796 | 183.000 | 64.000 | 119.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/_decoder_block | 659126 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/attention | 630506 | 350.000 | 41.000 | 309.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/model_list.22.attention.o_weight.weight | 630606 | 349.500 | 39.500 | 310.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_22/rope | 626398 | 301.500 | 39.500 | 262.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/attention | 659572 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/model_list.23.attention.o_weight.weight | 662408 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_00/attention \| token_0-1/layer_23/rope | 660747 | 168.000 | 26.000 | 142.000 |
| token_0-1/layer_00/attention \| token_shared/layer_shared/pack_kv | 489974 | 329.500 | 39.500 | 290.000 |
| token_0-1/layer_00/gqa_attention | 129945 | 190.500 | 39.500 | 151.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/_decoder_block | 148182 | 224.000 | 41.000 | 183.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_01/gqa_attention | 155588 | 193.500 | 39.500 | 154.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_02/_decoder_block | 179008 | 266.000 | 41.000 | 225.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_02/gqa_attention | 171391 | 147.500 | 63.500 | 84.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_03/_decoder_block | 218048 | 198.500 | 39.500 | 159.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_03/gqa_attention | 191913 | 149.000 | 63.000 | 86.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_04/_decoder_block | 241564 | 194.500 | 42.500 | 152.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_04/gqa_attention | 210189 | 304.500 | 39.500 | 265.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/_decoder_block | 611923 | 158.500 | 62.500 | 96.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_05/gqa_attention | 237487 | 302.500 | 39.500 | 263.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_06/_decoder_block | 279972 | 81.000 | 9.000 | 72.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_06/gqa_attention | 270654 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_07/_decoder_block | 294469 | 311.000 | 41.000 | 270.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_07/gqa_attention | 295549 | 208.500 | 39.500 | 169.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_08/_decoder_block | 317062 | 254.000 | 41.000 | 213.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_08/gqa_attention | 311279 | 158.000 | 63.000 | 95.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_09/_decoder_block | 358739 | 213.500 | 39.500 | 174.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_09/gqa_attention | 332225 | 162.000 | 63.000 | 99.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_10/_decoder_block | 382254 | 216.500 | 39.500 | 177.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_10/gqa_attention | 349776 | 311.000 | 41.000 | 270.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_11/_decoder_block | 397979 | 97.500 | 21.500 | 76.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_11/gqa_attention | 378029 | 323.500 | 39.500 | 284.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_12/_decoder_block | 421009 | 154.000 | 45.000 | 109.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_12/gqa_attention | 411323 | 220.500 | 39.500 | 181.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/_decoder_block | 444408 | 171.000 | 63.000 | 108.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_13/gqa_attention | 437314 | 223.500 | 39.500 | 184.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/_decoder_block | 462992 | 275.500 | 39.500 | 236.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_14/gqa_attention | 450343 | 180.000 | 64.000 | 116.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_15/_decoder_block | 499430 | 229.500 | 39.500 | 190.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_15/gqa_attention | 472178 | 177.000 | 63.000 | 114.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_16/_decoder_block | 522945 | 231.500 | 39.500 | 192.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_16/gqa_attention | 490487 | 335.000 | 41.000 | 294.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_17/_decoder_block | 537325 | 133.000 | 44.000 | 89.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_17/gqa_attention | 518142 | 338.000 | 41.000 | 297.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_18/_decoder_block | 561786 | 147.000 | 45.000 | 102.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_18/gqa_attention | 552072 | 235.500 | 39.500 | 196.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_19/_decoder_block | 586576 | 132.000 | 9.000 | 123.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_19/gqa_attention | 575675 | 238.500 | 39.500 | 199.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_20/_decoder_block | 600894 | 328.500 | 39.500 | 289.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_20/gqa_attention | 591245 | 188.000 | 63.000 | 125.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_21/_decoder_block | 623487 | 271.500 | 39.500 | 232.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_21/gqa_attention | 612545 | 193.000 | 63.000 | 130.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/_decoder_block | 655249 | 168.000 | 26.000 | 142.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_22/gqa_attention | 630666 | 350.000 | 41.000 | 309.000 |
| token_0-1/layer_00/gqa_attention \| token_0-1/layer_23/gqa_attention | 660898 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight | 130051 | 190.500 | 39.500 | 151.000 |
| token_0-1/layer_00/model_list.0.attention.o_weight.weight \| token_0-1/layer_22/_decoder_block | 623005 | 138.500 | 33.500 | 105.000 |
| token_0-1/layer_00/rope | 129873 | 190.500 | 39.500 | 151.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_01/rope | 145514 | 88.500 | 31.500 | 57.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_02/rope | 168804 | 112.000 | 45.000 | 67.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_03/rope | 192110 | 149.000 | 63.000 | 86.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_04/rope | 208640 | 287.000 | 41.000 | 246.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_05/rope | 247085 | 201.500 | 39.500 | 162.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_06/rope | 270614 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_07/rope | 284585 | 124.500 | 61.500 | 63.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_08/rope | 309136 | 146.000 | 45.000 | 101.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_09/rope | 333339 | 164.000 | 63.000 | 101.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_10/rope | 350284 | 320.000 | 41.000 | 279.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_11/rope | 377520 | 317.000 | 41.000 | 276.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_12/rope | 411291 | 220.500 | 39.500 | 181.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_13/rope | 425534 | 111.500 | 36.500 | 75.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_14/rope | 450372 | 175.000 | 64.000 | 111.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_15/rope | 473166 | 178.000 | 63.000 | 115.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_16/rope | 490403 | 334.500 | 39.500 | 295.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_17/rope | 518666 | 338.500 | 39.500 | 299.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_18/rope | 551982 | 235.500 | 39.500 | 196.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_19/rope | 573099 | 324.500 | 39.500 | 285.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_20/rope | 591011 | 192.500 | 63.500 | 129.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_21/rope | 612182 | 193.000 | 63.000 | 130.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_22/rope | 630526 | 349.500 | 39.500 | 310.000 |
| token_0-1/layer_00/rope \| token_0-1/layer_23/rope | 660768 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_01/_decoder_block | 154424 | 305.500 | 39.500 | 266.000 |
| token_0-1/layer_01/attention | 142134 | 145.000 | 63.500 | 81.500 |
| token_0-1/layer_01/gqa_attention | 155596 | 193.500 | 39.500 | 154.000 |
| token_0-1/layer_01/model_list.1.attention.o_weight.weight | 154419 | 297.500 | 39.500 | 258.000 |
| token_0-1/layer_01/rope | 145425 | 91.000 | 34.000 | 57.000 |
| token_0-1/layer_02/_decoder_block | 182439 | 308.500 | 39.500 | 269.000 |
| token_0-1/layer_02/attention | 17487 | 178.500 | 39.500 | 139.000 |
| token_0-1/layer_02/gqa_attention | 171908 | 148.000 | 64.000 | 84.000 |
| token_0-1/layer_02/model_list.2.attention.o_weight.weight | 17577 | 178.500 | 39.500 | 139.000 |
| token_0-1/layer_02/rope | 168800 | 112.000 | 45.000 | 67.000 |
| token_0-1/layer_03/_decoder_block | 210454 | 311.500 | 39.500 | 272.000 |
| token_0-1/layer_03/attention | 182364 | 302.500 | 39.500 | 263.000 |
| token_0-1/layer_03/gqa_attention | 191995 | 149.500 | 63.500 | 86.000 |
| token_0-1/layer_03/model_list.3.attention.o_weight.weight | 22071 | 178.500 | 39.500 | 139.000 |
| token_0-1/layer_03/rope | 192176 | 149.000 | 63.000 | 86.000 |
| token_0-1/layer_04/_decoder_block | 238464 | 306.500 | 39.500 | 267.000 |
| token_0-1/layer_04/attention | 210099 | 305.500 | 39.500 | 266.000 |
| token_0-1/layer_04/gqa_attention | 210179 | 305.500 | 39.500 | 266.000 |
| token_0-1/layer_04/model_list.4.attention.o_weight.weight | 210195 | 305.500 | 39.500 | 266.000 |
| token_0-1/layer_04/rope | 208642 | 287.500 | 39.500 | 248.000 |
| token_0-1/layer_05/_decoder_block | 266479 | 309.500 | 39.500 | 270.000 |
| token_0-1/layer_05/_decoder_block \| token_0-1/layer_23/swiglu | 639885 | 156.000 | 52.000 | 104.000 |
| token_0-1/layer_05/attention | 14290 | 284.500 | 39.500 | 245.000 |
| token_0-1/layer_05/gqa_attention | 238066 | 308.500 | 39.500 | 269.000 |
| token_0-1/layer_05/model_list.5.attention.o_weight.weight | 238469 | 314.500 | 39.500 | 275.000 |
| token_0-1/layer_05/rope | 247051 | 202.500 | 39.500 | 163.000 |
| token_0-1/layer_06/_decoder_block | 266484 | 317.500 | 39.500 | 278.000 |
| token_0-1/layer_06/attention | 269330 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_06/gqa_attention | 270638 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_06/model_list.6.attention.o_weight.weight | 270736 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_06/rope | 270566 | 205.500 | 39.500 | 166.000 |
| token_0-1/layer_07/_decoder_block | 294499 | 320.500 | 39.500 | 281.000 |
| token_0-1/layer_07/attention | 282206 | 153.500 | 63.500 | 90.000 |
| token_0-1/layer_07/gqa_attention | 295557 | 208.500 | 39.500 | 169.000 |
| token_0-1/layer_07/model_list.7.attention.o_weight.weight | 294494 | 312.500 | 39.500 | 273.000 |
| token_0-1/layer_07/rope | 284589 | 124.500 | 61.500 | 63.000 |
| token_0-1/layer_08/_decoder_block | 322514 | 323.500 | 39.500 | 284.000 |
| token_0-1/layer_08/attention | 44868 | 181.500 | 39.500 | 142.000 |
| token_0-1/layer_08/gqa_attention | 311283 | 158.000 | 63.000 | 95.000 |
| token_0-1/layer_08/model_list.8.attention.o_weight.weight | 44950 | 181.500 | 39.500 | 142.000 |
| token_0-1/layer_08/rope | 309214 | 146.000 | 45.000 | 101.000 |
| token_0-1/layer_09/_decoder_block | 350529 | 326.500 | 39.500 | 287.000 |
| token_0-1/layer_09/attention | 322479 | 317.500 | 39.500 | 278.000 |
| token_0-1/layer_09/gqa_attention | 333257 | 164.000 | 63.000 | 101.000 |
| token_0-1/layer_09/model_list.9.attention.o_weight.weight | 49452 | 181.500 | 39.500 | 142.000 |
| token_0-1/layer_09/rope | 333331 | 164.000 | 63.000 | 101.000 |
| token_0-1/layer_10/_decoder_block | 378539 | 321.500 | 39.500 | 282.000 |
| token_0-1/layer_10/attention | 350214 | 320.500 | 39.500 | 281.000 |
| token_0-1/layer_10/gqa_attention | 350294 | 320.500 | 39.500 | 281.000 |
| token_0-1/layer_10/model_list.10.attention.o_weight.weight | 350302 | 320.500 | 39.500 | 281.000 |
| token_0-1/layer_10/rope | 350286 | 320.500 | 39.500 | 281.000 |
| token_0-1/layer_11/_decoder_block | 406554 | 324.500 | 39.500 | 285.000 |
| token_0-1/layer_11/attention | 42265 | 287.500 | 39.500 | 248.000 |
| token_0-1/layer_11/gqa_attention | 378544 | 329.500 | 39.500 | 290.000 |
| token_0-1/layer_11/model_list.11.attention.o_weight.weight | 378565 | 327.500 | 39.500 | 288.000 |
| token_0-1/layer_11/rope | 377522 | 317.500 | 39.500 | 278.000 |
| token_0-1/layer_12/_decoder_block | 406580 | 330.500 | 39.500 | 291.000 |
| token_0-1/layer_12/attention | 410023 | 220.500 | 39.500 | 181.000 |
| token_0-1/layer_12/gqa_attention | 411331 | 220.500 | 39.500 | 181.000 |
| token_0-1/layer_12/model_list.12.attention.o_weight.weight | 406567 | 332.500 | 39.500 | 293.000 |
| token_0-1/layer_12/rope | 411257 | 220.500 | 39.500 | 181.000 |
| token_0-1/layer_13/_decoder_block | 434574 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_13/attention | 424660 | 139.500 | 61.500 | 78.000 |
| token_0-1/layer_13/gqa_attention | 436242 | 223.500 | 39.500 | 184.000 |
| token_0-1/layer_13/model_list.13.attention.o_weight.weight | 434569 | 327.500 | 39.500 | 288.000 |
| token_0-1/layer_13/rope | 425613 | 111.000 | 34.000 | 77.000 |
| token_0-1/layer_14/_decoder_block | 462589 | 338.500 | 39.500 | 299.000 |
| token_0-1/layer_14/attention | 72267 | 184.500 | 39.500 | 145.000 |
| token_0-1/layer_14/gqa_attention | 450327 | 180.000 | 64.000 | 116.000 |
| token_0-1/layer_14/model_list.14.attention.o_weight.weight | 72347 | 184.500 | 39.500 | 145.000 |
| token_0-1/layer_14/rope | 450048 | 183.000 | 64.000 | 119.000 |
| token_0-1/layer_15/_decoder_block | 490604 | 341.500 | 39.500 | 302.000 |
| token_0-1/layer_15/attention | 463416 | 211.500 | 39.500 | 172.000 |
| token_0-1/layer_15/gqa_attention | 472081 | 179.000 | 64.000 | 115.000 |
| token_0-1/layer_15/model_list.15.attention.o_weight.weight | 478063 | 186.000 | 64.000 | 122.000 |
| token_0-1/layer_15/rope | 473256 | 178.000 | 63.000 | 115.000 |
| token_0-1/layer_16/_decoder_block | 518614 | 336.500 | 39.500 | 297.000 |
| token_0-1/layer_16/attention | 490329 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_16/gqa_attention | 490489 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_16/model_list.16.attention.o_weight.weight | 490409 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_16/rope | 490401 | 335.500 | 39.500 | 296.000 |
| token_0-1/layer_17/_decoder_block | 546629 | 339.500 | 39.500 | 300.000 |
| token_0-1/layer_17/attention | 518619 | 344.500 | 39.500 | 305.000 |
| token_0-1/layer_17/gqa_attention | 518144 | 338.500 | 39.500 | 299.000 |
| token_0-1/layer_17/model_list.17.attention.o_weight.weight | 518160 | 338.500 | 39.500 | 299.000 |
| token_0-1/layer_17/rope | 518664 | 338.000 | 41.000 | 297.000 |
| token_0-1/layer_18/_decoder_block | 546167 | 341.500 | 39.500 | 302.000 |
| token_0-1/layer_18/attention | 550716 | 235.500 | 39.500 | 196.000 |
| token_0-1/layer_18/gqa_attention | 551926 | 235.500 | 39.500 | 196.000 |
| token_0-1/layer_18/model_list.18.attention.o_weight.weight | 546634 | 347.500 | 39.500 | 308.000 |
| token_0-1/layer_18/rope | 551950 | 235.500 | 39.500 | 196.000 |
| token_0-1/layer_19/_decoder_block | 574649 | 350.500 | 39.500 | 311.000 |
| token_0-1/layer_19/attention | 570524 | 294.500 | 39.500 | 255.000 |
| token_0-1/layer_19/gqa_attention | 568979 | 276.500 | 39.500 | 237.000 |
| token_0-1/layer_19/model_list.19.attention.o_weight.weight | 575635 | 238.500 | 39.500 | 199.000 |
| token_0-1/layer_19/rope | 574644 | 342.500 | 39.500 | 303.000 |
| token_0-1/layer_20/_decoder_block | 602664 | 353.500 | 39.500 | 314.000 |
| token_0-1/layer_20/attention | 590123 | 198.000 | 64.000 | 134.000 |
| token_0-1/layer_20/gqa_attention | 591270 | 192.000 | 63.000 | 129.000 |
| token_0-1/layer_20/model_list.20.attention.o_weight.weight | 602659 | 345.500 | 39.500 | 306.000 |
| token_0-1/layer_20/rope | 591013 | 192.500 | 63.500 | 129.000 |
| token_0-1/layer_21/_decoder_block | 630679 | 356.500 | 39.500 | 317.000 |
| token_0-1/layer_21/attention | 612156 | 194.000 | 64.000 | 130.000 |
| token_0-1/layer_21/gqa_attention | 612190 | 194.000 | 64.000 | 130.000 |
| token_0-1/layer_21/model_list.21.attention.o_weight.weight | 618138 | 201.000 | 64.000 | 137.000 |
| token_0-1/layer_21/rope | 612089 | 193.500 | 63.500 | 130.000 |
| token_0-1/layer_22/_decoder_block | 630420 | 350.500 | 39.500 | 311.000 |
| token_0-1/layer_22/_decoder_block \| token_shared/layer_shared/interface | 622992 | 139.000 | 30.000 | 109.000 |
| token_0-1/layer_22/attention | 630436 | 350.500 | 39.500 | 311.000 |
| token_0-1/layer_22/gqa_attention | 630516 | 350.500 | 39.500 | 311.000 |
| token_0-1/layer_22/model_list.22.attention.o_weight.weight | 630532 | 350.500 | 39.500 | 311.000 |
| token_0-1/layer_22/rope | 630524 | 350.500 | 39.500 | 311.000 |
| token_0-1/layer_23/_decoder_block | 688860 | 191.000 | 0.000 | 191.000 |
| token_0-1/layer_23/attention | 96676 | 275.000 | 41.000 | 234.000 |
| token_0-1/layer_23/gqa_attention | 660800 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_23/model_list.23.attention.o_weight.weight | 98223 | 293.500 | 39.500 | 254.000 |
| token_0-1/layer_23/model_list.23.post_Normal.weight | 92563 | 224.000 | 41.000 | 183.000 |
| token_0-1/layer_23/model_list.23.post_Normal.weight \| token_shared/layer_shared/final_normal.weight | 688866 | 190.500 | 1.500 | 189.000 |
| token_0-1/layer_23/rope | 660734 | 171.000 | 26.000 | 145.000 |
| token_0-1/layer_23/swiglu | 98231 | 293.500 | 39.500 | 254.000 |
| token_shared/layer_shared/final_normal.weight | 688868 | 191.000 | 0.000 | 191.000 |
| token_shared/layer_shared/interface | 518286 | 338.000 | 41.000 | 297.000 |
| token_shared/layer_shared/lm_head.weight | 126404 | 294.500 | 39.500 | 255.000 |
| token_shared/layer_shared/pack_kv | 518269 | 335.000 | 41.000 | 294.000 |

## 10. 全图峰值时的level分布

| 类型/level/分量 | 数量 | 字节 | MiB |
| --- | --- | --- | --- |
| ciphertext:L1:c2 | 67 | 140509184 | 134.000 |
| ciphertext:L2:c2 | 25 | 78643200 | 75.000 |
| ciphertext:L5:c2 | 18 | 113246208 | 108.000 |
| plaintext:L1:c1 | 2 | 2097152 | 2.000 |
| plaintext:L2:c1 | 10 | 15728640 | 15.000 |
| plaintext:L3:c1 | 7 | 14680064 | 14.000 |
| plaintext:L4:c1 | 1 | 2621440 | 2.500 |
| plaintext:L5:c1 | 2 | 6291456 | 6.000 |


完整机器可读数据：[compiled-analysis.json](evidence/compiled-analysis.json)。主报告：[MODEL_DATA_ANALYSIS.md](MODEL_DATA_ANALYSIS.md)。
