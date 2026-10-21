module {
  func.func @chains(%arg0: tensor<1x!ckks.poly<2 * 40 * 27>> {dist.device = -1 : i64, dist.rank = 0 : i64}, %arg1: tensor<1x!ckks.poly<2 * 40 * 27>> {dist.device = -1 : i64, dist.rank = 0 : i64}) -> (tensor<1x!ckks.poly<2 * 40 * 7>>, tensor<1x!ckks.poly<2 * 40 * 7>>) attributes {arg_scale = array<i64: 40, 40>, ckks.levels_per_logical_level = 5 : i64, ckks.logical_init_level = 5 : i64, dist.device_counts = array<i64: 1>, dist.inter_rank_communication_cost = 10000 : i64, dist.intra_rank_communication_cost = 1000 : i64, init_level = 27 : i64, res_scale = array<i64: 40, 40>, runtime.memory_planned, runtime.waterline = "{\22backend\22:\22auto\22,\22device_capacity_bytes\22:[8053063680],\22device_fractions\22:[0.5],\22device_reserve_bytes\22:[6442450944],\22host_budget_bytes\22:8589934592,\22lookahead_groups\22:128,\22pinned_budget_bytes\22:1073741824,\22raw_budget_bytes\22:67108864}"} {
    %0 = "dist.transfer"(%arg0) <{destination_device = 0 : i64, destination_rank = 0 : i64, initialization = true, source_device = -1 : i64, source_rank = 0 : i64, transfer_id = 0 : i64}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 48 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%arg0) <{initialization = true}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %1 = "dist.transfer"(%arg1) <{destination_device = 0 : i64, destination_rank = 0 : i64, initialization = true, source_device = -1 : i64, source_rank = 0 : i64, transfer_id = 1 : i64}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 49 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%arg1) <{initialization = true}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %2 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 50 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 27>>
    %3 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 51 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 22>>
    %4 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 52 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 17>>
    %5 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 53 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 12>>
    %6 = "ckks.mulcp"(%0, %2) {dist.device = 0 : i64, dist.logical_id = 3 : i64, dist.rank = 0 : i64, dist.schedule_finish = 410 : i64, dist.schedule_start = 402 : i64, runtime.value_id = 3 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%0) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %7 = "ckks.mulcp"(%1, %2) {dist.device = 0 : i64, dist.logical_id = 5 : i64, dist.rank = 0 : i64, dist.schedule_finish = 418 : i64, dist.schedule_start = 410 : i64, runtime.value_id = 5 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%1) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %8 = "ckks.rescalec"(%6) {dist.device = 0 : i64, dist.logical_id = 4 : i64, dist.rank = 0 : i64, dist.schedule_finish = 523 : i64, dist.schedule_start = 418 : i64, runtime.value_id = 4 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%6) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %9 = "ckks.rescalec"(%7) {dist.device = 0 : i64, dist.logical_id = 6 : i64, dist.rank = 0 : i64, dist.schedule_finish = 628 : i64, dist.schedule_start = 523 : i64, runtime.value_id = 6 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%7) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %10 = "ckks.mulcp"(%8, %3) {dist.device = 0 : i64, dist.logical_id = 8 : i64, dist.rank = 0 : i64, dist.schedule_finish = 635 : i64, dist.schedule_start = 628 : i64, runtime.value_id = 8 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%8) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    %11 = "ckks.mulcp"(%9, %3) {dist.device = 0 : i64, dist.logical_id = 10 : i64, dist.rank = 0 : i64, dist.schedule_finish = 642 : i64, dist.schedule_start = 635 : i64, runtime.value_id = 10 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%9) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    %12 = "ckks.rescalec"(%10) {dist.device = 0 : i64, dist.logical_id = 9 : i64, dist.rank = 0 : i64, dist.schedule_finish = 745 : i64, dist.schedule_start = 642 : i64, runtime.value_id = 9 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%10) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %13 = "ckks.rescalec"(%11) {dist.device = 0 : i64, dist.logical_id = 11 : i64, dist.rank = 0 : i64, dist.schedule_finish = 848 : i64, dist.schedule_start = 745 : i64, runtime.value_id = 11 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%11) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %14 = "ckks.mulcp"(%12, %4) {dist.device = 0 : i64, dist.logical_id = 13 : i64, dist.rank = 0 : i64, dist.schedule_finish = 855 : i64, dist.schedule_start = 848 : i64, runtime.value_id = 13 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%12) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    %15 = "ckks.mulcp"(%13, %4) {dist.device = 0 : i64, dist.logical_id = 15 : i64, dist.rank = 0 : i64, dist.schedule_finish = 862 : i64, dist.schedule_start = 855 : i64, runtime.value_id = 15 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%13) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    %16 = "ckks.rescalec"(%14) {dist.device = 0 : i64, dist.logical_id = 14 : i64, dist.rank = 0 : i64, dist.schedule_finish = 962 : i64, dist.schedule_start = 862 : i64, runtime.value_id = 14 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%14) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %17 = "ckks.rescalec"(%15) {dist.device = 0 : i64, dist.logical_id = 16 : i64, dist.rank = 0 : i64, dist.schedule_finish = 1062 : i64, dist.schedule_start = 962 : i64, runtime.value_id = 16 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%15) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %18 = "ckks.mulcp"(%16, %5) {dist.device = 0 : i64, dist.logical_id = 18 : i64, dist.rank = 0 : i64, dist.schedule_finish = 1069 : i64, dist.schedule_start = 1062 : i64, runtime.value_id = 18 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%16) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %19 = "ckks.mulcp"(%17, %5) {dist.device = 0 : i64, dist.logical_id = 20 : i64, dist.rank = 0 : i64, dist.schedule_finish = 1076 : i64, dist.schedule_start = 1069 : i64, runtime.value_id = 20 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%17) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %20 = "ckks.rescalec"(%18) {dist.device = 0 : i64, dist.logical_id = 19 : i64, dist.rank = 0 : i64, dist.schedule_finish = 1174 : i64, dist.schedule_start = 1076 : i64, runtime.value_id = 19 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%18) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %21 = "ckks.rescalec"(%19) {dist.device = 0 : i64, dist.logical_id = 21 : i64, dist.rank = 0 : i64, dist.schedule_finish = 1272 : i64, dist.schedule_start = 1174 : i64, runtime.value_id = 21 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%19) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %22 = "ckks.bootstrapc"(%20) {dist.device = 0 : i64, dist.logical_id = 22 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86272 : i64, dist.schedule_start = 1272 : i64, runtime.value_id = 22 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%20) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %23 = "ckks.bootstrapc"(%21) {dist.device = 0 : i64, dist.logical_id = 23 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171272 : i64, dist.schedule_start = 86272 : i64, runtime.value_id = 23 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%21) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %24 = "ckks.mulcp"(%22, %2) {dist.device = 0 : i64, dist.logical_id = 24 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171280 : i64, dist.schedule_start = 171272 : i64, runtime.value_id = 24 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%22) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %25 = "ckks.mulcp"(%23, %2) {dist.device = 0 : i64, dist.logical_id = 26 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171288 : i64, dist.schedule_start = 171280 : i64, runtime.value_id = 26 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%23) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    "dist.release"(%2) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 27>>) -> ()
    %26 = "ckks.rescalec"(%24) {dist.device = 0 : i64, dist.logical_id = 25 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171393 : i64, dist.schedule_start = 171288 : i64, runtime.value_id = 25 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%24) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %27 = "ckks.rescalec"(%25) {dist.device = 0 : i64, dist.logical_id = 27 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171498 : i64, dist.schedule_start = 171393 : i64, runtime.value_id = 27 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%25) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %28 = "ckks.mulcp"(%26, %3) {dist.device = 0 : i64, dist.logical_id = 28 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171505 : i64, dist.schedule_start = 171498 : i64, runtime.value_id = 28 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%26) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    %29 = "ckks.mulcp"(%27, %3) {dist.device = 0 : i64, dist.logical_id = 30 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171512 : i64, dist.schedule_start = 171505 : i64, runtime.value_id = 30 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%27) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    "dist.release"(%3) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 22>>) -> ()
    %30 = "ckks.rescalec"(%28) {dist.device = 0 : i64, dist.logical_id = 29 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171615 : i64, dist.schedule_start = 171512 : i64, runtime.value_id = 29 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%28) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %31 = "ckks.rescalec"(%29) {dist.device = 0 : i64, dist.logical_id = 31 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171718 : i64, dist.schedule_start = 171615 : i64, runtime.value_id = 31 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%29) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %32 = "ckks.mulcp"(%30, %4) {dist.device = 0 : i64, dist.logical_id = 32 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171725 : i64, dist.schedule_start = 171718 : i64, runtime.value_id = 32 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%30) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    %33 = "ckks.mulcp"(%31, %4) {dist.device = 0 : i64, dist.logical_id = 34 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171732 : i64, dist.schedule_start = 171725 : i64, runtime.value_id = 34 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%31) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    "dist.release"(%4) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 17>>) -> ()
    %34 = "ckks.rescalec"(%32) {dist.device = 0 : i64, dist.logical_id = 33 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171832 : i64, dist.schedule_start = 171732 : i64, runtime.value_id = 33 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%32) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %35 = "ckks.rescalec"(%33) {dist.device = 0 : i64, dist.logical_id = 35 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171932 : i64, dist.schedule_start = 171832 : i64, runtime.value_id = 35 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%33) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %36 = "ckks.mulcp"(%34, %5) {dist.device = 0 : i64, dist.logical_id = 36 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171939 : i64, dist.schedule_start = 171932 : i64, runtime.value_id = 36 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%34) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %37 = "ckks.mulcp"(%35, %5) {dist.device = 0 : i64, dist.logical_id = 38 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171946 : i64, dist.schedule_start = 171939 : i64, runtime.value_id = 38 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%35) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %38 = "ckks.rescalec"(%36) {dist.device = 0 : i64, dist.logical_id = 37 : i64, dist.rank = 0 : i64, dist.schedule_finish = 172044 : i64, dist.schedule_start = 171946 : i64, runtime.value_id = 37 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%36) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %39 = "ckks.rescalec"(%37) {dist.device = 0 : i64, dist.logical_id = 39 : i64, dist.rank = 0 : i64, dist.schedule_finish = 172142 : i64, dist.schedule_start = 172044 : i64, runtime.value_id = 39 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%37) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %40 = "ckks.bootstrapc"(%38) {dist.device = 0 : i64, dist.logical_id = 40 : i64, dist.rank = 0 : i64, dist.schedule_finish = 257142 : i64, dist.schedule_start = 172142 : i64, runtime.value_id = 40 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%38) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %41 = "ckks.bootstrapc"(%39) {dist.device = 0 : i64, dist.logical_id = 41 : i64, dist.rank = 0 : i64, dist.schedule_finish = 342142 : i64, dist.schedule_start = 257142 : i64, runtime.value_id = 41 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%39) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %42 = "ckks.modswitchc"(%40) <{downFactor = 15 : i64}> {dist.device = 0 : i64, dist.logical_id = 42 : i64, dist.rank = 0 : i64, dist.schedule_finish = 342180 : i64, dist.schedule_start = 342142 : i64, runtime.value_id = 42 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%40) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %43 = "ckks.modswitchc"(%41) <{downFactor = 15 : i64}> {dist.device = 0 : i64, dist.logical_id = 45 : i64, dist.rank = 0 : i64, dist.schedule_finish = 342218 : i64, dist.schedule_start = 342180 : i64, runtime.value_id = 45 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%41) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %44 = "ckks.mulcp"(%42, %5) {dist.device = 0 : i64, dist.logical_id = 43 : i64, dist.rank = 0 : i64, dist.schedule_finish = 342225 : i64, dist.schedule_start = 342218 : i64, runtime.value_id = 43 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%42) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %45 = "ckks.mulcp"(%43, %5) {dist.device = 0 : i64, dist.logical_id = 46 : i64, dist.rank = 0 : i64, dist.schedule_finish = 342232 : i64, dist.schedule_start = 342225 : i64, runtime.value_id = 46 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%43) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    "dist.release"(%5) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 12>>) -> ()
    %46 = "ckks.rescalec"(%44) {dist.device = 0 : i64, dist.logical_id = 44 : i64, dist.rank = 0 : i64, dist.schedule_finish = 342330 : i64, dist.schedule_start = 342232 : i64, runtime.value_id = 44 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%44) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %47 = "ckks.rescalec"(%45) {dist.device = 0 : i64, dist.logical_id = 47 : i64, dist.rank = 0 : i64, dist.schedule_finish = 342428 : i64, dist.schedule_start = 342330 : i64, runtime.value_id = 47 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%45) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    return %46, %47 : tensor<1x!ckks.poly<2 * 40 * 7>>, tensor<1x!ckks.poly<2 * 40 * 7>>
  }
}

