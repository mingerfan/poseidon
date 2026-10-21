module {
  func.func @chains(%arg0: tensor<1x!ckks.poly<2 * 40 * 27>> {dist.device = -1 : i64, dist.rank = 0 : i64}, %arg1: tensor<1x!ckks.poly<2 * 40 * 27>> {dist.device = -1 : i64, dist.rank = 0 : i64}) -> (tensor<1x!ckks.poly<2 * 40 * 7>>, tensor<1x!ckks.poly<2 * 40 * 7>>) attributes {arg_scale = array<i64: 40, 40>, ckks.levels_per_logical_level = 5 : i64, ckks.logical_init_level = 5 : i64, dist.device_counts = array<i64: 2>, dist.inter_rank_communication_cost = 10000 : i64, dist.intra_rank_communication_cost = 1000 : i64, init_level = 27 : i64, res_scale = array<i64: 40, 40>, runtime.memory_planned, runtime.waterline = "{\22backend\22:\22auto\22,\22device_capacity_bytes\22:[33285996544,33285996544],\22device_fractions\22:[0.5,0.5],\22device_reserve_bytes\22:[6442450944,6442450944],\22host_budget_bytes\22:8589934592,\22lookahead_groups\22:128,\22pinned_budget_bytes\22:1073741824,\22raw_budget_bytes\22:67108864}"} {
    %0 = "dist.transfer"(%arg0) <{destination_device = 0 : i64, destination_rank = 0 : i64, initialization = true, source_device = -1 : i64, source_rank = 0 : i64, transfer_id = 0 : i64}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 48 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%arg0) <{initialization = true}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %1 = "dist.transfer"(%arg1) <{destination_device = 1 : i64, destination_rank = 0 : i64, initialization = true, source_device = -1 : i64, source_rank = 0 : i64, transfer_id = 1 : i64}> {dist.device = 1 : i64, dist.rank = 0 : i64, runtime.value_id = 49 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%arg1) <{initialization = true}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %2 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 50 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 27>>
    %3 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 1 : i64, dist.rank = 0 : i64, runtime.value_id = 51 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 27>>
    %4 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 52 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 22>>
    %5 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 1 : i64, dist.rank = 0 : i64, runtime.value_id = 53 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 22>>
    %6 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 54 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 17>>
    %7 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 1 : i64, dist.rank = 0 : i64, runtime.value_id = 55 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 17>>
    %8 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 56 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 12>>
    %9 = "dist.prepare_plain"() <{payload = dense<1.010000e+00> : tensor<1xf64>}> {dist.device = 1 : i64, dist.rank = 0 : i64, runtime.value_id = 57 : i64} : () -> tensor<1x!ckks.poly<1 * 120 * 12>>
    %10 = "ckks.mulcp"(%0, %2) {dist.device = 0 : i64, dist.logical_id = 3 : i64, dist.rank = 0 : i64, dist.schedule_finish = 410 : i64, dist.schedule_start = 402 : i64, runtime.value_id = 3 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%0) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %11 = "ckks.mulcp"(%1, %3) {dist.device = 1 : i64, dist.logical_id = 5 : i64, dist.rank = 0 : i64, dist.schedule_finish = 410 : i64, dist.schedule_start = 402 : i64, runtime.value_id = 5 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%1) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %12 = "ckks.rescalec"(%10) {dist.device = 0 : i64, dist.logical_id = 4 : i64, dist.rank = 0 : i64, dist.schedule_finish = 515 : i64, dist.schedule_start = 410 : i64, runtime.value_id = 4 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%10) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %13 = "ckks.rescalec"(%11) {dist.device = 1 : i64, dist.logical_id = 6 : i64, dist.rank = 0 : i64, dist.schedule_finish = 515 : i64, dist.schedule_start = 410 : i64, runtime.value_id = 6 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%11) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %14 = "ckks.mulcp"(%12, %4) {dist.device = 0 : i64, dist.logical_id = 8 : i64, dist.rank = 0 : i64, dist.schedule_finish = 522 : i64, dist.schedule_start = 515 : i64, runtime.value_id = 8 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%12) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    %15 = "ckks.mulcp"(%13, %5) {dist.device = 1 : i64, dist.logical_id = 10 : i64, dist.rank = 0 : i64, dist.schedule_finish = 522 : i64, dist.schedule_start = 515 : i64, runtime.value_id = 10 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%13) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    %16 = "ckks.rescalec"(%14) {dist.device = 0 : i64, dist.logical_id = 9 : i64, dist.rank = 0 : i64, dist.schedule_finish = 625 : i64, dist.schedule_start = 522 : i64, runtime.value_id = 9 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%14) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %17 = "ckks.rescalec"(%15) {dist.device = 1 : i64, dist.logical_id = 11 : i64, dist.rank = 0 : i64, dist.schedule_finish = 625 : i64, dist.schedule_start = 522 : i64, runtime.value_id = 11 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%15) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %18 = "ckks.mulcp"(%16, %6) {dist.device = 0 : i64, dist.logical_id = 13 : i64, dist.rank = 0 : i64, dist.schedule_finish = 632 : i64, dist.schedule_start = 625 : i64, runtime.value_id = 13 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%16) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    %19 = "ckks.mulcp"(%17, %7) {dist.device = 1 : i64, dist.logical_id = 15 : i64, dist.rank = 0 : i64, dist.schedule_finish = 632 : i64, dist.schedule_start = 625 : i64, runtime.value_id = 15 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%17) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    %20 = "ckks.rescalec"(%18) {dist.device = 0 : i64, dist.logical_id = 14 : i64, dist.rank = 0 : i64, dist.schedule_finish = 732 : i64, dist.schedule_start = 632 : i64, runtime.value_id = 14 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%18) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %21 = "ckks.rescalec"(%19) {dist.device = 1 : i64, dist.logical_id = 16 : i64, dist.rank = 0 : i64, dist.schedule_finish = 732 : i64, dist.schedule_start = 632 : i64, runtime.value_id = 16 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%19) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %22 = "ckks.mulcp"(%20, %8) {dist.device = 0 : i64, dist.logical_id = 18 : i64, dist.rank = 0 : i64, dist.schedule_finish = 739 : i64, dist.schedule_start = 732 : i64, runtime.value_id = 18 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%20) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %23 = "ckks.mulcp"(%21, %9) {dist.device = 1 : i64, dist.logical_id = 20 : i64, dist.rank = 0 : i64, dist.schedule_finish = 739 : i64, dist.schedule_start = 732 : i64, runtime.value_id = 20 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%21) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %24 = "ckks.rescalec"(%22) {dist.device = 0 : i64, dist.logical_id = 19 : i64, dist.rank = 0 : i64, dist.schedule_finish = 837 : i64, dist.schedule_start = 739 : i64, runtime.value_id = 19 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%22) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %25 = "ckks.rescalec"(%23) {dist.device = 1 : i64, dist.logical_id = 21 : i64, dist.rank = 0 : i64, dist.schedule_finish = 837 : i64, dist.schedule_start = 739 : i64, runtime.value_id = 21 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%23) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %26 = "ckks.bootstrapc"(%24) {dist.device = 0 : i64, dist.logical_id = 22 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85837 : i64, dist.schedule_start = 837 : i64, runtime.value_id = 22 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%24) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %27 = "ckks.bootstrapc"(%25) {dist.device = 1 : i64, dist.logical_id = 23 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85837 : i64, dist.schedule_start = 837 : i64, runtime.value_id = 23 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%25) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %28 = "ckks.mulcp"(%26, %2) {dist.device = 0 : i64, dist.logical_id = 24 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85845 : i64, dist.schedule_start = 85837 : i64, runtime.value_id = 24 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%26) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    "dist.release"(%2) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 27>>) -> ()
    %29 = "ckks.mulcp"(%27, %3) {dist.device = 1 : i64, dist.logical_id = 26 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85845 : i64, dist.schedule_start = 85837 : i64, runtime.value_id = 26 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>, tensor<1x!ckks.poly<1 * 120 * 27>>) -> tensor<1x!ckks.poly<2 * 160 * 27>>
    "dist.release"(%27) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    "dist.release"(%3) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 27>>) -> ()
    %30 = "ckks.rescalec"(%28) {dist.device = 0 : i64, dist.logical_id = 25 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85950 : i64, dist.schedule_start = 85845 : i64, runtime.value_id = 25 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%28) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %31 = "ckks.rescalec"(%29) {dist.device = 1 : i64, dist.logical_id = 27 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85950 : i64, dist.schedule_start = 85845 : i64, runtime.value_id = 27 : i64} : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 22>>
    "dist.release"(%29) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 27>>) -> ()
    %32 = "ckks.mulcp"(%30, %4) {dist.device = 0 : i64, dist.logical_id = 28 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85957 : i64, dist.schedule_start = 85950 : i64, runtime.value_id = 28 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%30) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    "dist.release"(%4) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 22>>) -> ()
    %33 = "ckks.mulcp"(%31, %5) {dist.device = 1 : i64, dist.logical_id = 30 : i64, dist.rank = 0 : i64, dist.schedule_finish = 85957 : i64, dist.schedule_start = 85950 : i64, runtime.value_id = 30 : i64} : (tensor<1x!ckks.poly<2 * 40 * 22>>, tensor<1x!ckks.poly<1 * 120 * 22>>) -> tensor<1x!ckks.poly<2 * 160 * 22>>
    "dist.release"(%31) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 22>>) -> ()
    "dist.release"(%5) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 22>>) -> ()
    %34 = "ckks.rescalec"(%32) {dist.device = 0 : i64, dist.logical_id = 29 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86060 : i64, dist.schedule_start = 85957 : i64, runtime.value_id = 29 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%32) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %35 = "ckks.rescalec"(%33) {dist.device = 1 : i64, dist.logical_id = 31 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86060 : i64, dist.schedule_start = 85957 : i64, runtime.value_id = 31 : i64} : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> tensor<1x!ckks.poly<2 * 40 * 17>>
    "dist.release"(%33) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 22>>) -> ()
    %36 = "ckks.mulcp"(%34, %6) {dist.device = 0 : i64, dist.logical_id = 32 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86067 : i64, dist.schedule_start = 86060 : i64, runtime.value_id = 32 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%34) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    "dist.release"(%6) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 17>>) -> ()
    %37 = "ckks.mulcp"(%35, %7) {dist.device = 1 : i64, dist.logical_id = 34 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86067 : i64, dist.schedule_start = 86060 : i64, runtime.value_id = 34 : i64} : (tensor<1x!ckks.poly<2 * 40 * 17>>, tensor<1x!ckks.poly<1 * 120 * 17>>) -> tensor<1x!ckks.poly<2 * 160 * 17>>
    "dist.release"(%35) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 17>>) -> ()
    "dist.release"(%7) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 17>>) -> ()
    %38 = "ckks.rescalec"(%36) {dist.device = 0 : i64, dist.logical_id = 33 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86167 : i64, dist.schedule_start = 86067 : i64, runtime.value_id = 33 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%36) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %39 = "ckks.rescalec"(%37) {dist.device = 1 : i64, dist.logical_id = 35 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86167 : i64, dist.schedule_start = 86067 : i64, runtime.value_id = 35 : i64} : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%37) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 17>>) -> ()
    %40 = "ckks.mulcp"(%38, %8) {dist.device = 0 : i64, dist.logical_id = 36 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86174 : i64, dist.schedule_start = 86167 : i64, runtime.value_id = 36 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%38) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %41 = "ckks.mulcp"(%39, %9) {dist.device = 1 : i64, dist.logical_id = 38 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86174 : i64, dist.schedule_start = 86167 : i64, runtime.value_id = 38 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%39) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    %42 = "ckks.rescalec"(%40) {dist.device = 0 : i64, dist.logical_id = 37 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86272 : i64, dist.schedule_start = 86174 : i64, runtime.value_id = 37 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%40) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %43 = "ckks.rescalec"(%41) {dist.device = 1 : i64, dist.logical_id = 39 : i64, dist.rank = 0 : i64, dist.schedule_finish = 86272 : i64, dist.schedule_start = 86174 : i64, runtime.value_id = 39 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%41) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %44 = "ckks.bootstrapc"(%42) {dist.device = 0 : i64, dist.logical_id = 40 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171272 : i64, dist.schedule_start = 86272 : i64, runtime.value_id = 40 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%42) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %45 = "ckks.bootstrapc"(%43) {dist.device = 1 : i64, dist.logical_id = 41 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171272 : i64, dist.schedule_start = 86272 : i64, runtime.value_id = 41 : i64} : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 40 * 27>>
    "dist.release"(%43) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 7>>) -> ()
    %46 = "ckks.modswitchc"(%44) <{downFactor = 15 : i64}> {dist.device = 0 : i64, dist.logical_id = 42 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171310 : i64, dist.schedule_start = 171272 : i64, runtime.value_id = 42 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%44) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %47 = "ckks.modswitchc"(%45) <{downFactor = 15 : i64}> {dist.device = 1 : i64, dist.logical_id = 45 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171310 : i64, dist.schedule_start = 171272 : i64, runtime.value_id = 45 : i64} : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> tensor<1x!ckks.poly<2 * 40 * 12>>
    "dist.release"(%45) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 27>>) -> ()
    %48 = "ckks.mulcp"(%46, %8) {dist.device = 0 : i64, dist.logical_id = 43 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171317 : i64, dist.schedule_start = 171310 : i64, runtime.value_id = 43 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%46) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    "dist.release"(%8) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 12>>) -> ()
    %49 = "ckks.mulcp"(%47, %9) {dist.device = 1 : i64, dist.logical_id = 46 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171317 : i64, dist.schedule_start = 171310 : i64, runtime.value_id = 46 : i64} : (tensor<1x!ckks.poly<2 * 40 * 12>>, tensor<1x!ckks.poly<1 * 120 * 12>>) -> tensor<1x!ckks.poly<2 * 160 * 12>>
    "dist.release"(%47) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 12>>) -> ()
    "dist.release"(%9) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 120 * 12>>) -> ()
    %50 = "ckks.rescalec"(%48) {dist.device = 0 : i64, dist.logical_id = 44 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171415 : i64, dist.schedule_start = 171317 : i64, runtime.value_id = 44 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%48) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    %51 = "ckks.rescalec"(%49) {dist.device = 1 : i64, dist.logical_id = 47 : i64, dist.rank = 0 : i64, dist.schedule_finish = 171415 : i64, dist.schedule_start = 171317 : i64, runtime.value_id = 47 : i64} : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> tensor<1x!ckks.poly<2 * 40 * 7>>
    "dist.release"(%49) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 160 * 12>>) -> ()
    return %50, %51 : tensor<1x!ckks.poly<2 * 40 * 7>>, tensor<1x!ckks.poly<2 * 40 * 7>>
  }
}

