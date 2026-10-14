module {
  func.func @memory_smoke(%arg0: tensor<1x!ckks.poly<2 * 40 * 6>> {dist.device = -1 : i64, dist.rank = 0 : i64}) -> tensor<1x!ckks.poly<2 * 40 * 6>> attributes {dist.device_counts = array<i64: 1>, runtime.memory_planned} {
    %0 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<1xf64>}> {dist.device = -1 : i64, dist.logical_id = 1 : i64, dist.rank = 0 : i64, runtime.value_id = 1 : i64} : () -> tensor<1x!ckks.poly<1 * 40 * 6>>
    %1 = "dist.transfer"(%arg0) <{destination_device = 0 : i64, destination_rank = 0 : i64, initialization = true, source_device = -1 : i64, source_rank = 0 : i64, transfer_id = 0 : i64}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 8 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    "dist.release"(%arg0) <{initialization = true}> : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> ()
    %2 = "dist.transfer"(%0) <{destination_device = 0 : i64, destination_rank = 0 : i64, initialization = true, source_device = -1 : i64, source_rank = 0 : i64, transfer_id = 1 : i64}> {dist.device = 0 : i64, dist.rank = 0 : i64, runtime.value_id = 9 : i64} : (tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<1 * 40 * 6>>
    "dist.release"(%0) <{initialization = true}> : (tensor<1x!ckks.poly<1 * 40 * 6>>) -> ()
    %3 = "ckks.addcp"(%1, %2) {dist.device = 0 : i64, dist.logical_id = 2 : i64, dist.rank = 0 : i64, runtime.value_id = 2 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    "dist.release"(%1) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> ()
    %4 = "ckks.addcp"(%3, %2) {dist.device = 0 : i64, dist.logical_id = 3 : i64, dist.rank = 0 : i64, runtime.value_id = 3 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    "dist.release"(%3) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> ()
    %5 = "ckks.addcp"(%4, %2) {dist.device = 0 : i64, dist.logical_id = 4 : i64, dist.rank = 0 : i64, runtime.value_id = 4 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    "dist.release"(%4) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> ()
    %6 = "ckks.addcp"(%5, %2) {dist.device = 0 : i64, dist.logical_id = 5 : i64, dist.rank = 0 : i64, runtime.value_id = 5 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    "dist.release"(%5) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> ()
    "dist.release"(%2) <{initialization = false}> : (tensor<1x!ckks.poly<1 * 40 * 6>>) -> ()
    %7 = "ckks.rotatec"(%6) <{offset = array<i64: 1>}> {dist.device = 0 : i64, dist.logical_id = 6 : i64, dist.rank = 0 : i64, runtime.value_id = 6 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    "dist.release"(%6) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> ()
    %8 = "ckks.rotatec"(%7) <{offset = array<i64: -1>}> {dist.device = 0 : i64, dist.logical_id = 7 : i64, dist.rank = 0 : i64, runtime.value_id = 7 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    "dist.release"(%7) <{initialization = false}> : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> ()
    return %8 : tensor<1x!ckks.poly<2 * 40 * 6>>
  }
}

