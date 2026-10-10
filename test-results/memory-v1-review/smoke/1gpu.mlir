module {
  func.func @memory_smoke(%arg0: tensor<1x!ckks.poly<2 * 40 * 6>> {dist.rank = 0 : i64, dist.device = -1 : i64}) -> tensor<1x!ckks.poly<2 * 40 * 6>> attributes {dist.device_counts = array<i64: 1>} {
    %w = "ckks.encode"() <{payload = dense<1.0> : tensor<1xf64>}> {dist.logical_id = 1 : i64, dist.rank = 0 : i64, dist.device = -1 : i64} : () -> tensor<1x!ckks.poly<1 * 40 * 6>>
    %t0 = "dist.transfer"(%arg0) <{transfer_id = 0 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}> {dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    %t1 = "dist.transfer"(%w) <{transfer_id = 1 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}> {dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<1 * 40 * 6>>
    %v2 = "ckks.addcp"(%t0, %t1) {dist.logical_id = 2 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    %v3 = "ckks.addcp"(%v2, %t1) {dist.logical_id = 3 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    %v4 = "ckks.addcp"(%v3, %t1) {dist.logical_id = 4 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    %v5 = "ckks.addcp"(%v4, %t1) {dist.logical_id = 5 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>, tensor<1x!ckks.poly<1 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    %v6 = "ckks.rotatec"(%v5) <{offset = array<i64: 1>}> {dist.logical_id = 6 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    %v7 = "ckks.rotatec"(%v6) <{offset = array<i64: -1>}> {dist.logical_id = 7 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 6>>
    return %v7 : tensor<1x!ckks.poly<2 * 40 * 6>>
  }
}
