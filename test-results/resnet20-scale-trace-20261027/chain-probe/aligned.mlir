module {
  func.func @align(%arg0: tensor<1x!ckks.poly<2 * 76 * 8>>) -> tensor<1x!ckks.poly<2 * 40 * 24>> attributes {ckks.levels_per_logical_level = 1 : i64, ckks.logical_init_level = 24 : i64, ckks.rns_moduli = array<i64: 4255252481, 4255645697, 4258922497, 4260364289, 4261675009, 4264427521, 47509536769, 1099463786497, 1099461820417, 1099460902913, 1099460640769, 1099457495041, 1099455004673, 1099454218241, 1099453431809, 1099451465729, 1099451072513, 1099450810369, 1099450023937, 1099448057857, 1099446484993, 1099443732481, 1099442159617, 1099439800321, 1099438620673, 2135162881, 4272291841, 4272685057, 1040056321, 2135818241, 4273340417, 2142502913, 4274126849, 4276092929, 2144468993, 2146041857, 2146959361, 4277403649, 4279369729, 2147352577, 4280025089, 4280156161, 4281204737, 1038745601>, earth.arg_scales = array<f64: 75.609895558562386>, init_level = 24 : i64} {
    %0 = "ckks.modswitchc"(%arg0) <{downFactor = 1 : i64}> {runtime.scale_log2 = 75.609895558562386 : f64} : (tensor<1x!ckks.poly<2 * 76 * 8>>) -> tensor<1x!ckks.poly<2 * 76 * 7>>
    %1 = "ckks.encode"() <{payload = dense<1.000000e+00> : tensor<32768xf64>}> {runtime.scale_log2 = 39.857539755424682 : f64} : () -> tensor<1x!ckks.poly<1 * 40 * 7>>
    %2 = "ckks.mulcp"(%0, %1) {runtime.scale_log2 = 115.46743531398707 : f64} : (tensor<1x!ckks.poly<2 * 76 * 7>>, tensor<1x!ckks.poly<1 * 40 * 7>>) -> tensor<1x!ckks.poly<2 * 116 * 7>>
    %3 = "ckks.rescalec"(%2) {runtime.scale_log2 = 75.467498089019045 : f64} : (tensor<1x!ckks.poly<2 * 116 * 7>>) -> tensor<1x!ckks.poly<2 * 76 * 6>>
    %4 = "ckks.rescalec"(%3) {runtime.scale_log2 = 4.000000e+01 : f64} : (tensor<1x!ckks.poly<2 * 76 * 6>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %5 = "ckks.bootstrapc"(%4) {runtime.scale_log2 = 4.000000e+01 : f64} : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 24>>
    return %5 : tensor<1x!ckks.poly<2 * 40 * 24>>
  }
}
