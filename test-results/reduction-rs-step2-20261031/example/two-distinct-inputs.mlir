module {
  func.func @reduction(%arg0: tensor<1x!ckks.poly<2 * 40 * 9>>, %arg1: tensor<1x!ckks.poly<2 * 40 * 9>>) -> tensor<1x!ckks.poly<2 * 51 * 8>> attributes {ckks.rns_moduli = array<i64: 1073709057, 1073676289, 1073643521, 1073610753, 1073577985, 1073545217, 1073512449, 1073479681, 1073446913, 1073414145>, earth.arg_scales = array<f64: 4.000000e+01, 4.000000e+01>, init_level = 9 : i64} {
    %0 = "ckks.encode"() <{payload = dense<6.250000e-02> : tensor<8xf64>}> {runtime.scale_log2 = 4.000000e+01 : f64} : () -> tensor<1x!ckks.poly<1 * 40 * 9>>
    %1 = "ckks.mulcp"(%arg0, %0) {runtime.scale_log2 = 8.000000e+01 : f64} : (tensor<1x!ckks.poly<2 * 40 * 9>>, tensor<1x!ckks.poly<1 * 40 * 9>>) -> tensor<1x!ckks.poly<2 * 80 * 9>>
    %2 = "ckks.encode"() <{payload = dense<-1.250000e-01> : tensor<8xf64>}> {runtime.scale_log2 = 4.000000e+01 : f64} : () -> tensor<1x!ckks.poly<1 * 40 * 9>>
    %3 = "ckks.mulcp"(%arg1, %2) {runtime.scale_log2 = 8.000000e+01 : f64} : (tensor<1x!ckks.poly<2 * 40 * 9>>, tensor<1x!ckks.poly<1 * 40 * 9>>) -> tensor<1x!ckks.poly<2 * 80 * 9>>
    %4 = "ckks.addcc"(%1, %3) {runtime.scale_log2 = 8.000000e+01 : f64} : (tensor<1x!ckks.poly<2 * 80 * 9>>, tensor<1x!ckks.poly<2 * 80 * 9>>) -> tensor<1x!ckks.poly<2 * 80 * 9>>
    %5 = "ckks.rescalec"(%4) {runtime.scale_log2 = 50.00044034143658 : f64} : (tensor<1x!ckks.poly<2 * 80 * 9>>) -> tensor<1x!ckks.poly<2 * 51 * 8>>
    return %5 : tensor<1x!ckks.poly<2 * 51 * 8>>
  }
}

