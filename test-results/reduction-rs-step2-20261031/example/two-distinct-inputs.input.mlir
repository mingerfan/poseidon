module { func.func @reduction(%x: tensor<1x!ckks.poly<2 * 40 * 9>>, %z: tensor<1x!ckks.poly<2 * 40 * 9>>) -> (tensor<1x!ckks.poly<2 * 51 * 8>>) attributes {init_level = 9 : i64, ckks.rns_moduli = array<i64: 1073709057, 1073676289, 1073643521, 1073610753, 1073577985, 1073545217, 1073512449, 1073479681, 1073446913, 1073414145>, earth.arg_scales = array<f64: 4.000000e+01, 4.000000e+01>} {
%p0 = "ckks.encode"() {payload = dense<0.0625> : tensor<8xf64>, runtime.scale_log2 = 4.00000000000000000e+01 : f64} : () -> tensor<1x!ckks.poly<1 * 40 * 9>>
%m0 = "ckks.mulcp"(%x, %p0) {runtime.scale_log2 = 8.00000000000000000e+01 : f64} : (tensor<1x!ckks.poly<2 * 40 * 9>>, tensor<1x!ckks.poly<1 * 40 * 9>>) -> tensor<1x!ckks.poly<2 * 80 * 9>>
%r0_0 = "ckks.rescalec"(%m0) {runtime.scale_log2 = 5.00004403414365797e+01 : f64} : (tensor<1x!ckks.poly<2 * 80 * 9>>) -> tensor<1x!ckks.poly<2 * 51 * 8>>
%p1 = "ckks.encode"() {payload = dense<-0.125> : tensor<8xf64>, runtime.scale_log2 = 4.00000000000000000e+01 : f64} : () -> tensor<1x!ckks.poly<1 * 40 * 9>>
%m1 = "ckks.mulcp"(%z, %p1) {runtime.scale_log2 = 8.00000000000000000e+01 : f64} : (tensor<1x!ckks.poly<2 * 40 * 9>>, tensor<1x!ckks.poly<1 * 40 * 9>>) -> tensor<1x!ckks.poly<2 * 80 * 9>>
%r1_0 = "ckks.rescalec"(%m1) {runtime.scale_log2 = 5.00004403414365797e+01 : f64} : (tensor<1x!ckks.poly<2 * 80 * 9>>) -> tensor<1x!ckks.poly<2 * 51 * 8>>
%a0 = "ckks.addcc"(%r0_0, %r1_0) {runtime.scale_log2 = 5.00004403414365797e+01 : f64} : (tensor<1x!ckks.poly<2 * 51 * 8>>, tensor<1x!ckks.poly<2 * 51 * 8>>) -> tensor<1x!ckks.poly<2 * 51 * 8>>
return %a0 : tensor<1x!ckks.poly<2 * 51 * 8>>
}}
