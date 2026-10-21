module { func.func @graph(%x: tensor<1x!earth.ci<0 * 0>>, %y: tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>) {
%p = "earth.constant"() {value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%v0 = "earth.mul"(%x, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v1 = "earth.mul"(%v0, %v0) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v2 = "earth.mul"(%v1, %v1) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %v2 : tensor<1x!earth.ci<0 * 0>>
}}
