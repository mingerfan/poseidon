module { func.func @graph(%x: tensor<1x!earth.ci<0 * 0>>, %y: tensor<1x!earth.ci<0 * 0>>) -> (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) {
%p = "earth.constant"() {value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%v0 = "earth.mul"(%x, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v1 = "earth.mul"(%v0, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v2 = "earth.mul"(%y, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v3 = "earth.add"(%v1, %v2) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v4 = "earth.mul"(%v3, %x) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
%v5 = "earth.mul"(%v4, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %v5, %v2 : tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.ci<0 * 0>>
}}
