module { func.func @graph(%x: tensor<1x!earth.ci<0 * 25>>, %y: tensor<1x!earth.ci<0 * 25>>) -> (tensor<1x!earth.ci<0 * 25>>) {
%p = "earth.constant"() {value = dense<1.01> : tensor<8xf64>, rms_var = 1.01 : f64} : () -> tensor<1x!earth.pl<0 * 25>>
%v0 = "earth.rotate"(%x) <{offset = array<i64: 1>}> : (tensor<1x!earth.ci<0 * 25>>) -> tensor<1x!earth.ci<0 * 25>>
%v1 = "earth.mul"(%v0, %p) : (tensor<1x!earth.ci<0 * 25>>, tensor<1x!earth.pl<0 * 25>>) -> tensor<1x!earth.ci<0 * 25>>
%v2 = "earth.mul"(%v1, %p) : (tensor<1x!earth.ci<0 * 25>>, tensor<1x!earth.pl<0 * 25>>) -> tensor<1x!earth.ci<0 * 25>>
%v3 = "earth.mul"(%v2, %p) : (tensor<1x!earth.ci<0 * 25>>, tensor<1x!earth.pl<0 * 25>>) -> tensor<1x!earth.ci<0 * 25>>
return %v3 : tensor<1x!earth.ci<0 * 25>>
}}
