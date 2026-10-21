module { func.func @graph(%x: tensor<1x!earth.ci<0 * 45>>, %y: tensor<1x!earth.ci<0 * 45>>) -> (tensor<1x!earth.ci<0 * 45>>) {
%p = "earth.constant"() {value = dense<1.01> : tensor<1xf64>, rms_var = 1.01 : f64} : () -> tensor<1x!earth.pl<0 * 45>>
%v0 = "earth.mul"(%x, %p) : (tensor<1x!earth.ci<0 * 45>>, tensor<1x!earth.pl<0 * 45>>) -> tensor<1x!earth.ci<0 * 45>>
return %v0 : tensor<1x!earth.ci<0 * 45>>
}}
