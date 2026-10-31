module { func.func @probe(%x: tensor<1x!earth.ci<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>> {
%p = "earth.constant"() {value = dense<0.125> : tensor<1xf64>, rms_var = 0.125 : f64} : () -> tensor<1x!earth.pl<0 * 0>>
%y = "earth.mul"(%x, %p) : (tensor<1x!earth.ci<0 * 0>>, tensor<1x!earth.pl<0 * 0>>) -> tensor<1x!earth.ci<0 * 0>>
return %y : tensor<1x!earth.ci<0 * 0>>
}}